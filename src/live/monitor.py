"""
Monitoreo en tiempo real: captura, deteccion con marcas y conteo de una camara.

Cada vez que se inicia una camara se crea una SESION independiente con su propia
senal de parada. Asi, al cambiar de camara rapido:
- La sesion anterior termina sola en cuanto puede (aunque este cargando YOLO).
- Sus frames, contadores y errores quedan en su propia sesion y no pisan a la
  nueva (antes el hilo viejo seguia corriendo y al terminar apagaba al nuevo).

Para camaras en vivo (RTSP / USB) se usa LiveFrameReader, que siempre entrega el
frame mas reciente (sin retraso acumulado) y reconecta automaticamente.
Los videos grabados se leen frame a frame.
"""

import itertools
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.analysis.kitchen import KitchenAnalyzer
from src.db.database import PROJECT_ROOT, create_session_factory
from src.db.models import KitchenEvent, PipelineRun, MODE_DEMO, MODE_LIVE
from src.detection.tracker import Tracker
from src.detection.visualizer import draw_clean_zones, render_annotated_frame, save_event_snapshot
from src.events.zones import load_zones
from src.live.capture import (
    LiveFrameReader,
    STATE_CONNECTING,
    STATE_LIVE,
    STATE_RECONNECTING,
    STATE_STOPPED,
    is_live_source,
)
from src.utils.security import mask_rtsp_url, mask_text

VIEW_MODES = ("limpia", "solo_jarras", "tecnica")
VIEW_LABELS = {"limpia": "VISTA LIMPIA", "solo_jarras": "SOLO JARRAS", "tecnica": "VISTA TECNICA"}
TRACKER_CONFIG = "config/trackers/bytetrack_cocina.yaml"
STREAM_MAX_WIDTH = 960


class LiveSession:
    """Estado de UNA ejecucion del monitor (una camara o video)."""

    _ids = itertools.count(1)

    def __init__(self, camera_id, source, model, zones_file, confidence):
        self.id = next(self._ids)
        self.camera_id = camera_id
        self.source = source                       # Fuente real (puede tener la clave RTSP)
        self.public_source = mask_rtsp_url(source) # Lo unico que se muestra
        self.model = model
        self.zones_file = zones_file
        self.confidence = confidence
        self.is_live = is_live_source(source)
        # Identifica las fotos de evidencia de esta sesion (no se mezclan entre videos/dias)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tag = f"en_vivo_{stamp}" if self.is_live else f"demo_{Path(source).stem}_{stamp}"

        self.stop_event = threading.Event()
        self.thread = None
        self.running = True
        self.connection = STATE_CONNECTING
        self.reconnects = 0
        self.resolution = "-"
        self.fps_actual = 0.0
        self.lifted = 0
        self.returned = 0
        self.total_lifted = 0
        self.error = None

        self._jpeg_lock = threading.Lock()
        self._jpeg = None

    @property
    def latest_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def publish(self, image):
        """Codifica un frame para el streaming MJPEG (max 960 px de ancho)."""
        h, w = image.shape[:2]
        if w > STREAM_MAX_WIDTH:
            scale = STREAM_MAX_WIDTH / w
            image = cv2.resize(image, (STREAM_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._jpeg_lock:
                self._jpeg = buffer.tobytes()


def _message_frame(lines, base=None, size=(854, 480)):
    """Cuadro con texto (cargando, reconectando...) opcionalmente sobre el ultimo frame."""
    if base is not None:
        img = (base * 0.35).astype(np.uint8)
    else:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    y = img.shape[0] // 2 - 20 * (len(lines) - 1)
    for i, text in enumerate(lines):
        scale, color = (0.85, (255, 255, 255)) if i == 0 else (0.6, (180, 220, 255))
        cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)
        y += 45
    return img


class LiveMonitor:
    """Gestiona la sesion de monitoreo en tiempo real (una a la vez).

    Args:
        snapshots_dir: Carpeta donde se guardan las fotos de evidencia.
        tracker_factory: Crea el detector+tracker (se reemplaza en las pruebas).
        reader_factory: Crea el lector de camaras en vivo (se reemplaza en las pruebas).
        session_factory: Fabrica de sesiones de base de datos (None = la configurada).
        stop_timeout: Segundos que se espera a que termine la sesion anterior.
    """

    def __init__(self, snapshots_dir, tracker_factory=Tracker, reader_factory=LiveFrameReader,
                 session_factory=None, stop_timeout=3.0):
        self.snapshots_dir = Path(snapshots_dir)
        self.tracker_factory = tracker_factory
        self.reader_factory = reader_factory
        self._db_factory = session_factory
        self.stop_timeout = stop_timeout

        self.view_mode = "limpia"
        self.show_zones = True
        self._session = None
        self._control_lock = threading.Lock()  # Serializa start/stop concurrentes

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    @property
    def session(self):
        return self._session

    @property
    def running(self):
        session = self._session
        return bool(session and session.running and not session.stop_event.is_set())

    @property
    def latest_jpeg(self):
        session = self._session
        return session.latest_jpeg if session else None

    def set_display(self, view_mode=None, show_zones=None):
        """Actualiza el modo visual en caliente sin reiniciar la camara."""
        if view_mode in VIEW_MODES:
            self.view_mode = view_mode
        if show_zones is not None:
            self.show_zones = bool(show_zones)

    def start(self, source, camera_id, model, zones, confidence=0.25, view_mode=None, show_zones=None):
        """Inicia una sesion nueva (detiene la anterior). Las rutas ya deben estar validadas."""
        with self._control_lock:
            self._stop_locked()
            self.set_display(view_mode=view_mode, show_zones=show_zones)
            session = LiveSession(camera_id, source, model, zones, confidence)
            session.publish(_message_frame([
                f"Cargando camara {camera_id.upper()} y detector YOLO...",
                f"Fuente: {session.public_source}",
            ]))
            session.thread = threading.Thread(
                target=self._run_session, args=(session,), daemon=True,
                name=f"nex-live-{camera_id}-{session.id}",
            )
            self._session = session
            session.thread.start()
            return session

    def stop(self):
        with self._control_lock:
            self._stop_locked()

    def _stop_locked(self):
        session = self._session
        if not session:
            return
        session.stop_event.set()
        if session.thread and session.thread.is_alive() and session.thread is not threading.current_thread():
            # Si sigue ocupada (ej. cargando YOLO) terminara sola: su estado es independiente
            session.thread.join(timeout=self.stop_timeout)
        session.running = False

    def status(self):
        s = self._session
        return {
            "activo": self.running,
            "camera_id": s.camera_id if s else "cocina_1",
            "fuente": s.public_source if s else "",
            "modelo": s.model if s else "",
            "zonas": s.zones_file if s else "",
            "view_mode": self.view_mode,
            "show_zones": self.show_zones,
            "conexion": (s.connection if self.running else STATE_STOPPED) if s else STATE_STOPPED,
            "reconexiones": s.reconnects if s else 0,
            "resolucion": s.resolution if s else "-",
            "fps": round(s.fps_actual, 1) if s else 0.0,
            "jarras_levantadas_sesion": s.lifted if s else 0,
            "jarras_devueltas_sesion": s.returned if s else 0,
            "objetos_levantados_sesion": s.total_lifted if s else 0,
            "error": s.error if s else None,
        }

    # ------------------------------------------------------------------
    # Hilo de la sesion
    # ------------------------------------------------------------------
    def _db(self):
        if self._db_factory is None:
            self._db_factory = create_session_factory()
        return self._db_factory()

    def _run_session(self, session):
        reader = None
        cap = None
        try:
            zones = load_zones(PROJECT_ROOT / session.zones_file)

            if session.is_live:
                # El lector se conecta en segundo plano mientras carga YOLO
                reader = self.reader_factory(session.source).start()
                fps = None
            else:
                cap = cv2.VideoCapture(str(PROJECT_ROOT / session.source))
                if not cap.isOpened():
                    session.error = f"No se pudo abrir la fuente: {session.public_source}"
                    return
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                session.resolution = f"{w}x{h}"
                session.connection = STATE_LIVE
                # Videos de Nex a 60fps: saltar el logo del DVR de los primeros segundos
                if fps >= 45:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
                ok, first = cap.read()
                if ok:
                    self._publish_preview(session, first, zones)

            tracker = self.tracker_factory(
                model_path=PROJECT_ROOT / session.model,
                confidence=session.confidence,
                tracker_type=str(PROJECT_ROOT / TRACKER_CONFIG),
            )
            if session.stop_event.is_set():
                return  # Se cambio de camara mientras cargaba el modelo

            analyzer = KitchenAnalyzer(zones, jar_classes=None, min_frames=4)
            run_mode = MODE_LIVE if session.is_live else MODE_DEMO
            run_id = self._create_run(session, run_mode)

            if session.is_live:
                self._loop_live(session, reader, tracker, analyzer, run_id, run_mode)
            else:
                self._loop_file(session, cap, fps, tracker, analyzer, run_id, run_mode)
        except Exception as exc:  # noqa: BLE001 - se informa en el Dashboard
            session.error = mask_text(str(exc), session.source)
        finally:
            if reader is not None:
                reader.stop()
            if cap is not None:
                cap.release()
            session.running = False
            if session.connection != STATE_STOPPED:
                session.connection = STATE_STOPPED

    def _loop_live(self, session, reader, tracker, analyzer, run_id, run_mode):
        t0 = time.monotonic()
        last_seq = 0
        inferred = 0
        last_frame = None
        last_notice = 0.0
        hud = _HudState()

        while not session.stop_event.is_set():
            session.connection = reader.state
            session.reconnects = reader.reconnects
            item = reader.read_latest(last_seq, timeout=0.5)
            if item is None:
                # Sin frames nuevos: mostrar el estado de la conexion al operador
                now = time.monotonic()
                if reader.state in (STATE_RECONNECTING, STATE_CONNECTING) and now - last_notice > 1.0:
                    last_notice = now
                    title = ("Reconectando con la camara..." if reader.state == STATE_RECONNECTING
                             else "Conectando con la camara...")
                    session.publish(_message_frame(
                        [f"[{session.camera_id.upper()}] {title}",
                         f"Fuente: {session.public_source} | Reconexiones: {reader.reconnects}"],
                        base=last_frame,
                    ))
                    session.error = reader.error
                continue

            last_seq, frame = item
            session.error = None
            session.resolution = f"{frame.shape[1]}x{frame.shape[0]}"
            fps = reader.fps
            elapsed = time.monotonic() - t0
            # Indice de frame segun el reloj real (se descartan frames para no acumular retraso)
            frame_idx = int(elapsed * fps)
            inferred += 1
            session.fps_actual = inferred / max(0.1, elapsed)
            self._process_frame(session, frame, frame_idx, fps, tracker, analyzer, run_id, run_mode, hud)
            last_frame = frame

    def _loop_file(self, session, cap, fps, tracker, analyzer, run_id, run_mode):
        skip = 3 if fps >= 45 else 1
        frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        t0 = time.monotonic()
        inferred = 0
        hud = _HudState()
        while not session.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            inferred += 1
            session.fps_actual = inferred / max(0.1, time.monotonic() - t0)
            self._process_frame(session, frame, frame_idx, fps, tracker, analyzer, run_id, run_mode, hud)

    # ------------------------------------------------------------------
    def _publish_preview(self, session, frame, zones):
        preview = draw_clean_zones(frame, zones, subtle=True) if self.show_zones else frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (30, 34, 42), -1)
        cv2.putText(preview, f"[{session.camera_id.upper()}] Iniciando modelo YOLO...",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        session.publish(preview)

    def _create_run(self, session, run_mode):
        db = self._db()
        try:
            run = PipelineRun(
                created_at=datetime.now(),
                camera_id=session.camera_id,
                source=f"EN_VIVO: {session.public_source}",
                model=session.model,
                zones_file=session.zones_file,
                mode=run_mode,
                video_start=datetime.now(),
                start_time_estimated=not session.is_live,
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run.id
        finally:
            db.close()

    def _save_events(self, session, frame, results, new_events, analyzer, run_id, run_mode, frame_idx):
        db = self._db()
        try:
            for ev in new_events:
                ev.snapshot_file = save_event_snapshot(
                    frame, results, ev, analyzer.zones, self.snapshots_dir, session.camera_id,
                    tag=session.tag,
                )
                db.add(KitchenEvent(
                    run_id=run_id,
                    created_at=datetime.now(),
                    camera_id=session.camera_id,
                    mode=run_mode,
                    source=session.public_source,
                    frame=ev.frame,
                    time_s=ev.time_s,
                    track_id=ev.track_id,
                    class_name=ev.class_name,
                    zone=ev.zone,
                    event_type=ev.event_type,
                    snapshot_file=ev.snapshot_file,
                ))
            run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
            if run:
                run.frames_processed = frame_idx
                run.jarras_levantadas = analyzer.lifted
                run.jarras_devueltas = analyzer.returned
                run.objetos_levantados_total = analyzer.total_lifted
                run.objetos_devueltos_total = analyzer.total_returned
            db.commit()
        finally:
            db.close()

    def _process_frame(self, session, frame, frame_idx, fps, tracker, analyzer, run_id, run_mode, hud):
        now = time.monotonic()
        results = tracker.track(frame)
        new_events = analyzer.update(results, tracker.class_names, frame_idx, fps)
        session.lifted = analyzer.lifted
        session.returned = analyzer.returned
        session.total_lifted = analyzer.total_lifted

        if new_events and not session.stop_event.is_set():
            self._save_events(session, frame, results, new_events, analyzer, run_id, run_mode, frame_idx)
            for ev in new_events:
                action = "LEVANTADO" if ev.event_type == "salida" else "DEVUELTO"
                hud.last_event_text = f"{ev.class_name} #{ev.track_id} {action} ({ev.zone})"
                hud.last_event_time = now
                hud.recent_tracks[ev.track_id] = (ev.event_type, now + 3.0)

        # Resaltar 3 segundos los objetos que acaban de cruzar una zona
        hud.recent_tracks = {tid: v for tid, v in hud.recent_tracks.items() if v[1] > now}
        annotated = render_annotated_frame(
            frame=frame,
            results=results,
            class_names=tracker.class_names,
            zones=analyzer.zones,
            view_mode=self.view_mode,
            show_zones=self.show_zones,
            recent_track_events={tid: v[0] for tid, v in hud.recent_tracks.items()},
        )
        width = annotated.shape[1]
        cv2.rectangle(annotated, (0, 0), (width, 40), (30, 34, 42), -1)
        text = (f"[{session.camera_id.upper()} | {VIEW_LABELS.get(self.view_mode, self.view_mode.upper())}] "
                f"{session.fps_actual:.1f} FPS | Jarras Lev: {analyzer.lifted} | "
                f"Total Obj Lev: {analyzer.total_lifted}")
        if session.reconnects:
            text += f" | Reconexiones: {session.reconnects}"
        cv2.putText(annotated, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        if hud.last_event_text and now - hud.last_event_time < 3.0:
            cv2.rectangle(annotated, (8, 46), (min(width - 8, 680), 82), (62, 62, 249), -1)
            cv2.putText(annotated, f"EVENTO: {hud.last_event_text}", (16, 71),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        session.publish(annotated)


class _HudState:
    """Texto del ultimo evento y objetos resaltados en pantalla."""

    def __init__(self):
        self.last_event_text = ""
        self.last_event_time = 0.0
        self.recent_tracks = {}  # {track_id: (event_type, expira)}
