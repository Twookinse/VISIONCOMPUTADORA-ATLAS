"""
API REST y Servidor Unificado con FastAPI para Nex Vision AI.

Incluye:
- Endpoints de eventos y metricas por hora (/api/events, /api/metrics/hourly).
- Procesamiento de videos grabados bajo demanda (/api/videos, /api/pipeline/run).
- Monitoreo en TIEMPO REAL optimizado para camaras 720p RTSP/USB (/api/live/*)
  con streaming de video MJPEG en vivo hacia el Dashboard y guardado inmediato en SQLite.
- Entrega directa del Dashboard React compilado (dashboard/dist/) en http://localhost:8000.
"""

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import shutil
import cv2
from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.database import get_db, create_session_factory, PROJECT_ROOT
from src.db.models import PipelineRun, KitchenEvent, save_report_to_db
from src.detection.tracker import Tracker
from src.events.zones import load_zones, draw_zones
from src.analysis.kitchen import KitchenAnalyzer, DEFAULT_JAR_CLASSES, JAR_ONLY_CLASSES

OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEOS_DIR = PROJECT_ROOT / "videos"
DASHBOARD_DIST = PROJECT_ROOT / "dashboard" / "dist"


# =====================================================================
# MOTOR DE MONITOREO EN TIEMPO REAL (Optimizado para 720p @ 1000 KB/s)
# =====================================================================
class LiveMonitor:
    """Gestiona la captura y analisis en tiempo real (RTSP, Webcam o Video en vivo)."""

    def __init__(self):
        self.running = False
        self.source = ""
        self.model_name = "models/custom/jarras_nex_v1.pt"
        self.zones_file = "config/zones/casa_cocina.json"
        self.resolution = "1280x720"
        self.fps_actual = 0.0
        self.lifted = 0
        self.returned = 0
        self.total_lifted = 0
        self.latest_jpeg = None
        self.error = None
        self._thread = None
        self._lock = threading.Lock()

    def start(self, source: str, model: str = "models/custom/jarras_nex_v1.pt",
              zones: str = "config/zones/casa_cocina.json", confidence: float = 0.25):
        if self.running:
            self.stop()
        self.error = None
        self.source = source
        self.model_name = model
        self.zones_file = zones
        self.lifted = 0
        self.returned = 0
        self.total_lifted = 0
        self.fps_actual = 0.0

        # Crear cuadro inicial inmediato mientras carga el modelo YOLO
        import numpy as np
        placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Cargando modelo YOLO y abriendo video...",
                    (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(placeholder, f"Fuente: {source}",
                    (60, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1)
        ok, buf = cv2.imencode(".jpg", placeholder)
        if ok:
            with self._lock:
                self.latest_jpeg = buf.tobytes()

        self.running = True
        self._thread = threading.Thread(
            target=self._worker,
            args=(source, model, zones, confidence),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def status(self):
        return {
            "activo": self.running,
            "fuente": self.source,
            "modelo": self.model_name,
            "zonas": self.zones_file,
            "resolucion": self.resolution,
            "fps": round(self.fps_actual, 1),
            "jarras_levantadas_sesion": self.lifted,
            "jarras_devueltas_sesion": self.returned,
            "objetos_levantados_sesion": self.total_lifted,
            "error": self.error,
        }

    def _worker(self, source_str: str, model_rel: str, zones_rel: str, confidence: float):
        try:
            zones = load_zones(PROJECT_ROOT / zones_rel)

            # Si es RTSP, usar transporte TCP para evitar perdida de paquetes a 1000 KB/s
            if str(source_str).lower().startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"
                cap_source = source_str
                is_live_camera = True
            elif str(source_str).isdigit():
                cap_source = int(source_str)
                is_live_camera = True
            else:
                full_path = PROJECT_ROOT / source_str
                cap_source = str(full_path if full_path.exists() else source_str)
                is_live_camera = False

            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                self.error = f"No se pudo abrir la fuente: {source_str}"
                self.running = False
                return

            # Optimizar para camara 720p (1280x720) sin acumulacion de buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(cap_source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.resolution = f"{width}x{height}"

            # Mostrar el primer frame real del video de inmediato mientras inicia YOLO
            ret0, first_frame = cap.read()
            if ret0:
                preview = draw_zones(first_frame, zones)
                cv2.rectangle(preview, (0, 0), (width, 36), (0, 0, 0), -1)
                cv2.putText(preview, "Iniciando detector YOLO sobre el video...",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                ok0, buf0 = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok0:
                    with self._lock:
                        self.latest_jpeg = buf0.tobytes()

            tracker = Tracker(
                model_path=PROJECT_ROOT / model_rel,
                confidence=confidence,
                tracker_type=str(PROJECT_ROOT / "config/trackers/bytetrack_cocina.yaml"),
            )
            analyzer = KitchenAnalyzer(zones, jar_classes=DEFAULT_JAR_CLASSES, min_frames=3)

            # Crear registro de corrida en SQLite para esta sesion en vivo
            session_local = create_session_factory()
            db = session_local()
            try:
                live_run = PipelineRun(
                    created_at=datetime.now(),
                    source=f"EN_VIVO: {source_str}",
                    model=model_rel,
                    zones_file=zones_rel,
                )
                db.add(live_run)
                db.commit()
                db.refresh(live_run)
                run_id = live_run.id
            finally:
                db.close()

            frame_idx = 0
            t0 = time.time()
            last_event_text = ""
            last_event_frame = -999

            while self.running:
                # En camara en vivo, descartar frames viejos acumulados en buffer
                if is_live_camera:
                    cap.grab()
                ret, frame = cap.retrieve() if is_live_camera else cap.read()
                if not ret:
                    if not is_live_camera:
                        # Si se prueba el modo en vivo con un archivo de video, terminar al acabar
                        break
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                results = tracker.track(frame)
                new_events = analyzer.update(results, tracker.class_names, frame_idx, fps)

                self.lifted = analyzer.lifted
                self.returned = analyzer.returned
                self.total_lifted = analyzer.total_lifted

                # Si ocurrio un evento nuevo, guardarlo al instante en SQLite
                if new_events:
                    db = session_local()
                    try:
                        for ev in new_events:
                            action = "LEVANTADO/A" if ev.event_type == "salida" else "DEVUELTO/A"
                            last_event_text = f"{ev.class_name} #{ev.track_id} {action}"
                            last_event_frame = frame_idx
                            db.add(KitchenEvent(
                                run_id=run_id,
                                created_at=datetime.now(),
                                source=str(source_str),
                                frame=ev.frame,
                                time_s=ev.time_s,
                                track_id=ev.track_id,
                                class_name=ev.class_name,
                                zone=ev.zone,
                                event_type=ev.event_type,
                            ))
                        run_obj = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
                        if run_obj:
                            run_obj.frames_processed = frame_idx
                            run_obj.jarras_levantadas = analyzer.lifted
                            run_obj.jarras_devueltas = analyzer.returned
                            run_obj.objetos_levantados_total = analyzer.total_lifted
                            run_obj.objetos_devueltos_total = analyzer.total_returned
                        db.commit()
                    finally:
                        db.close()

                # Dibujar anotaciones para la vista en vivo del Dashboard
                annotated = draw_zones(results.plot(), analyzer.zones)
                cv2.rectangle(annotated, (0, 0), (width, 36), (0, 0, 0), -1)
                elapsed = max(0.1, time.time() - t0)
                self.fps_actual = frame_idx / elapsed
                hud = (f"EN VIVO ({width}x{height} | {self.fps_actual:.1f} FPS) | "
                       f"Jarras Lev: {analyzer.lifted} | Total Obj Lev: {analyzer.total_lifted}")
                cv2.putText(annotated, hud, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if frame_idx - last_event_frame < fps * 1.5:
                    cv2.putText(annotated, last_event_text, (10, 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self._lock:
                        self.latest_jpeg = buffer.tobytes()

            cap.release()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.running = False


live_monitor = LiveMonitor()


def seed_existing_reports_if_empty():
    """Si la base de datos esta vacia, importa los reportes *_conteo.json de output/."""
    session_local = create_session_factory()
    db = session_local()
    try:
        has_runs = db.query(PipelineRun).first() is not None
        if has_runs or not OUTPUT_DIR.exists():
            return
    finally:
        db.close()

    for report_file in sorted(OUTPUT_DIR.glob("*_conteo.json")):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
            save_report_to_db(report)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_existing_reports_if_empty()
    yield
    live_monitor.stop()


app = FastAPI(
    title="Nex Vision AI - API y Servidor de Cocina",
    description="API REST, Streaming en Tiempo Real (720p) y Dashboard para Nex.",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRunRequest(BaseModel):
    source: str = "videos/VIDEO 2 VISION COMPUTADORA.mp4"
    model: str = "models/custom/jarras_nex_v1.pt"
    zones: str = "config/zones/casa_cocina.json"
    tracker: str = "config/trackers/bytetrack_cocina.yaml"
    confidence: float = 0.25
    min_frames: int = 3
    skip_frames: int = 1
    classes: Optional[List[str]] = None


class LiveStartRequest(BaseModel):
    source: str = "0"
    model: str = "models/custom/jarras_nex_v1.pt"
    zones: str = "config/zones/casa_cocina.json"
    confidence: float = 0.25


@app.get("/api/status")
def api_status():
    return {
        "servicio": "Nex Vision AI - API de Cocina",
        "estado": "activo",
        "version": "0.5.0",
        "documentacion_swagger": "/docs",
        "monitoreo_en_vivo": live_monitor.status(),
    }


@app.get("/api/videos")
def list_available_videos():
    """Lista los archivos de video disponibles en la carpeta videos/."""
    videos = []
    if VIDEOS_DIR.exists():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for f in sorted(VIDEOS_DIR.glob(ext)):
                videos.append(f"videos/{f.name}")
    return {"videos": videos}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recibe un archivo de video subido por el usuario y lo guarda en videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "video_subido.mp4").name
    dest_path = VIDEOS_DIR / safe_name
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    rel_path = f"videos/{safe_name}"
    return {
        "mensaje": f"Video '{safe_name}' cargado correctamente",
        "video_path": rel_path,
    }


@app.get("/api/events")
def list_events(
    date: Optional[str] = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase (ej: Jarra, Vaso, Plato)"),
    event_type: Optional[str] = Query(None, description="Filtrar por tipo: 'salida' o 'entrada'"),
    zone: Optional[str] = Query(None, description="Filtrar por nombre de zona ROI"),
    limit: int = Query(200, ge=1, le=1000, description="Maximo de eventos a devolver"),
    db: Session = Depends(get_db),
):
    """Lista los eventos registrados en la base de datos con filtros opcionales."""
    query = db.query(KitchenEvent)

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha invalido. Usa YYYY-MM-DD")
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())
        query = query.filter(KitchenEvent.created_at >= start_dt, KitchenEvent.created_at <= end_dt)

    if class_name:
        query = query.filter(KitchenEvent.class_name == class_name)

    if event_type:
        query = query.filter(KitchenEvent.event_type == event_type)

    if zone:
        query = query.filter(KitchenEvent.zone == zone)

    events = query.order_by(KitchenEvent.id.desc()).limit(limit).all()
    return {
        "total": len(events),
        "eventos": [e.to_dict() for e in events],
    }


@app.get("/api/metrics/hourly")
def hourly_metrics(
    date: Optional[str] = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase especifica (ej: Jarra)"),
    db: Session = Depends(get_db),
):
    """Devuelve conteos de objetos levantados y devueltos agrupados por hora y por clase."""
    query = db.query(KitchenEvent)

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha invalido. Usa YYYY-MM-DD")
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())
        query = query.filter(KitchenEvent.created_at >= start_dt, KitchenEvent.created_at <= end_dt)

    if class_name:
        query = query.filter(KitchenEvent.class_name == class_name)

    events = query.all()

    hourly = {
        h: {
            "hora": f"{h:02d}:00",
            "jarras_levantadas": 0,
            "jarras_devueltas": 0,
            "objetos_levantados": 0,
            "objetos_devueltos": 0,
            "por_clase": {},
        }
        for h in range(24)
    }

    totales_por_clase = {}

    for ev in events:
        hour = ev.created_at.hour if ev.created_at else 0
        bucket = hourly[hour]
        cls_bucket = bucket["por_clase"].setdefault(
            ev.class_name, {"levantados": 0, "devueltos": 0}
        )
        cls_total = totales_por_clase.setdefault(
            ev.class_name, {"levantados": 0, "devueltos": 0}
        )

        if ev.event_type == "salida":
            bucket["objetos_levantados"] += 1
            cls_bucket["levantados"] += 1
            cls_total["levantados"] += 1
            if ev.class_name in JAR_ONLY_CLASSES:
                bucket["jarras_levantadas"] += 1
        else:
            bucket["objetos_devueltos"] += 1
            cls_bucket["devueltos"] += 1
            cls_total["devueltos"] += 1
            if ev.class_name in JAR_ONLY_CLASSES:
                bucket["jarras_devueltas"] += 1

    active_hours = [
        b for b in hourly.values()
        if b["objetos_levantados"] > 0 or b["objetos_devueltos"] > 0
    ]

    return {
        "fecha": date or "todas",
        "clase_filtro": class_name or "todas",
        "jarras_levantadas_total": sum(b["jarras_levantadas"] for b in hourly.values()),
        "objetos_levantados_total": sum(b["objetos_levantados"] for b in hourly.values()),
        "totales_por_clase": totales_por_clase,
        "horas_con_actividad": active_hours,
        "por_hora": list(hourly.values()),
    }


@app.get("/api/runs")
def list_runs(db: Session = Depends(get_db)):
    """Lista las ejecuciones del pipeline guardadas en la base de datos."""
    runs = db.query(PipelineRun).order_by(PipelineRun.id.desc()).all()
    return {
        "total": len(runs),
        "corridas": [r.to_dict() for r in runs],
    }


@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db)):
    """Devuelve el estado de las camaras configuradas (720p @ 1000 KB/s)."""
    last_run = db.query(PipelineRun).order_by(PipelineRun.id.desc()).first()
    zones_path = PROJECT_ROOT / "config/zones/casa_cocina.json"
    model_path = PROJECT_ROOT / "models/custom/jarras_nex_v1.pt"
    live_info = live_monitor.status()
    return {
        "camaras": [
            {
                "id": "cam-cocina-01",
                "nombre": "Camara 1 - Mesa de Cocina (Nex 720p)",
                "estado": "en_vivo" if live_info["activo"] else "activa",
                "resolucion": live_info["resolucion"] or "1280x720 (720p @ 1000 KB/s)",
                "zona_roi": "Mesa de cocina" if zones_path.exists() else "Sin configurar",
                "modelo": "jarras_nex_v1.pt (Custom)" if model_path.exists() else "yolo26n.pt",
                "ultima_actividad": last_run.created_at.isoformat() if last_run and last_run.created_at else None,
                "fuente_actual": live_info["fuente"] if live_info["activo"] else (
                    last_run.source if last_run else "videos/VIDEO 2 VISION COMPUTADORA.mp4"
                ),
            }
        ]
    }


@app.post("/api/live/start")
def start_live_monitoring(req: LiveStartRequest):
    """Inicia el monitoreo en tiempo real (RTSP 720p, Webcam 0 o archivo en vivo)."""
    live_monitor.start(
        source=req.source,
        model=req.model,
        zones=req.zones,
        confidence=req.confidence,
    )
    return {"mensaje": "Monitoreo en tiempo real iniciado", "estado": live_monitor.status()}


@app.post("/api/live/stop")
def stop_live_monitoring():
    """Detiene el monitoreo en tiempo real."""
    live_monitor.stop()
    return {"mensaje": "Monitoreo en tiempo real detenido", "estado": live_monitor.status()}


@app.get("/api/live/status")
def get_live_status():
    """Devuelve el estado actual del monitor en tiempo real."""
    return live_monitor.status()


@app.get("/api/live/stream")
async def stream_live_video():
    """Transmite el video procesado en tiempo real (MJPEG) sin bloquear hilos."""
    async def generate():
        while live_monitor.running:
            with live_monitor._lock:
                frame_bytes = live_monitor.latest_jpeg
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(req: PipelineRunRequest):
    """Procesa un video grabado con el pipeline de conteo y guarda los eventos en SQLite."""
    from src.detection.tracker import Tracker
    from src.events.zones import load_zones
    from src.analysis.kitchen import KitchenAnalyzer

    source_path = PROJECT_ROOT / req.source
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"No se encontro el video '{req.source}'")

    zones = load_zones(PROJECT_ROOT / req.zones)
    if not zones:
        raise HTTPException(status_code=400, detail=f"No hay zonas validas en '{req.zones}'")

    model_path = PROJECT_ROOT / req.model
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"No se encontro el modelo '{req.model}'")

    tracker = Tracker(
        model_path=model_path,
        confidence=req.confidence,
        tracker_type=str(PROJECT_ROOT / req.tracker),
    )
    analyzer = KitchenAnalyzer(
        zones,
        jar_classes=req.classes or DEFAULT_JAR_CLASSES,
        min_frames=req.min_frames,
    )

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"No se pudo abrir el video '{req.source}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % max(1, req.skip_frames) != 0:
                continue
            results = tracker.track(frame)
            analyzer.update(results, tracker.class_names, frame_count, fps)
    finally:
        cap.release()

    elapsed = time.time() - start_time
    report = {
        "fuente": req.source,
        "modelo": req.model,
        "zonas_archivo": req.zones,
        "frames_procesados": frame_count,
        "duracion_video_s": round(frame_count / fps, 1),
        "tiempo_proceso_s": round(elapsed, 1),
        **analyzer.summary(),
    }

    run_id = save_report_to_db(report)
    return {
        "run_id": run_id,
        "mensaje": "Video procesado y eventos guardados en SQLite",
        "reporte": report,
    }


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Canal WebSocket para enviar actualizaciones en vivo al Dashboard."""
    await websocket.accept()
    last_seen_id = -1
    try:
        while True:
            session_local = create_session_factory()
            db = session_local()
            try:
                latest = db.query(KitchenEvent).order_by(KitchenEvent.id.desc()).first()
                total_events = db.query(KitchenEvent).count()
                current_id = latest.id if latest else 0
                if current_id != last_seen_id:
                    last_seen_id = current_id
                    await websocket.send_json({
                        "type": "sync",
                        "total_events": total_events,
                        "latest_event": latest.to_dict() if latest else None,
                        "live": live_monitor.status(),
                        "timestamp": datetime.now().isoformat(),
                    })
            finally:
                db.close()
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Servir el Dashboard React compilado (dashboard/dist) directamente en /
if (DASHBOARD_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DASHBOARD_DIST / "assets")), name="assets")


@app.get("/")
def serve_dashboard_or_status():
    """Sirve el Dashboard React si esta compilado, o el JSON de estado de la API."""
    index_file = DASHBOARD_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return api_status()
