"""
API REST y Servidor Unificado Multi-Camara con FastAPI para Nex Vision AI.

Incluye:
- Gestion centralizada de las 3 camaras de Nex (config/cameras.json):
  cocina_1, cocina_2 y almacen, con soporte para configurar URLs RTSP
  directamente desde el Dashboard al desplegar en el local.
- Endpoints de eventos y metricas por hora filtrables por camara, fecha,
  zona ROI y clase de objeto (/api/events, /api/metrics/hourly).
- Monitoreo en TIEMPO REAL optimizado para camaras 720p RTSP y videos 1080p 60fps
  (/api/live/*) con cambio rapido de camara y streaming MJPEG fluido.
- Analisis de videos grabados con seguimiento de progreso en vivo (/api/pipeline/*).
- Entrega directa del Dashboard React compilado (dashboard/dist/) en http://localhost:8000.
"""

import asyncio
import json
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
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
from src.analysis.kitchen import (
    KitchenAnalyzer,
    DEFAULT_JAR_CLASSES,
    JAR_ONLY_CLASSES,
    normalize_class_name,
)
from src.config.cameras import (
    get_cameras,
    get_camera_by_id,
    update_camera_rtsp,
    detect_camera_for_source,
)

OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEOS_DIR = PROJECT_ROOT / "videos"
DASHBOARD_DIST = PROJECT_ROOT / "dashboard" / "dist"


def _clean_plot(results):
    """Dibuja las detecciones usando los nombres normalizados (ej. Jarra en vez de Jarra2)."""
    orig_names = results.names
    try:
        results.names = {
            k: (normalize_class_name(v) or v)
            for k, v in orig_names.items()
        }
        return results.plot()
    finally:
        results.names = orig_names


def _resolve_camera_and_zones(camera_id: Optional[str], source: Optional[str], zones_override: Optional[str]):
    """Resuelve camera_id, archivo de zonas y fuente a partir de config/cameras.json."""
    cam_cfg = get_camera_by_id(camera_id) if camera_id else None
    if not cam_cfg and source:
        cam_cfg = detect_camera_for_source(source)
    if not cam_cfg:
        cams = get_cameras()
        cam_cfg = cams[0] if cams else None

    resolved_cam_id = cam_cfg["id"] if cam_cfg else (camera_id or "cocina_1")
    resolved_zones = zones_override or (
        cam_cfg["zones_file"] if cam_cfg else "config/zones/cocina_1.json"
    )
    resolved_source = source
    if not resolved_source and cam_cfg:
        resolved_source = cam_cfg.get("rtsp_url") or cam_cfg.get("default_video") or "0"
    return resolved_cam_id, resolved_source, resolved_zones


# =====================================================================
# MOTOR DE MONITOREO EN TIEMPO REAL (Multi-Camara 720p RTSP / 1080p 60fps)
# =====================================================================
class LiveMonitor:
    """Gestiona la captura, visualizacion con marcas y conteo en tiempo real."""

    def __init__(self):
        self.running = False
        self.camera_id = "cocina_1"
        self.source = ""
        self.model_name = "models/custom/jarras_nex_v1.pt"
        self.zones_file = "config/zones/cocina_1.json"
        self.resolution = "1280x720"
        self.fps_actual = 0.0
        self.lifted = 0
        self.returned = 0
        self.total_lifted = 0
        self.latest_jpeg = None
        self.error = None
        self._thread = None
        self._lock = threading.Lock()

    def start(self, source: Optional[str] = None, camera_id: Optional[str] = None,
              model: str = "models/custom/jarras_nex_v1.pt",
              zones: Optional[str] = None, confidence: float = 0.25):
        if self.running:
            self.stop()

        cam_id, resolved_src, resolved_zones = _resolve_camera_and_zones(camera_id, source, zones)

        self.error = None
        self.camera_id = cam_id
        self.source = resolved_src
        self.model_name = model
        self.zones_file = resolved_zones
        self.lifted = 0
        self.returned = 0
        self.total_lifted = 0
        self.fps_actual = 0.0

        # Cuadro inicial inmediato mientras carga el modelo YOLO
        placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Cargando camara {cam_id.upper()} y detector YOLO...",
                    (50, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(placeholder, f"Fuente: {resolved_src}",
                    (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
        ok, buf = cv2.imencode(".jpg", placeholder)
        if ok:
            with self._lock:
                self.latest_jpeg = buf.tobytes()

        self.running = True
        self._thread = threading.Thread(
            target=self._worker,
            args=(cam_id, resolved_src, model, resolved_zones, confidence),
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
            "camera_id": self.camera_id,
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

    def _encode_stream_frame(self, annotated):
        """Redimensiona suavemente a max 960px de ancho para streaming MJPEG rapido."""
        h, w = annotated.shape[:2]
        if w > 960:
            scale = 960.0 / w
            annotated = cv2.resize(annotated, (960, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self.latest_jpeg = buffer.tobytes()

    def _worker(self, cam_id: str, source_str: str, model_rel: str, zones_rel: str, confidence: float):
        try:
            zones = load_zones(PROJECT_ROOT / zones_rel)

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

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(cap_source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.resolution = f"{width}x{height}"

            # Si es video grabado a 60fps de Nex, saltar los primeros segundos del logo DVR
            if not is_live_camera and fps >= 45:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
            skip_frames = 3 if (not is_live_camera and fps >= 45) else 1

            # Mostrar primer cuadro con las zonas de la camara mientras arranca YOLO
            ret0, first_frame = cap.read()
            if ret0:
                preview = draw_zones(first_frame, zones)
                cv2.rectangle(preview, (0, 0), (width, 40), (0, 0, 0), -1)
                cv2.putText(preview, f"[{cam_id.upper()}] Iniciando modelo YOLO...",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                self._encode_stream_frame(preview)

            tracker = Tracker(
                model_path=PROJECT_ROOT / model_rel,
                confidence=confidence,
                tracker_type=str(PROJECT_ROOT / "config/trackers/bytetrack_cocina.yaml"),
            )
            analyzer = KitchenAnalyzer(zones, jar_classes=None, min_frames=4)

            session_local = create_session_factory()
            db = session_local()
            try:
                live_run = PipelineRun(
                    created_at=datetime.now(),
                    camera_id=cam_id,
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
            inferred_idx = 0
            t0 = time.time()
            last_event_text = ""
            last_event_frame = -999

            while self.running:
                if is_live_camera:
                    cap.grab()
                ret, frame = cap.retrieve() if is_live_camera else cap.read()
                if not ret:
                    if not is_live_camera:
                        break
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                if not is_live_camera and frame_idx % skip_frames != 0:
                    continue

                inferred_idx += 1
                results = tracker.track(frame)
                new_events = analyzer.update(results, tracker.class_names, frame_idx, fps)

                self.lifted = analyzer.lifted
                self.returned = analyzer.returned
                self.total_lifted = analyzer.total_lifted

                if new_events:
                    db = session_local()
                    try:
                        for ev in new_events:
                            action = "LEVANTADO" if ev.event_type == "salida" else "DEVUELTO"
                            last_event_text = f"{ev.class_name} #{ev.track_id} {action} ({ev.zone})"
                            last_event_frame = frame_idx
                            db.add(KitchenEvent(
                                run_id=run_id,
                                created_at=datetime.now(),
                                camera_id=cam_id,
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

                annotated = draw_zones(_clean_plot(results), analyzer.zones)
                cv2.rectangle(annotated, (0, 0), (width, 40), (0, 0, 0), -1)
                elapsed = max(0.1, time.time() - t0)
                self.fps_actual = inferred_idx / elapsed
                hud = (f"[{cam_id.upper()}] {width}x{height} ({self.fps_actual:.1f} FPS) | "
                       f"Jarras Lev: {analyzer.lifted} | Total Obj Lev: {analyzer.total_lifted}")
                cv2.putText(annotated, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if frame_idx - last_event_frame < fps * 2.0:
                    cv2.putText(annotated, last_event_text, (10, 74),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

                self._encode_stream_frame(annotated)

            cap.release()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.running = False


# =====================================================================
# MOTOR DE ANALISIS DE VIDEOS EN SEGUNDO PLANO (Con barra de progreso %)
# =====================================================================
class PipelineJobManager:
    """Procesa videos largos (8-10 min a 60fps) sin congelar el navegador."""

    def __init__(self):
        self.running = False
        self.cancel_requested = False
        self.camera_id = ""
        self.source = ""
        self.progress_pct = 0.0
        self.frames_processed = 0
        self.total_frames = 0
        self.events_found = 0
        self.jarras_found = 0
        self.last_report = None
        self.error = None
        self._thread = None

    def start(self, source: str, camera_id: Optional[str] = None,
              model: str = "models/custom/jarras_nex_v1.pt",
              zones: Optional[str] = None, confidence: float = 0.25,
              min_frames: int = 4, skip_frames: int = 0, max_seconds: float = 0.0):
        if self.running:
            raise HTTPException(status_code=409, detail="Ya hay un video analizandose en este momento")

        cam_id, resolved_src, resolved_zones = _resolve_camera_and_zones(camera_id, source, zones)
        source_path = PROJECT_ROOT / resolved_src
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"No se encontro el video '{resolved_src}'")

        self.running = True
        self.cancel_requested = False
        self.camera_id = cam_id
        self.source = resolved_src
        self.progress_pct = 0.0
        self.frames_processed = 0
        self.total_frames = 0
        self.events_found = 0
        self.jarras_found = 0
        self.last_report = None
        self.error = None

        self._thread = threading.Thread(
            target=self._run_job,
            args=(cam_id, resolved_src, model, resolved_zones, confidence, min_frames, skip_frames, max_seconds),
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        self.cancel_requested = True

    def status(self):
        return {
            "activo": self.running,
            "camera_id": self.camera_id,
            "fuente": self.source,
            "progreso_pct": round(self.progress_pct, 1),
            "frames_procesados": self.frames_processed,
            "total_frames": self.total_frames,
            "eventos_encontrados": self.events_found,
            "jarras_encontradas": self.jarras_found,
            "ultimo_reporte": self.last_report,
            "error": self.error,
        }

    def _run_job(self, cam_id, source_rel, model_rel, zones_rel, confidence, min_frames, skip_frames, max_seconds):
        try:
            zones = load_zones(PROJECT_ROOT / zones_rel)
            if not zones:
                self.error = f"No hay zonas configuradas en '{zones_rel}'"
                return

            tracker = Tracker(
                model_path=PROJECT_ROOT / model_rel,
                confidence=confidence,
                tracker_type=str(PROJECT_ROOT / "config/trackers/bytetrack_cocina.yaml"),
            )
            analyzer = KitchenAnalyzer(zones, jar_classes=None, min_frames=min_frames)

            cap = cv2.VideoCapture(str(PROJECT_ROOT / source_rel))
            if not cap.isOpened():
                self.error = f"No se pudo abrir '{source_rel}'"
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            eff_skip = skip_frames if skip_frames >= 1 else (3 if fps >= 45 else 1)
            max_frames = int(max_seconds * fps) if max_seconds > 0 else total_frames
            self.total_frames = min(total_frames, max_frames) if max_frames > 0 else total_frames

            frame_count = 0
            start_time = time.time()

            try:
                while not self.cancel_requested:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if max_frames and frame_count > max_frames:
                        break
                    self.frames_processed = frame_count
                    self.progress_pct = min(99.0, (frame_count / max(1, self.total_frames)) * 100.0)
                    if frame_count % eff_skip != 0:
                        continue
                    results = tracker.track(frame)
                    analyzer.update(results, tracker.class_names, frame_count, fps)
                    self.events_found = len(analyzer.events)
                    self.jarras_found = analyzer.lifted
            finally:
                cap.release()

            elapsed = time.time() - start_time
            report = {
                "camera_id": cam_id,
                "fuente": source_rel,
                "modelo": model_rel,
                "zonas_archivo": zones_rel,
                "frames_procesados": frame_count,
                "duracion_video_s": round(frame_count / fps, 1),
                "tiempo_proceso_s": round(elapsed, 1),
                "cancelado": self.cancel_requested,
                **analyzer.summary(),
            }

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            report_path = OUTPUT_DIR / f"{Path(source_rel).stem}_conteo.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            save_report_to_db(report)
            self.last_report = report
            self.progress_pct = 100.0
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.running = False


live_monitor = LiveMonitor()
pipeline_job = PipelineJobManager()


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
    pipeline_job.cancel()


app = FastAPI(
    title="Nex Vision AI - API y Servidor Multi-Camara",
    description="API REST, Streaming en Tiempo Real (3 camaras de Nex) y Dashboard.",
    version="0.6.0",
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
    source: str = "videos/COCINA 1 VIDEO 1.mp4"
    camera_id: Optional[str] = None
    model: str = "models/custom/jarras_nex_v1.pt"
    zones: Optional[str] = None
    tracker: str = "config/trackers/bytetrack_cocina.yaml"
    confidence: float = 0.25
    min_frames: int = 4
    skip_frames: int = 0
    max_seconds: float = 0.0
    classes: Optional[List[str]] = None


class LiveStartRequest(BaseModel):
    source: Optional[str] = None
    camera_id: Optional[str] = "cocina_1"
    model: str = "models/custom/jarras_nex_v1.pt"
    zones: Optional[str] = None
    confidence: float = 0.25


class CameraUpdateRequest(BaseModel):
    rtsp_url: str = ""


@app.get("/api/status")
def api_status():
    return {
        "servicio": "Nex Vision AI - Servidor Multi-Camara",
        "estado": "activo",
        "version": "0.6.0",
        "documentacion_swagger": "/docs",
        "monitoreo_en_vivo": live_monitor.status(),
        "analisis_video": pipeline_job.status(),
    }


@app.get("/api/classes")
def list_classes(db: Session = Depends(get_db)):
    """Devuelve las clases normalizadas de objetos de cocina disponibles para filtrar."""
    db_classes = [r[0] for r in db.query(KitchenEvent.class_name).distinct().all() if r[0]]
    merged = list(dict.fromkeys(DEFAULT_JAR_CLASSES + sorted(db_classes)))
    return {"clases": merged}


@app.get("/api/videos")
def list_available_videos():
    """Lista los archivos de video en videos/ junto con su camara asignada."""
    videos = []
    videos_detail = []
    if VIDEOS_DIR.exists():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for f in sorted(VIDEOS_DIR.glob(ext)):
                rel = f"videos/{f.name}"
                cam = detect_camera_for_source(rel)
                videos.append(rel)
                videos_detail.append({
                    "path": rel,
                    "name": f.name,
                    "camera_id": cam["id"] if cam else "cocina_1",
                    "zones_file": cam["zones_file"] if cam else "config/zones/cocina_1.json",
                })
    return {"videos": videos, "detalle": videos_detail}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recibe un archivo de video subido por el usuario y lo guarda en videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "video_subido.mp4").name
    dest_path = VIDEOS_DIR / safe_name
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    rel_path = f"videos/{safe_name}"
    cam = detect_camera_for_source(rel_path)
    return {
        "mensaje": f"Video '{safe_name}' cargado correctamente",
        "video_path": rel_path,
        "camera_id": cam["id"] if cam else "cocina_1",
    }


@app.get("/api/events")
def list_events(
    camera_id: Optional[str] = Query(None, description="Filtrar por camara: cocina_1, cocina_2, almacen"),
    date: Optional[str] = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase (ej: Jarra, Taper, Olla)"),
    event_type: Optional[str] = Query(None, description="Filtrar por tipo: 'salida' o 'entrada'"),
    zone: Optional[str] = Query(None, description="Filtrar por nombre de zona ROI"),
    limit: int = Query(300, ge=1, le=2000, description="Maximo de eventos a devolver"),
    db: Session = Depends(get_db),
):
    """Lista los eventos registrados en SQLite con filtros opcionales."""
    query = db.query(KitchenEvent)

    if camera_id:
        query = query.filter(KitchenEvent.camera_id == camera_id)

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
    camera_id: Optional[str] = Query(None, description="Filtrar por camara: cocina_1, cocina_2, almacen"),
    date: Optional[str] = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase especifica (ej: Jarra)"),
    db: Session = Depends(get_db),
):
    """Devuelve conteos agrupados por hora, por clase y por zona."""
    query = db.query(KitchenEvent)

    if camera_id:
        query = query.filter(KitchenEvent.camera_id == camera_id)

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
    totales_por_zona = {}

    for ev in events:
        hour = ev.created_at.hour if ev.created_at else 0
        bucket = hourly[hour]
        cls_bucket = bucket["por_clase"].setdefault(
            ev.class_name, {"levantados": 0, "devueltos": 0}
        )
        cls_total = totales_por_clase.setdefault(
            ev.class_name, {"levantados": 0, "devueltos": 0}
        )
        zone_total = totales_por_zona.setdefault(
            ev.zone, {"levantados": 0, "devueltos": 0}
        )

        if ev.event_type == "salida":
            bucket["objetos_levantados"] += 1
            cls_bucket["levantados"] += 1
            cls_total["levantados"] += 1
            zone_total["levantados"] += 1
            if ev.class_name in JAR_ONLY_CLASSES:
                bucket["jarras_levantadas"] += 1
        else:
            bucket["objetos_devueltos"] += 1
            cls_bucket["devueltos"] += 1
            cls_total["devueltos"] += 1
            zone_total["devueltos"] += 1
            if ev.class_name in JAR_ONLY_CLASSES:
                bucket["jarras_devueltas"] += 1

    active_hours = [
        b for b in hourly.values()
        if b["objetos_levantados"] > 0 or b["objetos_devueltos"] > 0
    ]

    return {
        "camera_id": camera_id or "todas",
        "fecha": date or "todas",
        "clase_filtro": class_name or "todas",
        "jarras_levantadas_total": sum(b["jarras_levantadas"] for b in hourly.values()),
        "objetos_levantados_total": sum(b["objetos_levantados"] for b in hourly.values()),
        "totales_por_clase": totales_por_clase,
        "totales_por_zona": totales_por_zona,
        "horas_con_actividad": active_hours,
        "por_hora": list(hourly.values()),
    }


@app.get("/api/runs")
def list_runs(
    camera_id: Optional[str] = Query(None, description="Filtrar corridas por camara"),
    db: Session = Depends(get_db),
):
    """Lista las ejecuciones del pipeline guardadas en la base de datos."""
    query = db.query(PipelineRun)
    if camera_id:
        query = query.filter(PipelineRun.camera_id == camera_id)
    runs = query.order_by(PipelineRun.id.desc()).all()
    return {
        "total": len(runs),
        "corridas": [r.to_dict() for r in runs],
    }


@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db)):
    """Devuelve las 3 camaras de Nex configuradas en config/cameras.json y su estado."""
    cams_cfg = get_cameras()
    model_path = PROJECT_ROOT / "models/custom/jarras_nex_v1.pt"
    live_info = live_monitor.status()
    result = []

    for cam in cams_cfg:
        cam_id = cam["id"]
        zones_path = PROJECT_ROOT / cam["zones_file"]
        loaded_zones = load_zones(zones_path)
        last_run = (
            db.query(PipelineRun)
            .filter(PipelineRun.camera_id == cam_id)
            .order_by(PipelineRun.id.desc())
            .first()
        )
        events_count = db.query(KitchenEvent).filter(KitchenEvent.camera_id == cam_id).count()
        is_this_live = live_info["activo"] and live_info["camera_id"] == cam_id
        rtsp_url = (cam.get("rtsp_url") or "").strip()

        result.append({
            "id": cam_id,
            "nombre": cam["nombre"],
            "ubicacion": cam.get("ubicacion", ""),
            "estado": "en_vivo" if is_this_live else ("rtsp_listo" if rtsp_url else "modo_video"),
            "resolucion": live_info["resolucion"] if is_this_live else cam.get("resolucion_esperada", "720p"),
            "zones_file": cam["zones_file"],
            "zonas_lista": [z.name for z in loaded_zones],
            "zona_roi": f"{len(loaded_zones)} zonas configuradas" if loaded_zones else "Sin configurar",
            "rtsp_url": rtsp_url,
            "default_video": cam.get("default_video", ""),
            "modelo": "jarras_nex_v1.pt (Nex Oficial - 38 clases)" if model_path.exists() else "yolo26n.pt",
            "eventos_registrados": events_count,
            "ultima_actividad": last_run.created_at.isoformat() if last_run and last_run.created_at else None,
            "fuente_actual": live_info["fuente"] if is_this_live else (rtsp_url or cam.get("default_video", "")),
        })

    return {"camaras": result}


@app.put("/api/cameras/{camera_id}")
def save_camera_rtsp_endpoint(camera_id: str, req: CameraUpdateRequest):
    """Guarda el enlace RTSP de una camara en config/cameras.json (listo para el local de Nex)."""
    updated = update_camera_rtsp(camera_id, req.rtsp_url)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Camara '{camera_id}' no encontrada")
    return {
        "mensaje": f"Enlace RTSP guardado para {updated['nombre']}",
        "camara": updated,
    }


@app.post("/api/live/start")
def start_live_monitoring(req: LiveStartRequest):
    """Inicia el monitoreo/reproduccion con marcas para una camara de Nex."""
    live_monitor.start(
        source=req.source,
        camera_id=req.camera_id,
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


@app.post("/api/pipeline/start")
def start_pipeline_job(req: PipelineRunRequest):
    """Inicia el analisis de un video en segundo plano con progreso en vivo."""
    pipeline_job.start(
        source=req.source,
        camera_id=req.camera_id,
        model=req.model,
        zones=req.zones,
        confidence=req.confidence,
        min_frames=req.min_frames,
        skip_frames=req.skip_frames,
        max_seconds=req.max_seconds,
    )
    return {"mensaje": "Analisis iniciado en segundo plano", "estado": pipeline_job.status()}


@app.get("/api/pipeline/status")
def get_pipeline_job_status():
    """Consulta el progreso (%) del analisis de video en segundo plano."""
    return pipeline_job.status()


@app.post("/api/pipeline/cancel")
def cancel_pipeline_job():
    """Cancela un analisis de video largo guardando lo contado hasta ese momento."""
    pipeline_job.cancel()
    return {"mensaje": "Cancelacion solicitada", "estado": pipeline_job.status()}


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(req: PipelineRunRequest):
    """Procesa un video grabado de forma directa y guarda los eventos en SQLite."""
    cam_id, resolved_src, resolved_zones = _resolve_camera_and_zones(
        req.camera_id, req.source, req.zones
    )

    source_path = PROJECT_ROOT / resolved_src
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"No se encontro el video '{resolved_src}'")

    zones = load_zones(PROJECT_ROOT / resolved_zones)
    if not zones:
        raise HTTPException(status_code=400, detail=f"No hay zonas validas en '{resolved_zones}'")

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
        jar_classes=req.classes,
        min_frames=req.min_frames,
    )

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"No se pudo abrir el video '{resolved_src}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    eff_skip = req.skip_frames if req.skip_frames >= 1 else (3 if fps >= 45 else 1)
    max_frames = int(req.max_seconds * fps) if req.max_seconds > 0 else 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            if frame_count % eff_skip != 0:
                continue
            results = tracker.track(frame)
            analyzer.update(results, tracker.class_names, frame_count, fps)
    finally:
        cap.release()

    elapsed = time.time() - start_time
    report = {
        "camera_id": cam_id,
        "fuente": resolved_src,
        "modelo": req.model,
        "zonas_archivo": resolved_zones,
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
