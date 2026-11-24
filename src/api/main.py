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
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db.database import get_db, create_session_factory, PROJECT_ROOT
from src.db.models import (
    PipelineRun,
    KitchenEvent,
    save_report_to_db,
    MODE_VIDEO,
)
from src.db.queries import (
    FilterError,
    SHIFTS,
    apply_event_filters,
    event_hour_expr,
    get_jornada_start_hour,
    jornada_bounds,
    jornada_hour_order,
)
from src.utils.video_meta import resolve_video_start, compute_source_key
from src.detection.tracker import Tracker
from src.detection.visualizer import save_event_snapshot
from src.events.zones import load_zones
from src.live.monitor import LiveMonitor
from src.analysis.kitchen import (
    KitchenAnalyzer,
    DEFAULT_JAR_CLASSES,
    JAR_ONLY_CLASSES,
)
from src.config.cameras import (
    InvalidRtspUrl,
    get_cameras,
    get_camera_by_id,
    update_camera_rtsp,
    detect_camera_for_source,
    public_camera,
    unmask_source,
)
from src.config.settings import cors_origins, max_upload_bytes
from src.api.auth import AccessControlMiddleware
from src.utils.security import (
    UnsafePathError,
    mask_rtsp_url,
    mask_text,
    sanitize_upload_name,
    unique_destination,
    validate_model_path,
    validate_source,
    validate_tracker_path,
    validate_zones_path,
)

OUTPUT_DIR = PROJECT_ROOT / "output"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
VIDEOS_DIR = PROJECT_ROOT / "videos"
DASHBOARD_DIST = PROJECT_ROOT / "dashboard" / "dist"


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
    resolved_source = unmask_source(source)
    if not resolved_source and cam_cfg:
        resolved_source = cam_cfg.get("rtsp_url") or cam_cfg.get("default_video") or "0"
    return resolved_cam_id, resolved_source, resolved_zones


def _validate_inputs(camera_id: Optional[str], source: Optional[str], zones: Optional[str],
                     model: str, tracker: Optional[str] = None, allow_stream: bool = False):
    """Resuelve la camara y valida que todas las rutas apunten a archivos permitidos.

    Evita que la API lea o cargue archivos fuera de videos/, models/ y config/
    (un .pt arbitrario puede ejecutar codigo al cargarse).

    Returns:
        (camera_id, source, zones, model, tracker) normalizados.
    """
    cam_id, resolved_src, resolved_zones = _resolve_camera_and_zones(camera_id, source, zones)
    try:
        src = validate_source(resolved_src, allow_stream=allow_stream)
        zones_rel = validate_zones_path(resolved_zones)
        model_rel = validate_model_path(model)
        tracker_rel = validate_tracker_path(tracker) if tracker else None
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=mask_text(str(exc), resolved_src))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return cam_id, src, zones_rel, model_rel, tracker_rel


def _parse_video_start_or_400(source: str, video_start: Optional[str]):
    """Resuelve la hora real de inicio del video; 400 si el usuario envio un valor invalido."""
    start_dt, estimated = resolve_video_start(source, video_start)
    if video_start and estimated:
        raise HTTPException(
            status_code=400,
            detail="Fecha/hora de inicio invalida. Usa el formato YYYY-MM-DDTHH:MM",
        )
    return start_dt, estimated


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
              min_frames: int = 4, skip_frames: int = 0, max_seconds: float = 0.0,
              video_start: Optional[str] = None):
        if self.running:
            raise HTTPException(status_code=409, detail="Ya hay un video analizandose en este momento")

        cam_id, resolved_src, resolved_zones, model, _ = _validate_inputs(
            camera_id, source, zones, model, allow_stream=False
        )
        start_dt, start_estimated = _parse_video_start_or_400(resolved_src, video_start)

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
            args=(cam_id, resolved_src, model, resolved_zones, confidence, min_frames, skip_frames,
                  max_seconds, start_dt, start_estimated),
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

    def _run_job(self, cam_id, source_rel, model_rel, zones_rel, confidence, min_frames, skip_frames,
                 max_seconds, start_dt, start_estimated):
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
                    new_events = analyzer.update(results, tracker.class_names, frame_count, fps)
                    for ev in new_events:
                        ev.snapshot_file = save_event_snapshot(
                            frame, results, ev, analyzer.zones, SNAPSHOTS_DIR, cam_id
                        )
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
                "modo": MODE_VIDEO,
                "inicio_grabacion": start_dt.isoformat(),
                "inicio_estimado": start_estimated,
                "source_key": compute_source_key(PROJECT_ROOT / source_rel),
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


live_monitor = LiveMonitor(snapshots_dir=SNAPSHOTS_DIR)
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

# CORS solo para el Dashboard en modo desarrollo (Vite en :5173). En produccion el
# Dashboard se sirve desde el mismo origen y no necesita CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)
# Se agrega al final para que se ejecute primero: la propia PC entra libre y los
# demas equipos necesitan usuario/contrasena (ver src/api/auth.py).
app.add_middleware(AccessControlMiddleware)


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
    video_start: Optional[str] = None  # Hora real de inicio de la grabacion (YYYY-MM-DDTHH:MM)


class LiveStartRequest(BaseModel):
    source: Optional[str] = None
    camera_id: Optional[str] = "cocina_1"
    model: str = "models/custom/jarras_nex_v1.pt"
    zones: Optional[str] = None
    confidence: float = 0.25
    view_mode: Optional[str] = "limpia"
    show_zones: Optional[bool] = True


class LiveDisplayRequest(BaseModel):
    view_mode: Optional[str] = None
    show_zones: Optional[bool] = None


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
                start_dt, start_estimated = resolve_video_start(f.name)
                videos.append(rel)
                videos_detail.append({
                    "inicio_detectado": None if start_estimated else start_dt.isoformat(timespec="minutes"),
                    "path": rel,
                    "name": f.name,
                    "camera_id": cam["id"] if cam else "cocina_1",
                    "zones_file": cam["zones_file"] if cam else "config/zones/cocina_1.json",
                })
    return {"videos": videos, "detalle": videos_detail}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recibe un archivo de video subido por el usuario y lo guarda en videos/.

    Solo acepta extensiones de video, nunca sobrescribe un archivo existente
    (agrega ' (1)', ' (2)'...) y limita el tamano (NEX_MAX_UPLOAD_MB).
    """
    try:
        safe_name = sanitize_upload_name(file.filename)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = unique_destination(VIDEOS_DIR, safe_name)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    limit = max_upload_bytes()
    written = 0
    try:
        with open(tmp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"El video supera el limite de {limit // (1024 * 1024)} MB",
                    )
                buffer.write(chunk)
        tmp_path.replace(dest_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    safe_name = dest_path.name
    rel_path = f"videos/{safe_name}"
    cam = detect_camera_for_source(rel_path)
    return {
        "mensaje": f"Video '{safe_name}' cargado correctamente",
        "video_path": rel_path,
        "camera_id": cam["id"] if cam else "cocina_1",
    }


@app.get("/api/shifts")
def list_shifts():
    """Turnos disponibles y hora de inicio de la jornada (config/cameras.json)."""
    return {
        "jornada_inicio_hora": get_jornada_start_hour(),
        "turnos": {k: {"label": v["label"], "horas": v["hours"]} for k, v in SHIFTS.items()},
    }


@app.get("/api/events")
def list_events(
    camera_id: Optional[str] = Query(None, description="Filtrar por camara: cocina_1, cocina_2, almacen"),
    date: Optional[str] = Query(None, description="Filtrar por JORNADA YYYY-MM-DD (de 08:00 a 07:59 del dia siguiente)"),
    shift: Optional[str] = Query(None, description="Turno: todos, dia, tarde, noche"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase (ej: Jarra, Taper, Olla)"),
    event_type: Optional[str] = Query(None, description="Filtrar por tipo: 'salida' o 'entrada'"),
    zone: Optional[str] = Query(None, description="Filtrar por nombre de zona ROI"),
    include_demo: bool = Query(False, description="Incluir reproducciones con marcas de videos (demo)"),
    limit: int = Query(300, ge=1, le=2000, description="Maximo de eventos a devolver"),
    db: Session = Depends(get_db),
):
    """Lista los eventos registrados en SQLite con filtros opcionales.

    'total' es la cantidad real de eventos que cumplen el filtro; 'mostrados'
    la cantidad devuelta en esta respuesta (limitada por 'limit').
    """
    try:
        query = apply_event_filters(
            db.query(KitchenEvent), camera_id=camera_id, date=date, shift=shift,
            class_name=class_name, event_type=event_type, zone=zone, include_demo=include_demo,
        )
    except FilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    total = query.count()
    events = query.order_by(KitchenEvent.created_at.desc(), KitchenEvent.id.desc()).limit(limit).all()
    return {
        "total": total,
        "mostrados": len(events),
        "eventos": [e.to_dict() for e in events],
    }


@app.get("/api/metrics/hourly")
def hourly_metrics(
    camera_id: Optional[str] = Query(None, description="Filtrar por camara: cocina_1, cocina_2, almacen"),
    date: Optional[str] = Query(None, description="Filtrar por JORNADA YYYY-MM-DD (de 08:00 a 07:59 del dia siguiente)"),
    shift: Optional[str] = Query(None, description="Turno: todos, dia, tarde, noche"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase especifica (ej: Jarra)"),
    include_demo: bool = Query(False, description="Incluir reproducciones con marcas de videos (demo)"),
    db: Session = Depends(get_db),
):
    """Devuelve los KPIs y conteos agrupados por hora, clase y zona (calculados en SQL).

    Los totales consideran TODOS los eventos del filtro (sin limite de cantidad).
    'por_hora' viene ordenado desde la hora de inicio de la jornada.
    """
    hour_col = event_hour_expr().label("hora")
    try:
        query = apply_event_filters(
            db.query(
                hour_col,
                KitchenEvent.class_name,
                KitchenEvent.zone,
                KitchenEvent.event_type,
                func.count(KitchenEvent.id),
            ),
            camera_id=camera_id, date=date, shift=shift,
            class_name=class_name, include_demo=include_demo,
        )
    except FilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rows = query.group_by(
        hour_col, KitchenEvent.class_name, KitchenEvent.zone, KitchenEvent.event_type
    ).all()

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

    for hour, cls_name, zone_name, ev_type, count in rows:
        bucket = hourly[int(hour or 0) % 24]
        key = "levantados" if ev_type == "salida" else "devueltos"
        is_jar = cls_name in JAR_ONLY_CLASSES
        for target in (
            bucket["por_clase"].setdefault(cls_name, {"levantados": 0, "devueltos": 0}),
            totales_por_clase.setdefault(cls_name, {"levantados": 0, "devueltos": 0}),
            totales_por_zona.setdefault(zone_name, {"levantados": 0, "devueltos": 0}),
        ):
            target[key] += count
        if ev_type == "salida":
            bucket["objetos_levantados"] += count
            if is_jar:
                bucket["jarras_levantadas"] += count
        else:
            bucket["objetos_devueltos"] += count
            if is_jar:
                bucket["jarras_devueltas"] += count

    ordered = [hourly[h] for h in jornada_hour_order()]
    active_hours = [b for b in ordered if b["objetos_levantados"] or b["objetos_devueltos"]]
    jornada = None
    if date:
        j_start, j_end = jornada_bounds(date)
        jornada = {"inicio": j_start.isoformat(), "fin": j_end.isoformat()}

    objetos_lev = sum(b["objetos_levantados"] for b in ordered)
    objetos_dev = sum(b["objetos_devueltos"] for b in ordered)
    return {
        "camera_id": camera_id or "todas",
        "fecha": date or "todas",
        "jornada": jornada,
        "turno": shift or "todos",
        "clase_filtro": class_name or "todas",
        "jarras_levantadas_total": sum(b["jarras_levantadas"] for b in ordered),
        "jarras_devueltas_total": sum(b["jarras_devueltas"] for b in ordered),
        "objetos_levantados_total": objetos_lev,
        "objetos_devueltos_total": objetos_dev,
        "total_eventos": objetos_lev + objetos_dev,
        "totales_por_clase": totales_por_clase,
        "totales_por_zona": totales_por_zona,
        "horas_con_actividad": active_hours,
        "por_hora": ordered,
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
            "rtsp_url": mask_rtsp_url(rtsp_url),
            "default_video": cam.get("default_video", ""),
            "modelo": "jarras_nex_v1.pt (Nex Oficial - 38 clases)" if model_path.exists() else "yolo26n.pt",
            "eventos_registrados": events_count,
            "ultima_actividad": last_run.created_at.isoformat() if last_run and last_run.created_at else None,
            "fuente_actual": live_info["fuente"] if is_this_live else (
                mask_rtsp_url(rtsp_url) or cam.get("default_video", "")
            ),
        })

    return {"camaras": result}


@app.put("/api/cameras/{camera_id}")
def save_camera_rtsp_endpoint(camera_id: str, req: CameraUpdateRequest):
    """Guarda el enlace RTSP de una camara en config/cameras.json (listo para el local de Nex).

    La respuesta devuelve el enlace enmascarado (rtsp://usuario:****@IP/...).
    """
    try:
        updated = update_camera_rtsp(camera_id, req.rtsp_url)
    except InvalidRtspUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Camara '{camera_id}' no encontrada")
    return {
        "mensaje": f"Enlace RTSP guardado para {updated['nombre']}",
        "camara": public_camera(updated),
    }


@app.get("/api/snapshots/{filename}")
def get_event_snapshot(filename: str):
    """Sirve la foto de evidencia (snapshot) capturada en el instante de un evento."""
    safe_name = Path(filename).name
    file_path = SNAPSHOTS_DIR / safe_name
    if file_path.suffix.lower() != ".jpg" or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Captura no encontrada")
    return FileResponse(str(file_path), media_type="image/jpeg")


@app.post("/api/live/start")
def start_live_monitoring(req: LiveStartRequest):
    """Inicia el monitoreo/reproduccion con marcas para una camara de Nex."""
    cam_id, source, zones, model, _ = _validate_inputs(
        req.camera_id, req.source, req.zones, req.model, allow_stream=True
    )
    live_monitor.start(
        source=source,
        camera_id=cam_id,
        model=model,
        zones=zones,
        confidence=req.confidence,
        view_mode=req.view_mode,
        show_zones=req.show_zones,
    )
    return {"mensaje": "Monitoreo en tiempo real iniciado", "estado": live_monitor.status()}


@app.post("/api/live/display")
def update_live_display(req: LiveDisplayRequest):
    """Cambia en caliente el modo visual (limpia/solo_jarras/tecnica) y visibilidad de zonas."""
    live_monitor.set_display(view_mode=req.view_mode, show_zones=req.show_zones)
    return {"mensaje": "Modo visual actualizado", "estado": live_monitor.status()}


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
        last_sent, last_time = None, 0.0
        while live_monitor.running:
            frame_bytes = live_monitor.latest_jpeg
            now = time.monotonic()
            # Enviar solo frames nuevos (o repetir 1 vez por segundo para mantener la conexion)
            if frame_bytes and (frame_bytes is not last_sent or now - last_time > 1.0):
                last_sent, last_time = frame_bytes, now
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.04)

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
        video_start=req.video_start,
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
    cam_id, resolved_src, resolved_zones, model_rel, tracker_rel = _validate_inputs(
        req.camera_id, req.source, req.zones, req.model, tracker=req.tracker, allow_stream=False
    )
    source_path = PROJECT_ROOT / resolved_src

    zones = load_zones(PROJECT_ROOT / resolved_zones)
    if not zones:
        raise HTTPException(status_code=400, detail=f"No hay zonas validas en '{resolved_zones}'")

    tracker = Tracker(
        model_path=PROJECT_ROOT / model_rel,
        confidence=req.confidence,
        tracker_type=str(PROJECT_ROOT / tracker_rel),
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
            new_events = analyzer.update(results, tracker.class_names, frame_count, fps)
            for ev in new_events:
                ev.snapshot_file = save_event_snapshot(
                    frame, results, ev, analyzer.zones, SNAPSHOTS_DIR, cam_id
                )
    finally:
        cap.release()

    elapsed = time.time() - start_time
    report = {
        "camera_id": cam_id,
        "fuente": resolved_src,
        "modelo": model_rel,
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
