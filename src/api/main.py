"""
API REST con FastAPI para Nex Vision AI (Sprint 4).

Expone los eventos detectados en la cocina y las metricas por hora:
- GET  /api/events          Lista eventos con filtros por fecha, clase, tipo y zona.
- GET  /api/metrics/hourly  Conteos agrupados por hora (00:00 a 23:00) y por clase.
- GET  /api/runs            Historial de videos procesados.
- POST /api/pipeline/run    Ejecuta el conteo sobre un video y guarda en SQLite.

Ejecucion local:
    uvicorn src.api.main:app --reload
    Documentacion Swagger: http://localhost:8000/docs
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.database import get_db, create_session_factory, PROJECT_ROOT
from src.db.models import PipelineRun, KitchenEvent, save_report_to_db
from src.analysis.kitchen import DEFAULT_JAR_CLASSES, JAR_ONLY_CLASSES

OUTPUT_DIR = PROJECT_ROOT / "output"


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


app = FastAPI(
    title="Nex Vision AI - API de Cocina",
    description="API REST para consultar eventos de objetos de cocina y metricas operativas de Nex.",
    version="0.4.0",
    lifespan=lifespan,
)

# Permitir peticiones desde el Dashboard React (Sprint 5)
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


@app.get("/")
def root():
    """Ruta raiz de estado para evitar 404 al entrar a http://localhost:8000/."""
    return {
        "servicio": "Nex Vision AI - API de Cocina",
        "estado": "activo",
        "version": "0.4.0",
        "documentacion_swagger": "/docs",
        "endpoints": [
            "/api/events",
            "/api/metrics/hourly",
            "/api/runs",
            "/api/pipeline/run",
        ],
    }


@app.get("/api/events")
def list_events(
    date: Optional[str] = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    class_name: Optional[str] = Query(None, description="Filtrar por clase (ej: Jarra, Vaso, Plato)"),
    event_type: Optional[str] = Query(None, description="Filtrar por tipo: 'salida' (levantado) o 'entrada' (devuelto)"),
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
    """Devuelve el estado de las camaras configuradas para el Dashboard (Tarea 5.6)."""
    last_run = db.query(PipelineRun).order_by(PipelineRun.id.desc()).first()
    zones_path = PROJECT_ROOT / "config/zones/casa_cocina.json"
    model_path = PROJECT_ROOT / "models/custom/jarras_nex_v1.pt"
    return {
        "camaras": [
            {
                "id": "cam-cocina-01",
                "nombre": "Camara 1 - Mesa de Cocina (Nex)",
                "estado": "activa",
                "zona_roi": "Mesa de cocina" if zones_path.exists() else "Sin configurar",
                "modelo": "jarras_nex_v1.pt (Custom)" if model_path.exists() else "yolo26n.pt",
                "ultima_actividad": last_run.created_at.isoformat() if last_run and last_run.created_at else None,
                "fuente_actual": last_run.source if last_run else "videos/VIDEO 2 VISION COMPUTADORA.mp4",
            }
        ]
    }


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Canal WebSocket para enviar actualizaciones en vivo al Dashboard (Tarea 5.7)."""
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
                        "timestamp": datetime.now().isoformat(),
                    })
            finally:
                db.close()
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(req: PipelineRunRequest):
    """Procesa un video con el pipeline de conteo y guarda los eventos en SQLite."""
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
