"""
Analisis completo de un video (o camara desde la consola): UNA sola implementacion
usada por el Dashboard (analisis en segundo plano), por /api/pipeline/run y por
scripts/run_counting.py.

Flujo: abrir video -> YOLO + ByteTrack -> KitchenAnalyzer (eventos de zona) ->
fotos de evidencia -> reporte (dict) -> JSON en output/ + SQLite.

Los callbacks permiten a cada caller agregar lo suyo sin duplicar el bucle:
- on_progress(progress): barra de progreso del Dashboard.
- should_cancel(): boton "Detener y Guardar" del Dashboard.
- on_frame(frame, results, analyzer, new_events, frame_idx, fps): video anotado
  y ventana de vista previa del script (si devuelve False, se detiene).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from src.analysis.kitchen import KitchenAnalyzer
from src.config.cameras import FALLBACK_MODEL, get_default_model
from src.db.database import PROJECT_ROOT
from src.db.models import MODE_LIVE, MODE_VIDEO, save_report_to_db
from src.detection.tracker import Tracker
from src.detection.visualizer import save_event_snapshot
from src.events.zones import load_zones
from src.live.capture import is_live_source
from src.utils.security import mask_rtsp_url
from src.utils.video_meta import compute_source_key

DEFAULT_MODEL = FALLBACK_MODEL  # Se usa el "modelo_default" de config/cameras.json (ver get_default_model)
DEFAULT_TRACKER = "config/trackers/bytetrack_cocina.yaml"
DEFAULT_FPS = 30.0


class AnalysisError(RuntimeError):
    """No se pudo iniciar el analisis (sin zonas, video que no abre, etc.)."""


@dataclass
class AnalysisRequest:
    """Parametros de un analisis. Las rutas son relativas a la raiz del proyecto."""
    camera_id: str
    source: str
    zones_file: str
    model: str = field(default_factory=get_default_model)
    tracker: str = DEFAULT_TRACKER
    confidence: float = 0.25
    min_frames: int = 4
    skip_frames: int = 0          # 0 = automatico (3 en videos de 45+ fps, 1 en el resto)
    max_seconds: float = 0.0      # 0 = video completo
    classes: Optional[List[str]] = None
    video_start: Optional[datetime] = None  # Hora real de inicio de la grabacion
    start_estimated: bool = True

    @property
    def is_live(self):
        return is_live_source(self.source)

    @property
    def public_source(self):
        return mask_rtsp_url(self.source)

    @property
    def report_name(self):
        """Nombre base de los archivos de salida (reporte JSON, video anotado)."""
        if self.is_live:
            return f"{self.camera_id}_en_vivo"
        return Path(self.source).stem


@dataclass
class AnalysisProgress:
    frames_read: int = 0
    total_frames: Optional[int] = None   # None si el video no informa su largo
    frames_inferred: int = 0
    events: int = 0
    jars_lifted: int = 0

    @property
    def percent(self):
        """Porcentaje de avance, o None si no se conoce el largo del video."""
        if not self.total_frames:
            return None
        return min(99.0, self.frames_read * 100.0 / self.total_frames)


@dataclass
class _Clock:
    started: float = field(default_factory=time.time)

    def elapsed(self):
        return time.time() - self.started


def _sane_fps(value):
    """Algunos archivos de DVR informan 0 o valores absurdos (ej. 90000)."""
    return value if value and 1.0 <= value <= 240.0 else DEFAULT_FPS


def _open_source(request, capture_factory):
    if request.is_live:
        src = request.source.strip()
        return capture_factory(int(src) if src.isdigit() else src)
    return capture_factory(str(PROJECT_ROOT / request.source))


def analyze_video(request, snapshots_dir, tracker_factory=Tracker, capture_factory=cv2.VideoCapture,
                  on_progress: Optional[Callable] = None, should_cancel: Optional[Callable] = None,
                  on_frame: Optional[Callable] = None):
    """Analiza un video completo y devuelve el reporte (sin guardarlo).

    Raises:
        AnalysisError: si no hay zonas o la fuente no se puede abrir.
    """
    zones = load_zones(PROJECT_ROOT / request.zones_file)
    if not zones:
        raise AnalysisError(f"No hay zonas configuradas en '{request.zones_file}'. "
                            "Dibujalas con scripts/draw_zones.py")

    cap = _open_source(request, capture_factory)
    if not cap.isOpened():
        cap.release()
        raise AnalysisError(f"No se pudo abrir la fuente '{request.public_source}'")

    clock = _Clock()
    analyzer = None
    cancelled = False
    progress = AnalysisProgress()
    try:
        fps = _sane_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = int(request.max_seconds * fps) if request.max_seconds > 0 else None
        if request.is_live or frame_count <= 0:
            frame_count = None  # Largo desconocido: se procesa hasta que el video termine
        progress.total_frames = min(filter(None, (frame_count, max_frames)), default=None)
        skip = request.skip_frames if request.skip_frames >= 1 else (3 if fps >= 45 else 1)

        tracker = tracker_factory(
            model_path=PROJECT_ROOT / request.model,
            confidence=request.confidence,
            tracker_type=str(PROJECT_ROOT / request.tracker),
        )
        analyzer = KitchenAnalyzer(zones, jar_classes=request.classes, min_frames=request.min_frames)
        tag = request.report_name if not request.is_live else f"{request.report_name}_{datetime.now():%Y%m%d-%H%M%S}"
        clock = _Clock()  # El tiempo de proceso no incluye la carga del modelo

        try:
            while True:
                if should_cancel and should_cancel():
                    cancelled = True
                    break
                if max_frames and progress.frames_read >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                progress.frames_read += 1
                frame_idx = progress.frames_read
                if frame_idx % skip == 0:
                    progress.frames_inferred += 1
                    results = tracker.track(frame)
                    new_events = analyzer.update(results, tracker.class_names, frame_idx, fps)
                    for ev in new_events:
                        ev.snapshot_file = save_event_snapshot(
                            frame, results, ev, analyzer.zones, snapshots_dir, request.camera_id, tag=tag
                        )
                    progress.events = len(analyzer.events)
                    progress.jars_lifted = analyzer.lifted
                    if on_frame and on_frame(frame, results, analyzer, new_events, frame_idx, fps) is False:
                        cancelled = True
                        break
                if on_progress:
                    on_progress(progress)
        except KeyboardInterrupt:
            cancelled = True  # Ctrl+C en la consola: se guarda lo contado hasta ahora
    finally:
        cap.release()

    return build_report(request, analyzer, progress, fps, clock.elapsed(), cancelled)


def build_report(request, analyzer, progress, fps, elapsed_s, cancelled):
    """Arma el diccionario de reporte que se guarda en JSON y en SQLite."""
    start = request.video_start or datetime.now()
    return {
        "camera_id": request.camera_id,
        "fuente": request.public_source,
        "modelo": request.model,
        "zonas_archivo": request.zones_file,
        "frames_procesados": progress.frames_read,
        "frames_inferidos": progress.frames_inferred,
        "duracion_video_s": round(progress.frames_read / fps, 1),
        "tiempo_proceso_s": round(elapsed_s, 1),
        "cancelado": cancelled,
        "modo": MODE_LIVE if request.is_live else MODE_VIDEO,
        "inicio_grabacion": start.isoformat(),
        "inicio_estimado": bool(request.start_estimated and not request.is_live),
        "source_key": None if request.is_live else compute_source_key(PROJECT_ROOT / request.source),
        **analyzer.summary(),
    }


def save_report(report, output_dir, report_name):
    """Guarda el reporte en output/<nombre>_conteo.json y en SQLite.

    Returns:
        (ruta_json, run_id)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report_name}_conteo.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    run_id = save_report_to_db(report)
    return report_path, run_id
