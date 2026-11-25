"""
Modelos de tablas SQLAlchemy para almacenar corridas de video y eventos por camara.

- PipelineRun: metadata de cada ejecucion del pipeline (camara, video, modelo,
  hora real de inicio de la grabacion, modo y totales).
- KitchenEvent: cada evento individual (salida/levantado o entrada/devuelto
  de cualquier objeto movible de cocina: Jarra, Taper, Olla, Vaso, etc.).
  Su created_at es la HORA REAL del evento (inicio de la grabacion + time_s).

Modos de una corrida:
- "video": analisis completo de una grabacion (cuenta en las metricas).
- "live":  monitoreo de una camara real (RTSP / USB) (cuenta en las metricas).
- "demo":  reproduccion con marcas de un video grabado. Se guarda para ver la
           evidencia, pero se EXCLUYE de las metricas para no duplicar conteos.
- "prueba": analisis de un TRAMO de un video (--start-seconds / --max-seconds),
           para comparar con un conteo manual. Se EXCLUYE de las metricas y no
           reemplaza el analisis completo del mismo video.
"""

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, or_
from sqlalchemy.orm import relationship

from src.db.database import Base, create_session_factory, PROJECT_ROOT
from src.config.cameras import detect_camera_for_source
from src.utils.video_meta import parse_datetime_input, compute_source_key

MODE_VIDEO = "video"
MODE_LIVE = "live"
MODE_DEMO = "demo"
MODE_TEST = "prueba"
COUNTED_MODES = (MODE_VIDEO, MODE_LIVE)

SNAPSHOTS_DIR = PROJECT_ROOT / "output" / "snapshots"


class PipelineRun(Base):
    """Registro de una ejecucion de conteo sobre un video o camara."""
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)  # Cuando se analizo
    camera_id = Column(String, nullable=False, default="cocina_1", index=True)
    source = Column(String, nullable=False)
    model = Column(String, nullable=False)
    zones_file = Column(String, nullable=False)
    mode = Column(String, nullable=False, default=MODE_VIDEO, index=True)
    video_start = Column(DateTime, nullable=True)            # Hora real de inicio de la grabacion
    start_time_estimated = Column(Boolean, default=True)     # True si no se conocia la hora real
    source_key = Column(String, nullable=True, index=True)   # Huella del archivo (deduplicacion)
    frames_processed = Column(Integer, default=0)
    video_duration_s = Column(Float, default=0.0)
    process_time_s = Column(Float, default=0.0)
    jarras_levantadas = Column(Integer, default=0)
    jarras_devueltas = Column(Integer, default=0)
    objetos_levantados_total = Column(Integer, default=0)
    objetos_devueltos_total = Column(Integer, default=0)

    events = relationship(
        "KitchenEvent",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "camera_id": self.camera_id or "cocina_1",
            "source": self.source,
            "model": self.model,
            "zones_file": self.zones_file,
            "modo": self.mode or MODE_VIDEO,
            "inicio_grabacion": self.video_start.isoformat() if self.video_start else None,
            "inicio_estimado": bool(self.start_time_estimated),
            "frames_processed": self.frames_processed,
            "video_duration_s": self.video_duration_s,
            "process_time_s": self.process_time_s,
            "jarras_levantadas": self.jarras_levantadas,
            "jarras_devueltas": self.jarras_devueltas,
            "objetos_levantados_total": self.objetos_levantados_total,
            "objetos_devueltos_total": self.objetos_devueltos_total,
        }


class KitchenEvent(Base):
    """Evento individual detectado en una zona ROI de una camara."""
    __tablename__ = "kitchen_events"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)  # Hora REAL del evento
    camera_id = Column(String, nullable=False, default="cocina_1", index=True)
    mode = Column(String, nullable=False, default=MODE_VIDEO, index=True)
    source = Column(String, nullable=False, default="")
    frame = Column(Integer, nullable=False)
    time_s = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False, index=True)
    zone = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)  # "salida" o "entrada"
    snapshot_file = Column(String, nullable=True, default="")

    run = relationship("PipelineRun", back_populates="events")

    def to_dict(self):
        snap = (self.snapshot_file or "").strip()
        return {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "camera_id": self.camera_id or "cocina_1",
            "modo": self.mode or MODE_VIDEO,
            "source": self.source,
            "frame": self.frame,
            "time_s": self.time_s,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "zone": self.zone,
            "event_type": self.event_type,
            "accion": "levantado" if self.event_type == "salida" else "devuelto",
            "snapshot_file": snap,
            "snapshot_url": f"/api/snapshots/{snap}" if snap else None,
        }


def _delete_snapshot_files(filenames):
    """Elimina del disco las fotos de evidencia de corridas reemplazadas."""
    for name in filenames:
        if not name:
            continue
        try:
            (SNAPSHOTS_DIR / Path(name).name).unlink(missing_ok=True)
        except OSError:
            pass


def save_report_to_db(report, db_path=None, created_at=None, video_start=None, mode=None):
    """Guarda un diccionario de reporte (generado por el pipeline) en SQLite.

    - Cada evento se guarda con su hora real: inicio de la grabacion + time_s.
    - En modo "video", si el mismo archivo ya habia sido analizado (misma huella
      o misma ruta), la corrida anterior se REEMPLAZA para no duplicar conteos.

    Args:
        report: Diccionario del reporte (ver analyzer.summary()).
        db_path: Ruta alternativa de la base de datos (pruebas).
        created_at: Alias historico de video_start.
        video_start: Hora real de inicio de la grabacion (datetime o ISO).
        mode: "video" (default), "live" o "demo".

    Returns:
        ID del PipelineRun insertado.
    """
    session_local = create_session_factory(db_path=db_path)
    db = session_local()
    removed_snapshots = []
    try:
        source = report.get("fuente", "")
        mode = mode or report.get("modo") or MODE_VIDEO

        start = parse_datetime_input(video_start or created_at or report.get("inicio_grabacion"))
        estimated = bool(report.get("inicio_estimado", start is None))
        if start is None:
            start = datetime.now()
            estimated = True

        cam_id = report.get("camera_id")
        if not cam_id:
            detected = detect_camera_for_source(source)
            cam_id = detected["id"] if detected else "cocina_1"

        source_key = report.get("source_key")
        if source_key is None and mode == MODE_VIDEO and source:
            source_key = compute_source_key(PROJECT_ROOT / source)

        # Deduplicacion: reemplazar analisis anteriores del mismo video
        if mode == MODE_VIDEO:
            conditions = [PipelineRun.source == source]
            if source_key:
                conditions.append(PipelineRun.source_key == source_key)
            previous = (
                db.query(PipelineRun)
                .filter(PipelineRun.mode == MODE_VIDEO, or_(*conditions))
                .all()
            )
            for old_run in previous:
                removed_snapshots.extend(e.snapshot_file for e in old_run.events)
                db.delete(old_run)
            db.flush()

        run = PipelineRun(
            created_at=datetime.now(),
            camera_id=cam_id,
            source=source,
            model=report.get("modelo", ""),
            zones_file=report.get("zonas_archivo", ""),
            mode=mode,
            video_start=start,
            start_time_estimated=estimated,
            source_key=source_key,
            frames_processed=report.get("frames_procesados", 0),
            video_duration_s=report.get("duracion_video_s", 0.0),
            process_time_s=report.get("tiempo_proceso_s", 0.0),
            jarras_levantadas=report.get("jarras_levantadas", 0),
            jarras_devueltas=report.get("jarras_devueltas", 0),
            objetos_levantados_total=report.get(
                "objetos_levantados_total", report.get("jarras_levantadas", 0)
            ),
            objetos_devueltos_total=report.get(
                "objetos_devueltos_total", report.get("jarras_devueltas", 0)
            ),
        )
        db.add(run)
        db.flush()

        new_snapshots = set()
        for ev in report.get("eventos", []):
            snap = ev.get("snapshot_file", "")
            new_snapshots.add(snap)
            db.add(KitchenEvent(
                run_id=run.id,
                created_at=start + timedelta(seconds=float(ev.get("time_s", 0.0))),
                camera_id=cam_id,
                mode=mode,
                source=run.source,
                frame=ev["frame"],
                time_s=ev["time_s"],
                track_id=ev["track_id"],
                class_name=ev["class_name"],
                zone=ev["zone"],
                event_type=ev["event_type"],
                snapshot_file=snap,
            ))

        db.commit()
        # Solo borrar fotos antiguas que no sean reutilizadas por la corrida nueva
        _delete_snapshot_files(s for s in removed_snapshots if s not in new_snapshots)
        return run.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
