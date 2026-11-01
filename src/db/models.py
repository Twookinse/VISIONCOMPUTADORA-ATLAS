"""
Modelos de tablas SQLAlchemy para almacenar corridas de video y eventos.

- PipelineRun: guarda la metadata de cada ejecucion del pipeline (video, modelo,
  fecha/hora, totales de jarras y objetos levantados/devueltos).
- KitchenEvent: guarda cada evento individual (salida/levantado o entrada/devuelto
  de cualquier objeto de cocina como Jarra, Vaso, Plato, Olla, etc.).
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from src.db.database import Base, create_session_factory


class PipelineRun(Base):
    """Registro de una ejecucion de conteo sobre un video o camara."""
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    source = Column(String, nullable=False)
    model = Column(String, nullable=False)
    zones_file = Column(String, nullable=False)
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
            "source": self.source,
            "model": self.model,
            "zones_file": self.zones_file,
            "frames_processed": self.frames_processed,
            "video_duration_s": self.video_duration_s,
            "process_time_s": self.process_time_s,
            "jarras_levantadas": self.jarras_levantadas,
            "jarras_devueltas": self.jarras_devueltas,
            "objetos_levantados_total": self.objetos_levantados_total,
            "objetos_devueltos_total": self.objetos_devueltos_total,
        }


class KitchenEvent(Base):
    """Evento individual detectado en una zona de la cocina."""
    __tablename__ = "kitchen_events"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    source = Column(String, nullable=False, default="")
    frame = Column(Integer, nullable=False)
    time_s = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False, index=True)
    zone = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)  # "salida" o "entrada"

    run = relationship("PipelineRun", back_populates="events")

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "frame": self.frame,
            "time_s": self.time_s,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "zone": self.zone,
            "event_type": self.event_type,
            "accion": "levantado" if self.event_type == "salida" else "devuelto",
        }


def save_report_to_db(report, db_path=None, created_at=None):
    """Guarda un diccionario de reporte (generado por run_counting.py) en SQLite.

    Returns:
        ID del PipelineRun insertado.
    """
    session_local = create_session_factory(db_path=db_path)
    db = session_local()
    try:
        base_time = created_at or datetime.now()
        run = PipelineRun(
            created_at=base_time,
            source=report.get("fuente", ""),
            model=report.get("modelo", ""),
            zones_file=report.get("zonas_archivo", ""),
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

        for ev in report.get("eventos", []):
            db.add(KitchenEvent(
                run_id=run.id,
                created_at=base_time,
                source=run.source,
                frame=ev["frame"],
                time_s=ev["time_s"],
                track_id=ev["track_id"],
                class_name=ev["class_name"],
                zone=ev["zone"],
                event_type=ev["event_type"],
            ))

        db.commit()
        return run.id
    finally:
        db.close()
