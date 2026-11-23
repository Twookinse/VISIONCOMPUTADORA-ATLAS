"""
Conexion y sesion de base de datos SQLite con SQLAlchemy.

Por defecto guarda el archivo SQLite en output/nex_vision.db (ignorado por Git),
pero permite usar otra ruta o base en memoria mediante la variable de entorno
NEX_DB_URL (util para pruebas automatizadas).
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "output" / "nex_vision.db"

Base = declarative_base()
_ENGINES = {}


def get_db_url(db_path=None):
    """Obtiene la URL de conexion SQLite."""
    if db_path:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"
    env_url = os.environ.get("NEX_DB_URL")
    if env_url:
        return env_url
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def _table_columns(conn, table):
    return [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]


def _ensure_camera_id_columns(engine):
    """Migra BDs de versiones anteriores agregando las columnas nuevas.

    Al agregar la columna 'mode' por primera vez tambien se ordenan los datos viejos:
    - Las reproducciones con marcas de videos grabados ("EN_VIVO: videos/...") pasan
      a modo 'demo' (se excluyen de las metricas).
    - Los analisis repetidos del mismo video se reducen al mas reciente.
    """
    with engine.begin() as conn:
        for table in ("pipeline_runs", "kitchen_events"):
            cols = _table_columns(conn, table)
            if cols and "camera_id" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN camera_id VARCHAR DEFAULT 'cocina_1'"))
        ev_cols = _table_columns(conn, "kitchen_events")
        if ev_cols and "snapshot_file" not in ev_cols:
            conn.execute(text("ALTER TABLE kitchen_events ADD COLUMN snapshot_file VARCHAR DEFAULT ''"))

        run_cols = _table_columns(conn, "pipeline_runs")
        if not run_cols:
            return
        if "video_start" not in run_cols:
            conn.execute(text("ALTER TABLE pipeline_runs ADD COLUMN video_start DATETIME"))
        if "start_time_estimated" not in run_cols:
            conn.execute(text("ALTER TABLE pipeline_runs ADD COLUMN start_time_estimated BOOLEAN DEFAULT 1"))
        if "source_key" not in run_cols:
            conn.execute(text("ALTER TABLE pipeline_runs ADD COLUMN source_key VARCHAR"))

        if "mode" not in run_cols:
            conn.execute(text("ALTER TABLE pipeline_runs ADD COLUMN mode VARCHAR DEFAULT 'video'"))
            conn.execute(text("""
                UPDATE pipeline_runs SET mode = CASE
                    WHEN source LIKE 'EN_VIVO: rtsp%' OR source GLOB 'EN_VIVO: [0-9]*' THEN 'live'
                    WHEN source LIKE 'EN_VIVO:%' THEN 'demo'
                    ELSE 'video' END
            """))
            # Conservar solo el analisis mas reciente de cada video grabado
            dup_runs = """
                SELECT id FROM pipeline_runs WHERE mode = 'video' AND id NOT IN (
                    SELECT MAX(id) FROM pipeline_runs WHERE mode = 'video' GROUP BY source
                )
            """
            conn.execute(text(f"DELETE FROM kitchen_events WHERE run_id IN ({dup_runs})"))
            conn.execute(text(f"DELETE FROM pipeline_runs WHERE id IN ({dup_runs})"))

        if "mode" not in _table_columns(conn, "kitchen_events"):
            conn.execute(text("ALTER TABLE kitchen_events ADD COLUMN mode VARCHAR DEFAULT 'video'"))
            conn.execute(text("""
                UPDATE kitchen_events SET mode = COALESCE(
                    (SELECT mode FROM pipeline_runs WHERE pipeline_runs.id = kitchen_events.run_id),
                    'video'
                )
            """))


def create_session_factory(db_path=None):
    """Crea o reutiliza el motor SQLAlchemy, inicializa las tablas y devuelve SessionLocal."""
    url = get_db_url(db_path)
    if url not in _ENGINES:
        engine = create_engine(url, connect_args={"check_same_thread": False})
        from src.db import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        _ensure_camera_id_columns(engine)
        _ENGINES[url] = engine
    return sessionmaker(autocommit=False, autoflush=False, bind=_ENGINES[url])


def dispose_all_engines():
    """Cierra las conexiones abiertas (util al limpiar archivos temporales en Windows)."""
    for eng in _ENGINES.values():
        eng.dispose()
    _ENGINES.clear()


def get_db():
    """Dependencia para FastAPI que entrega una sesion de base de datos."""
    session_local = create_session_factory()
    db = session_local()
    try:
        yield db
    finally:
        db.close()
