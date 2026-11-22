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


def _ensure_camera_id_columns(engine):
    """Agrega la columna camera_id si la base de datos fue creada con una version anterior."""
    with engine.begin() as conn:
        for table in ("pipeline_runs", "kitchen_events"):
            cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
            if cols and "camera_id" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN camera_id VARCHAR DEFAULT 'cocina_1'"))


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
