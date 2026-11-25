"""
Ajustes del servidor leidos desde variables de entorno o desde un archivo .env
en la raiz del proyecto (ignorado por Git). Ver .env.example.

Las variables ya definidas en el sistema tienen prioridad sobre el archivo .env.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
DEFAULT_MAX_UPLOAD_MB = 4096


def load_env_file(path=ENV_FILE):
    """Carga pares CLAVE=valor de un archivo .env sin sobrescribir el entorno."""
    path = Path(path)
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()


def auth_user():
    """Usuario para el acceso remoto (HTTP Basic). Default: 'nex'."""
    return os.environ.get("NEX_USER", "nex").strip() or "nex"


def auth_password():
    """Contrasena para el acceso remoto. Vacia = acceso remoto bloqueado."""
    return os.environ.get("NEX_PASSWORD", "").strip()


def cors_origins():
    """Origenes permitidos para el Dashboard en modo desarrollo (Vite)."""
    raw = os.environ.get("NEX_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def max_upload_bytes():
    """Tamano maximo de un video subido desde el Dashboard."""
    try:
        mb = int(os.environ.get("NEX_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return max(1, mb) * 1024 * 1024
