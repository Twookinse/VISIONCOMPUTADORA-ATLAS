"""
Utilidades de seguridad:

- Enmascarar credenciales de enlaces RTSP (rtsp://admin:****@IP/...) para que la
  clave del DVR nunca se muestre en el Dashboard, la API, la base de datos ni
  sobre el video.
- Validar que las rutas recibidas por la API (modelo, zonas, tracker, video)
  apunten a archivos permitidos dentro del proyecto. Evita que se lea o cargue
  cualquier archivo del disco (ej. un .pt malicioso, que puede ejecutar codigo).
- Limpiar nombres de archivos subidos desde el Dashboard.
"""

import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MASK = "****"
STREAM_SCHEMES = ("rtsp://", "rtsps://")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MODEL_EXTENSIONS = {".pt", ".onnx"}

# Carpetas permitidas para cada tipo de archivo (relativas a la raiz del proyecto)
VIDEOS_DIR = "videos"
MODELS_DIR = "models"
ZONES_DIR = "config/zones"
TRACKERS_DIR = "config/trackers"

# Parametros de query donde algunos DVR (Dahua, XMEye...) ponen la clave
_SECRET_QUERY_KEYS = {"password", "passwd", "pass", "pwd", "contrasena", "token"}


class UnsafePathError(ValueError):
    """La ruta solicitada no esta permitida (se responde con HTTP 400)."""


# ---------------------------------------------------------------------
# Credenciales RTSP
# ---------------------------------------------------------------------
def is_stream_url(value):
    return str(value or "").strip().lower().startswith(STREAM_SCHEMES)


def mask_rtsp_url(url):
    """Oculta la clave de un enlace RTSP. Otros valores se devuelven sin cambios.

    Ejemplos:
        rtsp://admin:clave@192.168.1.10:554/ch1 -> rtsp://admin:****@192.168.1.10:554/ch1
        rtsp://192.168.1.10/live?user=admin&password=x -> ...?user=admin&password=****
    """
    text = str(url or "").strip()
    if not is_stream_url(text):
        return text
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.split("://", 1)[0] + "://" + MASK

    netloc = parts.netloc
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:{MASK}@{hostport}" if user else f"{MASK}@{hostport}"

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        if any(k.lower() in _SECRET_QUERY_KEYS for k, _ in pairs):
            query = urlencode(
                [(k, MASK if k.lower() in _SECRET_QUERY_KEYS else v) for k, v in pairs],
                safe="*",
            )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def is_masked(url):
    return MASK in str(url or "")


def mask_text(text, secret_url):
    """Reemplaza un enlace RTSP real por su version enmascarada dentro de un texto."""
    text = str(text or "")
    secret_url = str(secret_url or "").strip()
    if secret_url and is_stream_url(secret_url):
        text = text.replace(secret_url, mask_rtsp_url(secret_url))
    return text


# ---------------------------------------------------------------------
# Rutas de archivos
# ---------------------------------------------------------------------
def _to_relative(path):
    return path.relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_project_file(rel_path, base_dir, allowed_extensions, must_exist=True):
    """Valida que rel_path este dentro de PROJECT_ROOT/base_dir con una extension permitida.

    Returns:
        Ruta relativa normalizada (ej. "models/custom/jarras_nex_v1.pt").

    Raises:
        UnsafePathError: ruta fuera de la carpeta permitida o extension no valida.
        FileNotFoundError: el archivo no existe (si must_exist=True).
    """
    if not rel_path or not str(rel_path).strip():
        raise UnsafePathError("Ruta vacia")
    base = (PROJECT_ROOT / base_dir).resolve()
    candidate = (PROJECT_ROOT / str(rel_path).strip()).resolve()
    if not candidate.is_relative_to(base):
        raise UnsafePathError(f"La ruta '{rel_path}' debe estar dentro de '{base_dir}/'")
    if candidate.suffix.lower() not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise UnsafePathError(f"Extension no permitida en '{rel_path}' (usa: {allowed})")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"No se encontro el archivo '{rel_path}'")
    return _to_relative(candidate)


def validate_model_path(rel_path):
    return resolve_project_file(rel_path, MODELS_DIR, MODEL_EXTENSIONS)


def validate_zones_path(rel_path):
    return resolve_project_file(rel_path, ZONES_DIR, {".json"})


def validate_tracker_path(rel_path):
    return resolve_project_file(rel_path, TRACKERS_DIR, {".yaml", ".yml"})


def validate_video_path(rel_path):
    return resolve_project_file(rel_path, VIDEOS_DIR, VIDEO_EXTENSIONS)


def validate_source(source, allow_stream=True):
    """Valida la fuente de video.

    Acepta: enlace rtsp:// o rtsps:// (si allow_stream), indice de camara USB
    ("0".."9") o un video dentro de videos/.

    Returns:
        La fuente normalizada (texto).
    """
    src = str(source or "").strip()
    if not src:
        raise UnsafePathError("No se indico una fuente de video")
    if is_stream_url(src) or (src.isdigit() and len(src) == 1):
        if not allow_stream:
            raise UnsafePathError("Esta operacion solo acepta videos grabados dentro de videos/")
        return src
    return validate_video_path(src)


# ---------------------------------------------------------------------
# Subida de archivos
# ---------------------------------------------------------------------
_UNSAFE_NAME_CHARS = re.compile(r"[^\w\s.\-()]", re.UNICODE)


def sanitize_upload_name(filename, default="video_subido.mp4"):
    """Nombre seguro para un video subido: sin carpetas ni caracteres raros."""
    # Quitar cualquier carpeta, tanto con "/" como con "\\" (igual en Windows y Linux)
    name = re.split(r"[\\/]", str(filename or ""))[-1]
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip(" .")
    if not name:
        name = default
    stem, suffix = Path(name).stem[:120], Path(name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise UnsafePathError(f"Solo se permiten videos ({allowed})")
    return f"{stem}{suffix}"


def unique_destination(directory, name):
    """Evita sobrescribir: 'video.mp4' -> 'video (1).mp4' si ya existe."""
    directory = Path(directory)
    dest = directory / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while dest.exists():
        dest = directory / f"{stem} ({n}){suffix}"
        n += 1
    return dest
