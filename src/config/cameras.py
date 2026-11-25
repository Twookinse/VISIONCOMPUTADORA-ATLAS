"""
Carga y gestion de la configuracion multicamara (config/cameras.json).

Centraliza las 3 camaras de Nex (cocina_1, cocina_2, almacen) para que al llevar
el sistema al local solo sea necesario colocar el enlace rtsp_url de cada camara
sin modificar codigo fuente.

config/cameras.json NO se sube a Git porque guarda los enlaces RTSP con el usuario
y la clave del DVR. En Git solo esta la plantilla config/cameras.example.json, que
se copia automaticamente a config/cameras.json la primera vez.
"""

import json
import shutil
from pathlib import Path

from src.utils.security import is_masked, is_stream_url, mask_rtsp_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAMERAS_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.json"
CAMERAS_EXAMPLE_PATH = PROJECT_ROOT / "config" / "cameras.example.json"
FALLBACK_MODEL = "models/custom/jarras_nex_v1.pt"


class InvalidRtspUrl(ValueError):
    """El enlace indicado no es un RTSP valido (se responde con HTTP 400)."""


def _ensure_local_config(cfg_path):
    """Crea config/cameras.json desde la plantilla si todavia no existe."""
    if cfg_path == CAMERAS_CONFIG_PATH and not cfg_path.exists() and CAMERAS_EXAMPLE_PATH.exists():
        shutil.copyfile(CAMERAS_EXAMPLE_PATH, cfg_path)


def load_cameras_config(path=None):
    """Lee config/cameras.json y devuelve el diccionario completo."""
    cfg_path = Path(path) if path else CAMERAS_CONFIG_PATH
    _ensure_local_config(cfg_path)
    if not cfg_path.exists():
        return {"modelo_default": FALLBACK_MODEL, "camaras": []}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cameras_config(data, path=None):
    """Guarda cambios en config/cameras.json (ej. al actualizar enlaces RTSP)."""
    cfg_path = Path(path) if path else CAMERAS_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_default_model(path=None):
    """Modelo YOLO a usar si no se indica otro ("modelo_default" en config/cameras.json).

    Para pasar a un modelo nuevo (ej. jarras_nex_v2.pt) basta con cambiar esa linea.
    """
    model = (load_cameras_config(path).get("modelo_default") or "").strip()
    return model or FALLBACK_MODEL


def get_cameras(path=None):
    """Devuelve la lista de camaras configuradas."""
    return load_cameras_config(path).get("camaras", [])


def get_camera_by_id(camera_id, path=None):
    """Busca una camara por su ID ('cocina_1', 'cocina_2', 'almacen')."""
    if not camera_id:
        return None
    for cam in get_cameras(path):
        if cam.get("id") == camera_id:
            return cam
    return None


def validate_rtsp_url(rtsp_url):
    """Acepta vacio (sin camara), rtsp://, rtsps:// o un indice de camara USB (0-9)."""
    url = (rtsp_url or "").strip()
    if not url or is_stream_url(url) or (url.isdigit() and len(url) == 1):
        return url
    raise InvalidRtspUrl("El enlace debe comenzar con rtsp:// (o ser 0-9 para una camara USB)")


def update_camera_rtsp(camera_id, rtsp_url, path=None):
    """Actualiza el enlace RTSP de una camara en config/cameras.json.

    Si se recibe la version enmascarada del enlace actual (rtsp://admin:****@...),
    se conserva el enlace guardado: el Dashboard nunca conoce la clave real.
    """
    url = validate_rtsp_url(rtsp_url)
    data = load_cameras_config(path)
    for cam in data.get("camaras", []):
        if cam.get("id") == camera_id:
            current = (cam.get("rtsp_url") or "").strip()
            if is_masked(url):
                if not current or url != mask_rtsp_url(current):
                    raise InvalidRtspUrl(
                        "Escribe el enlace RTSP completo con la clave real (no el enlace oculto con ****)"
                    )
                return cam
            cam["rtsp_url"] = url
            save_cameras_config(data, path)
            return cam
    return None


def public_camera(cam):
    """Copia de la camara con el enlace RTSP enmascarado (para responder a la API)."""
    data = dict(cam or {})
    data["rtsp_url"] = mask_rtsp_url(data.get("rtsp_url", ""))
    return data


def unmask_source(source, path=None):
    """Traduce un enlace enmascarado (rtsp://admin:****@...) al enlace real guardado.

    Devuelve la fuente sin cambios si no esta enmascarada o no coincide con ninguna camara.
    """
    src = (source or "").strip()
    if not is_masked(src):
        return source
    for cam in get_cameras(path):
        real = (cam.get("rtsp_url") or "").strip()
        if real and mask_rtsp_url(real) == src:
            return real
    return source


def detect_camera_for_source(source, path=None):
    """Intenta deducir a que camara pertenece un archivo de video o URL RTSP.

    Ejemplo:
        'videos/COCINA 1 VIDEO 1.mp4' -> camara 'cocina_1'
        'videos/COCINA 2 VIDEO 3.mp4' -> camara 'cocina_2'
        'videos/ALMACEN VIDEO 1.mp4'  -> camara 'almacen'
    """
    cameras = get_cameras(path)
    if not cameras:
        return None

    src_str = str(source or "")
    src_upper = Path(src_str).name.upper() if src_str else ""

    # 1. Coincidencia exacta por URL RTSP configurada (real o enmascarada)
    for cam in cameras:
        rtsp = (cam.get("rtsp_url") or "").strip()
        if rtsp and src_str.strip() in (rtsp, mask_rtsp_url(rtsp)):
            return cam

    # 2. Coincidencia por prefijo de nombre de video (COCINA 1, COCINA 2, ALMACEN)
    for cam in cameras:
        prefix = (cam.get("video_prefix") or "").upper()
        if prefix and prefix in src_upper:
            return cam
        cam_id_norm = cam.get("id", "").upper().replace("_", " ")
        if cam_id_norm and cam_id_norm in src_upper:
            return cam

    # 3. Fallback a la primera camara activa
    return cameras[0]
