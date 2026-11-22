"""
Carga y gestion de la configuracion multicamara (config/cameras.json).

Centraliza las 3 camaras de Nex (cocina_1, cocina_2, almacen) para que al llevar
el sistema al local solo sea necesario colocar el enlace rtsp_url de cada camara
sin modificar codigo fuente.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAMERAS_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.json"


def load_cameras_config(path=None):
    """Lee config/cameras.json y devuelve el diccionario completo."""
    cfg_path = Path(path) if path else CAMERAS_CONFIG_PATH
    if not cfg_path.exists():
        return {"modelo_default": "models/custom/jarras_nex_v1.pt", "camaras": []}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cameras_config(data, path=None):
    """Guarda cambios en config/cameras.json (ej. al actualizar enlaces RTSP)."""
    cfg_path = Path(path) if path else CAMERAS_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def update_camera_rtsp(camera_id, rtsp_url, path=None):
    """Actualiza el enlace RTSP de una camara en config/cameras.json."""
    data = load_cameras_config(path)
    for cam in data.get("camaras", []):
        if cam.get("id") == camera_id:
            cam["rtsp_url"] = (rtsp_url or "").strip()
            save_cameras_config(data, path)
            return cam
    return None


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

    # 1. Coincidencia exacta por URL RTSP configurada
    for cam in cameras:
        rtsp = (cam.get("rtsp_url") or "").strip()
        if rtsp and rtsp == src_str.strip():
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
