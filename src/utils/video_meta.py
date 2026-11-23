"""
Metadatos de videos grabados: hora real de inicio y huella unica del archivo.

- resolve_video_start(): determina a que hora real comenzo la grabacion para que
  cada evento se guarde con su hora verdadera (inicio + segundo del evento) y no
  con la hora en que se ejecuto el analisis.
  Prioridad: valor indicado por el usuario > fecha/hora en el nombre del archivo
  del DVR > hora actual (marcada como "estimada").
- compute_source_key(): huella del contenido del video para detectar que el mismo
  archivo se esta analizando otra vez (aunque se haya renombrado).
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

# Formatos comunes de DVR/NVR: 20261122213000, 20261122_213000,
# 2026-11-22_21-30-00, 2026-11-22 21.30.00, CH01_20261122T213000, etc.
_FILENAME_DT_RE = re.compile(
    r"(?<!\d)(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ T_\-.]?(\d{2})[-_.:h]?(\d{2})(?:[-_.:m]?(\d{2}))?(?!\d)"
)


def parse_start_from_filename(name):
    """Intenta extraer la fecha y hora de inicio desde el nombre del archivo."""
    stem = Path(str(name or "")).stem
    for match in _FILENAME_DT_RE.finditer(stem):
        y, mo, d, h, mi, s = match.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
        except ValueError:
            continue
    return None


def parse_datetime_input(value):
    """Convierte un texto ISO ('2026-11-22T21:30' o '2026-11-22 21:30:00') a datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def resolve_video_start(source, override=None):
    """Devuelve (inicio_datetime, es_estimado).

    Args:
        source: Ruta o nombre del video.
        override: Fecha/hora indicada manualmente (texto ISO o datetime).
    """
    manual = parse_datetime_input(override)
    if manual:
        return manual, False
    from_name = parse_start_from_filename(source)
    if from_name:
        return from_name, False
    return datetime.now(), True


def compute_source_key(path, chunk_size=4 * 1024 * 1024):
    """Huella rapida del video (tamano + primeros y ultimos 4 MB). None si no existe."""
    path = Path(path)
    if not path.is_file():
        return None
    size = path.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(chunk_size))
        if size > chunk_size * 2:
            f.seek(-chunk_size, 2)
            digest.update(f.read(chunk_size))
    return digest.hexdigest()
