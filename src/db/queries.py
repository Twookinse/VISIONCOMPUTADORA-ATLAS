"""
Filtros comunes de consulta de eventos: camara, jornada, turno, clase, zona y modo.

Jornada: el karaoke trabaja de noche, por lo que la "fecha" de un evento no es el
dia calendario sino la JORNADA. Por defecto una jornada D va desde D a las 08:00
hasta D+1 a las 07:59 (configurable con "jornada_inicio_hora" en config/cameras.json).
Asi, el turno noche (21:00 - 04:59) queda completo dentro de la misma jornada.
"""

from datetime import datetime, time, timedelta

from sqlalchemy import Integer, cast, func

from src.config.cameras import load_cameras_config
from src.db.models import KitchenEvent, COUNTED_MODES

DEFAULT_JORNADA_START_HOUR = 8

SHIFTS = {
    "todos": {"label": "Todo el dia (24 horas)", "hours": None},
    "dia": {"label": "Turno Manana / Mediodia (08:00 - 15:59)", "hours": list(range(8, 16))},
    "tarde": {"label": "Turno Tarde (16:00 - 20:59)", "hours": list(range(16, 21))},
    "noche": {"label": "Turno Noche Karaoke (21:00 - 04:59)", "hours": [21, 22, 23, 0, 1, 2, 3, 4]},
}


class FilterError(ValueError):
    """Parametro de filtro invalido (se responde con HTTP 400)."""


def get_jornada_start_hour():
    try:
        hour = int(load_cameras_config().get("jornada_inicio_hora", DEFAULT_JORNADA_START_HOUR))
    except (TypeError, ValueError):
        hour = DEFAULT_JORNADA_START_HOUR
    return hour if 0 <= hour <= 23 else DEFAULT_JORNADA_START_HOUR


def jornada_bounds(date_str, start_hour=None):
    """Devuelve (inicio, fin) de la jornada indicada (fin exclusivo)."""
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise FilterError("Formato de fecha invalido. Usa YYYY-MM-DD")
    hour = get_jornada_start_hour() if start_hour is None else start_hour
    start = datetime.combine(target, time(hour=hour))
    return start, start + timedelta(days=1)


def jornada_hour_order(start_hour=None):
    """Horas del dia ordenadas desde el inicio de la jornada (ej. 08..23, 00..07)."""
    hour = get_jornada_start_hour() if start_hour is None else start_hour
    return [(hour + i) % 24 for i in range(24)]


def event_hour_expr():
    """Expresion SQL (SQLite) con la hora 0-23 del evento."""
    return cast(func.strftime("%H", KitchenEvent.created_at), Integer)


def apply_event_filters(query, camera_id=None, date=None, shift=None, class_name=None,
                        event_type=None, zone=None, include_demo=False):
    """Aplica los filtros estandar a una consulta sobre KitchenEvent."""
    if not include_demo:
        query = query.filter(KitchenEvent.mode.in_(COUNTED_MODES))
    if camera_id:
        query = query.filter(KitchenEvent.camera_id == camera_id)
    if date:
        start, end = jornada_bounds(date)
        query = query.filter(KitchenEvent.created_at >= start, KitchenEvent.created_at < end)
    if shift and shift != "todos":
        if shift not in SHIFTS:
            raise FilterError(f"Turno invalido. Usa: {', '.join(SHIFTS)}")
        query = query.filter(event_hour_expr().in_(SHIFTS[shift]["hours"]))
    if class_name:
        query = query.filter(KitchenEvent.class_name == class_name)
    if event_type:
        query = query.filter(KitchenEvent.event_type == event_type)
    if zone:
        query = query.filter(KitchenEvent.zone == zone)
    return query
