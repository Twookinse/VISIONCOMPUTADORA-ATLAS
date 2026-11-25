"""
Motor de eventos de zona.

Recibe la posicion de cada objeto rastreado (por su ID) y detecta cuando
entra o sale de una zona. Es generico: no sabe nada de jarras ni cocinas,
solo de objetos y zonas. La interpretacion (ej: "salida de la mesa = jarra
levantada") se hace en src/analysis/.

Reglas:

1. UNA ubicacion por objeto. Cada objeto esta en una sola zona o "fuera"
   (en la mano, en el aire). Si las zonas se superponen, el objeto se queda en
   la zona donde ya estaba; si es nuevo, se elige la zona donde queda mas adentro.
   Asi un objeto en la interseccion de dos zonas no genera eventos dobles.

2. Histeresis temporal (min_frames). Un cambio de ubicacion solo se confirma si
   se mantiene varios frames seguidos (evita detecciones que parpadean).

3. Histeresis espacial (margin_ratio). Para SALIR de su zona el objeto debe
   alejarse del borde un margen minimo. Un objeto apoyado justo en el borde de
   la mesa no "sale y entra" cada vez que la caja de deteccion tiembla.

4. Cambios de ID del tracker. Si una mano tapa un objeto y el tracker le asigna
   un ID nuevo, el ID nuevo hereda el estado del anterior cuando: es de la misma
   clase, el ID anterior dejo de verse hace poco (lost_seconds) y aparecio cerca
   (reid_max_dist_ratio). Asi se cuenta la salida aunque el objeto reaparezca ya
   levantado, y no se cuenta una "entrada" sin su "salida".

5. Memoria acotada. Los objetos que no se ven por mas de lost_seconds se
   olvidan, para que una sesion en vivo de muchas horas no acumule estado.

Un objeto que pasa directamente de una zona a otra (ej. de la mesa al lavado)
genera "salida" de la primera y "entrada" a la segunda: fue levantado de un
lugar y dejado en otro.
"""

import math
from dataclasses import dataclass, asdict

import cv2

OUTSIDE = None  # Ubicacion "fuera de todas las zonas"
_UNKNOWN = object()  # Ubicacion todavia no confirmada


@dataclass
class ZoneEvent:
    """Un evento de entrada o salida de una zona."""
    frame: int
    time_s: float
    track_id: int
    class_name: str
    zone: str
    event_type: str  # "entrada" o "salida"
    snapshot_file: str = ""

    def to_dict(self):
        return asdict(self)


class _TrackState:
    """Estado interno de un objeto rastreado."""

    __slots__ = ("track_id", "class_name", "confirmed", "candidate", "count",
                 "last_seen_s", "last_xy")

    def __init__(self, track_id, class_name, now_s, xy):
        self.track_id = track_id
        self.class_name = class_name
        self.confirmed = _UNKNOWN   # Zona confirmada (nombre), OUTSIDE o _UNKNOWN
        self.candidate = _UNKNOWN   # Ubicacion observada que aun no se confirma
        self.count = 0              # Frames seguidos observando el candidato
        self.last_seen_s = now_s
        self.last_xy = xy


class ZoneEventEngine:
    """Detecta entradas y salidas de objetos en zonas.

    Args:
        zones: Lista de Zone (ver src/events/zones.py).
        min_frames: Frames procesados seguidos que debe mantenerse un cambio
            para confirmarlo. Mas alto = menos falsos eventos pero mas lento.
        margin_ratio: Margen para salir de una zona, como fraccion del lado menor
            del frame (0.01 = ~7 px en 720p, ~11 px en 1080p).
        lost_seconds: Tiempo que se recuerda un objeto que dejo de verse
            (para heredar su estado si reaparece con otro ID).
        reid_max_dist_ratio: Distancia maxima, como fraccion de la diagonal del
            frame, para considerar que un ID nuevo es el mismo objeto perdido.
    """

    def __init__(self, zones, min_frames=3, margin_ratio=0.01,
                 lost_seconds=6.0, reid_max_dist_ratio=0.15):
        self.zones = list(zones)
        self.min_frames = max(1, int(min_frames))
        self.margin_ratio = margin_ratio
        self.lost_seconds = lost_seconds
        self.reid_max_dist_ratio = reid_max_dist_ratio
        self._tracks = {}            # track_id -> _TrackState
        self._polygons = {}          # (ancho, alto) -> {zona: poligono en pixeles}

    # -----------------------------------------------------------------
    # Geometria
    # -----------------------------------------------------------------
    def _zone_polygons(self, width, height):
        key = (width, height)
        if key not in self._polygons:
            self._polygons = {key: {z.name: z.to_pixels(width, height) for z in self.zones}}
        return self._polygons[key]

    def _locate(self, x, y, previous, polygons, margin):
        """Devuelve la zona donde esta el punto (x, y), o OUTSIDE.

        Distancia con signo de cv2.pointPolygonTest: > 0 adentro, < 0 afuera.
        """
        point = (float(x), float(y))
        if isinstance(previous, str) and previous in polygons:
            # Sigue en su zona mientras no se aleje del borde mas que el margen
            if cv2.pointPolygonTest(polygons[previous], point, True) >= -margin:
                return previous
        best, best_dist = OUTSIDE, -1.0
        for name, poly in polygons.items():
            dist = cv2.pointPolygonTest(poly, point, True)
            if dist >= 0 and dist > best_dist:
                best, best_dist = name, dist
        return best

    # -----------------------------------------------------------------
    # Cambios de ID
    # -----------------------------------------------------------------
    def _adopt_lost_track(self, track_id, class_name, xy, now_s, seen_ids, max_dist):
        """Busca un objeto perdido que probablemente sea este ID nuevo."""
        best, best_dist = None, max_dist
        for state in self._tracks.values():
            if state.track_id in seen_ids or state.class_name != class_name:
                continue
            if state.confirmed is _UNKNOWN or now_s - state.last_seen_s > self.lost_seconds:
                continue
            dist = math.dist(xy, state.last_xy)
            if dist <= best_dist:
                best, best_dist = state, dist
        if best is None:
            return None
        del self._tracks[best.track_id]
        best.track_id = track_id
        best.candidate, best.count = _UNKNOWN, 0
        return best

    def _forget_old_tracks(self, now_s):
        stale = [tid for tid, s in self._tracks.items() if now_s - s.last_seen_s > self.lost_seconds]
        for tid in stale:
            del self._tracks[tid]

    # -----------------------------------------------------------------
    # API publica
    # -----------------------------------------------------------------
    @property
    def tracked_count(self):
        """Cantidad de objetos en memoria (util para monitoreo y pruebas)."""
        return len(self._tracks)

    def update(self, detections, frame_idx, fps, width, height):
        """Procesa las detecciones de un frame y devuelve los eventos nuevos.

        Args:
            detections: Lista de dicts con track_id, class_name, cx, cy
                (cx, cy = punto de apoyo del objeto en pixeles).
            frame_idx: Numero de frame actual.
            fps: FPS del video, para calcular el tiempo del evento.
            width, height: Tamano del frame en pixeles.

        Returns:
            Lista de ZoneEvent generados en este frame.
        """
        fps = fps or 30.0
        now_s = frame_idx / fps
        polygons = self._zone_polygons(width, height)
        margin = self.margin_ratio * min(width, height)
        max_reid_dist = self.reid_max_dist_ratio * math.hypot(width, height)
        seen_ids = {det["track_id"] for det in detections}

        events = []
        for det in detections:
            track_id, class_name = det["track_id"], det["class_name"]
            xy = (float(det["cx"]), float(det["cy"]))

            state = self._tracks.get(track_id)
            if state is None:
                state = self._adopt_lost_track(track_id, class_name, xy, now_s, seen_ids, max_reid_dist)
                if state is None:
                    state = _TrackState(track_id, class_name, now_s, xy)
                self._tracks[track_id] = state

            reference = state.candidate if state.candidate is not _UNKNOWN else state.confirmed
            location = self._locate(xy[0], xy[1], reference, polygons, margin)
            state.last_seen_s, state.last_xy = now_s, xy

            # Histeresis temporal: contar frames seguidos con la misma ubicacion
            if location == state.candidate:
                state.count += 1
            else:
                state.candidate, state.count = location, 1
            if state.count < self.min_frames or location == state.confirmed:
                continue

            previous = state.confirmed
            state.confirmed = location
            if previous is _UNKNOWN:
                continue  # Primera confirmacion: es el estado inicial, no un evento

            event_args = dict(frame=frame_idx, time_s=round(now_s, 2),
                              track_id=track_id, class_name=class_name)
            if previous is not OUTSIDE:
                events.append(ZoneEvent(zone=previous, event_type="salida", **event_args))
            if location is not OUTSIDE:
                events.append(ZoneEvent(zone=location, event_type="entrada", **event_args))

        self._forget_old_tracks(now_s)
        return events
