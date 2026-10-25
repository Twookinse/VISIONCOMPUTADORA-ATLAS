"""
Motor de eventos de zona.

Recibe la posicion de cada objeto rastreado (por su ID) y detecta cuando
entra o sale de una zona. Es generico: no sabe nada de jarras ni cocinas,
solo de objetos y zonas. La interpretacion (ej: "salida de la mesa = jarra
levantada") se hace en src/analysis/.

Para evitar falsos eventos por detecciones que parpadean, un cambio de
estado solo se confirma si se mantiene durante varios frames seguidos
(histeresis).
"""

from dataclasses import dataclass, asdict


@dataclass
class ZoneEvent:
    """Un evento de entrada o salida de una zona."""
    frame: int
    time_s: float
    track_id: int
    class_name: str
    zone: str
    event_type: str  # "entrada" o "salida"

    def to_dict(self):
        return asdict(self)


class ZoneEventEngine:
    """Detecta entradas y salidas de objetos en zonas.

    Args:
        zones: Lista de Zone (ver src/events/zones.py).
        min_frames: Frames consecutivos que debe mantenerse un cambio
            para confirmarlo. Mas alto = menos falsos eventos pero mas lento.
    """

    def __init__(self, zones, min_frames=3):
        self.zones = zones
        self.min_frames = min_frames
        # Estado por (track_id, zona): estado confirmado y cambio pendiente
        self._state = {}

    def update(self, detections, frame_idx, fps, width, height):
        """Procesa las detecciones de un frame y devuelve los eventos nuevos.

        Args:
            detections: Lista de dicts con track_id, class_name, cx, cy
                (cx, cy = centro del objeto en pixeles).
            frame_idx: Numero de frame actual.
            fps: FPS del video, para calcular el tiempo del evento.
            width, height: Tamano del frame en pixeles.

        Returns:
            Lista de ZoneEvent generados en este frame.
        """
        events = []
        for det in detections:
            for zone in self.zones:
                inside = zone.contains(det["cx"], det["cy"], width, height)
                key = (det["track_id"], zone.name)
                state = self._state.setdefault(
                    key, {"confirmed": None, "candidate": inside, "count": 0}
                )

                # Contar cuantos frames seguidos se observa el mismo estado
                if inside == state["candidate"]:
                    state["count"] += 1
                else:
                    state["candidate"] = inside
                    state["count"] = 1

                if state["count"] < self.min_frames:
                    continue

                if state["confirmed"] is None:
                    # Primera vez que se confirma: es el estado inicial, no un evento
                    state["confirmed"] = inside
                elif inside != state["confirmed"]:
                    state["confirmed"] = inside
                    events.append(ZoneEvent(
                        frame=frame_idx,
                        time_s=round(frame_idx / fps, 2),
                        track_id=det["track_id"],
                        class_name=det["class_name"],
                        zone=zone.name,
                        event_type="entrada" if inside else "salida",
                    ))
        return events
