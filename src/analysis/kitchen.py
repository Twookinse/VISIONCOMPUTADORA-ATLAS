"""
Logica especifica de cocina: contar jarras levantadas.

Regla: una jarra que estaba dentro de una zona de reposo (la mesa) y sale
de ella se cuenta como "levantada". Cuando vuelve a entrar se cuenta como
"devuelta".

Mientras no tengamos el modelo custom (Sprint 3), la jarra no existe como
clase en YOLO. El modelo generico la detecta como "cup", "bottle" o "vase",
por eso esas son las clases por defecto.
"""

from src.events.event_engine import ZoneEventEngine

DEFAULT_JAR_CLASSES = ["cup", "bottle", "vase"]


class KitchenAnalyzer:
    """Cuenta jarras levantadas y devueltas a partir del tracking.

    Args:
        zones: Lista de Zone cargadas del JSON.
        jar_classes: Nombres de clase que se consideran jarra.
        min_frames: Frames para confirmar un cambio (ver ZoneEventEngine).
    """

    def __init__(self, zones, jar_classes=None, min_frames=3):
        self.jar_classes = set(jar_classes or DEFAULT_JAR_CLASSES)

        # Solo se usan las zonas de tipo "reposo". Si no hay ninguna,
        # se usan todas para no dejar el sistema sin zonas.
        rest_zones = [z for z in zones if z.zone_type == "reposo"]
        if not rest_zones:
            print("Aviso: no hay zonas de tipo 'reposo', se usaran todas las zonas")
            rest_zones = zones
        self.zones = rest_zones

        self.engine = ZoneEventEngine(self.zones, min_frames=min_frames)
        self.events = []

    @property
    def lifted(self):
        return sum(1 for e in self.events if e.event_type == "salida")

    @property
    def returned(self):
        return sum(1 for e in self.events if e.event_type == "entrada")

    def update(self, results, class_names, frame_idx, fps):
        """Procesa el resultado del tracker de un frame.

        Args:
            results: Objeto Results de ultralytics (de Tracker.track).
            class_names: Diccionario id -> nombre de clase del modelo.
            frame_idx: Numero de frame actual.
            fps: FPS del video.

        Returns:
            Lista de ZoneEvent nuevos en este frame.
        """
        if results.boxes.id is None:
            return []

        height, width = results.orig_shape
        detections = []
        for box in results.boxes:
            class_name = class_names[int(box.cls[0])]
            if class_name not in self.jar_classes:
                continue
            # Punto de apoyo: centro de la parte inferior del rectangulo.
            # Es donde el objeto toca la superficie. Si la base esta en la
            # mesa la jarra esta en reposo; si la base sube, fue levantada.
            x1, _, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "track_id": int(box.id[0]),
                "class_name": class_name,
                "cx": (x1 + x2) / 2,
                "cy": y2,
            })

        new_events = self.engine.update(detections, frame_idx, fps, width, height)
        self.events.extend(new_events)
        return new_events

    def summary(self):
        """Resumen del conteo para el reporte final."""
        return {
            "jarras_levantadas": self.lifted,
            "jarras_devueltas": self.returned,
            "clases_consideradas_jarra": sorted(self.jar_classes),
            "zonas": [z.name for z in self.zones],
            "eventos": [e.to_dict() for e in self.events],
        }
