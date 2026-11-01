"""
Logica especifica de cocina: contar jarras y objetos de cocina levantados.

Regla: un objeto que estaba dentro de una zona de reposo (la mesa) y sale
de ella se cuenta como "levantado". Cuando vuelve a entrar se cuenta como
"devuelto".

Con el modelo custom (Sprint 3, jarras_nex_v1.pt), el sistema reconoce
directamente las clases reales de cocina (Jarra, Vaso, Plato, Olla, etc.),
ignorando estructuras fijas (Mesa, Base, Persona) y etiquetas descartadas (Jarra2).
"""

from src.events.event_engine import ZoneEventEngine

# Clases movibles de cocina del modelo custom + respaldo COCO
DEFAULT_JAR_CLASSES = [
    "Jarra",
    "Vaso",
    "Plato",
    "Olla",
    "Cuchillo",
    "Tijera",
    "Envase de cafe",
    "Envase de sal",
    "Pinia",
    "Trapo",
    "Papel Higienico",
    "cup",
    "bottle",
    "vase",
]

# Clases que representan especificamente una jarra para el indicador principal
JAR_ONLY_CLASSES = {"Jarra", "jarra", "cup", "bottle", "vase"}

# Etiquetas que siempre deben ignorarse
IGNORED_CLASSES = {"Jarra2"}


class KitchenAnalyzer:
    """Cuenta jarras y objetos de cocina levantados y devueltos a partir del tracking.

    Args:
        zones: Lista de Zone cargadas del JSON.
        jar_classes: Nombres de clase movibles a monitorear en las zonas.
        min_frames: Frames para confirmar un cambio (ver ZoneEventEngine).
    """

    def __init__(self, zones, jar_classes=None, min_frames=3):
        classes = set(jar_classes or DEFAULT_JAR_CLASSES) - IGNORED_CLASSES
        self.jar_classes = classes

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
        """Cantidad de jarras levantadas (mantiene compatibilidad con el validador)."""
        return sum(
            1 for e in self.events
            if e.event_type == "salida" and e.class_name in JAR_ONLY_CLASSES
        )

    @property
    def returned(self):
        """Cantidad de jarras devueltas."""
        return sum(
            1 for e in self.events
            if e.event_type == "entrada" and e.class_name in JAR_ONLY_CLASSES
        )

    @property
    def total_lifted(self):
        """Total de todos los objetos de cocina levantados."""
        return sum(1 for e in self.events if e.event_type == "salida")

    @property
    def total_returned(self):
        """Total de todos los objetos de cocina devueltos."""
        return sum(1 for e in self.events if e.event_type == "entrada")

    def counts_by_class(self):
        """Desglose de objetos levantados y devueltos por cada clase."""
        counts = {}
        for e in self.events:
            item = counts.setdefault(e.class_name, {"levantados": 0, "devueltos": 0})
            if e.event_type == "salida":
                item["levantados"] += 1
            else:
                item["devueltos"] += 1
        return counts

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
            if class_name in IGNORED_CLASSES or class_name not in self.jar_classes:
                continue
            # Punto de apoyo: centro de la parte inferior del rectangulo.
            # Es donde el objeto toca la superficie. Si la base esta en la
            # mesa el objeto esta en reposo; si la base sube, fue levantado.
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
            "objetos_levantados_total": self.total_lifted,
            "objetos_devueltos_total": self.total_returned,
            "conteo_por_clase": self.counts_by_class(),
            "clases_monitoreadas": sorted(self.jar_classes),
            "zonas": [z.name for z in self.zones],
            "eventos": [e.to_dict() for e in self.events],
        }
