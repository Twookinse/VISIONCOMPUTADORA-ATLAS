"""
Logica especifica de cocina para Nex: contar jarras y utensilios levantados.

Regla: un objeto movible que estaba dentro de una zona ROI (mesa, estante,
lavado, etc.) y sale de ella se cuenta como "levantado" (salida). Cuando
vuelve a entrar se cuenta como "devuelto" (entrada).

Ademas, normaliza las etiquetas numeradas del modelo custom de Roboflow
(ej. Jarra 1, Jarra1..Jarra5 -> "Jarra"; Taper2..Taper5 -> "Taper";
olla/olla1 -> "Olla") e ignora elementos fijos y personas (Persona1..6,
Microondas, Basurero, Licuadora) para evitar falsos conteos.
"""

from src.events.event_engine import ZoneEventEngine

# Mapeo de etiquetas del modelo custom de Nex hacia nombres limpios y unificados
CLASS_ALIASES = {
    # Jarras (todas unificadas en "Jarra")
    "Jarra 1": "Jarra",
    "Jarra1": "Jarra",
    "Jarra2": "Jarra",
    "Jarra3": "Jarra",
    "Jarra4": "Jarra",
    "Jarra5": "Jarra",
    "Jarra": "Jarra",
    "jarra": "Jarra",
    # Jarra de licuadora (movible, separada para claridad)
    "JarraLicuadora": "Jarra Licuadora",
    # Tapers / contenedores
    "Taper": "Taper",
    "Taper2": "Taper",
    "Taper3": "Taper",
    "Taper4": "Taper",
    "Taper5": "Taper",
    # Ollas y tapas
    "olla": "Olla",
    "olla1": "Olla",
    "Olla": "Olla",
    "Tapa de olla": "Tapa de Olla",
    "Tapa1": "Tapa de Olla",
    # Vasos, cervezas y botellas
    "Vaso1": "Vaso",
    "Vaso": "Vaso",
    "Cerveza": "Cerveza",
    "CajaCerveza": "Caja de Cerveza",
    "botella": "Botella",
    "Botella": "Botella",
    # Utensilios y envases de cocina
    "Cuchillo1": "Cuchillo",
    "Cuchillo": "Cuchillo",
    "Cernidor": "Cernidor",
    "Balde": "Balde",
    "Envase Ketchup": "Envase Ketchup",
    "Envase Llajua": "Envase Llajua",
    "Envase salsa": "Envase Salsa",
    "Envase para platanos": "Envase Platanos",
    "Productos culinarios": "Productos Culinarios",
    # Respaldo COCO (solo si se usa un modelo preentrenado)
    "cup": "Vaso",
    "bottle": "Botella",
    "vase": "Jarra",
}

# Clases fijas o personas que NO deben contarse como objetos levantados
IGNORED_CLASSES = {
    "Persona",
    "Persona1",
    "Persona2",
    "Persona3",
    "Persona4",
    "Persona5",
    "Persona6",
    "person",
    "Microondas",
    "Basurero",
    "Licuadora",
    "Licuadora2",
    "Mesa",
    "Base",
}

# Lista limpia de clases movibles principales monitoreadas por defecto
DEFAULT_JAR_CLASSES = [
    "Jarra",
    "Jarra Licuadora",
    "Olla",
    "Tapa de Olla",
    "Vaso",
    "Taper",
    "Cerveza",
    "Caja de Cerveza",
    "Botella",
    "Cuchillo",
    "Cernidor",
    "Balde",
    "Envase Ketchup",
    "Envase Llajua",
    "Envase Salsa",
    "Envase Platanos",
    "Productos Culinarios",
]

# Clases que suman en el indicador principal de Jarras
JAR_ONLY_CLASSES = {"Jarra", "Jarra Licuadora", "jarra"}


def normalize_class_name(raw_name):
    """Devuelve el nombre unificado de una clase o None si debe ignorarse."""
    if not raw_name:
        return None
    name = str(raw_name).strip()
    if name in IGNORED_CLASSES:
        return None
    if name in CLASS_ALIASES:
        return CLASS_ALIASES[name]
    # Ignorar automaticamente cualquier futura etiqueta "PersonaN"
    if name.lower().startswith("persona"):
        return None
    return name


class KitchenAnalyzer:
    """Cuenta jarras y objetos de cocina levantados y devueltos a partir del tracking.

    Args:
        zones: Lista de Zone cargadas del JSON de la camara correspondiente.
        jar_classes: Lista opcional de clases normalizadas a monitorear.
            Si es None, monitorea todas las clases movibles (ignorando fijos/personas).
        min_frames: Frames consecutivos para confirmar un cambio de zona.
        **engine_options: Ajustes opcionales del motor de eventos (margin_ratio,
            lost_seconds, reid_max_dist_ratio). Ver ZoneEventEngine.
    """

    def __init__(self, zones, jar_classes=None, min_frames=3, **engine_options):
        if jar_classes is None:
            self.monitor_all_movable = True
            self.jar_classes = set(DEFAULT_JAR_CLASSES)
        else:
            self.monitor_all_movable = False
            normalized = set()
            for c in jar_classes:
                norm = normalize_class_name(c)
                if norm:
                    normalized.add(norm)
            self.jar_classes = normalized

        # Si hay zonas de tipo "reposo" se priorizan; si no (como en cocina_1/2/almacen
        # donde son "general" y "servicio"), se utilizan todas las zonas configuradas.
        rest_zones = [z for z in zones if z.zone_type.lower() == "reposo"]
        self.zones = rest_zones if rest_zones else list(zones)

        self.engine = ZoneEventEngine(self.zones, min_frames=min_frames, **engine_options)
        self.events = []

    @property
    def lifted(self):
        """Cantidad de jarras levantadas."""
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
        """Total de todos los objetos movibles levantados."""
        return sum(1 for e in self.events if e.event_type == "salida")

    @property
    def total_returned(self):
        """Total de todos los objetos movibles devueltos."""
        return sum(1 for e in self.events if e.event_type == "entrada")

    def counts_by_class(self):
        """Desglose de objetos levantados y devueltos por cada clase unificada."""
        counts = {}
        for e in self.events:
            item = counts.setdefault(e.class_name, {"levantados": 0, "devueltos": 0})
            if e.event_type == "salida":
                item["levantados"] += 1
            else:
                item["devueltos"] += 1
        return counts

    def counts_by_zone(self):
        """Desglose de movimientos por zona ROI."""
        counts = {}
        for e in self.events:
            item = counts.setdefault(e.zone, {"levantados": 0, "devueltos": 0})
            if e.event_type == "salida":
                item["levantados"] += 1
            else:
                item["devueltos"] += 1
        return counts

    def update(self, results, class_names, frame_idx, fps):
        """Procesa el resultado del tracker de un frame."""
        if results.boxes.id is None:
            return []

        height, width = results.orig_shape
        detections = []
        for box in results.boxes:
            raw_name = class_names[int(box.cls[0])]
            clean_name = normalize_class_name(raw_name)
            if not clean_name:
                continue
            if not self.monitor_all_movable and clean_name not in self.jar_classes:
                continue
            # Punto de apoyo: centro de la base inferior del objeto
            x1, _, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "track_id": int(box.id[0]),
                "class_name": clean_name,
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
            "conteo_por_zona": self.counts_by_zone(),
            "clases_monitoreadas": sorted(self.jar_classes),
            "zonas": [z.name for z in self.zones],
            "eventos": [e.to_dict() for e in self.events],
        }
