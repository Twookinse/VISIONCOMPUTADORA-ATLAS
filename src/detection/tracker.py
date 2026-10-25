"""
Wrapper de YOLO para deteccion con tracking de objetos.

A diferencia de detector.py (que analiza cada frame de forma independiente),
el tracker mantiene un ID unico por objeto entre frames consecutivos.
Internamente usa ByteTrack, que viene integrado en ultralytics.
"""

from pathlib import Path
from ultralytics import YOLO


class Tracker:
    """Detecta y rastrea objetos entre frames usando YOLO + ByteTrack.

    La diferencia con Detector es que aqui se usa model.track() en lugar
    de model.predict(). Esto activa el algoritmo de tracking que asigna
    un ID persistente a cada objeto.

    Args:
        model_path: Ruta al archivo .pt del modelo YOLO.
        confidence: Confianza minima para aceptar una deteccion (0.0 a 1.0).
        classes: Lista de IDs de clases COCO a detectar. None detecta todas.
        tracker_type: Algoritmo de tracking ("bytetrack.yaml" o "botsort.yaml").
    """

    def __init__(self, model_path, confidence=0.25, classes=None,
                 tracker_type="bytetrack.yaml"):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.classes = classes
        self.tracker_type = tracker_type

        # Mapeo de IDs a nombres de clase
        self.class_names = self.model.names

    def track(self, frame):
        """Ejecuta deteccion + tracking sobre un frame.

        Debe llamarse sobre frames consecutivos del mismo video
        para que el tracking funcione correctamente.

        Args:
            frame: Imagen en formato numpy array (BGR).

        Returns:
            Objeto Results de ultralytics con las detecciones.
            Ademas de boxes.xyxy, boxes.cls y boxes.conf,
            ahora boxes.id contiene el ID de tracking de cada objeto.
            boxes.id es None si no se pudo asignar ID en ese frame.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            classes=self.classes,
            tracker=self.tracker_type,
            persist=True,
            verbose=False,
        )
        return results[0]

    def reset(self):
        """Reinicia el tracker (para comenzar un video nuevo).

        Sin esto, los IDs del video anterior se mezclan con el nuevo.
        """
        self.model = YOLO(str(self.model.ckpt_path))

    def get_class_name(self, class_id):
        """Devuelve el nombre de una clase dado su ID numerico."""
        return self.class_names.get(int(class_id), "desconocido")
