"""
Wrapper de YOLO para deteccion de objetos.

Encapsula la carga del modelo y la inferencia sobre un frame.
El modelo y los parametros se pasan por argumento, sin rutas fijas.
"""

from pathlib import Path
from ultralytics import YOLO


class Detector:
    """Detecta objetos en un frame usando un modelo YOLO.

    Args:
        model_path: Ruta al archivo .pt del modelo YOLO.
        confidence: Confianza minima para aceptar una deteccion (0.0 a 1.0).
        classes: Lista de IDs de clases COCO a detectar. None detecta todas.
    """

    def __init__(self, model_path, confidence=0.25, classes=None):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.classes = classes

        # Guardar el mapeo de IDs a nombres de clase
        self.class_names = self.model.names

    def detect(self, frame):
        """Ejecuta la deteccion sobre un frame.

        Args:
            frame: Imagen en formato numpy array (BGR, como lo entrega OpenCV).

        Returns:
            Objeto Results de ultralytics con las detecciones.
            Acceder a results.boxes para obtener bounding boxes,
            clases y confianzas.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False,
        )
        return results[0]

    def get_class_name(self, class_id):
        """Devuelve el nombre de una clase dado su ID numerico."""
        return self.class_names.get(int(class_id), "desconocido")
