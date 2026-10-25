"""
Carga y manejo de zonas de interes (ROI).

Las zonas se leen de un archivo JSON (ej: config/zones/kitchen.json)
con coordenadas normalizadas de 0 a 1. Aqui se convierten a pixeles
segun la resolucion real del video y se ofrece la verificacion de si
un punto esta dentro de una zona.
"""

import json
from pathlib import Path

import cv2
import numpy as np


class Zone:
    """Una zona poligonal dentro del frame.

    Args:
        name: Nombre de la zona (ej: "mesa").
        zone_type: Tipo de zona (ej: "reposo").
        points: Lista de puntos [x, y] normalizados de 0 a 1.
        description: Texto opcional que explica la zona.
    """

    def __init__(self, name, zone_type, points, description=""):
        self.name = name
        self.zone_type = zone_type
        self.points = points
        self.description = description

    def to_pixels(self, width, height):
        """Devuelve los puntos de la zona en pixeles como array de numpy."""
        return np.array(
            [[int(x * width), int(y * height)] for x, y in self.points],
            dtype=np.int32,
        )

    def contains(self, x, y, width, height):
        """Indica si el punto (x, y) en pixeles esta dentro de la zona."""
        polygon = self.to_pixels(width, height)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def to_dict(self):
        return {
            "nombre": self.name,
            "tipo": self.zone_type,
            "descripcion": self.description,
            "puntos": [[round(x, 4), round(y, 4)] for x, y in self.points],
        }


def load_zones(path):
    """Lee un archivo JSON de zonas y devuelve una lista de Zone."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Zone(
            name=z["nombre"],
            zone_type=z.get("tipo", "general"),
            points=z["puntos"],
            description=z.get("descripcion", ""),
        )
        for z in data.get("zonas", [])
    ]


def save_zones(path, zones, description=""):
    """Guarda una lista de Zone en un archivo JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "descripcion": description or (
            "Zonas de interes (ROI). Coordenadas normalizadas de 0 a 1 "
            "respecto al ancho y alto del frame."
        ),
        "zonas": [z.to_dict() for z in zones],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def draw_zones(frame, zones, color=(0, 200, 255)):
    """Dibuja las zonas sobre una copia del frame y la devuelve."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    overlay = annotated.copy()
    for zone in zones:
        polygon = zone.to_pixels(width, height)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(annotated, [polygon], True, color, 2)
        x, y = polygon[0]
        cv2.putText(annotated, zone.name, (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Relleno semitransparente para no tapar los objetos
    return cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0)
