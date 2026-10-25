"""
Prueba del motor de eventos con datos simulados (sin video ni YOLO).

Sirve para comprobar que la logica de entrada/salida funciona por si sola,
independiente de la calidad del modelo de deteccion.

Uso:
    python tests/test_event_engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events.zones import Zone
from src.events.event_engine import ZoneEventEngine

W, H = 100, 100
# Zona = mitad inferior del frame (la "mesa")
mesa = Zone("mesa", "reposo", [[0, 0.5], [1, 0.5], [1, 1], [0, 1]])


def simulate(positions, min_frames=3):
    """positions: lista de cy por frame para un solo objeto con ID 1."""
    engine = ZoneEventEngine([mesa], min_frames=min_frames)
    events = []
    for i, cy in enumerate(positions):
        det = [{"track_id": 1, "class_name": "cup", "cx": 50, "cy": cy}]
        events += engine.update(det, i, 30, W, H)
    return [e.event_type for e in events]


def test_levantar_y_devolver():
    # 5 frames en la mesa, 5 arriba, 5 en la mesa
    assert simulate([80] * 5 + [20] * 5 + [80] * 5) == ["salida", "entrada"]


def test_parpadeo_no_cuenta():
    # Sale 1 solo frame (deteccion erronea): no debe generar evento
    assert simulate([80] * 5 + [20] + [80] * 5) == []


def test_aparece_fuera_no_cuenta():
    # Objeto que aparece ya fuera de la zona: no es un levantamiento
    assert simulate([20] * 10) == []


if __name__ == "__main__":
    for test in (test_levantar_y_devolver, test_parpadeo_no_cuenta, test_aparece_fuera_no_cuenta):
        test()
        print(f"OK  {test.__name__}")
