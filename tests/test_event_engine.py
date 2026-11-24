"""
Prueba del motor de eventos con datos simulados (sin video ni YOLO).

Sirve para comprobar que la logica de entrada/salida funciona por si sola,
independiente de la calidad del modelo de deteccion.

Uso:
    python -m pytest tests/test_event_engine.py -v
    python tests/test_event_engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events.zones import Zone
from src.events.event_engine import ZoneEventEngine

W, H = 100, 100
FPS = 30
# Zona = mitad inferior del frame (la "mesa")
mesa = Zone("mesa", "reposo", [[0, 0.5], [1, 0.5], [1, 1], [0, 1]])

# Escenario con varias zonas en un frame de 1000x1000 px
BW, BH = 1000, 1000
mesa_izq = Zone("mesa", "general", [[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]])
lavado = Zone("lavado", "servicio", [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]])
# Estante que se superpone con la mesa en x = 0.4 .. 0.5
estante = Zone("estante", "general", [[0.4, 0.5], [0.7, 0.5], [0.7, 1], [0.4, 1]])

GONE = None  # Frame en que el objeto no se detecta


def simulate(positions, min_frames=3):
    """positions: lista de cy por frame para un solo objeto con ID 1."""
    engine = ZoneEventEngine([mesa], min_frames=min_frames)
    events = []
    for i, cy in enumerate(positions):
        det = [{"track_id": 1, "class_name": "cup", "cx": 50, "cy": cy}]
        events += engine.update(det, i, FPS, W, H)
    return [e.event_type for e in events]


def run(frames, zones, size=(BW, BH), **options):
    """frames: lista por frame de detecciones (track_id, clase, x, y) o GONE.

    Devuelve los eventos como tuplas (track_id, zona, tipo) y el motor.
    """
    engine = ZoneEventEngine(zones, min_frames=3, **options)
    out = []
    for i, frame in enumerate(frames):
        dets = []
        for item in ([] if frame is GONE else frame):
            tid, cls, x, y = item
            dets.append({"track_id": tid, "class_name": cls, "cx": x, "cy": y})
        out += [(e.track_id, e.zone, e.event_type) for e in engine.update(dets, i, FPS, *size)]
    return out, engine


def hold(n, *dets):
    """n frames iguales con las detecciones indicadas."""
    return [list(dets)] * n


# ---------------------------------------------------------------------
# Casos basicos (comportamiento original)
# ---------------------------------------------------------------------
def test_levantar_y_devolver():
    # 5 frames en la mesa, 5 arriba, 5 en la mesa
    assert simulate([80] * 5 + [20] * 5 + [80] * 5) == ["salida", "entrada"]


def test_parpadeo_no_cuenta():
    # Sale 1 solo frame (deteccion erronea): no debe generar evento
    assert simulate([80] * 5 + [20] + [80] * 5) == []


def test_aparece_fuera_no_cuenta():
    # Objeto que aparece ya fuera de la zona: no es un levantamiento
    assert simulate([20] * 10) == []


# ---------------------------------------------------------------------
# Zonas superpuestas y bordes
# ---------------------------------------------------------------------
def test_zonas_superpuestas_un_solo_evento():
    # Jarra en la interseccion mesa/estante: estaba en la mesa y se queda ahi.
    # Al levantarla se registra UNA salida (antes salian dos: mesa y estante).
    frames = (hold(5, (1, "Jarra", 300, 800))        # solo mesa
              + hold(5, (1, "Jarra", 450, 800))      # interseccion mesa/estante
              + hold(5, (1, "Jarra", 450, 300)))     # levantada
    events, _ = run(frames, [mesa_izq, estante])
    assert events == [(1, "mesa", "salida")]


def test_objeto_nuevo_en_interseccion_elige_zona_mas_profunda():
    frames = hold(5, (1, "Jarra", 480, 800)) + hold(5, (1, "Jarra", 480, 300))
    events, _ = run(frames, [mesa_izq, estante])
    # x=480 esta a 20 px del borde de la mesa y a 80 px del borde del estante
    assert events == [(1, "estante", "salida")]


def test_temblor_en_el_borde_no_cuenta():
    # Jarra apoyada en el borde de la mesa (y=500): la caja tiembla +-4 px y se
    # queda varios frames de cada lado del borde (no basta la histeresis temporal)
    jitter = ([496] * 5 + [503] * 5 + [495] * 6 + [504] * 4) * 5
    frames = [[(1, "Jarra", 300, y)] for y in [520] * 5 + jitter]
    events, _ = run(frames, [mesa_izq])
    assert events == []


def test_alejarse_del_borde_si_cuenta():
    frames = hold(5, (1, "Jarra", 300, 520)) + hold(5, (1, "Jarra", 300, 470))
    events, _ = run(frames, [mesa_izq])
    assert events == [(1, "mesa", "salida")]


def test_traslado_de_mesa_a_lavado():
    # Levantada de la mesa y dejada en el lavado: salida + entrada
    frames = hold(5, (1, "Jarra", 200, 800)) + hold(5, (1, "Jarra", 800, 800))
    events, _ = run(frames, [mesa_izq, lavado])
    assert events == [(1, "mesa", "salida"), (1, "lavado", "entrada")]


# ---------------------------------------------------------------------
# Cambios de ID del tracker (oclusiones)
# ---------------------------------------------------------------------
def test_cambio_de_id_al_levantar_cuenta_salida_y_entrada():
    # La mano tapa la jarra #1 al tomarla; reaparece levantada como #2 y se devuelve.
    frames = (hold(5, (1, "Jarra", 300, 520))
              + [GONE] * 10
              + hold(5, (2, "Jarra", 300, 440))
              + hold(5, (2, "Jarra", 300, 700)))
    events, _ = run(frames, [mesa_izq])
    assert events == [(2, "mesa", "salida"), (2, "mesa", "entrada")]


def test_cambio_de_id_quieto_no_genera_eventos():
    # Una persona pasa por delante; la jarra vuelve con otro ID en el mismo lugar.
    frames = hold(5, (1, "Jarra", 300, 800)) + [GONE] * 10 + hold(10, (2, "Jarra", 302, 801))
    events, _ = run(frames, [mesa_izq])
    assert events == []


def test_no_hereda_de_otra_clase_ni_de_lejos():
    base = hold(5, (1, "Jarra", 300, 520)) + [GONE] * 10
    # Otra clase en el mismo lugar: es otro objeto
    events, _ = run(base + hold(5, (2, "Vaso", 300, 440)), [mesa_izq])
    assert events == []
    # Misma clase pero lejos (otra jarra en otra parte de la cocina)
    events, _ = run(base + hold(5, (2, "Jarra", 900, 100)), [mesa_izq])
    assert events == []


def test_no_hereda_de_un_objeto_que_sigue_visible():
    # Dos jarras: la #1 sigue en la mesa y aparece la #2 cerca. No se mezclan.
    frames = (hold(5, (1, "Jarra", 300, 520))
              + hold(5, (1, "Jarra", 300, 520), (2, "Jarra", 340, 440))
              + hold(5, (1, "Jarra", 300, 520), (2, "Jarra", 340, 700)))
    events, _ = run(frames, [mesa_izq])
    assert events == [(2, "mesa", "entrada")]


def test_no_hereda_despues_del_tiempo_limite():
    frames = hold(5, (1, "Jarra", 300, 520)) + [GONE] * 300 + hold(5, (2, "Jarra", 300, 440))
    events, _ = run(frames, [mesa_izq], lost_seconds=2.0)
    assert events == []


# ---------------------------------------------------------------------
# Memoria
# ---------------------------------------------------------------------
def test_memoria_acotada_en_sesiones_largas():
    # 10.000 objetos distintos, uno por frame (8+ minutos de video a 30 fps)
    frames = [[(i, "Jarra", 300, 800)] for i in range(10_000)]
    _, engine = run(frames, [mesa_izq, lavado], lost_seconds=6.0)
    assert engine.tracked_count <= 6 * FPS + 1


def test_mismo_id_que_vuelve_conserva_su_estado():
    # El tracker recupera el mismo ID tras una oclusion: no hay eventos falsos
    frames = hold(5, (1, "Jarra", 300, 800)) + [GONE] * 30 + hold(5, (1, "Jarra", 300, 800))
    events, engine = run(frames, [mesa_izq])
    assert events == []
    assert engine.tracked_count == 1


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
