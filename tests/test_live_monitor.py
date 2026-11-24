"""
Pruebas del monitoreo en vivo con camaras y detector simulados (sin RTSP real
ni modelo YOLO): reconexion, frames sin retraso acumulado y cambio rapido de camara.

Uso:
    python -m pytest tests/test_live_monitor.py -v
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import create_session_factory
from src.live.capture import STATE_LIVE, STATE_RECONNECTING, LiveFrameReader, is_live_source
from src.live.monitor import LiveMonitor

RTSP_A = "rtsp://admin:ClaveA@10.0.0.1:554/ch1"
RTSP_B = "rtsp://admin:ClaveB@10.0.0.2:554/ch1"


def wait_until(condition, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return False


# ---------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------
class FakeCapture:
    """Camara simulada: entrega `frames` cuadros y luego falla (o sigue para siempre)."""

    def __init__(self, frames=None, value=0, delay=0.0):
        self.remaining = frames
        self.value = value
        self.delay = delay
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 0

    def set(self, prop, value):
        return True

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.remaining is not None:
            if self.remaining <= 0:
                return False, None
            self.remaining -= 1
        return True, np.full((72, 128, 3), self.value, dtype=np.uint8)

    def release(self):
        self.opened = False


class _Boxes(list):
    id = None


class _Results:
    def __init__(self, frame):
        self.boxes = _Boxes()
        self.orig_shape = frame.shape[:2]


class FakeTracker:
    """Detector sin detecciones; puede tardar en "cargar" como YOLO."""

    load_delay = 0.0
    created = []

    def __init__(self, model_path, confidence, tracker_type):
        time.sleep(self.load_delay)
        self.class_names = {0: "Jarra"}
        FakeTracker.created.append(self)

    def track(self, frame):
        time.sleep(0.01)
        return _Results(frame)


def fake_reader_factory(values):
    """Lector en vivo cuyas camaras devuelven un color distinto por enlace RTSP."""
    def factory(source):
        return LiveFrameReader(
            source,
            open_fn=lambda src: FakeCapture(value=values[src], delay=0.005),
            stall_seconds=0.2,
            reconnect_delays=(0.05,),
        )
    return factory


@pytest.fixture
def monitor(tmp_path):
    FakeTracker.load_delay = 0.0
    FakeTracker.created = []
    mon = LiveMonitor(
        snapshots_dir=tmp_path / "snapshots",
        tracker_factory=FakeTracker,
        reader_factory=fake_reader_factory({RTSP_A: 40, RTSP_B: 200}),
        session_factory=create_session_factory(db_path=tmp_path / "live.db"),
        stop_timeout=0.2,
    )
    yield mon
    mon.stop()


def stream_value(mon):
    """Color promedio del ultimo JPEG publicado (identifica la camara)."""
    import cv2

    data = mon.latest_jpeg
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return int(img[60:, :].mean())  # Debajo del HUD


# ---------------------------------------------------------------------
# Lector en vivo
# ---------------------------------------------------------------------
def test_is_live_source():
    assert is_live_source(RTSP_A)
    assert is_live_source("0")
    assert not is_live_source("videos/COCINA 1 VIDEO 1.mp4")


def test_reconecta_cuando_la_camara_se_corta():
    opens = []

    def open_fn(src):
        opens.append(time.monotonic())
        # La primera conexion entrega 5 cuadros y se "corta"; la segunda funciona
        return FakeCapture(frames=5 if len(opens) == 1 else None, delay=0.002)

    reader = LiveFrameReader("rtsp://x/1", open_fn=open_fn, stall_seconds=0.1, reconnect_delays=(0.05,))
    reader.start()
    try:
        assert wait_until(lambda: reader.reconnects >= 1)
        assert wait_until(lambda: reader.state == STATE_LIVE and reader.frames_read > 20)
        assert len(opens) >= 2
    finally:
        reader.stop()


def test_reintenta_si_la_camara_no_responde_al_inicio():
    attempts = []

    def open_fn(src):
        attempts.append(1)
        cap = FakeCapture()
        cap.opened = len(attempts) >= 3  # El DVR tarda en arrancar
        return cap

    reader = LiveFrameReader("rtsp://x/1", open_fn=open_fn, reconnect_delays=(0.02,))
    reader.start()
    try:
        assert wait_until(lambda: reader.state == STATE_LIVE)
        assert len(attempts) >= 3
    finally:
        reader.stop()


def test_estado_reconectando_mientras_no_hay_senal():
    opens = []

    def open_fn(src):
        # Entrega 2 cuadros y luego la camara queda caida (no vuelve a abrir)
        opens.append(1)
        cap = FakeCapture(frames=2)
        cap.opened = len(opens) == 1
        return cap

    reader = LiveFrameReader("rtsp://x/1", open_fn=open_fn, stall_seconds=0.05, reconnect_delays=(0.05,))
    reader.start()
    try:
        assert wait_until(lambda: reader.reconnects >= 1 and len(opens) >= 3)
        assert reader.state == STATE_RECONNECTING
        assert "No se pudo conectar" in (reader.error or "")
    finally:
        reader.stop()


def test_entrega_siempre_el_frame_mas_reciente():
    # Camara rapida y consumidor lento (como YOLO en CPU): no se acumulan frames viejos
    reader = LiveFrameReader("rtsp://x/1", open_fn=lambda src: FakeCapture(delay=0.002))
    reader.start()
    try:
        first = reader.read_latest(0, timeout=2)
        assert first is not None
        time.sleep(0.2)  # "Inferencia" lenta
        second = reader.read_latest(first[0], timeout=2)
        assert second[0] - first[0] > 10          # Se salto los frames intermedios
        assert reader.frames_read - second[0] <= 2  # Y es practicamente el ultimo
    finally:
        reader.stop()


# ---------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------
def test_cambio_rapido_de_camara_no_mezcla_sesiones(monitor):
    # El modelo tarda mas que el tiempo de espera del stop (como YOLO en CPU)
    FakeTracker.load_delay = 0.6
    session_a = monitor.start(RTSP_A, "cocina_1", "models/x.pt", "config/zones/cocina_1.json")
    time.sleep(0.05)
    session_b = monitor.start(RTSP_B, "cocina_2", "models/x.pt", "config/zones/cocina_2.json")

    # La sesion A termina sola despues de cargar el modelo, sin procesar frames
    assert wait_until(lambda: not session_a.thread.is_alive(), timeout=5)
    # ...y al terminar NO apaga la sesion B (antes ponia running=False global)
    assert wait_until(lambda: monitor.running and session_b.connection == STATE_LIVE, timeout=5)
    assert monitor.status()["camera_id"] == "cocina_2"
    time.sleep(0.3)
    assert monitor.running
    # El video que se transmite es el de la camara B (no se mezclan frames de A)
    assert wait_until(lambda: abs(stream_value(monitor) - 200) < 30, timeout=3)


def test_stop_detiene_la_sesion(monitor):
    session = monitor.start(RTSP_A, "cocina_1", "models/x.pt", "config/zones/cocina_1.json")
    assert wait_until(lambda: session.fps_actual > 0)
    monitor.stop()
    assert not monitor.running
    assert wait_until(lambda: not session.thread.is_alive(), timeout=3)
    assert monitor.status()["conexion"] == "detenido"


def test_status_no_expone_la_clave_rtsp(monitor):
    monitor.start(RTSP_A, "cocina_1", "models/x.pt", "config/zones/cocina_1.json")
    assert wait_until(lambda: monitor.status()["conexion"] == STATE_LIVE)
    status = monitor.status()
    assert "ClaveA" not in str(status)
    assert status["fuente"] == "rtsp://admin:****@10.0.0.1:554/ch1"


def test_arranques_simultaneos_dejan_una_sola_sesion(monitor):
    # Dos peticiones /api/live/start al mismo tiempo (ej. doble clic)
    threads = [
        threading.Thread(target=monitor.start, args=(src, cam, "models/x.pt", "config/zones/cocina_1.json"))
        for src, cam in ((RTSP_A, "cocina_1"), (RTSP_B, "cocina_2"))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    current = monitor.session
    assert wait_until(lambda: current.connection == STATE_LIVE)
    time.sleep(0.3)
    alive = [t for t in threading.enumerate() if t.name.startswith("nex-live-cocina")]
    assert alive == [current.thread]
