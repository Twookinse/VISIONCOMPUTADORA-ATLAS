"""
Captura de video para camaras en vivo (RTSP / USB).

LiveFrameReader lee la camara en un hilo propio y conserva SOLO el ultimo frame:
- Sin retraso acumulado: si YOLO es mas lento que la camara, se descartan los
  frames intermedios en lugar de encolarlos, y la inferencia siempre trabaja
  sobre la imagen actual.
- Reconexion automatica: si la camara deja de entregar frames (corte de red,
  DVR reiniciado), se cierra la conexion y se vuelve a abrir con esperas
  crecientes (1, 2, 4, 8, 10 s...), sin detener el monitoreo.

Los videos grabados NO usan este lector: se leen frame a frame para no perder
ningun evento del conteo.
"""

import os
import threading
import time

import cv2

from src.utils.security import is_stream_url, mask_rtsp_url

# Opciones de FFmpeg para RTSP: TCP (sin perdida de paquetes) y sin buffer
RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
OPEN_TIMEOUT_MS = 8000
READ_TIMEOUT_MS = 5000

STATE_CONNECTING = "conectando"
STATE_LIVE = "en_vivo"
STATE_RECONNECTING = "reconectando"
STATE_STOPPED = "detenido"


def is_live_source(source):
    """True si la fuente es una camara en vivo (RTSP o indice USB)."""
    src = str(source or "").strip()
    return is_stream_url(src) or src.isdigit()


def open_capture(source):
    """Abre una camara en vivo con timeouts (para que un corte no bloquee para siempre)."""
    src = str(source).strip()
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS
    params = []
    for name, value in (("CAP_PROP_OPEN_TIMEOUT_MSEC", OPEN_TIMEOUT_MS),
                        ("CAP_PROP_READ_TIMEOUT_MSEC", READ_TIMEOUT_MS)):
        prop = getattr(cv2, name, None)
        if prop is not None:
            params += [prop, value]
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LiveFrameReader:
    """Hilo lector que mantiene el frame mas reciente de una camara en vivo.

    Args:
        source: Enlace RTSP o indice de camara USB.
        open_fn: Funcion que abre la fuente y devuelve un objeto tipo cv2.VideoCapture.
        stall_seconds: Segundos sin frames validos antes de reconectar.
        reconnect_delays: Esperas entre intentos de reconexion (el ultimo se repite).
    """

    def __init__(self, source, open_fn=open_capture, stall_seconds=5.0,
                 reconnect_delays=(1.0, 2.0, 4.0, 8.0, 10.0)):
        self.source = str(source)
        self.public_source = mask_rtsp_url(self.source)
        self._open_fn = open_fn
        self.stall_seconds = stall_seconds
        self.reconnect_delays = tuple(reconnect_delays) or (1.0,)

        self.state = STATE_CONNECTING
        self.reconnects = 0
        self.error = None
        self.fps = 25.0
        self.width = 0
        self.height = 0

        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="nex-live-reader")
        self._thread.start()
        return self

    def stop(self, timeout=2.0):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.state = STATE_STOPPED

    @property
    def frames_read(self):
        return self._seq

    def read_latest(self, last_seq=0, timeout=1.0):
        """Espera un frame mas nuevo que last_seq.

        Returns:
            (seq, frame) del frame mas reciente, o None si no llego ninguno en timeout.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._seq <= last_seq:
                return None
            return self._seq, self._frame

    # ------------------------------------------------------------------
    def _open(self):
        try:
            cap = self._open_fn(self.source)
        except Exception as exc:  # noqa: BLE001 - cualquier error de apertura es reintentable
            self.error = f"No se pudo abrir la camara {self.public_source}: {exc}"
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.error = f"No se pudo conectar con la camara {self.public_source}"
            return None
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1.0 <= self.fps <= 120.0:
            self.fps = 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        return cap

    def _wait_before_retry(self, attempt):
        delay = self.reconnect_delays[min(attempt, len(self.reconnect_delays) - 1)]
        self._stop.wait(delay)

    def _run(self):
        cap = None
        attempt = 0
        last_ok = time.monotonic()
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    if cap is None:
                        self.state = STATE_RECONNECTING if (self.reconnects or attempt) else STATE_CONNECTING
                        self._wait_before_retry(attempt)
                        attempt += 1
                        continue
                    last_ok = time.monotonic()

                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._cond:
                        self._frame = frame
                        self._seq += 1
                        self._cond.notify_all()
                    if frame.shape[1] and frame.shape[0]:
                        self.height, self.width = frame.shape[:2]
                    self.state = STATE_LIVE
                    self.error = None
                    attempt = 0
                    last_ok = time.monotonic()
                    continue

                if time.monotonic() - last_ok >= self.stall_seconds:
                    # La camara dejo de responder: cerrar y reconectar
                    cap.release()
                    cap = None
                    self.reconnects += 1
                    self.state = STATE_RECONNECTING
                    self.error = f"Se perdio la senal de {self.public_source}; reconectando..."
                else:
                    self._stop.wait(0.02)
        finally:
            if cap is not None:
                cap.release()
            self.state = STATE_STOPPED
