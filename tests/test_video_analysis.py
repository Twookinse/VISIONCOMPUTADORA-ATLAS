"""
Pruebas del analisis de videos unificado (src/pipeline/) con video y detector
simulados: no necesitan el modelo YOLO ni videos reales.

Uso:
    python -m pytest tests/test_video_analysis.py -v
"""

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

import src.api.main as api
import src.pipeline.video_analysis as va
from src.events.zones import Zone
from src.pipeline.jobs import JobBusyError, VideoAnalysisJobs
from src.pipeline.video_analysis import AnalysisRequest, analyze_video

W, H = 200, 100
MESA = Zone("Mesa de Centro", "general", [[0, 0.5], [1, 0.5], [1, 1], [0, 1]])


# ---------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------
class FakeVideo:
    """Video simulado. El numero de frame va codificado en el pixel (0, 0)."""

    def __init__(self, frames=120, fps=30.0, reported_count=None, delay=0.0):
        self.frames = frames
        self.fps = fps
        self.reported_count = frames if reported_count is None else reported_count
        self.delay = delay
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_count
        return 0

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.read_count >= self.frames:
            return False, None
        self.read_count += 1
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[0, 0, 0], frame[0, 0, 1] = self.read_count % 256, self.read_count // 256
        return True, frame

    def release(self):
        self.released = True


class _Box:
    def __init__(self, track_id, xyxy):
        self.cls = [0]
        self.id = [track_id]
        self.conf = [0.9]
        self.xyxy = [np.array(xyxy, dtype=float)]


class _Boxes(list):
    id = [1]


class _Results:
    def __init__(self, frame, boxes):
        self.orig_shape = frame.shape[:2]
        self.boxes = _Boxes(boxes)
        self.names = {0: "Jarra"}


class FakeTracker:
    """Una jarra en la mesa durante los primeros `lift_at` frames y luego levantada."""

    lift_at = 60

    def __init__(self, model_path, confidence, tracker_type):
        self.class_names = {0: "Jarra"}

    def track(self, frame):
        idx = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        bottom = 90 if idx <= self.lift_at else 20  # Mesa = mitad inferior
        return _Results(frame, [_Box(1, [90, bottom - 15, 110, bottom])])


@pytest.fixture(autouse=True)
def fake_zones(monkeypatch):
    monkeypatch.setattr(va, "load_zones", lambda path: [MESA])


def request(**overrides):
    data = dict(camera_id="cocina_1", source="videos/COCINA 1 VIDEO 2.mp4",
                zones_file="config/zones/cocina_1.json", model="models/custom/x.pt")
    data.update(overrides)
    return AnalysisRequest(**data)


def run(video, req=None, tmp_path=None, **kwargs):
    return analyze_video(req or request(), snapshots_dir=tmp_path,
                         tracker_factory=FakeTracker, capture_factory=lambda src: video, **kwargs)


# ---------------------------------------------------------------------
# Motor de analisis
# ---------------------------------------------------------------------
def test_cuenta_el_levantamiento_y_guarda_la_foto(tmp_path):
    report = run(FakeVideo(frames=120), tmp_path=tmp_path)
    assert report["jarras_levantadas"] == 1
    assert report["frames_procesados"] == 120
    assert report["modo"] == "video"
    snaps = [e["snapshot_file"] for e in report["eventos"]]
    assert snaps and all((tmp_path / s).is_file() for s in snaps)


def test_procesa_todo_aunque_el_video_no_informe_su_largo(tmp_path):
    # Antes: CAP_PROP_FRAME_COUNT = 0 -> se procesaba 1 solo frame
    report = run(FakeVideo(frames=300, reported_count=0), tmp_path=tmp_path)
    assert report["frames_procesados"] == 300
    assert report["jarras_levantadas"] == 1


def test_limite_de_segundos(tmp_path):
    video = FakeVideo(frames=300, fps=30)
    report = run(video, req=request(max_seconds=2), tmp_path=tmp_path)
    assert report["frames_procesados"] == 60
    assert video.read_count == 60
    assert report["duracion_video_s"] == 2.0


def test_salto_automatico_en_videos_de_60fps(tmp_path):
    report = run(FakeVideo(frames=180, fps=60), tmp_path=tmp_path)
    assert report["frames_inferidos"] == 60  # 1 de cada 3


def test_fps_invalido_usa_30(tmp_path):
    report = run(FakeVideo(frames=90, fps=90000), tmp_path=tmp_path)
    assert report["duracion_video_s"] == 3.0


def test_cancelar_guarda_lo_contado(tmp_path):
    video = FakeVideo(frames=1000)
    report = run(video, tmp_path=tmp_path, should_cancel=lambda: video.read_count >= 100)
    assert report["cancelado"] is True
    assert report["frames_procesados"] == 100
    assert report["jarras_levantadas"] == 1
    assert video.released


def test_hora_real_de_la_grabacion(tmp_path):
    start = datetime(2026, 11, 22, 21, 30)
    report = run(FakeVideo(), req=request(video_start=start, start_estimated=False), tmp_path=tmp_path)
    assert report["inicio_grabacion"] == "2026-11-22T21:30:00"
    assert report["inicio_estimado"] is False


def test_fuente_rtsp_desde_consola(tmp_path):
    req = request(source="rtsp://admin:Clave@10.0.0.5:554/ch1")
    report = run(FakeVideo(frames=120, reported_count=0), req=req, tmp_path=tmp_path)
    assert report["modo"] == "live"          # Antes se guardaba como "video" y se deduplicaba
    assert report["fuente"] == "rtsp://admin:****@10.0.0.5:554/ch1"
    assert report["source_key"] is None
    assert req.report_name == "cocina_1_en_vivo"


def test_fotos_de_videos_distintos_no_se_sobrescriben(tmp_path):
    a = run(FakeVideo(), req=request(source="videos/COCINA 1 VIDEO 1.mp4"), tmp_path=tmp_path)
    b = run(FakeVideo(), req=request(source="videos/COCINA 1 VIDEO 2.mp4"), tmp_path=tmp_path)
    snaps_a = {e["snapshot_file"] for e in a["eventos"]}
    snaps_b = {e["snapshot_file"] for e in b["eventos"]}
    assert snaps_a and snaps_b and not (snaps_a & snaps_b)
    assert all("COCINA_1_VIDEO_1" in s for s in snaps_a)


def test_on_frame_puede_detener(tmp_path):
    report = run(FakeVideo(frames=500), tmp_path=tmp_path,
                 on_frame=lambda *args: args[4] < 50)  # args[4] = frame_idx
    assert report["cancelado"] is True
    assert report["frames_procesados"] == 50


# ---------------------------------------------------------------------
# Analisis en segundo plano
# ---------------------------------------------------------------------
def make_jobs(tmp_path, video_factory):
    saved = []

    def fake_save(report, output_dir, name):
        saved.append((report, name))
        return output_dir / f"{name}_conteo.json", len(saved)

    jobs = VideoAnalysisJobs(tmp_path / "snaps", tmp_path, tracker_factory=FakeTracker,
                             capture_factory=lambda src: video_factory(), save_fn=fake_save)
    return jobs, saved


def test_job_en_segundo_plano(tmp_path):
    jobs, saved = make_jobs(tmp_path, lambda: FakeVideo(frames=120))
    job = jobs.start(request())
    assert job.done.wait(5)
    assert job.error is None and job.run_id == 1
    assert saved[0][1] == "COCINA 1 VIDEO 2"
    status = jobs.status()
    assert status["activo"] is False and status["progreso_pct"] == 100.0
    assert status["ultimo_reporte"]["jarras_levantadas"] == 1


def test_un_solo_analisis_a_la_vez(tmp_path):
    jobs, _ = make_jobs(tmp_path, lambda: FakeVideo(frames=10_000, delay=0.001))
    jobs.start(request())
    with pytest.raises(JobBusyError):
        jobs.start(request())
    jobs.cancel()
    assert jobs.current.done.wait(5)
    assert jobs.current.report["cancelado"] is True


def test_inicios_simultaneos_solo_uno_gana(tmp_path):
    jobs, _ = make_jobs(tmp_path, lambda: FakeVideo(frames=10_000, delay=0.001))
    results = []

    def try_start():
        try:
            jobs.start(request())
            results.append("ok")
        except JobBusyError:
            results.append("ocupado")

    threads = [threading.Thread(target=try_start) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    jobs.cancel()
    jobs.current.done.wait(5)
    assert results.count("ok") == 1


def test_progreso_desconocido(tmp_path):
    jobs, _ = make_jobs(tmp_path, lambda: FakeVideo(frames=10_000, reported_count=0, delay=0.001))
    jobs.start(request())
    time.sleep(0.2)
    status = jobs.status()
    assert status["progreso_conocido"] is False
    assert status["frames_procesados"] > 0
    jobs.cancel()
    jobs.current.done.wait(5)


def test_error_queda_en_el_estado(tmp_path):
    class Broken(FakeTracker):
        def track(self, frame):
            raise RuntimeError("fallo en rtsp://admin:Clave@10.0.0.5/ch1")

    jobs, _ = make_jobs(tmp_path, lambda: FakeVideo())
    jobs.tracker_factory = Broken
    job = jobs.start(request(source="videos/x.mp4"))
    assert job.done.wait(5)
    assert "fallo" in jobs.status()["error"]
    assert jobs.running is False


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
@pytest.fixture
def api_files():
    """Video y modelo de prueba temporales dentro de videos/ y models/ (rutas permitidas)."""
    video = PROJECT_ROOT / "videos" / "_prueba_pipeline.mp4"
    model = PROJECT_ROOT / "models" / "custom" / "_prueba_pipeline.pt"
    for path in (video, model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    yield video, model
    for path in (video, model):
        path.unlink(missing_ok=True)


def test_api_pipeline_run_usa_el_mismo_motor(tmp_path, monkeypatch, api_files):
    jobs, saved = make_jobs(tmp_path, lambda: FakeVideo(frames=120))
    monkeypatch.setattr(api, "pipeline_job", jobs)
    client = TestClient(api.app, client=("127.0.0.1", 50000))
    body = {"source": "videos/_prueba_pipeline.mp4", "camera_id": "cocina_1",
            "model": "models/custom/_prueba_pipeline.pt", "video_start": "2026-11-22T21:30"}
    res = client.post("/api/pipeline/run", json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["run_id"] == 1
    assert data["reporte"]["jarras_levantadas"] == 1
    assert data["reporte"]["inicio_grabacion"] == "2026-11-22T21:30:00"  # Antes se ignoraba


def test_api_pipeline_run_responde_409_si_esta_ocupado(tmp_path, monkeypatch, api_files):
    jobs, _ = make_jobs(tmp_path, lambda: FakeVideo(frames=10_000, delay=0.001))
    monkeypatch.setattr(api, "pipeline_job", jobs)
    jobs.start(request())
    client = TestClient(api.app, client=("127.0.0.1", 50000))
    body = {"source": "videos/_prueba_pipeline.mp4", "camera_id": "cocina_1",
            "model": "models/custom/_prueba_pipeline.pt"}
    assert client.post("/api/pipeline/run", json=body).status_code == 409
    assert client.post("/api/pipeline/start", json=body).status_code == 409
    jobs.cancel()
    jobs.current.done.wait(5)


# ---------------------------------------------------------------------
# Analisis de un tramo del video (para comparar con un conteo manual)
# ---------------------------------------------------------------------
class SeekableVideo(FakeVideo):
    """FakeVideo que permite saltar a un frame (CAP_PROP_POS_FRAMES)."""

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.read_count = int(value)
        return True


def test_tramo_del_video_conserva_los_tiempos_reales(tmp_path):
    FakeTracker.lift_at = 4500  # la jarra se levanta en el frame 4500 = segundo 150 (30 fps)
    try:
        video = SeekableVideo(frames=9000, fps=30)
        req = request(start_seconds=120, max_seconds=60)
        report = run(video, req=req, tmp_path=tmp_path)
    finally:
        FakeTracker.lift_at = 60
    assert report["frames_procesados"] == 1800          # solo 60 s
    assert report["inicio_tramo_s"] == 120.0
    assert report["jarras_levantadas"] == 1
    assert 150 <= report["eventos"][0]["time_s"] <= 151  # tiempo del video, no del tramo
    assert report["modo"] == "prueba"                    # no cuenta en las metricas
    assert req.report_name == "COCINA 1 VIDEO 2_tramo_120-180s"


def test_video_completo_sigue_en_modo_video(tmp_path):
    req = request()
    report = run(FakeVideo(), req=req, tmp_path=tmp_path)
    assert report["modo"] == "video" and report["inicio_tramo_s"] == 0.0
    assert not req.is_partial and req.report_name == "COCINA 1 VIDEO 2"


def test_tramo_fuera_del_video_es_error(tmp_path):
    with pytest.raises(va.AnalysisError, match="no se puede empezar"):
        run(SeekableVideo(frames=300, fps=30), req=request(start_seconds=60), tmp_path=tmp_path)


def test_tramo_no_reemplaza_el_analisis_completo(tmp_path):
    from src.db.database import create_session_factory
    from src.db.models import PipelineRun, save_report_to_db

    db_path = tmp_path / "tramo.db"
    full = run(FakeVideo(), req=request(), tmp_path=tmp_path)
    save_report_to_db(full, db_path=db_path)
    partial = run(SeekableVideo(frames=300), req=request(start_seconds=2, max_seconds=5), tmp_path=tmp_path)
    save_report_to_db(partial, db_path=db_path)
    db = create_session_factory(db_path=db_path)()
    try:
        modes = sorted(r.mode for r in db.query(PipelineRun).all())
    finally:
        db.close()
    assert modes == ["prueba", "video"]  # antes el tramo borraba el analisis completo
