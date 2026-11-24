"""
Pruebas de seguridad: credenciales RTSP ocultas, validacion de rutas, subida
de videos y control de acceso desde otros equipos de la red.

No necesitan el modelo YOLO ni videos reales.

Uso:
    python -m pytest tests/test_security.py -v
"""

import base64
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

import src.api.main as api
from src.config import cameras as cameras_cfg
from src.db.database import create_session_factory, get_db
from src.utils.security import (
    UnsafePathError,
    mask_rtsp_url,
    sanitize_upload_name,
    unique_destination,
    validate_model_path,
    validate_source,
    validate_zones_path,
)

REAL_RTSP = "rtsp://admin:Clave.Secreta1@192.168.1.64:554/Streaming/Channels/101"
MASKED_RTSP = "rtsp://admin:****@192.168.1.64:554/Streaming/Channels/101"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def cameras_file(tmp_path, monkeypatch):
    """config/cameras.json temporal con una camara que tiene clave RTSP."""
    cfg = json.loads((PROJECT_ROOT / "config" / "cameras.example.json").read_text(encoding="utf-8"))
    cfg["camaras"][0]["rtsp_url"] = REAL_RTSP
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(cameras_cfg, "CAMERAS_CONFIG_PATH", path)
    return path


@pytest.fixture
def local_client(tmp_path):
    """Cliente que simula la propia PC del servidor (127.0.0.1) con una BD temporal."""
    session_local = create_session_factory(db_path=tmp_path / "test.db")

    def override_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    api.app.dependency_overrides[get_db] = override_db
    yield TestClient(api.app, client=("127.0.0.1", 50000))
    api.app.dependency_overrides.clear()


def remote_client():
    """Cliente que simula un celular u otra PC de la red del local."""
    return TestClient(api.app, client=("192.168.1.50", 50000))


def basic_auth(user, password):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ---------------------------------------------------------------------
# Enmascarado de credenciales RTSP
# ---------------------------------------------------------------------
@pytest.mark.parametrize("url, expected", [
    (REAL_RTSP, MASKED_RTSP),
    ("rtsp://192.168.1.64:554/live", "rtsp://192.168.1.64:554/live"),
    ("rtsp://admin@192.168.1.64/live", "rtsp://admin:****@192.168.1.64/live"),
    ("rtsp://192.168.1.64/cam?channel=1&user=admin&password=x1",
     "rtsp://192.168.1.64/cam?channel=1&user=admin&password=****"),
    ("videos/COCINA 1 VIDEO 1.mp4", "videos/COCINA 1 VIDEO 1.mp4"),
    ("", ""),
])
def test_mask_rtsp_url(url, expected):
    assert mask_rtsp_url(url) == expected


def test_clave_con_arroba_queda_oculta():
    masked = mask_rtsp_url("rtsp://admin:cl@ve@10.0.0.5/ch1")
    assert "cl@ve" not in masked
    assert masked.endswith("@10.0.0.5/ch1")


# ---------------------------------------------------------------------
# Validacion de rutas
# ---------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "/etc/passwd",
    "C:/Windows/win.ini",
    "../models/x.pt",
    "videos/../../secreto.pt",
    "videos/malicioso.pt",     # un .pt subido como "video" no puede cargarse como modelo
    "output/nex_vision.db",
])
def test_modelo_fuera_de_models_rechazado(bad):
    with pytest.raises(UnsafePathError):
        validate_model_path(bad)


def test_modelo_inexistente_da_404():
    with pytest.raises(FileNotFoundError):
        validate_model_path("models/custom/no_existe.pt")


def test_zonas_validas_y_traversal():
    assert validate_zones_path("config/zones/cocina_1.json") == "config/zones/cocina_1.json"
    assert validate_zones_path("config/zones/../zones/cocina_2.json") == "config/zones/cocina_2.json"
    with pytest.raises(UnsafePathError):
        validate_zones_path("config/cameras.example.json")


def test_fuentes_permitidas():
    assert validate_source(REAL_RTSP) == REAL_RTSP
    assert validate_source("0") == "0"
    with pytest.raises(UnsafePathError):
        validate_source(REAL_RTSP, allow_stream=False)
    with pytest.raises(UnsafePathError):
        validate_source("http://10.0.0.5/video.mjpg")
    with pytest.raises(UnsafePathError):
        validate_source("/etc/passwd")


# ---------------------------------------------------------------------
# Nombres de videos subidos
# ---------------------------------------------------------------------
def test_sanitize_upload_name():
    assert sanitize_upload_name("COCINA 1 VIDEO 1.mp4") == "COCINA 1 VIDEO 1.mp4"
    assert sanitize_upload_name("..\\..\\Cocina<2>.MOV") == "Cocina_2_.mov"
    for bad in ("modelo.pt", "../../evil.pt", "script.bat", "sin_extension"):
        with pytest.raises(UnsafePathError):
            sanitize_upload_name(bad)


def test_unique_destination_no_sobrescribe(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"1")
    (tmp_path / "video (1).mp4").write_bytes(b"2")
    assert unique_destination(tmp_path, "video.mp4").name == "video (2).mp4"


# ---------------------------------------------------------------------
# Configuracion de camaras
# ---------------------------------------------------------------------
def test_rtsp_enmascarado_conserva_clave_real(cameras_file):
    cam = cameras_cfg.update_camera_rtsp("cocina_1", MASKED_RTSP)
    assert cam["rtsp_url"] == REAL_RTSP
    stored = json.loads(cameras_file.read_text(encoding="utf-8"))
    assert stored["camaras"][0]["rtsp_url"] == REAL_RTSP


def test_rtsp_invalido_rechazado(cameras_file):
    with pytest.raises(cameras_cfg.InvalidRtspUrl):
        cameras_cfg.update_camera_rtsp("cocina_1", "http://evil.example/stream")
    with pytest.raises(cameras_cfg.InvalidRtspUrl):
        # Enlace enmascarado que no corresponde al guardado
        cameras_cfg.update_camera_rtsp("cocina_1", "rtsp://otro:****@10.0.0.9/ch1")


def test_unmask_source(cameras_file):
    assert cameras_cfg.unmask_source(MASKED_RTSP) == REAL_RTSP
    assert cameras_cfg.unmask_source("videos/a.mp4") == "videos/a.mp4"


def test_config_local_se_crea_desde_plantilla(tmp_path, monkeypatch):
    local = tmp_path / "cameras.json"
    example = tmp_path / "cameras.example.json"
    example.write_text(json.dumps({"camaras": [{"id": "x", "rtsp_url": ""}]}), encoding="utf-8")
    monkeypatch.setattr(cameras_cfg, "CAMERAS_CONFIG_PATH", local)
    monkeypatch.setattr(cameras_cfg, "CAMERAS_EXAMPLE_PATH", example)
    assert cameras_cfg.get_cameras()[0]["id"] == "x"
    assert local.exists()


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
def test_api_cameras_no_expone_clave(cameras_file, local_client):
    res = local_client.get("/api/cameras")
    assert res.status_code == 200
    assert "Clave.Secreta1" not in res.text
    assert res.json()["camaras"][0]["rtsp_url"] == MASKED_RTSP


def test_api_guardar_rtsp_responde_enmascarado(cameras_file, local_client):
    res = local_client.put("/api/cameras/cocina_2", json={"rtsp_url": REAL_RTSP})
    assert res.status_code == 200
    assert "Clave.Secreta1" not in res.text
    bad = local_client.put("/api/cameras/cocina_2", json={"rtsp_url": "file:///etc/passwd"})
    assert bad.status_code == 400


@pytest.mark.parametrize("payload", [
    {"model": "videos/malicioso.pt"},
    {"model": "/tmp/malicioso.pt"},
    {"zones": "/etc/passwd"},
    {"tracker": "../../../etc/passwd"},
    {"source": "/etc/passwd"},
    {"source": REAL_RTSP},          # el analisis por lotes solo acepta videos grabados
])
def test_api_pipeline_rechaza_rutas_peligrosas(payload, local_client):
    body = {"source": "videos/COCINA 1 VIDEO 1.mp4", "camera_id": "cocina_1", **payload}
    res = local_client.post("/api/pipeline/run", json=body)
    assert res.status_code in (400, 404)
    assert "Clave.Secreta1" not in res.text


def test_api_live_rechaza_modelo_fuera_de_models(local_client):
    res = local_client.post("/api/live/start", json={"source": "0", "model": "videos/malicioso.pt"})
    assert res.status_code == 400
    assert api.live_monitor.running is False


def test_api_upload_rechaza_no_videos(local_client):
    res = local_client.post("/api/videos/upload", files={"file": ("modelo.pt", b"x", "application/octet-stream")})
    assert res.status_code == 400


def test_api_upload_no_sobrescribe(local_client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "VIDEOS_DIR", tmp_path)
    (tmp_path / "COCINA 1 VIDEO 1.mp4").write_bytes(b"original")
    res = local_client.post(
        "/api/videos/upload", files={"file": ("COCINA 1 VIDEO 1.mp4", b"nuevo", "video/mp4")}
    )
    assert res.status_code == 200
    assert res.json()["video_path"] == "videos/COCINA 1 VIDEO 1 (1).mp4"
    assert (tmp_path / "COCINA 1 VIDEO 1.mp4").read_bytes() == b"original"


def test_api_upload_limite_de_tamano(local_client, tmp_path, monkeypatch):
    videos_dir = tmp_path / "videos"
    monkeypatch.setattr(api, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(api, "max_upload_bytes", lambda: 10)
    res = local_client.post("/api/videos/upload", files={"file": ("grande.mp4", b"x" * 50, "video/mp4")})
    assert res.status_code == 413
    assert list(videos_dir.iterdir()) == []  # ni el video ni el archivo .part quedan en disco


def test_api_snapshot_solo_jpg(local_client):
    assert local_client.get("/api/snapshots/..%2F..%2Fconfig%2Fcameras.json").status_code == 404


# ---------------------------------------------------------------------
# Control de acceso
# ---------------------------------------------------------------------
def test_remoto_bloqueado_sin_contrasena(monkeypatch):
    monkeypatch.setenv("NEX_PASSWORD", "")
    res = remote_client().get("/api/status")
    assert res.status_code == 403
    assert "NEX_PASSWORD" in res.json()["detail"]


def test_remoto_con_contrasena(monkeypatch):
    monkeypatch.setenv("NEX_USER", "nex")
    monkeypatch.setenv("NEX_PASSWORD", "cocina2026")
    client = remote_client()
    sin_login = client.get("/api/status")
    assert sin_login.status_code == 401
    assert sin_login.headers["www-authenticate"].startswith("Basic")
    assert client.get("/api/status", headers=basic_auth("nex", "incorrecta")).status_code == 401
    assert client.get("/api/status", headers=basic_auth("nex", "cocina2026")).status_code == 200


def test_websocket_remoto_bloqueado(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("NEX_PASSWORD", "")
    with pytest.raises(WebSocketDisconnect):
        with remote_client().websocket_connect("/ws/events") as ws:
            ws.receive_json()


def test_cors_no_permite_cualquier_origen(local_client):
    res = local_client.get("/api/status", headers={"Origin": "http://sitio-malicioso.com"})
    assert "access-control-allow-origin" not in res.headers
    ok = local_client.get("/api/status", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
