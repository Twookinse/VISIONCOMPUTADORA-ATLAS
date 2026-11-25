"""
Pruebas de la base de datos SQLite, normalizacion de clases Nex y endpoints Multi-Camara.

Uso:
    python -m pytest tests/test_api.py -v
    python tests/test_api.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Usar una base de datos temporal de prueba en output/
TEST_DB_PATH = PROJECT_ROOT / "output" / "test_nex_vision.db"
TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # output/ no existe en un clon nuevo
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()
os.environ["NEX_DB_URL"] = f"sqlite:///{TEST_DB_PATH}"

from fastapi.testclient import TestClient
from src.api.main import app
from src.analysis.kitchen import normalize_class_name
from src.db.database import dispose_all_engines
from src.db.models import save_report_to_db

# Simula la propia PC del servidor (los demas equipos necesitan contrasena)
client = TestClient(app, client=("127.0.0.1", 50000))


def _cleanup_test_db():
    dispose_all_engines()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture(scope="module", autouse=True)
def sample_data():
    """Con pytest: carga los datos de ejemplo una vez y borra la BD al terminar."""
    setup_sample_data()
    yield
    _cleanup_test_db()


def setup_sample_data():
    fixed_time = datetime(2026, 11, 22, 21, 30, 0)

    report_cam1 = {
        "camera_id": "cocina_1",
        "fuente": "videos/COCINA 1 VIDEO 1.mp4",
        "modelo": "models/custom/jarras_nex_v1.pt",
        "zonas_archivo": "config/zones/cocina_1.json",
        "frames_procesados": 1800,
        "duracion_video_s": 30.0,
        "tiempo_proceso_s": 12.0,
        "jarras_levantadas": 2,
        "jarras_devueltas": 1,
        "objetos_levantados_total": 3,
        "objetos_devueltos_total": 1,
        "eventos": [
            {
                "frame": 300,
                "time_s": 5.0,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de Centro",
                "event_type": "salida",
            },
            {
                "frame": 420,
                "time_s": 7.0,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de Centro",
                "event_type": "entrada",
            },
            {
                "frame": 600,
                "time_s": 10.0,
                "track_id": 2,
                "class_name": "Jarra",
                "zone": "Mesa de Centro",
                "event_type": "salida",
            },
            {
                "frame": 900,
                "time_s": 15.0,
                "track_id": 5,
                "class_name": "Taper",
                "zone": "Mesa de Centro",
                "event_type": "salida",
            },
        ],
    }

    report_cam2 = {
        "camera_id": "cocina_2",
        "fuente": "videos/COCINA 2 VIDEO 1.mp4",
        "modelo": "models/custom/jarras_nex_v1.pt",
        "zonas_archivo": "config/zones/cocina_2.json",
        "frames_procesados": 1800,
        "duracion_video_s": 30.0,
        "tiempo_proceso_s": 12.0,
        "jarras_levantadas": 0,
        "jarras_devueltas": 0,
        "objetos_levantados_total": 1,
        "objetos_devueltos_total": 0,
        "eventos": [
            {
                "frame": 500,
                "time_s": 8.33,
                "track_id": 9,
                "class_name": "Olla",
                "zone": "Mesa de Centro",
                "event_type": "salida",
            },
        ],
    }

    save_report_to_db(report_cam1, db_path=TEST_DB_PATH, created_at=fixed_time)
    save_report_to_db(report_cam2, db_path=TEST_DB_PATH, created_at=fixed_time)


def test_class_normalization():
    assert normalize_class_name("Jarra 1") == "Jarra"
    assert normalize_class_name("Jarra2") == "Jarra"
    assert normalize_class_name("Jarra5") == "Jarra"
    assert normalize_class_name("JarraLicuadora") == "Jarra Licuadora"
    assert normalize_class_name("Taper4") == "Taper"
    assert normalize_class_name("olla1") == "Olla"
    assert normalize_class_name("Vaso1") == "Vaso"
    assert normalize_class_name("CajaCerveza") == "Caja de Cerveza"
    # Fijos y personas deben ignorarse
    assert normalize_class_name("Persona1") is None
    assert normalize_class_name("Persona6") is None
    assert normalize_class_name("Microondas") is None
    assert normalize_class_name("Basurero") is None
    assert normalize_class_name("Licuadora2") is None


def test_list_events_and_camera_filters():
    res = client.get("/api/events")
    assert res.status_code == 200
    assert res.json()["total"] == 5

    # Filtrar solo cocina_1
    res_c1 = client.get("/api/events?camera_id=cocina_1")
    assert res_c1.status_code == 200
    assert res_c1.json()["total"] == 4

    # Filtrar solo cocina_2
    res_c2 = client.get("/api/events?camera_id=cocina_2")
    assert res_c2.status_code == 200
    assert res_c2.json()["total"] == 1
    assert res_c2.json()["eventos"][0]["class_name"] == "Olla"

    # Filtrar Jarra salida
    res_jar = client.get("/api/events?camera_id=cocina_1&class_name=Jarra&event_type=salida")
    assert res_jar.status_code == 200
    assert res_jar.json()["total"] == 2


def test_hourly_metrics_and_cameras():
    res = client.get("/api/metrics/hourly?date=2026-11-22&camera_id=cocina_1")
    assert res.status_code == 200
    data = res.json()
    assert data["jarras_levantadas_total"] == 2
    assert data["objetos_levantados_total"] == 3
    assert data["totales_por_clase"]["Jarra"]["levantados"] == 2
    assert data["totales_por_clase"]["Taper"]["levantados"] == 1

    # Verificar las 3 camaras de Nex configuradas
    cams_res = client.get("/api/cameras")
    assert cams_res.status_code == 200
    cams = cams_res.json()["camaras"]
    cam_ids = [c["id"] for c in cams]
    assert "cocina_1" in cam_ids
    assert "cocina_2" in cam_ids
    assert "almacen" in cam_ids


def test_runs_classes_and_swagger():
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert res.json()["total"] == 2

    cls_res = client.get("/api/classes")
    assert cls_res.status_code == 200
    assert "Jarra" in cls_res.json()["clases"]

    docs = client.get("/docs")
    assert docs.status_code == 200


if __name__ == "__main__":
    setup_sample_data()
    for test in (
        test_class_normalization,
        test_list_events_and_camera_filters,
        test_hourly_metrics_and_cameras,
        test_runs_classes_and_swagger,
    ):
        test()
        print(f"OK  {test.__name__}")
    _cleanup_test_db()
