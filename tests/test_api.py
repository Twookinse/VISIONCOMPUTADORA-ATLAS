"""
Pruebas de la base de datos SQLite y endpoints de FastAPI (Sprint 4).

Uso:
    python tests/test_api.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Usar una base de datos temporal de prueba en output/
TEST_DB_PATH = PROJECT_ROOT / "output" / "test_nex_vision.db"
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()
os.environ["NEX_DB_URL"] = f"sqlite:///{TEST_DB_PATH}"

from fastapi.testclient import TestClient
from src.api.main import app
from src.db.database import dispose_all_engines
from src.db.models import save_report_to_db

client = TestClient(app)


def setup_sample_data():
    sample_report = {
        "fuente": "videos/VIDEO 2 VISION COMPUTADORA.mp4",
        "modelo": "models/custom/jarras_nex_v1.pt",
        "zonas_archivo": "config/zones/casa_cocina.json",
        "frames_procesados": 1575,
        "duracion_video_s": 52.5,
        "tiempo_proceso_s": 70.0,
        "jarras_levantadas": 2,
        "jarras_devueltas": 2,
        "objetos_levantados_total": 3,
        "objetos_devueltos_total": 2,
        "eventos": [
            {
                "frame": 145,
                "time_s": 4.83,
                "track_id": 7,
                "class_name": "Vaso",
                "zone": "Mesa de cocina",
                "event_type": "salida",
            },
            {
                "frame": 386,
                "time_s": 12.87,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de cocina",
                "event_type": "salida",
            },
            {
                "frame": 415,
                "time_s": 13.84,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de cocina",
                "event_type": "entrada",
            },
            {
                "frame": 451,
                "time_s": 15.04,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de cocina",
                "event_type": "salida",
            },
            {
                "frame": 481,
                "time_s": 16.04,
                "track_id": 1,
                "class_name": "Jarra",
                "zone": "Mesa de cocina",
                "event_type": "entrada",
            },
        ],
    }
    fixed_time = datetime(2026, 11, 1, 21, 30, 0)
    return save_report_to_db(sample_report, db_path=TEST_DB_PATH, created_at=fixed_time)


def test_list_events_and_filters():
    res = client.get("/api/events")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 5

    # Filtrar solo por Jarra y tipo salida (levantadas)
    res_jar = client.get("/api/events?class_name=Jarra&event_type=salida")
    assert res_jar.status_code == 200
    jar_data = res_jar.json()
    assert jar_data["total"] == 2
    assert all(e["class_name"] == "Jarra" and e["accion"] == "levantado" for e in jar_data["eventos"])

    # Filtrar por fecha
    res_date = client.get("/api/events?date=2026-11-01")
    assert res_date.status_code == 200
    assert res_date.json()["total"] == 5

    res_empty_date = client.get("/api/events?date=2026-01-01")
    assert res_empty_date.status_code == 200
    assert res_empty_date.json()["total"] == 0


def test_hourly_metrics():
    res = client.get("/api/metrics/hourly?date=2026-11-01")
    assert res.status_code == 200
    data = res.json()
    assert data["jarras_levantadas_total"] == 2
    assert data["objetos_levantados_total"] == 3
    assert data["totales_por_clase"]["Jarra"]["levantados"] == 2
    assert data["totales_por_clase"]["Vaso"]["levantados"] == 1

    # Verificar que la hora 21:00 concentre los eventos
    hour_21 = next(h for h in data["por_hora"] if h["hora"] == "21:00")
    assert hour_21["jarras_levantadas"] == 2
    assert hour_21["objetos_levantados"] == 3


def test_runs_and_swagger():
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert res.json()["total"] == 1

    docs = client.get("/docs")
    assert docs.status_code == 200


if __name__ == "__main__":
    setup_sample_data()
    for test in (test_list_events_and_filters, test_hourly_metrics, test_runs_and_swagger):
        test()
        print(f"OK  {test.__name__}")
    dispose_all_engines()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
