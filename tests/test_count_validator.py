"""
Pruebas del validador de conteo (scripts/count_validator.py): comparacion evento
por evento contra el conteo manual y compatibilidad con el formato de solo totales.

Uso:
    python -m pytest tests/test_count_validator.py -v
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from count_validator import evaluate_report, match_events


def ev(t, cls="Jarra", kind="salida", zone="Mesa de Centro", tid=1):
    return {"time_s": t, "class_name": cls, "event_type": kind, "zone": zone, "track_id": tid,
            "snapshot_file": f"foto_{t}.jpg"}


def write_report(tmp_path, events, start=0.0, duration=600.0, name="COCINA 2 VIDEO 2.mp4"):
    path = tmp_path / "reporte_conteo.json"
    path.write_text(json.dumps({
        "fuente": f"videos/{name}", "modelo": "models/custom/jarras_nex_v2.pt",
        "inicio_tramo_s": start, "duracion_video_s": duration,
        "jarras_levantadas": sum(1 for e in events if e["event_type"] == "salida"), "eventos": events,
    }), encoding="utf-8")
    return path


def test_match_empareja_el_mas_cercano_y_una_sola_vez():
    manual = [{"segundo": 10, "clase": "Jarra"}, {"segundo": 12, "clase": "Jarra"}]
    auto = [{"time_s": 11.8, "clase": "Jarra"}, {"time_s": 30, "clase": "Jarra"}]
    pairs, missed, extra = match_events(manual, auto, tolerance=3)
    assert [(m["segundo"], a["time_s"]) for m, a in pairs] == [(12, 11.8)]
    assert [m["segundo"] for m in missed] == [10]
    assert [a["time_s"] for a in extra] == [30]


def test_match_respeta_la_clase():
    pairs, missed, extra = match_events([{"segundo": 10, "clase": "Jarra"}],
                                        [{"time_s": 10, "clase": "Taper"}], tolerance=3)
    assert not pairs and len(missed) == 1 and len(extra) == 1


def test_evento_por_evento(tmp_path):
    # Tramo 120-210 s: 3 levantamientos reales. La IA acierta 2, pierde 1 y cuenta 1 de mas.
    gt = {"COCINA 2 VIDEO 2.mp4": {
        "revisado_desde_s": 120, "revisado_hasta_s": 210,
        "levantamientos": [{"segundo": 131.5}, {"segundo": 150, "clase": "Jarra2"}, {"segundo": 190, "nota": "tapada"}],
    }}
    events = [ev(132.4), ev(149), ev(170), ev(140, kind="entrada"), ev(140, cls="Taper"), ev(300)]
    r = evaluate_report(write_report(tmp_path, events, start=120, duration=90), gt)
    assert r["tipo"] == "eventos"
    assert r["tramo_s"] == [120, 210]
    assert (r["reales"], r["detectadas_ia"], r["aciertos"]) == (3, 3, 2)
    assert r["perdidas"] == [{"segundo": 190, "nota": "tapada"}]
    assert [a["segundo"] for a in r["de_mas"]] == [170]
    assert r["de_mas"][0]["foto"] == "foto_170.jpg"
    assert r["recall_pct"] == 66.7 and r["precision_pct"] == 66.7
    # Mismo total (3 vs 3): con solo totales habria dado 100%
    assert r["precision_total_pct"] == 100.0


def test_solo_se_compara_el_tramo_analizado(tmp_path):
    gt = {"COCINA 2 VIDEO 2.mp4": {"levantamientos": [{"segundo": 50}, {"segundo": 400}]}}
    r = evaluate_report(write_report(tmp_path, [ev(51)], start=0, duration=120), gt)
    assert r["tramo_s"] == [0, 120]
    assert (r["reales"], r["aciertos"]) == (1, 1)


def test_formato_anterior_de_totales(tmp_path):
    gt = {"COCINA 2 VIDEO 2.mp4": {"jarras_levantadas": 4, "visibles": 3, "tapadas": 1}}
    r = evaluate_report(write_report(tmp_path, [ev(1), ev(2), ev(3)]), gt)
    assert r["tipo"] == "total"
    assert r["precision_total_pct"] == 75.0 and r["precision_visibles_pct"] == 100.0


def test_video_sin_conteo_manual(tmp_path):
    assert evaluate_report(write_report(tmp_path, [ev(1)], name="OTRO.mp4"), {}) is None


def test_plantilla_del_conteo_manual_es_valida():
    data = json.loads((PROJECT_ROOT / "tests" / "ground_truth" / "conteo_manual.json").read_text(encoding="utf-8"))
    example = data["videos"]["_ejemplo_por_eventos"]
    assert all("segundo" in e for e in example["levantamientos"])
