"""
Pruebas de la fusion de clases del dataset (config/class_mapping.json y
scripts/merge_dataset_classes.py) y de su compatibilidad con el sistema.

Uso:
    python -m pytest tests/test_class_mapping.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_dataset_classes import build_index, load_mapping, merge_dataset, remap_label_text
from src.analysis.kitchen import JAR_ONLY_CLASSES, normalize_class_name
from src.config import cameras as cameras_cfg

MAPPING = PROJECT_ROOT / "config" / "class_mapping.json"

# Clases del modelo jarras_nex_v1.pt / dataset nex-cocina-oficial v1 (en su orden)
V1_CLASSES = [
    "Balde", "Basurero", "CajaCerveza", "Cernidor", "Cerveza", "Cuchillo1", "Envase Ketchup",
    "Envase Llajua", "Envase para platanos", "Envase salsa", "Jarra 1", "Jarra1", "Jarra2", "Jarra3",
    "Jarra4", "Jarra5", "JarraLicuadora", "Licuadora", "Licuadora2", "Microondas", "Persona1",
    "Persona2", "Persona3", "Persona4", "Persona5", "Persona6", "Productos culinarios", "Tapa de olla",
    "Tapa1", "Taper", "Taper2", "Taper3", "Taper4", "Taper5", "Vaso1", "botella", "olla", "olla1",
]


# ---------------------------------------------------------------------
# Mapeo
# ---------------------------------------------------------------------
def test_mapeo_cubre_las_38_clases_del_modelo_v1():
    new_names, fusion = load_mapping(MAPPING)
    assert sorted(fusion) == sorted(V1_CLASSES)
    assert len(new_names) == 21
    assert len(set(new_names)) == len(new_names)


def test_modelo_v1_y_v2_dan_los_mismos_nombres_en_el_sistema():
    # Con el modelo actual (38 clases) o el nuevo (21), el conteo ve los mismos nombres
    _, fusion = load_mapping(MAPPING)
    for old, new in fusion.items():
        assert normalize_class_name(old) == normalize_class_name(new), (old, new)


def test_clases_nuevas_se_cuentan_o_se_ignoran_como_corresponde():
    new_names, _ = load_mapping(MAPPING)
    ignored = {n for n in new_names if normalize_class_name(n) is None}
    assert ignored == {"Persona", "Licuadora", "Microondas", "Basurero"}
    for name in set(new_names) - ignored:
        assert normalize_class_name(name) == name
    assert {"Jarra", "Jarra Licuadora"} <= JAR_ONLY_CLASSES


# ---------------------------------------------------------------------
# Script de fusion
# ---------------------------------------------------------------------
def test_remap_poligonos_y_cajas():
    index = {0: 1, 1: 1, 2: None}
    text = "0 0.1 0.1 0.2 0.1 0.2 0.2\n1 0.5 0.5 0.1 0.1\n2 0.3 0.3 0.1 0.1\n\n"
    new_text, dropped = remap_label_text(text, index)
    assert new_text == "1 0.1 0.1 0.2 0.1 0.2 0.2\n1 0.5 0.5 0.1 0.1\n"
    assert dropped == 1


def test_clase_sin_mapeo_es_error_salvo_drop_unmapped():
    new_names, fusion = load_mapping(MAPPING)
    with pytest.raises(ValueError, match="sin mapeo"):
        build_index(V1_CLASSES + ["Plato"], new_names, fusion)
    index = build_index(V1_CLASSES + ["Plato"], new_names, fusion, drop_unmapped=True)
    assert index[len(V1_CLASSES)] is None


def make_dataset(root):
    """Mini dataset con el formato de exportacion de Roboflow."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({
        "train": "../train/images", "val": "../valid/images", "nc": len(V1_CLASSES), "names": V1_CLASSES,
    }), encoding="utf-8")
    labels = {
        "train": {"a.jpg": "11 0.1 0.1 0.2 0.1 0.2 0.2\n14 0.5 0.5 0.6 0.5 0.6 0.6\n20 0.3 0.3 0.4 0.3 0.4 0.4\n",
                  "b.jpg": ""},
        "valid": {"c.jpg": "36 0.5 0.5 0.1 0.1\n37 0.2 0.2 0.1 0.1\n"},
    }
    for split, files in labels.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, text in files.items():
            (root / split / "images" / name).write_bytes(b"jpg")
            (root / split / "labels" / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")


def test_merge_dataset(tmp_path):
    src, dst = tmp_path / "roboflow", tmp_path / "v2"
    make_dataset(src)
    summary = merge_dataset(src, dst, MAPPING)

    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == 21 and data["names"][0] == "Jarra"
    assert Path(data["path"]) == dst.resolve()
    assert "test" not in data  # no habia carpeta test
    assert summary["cajas_por_clase"]["Jarra"]["train"] == 2      # Jarra1 + Jarra4
    assert summary["cajas_por_clase"]["Persona"]["train"] == 1
    assert summary["cajas_por_clase"]["Olla"]["valid"] == 2       # olla + olla1
    assert (dst / "train" / "labels" / "b.txt").read_text() == ""  # imagen sin objetos se conserva
    assert (dst / "train" / "images" / "a.jpg").read_bytes() == b"jpg"
    jarra = data["names"].index("Jarra")
    rows = (dst / "train" / "labels" / "a.txt").read_text().splitlines()
    assert [int(r.split()[0]) for r in rows[:2]] == [jarra, jarra]
    assert json.loads((dst / "resumen_fusion.json").read_text(encoding="utf-8"))["clases_antes"] == 38


def test_merge_no_sobrescribe_destino(tmp_path):
    src, dst = tmp_path / "roboflow", tmp_path / "v2"
    make_dataset(src)
    dst.mkdir()
    (dst / "algo.txt").write_text("x")
    with pytest.raises(FileExistsError):
        merge_dataset(src, dst, MAPPING)


# ---------------------------------------------------------------------
# Modelo por defecto configurable
# ---------------------------------------------------------------------
def test_modelo_por_defecto_desde_config(tmp_path):
    cfg = tmp_path / "cameras.json"
    cfg.write_text(json.dumps({"modelo_default": "models/custom/jarras_nex_v2.pt", "camaras": []}))
    assert cameras_cfg.get_default_model(cfg) == "models/custom/jarras_nex_v2.pt"
    cfg.write_text(json.dumps({"camaras": []}))
    assert cameras_cfg.get_default_model(cfg) == "models/custom/jarras_nex_v1.pt"


def test_notebook_de_colab_esta_al_dia():
    # El notebook incluye copias del script de fusion y del mapeo: deben coincidir
    sys.path.insert(0, str(PROJECT_ROOT / "training"))
    from build_notebook import NOTEBOOK, build

    assert NOTEBOOK.read_text(encoding="utf-8") == build(), (
        "training/entrenar_modelo_nex.ipynb esta desactualizado: ejecuta python training/build_notebook.py"
    )
