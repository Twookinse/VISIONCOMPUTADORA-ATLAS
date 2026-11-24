"""
Fusiona las clases de un dataset YOLO (exportado de Roboflow) segun un archivo de mapeo.

No hace falta volver a etiquetar: solo se reescribe el numero de clase de cada
etiqueta (cajas o poligonos) y se genera un data.yaml nuevo.

Uso:
    python scripts/merge_dataset_classes.py --src datasets/nex_roboflow --dst datasets/nex_v2
    python scripts/merge_dataset_classes.py --src /content/nex-cocina-oficial-1 --dst /content/nex_v2 \\
        --mapping config/class_mapping.json

Este archivo solo usa la libreria estandar de Python y PyYAML (incluido con
Ultralytics), para poder ejecutarse tal cual en Google Colab.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
DEFAULT_MAPPING = Path(__file__).resolve().parent.parent / "config" / "class_mapping.json"


def load_mapping(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    new_names = data["clases"]
    fusion = data["fusion"]
    unknown_targets = sorted(set(fusion.values()) - set(new_names))
    if unknown_targets:
        raise ValueError(f"El mapeo apunta a clases que no estan en 'clases': {unknown_targets}")
    return new_names, fusion


def load_source_names(src):
    data = yaml.safe_load((Path(src) / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def build_index(old_names, new_names, fusion, drop_unmapped=False):
    """Devuelve {id_viejo: id_nuevo} (o None para descartar la clase)."""
    missing = [n for n in old_names if n not in fusion]
    if missing and not drop_unmapped:
        raise ValueError(
            f"Clases del dataset sin mapeo: {missing}. Agregalas a config/class_mapping.json "
            "o usa --drop-unmapped para descartarlas."
        )
    new_index = {name: i for i, name in enumerate(new_names)}
    return {i: (new_index[fusion[n]] if n in fusion else None) for i, n in enumerate(old_names)}


def remap_label_text(text, index):
    """Reescribe el numero de clase de cada fila (cajas o poligonos). Descarta clases None."""
    out, dropped = [], 0
    for row in text.splitlines():
        parts = row.split()
        if not parts:
            continue
        new_id = index.get(int(parts[0]))
        if new_id is None:
            dropped += 1
            continue
        out.append(" ".join([str(new_id)] + parts[1:]))
    return ("\n".join(out) + "\n") if out else "", dropped


def _link_or_copy(src, dst):
    """Usa un enlace duro (no ocupa espacio extra) y si no se puede, copia."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_dataset(src, dst, mapping_path=DEFAULT_MAPPING, drop_unmapped=False):
    src, dst = Path(src), Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"La carpeta destino '{dst}' ya existe y no esta vacia")

    new_names, fusion = load_mapping(mapping_path)
    old_names = load_source_names(src)
    index = build_index(old_names, new_names, fusion, drop_unmapped)

    counts = {s: Counter() for s in SPLITS}
    images = Counter()
    dropped = 0
    for split in SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.is_dir():
            continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            _link_or_copy(img, dst / split / "images" / img.name)
            images[split] += 1
            lbl = lbl_dir / f"{img.stem}.txt"
            text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
            new_text, n_drop = remap_label_text(text, index)
            dropped += n_drop
            (dst / split / "labels" / f"{img.stem}.txt").write_text(new_text, encoding="utf-8")
            for row in new_text.splitlines():
                counts[split][new_names[int(row.split()[0])]] += 1

    data_yaml = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if images["test"] else None,
        "nc": len(new_names),
        "names": new_names,
    }
    data_yaml = {k: v for k, v in data_yaml.items() if v is not None}
    (dst / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary = {
        "origen": str(src),
        "clases_antes": len(old_names),
        "clases_despues": len(new_names),
        "imagenes": dict(images),
        "etiquetas_descartadas": dropped,
        "cajas_por_clase": {name: {s: counts[s][name] for s in SPLITS} for name in new_names},
    }
    (dst / "resumen_fusion.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def print_summary(summary):
    print(f"Clases: {summary['clases_antes']} -> {summary['clases_despues']}")
    print(f"Imagenes: {summary['imagenes']}")
    if summary["etiquetas_descartadas"]:
        print(f"Etiquetas descartadas (clases sin mapeo): {summary['etiquetas_descartadas']}")
    print(f"\n{'Clase':<22}{'train':>8}{'valid':>8}{'test':>8}")
    for name, c in summary["cajas_por_clase"].items():
        warn = "  <- pocos ejemplos" if c["valid"] < 5 else ""
        print(f"{name:<22}{c['train']:>8}{c['valid']:>8}{c['test']:>8}{warn}")


def main():
    parser = argparse.ArgumentParser(description="Fusionar clases de un dataset YOLO")
    parser.add_argument("--src", required=True, help="Carpeta del dataset exportado (con data.yaml)")
    parser.add_argument("--dst", required=True, help="Carpeta nueva donde crear el dataset fusionado")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="Archivo JSON de mapeo de clases")
    parser.add_argument("--drop-unmapped", action="store_true",
                        help="Descartar las clases que no esten en el mapeo (por defecto es un error)")
    args = parser.parse_args()
    try:
        summary = merge_dataset(args.src, args.dst, args.mapping, args.drop_unmapped)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")
    print_summary(summary)
    print(f"\nDataset fusionado en: {Path(args.dst).resolve()}")


if __name__ == "__main__":
    main()
