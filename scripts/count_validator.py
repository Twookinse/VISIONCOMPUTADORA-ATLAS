"""
Script para validar el conteo automatico contra el conteo manual (ground truth).

Compara los reportes generados por scripts/run_counting.py (*_conteo.json)
con los datos anotados manualmente en tests/ground_truth/conteo_manual.json.

Uso:
    python scripts/count_validator.py
    python scripts/count_validator.py --report "output/VIDEO 2 VISION COMPUTADORA_conteo.json"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar conteo automatico de IA contra conteo manual"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Ruta a un archivo *_conteo.json especifico (opcional). "
             "Si no se indica, evalua todos los reportes en output/.",
    )
    parser.add_argument(
        "--ground-truth",
        default="tests/ground_truth/conteo_manual.json",
        help="Archivo con el conteo manual (default: tests/ground_truth/conteo_manual.json)",
    )
    parser.add_argument(
        "--output",
        default="output/validacion_conteo.json",
        help="Ruta donde guardar el resumen de validacion (default: output/validacion_conteo.json)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=70.0,
        help="Porcentaje minimo esperado para aprobar la meta (default: 70.0)",
    )
    return parser.parse_args()


def calc_accuracy(detected, expected):
    """Calcula el porcentaje de acierto penalizando tanto faltantes como excesos."""
    if expected <= 0:
        return 100.0 if detected == 0 else 0.0
    error = abs(detected - expected)
    return max(0.0, round((1.0 - error / expected) * 100.0, 1))


def evaluate_report(report_path, gt_videos):
    """Compara un archivo *_conteo.json contra el diccionario de ground truth."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    source_file = Path(report.get("fuente", "")).name
    if source_file not in gt_videos:
        return None

    gt = gt_videos[source_file]
    expected_total = gt.get("jarras_levantadas", 0)
    expected_visible = gt.get("visibles", expected_total)
    detected = report.get("jarras_levantadas", 0)

    acc_total = calc_accuracy(detected, expected_total)
    acc_visible = calc_accuracy(detected, expected_visible)

    return {
        "video": source_file,
        "reporte": str(report_path.relative_to(PROJECT_ROOT)),
        "modelo": report.get("modelo", "desconocido"),
        "reales_total": expected_total,
        "reales_visibles": expected_visible,
        "reales_tapadas": gt.get("tapadas", 0),
        "detectadas_ia": detected,
        "diferencia": detected - expected_total,
        "precision_total_pct": acc_total,
        "precision_visibles_pct": acc_visible,
        "notas": gt.get("notas", ""),
    }


def main():
    args = parse_args()
    gt_path = PROJECT_ROOT / args.ground_truth

    if not gt_path.exists():
        print(f"Error: no se encontro el archivo de conteo manual en '{args.ground_truth}'")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_videos = gt_data.get("videos", {})

    if args.report:
        report_files = [PROJECT_ROOT / args.report]
    else:
        output_dir = PROJECT_ROOT / "output"
        report_files = sorted(output_dir.glob("*_conteo.json"))

    if not report_files:
        print("Error: no se encontraron reportes *_conteo.json para evaluar.")
        print("Ejecuta primero: python scripts/run_counting.py --source <video>")
        sys.exit(1)

    results = []
    for r_path in report_files:
        if not r_path.exists():
            print(f"Aviso: no existe el archivo '{r_path}'")
            continue
        res = evaluate_report(r_path, gt_videos)
        if res is not None:
            results.append(res)
        else:
            print(f"Aviso: '{r_path.name}' no tiene entrada correspondiente en {args.ground_truth}")

    if not results:
        print("No se pudo validar ningun video contra el conteo manual.")
        sys.exit(1)

    print("=" * 65)
    print("VALIDACION DE CONTEO: IA vs MANUAL (GROUND TRUTH)")
    print("=" * 65)

    for r in results:
        cumple = "SI" if r["precision_total_pct"] >= args.target else "NO"
        print(f"\nVideo:   {r['video']}")
        print(f"Modelo:  {r['modelo']}")
        print(f"  De {r['reales_total']} jarras reales ({r['reales_visibles']} visibles, "
              f"{r['reales_tapadas']} tapadas), el sistema detecto {r['detectadas_ia']}.")
        print(f"  Precision total:         {r['precision_total_pct']}%")
        print(f"  Precision (visibles):    {r['precision_visibles_pct']}%")
        print(f"  Cumple meta (>={args.target:.0f}%):     {cumple}")
        if r["notas"]:
            print(f"  Notas: {r['notas']}")

    avg_total = round(sum(r["precision_total_pct"] for r in results) / len(results), 1)
    avg_vis = round(sum(r["precision_visibles_pct"] for r in results) / len(results), 1)

    summary = {
        "meta_precision_pct": args.target,
        "precision_promedio_total_pct": avg_total,
        "precision_promedio_visibles_pct": avg_vis,
        "cumple_meta": avg_total >= args.target,
        "evaluaciones": results,
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"Precision promedio total: {avg_total}% (Meta: >={args.target:.0f}%)")
    print(f"Reporte guardado en:      {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
