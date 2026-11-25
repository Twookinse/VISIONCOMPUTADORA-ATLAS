"""
Script para validar el conteo automatico contra el conteo manual (ground truth).

Compara los reportes generados por scripts/run_counting.py (*_conteo.json)
con los datos anotados a mano en tests/ground_truth/conteo_manual.json.

Dos formas de anotar un video en conteo_manual.json:

1. Evento por evento (recomendado): el segundo del video en que se levanta cada
   jarra. Se comparan uno a uno (con una tolerancia de +-3 s) y se informa cuales
   acerto el sistema, cuales se le escaparon y cuales conto de mas.

    "COCINA 2 VIDEO 2.mp4": {
      "revisado_desde_s": 120,
      "revisado_hasta_s": 210,
      "levantamientos": [
        {"segundo": 131.5, "clase": "Jarra", "nota": "jarra de la mesa central"},
        {"segundo": 178, "clase": "Jarra"}
      ]
    }

2. Solo el total (formato anterior): "jarras_levantadas": 3

Uso:
    python scripts/run_counting.py --source "videos/COCINA 2 VIDEO 2.mp4" --start-seconds 120 --max-seconds 90
    python scripts/count_validator.py
    python scripts/count_validator.py --report "output/COCINA 2 VIDEO 2_tramo_120-210s_conteo.json"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.kitchen import normalize_class_name  # noqa: E402

DEFAULT_CLASS = "Jarra"


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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Segundos de diferencia aceptados entre el evento manual y el automatico (default: 3)",
    )
    return parser.parse_args()


def calc_accuracy(detected, expected):
    """Calcula el porcentaje de acierto penalizando tanto faltantes como excesos."""
    if expected <= 0:
        return 100.0 if detected == 0 else 0.0
    error = abs(detected - expected)
    return max(0.0, round((1.0 - error / expected) * 100.0, 1))


def _pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def _fmt(pct):
    return "-" if pct is None else f"{pct}%"


def analyzed_range(report):
    """Segundos del video que cubre el reporte (inicio, fin)."""
    start = float(report.get("inicio_tramo_s", 0.0))
    return start, start + float(report.get("duracion_video_s", 0.0))


def match_events(manual, automatic, tolerance):
    """Empareja eventos manuales y automaticos de la misma clase dentro de la tolerancia.

    Empareja primero los pares mas cercanos en el tiempo; cada evento se usa una sola vez.

    Returns:
        (pares, perdidos, de_mas) - pares = [(manual, automatico)]
    """
    candidates = sorted(
        (abs(m["segundo"] - a["time_s"]), i, j)
        for i, m in enumerate(manual)
        for j, a in enumerate(automatic)
        if m["clase"] == a["clase"] and abs(m["segundo"] - a["time_s"]) <= tolerance
    )
    used_m, used_a, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_m or j in used_a:
            continue
        used_m.add(i)
        used_a.add(j)
        pairs.append((manual[i], automatic[j]))
    missed = [m for i, m in enumerate(manual) if i not in used_m]
    extra = [a for j, a in enumerate(automatic) if j not in used_a]
    return pairs, missed, extra


def evaluate_events(report, gt, tolerance):
    """Comparacion evento por evento dentro del tramo analizado y revisado a mano."""
    start, end = analyzed_range(report)
    start = max(start, float(gt.get("revisado_desde_s", start)))
    end = min(end, float(gt.get("revisado_hasta_s", end)))

    manual = [
        {"segundo": float(e["segundo"]), "clase": normalize_class_name(e.get("clase", DEFAULT_CLASS)) or DEFAULT_CLASS,
         "nota": e.get("nota", "")}
        for e in gt["levantamientos"]
        if start <= float(e["segundo"]) <= end
    ]
    classes = {m["clase"] for m in manual} or {DEFAULT_CLASS}
    automatic = [
        {"time_s": float(e["time_s"]), "clase": e["class_name"], "zona": e.get("zone", ""),
         "track_id": e.get("track_id"), "foto": e.get("snapshot_file", "")}
        for e in report.get("eventos", [])
        if e.get("event_type") == "salida" and e.get("class_name") in classes and start <= float(e["time_s"]) <= end
    ]
    pairs, missed, extra = match_events(manual, automatic, tolerance)
    return {
        "tramo_s": [round(start, 1), round(end, 1)],
        "clases": sorted(classes),
        "reales": len(manual),
        "detectadas_ia": len(automatic),
        "aciertos": len(pairs),
        "perdidas": [{"segundo": m["segundo"], "nota": m["nota"]} for m in missed],
        "de_mas": [{"segundo": a["time_s"], "zona": a["zona"], "track_id": a["track_id"], "foto": a["foto"]}
                   for a in extra],
        # Recall: de las jarras levantadas de verdad, cuantas conto el sistema
        "recall_pct": _pct(len(pairs), len(manual)),
        # Precision: de lo que conto el sistema, cuanto era real
        "precision_pct": _pct(len(pairs), len(automatic)),
        "precision_total_pct": calc_accuracy(len(automatic), len(manual)),
    }


def evaluate_report(report_path, gt_videos, tolerance=3.0):
    """Compara un archivo *_conteo.json contra el diccionario de ground truth."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    source_file = Path(report.get("fuente", "")).name
    if source_file not in gt_videos:
        return None

    gt = gt_videos[source_file]
    base = {
        "video": source_file,
        "reporte": str(Path(report_path).resolve().relative_to(PROJECT_ROOT))
        if Path(report_path).resolve().is_relative_to(PROJECT_ROOT) else str(report_path),
        "modelo": report.get("modelo", "desconocido"),
        "notas": gt.get("notas", ""),
    }
    if "levantamientos" in gt:
        return {**base, "tipo": "eventos", **evaluate_events(report, gt, tolerance)}

    expected_total = gt.get("jarras_levantadas", 0)
    expected_visible = gt.get("visibles", expected_total)
    detected = report.get("jarras_levantadas", 0)
    return {
        **base,
        "tipo": "total",
        "reales_total": expected_total,
        "reales_visibles": expected_visible,
        "reales_tapadas": gt.get("tapadas", 0),
        "detectadas_ia": detected,
        "diferencia": detected - expected_total,
        "precision_total_pct": calc_accuracy(detected, expected_total),
        "precision_visibles_pct": calc_accuracy(detected, expected_visible),
    }


def print_result(r, target, tolerance):
    print(f"\nVideo:   {r['video']}")
    print(f"Reporte: {r['reporte']}")
    print(f"Modelo:  {r['modelo']}")
    if r["tipo"] == "eventos":
        print(f"  Tramo comparado: {r['tramo_s'][0]:g} s a {r['tramo_s'][1]:g} s (tolerancia +-{tolerance:g} s)")
        print(f"  Levantamientos reales: {r['reales']} | contados por la IA: {r['detectadas_ia']} "
              f"| aciertos: {r['aciertos']}")
        print(f"  Recall (reales que detecto):     {_fmt(r['recall_pct'])}")
        print(f"  Precision (conteos correctos):   {_fmt(r['precision_pct'])}")
        for m in r["perdidas"]:
            nota = f" ({m['nota']})" if m["nota"] else ""
            print(f"    - SE LE ESCAPO: segundo {m['segundo']:g}{nota}")
        for a in r["de_mas"]:
            print(f"    - CONTO DE MAS: segundo {a['segundo']:g}, zona '{a['zona']}', foto: {a['foto']}")
        cumple = "SI" if (r["recall_pct"] or 0) >= target and (r["precision_pct"] or 0) >= target else "NO"
    else:
        print(f"  De {r['reales_total']} jarras reales ({r['reales_visibles']} visibles, "
              f"{r['reales_tapadas']} tapadas), el sistema detecto {r['detectadas_ia']}.")
        print(f"  Precision total:         {r['precision_total_pct']}%")
        print(f"  Precision (visibles):    {r['precision_visibles_pct']}%")
        print("  (Solo compara totales. Anota los segundos de cada levantamiento para comparar evento por evento)")
        cumple = "SI" if r["precision_total_pct"] >= target else "NO"
    print(f"  Cumple meta (>={target:.0f}%):     {cumple}")
    if r["notas"]:
        print(f"  Notas: {r['notas']}")


def main():
    args = parse_args()
    gt_path = PROJECT_ROOT / args.ground_truth

    if not gt_path.exists():
        print(f"Error: no se encontro el archivo de conteo manual en '{args.ground_truth}'")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_videos = {k: v for k, v in gt_data.get("videos", {}).items() if not k.startswith("_")}

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
        res = evaluate_report(r_path, gt_videos, args.tolerance)
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
        print_result(r, args.target, args.tolerance)

    event_results = [r for r in results if r["tipo"] == "eventos"]
    summary = {
        "meta_precision_pct": args.target,
        "tolerancia_s": args.tolerance,
        "evaluaciones": results,
    }
    print("\n" + "=" * 65)
    if event_results:
        real = sum(r["reales"] for r in event_results)
        auto = sum(r["detectadas_ia"] for r in event_results)
        hits = sum(r["aciertos"] for r in event_results)
        summary.update({"recall_global_pct": _pct(hits, real), "precision_global_pct": _pct(hits, auto)})
        print(f"Evento por evento: {hits} aciertos de {real} levantamientos reales y {auto} contados por la IA")
        print(f"Recall global: {_fmt(summary['recall_global_pct'])} | "
              f"Precision global: {_fmt(summary['precision_global_pct'])}")
    avg_total = round(sum(r["precision_total_pct"] for r in results) / len(results), 1)
    summary.update({"precision_promedio_total_pct": avg_total, "cumple_meta": avg_total >= args.target})
    print(f"Precision promedio de totales: {avg_total}% (Meta: >={args.target:.0f}%)")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en:      {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
