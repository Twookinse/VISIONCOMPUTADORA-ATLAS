"""
Script para contar jarras levantadas en un video o camara.

Combina tracking (src/detection/tracker.py), zonas (config/zones/*.json)
y la logica de cocina (src/analysis/kitchen.py).

Uso:
    python scripts/run_counting.py --source "videos/VIDEO 2 VISION COMPUTADORA.mp4"
    python scripts/run_counting.py --source 0 --show
    python scripts/run_counting.py --source rtsp://ip:554/stream --zones config/zones/nex_cocina_cam1.json

Genera en output/:
    <video>_conteo.mp4   Video con zonas, objetos y contador dibujados
    <video>_conteo.json  Reporte con el total y cada evento (frame, segundo, ID)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.tracker import Tracker
from src.events.zones import load_zones, draw_zones
from src.analysis.kitchen import KitchenAnalyzer, DEFAULT_JAR_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Contar jarras levantadas")
    parser.add_argument("--source", required=True,
                        help="Video, 0 para camara web, o URL RTSP")
    parser.add_argument("--model", default="models/custom/jarras_nex_v1.pt",
                        help="Modelo YOLO (default: models/custom/jarras_nex_v1.pt)")
    parser.add_argument("--zones", default="config/zones/casa_cocina.json",
                        help="Archivo de zonas (default: config/zones/casa_cocina.json)")
    parser.add_argument("--tracker", default="config/trackers/bytetrack_cocina.yaml",
                        help="Configuracion del tracker (default: config/trackers/bytetrack_cocina.yaml)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_JAR_CLASSES,
                        help="Clases movibles a monitorear en las zonas")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confianza minima de deteccion (default: 0.25)")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="Frames para confirmar una salida o entrada (default: 3)")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Procesar 1 de cada N frames (default: 1)")
    parser.add_argument("--output", default="output",
                        help="Carpeta de salida (default: output/)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar el video mientras se procesa")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    zones = load_zones(PROJECT_ROOT / args.zones)
    if not zones:
        print(f"Error: no hay zonas en '{args.zones}'. Dibujalas con scripts/draw_zones.py")
        sys.exit(1)

    tracker = Tracker(
        model_path=PROJECT_ROOT / args.model,
        confidence=args.confidence,
        tracker_type=str(PROJECT_ROOT / args.tracker),
    )
    analyzer = KitchenAnalyzer(zones, jar_classes=args.classes, min_frames=args.min_frames)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la fuente '{args.source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_file = total_frames > 0

    print(f"Fuente: {args.source}")
    print(f"Modelo: {args.model}")
    print(f"Zonas usadas: {[z.name for z in analyzer.zones]}")
    print(f"Clases monitoreadas: {sorted(analyzer.jar_classes)}")
    print()

    writer = None
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(str(source)).stem if isinstance(source, str) else "camara"
    video_path = output_dir / f"{source_name}_conteo.mp4"
    report_path = output_dir / f"{source_name}_conteo.json"
    if is_file:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    frame_count = 0
    last_event_text = ""
    last_event_frame = -999
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % args.skip_frames != 0:
                continue

            results = tracker.track(frame)
            new_events = analyzer.update(results, tracker.class_names, frame_count, fps)

            for event in new_events:
                action = "LEVANTADO/A" if event.event_type == "salida" else "DEVUELTO/A"
                last_event_text = f"{event.class_name} #{event.track_id} {action}"
                last_event_frame = frame_count
                print(f"  [{event.time_s:6.1f}s] {event.class_name} #{event.track_id} "
                      f"{event.event_type} de '{event.zone}'")

            # Dibujar objetos, zonas y contador
            annotated = draw_zones(results.plot(), analyzer.zones)
            cv2.rectangle(annotated, (0, 0), (width, 34), (0, 0, 0), -1)
            hud_text = (f"Jarras Lev: {analyzer.lifted} Dev: {analyzer.returned} | "
                        f"Total Objetos Lev: {analyzer.total_lifted}")
            cv2.putText(annotated, hud_text,
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            # Mostrar el ultimo evento durante ~1.5 segundos
            if frame_count - last_event_frame < fps * 1.5:
                cv2.putText(annotated, last_event_text, (10, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if writer:
                # Se repite el frame para que el video de salida conserve la duracion real
                for _ in range(args.skip_frames):
                    writer.write(annotated)
            if args.show:
                cv2.imshow("Conteo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nDetenido por el usuario")

    finally:
        elapsed = time.time() - start_time
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        report = {
            "fuente": str(args.source),
            "modelo": args.model,
            "zonas_archivo": args.zones,
            "frames_procesados": frame_count,
            "duracion_video_s": round(frame_count / fps, 1),
            "tiempo_proceso_s": round(elapsed, 1),
            **analyzer.summary(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print()
        print(f"Jarras: {analyzer.lifted} levantadas y {analyzer.returned} devueltas")
        print(f"Total objetos de cocina: {analyzer.total_lifted} levantados "
              f"y {analyzer.total_returned} devueltos")
        by_class = analyzer.counts_by_class()
        if by_class:
            print("Desglose por clase:")
            for cls_name, c in sorted(by_class.items()):
                print(f"  - {cls_name}: {c['levantados']} levantados, {c['devueltos']} devueltos")
        print(f"Reporte: {report_path}")
        if writer:
            print(f"Video:   {video_path}")


if __name__ == "__main__":
    main()
