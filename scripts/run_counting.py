"""
Script para contar jarras y utensilios levantados en un video o camara de Nex.

Detecta automaticamente la camara y sus zonas ROI desde config/cameras.json
(cocina_1, cocina_2, almacen) y guarda el reporte en output/ y en SQLite.

Uso:
    python scripts/run_counting.py --source "videos/COCINA 1 VIDEO 1.mp4"
    python scripts/run_counting.py --camera cocina_2
    python scripts/run_counting.py --source rtsp://ip:554/stream --camera cocina_1 --show
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
from src.detection.visualizer import render_annotated_frame, save_event_snapshot
from src.events.zones import load_zones, draw_zones
from src.analysis.kitchen import KitchenAnalyzer
from src.config.cameras import get_camera_by_id, detect_camera_for_source
from src.db.models import save_report_to_db


def parse_args():
    parser = argparse.ArgumentParser(description="Contar objetos de cocina levantados (Nex)")
    parser.add_argument("--source", default=None,
                        help="Video, 0 para camara web, o URL RTSP (opcional si usas --camera)")
    parser.add_argument("--camera", default=None,
                        help="ID de camara de config/cameras.json: cocina_1, cocina_2 o almacen")
    parser.add_argument("--model", default="models/custom/jarras_nex_v1.pt",
                        help="Modelo YOLO (default: models/custom/jarras_nex_v1.pt)")
    parser.add_argument("--zones", default=None,
                        help="Archivo de zonas (si se omite, se detecta segun la camara)")
    parser.add_argument("--tracker", default="config/trackers/bytetrack_cocina.yaml",
                        help="Configuracion del tracker")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Clases especificas a monitorear (si se omite, monitorea todas las movibles)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confianza minima de deteccion (default: 0.25)")
    parser.add_argument("--min-frames", type=int, default=4,
                        help="Frames procesados para confirmar una salida o entrada (default: 4)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Procesar 1 de cada N frames (0 = automatico: 3 en 60fps, 1 en 30fps)")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Opcional: limitar analisis a los primeros N segundos (0 = video completo)")
    parser.add_argument("--no-video", action="store_true",
                        help="No guardar video MP4 anotado (mas rapido, solo JSON + SQLite)")
    parser.add_argument("--output", default="output",
                        help="Carpeta de salida (default: output/)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar el video mientras se procesa")
    return parser.parse_args()


def main():
    args = parse_args()

    cam_cfg = get_camera_by_id(args.camera) if args.camera else None
    if not args.source:
        if cam_cfg:
            args.source = cam_cfg.get("rtsp_url") or cam_cfg.get("default_video")
        else:
            print("Error: debes indicar --source <video/rtsp> o --camera <cocina_1|cocina_2|almacen>")
            sys.exit(1)

    if not cam_cfg:
        cam_cfg = detect_camera_for_source(args.source)

    camera_id = cam_cfg["id"] if cam_cfg else "cocina_1"
    zones_rel = args.zones or (cam_cfg["zones_file"] if cam_cfg else "config/zones/cocina_1.json")

    source = int(args.source) if str(args.source).isdigit() else args.source

    zones = load_zones(PROJECT_ROOT / zones_rel)
    if not zones:
        print(f"Error: no hay zonas en '{zones_rel}'. Dibujalas con scripts/draw_zones.py")
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

    skip_frames = args.skip_frames if args.skip_frames >= 1 else (3 if fps >= 45 else 1)
    max_frames = int(args.max_seconds * fps) if args.max_seconds > 0 else 0

    print(f"Camara: {camera_id} ({cam_cfg['nombre'] if cam_cfg else camera_id})")
    print(f"Fuente: {args.source} ({width}x{height} @ {fps:.1f} FPS, skip={skip_frames})")
    print(f"Modelo: {args.model}")
    print(f"Zonas ({len(analyzer.zones)}): {[z.name for z in analyzer.zones]}")
    print()

    writer = None
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(str(source)).stem if isinstance(source, str) else camera_id
    video_path = output_dir / f"{source_name}_conteo.mp4"
    report_path = output_dir / f"{source_name}_conteo.json"
    if is_file and not args.no_video:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    frame_count = 0
    processed_count = 0
    last_event_text = ""
    last_event_frame = -999
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            if frame_count % skip_frames != 0:
                continue

            processed_count += 1
            results = tracker.track(frame)
            new_events = analyzer.update(results, tracker.class_names, frame_count, fps)

            for event in new_events:
                event.snapshot_file = save_event_snapshot(
                    frame, results, event, analyzer.zones, output_dir / "snapshots", camera_id
                )
                action = "LEVANTADO/A" if event.event_type == "salida" else "DEVUELTO/A"
                last_event_text = f"{event.class_name} #{event.track_id} {action} ({event.zone})"
                last_event_frame = frame_count
                print(f"  [{event.time_s:6.1f}s] {event.class_name} #{event.track_id} "
                      f"{event.event_type} de '{event.zone}' -> foto: {event.snapshot_file}")

            if writer or args.show:
                annotated = render_annotated_frame(
                    frame=frame,
                    results=results,
                    class_names=tracker.class_names,
                    zones=analyzer.zones,
                    view_mode="limpia",
                    show_zones=True,
                )
                cv2.rectangle(annotated, (0, 0), (width, 40), (0, 0, 0), -1)
                hud_text = (f"[{camera_id.upper()}] Jarras Lev: {analyzer.lifted} "
                            f"Dev: {analyzer.returned} | Total Obj Lev: {analyzer.total_lifted}")
                cv2.putText(annotated, hud_text,
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                if frame_count - last_event_frame < fps * 1.5:
                    cv2.putText(annotated, last_event_text, (10, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if writer:
                    for _ in range(skip_frames):
                        writer.write(annotated)
                if args.show:
                    cv2.imshow("Conteo Nex", annotated)
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
            "camera_id": camera_id,
            "fuente": str(args.source),
            "modelo": args.model,
            "zonas_archivo": zones_rel,
            "frames_procesados": frame_count,
            "frames_inferidos": processed_count,
            "duracion_video_s": round(frame_count / fps, 1),
            "tiempo_proceso_s": round(elapsed, 1),
            **analyzer.summary(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        run_id = save_report_to_db(report)

        print()
        print(f"Camara: {camera_id}")
        print(f"Jarras: {analyzer.lifted} levantadas y {analyzer.returned} devueltas")
        print(f"Total objetos de cocina: {analyzer.total_lifted} levantados "
              f"y {analyzer.total_returned} devueltos")
        by_class = analyzer.counts_by_class()
        if by_class:
            print("Desglose por clase:")
            for cls_name, c in sorted(by_class.items()):
                print(f"  - {cls_name}: {c['levantados']} levantados, {c['devueltos']} devueltos")
        print(f"Reporte: {report_path}")
        print(f"Guardado en BD SQLite (corrida #{run_id})")
        if writer:
            print(f"Video:   {video_path}")


if __name__ == "__main__":
    main()
