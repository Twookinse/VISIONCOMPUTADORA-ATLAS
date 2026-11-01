"""
Script para procesar un video con deteccion + tracking de objetos.

A diferencia de run_detection.py, aqui cada objeto mantiene un ID
unico entre frames. Esto permite saber que la "jarra #3" del frame
anterior es la misma del frame actual.

Uso:
    python scripts/run_tracking.py --source videos/mi_video.mp4
    python scripts/run_tracking.py --source 0                     (camara web)
    python scripts/run_tracking.py --source rtsp://ip:554/stream  (camara IP)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Agregar la raiz del proyecto al path para importar src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.tracker import Tracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Procesar video con deteccion y tracking de objetos"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Fuente de video: ruta a archivo, 0 para camara web, o URL RTSP",
    )
    parser.add_argument(
        "--model",
        default="models/custom/jarras_nex_v1.pt",
        help="Ruta al modelo YOLO (default: models/custom/jarras_nex_v1.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confianza minima de deteccion (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Carpeta de salida para el video procesado (default: output/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar el video en una ventana mientras se procesa",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Procesar 1 de cada N frames (default: 1, sin saltar)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determinar si la fuente es una camara (numero) o un archivo/URL
    source = int(args.source) if args.source.isdigit() else args.source

    # Crear el tracker
    model_path = PROJECT_ROOT / args.model
    tracker = Tracker(
        model_path=model_path,
        confidence=args.confidence,
    )

    # Abrir la fuente de video
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
    print(f"Resolucion: {width}x{height} | FPS: {fps:.1f}", end="")
    if is_file:
        print(f" | Frames: {total_frames} | Duracion: {total_frames / fps:.1f}s")
    else:
        print(" | En vivo")
    print(f"Modelo: {args.model}")
    print(f"Confianza minima: {args.confidence}")
    print()

    # Preparar video de salida
    writer = None
    if is_file:
        output_dir = PROJECT_ROOT / args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(str(source)).stem if isinstance(source, str) else "webcam"
        output_path = output_dir / f"{source_name}_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Video de salida: {output_path}")

    # Llevar registro de todos los IDs que aparecieron
    all_tracked_ids = {}  # {track_id: class_name}

    # Procesar frames
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Saltar frames si se configuro
            if frame_count % args.skip_frames != 0:
                if writer:
                    writer.write(frame)
                continue

            # Detectar + trackear
            results = tracker.track(frame)

            # Registrar los IDs de este frame
            if results.boxes.id is not None:
                for box in results.boxes:
                    track_id = int(box.id[0])
                    cls_name = tracker.get_class_name(int(box.cls[0]))
                    if track_id not in all_tracked_ids:
                        all_tracked_ids[track_id] = cls_name

            # Dibujar con el metodo nativo de ultralytics
            # (incluye rectangulos, clase, confianza e ID de tracking)
            annotated = results.plot()

            # Mostrar info en el frame
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            n_objects = len(results.boxes)
            info = f"Frame {frame_count}"
            if is_file:
                info += f"/{total_frames}"
            info += f" | {current_fps:.1f} FPS | {n_objects} objetos"
            info += f" | IDs totales: {len(all_tracked_ids)}"
            cv2.putText(
                annotated, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

            # Escribir al video de salida
            if writer:
                writer.write(annotated)

            # Mostrar en ventana si se pidio
            if args.show:
                cv2.imshow("Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nDetenido por el usuario (tecla Q)")
                    break

            # Progreso cada 100 frames
            if frame_count % 100 == 0:
                progress = f"{frame_count}/{total_frames}" if is_file else str(frame_count)
                print(f"  Procesando... {progress} frames | {current_fps:.1f} FPS | IDs: {len(all_tracked_ids)}")

    except KeyboardInterrupt:
        print("\nDetenido por el usuario (Ctrl+C)")

    finally:
        elapsed = time.time() - start_time
        print(f"\nResumen:")
        print(f"  Frames procesados: {frame_count}")
        print(f"  Tiempo total: {elapsed:.1f}s")
        print(f"  FPS promedio: {frame_count / elapsed if elapsed > 0 else 0:.1f}")
        print(f"  Objetos unicos rastreados: {len(all_tracked_ids)}")
        print()
        print(f"  Objetos detectados por clase:")
        # Contar cuantos IDs hubo por clase
        class_counts = {}
        for cls_name in all_tracked_ids.values():
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls_name}: {count} objetos unicos")

        cap.release()
        if writer:
            writer.release()
            print(f"\n  Video guardado en: {output_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
