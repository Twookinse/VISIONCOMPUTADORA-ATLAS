"""
Script para procesar un video con deteccion de objetos YOLO.

Uso:
    python scripts/run_detection.py --source videos/mi_video.mp4
    python scripts/run_detection.py --source 0                     (camara web)
    python scripts/run_detection.py --source rtsp://ip:554/stream  (camara IP)

Genera un video de salida en la carpeta output/ con los rectangulos
de deteccion dibujados sobre cada frame.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Agregar la raiz del proyecto al path para importar src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import Detector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Procesar video con deteccion de objetos YOLO"
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
        help="Procesar 1 de cada N frames para mayor velocidad (default: 1, sin saltar)",
    )
    return parser.parse_args()


def draw_detections(frame, results, detector):
    """Dibuja los rectangulos de deteccion sobre el frame.

    Usa el metodo plot() nativo de ultralytics que genera
    anotaciones de alta calidad con colores por clase.

    Args:
        frame: Imagen numpy array (BGR). Se usa como fallback.
        results: Objeto Results de ultralytics.
        detector: Instancia de Detector (reservado para uso futuro).

    Returns:
        Frame con las detecciones dibujadas.
    """
    return results.plot()


def main():
    args = parse_args()

    # Determinar si la fuente es una camara (numero) o un archivo/URL
    source = int(args.source) if args.source.isdigit() else args.source

    # Crear el detector
    model_path = PROJECT_ROOT / args.model
    detector = Detector(
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
    print(f"Saltar frames: cada {args.skip_frames}")
    print()

    # Preparar video de salida (solo para archivos, no en vivo)
    writer = None
    if is_file:
        output_dir = PROJECT_ROOT / args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(str(source)).stem if isinstance(source, str) else "webcam"
        output_path = output_dir / f"{source_name}_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Video de salida: {output_path}")

    # Procesar frames
    frame_count = 0
    detection_count = 0
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

            # Detectar
            results = detector.detect(frame)
            n_detections = len(results.boxes)
            detection_count += n_detections

            # Dibujar
            annotated = draw_detections(frame, results, detector)

            # Mostrar info en el frame
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            info = f"Frame {frame_count}"
            if is_file:
                info += f"/{total_frames}"
            info += f" | {current_fps:.1f} FPS | {n_detections} objetos"
            cv2.putText(
                annotated, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # Escribir al video de salida
            if writer:
                writer.write(annotated)

            # Mostrar en ventana si se pidio
            if args.show:
                cv2.imshow("Detecciones", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nDetenido por el usuario (tecla Q)")
                    break

            # Progreso cada 100 frames
            if frame_count % 100 == 0:
                progress = f"{frame_count}/{total_frames}" if is_file else str(frame_count)
                print(f"  Procesando... {progress} frames | {current_fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\nDetenido por el usuario (Ctrl+C)")

    finally:
        elapsed = time.time() - start_time
        print(f"\nResumen:")
        print(f"  Frames procesados: {frame_count}")
        print(f"  Tiempo total: {elapsed:.1f}s")
        print(f"  FPS promedio: {frame_count / elapsed if elapsed > 0 else 0:.1f}")
        print(f"  Detecciones totales: {detection_count}")

        cap.release()
        if writer:
            writer.release()
            print(f"  Video guardado en: {output_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
