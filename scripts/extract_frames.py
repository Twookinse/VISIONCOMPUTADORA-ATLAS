"""
Script para extraer frames de uno o varios videos para armar el dataset (Sprint 3).

Guarda imagenes .jpg cada N frames para luego subirlas a Roboflow o CVAT
y etiquetar las clases custom (ej: jarra, vaso, plato).

Uso:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --every 15
    python scripts/extract_frames.py --source "videos/VIDEO 2 VISION COMPUTADORA.mp4" --every 30
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extraer frames de videos para etiquetado de dataset YOLO"
    )
    parser.add_argument(
        "--source",
        default="videos",
        help="Ruta a un archivo de video o carpeta con videos (default: videos/)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=30,
        help="Guardar 1 frame cada N frames (default: 30, aprox. 1 por segundo a 30 FPS)",
    )
    parser.add_argument(
        "--output",
        default="output/frames",
        help="Carpeta donde se guardaran las imagenes .jpg (default: output/frames)",
    )
    return parser.parse_args()


def get_video_files(source_path):
    """Obtiene la lista de videos a procesar a partir de un archivo o carpeta."""
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        files = []
        for ext in exts:
            files.extend(sorted(source_path.glob(ext)))
        return files
    return []


def extract_from_video(video_path, output_dir, every_n):
    """Extrae 1 de cada every_n frames de un video y los guarda como JPG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir '{video_path}'")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prefix = video_path.stem.replace(" ", "_")

    print(f"Procesando: {video_path.name} ({total_frames} frames, {fps:.1f} FPS)")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            filename = f"{prefix}_frame_{frame_idx:05d}.jpg"
            out_file = output_dir / filename
            cv2.imwrite(str(out_file), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"  -> {saved_count} imagenes guardadas de '{video_path.name}'")
    return saved_count


def main():
    args = parse_args()
    if args.every < 1:
        print("Error: --every debe ser mayor o igual a 1")
        sys.exit(1)

    source_path = PROJECT_ROOT / args.source
    videos = get_video_files(source_path)

    if not videos:
        print(f"Error: no se encontraron videos en '{args.source}'")
        sys.exit(1)

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extrayendo 1 frame cada {args.every} frames...")
    print(f"Carpeta de destino: {output_dir}\n")

    total_saved = 0
    for v_path in videos:
        total_saved += extract_from_video(v_path, output_dir, args.every)

    print("\n" + "=" * 60)
    print(f"Total de imagenes extraidas: {total_saved}")
    print(f"Ubicacion: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
