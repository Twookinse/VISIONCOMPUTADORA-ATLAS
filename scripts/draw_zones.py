"""
Herramienta visual para dibujar zonas de interes (ROI) sobre un frame.

Uso:
    python scripts/draw_zones.py --source "videos/VIDEO 2 VISION COMPUTADORA.mp4"
    python scripts/draw_zones.py --source 0                (camara web)
    python scripts/draw_zones.py --source video.mp4 --frame 300

Controles en la ventana:
    Clic izquierdo   Agregar un punto a la zona actual
    Enter            Cerrar la zona actual (pide el nombre en la terminal)
    N                Adelantar el video (~2 segundos) para ver mejor el fondo
    B                Retroceder el video (~2 segundos)
    Z                Deshacer el ultimo punto
    D                Borrar la ultima zona
    S                Guardar manualmente (no es necesario, se guarda solo)
    Q                Salir

Cada vez que se agrega o borra una zona se guarda automaticamente.
Despues de escribir el nombre en la terminal, haz clic en la ventana
de la imagen para que vuelva a recibir las teclas.

Las coordenadas se guardan normalizadas (0 a 1), asi la misma configuracion
sirve aunque la camara de produccion tenga otra resolucion.
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.events.zones import Zone, load_zones, save_zones, draw_zones

WINDOW_NAME = "Dibujar zonas"


def parse_args():
    parser = argparse.ArgumentParser(description="Dibujar zonas ROI sobre un frame")
    parser.add_argument("--source", required=True,
                        help="Video, 0 para camara web, o URL RTSP")
    parser.add_argument("--frame", type=int, default=300,
                        help="Numero de frame a usar como fondo (default: 300, salta el logo inicial)")
    parser.add_argument("--zones", default="config/zones/casa_cocina.json",
                        help="Archivo JSON de zonas (default: config/zones/casa_cocina.json)")
    return parser.parse_args()


def read_background(source, frame_number):
    """Obtiene el frame sobre el que se van a dibujar las zonas."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    zones_path = PROJECT_ROOT / args.zones

    frame_num = args.frame
    background = read_background(source, frame_num)
    if background is None:
        frame_num = 0
        background = read_background(source, 0)
    if background is None:
        print(f"Error: no se pudo leer un frame de '{args.source}'")
        sys.exit(1)

    height, width = background.shape[:2]
    zones = load_zones(zones_path)
    current_points = []  # puntos en pixeles de la zona que se esta dibujando

    print(f"Resolucion real del video: {width}x{height} (mostrando frame #{frame_num})")
    print(f"Zonas cargadas: {len(zones)} desde {args.zones}")
    print("Clic: punto | Enter: cerrar zona | N/B: adelantar/atrasar video | Z: deshacer | D: borrar zona | Q: salir")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    # WINDOW_NORMAL permite redimensionar la ventana a 1280x720 para que un video
    # 1920x1080 se vea completo en pantalla sin recortarse ni verse con zoom.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        view = draw_zones(background, zones)

        # Dibujar la zona en construccion
        for i, point in enumerate(current_points):
            cv2.circle(view, point, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(view, current_points[i - 1], point, (0, 255, 0), 2)

        cv2.putText(view, f"Frame: {frame_num} (N:+2s / B:-2s) | Zonas: {len(zones)} | Puntos: {len(current_points)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, view)
        key = cv2.waitKey(30) & 0xFF

        # Si se cerro la ventana con la X, terminar el programa
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("n"):
            next_bg = read_background(source, frame_num + 120)
            if next_bg is not None:
                frame_num += 120
                background = next_bg
                print(f"Frame adelantado a #{frame_num}")

        elif key == ord("b"):
            prev_frame = max(0, frame_num - 120)
            prev_bg = read_background(source, prev_frame)
            if prev_bg is not None:
                frame_num = prev_frame
                background = prev_bg
                print(f"Frame retrocedido a #{frame_num}")

        elif key == 13:  # Enter
            if len(current_points) < 3:
                print("Una zona necesita al menos 3 puntos")
                continue
            name = input("Nombre de la zona: ").strip() or f"zona_{len(zones) + 1}"
            zone_type = input("Tipo (reposo/servicio/general) [reposo]: ").strip() or "reposo"
            normalized = [[x / width, y / height] for x, y in current_points]
            zones.append(Zone(name, zone_type, normalized))
            current_points.clear()
            save_zones(zones_path, zones)
            print(f"Zona '{name}' agregada y guardada en {args.zones}")
            print("Haz clic en la ventana de la imagen para seguir (Q para salir)")

        elif key == ord("z") and current_points:
            current_points.pop()

        elif key == ord("d") and zones:
            removed = zones.pop()
            save_zones(zones_path, zones)
            print(f"Zona '{removed.name}' eliminada y cambios guardados")

        elif key == ord("s"):
            save_zones(zones_path, zones)
            print(f"{len(zones)} zonas guardadas en {args.zones}")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
