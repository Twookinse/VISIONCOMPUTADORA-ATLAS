"""
Herramienta visual para dibujar zonas de interes (ROI) sobre un frame.

Uso:
    python scripts/draw_zones.py --source "videos/VIDEO 2 VISION COMPUTADORA.mp4"
    python scripts/draw_zones.py --source 0                (camara web)
    python scripts/draw_zones.py --source video.mp4 --frame 300

Controles en la ventana:
    Clic izquierdo   Agregar un punto a la zona actual
    Enter            Cerrar la zona actual (pide el nombre en la terminal)
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
    parser.add_argument("--frame", type=int, default=0,
                        help="Numero de frame a usar como fondo (default: 0)")
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

    background = read_background(source, args.frame)
    if background is None:
        print(f"Error: no se pudo leer un frame de '{args.source}'")
        sys.exit(1)

    height, width = background.shape[:2]
    zones = load_zones(zones_path)
    current_points = []  # puntos en pixeles de la zona que se esta dibujando

    print(f"Zonas cargadas: {len(zones)} desde {args.zones}")
    print("Clic: punto | Enter: cerrar zona | Z: deshacer | D: borrar zona | S: guardar | Q: salir")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        view = draw_zones(background, zones)

        # Dibujar la zona en construccion
        for i, point in enumerate(current_points):
            cv2.circle(view, point, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(view, current_points[i - 1], point, (0, 255, 0), 2)

        cv2.putText(view, f"Zonas: {len(zones)} | Puntos actuales: {len(current_points)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, view)
        key = cv2.waitKey(30) & 0xFF

        # Si se cerro la ventana con la X, terminar el programa
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == 13:  # Enter
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
