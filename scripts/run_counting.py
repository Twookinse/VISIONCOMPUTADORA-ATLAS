"""
Script para contar jarras y utensilios levantados en un video o camara de Nex.

Detecta automaticamente la camara y sus zonas ROI desde config/cameras.json
(cocina_1, cocina_2, almacen) y guarda el reporte en output/ y en SQLite.
Usa el mismo motor de analisis que el Dashboard (src/pipeline/video_analysis.py).

Uso:
    python scripts/run_counting.py --source "videos/COCINA 1 VIDEO 1.mp4"
    python scripts/run_counting.py --source "videos/COCINA 1 VIDEO 1.mp4" --video-start 2026-11-22T21:30
    python scripts/run_counting.py --source "videos/COCINA 2 VIDEO 2.mp4" --start-seconds 120 --max-seconds 90
    python scripts/run_counting.py --camera cocina_2
    python scripts/run_counting.py --source rtsp://ip:554/stream --camera cocina_1 --show
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.cameras import detect_camera_for_source, get_camera_by_id, get_default_model
from src.detection.visualizer import render_annotated_frame
from src.pipeline.video_analysis import (
    DEFAULT_TRACKER,
    AnalysisError,
    AnalysisRequest,
    analyze_video,
    save_report,
)
from src.utils.video_meta import resolve_video_start


def parse_args():
    parser = argparse.ArgumentParser(description="Contar objetos de cocina levantados (Nex)")
    parser.add_argument("--source", default=None,
                        help="Video, 0 para camara web, o URL RTSP (opcional si usas --camera)")
    parser.add_argument("--camera", default=None,
                        help="ID de camara de config/cameras.json: cocina_1, cocina_2 o almacen")
    parser.add_argument("--model", default=None,
                        help="Modelo YOLO (default: 'modelo_default' de config/cameras.json)")
    parser.add_argument("--zones", default=None,
                        help="Archivo de zonas (si se omite, se detecta segun la camara)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER,
                        help="Configuracion del tracker")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Clases especificas a monitorear (si se omite, monitorea todas las movibles)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confianza minima de deteccion (default: 0.25)")
    parser.add_argument("--min-frames", type=int, default=4,
                        help="Frames procesados para confirmar una salida o entrada (default: 4)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Procesar 1 de cada N frames (0 = automatico: 3 en 60fps, 1 en 30fps)")
    parser.add_argument("--start-seconds", type=float, default=0.0,
                        help="Opcional: segundo del video desde donde empezar (ej. 120 = minuto 2)")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Opcional: analizar solo N segundos (0 = hasta el final). Un tramo se guarda "
                             "en modo 'prueba': no cuenta en las metricas ni reemplaza el analisis completo")
    parser.add_argument("--video-start", default=None,
                        help="Hora real de inicio de la grabacion (YYYY-MM-DDTHH:MM). Si se omite, "
                             "se intenta leer del nombre del archivo del DVR")
    parser.add_argument("--no-video", action="store_true",
                        help="No guardar video MP4 anotado (mas rapido, solo JSON + SQLite)")
    parser.add_argument("--output", default="output",
                        help="Carpeta de salida (default: output/)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar el video mientras se procesa (Q para detener)")
    return parser.parse_args()


def build_request(args):
    cam_cfg = get_camera_by_id(args.camera) if args.camera else None
    source = args.source
    if not source:
        if not cam_cfg:
            sys.exit("Error: debes indicar --source <video/rtsp> o --camera <cocina_1|cocina_2|almacen>")
        source = cam_cfg.get("rtsp_url") or cam_cfg.get("default_video")
    if not cam_cfg:
        cam_cfg = detect_camera_for_source(source)

    request = AnalysisRequest(
        camera_id=cam_cfg["id"] if cam_cfg else "cocina_1",
        source=str(source),
        zones_file=args.zones or (cam_cfg["zones_file"] if cam_cfg else "config/zones/cocina_1.json"),
        model=args.model or get_default_model(),
        tracker=args.tracker,
        confidence=args.confidence,
        min_frames=args.min_frames,
        skip_frames=args.skip_frames,
        max_seconds=args.max_seconds,
        start_seconds=args.start_seconds,
        classes=args.classes,
    )
    if not request.is_live:
        start, estimated = resolve_video_start(request.source, args.video_start)
        if args.video_start and estimated:
            sys.exit("Error: --video-start invalido. Usa el formato YYYY-MM-DDTHH:MM")
        request.video_start, request.start_estimated = start, estimated
    if not (PROJECT_ROOT / request.model).is_file():
        sys.exit(f"Error: no se encontro el modelo '{request.model}'")
    return request, cam_cfg


class AnnotatedOutput:
    """Video MP4 anotado, ventana de vista previa y mensajes en consola."""

    def __init__(self, request, video_path, write_video, show):
        self.request = request
        self.video_path = video_path
        self.write_video = write_video
        self.show = show
        self.writer = None
        self.last_event_text = ""
        self.last_event_frame = -999
        self.last_frame_idx = None

    def __call__(self, frame, results, analyzer, new_events, frame_idx, fps):
        for event in new_events:
            action = "LEVANTADO/A" if event.event_type == "salida" else "DEVUELTO/A"
            self.last_event_text = f"{event.class_name} #{event.track_id} {action} ({event.zone})"
            self.last_event_frame = frame_idx
            print(f"  [{event.time_s:6.1f}s] {event.class_name} #{event.track_id} "
                  f"{event.event_type} de '{event.zone}' -> foto: {event.snapshot_file}")

        if not (self.write_video or self.show):
            return True

        annotated = render_annotated_frame(
            frame=frame, results=results, class_names=results.names if hasattr(results, "names") else {},
            zones=analyzer.zones, view_mode="limpia", show_zones=True,
        )
        height, width = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (width, 40), (0, 0, 0), -1)
        hud = (f"[{self.request.camera_id.upper()}] Jarras Lev: {analyzer.lifted} "
               f"Dev: {analyzer.returned} | Total Obj Lev: {analyzer.total_lifted}")
        cv2.putText(annotated, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        if frame_idx - self.last_event_frame < fps * 1.5:
            cv2.putText(annotated, self.last_event_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.write_video:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                              fps, (width, height))
            # Repetir el frame anotado para cubrir los frames que no se analizaron
            # (en un tramo, el primer frame no se repite por los frames saltados al inicio)
            repeat = 1 if self.last_frame_idx is None else max(1, frame_idx - self.last_frame_idx)
            for _ in range(repeat):
                self.writer.write(annotated)
        self.last_frame_idx = frame_idx

        if self.show:
            cv2.imshow("Conteo Nex", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self):
        if self.writer:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    request, cam_cfg = build_request(args)
    output_dir = PROJECT_ROOT / args.output

    print(f"Camara: {request.camera_id} ({cam_cfg['nombre'] if cam_cfg else request.camera_id})")
    print(f"Fuente: {request.public_source}")
    print(f"Modelo: {request.model}")
    print(f"Zonas:  {request.zones_file}")
    if request.is_partial:
        fin = f"{request.start_seconds + request.max_seconds:g} s" if request.max_seconds > 0 else "el final"
        print(f"Tramo:  desde {request.start_seconds:g} s hasta {fin} (modo prueba: no cuenta en las metricas)")
    if not request.is_live:
        estimado = " (estimada)" if request.start_estimated else ""
        print(f"Inicio de la grabacion: {request.video_start:%Y-%m-%d %H:%M}{estimado}")
    print("Ctrl+C para detener y guardar lo contado hasta el momento\n")

    video_path = output_dir / f"{request.report_name}_conteo.mp4"
    output = AnnotatedOutput(request, video_path,
                             write_video=not request.is_live and not args.no_video, show=args.show)
    try:
        report = analyze_video(request, snapshots_dir=output_dir / "snapshots", on_frame=output)
    except AnalysisError as exc:
        sys.exit(f"Error: {exc}")
    finally:
        output.close()

    report_path, run_id = save_report(report, output_dir, request.report_name)

    print()
    if report["cancelado"]:
        print("Analisis detenido por el usuario (se guardo lo contado hasta ese momento)")
    print(f"Camara: {request.camera_id}")
    print(f"Jarras: {report['jarras_levantadas']} levantadas y {report['jarras_devueltas']} devueltas")
    print(f"Total objetos de cocina: {report['objetos_levantados_total']} levantados "
          f"y {report['objetos_devueltos_total']} devueltos")
    if report["conteo_por_clase"]:
        print("Desglose por clase:")
        for cls_name, c in sorted(report["conteo_por_clase"].items()):
            print(f"  - {cls_name}: {c['levantados']} levantados, {c['devueltos']} devueltos")
    print(f"Reporte: {report_path}")
    print(f"Guardado en BD SQLite (corrida #{run_id})")
    if output.writer:
        print(f"Video:   {video_path}")


if __name__ == "__main__":
    main()
