"""
Renderizado limpio de video y generacion de capturas (snapshots) por evento.

Evita el amontonamiento visual de cajas de YOLO permitiendo elegir entre:
- 'limpia':      Oculta objetos fijos/personas, marca objetos quietos con un
                 punto sutil, dibuja borde fino en Jarras y resalta en color
                 brillante cualquier objeto que acaba de cruzar una zona.
- 'solo_jarras': Muestra unicamente Jarras y objetos con evento reciente.
- 'tecnica':     Muestra todas las detecciones (nombres normalizados) para depuracion.
"""

from pathlib import Path
import cv2
import numpy as np

from src.analysis.kitchen import normalize_class_name, JAR_ONLY_CLASSES

# Colores BGR para OpenCV
COLOR_JAR = (48, 161, 252)         # Ambar / naranja
COLOR_LIFTED = (62, 62, 249)       # Rojo vivo (levantado / salida)
COLOR_RETURNED = (144, 204, 73)    # Verde limpio (devuelto / entrada)
COLOR_OBJECT_DOT = (254, 175, 97)  # Azul claro discreto
COLOR_ZONE_LINE = (48, 180, 240)   # Amarillo/ambar sobrio
COLOR_TECH = (210, 170, 60)


def draw_clean_zones(frame, zones, subtle=True):
    """Dibuja las zonas ROI de forma limpia (solo borde fino en modo sutil)."""
    if not zones:
        return frame
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if subtle:
        overlay = annotated.copy()
        for zone in zones:
            polygon = zone.to_pixels(width, height)
            cv2.fillPoly(overlay, [polygon], COLOR_ZONE_LINE)
            cv2.polylines(annotated, [polygon], True, COLOR_ZONE_LINE, 1, cv2.LINE_AA)
            x, y = polygon[0]
            cv2.putText(
                annotated, zone.name, (x + 6, max(20, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE_LINE, 1, cv2.LINE_AA,
            )
        return cv2.addWeighted(overlay, 0.06, annotated, 0.94, 0)

    overlay = annotated.copy()
    for zone in zones:
        polygon = zone.to_pixels(width, height)
        cv2.fillPoly(overlay, [polygon], COLOR_ZONE_LINE)
        cv2.polylines(annotated, [polygon], True, COLOR_ZONE_LINE, 2, cv2.LINE_AA)
        x, y = polygon[0]
        cv2.putText(
            annotated, zone.name, (x + 6, max(22, y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ZONE_LINE, 2, cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.16, annotated, 0.84, 0)


def _draw_label(img, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.52):
    """Dibuja una pastilla de texto compacta y legible sin tapar demasiado el video."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(th + 6, y)
    cv2.rectangle(img, (x, y_top - th - 6), (x + tw + 8, y_top + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 4, y_top - 4), font, scale, text_color, thickness, cv2.LINE_AA)


def render_annotated_frame(
    frame,
    results,
    class_names,
    zones,
    view_mode="limpia",
    show_zones=True,
    recent_track_events=None,
):
    """Renderiza el frame segun el modo de visualizacion elegido.

    Args:
        frame: Frame BGR original.
        results: Resultado de Ultralytics Tracker.track().
        class_names: Diccionario id -> nombre crudo del modelo.
        zones: Lista de Zone.
        view_mode: 'limpia', 'solo_jarras' o 'tecnica'.
        show_zones: Si True dibuja las zonas ROI.
        recent_track_events: Dict {track_id: event_type} para resaltar objetos
            que acaban de disparar un evento ('salida' o 'entrada').
    """
    recent = recent_track_events or {}
    annotated = frame.copy()

    if show_zones:
        annotated = draw_clean_zones(annotated, zones, subtle=(view_mode != "tecnica"))

    if results.boxes is None or len(results.boxes) == 0:
        return annotated

    for box in results.boxes:
        raw_name = class_names[int(box.cls[0])]
        clean_name = normalize_class_name(raw_name)
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx, cy = int((x1 + x2) / 2), y2

        # En modos no tecnicos, ocultar elementos fijos y personas
        if view_mode != "tecnica" and clean_name is None:
            continue

        label_name = clean_name or raw_name
        is_jar = clean_name in JAR_ONLY_CLASSES
        recent_ev = recent.get(track_id) if track_id is not None else None

        # 1. Si el objeto acaba de generar evento -> SIEMPRE resaltar con claridad
        if recent_ev:
            color = COLOR_LIFTED if recent_ev == "salida" else COLOR_RETURNED
            tag = "LEVANTADO" if recent_ev == "salida" else "DEVUELTO"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            id_txt = f" #{track_id}" if track_id is not None else ""
            _draw_label(annotated, f"{label_name}{id_txt} - {tag}", x1, y1 - 4, color, scale=0.6)
            continue

        # 2. Modo 'solo_jarras': solo dibuja jarras (muy despejado)
        if view_mode == "solo_jarras":
            if not is_jar:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_JAR, 2, cv2.LINE_AA)
            id_txt = f" #{track_id}" if track_id is not None else ""
            _draw_label(annotated, f"{label_name}{id_txt}", x1, y1 - 4, COLOR_JAR, text_color=(20, 20, 20))
            continue

        # 3. Modo 'limpia': jarras con marco fino + demas objetos movibles solo con punto en base
        if view_mode == "limpia":
            if is_jar:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_JAR, 2, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 4, COLOR_JAR, -1, cv2.LINE_AA)
                id_txt = f" #{track_id}" if track_id is not None else ""
                _draw_label(annotated, f"{label_name}{id_txt}", x1, y1 - 4, COLOR_JAR, text_color=(20, 20, 20))
            else:
                # Punto sutil en la base para no saturar la pantalla con 15 cuadros
                cv2.circle(annotated, (cx, cy), 5, COLOR_OBJECT_DOT, -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 6, (40, 40, 40), 1, cv2.LINE_AA)
            continue

        # 4. Modo 'tecnica': todo dibujado con nombres normalizados
        color = COLOR_JAR if is_jar else (COLOR_TECH if clean_name else (140, 140, 140))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        id_txt = f" #{track_id}" if track_id is not None else ""
        _draw_label(annotated, f"{label_name}{id_txt} {conf:.2f}", x1, y1 - 4, color, text_color=(20, 20, 20))

    return annotated


def save_event_snapshot(frame, results, event, zones, snapshots_dir, camera_id):
    """Guarda una captura limpia del instante exacto de un evento para la bitacora.

    Resalta unicamente el objeto que disparo el evento y su zona ROI.

    Returns:
        Nombre del archivo .jpg guardado (relativo a snapshots_dir).
    """
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snap = frame.copy()
    # Dibujar unicamente la zona involucrada en este evento
    target_zones = [z for z in zones if z.name == event.zone]
    snap = draw_clean_zones(snap, target_zones or zones, subtle=True)

    color = COLOR_LIFTED if event.event_type == "salida" else COLOR_RETURNED
    action_txt = "LEVANTADO (Salida)" if event.event_type == "salida" else "DEVUELTO (Entrada)"

    # Resaltar la caja del objeto exacto que disparo el evento
    if results is not None and results.boxes is not None and results.boxes.id is not None:
        for box in results.boxes:
            if int(box.id[0]) == event.track_id:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cv2.rectangle(snap, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                _draw_label(
                    snap,
                    f"{event.class_name} #{event.track_id} - {action_txt}",
                    x1,
                    y1 - 6,
                    color,
                    scale=0.65,
                )
                break

    # Encabezado informativo superior
    h, w = snap.shape[:2]
    cv2.rectangle(snap, (0, 0), (w, 42), (30, 34, 42), -1)
    header = f"[{camera_id.upper()}] {event.class_name} #{event.track_id} | {action_txt} | Zona: {event.zone} | {event.time_s}s"
    cv2.putText(snap, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Redimensionar a max 860px para que cargue al instante en el Dashboard
    if w > 860:
        scale = 860.0 / w
        snap = cv2.resize(snap, (860, int(h * scale)), interpolation=cv2.INTER_AREA)

    safe_cls = event.class_name.replace(" ", "_")
    filename = f"{camera_id}_{safe_cls}_id{event.track_id}_f{event.frame}_{event.event_type}.jpg"
    out_path = snapshots_dir / filename
    cv2.imwrite(str(out_path), snap, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return filename
