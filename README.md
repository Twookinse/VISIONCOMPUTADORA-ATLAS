# VISIONCOMPUTADORA-ATLAS (Nex Vision AI)

Sistema de analisis por vision computacional multi-camara para la cocina y almacen del karaoke Nex (La Paz). Utiliza las 3 camaras del local (`Cocina 1`, `Cocina 2` y `Almacen`) para registrar automaticamente movimientos de jarras, utensilios e insumos y mostrarlos en un panel web en tiempo real.

## Estructura del proyecto

- `config/cameras.json` - Configuracion central de las 3 camaras (enlaces RTSP y archivo de zonas ROI de cada una).
- `config/zones/` - Poligonos de zonas ROI por camara (`cocina_1.json`, `cocina_2.json`, `almacen.json`).
- `config/trackers/` - Parametros de ByteTrack ajustados para oclusiones en cocina.
- `models/custom/` - Modelo YOLOv11 custom entrenado con las camaras reales de Nex (`jarras_nex_v1.pt`).
- `src/detection/` - Wrappers de deteccion y tracking (YOLO + ByteTrack).
- `src/events/` - Manejo de zonas ROI y motor de eventos de entrada/salida con histeresis.
- `src/analysis/` - Logica de cocina y normalizacion de clases de Nex.
- `src/config/` - Carga y actualizacion de configuracion multi-camara.
- `src/db/` - Modelos SQLAlchemy y persistencia en SQLite (`output/nex_vision.db`).
- `src/api/` - Backend FastAPI (API REST, WebSocket en vivo y streaming MJPEG).
- `dashboard/` - Frontend en React + Vite con selector multi-camara y filtros.
- `scripts/` - Utilidades CLI (`run_counting.py`, `draw_zones.py`, `extract_frames.py`, `count_validator.py`).
- `Iniciar_Nex.bat` - Lanzador unificado de 1 clic (inicia API + BD + Dashboard en `http://localhost:8000`).

## Uso rapido (Modo Local / Pruebas)

1. Hacer doble clic en `Iniciar_Nex.bat` (abre automaticamente `http://localhost:8000`).
2. Usar las pestanas superiores para alternar entre **Cocina 1**, **Cocina 2** y **Almacen**.
3. Reproducir grabaciones con marcas en vivo o subir un video para contar automaticamente.

## Que cambiar al instalar en el local de Nex (Solo 1 paso)

No es necesario modificar codigo fuente. Al conectar la PC a la red local del karaoke Nex:

1. Abrir `http://localhost:8000` (o editar `config/cameras.json`).
2. Pegar el enlace `rtsp://usuario:clave@IP_DVR:554/...` correspondiente a cada camara (`cocina_1`, `cocina_2`, `almacen`) y presionar **Guardar RTSP**.
3. Presionar **Conectar en Vivo** para comenzar el monitoreo en tiempo real.
