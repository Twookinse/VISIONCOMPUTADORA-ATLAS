# VISIONCOMPUTADORA-ATLAS (Nex Vision AI)

Sistema de analisis por vision computacional multi-camara para la cocina y almacen del karaoke Nex (La Paz). Utiliza las 3 camaras del local (`Cocina 1`, `Cocina 2` y `Almacen`) para registrar automaticamente movimientos de jarras, utensilios e insumos y mostrarlos en un panel web en tiempo real.

## Estructura del proyecto

- `config/cameras.example.json` - Plantilla de la configuracion de las 3 camaras. Al arrancar, el servidor crea `config/cameras.json` (ignorado por Git) donde se guardan los enlaces RTSP con la clave del DVR.
- `config/zones/` - Poligonos de zonas ROI por camara (`cocina_1.json`, `cocina_2.json`, `almacen.json`).
- `config/trackers/` - Parametros de ByteTrack ajustados para oclusiones en cocina.
- `models/custom/` - Modelo YOLOv11 custom entrenado con las camaras reales de Nex (`jarras_nex_v1.pt`).
- `src/detection/` - Wrappers de deteccion y tracking (YOLO + ByteTrack).
- `src/events/` - Manejo de zonas ROI y motor de eventos de entrada/salida con histeresis.
- `src/analysis/` - Logica de cocina y normalizacion de clases de Nex.
- `src/config/` - Carga y actualizacion de configuracion multi-camara.
- `src/db/` - Modelos SQLAlchemy y persistencia en SQLite (`output/nex_vision.db`).
- `src/api/` - Backend FastAPI (API REST, WebSocket en vivo y streaming MJPEG).
- `src/live/` - Monitoreo en vivo: lector de camaras RTSP/USB (ultimo frame, reconexion automatica) y sesiones de conteo en tiempo real.
- `dashboard/` - Frontend en React + Vite con selector multi-camara y filtros.
- `scripts/` - Utilidades CLI (`run_counting.py`, `draw_zones.py`, `extract_frames.py`, `count_validator.py`).
- `Iniciar_Nex.bat` - Lanzador unificado de 1 clic (inicia API + BD + Dashboard en `http://localhost:8000`).

## Uso rapido (Modo Local / Pruebas)

1. Hacer doble clic en `Iniciar_Nex.bat` (abre automaticamente `http://localhost:8000`).
2. Usar las pestanas superiores para alternar entre **Cocina 1**, **Cocina 2** y **Almacen**.
3. Reproducir grabaciones con marcas en vivo o subir un video para contar automaticamente.

## Que cambiar al instalar en el local de Nex (Solo 1 paso)

No es necesario modificar codigo fuente. Al conectar la PC a la red local del karaoke Nex:

1. Abrir `http://localhost:8000` (o editar `config/cameras.json`, nunca la plantilla `cameras.example.json`).
2. Pegar el enlace `rtsp://usuario:clave@IP_DVR:554/...` correspondiente a cada camara (`cocina_1`, `cocina_2`, `almacen`) y presionar **Guardar RTSP**.
3. Presionar **Conectar en Vivo** para comenzar el monitoreo en tiempo real.

## Seguridad

- **Claves RTSP:** se guardan solo en `config/cameras.json`, que no se sube a GitHub. El Dashboard, la API, la base de datos y el video muestran siempre el enlace oculto (`rtsp://admin:****@IP/...`). Para cambiar la clave hay que escribir el enlace completo de nuevo.
- **Acceso desde otros equipos (celular, otra PC):** desde la propia PC del servidor se entra sin contrasena. Para permitir el acceso desde la red del local, copiar `.env.example` como `.env` y definir `NEX_PASSWORD` (el usuario por defecto es `nex`). Si `NEX_PASSWORD` esta vacia, el acceso remoto queda bloqueado.
- **Archivos permitidos:** la API solo carga modelos de `models/`, zonas de `config/zones/`, trackers de `config/trackers/` y videos de `videos/`. La subida de videos acepta solo `.mp4`, `.avi`, `.mov` y `.mkv`, nunca sobrescribe un archivo existente y tiene un limite de tamano (`NEX_MAX_UPLOAD_MB`, 4096 MB por defecto).

## Pruebas

```
python -m pytest tests/test_security.py tests/test_event_engine.py -v
python tests/test_api.py
```
