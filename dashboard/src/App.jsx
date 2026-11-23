import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react'

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (window.location.port === '5173' ? 'http://localhost:8000' : window.location.origin)

const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/events'

const SHIFTS = {
  todos: { label: 'Todo el dia (24 horas)', check: () => true },
  dia: { label: 'Turno Manana / Mediodia (08:00 - 15:59)', check: (h) => h >= 8 && h < 16 },
  tarde: { label: 'Turno Tarde (16:00 - 20:59)', check: (h) => h >= 16 && h < 21 },
  noche: { label: 'Turno Noche Karaoke (21:00 - 04:59)', check: (h) => h >= 21 || h < 5 },
}

const CAMERA_LABELS = {
  cocina_1: 'Cocina 1',
  cocina_2: 'Cocina 2',
  almacen: 'Almacen',
}

export default function App() {
  const [events, setEvents] = useState([])
  const [hourly, setHourly] = useState([])
  const [totalsByClass, setTotalsByClass] = useState({})
  const [totalsByZone, setTotalsByZone] = useState({})
  const [cameras, setCameras] = useState([])
  const [availableClasses, setAvailableClasses] = useState([])
  const [videosDetail, setVideosDetail] = useState([])
  const [liveStatus, setLiveStatus] = useState({ activo: false })
  const [jobStatus, setJobStatus] = useState({ activo: false, progreso_pct: 0 })
  const [wsConnected, setWsConnected] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [loading, setLoading] = useState(true)

  // Camara activa para filtros y operacion ("" = todas)
  const [cameraFilter, setCameraFilter] = useState('')
  const [activeCamId, setActiveCamId] = useState('cocina_1')

  // Controles de video, visualizacion limpia y RTSP
  const [selectedVideo, setSelectedVideo] = useState('')
  const [rtspDraft, setRtspDraft] = useState('')
  const [viewMode, setViewMode] = useState('limpia')
  const [showZones, setShowZones] = useState(true)
  const [modalEvent, setModalEvent] = useState(null)
  const [uploadingVideo, setUploadingVideo] = useState(false)
  const [actionMsg, setActionMsg] = useState('')
  const [lastReport, setLastReport] = useState(null)
  const [streamKey, setStreamKey] = useState(1)
  const fileInputRef = useRef(null)

  // Filtros adicionales
  const [dateFilter, setDateFilter] = useState('')
  const [shiftFilter, setShiftFilter] = useState('todos')
  const [classFilter, setClassFilter] = useState('')

  const fetchData = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (cameraFilter) params.set('camera_id', cameraFilter)
      if (dateFilter) params.set('date', dateFilter)
      if (classFilter) params.set('class_name', classFilter)

      const [evRes, hrRes, camRes, vidRes, liveRes, jobRes, clsRes] = await Promise.all([
        fetch(`${API_BASE}/api/events?${params.toString()}`),
        fetch(`${API_BASE}/api/metrics/hourly?${params.toString()}`),
        fetch(`${API_BASE}/api/cameras`),
        fetch(`${API_BASE}/api/videos`),
        fetch(`${API_BASE}/api/live/status`),
        fetch(`${API_BASE}/api/pipeline/status`),
        fetch(`${API_BASE}/api/classes`),
      ])

      if (evRes.ok && hrRes.ok) {
        const evData = await evRes.json()
        const hrData = await hrRes.json()
        setEvents(evData.eventos || [])
        setHourly(hrData.por_hora || [])
        setTotalsByClass(hrData.totales_por_clase || {})
        setTotalsByZone(hrData.totales_por_zona || {})
        setApiOnline(true)
      }
      if (camRes.ok) {
        const camData = await camRes.json()
        const cams = camData.camaras || []
        setCameras(cams)
      }
      if (vidRes.ok) {
        const vidData = await vidRes.json()
        const details = vidData.detalle || []
        setVideosDetail(details)
        if (details.length > 0 && !selectedVideo) {
          setSelectedVideo(details[0].path)
        }
      }
      if (liveRes.ok) {
        const lData = await liveRes.json()
        setLiveStatus(lData)
        if (lData.view_mode) setViewMode(lData.view_mode)
        if (typeof lData.show_zones === 'boolean') setShowZones(lData.show_zones)
      }
      if (jobRes.ok) {
        const jData = await jobRes.json()
        setJobStatus(jData)
        if (!jData.activo && jData.ultimo_reporte) {
          setLastReport(jData.ultimo_reporte)
        }
      }
      if (clsRes.ok) {
        const cData = await clsRes.json()
        setAvailableClasses(cData.clases || [])
      }
    } catch (err) {
      setApiOnline(false)
    } finally {
      setLoading(false)
    }
  }, [cameraFilter, dateFilter, classFilter, selectedVideo])

  // Sincronizar borrador RTSP y video sugerido cuando cambia la camara activa
  useEffect(() => {
    const cam = cameras.find((c) => c.id === activeCamId)
    if (cam) {
      setRtspDraft(cam.rtsp_url || '')
    }
  }, [activeCamId, cameras])

  useEffect(() => {
    fetchData()
    const intervalMs = jobStatus.activo ? 1500 : 4000
    const timer = setInterval(fetchData, intervalMs)
    return () => clearInterval(timer)
  }, [fetchData, jobStatus.activo])

  useEffect(() => {
    let ws
    let reconnectTimer

    const connectWs = () => {
      try {
        ws = new WebSocket(WS_URL)
        ws.onopen = () => setWsConnected(true)
        ws.onmessage = () => fetchData()
        ws.onclose = () => {
          setWsConnected(false)
          reconnectTimer = setTimeout(connectWs, 4000)
        }
        ws.onerror = () => setWsConnected(false)
      } catch (e) {
        setWsConnected(false)
      }
    }

    connectWs()
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws) ws.close()
    }
  }, [fetchData])

  const handleSelectCameraTab = (camId) => {
    setCameraFilter(camId)
    if (camId) {
      setActiveCamId(camId)
      const camVideo = videosDetail.find((v) => v.camera_id === camId)
      if (camVideo) {
        setSelectedVideo(camVideo.path)
      }
    }
  }

  const handleVideoDropdownChange = (vidPath) => {
    setSelectedVideo(vidPath)
    const info = videosDetail.find((v) => v.path === vidPath)
    if (info && info.camera_id) {
      setActiveCamId(info.camera_id)
    }
  }

  const handleSaveRtsp = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/cameras/${activeCamId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: rtspDraft }),
      })
      if (res.ok) {
        const data = await res.json()
        setActionMsg(data.mensaje)
        fetchData()
      } else {
        setActionMsg('No se pudo guardar el enlace RTSP.')
      }
    } catch (e) {
      setActionMsg('Error al guardar el enlace RTSP.')
    }
  }

  // Cerrar visor de foto con tecla Escape
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setModalEvent(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const handleDisplayChange = async (newMode, newShowZones) => {
    const targetMode = newMode !== undefined ? newMode : viewMode
    const targetZones = newShowZones !== undefined ? newShowZones : showZones
    setViewMode(targetMode)
    setShowZones(targetZones)
    try {
      const res = await fetch(`${API_BASE}/api/live/display`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ view_mode: targetMode, show_zones: targetZones }),
      })
      if (res.ok) {
        const data = await res.json()
        setLiveStatus(data.estado)
      }
    } catch (e) {
      // Silencioso si la vista no esta activa aun
    }
  }

  const handleStartLive = async (sourceToUse, camIdOverride = null) => {
    const targetCam = camIdOverride || activeCamId
    const src = (sourceToUse || '').trim()
    if (!src) {
      setActionMsg(
        `Aun no hay enlace RTSP configurado para ${CAMERA_LABELS[targetCam] || targetCam}. Usa "Reproducir con Marcas" con un video o pega el enlace RTSP cuando estes en el local.`
      )
      return
    }
    setActiveCamId(targetCam)
    setActionMsg(`Abriendo vista con marcas para ${CAMERA_LABELS[targetCam] || targetCam}: "${src}"...`)
    setLiveStatus({
      activo: true,
      camera_id: targetCam,
      fuente: src,
      resolucion: 'Cargando...',
      view_mode: viewMode,
      show_zones: showZones,
    })
    setStreamKey(Date.now())
    try {
      const res = await fetch(`${API_BASE}/api/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: src,
          camera_id: targetCam,
          view_mode: viewMode,
          show_zones: showZones,
        }),
      })
      const data = await res.json()
      setLiveStatus(data.estado || { activo: true, camera_id: targetCam, fuente: src })
      setActionMsg(`Transmitiendo con marcas [${CAMERA_LABELS[targetCam] || targetCam}]: ${src}`)
    } catch (e) {
      setActionMsg('No se pudo iniciar la transmision en vivo.')
      setLiveStatus({ activo: false })
    }
  }

  const handleStopLive = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/live/stop`, { method: 'POST' })
      const data = await res.json()
      setLiveStatus(data.estado || { activo: false })
      setActionMsg('Vista en directo detenida.')
      fetchData()
    } catch (e) {
      setActionMsg('Error al detener la vista en directo.')
    }
  }

  const startBackgroundAnalysis = async (videoPath, camIdOverride = null) => {
    if (!videoPath) return
    const info = videosDetail.find((v) => v.path === videoPath)
    const targetCam = camIdOverride || (info ? info.camera_id : activeCamId)
    setLastReport(null)
    setActionMsg(
      `Iniciando conteo de "${videoPath.replace('videos/', '')}" con las zonas de ${CAMERA_LABELS[targetCam] || targetCam}...`
    )
    try {
      const res = await fetch(`${API_BASE}/api/pipeline/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: videoPath,
          camera_id: targetCam,
          skip_frames: 0,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setJobStatus(data.estado)
      } else {
        const err = await res.json().catch(() => ({}))
        setActionMsg(err.detail || 'No se pudo iniciar el analisis del video.')
      }
    } catch (e) {
      setActionMsg('Error de comunicacion con el servidor.')
    }
  }

  const handleCancelJob = async () => {
    try {
      await fetch(`${API_BASE}/api/pipeline/cancel`, { method: 'POST' })
      setActionMsg('Deteniendo analisis y guardando los eventos encontrados hasta ahora...')
    } catch (e) {
      setActionMsg('No se pudo cancelar el analisis.')
    }
  }

  const handleUploadAndAnalyze = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadingVideo(true)
    setActionMsg(`Cargando archivo "${file.name}"...`)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const upRes = await fetch(`${API_BASE}/api/videos/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!upRes.ok) {
        setActionMsg('No se pudo subir el archivo de video.')
        setUploadingVideo(false)
        return
      }

      const upData = await upRes.json()
      const newVideoPath = upData.video_path
      const detectedCam = upData.camera_id || activeCamId
      setSelectedVideo(newVideoPath)
      setActiveCamId(detectedCam)
      setUploadingVideo(false)

      await startBackgroundAnalysis(newVideoPath, detectedCam)
    } catch (err) {
      setActionMsg('Error al subir el video desde la computadora.')
      setUploadingVideo(false)
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const shiftChecker = SHIFTS[shiftFilter]?.check || (() => true)

  const filteredHourly = useMemo(() => {
    return hourly.filter((item) => {
      const h = parseInt(item.hora.split(':')[0], 10)
      return shiftChecker(h)
    })
  }, [hourly, shiftFilter])

  const filteredEvents = useMemo(() => {
    return events.filter((ev) => {
      if (!ev.created_at) return true
      const h = new Date(ev.created_at).getHours()
      return shiftChecker(h)
    })
  }, [events, shiftFilter])

  const kpis = useMemo(() => {
    let jarrasLev = 0
    let jarrasDev = 0
    let objLev = 0
    let objDev = 0
    for (const ev of filteredEvents) {
      const isJar = ev.class_name === 'Jarra' || ev.class_name === 'Jarra Licuadora'
      if (ev.event_type === 'salida') {
        objLev += 1
        if (isJar) jarrasLev += 1
      } else {
        objDev += 1
        if (isJar) jarrasDev += 1
      }
    }
    return { jarrasLev, jarrasDev, objLev, objDev }
  }, [filteredEvents])

  const maxHourlyValue = useMemo(() => {
    const vals = filteredHourly.map((h) =>
      Math.max(h.objetos_levantados || 0, h.objetos_devueltos || 0)
    )
    return Math.max(1, ...vals)
  }, [filteredHourly])

  const activeCameraObj = cameras.find((c) => c.id === activeCamId)

  return (
    <div className="dashboard">
      <header className="topbar">
        <div>
          <h1>Nex Vision — Control Multi-Camara de Cocina</h1>
          <p className="subtitle">
            Monitoreo de jarras, utensilios e insumos en Cocina 1, Cocina 2 y Almacen (La Paz)
          </p>
        </div>
        <div className="status-badges">
          <span className={`badge ${apiOnline ? 'online' : 'offline'}`}>
            API: {apiOnline ? 'EN LINEA' : 'OFFLINE'}
          </span>
          <span className={`badge ${wsConnected ? 'online' : 'warn'}`}>
            WS: {wsConnected ? 'ACTIVO' : 'ESPERA'}
          </span>
          <span className={`badge ${liveStatus.activo ? 'online' : 'warn'}`}>
            CAMARA:{' '}
            {liveStatus.activo
              ? `${(CAMERA_LABELS[liveStatus.camera_id] || liveStatus.camera_id).toUpperCase()} (${liveStatus.fps || 0} fps)`
              : 'EN PAUSA'}
          </span>
        </div>
      </header>

      {/* SELECTOR RAPIDO DE CAMARAS (COCINA 1 / COCINA 2 / ALMACEN) */}
      <nav className="camera-tabs">
        <button
          className={`cam-tab ${cameraFilter === '' ? 'active' : ''}`}
          onClick={() => handleSelectCameraTab('')}
        >
          Todas las Camaras ({cameras.length})
        </button>
        {cameras.map((cam) => {
          const isLive = liveStatus.activo && liveStatus.camera_id === cam.id
          return (
            <button
              key={cam.id}
              className={`cam-tab ${cameraFilter === cam.id ? 'active' : ''}`}
              onClick={() => handleSelectCameraTab(cam.id)}
            >
              <span>{cam.nombre}</span>
              <span className={`tab-dot ${isLive ? 'live' : ''}`} />
            </button>
          )
        })}
      </nav>

      {/* CENTRO DE CONTROL: GRABACIONES + CONFIGURACION RTSP POR CAMARA */}
      <section className="panel control-center">
        <div className="control-grid">
          {/* Bloque 1: Grabaciones de Nex o subir video */}
          <div className="control-box">
            <h2>1. Revisar Grabacion de Camaras Nex</h2>
            <p className="helper-text">
              Selecciona una grabacion o sube un video nuevo. El sistema asigna automaticamente las zonas ROI de{' '}
              <code>{CAMERA_LABELS[activeCamId] || activeCamId}</code>.
            </p>

            <div className="upload-row">
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={handleUploadAndAnalyze}
              />
              <button
                className="btn-primary"
                disabled={uploadingVideo || jobStatus.activo}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                {uploadingVideo ? 'Subiendo video...' : 'Subir video de mi PC y Contar'}
              </button>
              <span className="or-divider">o elegir grabacion local:</span>
            </div>

            <div className="control-row">
              <select
                value={selectedVideo}
                onChange={(e) => handleVideoDropdownChange(e.target.value)}
              >
                {videosDetail.map((v) => (
                  <option key={v.path} value={v.path}>
                    [{CAMERA_LABELS[v.camera_id] || v.camera_id}] {v.name}
                  </option>
                ))}
              </select>
              <button
                className="btn-secondary"
                disabled={jobStatus.activo || uploadingVideo || !selectedVideo}
                onClick={() => startBackgroundAnalysis(selectedVideo)}
              >
                {jobStatus.activo ? 'Analizando...' : 'Contar Video'}
              </button>
              <button
                className="btn-outline"
                disabled={!selectedVideo}
                onClick={() => {
                  const info = videosDetail.find((v) => v.path === selectedVideo)
                  handleStartLive(selectedVideo, info ? info.camera_id : activeCamId)
                }}
              >
                Reproducir con Marcas
              </button>
            </div>
          </div>

          {/* Bloque 2: Cambio de camara y configuracion RTSP lista para el local */}
          <div className="control-box">
            <h2>2. Camaras en Tiempo Real (Listo para RTSP en Local)</h2>
            <p className="helper-text">
              Elige la camara de Nex. Cuando instales el sistema en el local, pega su enlace <code>rtsp://...</code> y presiona Guardar RTSP.
            </p>

            <div className="control-row" style={{ marginBottom: '0.55rem' }}>
              <select
                value={activeCamId}
                onChange={(e) => setActiveCamId(e.target.value)}
              >
                {cameras.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.nombre} ({c.zona_roi})
                  </option>
                ))}
              </select>
              <button
                className="btn-outline"
                onClick={() => {
                  const cam = cameras.find((c) => c.id === activeCamId)
                  const src = (cam && cam.default_video) || selectedVideo
                  handleStartLive(src, activeCamId)
                }}
              >
                Probar Vista {CAMERA_LABELS[activeCamId] || activeCamId}
              </button>
            </div>

            <div className="control-row">
              <input
                type="text"
                value={rtspDraft}
                onChange={(e) => setRtspDraft(e.target.value)}
                placeholder="rtsp://usuario:clave@IP_DVR:554/stream (o 0 para USB)"
              />
              <button className="btn-secondary" onClick={handleSaveRtsp}>
                Guardar RTSP
              </button>
              {!liveStatus.activo ? (
                <button
                  className="btn-primary"
                  onClick={() => handleStartLive(rtspDraft, activeCamId)}
                >
                  Conectar en Vivo
                </button>
              ) : (
                <button className="btn-danger" onClick={handleStopLive}>
                  Detener Vista
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Barra de progreso del analisis de video largo */}
        {jobStatus.activo && (
          <div className="progress-panel">
            <div className="progress-header">
              <strong>
                Analizando [{CAMERA_LABELS[jobStatus.camera_id] || jobStatus.camera_id}]{' '}
                {jobStatus.fuente.replace('videos/', '')} — {jobStatus.progreso_pct}%
              </strong>
              <span>
                Eventos detectados: <strong>{jobStatus.eventos_encontrados}</strong> (Jarras:{' '}
                <strong>{jobStatus.jarras_encontradas}</strong>)
              </span>
              <button className="btn-danger" onClick={handleCancelJob}>
                Detener y Guardar
              </button>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.max(3, jobStatus.progreso_pct)}%` }}
              />
            </div>
          </div>
        )}

        {actionMsg && <div className="action-banner">{actionMsg}</div>}

        {liveStatus.error && (
          <div className="action-banner" style={{ borderColor: '#f93e3e', color: '#b41d1d' }}>
            Error en fuente de video: {liveStatus.error}
          </div>
        )}

        {/* Resumen inmediato al terminar un analisis */}
        {lastReport && !jobStatus.activo && (
          <div className="report-summary-box">
            <h3>
              Ultimo analisis completado [{CAMERA_LABELS[lastReport.camera_id] || lastReport.camera_id}]:{' '}
              {lastReport.fuente.replace('videos/', '')}
            </h3>
            <p>
              Duracion analizada: <strong>{lastReport.duracion_video_s}s</strong> | Jarras levantadas:{' '}
              <strong>{lastReport.jarras_levantadas}</strong> | Total salidas de zona:{' '}
              <strong>{lastReport.objetos_levantados_total}</strong>
            </p>
            {lastReport.conteo_por_clase && Object.keys(lastReport.conteo_por_clase).length > 0 && (
              <div className="report-tags">
                {Object.entries(lastReport.conteo_por_clase).map(([cls, c]) => (
                  <span key={cls} className="report-tag">
                    {cls}: {c.levantados} lev. / {c.devueltos} dev.
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Reproductor MJPEG en vivo con selector de camara y modo visual limpio */}
        {liveStatus.activo && (
          <div className="live-stream-box">
            <div className="live-stream-header">
              <div>
                <strong>
                  Vista en Directo [{CAMERA_LABELS[liveStatus.camera_id] || liveStatus.camera_id}] —{' '}
                  {liveStatus.fuente.replace('videos/', '')} ({liveStatus.resolucion})
                </strong>
                <div className="live-controls-bar">
                  <div className="live-quick-switch">
                    <span>Camara:</span>
                    {cameras.map((c) => (
                      <button
                        key={c.id}
                        className={`btn-mini ${liveStatus.camera_id === c.id ? 'active' : ''}`}
                        onClick={() => handleStartLive(c.rtsp_url || c.default_video, c.id)}
                      >
                        {CAMERA_LABELS[c.id] || c.nombre}
                      </button>
                    ))}
                  </div>
                  <div className="live-quick-switch">
                    <span>Modo visual:</span>
                    <button
                      className={`btn-mini ${viewMode === 'limpia' ? 'active-blue' : ''}`}
                      title="Oculta cajas amontonadas, resalta Jarras y eventos recientes"
                      onClick={() => handleDisplayChange('limpia', showZones)}
                    >
                      Vista Limpia
                    </button>
                    <button
                      className={`btn-mini ${viewMode === 'solo_jarras' ? 'active-blue' : ''}`}
                      title="Muestra unicamente Jarras y el objeto que acaba de moverse"
                      onClick={() => handleDisplayChange('solo_jarras', showZones)}
                    >
                      Solo Jarras
                    </button>
                    <button
                      className={`btn-mini ${viewMode === 'tecnica' ? 'active-blue' : ''}`}
                      title="Muestra todas las detecciones y porcentajes (modo tecnico)"
                      onClick={() => handleDisplayChange('tecnica', showZones)}
                    >
                      Vista Tecnica
                    </button>
                    <button
                      className={`btn-mini ${showZones ? 'active' : ''}`}
                      onClick={() => handleDisplayChange(viewMode, !showZones)}
                    >
                      Zonas ROI: {showZones ? 'SI' : 'NO'}
                    </button>
                  </div>
                </div>
              </div>
              <button className="btn-danger" onClick={handleStopLive}>
                Cerrar Vista
              </button>
            </div>
            <img
              src={`${API_BASE}/api/live/stream?session=${streamKey}`}
              alt="Vista de camara de cocina"
              className="live-video-img"
            />
          </div>
        )}
      </section>

      {/* FILTROS DE CONSULTA */}
      <section className="filters-bar">
        <div className="filter-group">
          <label>Camara</label>
          <select value={cameraFilter} onChange={(e) => handleSelectCameraTab(e.target.value)}>
            <option value="">Todas las camaras</option>
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.nombre}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Fecha</label>
          <input
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
          {dateFilter && (
            <button className="btn-clear" onClick={() => setDateFilter('')}>
              Mostrar todas
            </button>
          )}
        </div>

        <div className="filter-group">
          <label>Turno</label>
          <select value={shiftFilter} onChange={(e) => setShiftFilter(e.target.value)}>
            {Object.entries(SHIFTS).map(([key, s]) => (
              <option key={key} value={key}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Objeto</label>
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
            <option value="">Todos los objetos</option>
            {availableClasses.map((cls) => (
              <option key={cls} value={cls}>
                {cls}
              </option>
            ))}
          </select>
        </div>

        <button className="btn-secondary" onClick={fetchData}>
          Refrescar
        </button>
      </section>

      {/* TARJETAS KPI */}
      <section className="kpi-grid">
        <div className="kpi-card highlight">
          <span className="kpi-title">Jarras Levantadas</span>
          <strong className="kpi-value">{kpis.jarrasLev}</strong>
          <span className="kpi-sub">Devueltas a zona: {kpis.jarrasDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Total Objetos Levantados</span>
          <strong className="kpi-value">{kpis.objLev}</strong>
          <span className="kpi-sub">Devueltos a zona: {kpis.objDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Movimientos en Bitacora</span>
          <strong className="kpi-value">{filteredEvents.length}</strong>
          <span className="kpi-sub">
            Filtro: {cameraFilter ? CAMERA_LABELS[cameraFilter] : 'Todas las camaras'}
          </span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Camaras Configuradas</span>
          <strong className="kpi-value">{cameras.length}</strong>
          <span className="kpi-sub">
            {activeCameraObj ? activeCameraObj.zona_roi : 'Cocina 1, Cocina 2, Almacen'}
          </span>
        </div>
      </section>

      <div className="main-grid">
        {/* GRAFICO POR HORA */}
        <section className="panel chart-panel">
          <div className="panel-header">
            <h2>Actividad por Hora (Salidas vs Entradas en Zonas ROI)</h2>
            <div className="legend">
              <span className="dot lifted-dot" /> Levantado (salida)
              <span className="dot returned-dot" /> Devuelto (entrada)
            </div>
          </div>

          <div className="bar-chart">
            {filteredHourly.map((item) => {
              const levPct = Math.round(((item.objetos_levantados || 0) / maxHourlyValue) * 100)
              const devPct = Math.round(((item.objetos_devueltos || 0) / maxHourlyValue) * 100)
              const hasActivity = item.objetos_levantados > 0 || item.objetos_devueltos > 0
              return (
                <div
                  key={item.hora}
                  className={`bar-col ${hasActivity ? 'active-hour' : ''}`}
                  title={`${item.hora} -> Levantados: ${item.objetos_levantados}, Devueltos: ${item.objetos_devueltos}`}
                >
                  <div className="bars-wrap">
                    <div className="bar bar-lifted" style={{ height: `${levPct}%` }}>
                      {item.objetos_levantados > 0 && <span>{item.objetos_levantados}</span>}
                    </div>
                    <div className="bar bar-returned" style={{ height: `${devPct}%` }}>
                      {item.objetos_devueltos > 0 && <span>{item.objetos_devueltos}</span>}
                    </div>
                  </div>
                  <span className="bar-label">{item.hora.slice(0, 2)}h</span>
                </div>
              )
            })}
          </div>

          {/* Resumen rapido por zona ROI debajo del grafico */}
          {Object.keys(totalsByZone).length > 0 && (
            <div className="zones-summary">
              <h3>Actividad por Zona ROI</h3>
              <div className="report-tags">
                {Object.entries(totalsByZone).map(([zName, c]) => (
                  <span key={zName} className="zone-chip">
                    {zName}: <strong>{c.levantados}</strong> sal. / <strong>{c.devueltos}</strong> ent.
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* CONTEO POR OBJETO Y FICHAS DE LAS 3 CAMARAS */}
        <aside className="side-panels">
          <section className="panel">
            <h2>Resumen por Tipo de Objeto</h2>
            {Object.keys(totalsByClass).length === 0 ? (
              <p className="empty-msg">Sin registros para los filtros actuales.</p>
            ) : (
              <ul className="class-list">
                {Object.entries(totalsByClass)
                  .sort((a, b) => b[1].levantados - a[1].levantados)
                  .map(([clsName, counts]) => (
                    <li key={clsName} className="class-item">
                      <span className="class-name">{clsName}</span>
                      <div className="class-badges">
                        <span className="pill lifted-pill">{counts.levantados} sal.</span>
                        <span className="pill returned-pill">{counts.devueltos} ent.</span>
                      </div>
                    </li>
                  ))}
              </ul>
            )}
          </section>

          <section className="panel">
            <h2>Camaras de Nex ({cameras.length})</h2>
            <div className="cameras-stack">
              {cameras.map((cam) => (
                <div
                  key={cam.id}
                  className={`camera-card ${activeCamId === cam.id ? 'selected-cam' : ''}`}
                >
                  <div className="camera-header">
                    <strong>{cam.nombre}</strong>
                    <span
                      className={`cam-status ${
                        cam.estado === 'en_vivo'
                          ? 'live'
                          : cam.estado === 'rtsp_listo'
                          ? 'active'
                          : 'video'
                      }`}
                    >
                      {cam.estado === 'en_vivo'
                        ? 'EN VIVO'
                        : cam.estado === 'rtsp_listo'
                        ? 'RTSP LISTO'
                        : 'MODO VIDEO'}
                    </span>
                  </div>
                  <p>
                    <strong>Zonas:</strong> {cam.zona_roi}
                  </p>
                  <p>
                    <strong>RTSP:</strong>{' '}
                    {cam.rtsp_url ? <code>{cam.rtsp_url}</code> : <span className="muted-sm">Pendiente (se configura en local)</span>}
                  </p>
                  <p>
                    <strong>Eventos en BD:</strong> {cam.eventos_registrados}
                  </p>
                  <div className="cam-card-actions">
                    <button
                      className="btn-mini"
                      onClick={() => handleSelectCameraTab(cam.id)}
                    >
                      Filtrar
                    </button>
                    <button
                      className="btn-mini active"
                      onClick={() => handleStartLive(cam.rtsp_url || cam.default_video, cam.id)}
                    >
                      Ver con Marcas
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>

      {/* TABLA DE BITACORA */}
      <section className="panel table-panel">
        <h2>Detalle de Movimientos Registrados ({filteredEvents.length})</h2>
        {loading ? (
          <p className="empty-msg">Cargando registros...</p>
        ) : filteredEvents.length === 0 ? (
          <p className="empty-msg">Sin movimientos para los filtros seleccionados.</p>
        ) : (
          <div className="table-responsive">
            <table className="events-table">
              <thead>
                <tr>
                  <th>Nro</th>
                  <th>Camara</th>
                  <th>Objeto</th>
                  <th>Track</th>
                  <th>Movimiento</th>
                  <th>Zona ROI</th>
                  <th>Momento</th>
                  <th>Evidencia</th>
                  <th>Fecha y Hora</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.slice(0, 150).map((ev) => (
                  <tr key={ev.id}>
                    <td>#{ev.id}</td>
                    <td>
                      <span className="cam-chip">
                        {CAMERA_LABELS[ev.camera_id] || ev.camera_id}
                      </span>
                    </td>
                    <td>
                      <strong>{ev.class_name}</strong>
                    </td>
                    <td>#{ev.track_id}</td>
                    <td>
                      <span
                        className={`action-tag ${
                          ev.event_type === 'salida' ? 'tag-lifted' : 'tag-returned'
                        }`}
                      >
                        {ev.event_type === 'salida' ? 'LEVANTADO (Salida)' : 'DEVUELTO (Entrada)'}
                      </span>
                    </td>
                    <td>{ev.zone}</td>
                    <td>
                      {ev.time_s}s (f.{ev.frame})
                    </td>
                    <td>
                      {ev.snapshot_url ? (
                        <button
                          className="btn-snapshot"
                          onClick={() => setModalEvent(ev)}
                        >
                          Ver Foto
                        </button>
                      ) : (
                        <span className="muted-sm">-</span>
                      )}
                    </td>
                    <td>
                      {ev.created_at
                        ? new Date(ev.created_at).toLocaleString('es-BO')
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* VISOR MODAL DE FOTO DE EVIDENCIA (SNAPSHOT) */}
      {modalEvent && (
        <div className="modal-backdrop" onClick={() => setModalEvent(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3>
                  Evidencia #{modalEvent.id} — {modalEvent.class_name} (#{modalEvent.track_id})
                </h3>
                <p className="subtitle">
                  {CAMERA_LABELS[modalEvent.camera_id] || modalEvent.camera_id} ·{' '}
                  {modalEvent.event_type === 'salida' ? 'LEVANTADO (Salida)' : 'DEVUELTO (Entrada)'} · Zona:{' '}
                  <strong>{modalEvent.zone}</strong> · Segundo {modalEvent.time_s}s
                </p>
              </div>
              <button className="btn-danger" onClick={() => setModalEvent(null)}>
                Cerrar
              </button>
            </div>
            <img
              src={`${API_BASE}${modalEvent.snapshot_url}`}
              alt={`Captura del evento ${modalEvent.id}`}
              className="modal-snapshot-img"
            />
          </div>
        </div>
      )}
    </div>
  )
}
