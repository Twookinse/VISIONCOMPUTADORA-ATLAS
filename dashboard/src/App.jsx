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

const KNOWN_CLASSES = [
  'Jarra',
  'Vaso',
  'Plato',
  'Olla',
  'Cuchillo',
  'Tijera',
  'Pinia',
  'Envase de cafe',
  'Envase de sal',
  'Trapo',
  'Papel Higienico',
]

export default function App() {
  const [events, setEvents] = useState([])
  const [hourly, setHourly] = useState([])
  const [totalsByClass, setTotalsByClass] = useState({})
  const [cameras, setCameras] = useState([])
  const [availableVideos, setAvailableVideos] = useState([])
  const [liveStatus, setLiveStatus] = useState({ activo: false })
  const [wsConnected, setWsConnected] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [loading, setLoading] = useState(true)

  // Controles de camara y carga de videos
  const [liveSource, setLiveSource] = useState('0')
  const [selectedVideo, setSelectedVideo] = useState('videos/VIDEO 2 VISION COMPUTADORA.mp4')
  const [processingVideo, setProcessingVideo] = useState(false)
  const [uploadingVideo, setUploadingVideo] = useState(false)
  const [actionMsg, setActionMsg] = useState('')
  const [lastReport, setLastReport] = useState(null)
  const [streamKey, setStreamKey] = useState(1)
  const fileInputRef = useRef(null)

  // Filtros
  const [dateFilter, setDateFilter] = useState('')
  const [shiftFilter, setShiftFilter] = useState('todos')
  const [classFilter, setClassFilter] = useState('')

  const fetchData = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (dateFilter) params.set('date', dateFilter)
      if (classFilter) params.set('class_name', classFilter)

      const [evRes, hrRes, camRes, vidRes, liveRes] = await Promise.all([
        fetch(`${API_BASE}/api/events?${params.toString()}`),
        fetch(`${API_BASE}/api/metrics/hourly?${params.toString()}`),
        fetch(`${API_BASE}/api/cameras`),
        fetch(`${API_BASE}/api/videos`),
        fetch(`${API_BASE}/api/live/status`),
      ])

      if (evRes.ok && hrRes.ok) {
        const evData = await evRes.json()
        const hrData = await hrRes.json()
        setEvents(evData.eventos || [])
        setHourly(hrData.por_hora || [])
        setTotalsByClass(hrData.totales_por_clase || {})
        setApiOnline(true)
      }
      if (camRes.ok) {
        const camData = await camRes.json()
        setCameras(camData.camaras || [])
      }
      if (vidRes.ok) {
        const vidData = await vidRes.json()
        const vids = vidData.videos || []
        setAvailableVideos(vids)
        if (vids.length > 0 && !selectedVideo) {
          setSelectedVideo(vids[0])
        }
      }
      if (liveRes.ok) {
        const lData = await liveRes.json()
        setLiveStatus(lData)
      }
    } catch (err) {
      setApiOnline(false)
    } finally {
      setLoading(false)
    }
  }, [dateFilter, classFilter, selectedVideo])

  useEffect(() => {
    fetchData()
    const timer = setInterval(fetchData, 5000)
    return () => clearInterval(timer)
  }, [fetchData])

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

  const handleStartLive = async (sourceToUse) => {
    setActionMsg(`Abriendo reproduccion con marcas: "${sourceToUse}"...`)
    setLiveStatus({ activo: true, fuente: sourceToUse, resolucion: 'Cargando...' })
    setStreamKey(Date.now())
    try {
      const res = await fetch(`${API_BASE}/api/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceToUse }),
      })
      const data = await res.json()
      setLiveStatus(data.estado || { activo: true, fuente: sourceToUse })
      setActionMsg(`Reproduciendo con marcas en vivo: ${sourceToUse}`)
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
      setActionMsg('Transmision detenida.')
      fetchData()
    } catch (e) {
      setActionMsg('Error al detener la transmision.')
    }
  }

  const runVideoAnalysis = async (videoPath) => {
    setProcessingVideo(true)
    setLastReport(null)
    setActionMsg(`Analizando grabacion "${videoPath}"... Esto puede tardar cerca de 1 minuto.`)
    try {
      const res = await fetch(`${API_BASE}/api/pipeline/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: videoPath }),
      })
      if (res.ok) {
        const data = await res.json()
        setLastReport(data.reporte)
        setActionMsg(
          `Revision completada: Se detectaron ${data.reporte.jarras_levantadas} jarras levantadas y ${data.reporte.objetos_levantados_total} movimientos totales.`
        )
        fetchData()
      } else {
        setActionMsg('No se pudo analizar el video seleccionado.')
      }
    } catch (e) {
      setActionMsg('Error de comunicacion con el servidor.')
    } finally {
      setProcessingVideo(false)
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
      setSelectedVideo(newVideoPath)
      setUploadingVideo(false)

      // Una vez subido, lo analiza automaticamente
      await runVideoAnalysis(newVideoPath)
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
      if (ev.event_type === 'salida') {
        objLev += 1
        if (ev.class_name === 'Jarra') jarrasLev += 1
      } else {
        objDev += 1
        if (ev.class_name === 'Jarra') jarrasDev += 1
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

  return (
    <div className="dashboard">
      <header className="topbar">
        <div>
          <h1>Bitacora de Cocina — Karaoke Nex</h1>
          <p className="subtitle">
            Registro de salida de jarras, vasos y platos en mesa de despacho (La Paz)
          </p>
        </div>
        <div className="status-badges">
          <span className={`badge ${apiOnline ? 'online' : 'offline'}`}>
            Sistema: {apiOnline ? 'En linea' : 'Desconectado'}
          </span>
          <span className={`badge ${wsConnected ? 'online' : 'warn'}`}>
            Sincronizacion: {wsConnected ? 'Automatica' : 'Manual'}
          </span>
          <span className={`badge ${liveStatus.activo ? 'online' : 'warn'}`}>
            Camara: {liveStatus.activo ? `Transmitiendo (${liveStatus.fps || 0} fps)` : 'En pausa'}
          </span>
        </div>
      </header>

      {/* SECCION DE OPERACION: SUBIR VIDEO O CONECTAR CAMARA */}
      <section className="panel control-center">
        <div className="control-grid">
          {/* Bloque 1: Revisar grabacion (subir video o elegir existente) */}
          <div className="control-box">
            <h2>Revisar Grabacion de Cocina (Ej. Video de ayer)</h2>
            <p className="helper-text">
              Puedes subir cualquier video desde tu computadora para contar cuantas jarras y objetos se levantaron, o elegir uno ya guardado.
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
                disabled={uploadingVideo || processingVideo}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                {uploadingVideo
                  ? 'Subiendo video...'
                  : 'Subir video de mi PC y Contar'}
              </button>
              <span className="or-divider">o elegir guardado:</span>
            </div>

            <div className="control-row">
              <select
                value={selectedVideo}
                onChange={(e) => setSelectedVideo(e.target.value)}
              >
                {availableVideos.map((v) => (
                  <option key={v} value={v}>
                    {v.replace('videos/', '')}
                  </option>
                ))}
              </select>
              <button
                className="btn-secondary"
                disabled={processingVideo || uploadingVideo}
                onClick={() => runVideoAnalysis(selectedVideo)}
              >
                {processingVideo ? 'Contando...' : 'Contar Video'}
              </button>
              <button
                className="btn-outline"
                onClick={() => handleStartLive(selectedVideo)}
              >
                Reproducir con Marcas
              </button>
            </div>
          </div>

          {/* Bloque 2: Camara en tiempo real */}
          <div className="control-box">
            <h2>Camara de Cocina en Tiempo Real (720p)</h2>
            <p className="helper-text">
              Para conectar la camara del local en vivo escribe su direccion RTSP (720p @ 1000 KB/s) o <code>0</code> para camara USB.
            </p>
            <div className="control-row">
              <input
                type="text"
                value={liveSource}
                onChange={(e) => setLiveSource(e.target.value)}
                placeholder="0 o rtsp://usuario:clave@192.168.1.50:554/stream"
              />
              {!liveStatus.activo ? (
                <button
                  className="btn-primary"
                  onClick={() => handleStartLive(liveSource)}
                >
                  Encender Vista en Vivo
                </button>
              ) : (
                <button className="btn-danger" onClick={handleStopLive}>
                  Detener Camara
                </button>
              )}
            </div>
          </div>
        </div>

        {actionMsg && <div className="action-banner">{actionMsg}</div>}

        {/* Resumen inmediato cuando el dueno termina de analizar un video */}
        {lastReport && (
          <div className="report-summary-box">
            <h3>Resultado del video analizado: {lastReport.fuente.replace('videos/', '')}</h3>
            <p>
              Duracion: <strong>{lastReport.duracion_video_s}s</strong> | Jarras levantadas:{' '}
              <strong>{lastReport.jarras_levantadas}</strong> | Jarras devueltas:{' '}
              <strong>{lastReport.jarras_devueltas}</strong> | Total movimientos en mesa:{' '}
              <strong>{lastReport.objetos_levantados_total} salidas</strong>
            </p>
            {lastReport.conteo_por_clase && (
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

        {liveStatus.error && (
          <div className="action-banner" style={{ borderColor: '#f93e3e', color: '#b41d1d' }}>
            Error en fuente de video: {liveStatus.error}
          </div>
        )}

        {liveStatus.activo && (
          <div className="live-stream-box">
            <div className="live-stream-header">
              <strong>
                Vista en Directo — {liveStatus.fuente} ({liveStatus.resolucion || 'Cargando...'}) |{' '}
                Jarras Lev: {liveStatus.jarras_levantadas_sesion || 0} | Total Obj Lev:{' '}
                {liveStatus.objetos_levantados_sesion || 0}
              </strong>
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
          <label>Filtrar por Fecha</label>
          <input
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
          {dateFilter && (
            <button className="btn-clear" onClick={() => setDateFilter('')}>
              Mostrar todas las fechas
            </button>
          )}
        </div>

        <div className="filter-group">
          <label>Horario / Turno</label>
          <select value={shiftFilter} onChange={(e) => setShiftFilter(e.target.value)}>
            {Object.entries(SHIFTS).map(([key, s]) => (
              <option key={key} value={key}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Utensilio / Objeto</label>
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
            <option value="">Todos los registrados</option>
            {KNOWN_CLASSES.map((cls) => (
              <option key={cls} value={cls}>
                {cls}
              </option>
            ))}
          </select>
        </div>

        <button className="btn-secondary" onClick={fetchData}>
          Refrescar Tabla
        </button>
      </section>

      {/* TARJETAS DE RESUMEN */}
      <section className="kpi-grid">
        <div className="kpi-card highlight">
          <span className="kpi-title">Jarras Levantadas</span>
          <strong className="kpi-value">{kpis.jarrasLev}</strong>
          <span className="kpi-sub">Puestas de vuelta en mesa: {kpis.jarrasDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Total Utensilios Levantados</span>
          <strong className="kpi-value">{kpis.objLev}</strong>
          <span className="kpi-sub">Puestos en mesa: {kpis.objDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Movimientos en Bitacora</span>
          <strong className="kpi-value">{filteredEvents.length}</strong>
          <span className="kpi-sub">Con los filtros seleccionados</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Tipos de Objetos Vistos</span>
          <strong className="kpi-value">{Object.keys(totalsByClass).length}</strong>
          <span className="kpi-sub">Reconocimiento propio de cocina</span>
        </div>
      </section>

      <div className="main-grid">
        {/* GRAFICO POR HORA */}
        <section className="panel chart-panel">
          <div className="panel-header">
            <h2>Movimiento por Hora en Mesa de Cocina</h2>
            <div className="legend">
              <span className="dot lifted-dot" /> Salida de mesa (levantado)
              <span className="dot returned-dot" /> Entrada a mesa (devuelto)
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
        </section>

        {/* CONTEO POR OBJETO Y CAMARAS */}
        <aside className="side-panels">
          <section className="panel">
            <h2>Resumen por Tipo de Objeto</h2>
            {Object.keys(totalsByClass).length === 0 ? (
              <p className="empty-msg">No hay registros para este filtro.</p>
            ) : (
              <ul className="class-list">
                {Object.entries(totalsByClass).map(([clsName, counts]) => (
                  <li key={clsName} className="class-item">
                    <span className="class-name">{clsName}</span>
                    <div className="class-badges">
                      <span className="pill lifted-pill">
                        {counts.levantados} salidas
                      </span>
                      <span className="pill returned-pill">
                        {counts.devueltos} entradas
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="panel">
            <h2>Ficha de Camara</h2>
            {cameras.length === 0 ? (
              <p className="empty-msg">Consultando camara...</p>
            ) : (
              cameras.map((cam) => (
                <div key={cam.id} className="camera-card">
                  <div className="camera-header">
                    <strong>{cam.nombre}</strong>
                    <span className="cam-status active">
                      {cam.estado.toUpperCase()}
                    </span>
                  </div>
                  <p><strong>Formato:</strong> {cam.resolucion}</p>
                  <p><strong>Zona vigilada:</strong> {cam.zona_roi}</p>
                  <p><strong>Archivo / Stream:</strong> {cam.fuente_actual}</p>
                </div>
              ))
            )}
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
                  <th>Objeto</th>
                  <th>Identificador</th>
                  <th>Movimiento</th>
                  <th>Zona</th>
                  <th>Momento en Video</th>
                  <th>Fecha y Hora</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.map((ev) => (
                  <tr key={ev.id}>
                    <td>#{ev.id}</td>
                    <td>
                      <strong>{ev.class_name}</strong>
                    </td>
                    <td>Pieza #{ev.track_id}</td>
                    <td>
                      <span
                        className={`action-tag ${
                          ev.event_type === 'salida' ? 'tag-lifted' : 'tag-returned'
                        }`}
                      >
                        {ev.event_type === 'salida' ? 'Levantado de mesa' : 'Puesto en mesa'}
                      </span>
                    </td>
                    <td>{ev.zone}</td>
                    <td>{ev.time_s} seg (cuadro {ev.frame})</td>
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
    </div>
  )
}
