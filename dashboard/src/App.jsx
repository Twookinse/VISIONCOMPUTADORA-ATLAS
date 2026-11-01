import React, { useEffect, useState, useCallback, useMemo } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/events'

const SHIFTS = {
  todos: { label: 'Todos los turnos (24h)', check: () => true },
  dia: { label: 'Turno Dia (08:00 - 15:59)', check: (h) => h >= 8 && h < 16 },
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
  const [wsConnected, setWsConnected] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [loading, setLoading] = useState(true)

  // Filtros (Tarea 5.5)
  const [dateFilter, setDateFilter] = useState('')
  const [shiftFilter, setShiftFilter] = useState('todos')
  const [classFilter, setClassFilter] = useState('')

  const fetchData = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (dateFilter) params.set('date', dateFilter)
      if (classFilter) params.set('class_name', classFilter)

      const [evRes, hrRes, camRes] = await Promise.all([
        fetch(`${API_BASE}/api/events?${params.toString()}`),
        fetch(`${API_BASE}/api/metrics/hourly?${params.toString()}`),
        fetch(`${API_BASE}/api/cameras`),
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
    } catch (err) {
      setApiOnline(false)
    } finally {
      setLoading(false)
    }
  }, [dateFilter, classFilter])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Conexion WebSocket en vivo (Tarea 5.7)
  useEffect(() => {
    let ws
    let reconnectTimer

    const connectWs = () => {
      try {
        ws = new WebSocket(WS_URL)
        ws.onopen = () => setWsConnected(true)
        ws.onmessage = () => {
          fetchData()
        }
        ws.onclose = () => {
          setWsConnected(false)
          reconnectTimer = setTimeout(connectWs, 4000)
        }
        ws.onerror = () => {
          setWsConnected(false)
        }
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

  // Filtrar horas y eventos segun el turno seleccionado
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
          <h1>Nex Vision AI — Panel de Cocina</h1>
          <p className="subtitle">
            Monitoreo automatico de jarras y utensilios en tiempo real (La Paz)
          </p>
        </div>
        <div className="status-badges">
          <span className={`badge ${apiOnline ? 'online' : 'offline'}`}>
            API: {apiOnline ? 'Conectada' : 'Sin conexion'}
          </span>
          <span className={`badge ${wsConnected ? 'online' : 'warn'}`}>
            WebSocket: {wsConnected ? 'En vivo' : 'Espera'}
          </span>
        </div>
      </header>

      {/* Filtros (Tarea 5.5) */}
      <section className="filters-bar">
        <div className="filter-group">
          <label>Fecha</label>
          <input
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
          {dateFilter && (
            <button className="btn-clear" onClick={() => setDateFilter('')}>
              Ver todas
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
          <label>Tipo de Objeto</label>
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
            <option value="">Todos los objetos</option>
            {KNOWN_CLASSES.map((cls) => (
              <option key={cls} value={cls}>
                {cls}
              </option>
            ))}
          </select>
        </div>

        <button className="btn-refresh" onClick={fetchData}>
          Actualizar
        </button>
      </section>

      {/* KPIs principales */}
      <section className="kpi-grid">
        <div className="kpi-card highlight">
          <span className="kpi-title">Jarras Levantadas</span>
          <strong className="kpi-value">{kpis.jarrasLev}</strong>
          <span className="kpi-sub">Devueltas a mesa: {kpis.jarrasDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Total Objetos Levantados</span>
          <strong className="kpi-value">{kpis.objLev}</strong>
          <span className="kpi-sub">Devueltos a mesa: {kpis.objDev}</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Eventos Registrados</span>
          <strong className="kpi-value">{filteredEvents.length}</strong>
          <span className="kpi-sub">Segun filtros activos</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-title">Clases Activas</span>
          <strong className="kpi-value">{Object.keys(totalsByClass).length}</strong>
          <span className="kpi-sub">Modelo custom v1 (mAP 96.1%)</span>
        </div>
      </section>

      <div className="main-grid">
        {/* Grafico de barras por hora (Tarea 5.3) */}
        <section className="panel chart-panel">
          <div className="panel-header">
            <h2>Actividad por Hora (Objetos Levantados vs Devueltos)</h2>
            <div className="legend">
              <span className="dot lifted-dot" /> Levantados (salida)
              <span className="dot returned-dot" /> Devueltos (entrada)
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

        {/* Desglose por objeto y Estado de Camaras (Tarea 5.6) */}
        <aside className="side-panels">
          <section className="panel">
            <h2>Conteo por Objeto de Cocina</h2>
            {Object.keys(totalsByClass).length === 0 ? (
              <p className="empty-msg">Sin eventos para los filtros seleccionados.</p>
            ) : (
              <ul className="class-list">
                {Object.entries(totalsByClass).map(([clsName, counts]) => (
                  <li key={clsName} className="class-item">
                    <span className="class-name">{clsName}</span>
                    <div className="class-badges">
                      <span className="pill lifted-pill">
                        {counts.levantados} lev.
                      </span>
                      <span className="pill returned-pill">
                        {counts.devueltos} dev.
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="panel">
            <h2>Estado de Camaras</h2>
            {cameras.length === 0 ? (
              <p className="empty-msg">Verificando estado de camaras...</p>
            ) : (
              cameras.map((cam) => (
                <div key={cam.id} className="camera-card">
                  <div className="camera-header">
                    <strong>{cam.nombre}</strong>
                    <span className={`cam-status ${cam.estado === 'activa' ? 'active' : 'inactive'}`}>
                      {cam.estado.toUpperCase()}
                    </span>
                  </div>
                  <p><strong>Zona ROI:</strong> {cam.zona_roi}</p>
                  <p><strong>Modelo IA:</strong> {cam.modelo}</p>
                  <p><strong>Fuente:</strong> {cam.fuente_actual}</p>
                </div>
              ))
            )}
          </section>
        </aside>
      </div>

      {/* Tabla de eventos recientes (Tarea 5.4) */}
      <section className="panel table-panel">
        <h2>Eventos Recientes en Cocina ({filteredEvents.length})</h2>
        {loading ? (
          <p className="empty-msg">Cargando eventos desde la API...</p>
        ) : filteredEvents.length === 0 ? (
          <p className="empty-msg">No hay eventos registrados con los filtros actuales.</p>
        ) : (
          <div className="table-responsive">
            <table className="events-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Objeto</th>
                  <th>Track ID</th>
                  <th>Accion</th>
                  <th>Zona ROI</th>
                  <th>Segundo Video</th>
                  <th>Fecha / Hora Registro</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.map((ev) => (
                  <tr key={ev.id}>
                    <td>#{ev.id}</td>
                    <td>
                      <strong>{ev.class_name}</strong>
                    </td>
                    <td>Obj #{ev.track_id}</td>
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
                    <td>{ev.time_s}s (frame {ev.frame})</td>
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
