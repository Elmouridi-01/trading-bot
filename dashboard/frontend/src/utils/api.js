const BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

const get = async (path) => {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export const api = {
  overview:    ()      => get('/api/overview'),
  signals:     (p)     => get('/api/signals?'     + new URLSearchParams(p)),
  orders:      (p)     => get('/api/orders?'      + new URLSearchParams(p)),
  performance: (period)=> get(`/api/performance?period=${period || 'all'}`),
  equityCurve: ()      => get('/api/equity-curve'),
}

export const fmt     = (n, d=2)  => (n == null || isNaN(n)) ? '—' : Number(n).toFixed(d)
export const fmtUSD  = (n)       => n == null ? '—' : (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2)
export const fmtPct  = (n)       => n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(2) + '%'
export const fmtTime = (ts)      => ts ? new Date(ts).toLocaleTimeString('en-US', { hour12: false }) : '—'
export const fmtDate = (ts)      => ts ? new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '—'

export const regimeColor = (r) =>
  !r                    ? '#6b7fa3' :
  r.includes('up')      ? '#00ff9d' :
  r.includes('down')    ? '#ff3366' :
  r.includes('volatile')? '#ffd166' : '#00d4ff'

export const regimeLabel = (r) => r ? r.toUpperCase().replace('_', ' ') : 'UNKNOWN'