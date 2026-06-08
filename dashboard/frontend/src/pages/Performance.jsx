import { useState, useEffect } from 'react'
import { api, fmt, fmtUSD } from '../utils/api'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, ReferenceLine
} from 'recharts'
import clsx from 'clsx'

const PERIODS = [
  { id: 'week',  label: '7 Days'   },
  { id: 'month', label: '30 Days'  },
  { id: 'all',   label: 'All Time' },
]

export default function Performance() {
  const [period,  setPeriod]  = useState('all')
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.performance(period)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [period])

  const m      = data?.metrics    || {}
  const curve  = data?.equity_curve || []
  const byS    = data?.by_strategy  || {}
  const trades = data?.trades       || []

  const pnlHist = (() => {
    const pnls = trades.map(t => t.pnl || 0).filter(p => p !== 0)
    if (pnls.length < 2) return []
    const min = Math.min(...pnls), max = Math.max(...pnls)
    const step = (max - min) / 10 || 1
    const b = Array.from({ length: 10 }, (_, i) => ({
      r: (min + step * i).toFixed(0), c: 0, pos: (min + step * i) >= 0
    }))
    pnls.forEach(p => { const idx = Math.min(Math.floor((p - min) / step), 9); b[idx].c++ })
    return b
  })()

  const monthly = (() => {
    const map = {}
    trades.forEach(t => {
      const m = (t.timestamp || '').slice(0, 7)
      if (m) map[m] = (map[m] || 0) + (t.pnl || 0)
    })
    return Object.entries(map).map(([m, p]) => ({ m, p: +p.toFixed(2) }))
  })()

  const kpis = [
    { label: 'Total P&L',     val: fmtUSD(m.total_pnl),     color: (m.total_pnl||0)>=0 ? 'text-hex-green':'text-hex-red' },
    { label: 'Win Rate',      val: `${m.win_rate||0}%`,     color: 'text-hex-accent', sub: `${m.total_trades||0} trades` },
    { label: 'Profit Factor', val: m.profit_factor||'—',    color: (m.profit_factor||0)>1?'text-hex-green':'text-hex-red' },
    { label: 'Sharpe Ratio',  val: m.sharpe_ratio||'—',     color: (m.sharpe_ratio||0)>1?'text-hex-green':'text-hex-yellow' },
    { label: 'Max Drawdown',  val: `-${m.max_drawdown||0}%`,color: (m.max_drawdown||0)>10?'text-hex-red':'text-hex-yellow' },
    { label: 'Avg Win',       val: fmtUSD(m.avg_win),       color: 'text-hex-green' },
    { label: 'Avg Loss',      val: fmtUSD(m.avg_loss),      color: 'text-hex-red' },
    { label: 'Best Trade',    val: fmtUSD(m.best_trade),    color: 'text-hex-green' },
  ]

  return (
    <div className="p-6 space-y-6 animate-fade-in">

      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display text-2xl font-bold text-hex-text">Performance</h1>
          <p className="text-hex-sub text-sm font-mono mt-0.5">Detailed trading analytics</p>
        </div>
        <div className="flex gap-1">
          {PERIODS.map(p => (
            <button key={p.id} onClick={() => setPeriod(p.id)}
              className={clsx('px-3 py-1.5 rounded-lg text-xs font-mono transition-all',
                period === p.id
                  ? 'bg-hex-accent/20 text-hex-accent border border-hex-accent/30'
                  : 'text-hex-sub border border-hex-border hover:text-hex-text'
              )}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => <div key={i} className="shimmer h-24 rounded-xl" />)}
        </div>
      ) : !m.total_trades ? (
        <div className="card text-center py-20 text-hex-sub font-mono">No trades in this period</div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {kpis.map(({ label, val, color, sub }) => (
              <div key={label} className="stat-box">
                <span className="stat-label">{label}</span>
                <span className={clsx('stat-value text-xl font-mono', color)}>{val}</span>
                {sub && <span className="text-hex-sub text-xs font-mono">{sub}</span>}
              </div>
            ))}
          </div>

          {curve.length > 1 && (
            <div className="chart-container p-4">
              <div className="font-display text-sm font-semibold text-hex-text mb-4">Equity Curve</div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={curve} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
                  <defs>
                    <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#00ff9d" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#00ff9d" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
                  <XAxis dataKey="t" tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tickLine={false} axisLine={false} tickFormatter={v => `$${v.toFixed(0)}`} />
                  <Tooltip contentStyle={{ background: '#0c1220', border: '1px solid #1a2540', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                  <Area type="monotone" dataKey="v" stroke="#00ff9d" strokeWidth={2} fill="url(#g2)" dot={false} name="Portfolio" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {pnlHist.length > 0 && (
              <div className="chart-container p-4">
                <div className="font-display text-sm font-semibold text-hex-text mb-4">P&L Distribution</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={pnlHist} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
                    <XAxis dataKey="r" tickLine={false} />
                    <YAxis tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ background: '#0c1220', border: '1px solid #1a2540', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                    <Bar dataKey="c" name="Trades" radius={[2,2,0,0]}>
                      {pnlHist.map((e, i) => <Cell key={i} fill={e.pos ? '#00ff9d' : '#ff3366'} fillOpacity={0.8} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {monthly.length > 0 && (
              <div className="chart-container p-4">
                <div className="font-display text-sm font-semibold text-hex-text mb-4">Monthly P&L</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={monthly} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
                    <XAxis dataKey="m" tickLine={false} />
                    <YAxis tickLine={false} axisLine={false} />
                    <ReferenceLine y={0} stroke="#3a4a6b" />
                    <Tooltip formatter={v => [`$${v.toFixed(2)}`, 'P&L']} contentStyle={{ background: '#0c1220', border: '1px solid #1a2540', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                    <Bar dataKey="p" radius={[2,2,0,0]}>
                      {monthly.map((e, i) => <Cell key={i} fill={e.p >= 0 ? '#00ff9d' : '#ff3366'} fillOpacity={0.8} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {Object.keys(byS).length > 0 && (
            <div className="card">
              <div className="font-display text-sm font-semibold text-hex-text mb-4">Strategy Breakdown</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {Object.entries(byS).map(([name, d]) => {
                  const wr    = d.wins / (d.trades || 1) * 100
                  const color = d.pnl >= 0 ? '#00ff9d' : '#ff3366'
                  return (
                    <div key={name} className="card p-3 flex items-center gap-3">
                      <div className="w-2 h-8 rounded-full" style={{ background: color }} />
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-xs font-semibold text-hex-text truncate">{name}</div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <div className="flex-1 h-1 bg-hex-muted rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${wr}%`, background: color }} />
                          </div>
                          <span className="text-hex-sub text-xs font-mono">{wr.toFixed(0)}%</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-sm" style={{ color }}>{fmtUSD(d.pnl)}</div>
                        <div className="text-hex-sub text-xs font-mono">{d.trades} trades</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}