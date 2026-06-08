import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api, fmt, fmtUSD, fmtPct, fmtTime, regimeColor, regimeLabel } from '../utils/api'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid
} from 'recharts'
import { TrendingUp, TrendingDown, Activity, Shield, BarChart3 } from 'lucide-react'
import clsx from 'clsx'

function RegimePill({ symbol, regime }) {
  const color = regimeColor(regime)
  return (
    <div className="card p-3 flex items-center justify-between gap-2">
      <div>
        <div className="font-mono text-xs text-hex-sub mb-0.5">{symbol}</div>
        <div className="font-mono text-sm font-semibold" style={{ color }}>
          {regimeLabel(regime)}
        </div>
      </div>
      <div className="w-8 h-8 hex-shape flex items-center justify-center"
           style={{ background: `${color}15`, border: `1px solid ${color}30` }}>
        {regime?.includes('up')       && <TrendingUp   size={14} style={{ color }} />}
        {regime?.includes('down')     && <TrendingDown size={14} style={{ color }} />}
        {regime?.includes('sideways') && <Activity     size={14} style={{ color }} />}
        {regime?.includes('volatile') && <Shield       size={14} style={{ color }} />}
        {!regime                      && <BarChart3    size={14} style={{ color }} />}
      </div>
    </div>
  )
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="card p-2.5 text-xs border border-hex-accent/20">
      <div className="text-hex-sub font-mono mb-1">{label}</div>
      <div className="text-hex-accent font-mono font-semibold">
        ${Number(payload[0]?.value || 0).toFixed(2)}
      </div>
    </div>
  )
}

export default function Overview() {
  const portfolio  = useStore(s => s.portfolio)
  const regimes    = useStore(s => s.regimes)
  const signals    = useStore(s => s.signals)
  const orders     = useStore(s => s.orders)
  const flashCells = useStore(s => s.flashCells)

  const [curve,    setCurve]   = useState([])
  const [loading,  setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.overview(), api.equityCurve()])
      .then(([ov, raw]) => {
        setCurve(raw.map(r => ({ t: (r.timestamp || '').slice(11, 16), v: Number(r.total_value) })))
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const pf       = portfolio || {}
  const tv       = Number(pf.total_value   || 10000)
  const pnl      = Number(pf.total_pnl     || 0)
  const pnlPct   = Number(pf.total_pnl_pct || 0)
  const dd       = Number(pf.drawdown_pct  || 0)
  const openPos  = Number(pf.open_positions || 0)
  const pos      = pnl >= 0
  const wins     = orders.filter(o => (o.pnl || 0) > 0).length
  const winRate  = orders.length ? (wins / orders.length * 100).toFixed(1) : '—'
  const buys     = signals.slice(0, 50).filter(s => s.side === 'buy').length
  const sells    = signals.slice(0, 50).filter(s => s.side === 'sell').length

  return (
    <div className="p-6 space-y-6 animate-fade-in">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-hex-text tracking-tight">System Overview</h1>
          <p className="text-hex-sub text-sm font-mono mt-0.5">Real-time trading dashboard • Paper mode</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="live-dot" />
          <span className="font-mono text-xs text-hex-green">LIVE FEED</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">

        <div className={clsx('stat-box card-glow',
          flashCells['pv'] === 'up'   && 'ring-1 ring-hex-green/40',
          flashCells['pv'] === 'down' && 'ring-1 ring-hex-red/40',
        )}>
          <span className="stat-label">Portfolio Value</span>
          <span className={clsx('font-mono font-bold text-xl',
            flashCells['pv'] === 'up'   ? 'text-hex-green' :
            flashCells['pv'] === 'down' ? 'text-hex-red'   : 'text-hex-text'
          )}>
            ${tv.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
          <span className={pos ? 'stat-delta-up' : 'stat-delta-down'}>{fmtPct(pnlPct)} all time</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Total P&L</span>
          <span className={clsx('font-mono font-bold text-xl', pos ? 'text-hex-green' : 'text-hex-red')}>
            {fmtUSD(pnl)}
          </span>
          <span className="text-hex-sub text-xs font-mono">realized + unrealized</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Win Rate</span>
          <span className="font-mono font-bold text-xl text-hex-accent">{winRate}%</span>
          <span className="text-hex-sub text-xs font-mono">{orders.length} total orders</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Drawdown</span>
          <span className={clsx('font-mono font-bold text-xl',
            dd > 10 ? 'text-hex-red' : dd > 5 ? 'text-hex-yellow' : 'text-hex-green'
          )}>
            -{fmt(dd)}%
          </span>
          <span className="text-hex-sub text-xs font-mono">from peak</span>
        </div>
      </div>

      {/* Chart + Regimes */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        <div className="chart-container scan-line lg:col-span-2 p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="font-display text-sm font-semibold text-hex-text">Equity Curve</div>
            <span className="badge-blue">Portfolio Value</span>
          </div>
          {loading ? (
            <div className="shimmer h-48 w-full rounded-lg" />
          ) : curve.length > 1 ? (
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={curve} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00d4ff" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
                <XAxis dataKey="t" tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis tickLine={false} axisLine={false} tickFormatter={v => `$${v.toFixed(0)}`} />
                <Tooltip content={<ChartTip />} />
                <Area type="monotone" dataKey="v" stroke="#00d4ff" strokeWidth={2}
                      fill="url(#g1)" dot={false} activeDot={{ r: 4, fill: '#00d4ff' }} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-48 flex items-center justify-center text-hex-sub text-xs font-mono">
              No equity data yet
            </div>
          )}
        </div>

        <div className="card space-y-2">
          <div className="font-display text-sm font-semibold text-hex-text mb-1">Market Regimes</div>
          {Object.keys(regimes).length > 0
            ? Object.entries(regimes).map(([s, r]) => <RegimePill key={s} symbol={s} regime={r} />)
            : ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'].map(s => <RegimePill key={s} symbol={s} regime="sideways" />)
          }
          <div className="border-t border-hex-border pt-3 mt-1">
            <div className="text-hex-sub text-xs font-mono mb-2">SIGNAL FLOW (last 50)</div>
            <div className="flex gap-2">
              {[
                { label: 'BUY',  val: buys,   color: 'text-hex-green'  },
                { label: 'SELL', val: sells,  color: 'text-hex-red'    },
                { label: 'OPEN', val: openPos, color: 'text-hex-accent' },
              ].map(({ label, val, color }) => (
                <div key={label} className="flex-1 card p-2 text-center">
                  <div className={clsx('font-mono font-bold', color)}>{val}</div>
                  <div className="text-hex-sub text-xs font-mono">{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Signals */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="font-display text-sm font-semibold text-hex-text">Recent Signals</div>
            <span className="badge-blue">{signals.length}</span>
          </div>
          {signals.length === 0 ? (
            <div className="text-hex-sub text-xs font-mono text-center py-8">Waiting for signals…</div>
          ) : (
            <table className="data-table">
              <thead><tr><th>Symbol</th><th>Side</th><th>Strategy</th><th>Strength</th><th>Time</th></tr></thead>
              <tbody>
                {signals.slice(0, 8).map((s, i) => (
                  <tr key={i}>
                    <td className="text-hex-text font-semibold">{s.symbol}</td>
                    <td><span className={s.side === 'buy' ? 'badge-green' : 'badge-red'}>{s.side?.toUpperCase()}</span></td>
                    <td className="text-hex-sub">{s.strategy}</td>
                    <td>
                      <div className="flex items-center gap-1.5">
                        <div className="w-12 h-1 bg-hex-muted rounded-full overflow-hidden">
                          <div className="h-full bg-hex-accent rounded-full"
                               style={{ width: `${Math.min((s.strength / 2) * 100, 100)}%` }} />
                        </div>
                        <span className="text-hex-sub text-xs">{fmt(s.strength)}</span>
                      </div>
                    </td>
                    <td className="text-hex-dim">{fmtTime(s.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Orders */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="font-display text-sm font-semibold text-hex-text">Recent Orders</div>
            <span className="badge-blue">{orders.length}</span>
          </div>
          {orders.length === 0 ? (
            <div className="text-hex-sub text-xs font-mono text-center py-8">No orders yet</div>
          ) : (
            <table className="data-table">
              <thead><tr><th>Symbol</th><th>Side</th><th>Price</th><th>P&L</th><th>Strategy</th></tr></thead>
              <tbody>
                {orders.slice(0, 8).map((o, i) => (
                  <tr key={i}>
                    <td className="text-hex-text font-semibold">{o.symbol}</td>
                    <td><span className={o.side === 'buy' ? 'badge-green' : 'badge-red'}>{o.side?.toUpperCase()}</span></td>
                    <td className="text-hex-sub">${Number(o.filled_price || 0).toFixed(2)}</td>
                    <td className={o.pnl > 0 ? 'text-hex-green' : o.pnl < 0 ? 'text-hex-red' : 'text-hex-sub'}>
                      {o.pnl != null ? fmtUSD(o.pnl) : '—'}
                    </td>
                    <td className="text-hex-sub">{o.strategy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}