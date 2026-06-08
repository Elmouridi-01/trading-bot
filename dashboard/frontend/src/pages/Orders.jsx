import { useState, useEffect } from 'react'
import { useStore } from '../store'
import { api, fmt, fmtUSD, fmtTime, fmtDate } from '../utils/api'
import { BookOpen } from 'lucide-react'
import clsx from 'clsx'

export default function Orders() {
  const orders    = useStore(s => s.orders)
  const setOrders = useStore(s => s.setOrders)
  const [loading, setLoading] = useState(!orders.length)

  useEffect(() => {
    if (!orders.length) {
      api.orders({ limit: 300 })
        .then(d => { setOrders(d); setLoading(false) })
        .catch(() => setLoading(false))
    } else { setLoading(false) }
  }, [])

  const totalPnl = orders.reduce((a, o) => a + (o.pnl || 0), 0)
  const wins     = orders.filter(o => (o.pnl || 0) > 0).length
  const losses   = orders.filter(o => (o.pnl || 0) < 0).length
  const wr       = orders.length ? (wins / orders.length * 100).toFixed(1) : 0

  return (
    <div className="p-6 space-y-6 animate-fade-in">

      <div>
        <h1 className="font-display text-2xl font-bold text-hex-text">Orders</h1>
        <p className="text-hex-sub text-sm font-mono mt-0.5">Complete order history</p>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label:'Total',    val: orders.length, color:'text-hex-text'   },
          { label:'Winners',  val: wins,          color:'text-hex-green'  },
          { label:'Losers',   val: losses,        color:'text-hex-red'    },
          { label:'Win Rate', val: `${wr}%`,      color:'text-hex-accent' },
        ].map(({ label, val, color }) => (
          <div key={label} className="stat-box">
            <span className="stat-label">{label}</span>
            <span className={clsx('stat-value text-2xl font-mono', color)}>{val}</span>
          </div>
        ))}
      </div>

      <div className="card overflow-hidden">
        {loading ? (
          <div className="space-y-2 p-4">{[...Array(10)].map((_, i) => <div key={i} className="shimmer h-8 rounded" />)}</div>
        ) : orders.length === 0 ? (
          <div className="text-center py-20">
            <BookOpen size={32} className="mx-auto mb-3 text-hex-dim" />
            <div className="text-hex-sub text-sm font-mono">No orders yet</div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr><th>Date</th><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Value</th><th>P&L</th><th>Strategy</th><th>Status</th></tr>
              </thead>
              <tbody>
                {orders.map((o, i) => {
                  const val = (o.quantity || 0) * (o.filled_price || 0)
                  return (
                    <tr key={i}>
                      <td className="text-hex-dim">{fmtDate(o.timestamp)}</td>
                      <td className="text-hex-dim">{fmtTime(o.timestamp)}</td>
                      <td className="text-hex-text font-semibold">{o.symbol}</td>
                      <td><span className={o.side === 'buy' ? 'badge-green' : 'badge-red'}>{o.side?.toUpperCase()}</span></td>
                      <td className="text-hex-sub">{fmt(o.quantity, 4)}</td>
                      <td className="text-hex-sub">${Number(o.filled_price || 0).toFixed(2)}</td>
                      <td className="text-hex-sub">${val.toFixed(2)}</td>
                      <td className={clsx('font-semibold',
                        (o.pnl||0)>0?'text-hex-green':(o.pnl||0)<0?'text-hex-red':'text-hex-sub'
                      )}>
                        {o.pnl != null ? fmtUSD(o.pnl) : '—'}
                      </td>
                      <td><span className="badge-purple">{o.strategy || '—'}</span></td>
                      <td><span className={o.status==='filled'?'badge-green':'badge-yellow'}>{o.status?.toUpperCase()||'FILLED'}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {orders.length > 0 && (
        <div className="card flex items-center justify-between">
          <span className="text-hex-sub text-xs font-mono">{orders.length} orders</span>
          <span className={clsx('font-mono text-sm font-semibold', totalPnl>=0?'text-hex-green':'text-hex-red')}>
            Total P&L: {fmtUSD(totalPnl)}
          </span>
        </div>
      )}
    </div>
  )
}