import { useState, useEffect } from 'react'
import { useStore } from '../store'
import { api, fmt, fmtTime } from '../utils/api'
import { Zap, Filter } from 'lucide-react'
import clsx from 'clsx'

const STRATEGIES = ['All','momentum','mean_reversion','trend_following','ai_strategy','vwap_reversion']
const SYMBOLS    = ['All','BTC/USDT','ETH/USDT','SOL/USDT']
const SIDES      = ['All','buy','sell']

function Chips({ options, value, onChange }) {
  return (
    <div className="flex gap-1 flex-wrap">
      {options.map(o => (
        <button key={o} onClick={() => onChange(o)}
          className={clsx('px-2 py-0.5 rounded text-xs font-mono transition-all',
            value === o
              ? 'bg-hex-accent/20 text-hex-accent border border-hex-accent/30'
              : 'text-hex-sub border border-hex-border hover:text-hex-text'
          )}>
          {o}
        </button>
      ))}
    </div>
  )
}

export default function Signals() {
  const signals    = useStore(s => s.signals)
  const setSignals = useStore(s => s.setSignals)

  const [sym,     setSym]     = useState('All')
  const [side,    setSide]    = useState('All')
  const [strat,   setStrat]   = useState('All')
  const [loading, setLoading] = useState(!signals.length)

  useEffect(() => {
    if (!signals.length) {
      api.signals({ limit: 300 })
        .then(d => { setSignals(d); setLoading(false) })
        .catch(() => setLoading(false))
    } else { setLoading(false) }
  }, [])

  const filtered = signals.filter(s =>
    (sym   === 'All' || s.symbol   === sym)  &&
    (side  === 'All' || s.side     === side) &&
    (strat === 'All' || s.strategy === strat)
  )

  const buys   = filtered.filter(s => s.side === 'buy').length
  const sells  = filtered.filter(s => s.side === 'sell').length
  const avgStr = filtered.length
    ? (filtered.reduce((a, s) => a + (s.strength || 0), 0) / filtered.length).toFixed(2) : '—'

  return (
    <div className="p-6 space-y-6 animate-fade-in">

      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-hex-text">Signals</h1>
          <p className="text-hex-sub text-sm font-mono mt-0.5">Live trading signals log</p>
        </div>
        <span className="live-dot" />
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label:'Total',    val: filtered.length, color:'text-hex-text'   },
          { label:'BUY',      val: buys,            color:'text-hex-green'  },
          { label:'SELL',     val: sells,           color:'text-hex-red'    },
          { label:'Avg Str.', val: avgStr,          color:'text-hex-accent' },
        ].map(({ label, val, color }) => (
          <div key={label} className="stat-box">
            <span className="stat-label">{label}</span>
            <span className={clsx('stat-value text-2xl font-mono', color)}>{val}</span>
          </div>
        ))}
      </div>

      <div className="card space-y-3">
        <div className="flex items-center gap-2 text-hex-sub text-xs font-mono">
          <Filter size={12} /> FILTERS
        </div>
        <div className="flex flex-wrap gap-4">
          <div><span className="text-hex-sub text-xs font-mono mr-2">Symbol:</span><Chips options={SYMBOLS} value={sym} onChange={setSym} /></div>
          <div><span className="text-hex-sub text-xs font-mono mr-2">Side:</span><Chips options={SIDES} value={side} onChange={setSide} /></div>
          <div><span className="text-hex-sub text-xs font-mono mr-2">Strategy:</span><Chips options={STRATEGIES} value={strat} onChange={setStrat} /></div>
        </div>
        <div className="text-hex-dim text-xs font-mono">{filtered.length} results</div>
      </div>

      <div className="card overflow-hidden">
        {loading ? (
          <div className="space-y-2 p-4">{[...Array(8)].map((_, i) => <div key={i} className="shimmer h-8 rounded" />)}</div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-16 text-hex-sub font-mono">
            <Zap size={28} className="mx-auto mb-3 opacity-30" />
            No signals match filters
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Strategy</th><th>Strength</th><th>Price</th><th>Reason</th></tr>
              </thead>
              <tbody>
                {filtered.slice(0, 150).map((s, i) => (
                  <tr key={i}>
                    <td className="text-hex-dim whitespace-nowrap">{fmtTime(s.timestamp)}</td>
                    <td className="text-hex-text font-semibold">{s.symbol}</td>
                    <td><span className={s.side === 'buy' ? 'badge-green' : 'badge-red'}>{s.side?.toUpperCase()}</span></td>
                    <td><span className="badge-purple">{s.strategy}</span></td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="w-14 h-1.5 bg-hex-muted rounded-full overflow-hidden">
                          <div className="h-full rounded-full transition-all" style={{
                            width: `${Math.min((s.strength / 2) * 100, 100)}%`,
                            background: s.strength > 1.2 ? '#00ff9d' : s.strength > 0.7 ? '#00d4ff' : '#ffd166'
                          }} />
                        </div>
                        <span className="text-xs font-mono text-hex-text">{fmt(s.strength)}</span>
                      </div>
                    </td>
                    <td className="text-hex-sub font-mono">{s.price ? `$${Number(s.price).toFixed(2)}` : '—'}</td>
                    <td className="text-hex-dim text-xs max-w-xs truncate">{s.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}