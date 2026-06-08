import { useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './store'
import Overview    from './pages/Overview'
import Signals     from './pages/Signals'
import Performance from './pages/Performance'
import Orders      from './pages/Orders'
import { LayoutDashboard, Zap, TrendingUp, BookOpen, WifiOff } from 'lucide-react'
import clsx from 'clsx'

const NAV = [
  { id: 'overview',    label: 'Overview',    icon: LayoutDashboard },
  { id: 'signals',     label: 'Signals',     icon: Zap },
  { id: 'performance', label: 'Performance', icon: TrendingUp },
  { id: 'orders',      label: 'Orders',      icon: BookOpen },
]

export default function App() {
  const [page, setPage] = useState('overview')
  const connected = useStore(s => s.connected)
  const portfolio = useStore(s => s.portfolio)
  const flashCells= useStore(s => s.flashCells)

  useWebSocket()

  const tv  = Number(portfolio?.total_value  || 10000)
  const pnl = Number(portfolio?.total_pnl    || 0)
  const now = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })

  return (
    <div className="flex h-screen overflow-hidden hex-grid-bg">

      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-hex-surface border-r border-hex-border flex flex-col relative overflow-hidden">

        {/* Hex pattern */}
        <div className="absolute inset-0 pointer-events-none opacity-20"
          style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg width='40' height='35' viewBox='0 0 40 35' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M20 0 L40 11.5 L40 23.5 L20 35 L0 23.5 L0 11.5 Z' fill='none' stroke='%231a2540' stroke-width='1'/%3E%3C/svg%3E")`, backgroundSize: '40px 35px' }} />

        {/* Logo */}
        <div className="p-5 border-b border-hex-border relative">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 hex-shape bg-hex-accent/20 border border-hex-accent/40
                            flex items-center justify-center text-base animate-glow">
              ⚡
            </div>
            <div>
              <div className="font-display font-bold text-hex-text text-sm tracking-tight">AI Trading</div>
              <div className="font-mono text-xs text-hex-sub">SYSTEM v2.0</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1 relative">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setPage(id)}
                    className={clsx('nav-item w-full', page === id && 'active')}>
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {/* Status */}
        <div className="p-3 border-t border-hex-border space-y-2 relative">
          <div className="card p-2.5 flex items-center gap-2">
            {connected
              ? <><span className="live-dot" /><span className="text-hex-green text-xs font-mono">LIVE</span></>
              : <><WifiOff size={12} className="text-hex-red" /><span className="text-hex-red text-xs font-mono">OFFLINE</span></>}
            <span className="text-hex-dim text-xs font-mono ml-auto">{now}</span>
          </div>

          {portfolio && (
            <div className={clsx('card p-2.5 transition-all duration-300',
              flashCells['pv'] === 'up'   && 'ring-1 ring-hex-green/50',
              flashCells['pv'] === 'down' && 'ring-1 ring-hex-red/50',
            )}>
              <div className="text-hex-sub text-xs font-mono mb-1">PORTFOLIO</div>
              <div className="text-hex-text font-mono text-sm font-semibold">
                ${tv.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </div>
              <div className={clsx('text-xs font-mono', pnl >= 0 ? 'text-hex-green' : 'text-hex-red')}>
                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)} USDT
              </div>
            </div>
          )}

          <div className="text-center">
            <span className="badge-yellow">PAPER TRADING</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        {page === 'overview'    && <Overview />}
        {page === 'signals'     && <Signals />}
        {page === 'performance' && <Performance />}
        {page === 'orders'      && <Orders />}
      </main>
    </div>
  )
}