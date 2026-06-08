import { useEffect, useRef } from 'react'
import { useStore } from '../store'

const WS_URL = (() => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host  = import.meta.env.DEV ? 'localhost:8000' : window.location.host
  return `${proto}://${host}/ws`
})()

export function useWebSocket() {
  const ws    = useRef(null)
  const retry = useRef(null)
  const store = useStore()

  const connect = () => {
    if (ws.current?.readyState === WebSocket.OPEN) return
    const socket = new WebSocket(WS_URL)
    ws.current   = socket

    socket.onopen  = ()  => store.setConnected(true)
    socket.onclose = ()  => {
      store.setConnected(false)
      retry.current = setTimeout(connect, 3000)
    }
    socket.onerror = ()  => socket.close()
    socket.onmessage = ({ data }) => {
      try { handle(JSON.parse(data), store) } catch (e) {}
    }
  }

  useEffect(() => {
    connect()
    return () => { clearTimeout(retry.current); ws.current?.close() }
  }, [])
}

function handle(msg, store) {
  const { type, data } = msg

  if (type === 'snapshot') {
    if (data.portfolio) store.setPortfolio(data.portfolio)
    if (data.regimes)   store.setRegimes(data.regimes)
  }
  else if (type === 'tick') {
    if (data.portfolio) {
      const prev = store.portfolio
      if (prev && data.portfolio.total_value !== prev.total_value) {
        store.flash('pv', data.portfolio.total_value > prev.total_value ? 'up' : 'down')
      }
      store.setPortfolio(data.portfolio)
    }
    store.setLastTick(data.ts)
    if (data.latest_signals?.length) store.addSignals(data.latest_signals)
    if (data.latest_orders?.length)  store.addOrders(data.latest_orders)
  }
  else if (type === 'new_signals') store.addSignals(data)
  else if (type === 'new_orders')  store.addOrders(data)
}