import { create } from 'zustand'

export const useStore = create((set, get) => ({
  connected:  false,
  portfolio:  null,
  signals:    [],
  orders:     [],
  regimes:    {},
  lastTick:   null,
  flashCells: {},

  setConnected: (v) => set({ connected: v }),
  setPortfolio: (p) => set({ portfolio: p }),
  setRegimes:   (r) => set({ regimes: r }),
  setLastTick:  (t) => set({ lastTick: t }),

  setSignals: (s) => set({ signals: s }),
  addSignals: (arr) => set((s) => ({
    signals: [...arr, ...s.signals].slice(0, 300)
  })),

  setOrders: (o) => set({ orders: o }),
  addOrders: (arr) => set((s) => ({
    orders: [...arr, ...s.orders].slice(0, 200)
  })),

  flash: (key, dir) => {
    set((s) => ({ flashCells: { ...s.flashCells, [key]: dir } }))
    setTimeout(() => set((s) => {
      const c = { ...s.flashCells }; delete c[key]; return { flashCells: c }
    }), 800)
  },
}))