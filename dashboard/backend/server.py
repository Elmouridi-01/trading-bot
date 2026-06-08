"""
dashboard/backend/server.py

FastAPI + WebSocket dashboard that reads the live SQLite DB (data/trading.db).

FIX H9 (security):
  - Binds to DASHBOARD_HOST (default 127.0.0.1 = local only).
  - CORS restricted to localhost origins by default (override via DASHBOARD_CORS).
  - Optional bearer token (DASHBOARD_TOKEN). When set, every /api/* request and
    the /ws socket must present it (Authorization: Bearer <t> OR ?token=<t>).
  - If bound to a NON-local host without a token, the server refuses to start,
    so the dashboard can never be exposed on the network unauthenticated.

FIX M7 (schema):
  - Reads the real schema written by data/storage/database.py:
      * signals / orders use `created_at` (not `timestamp`)
      * portfolio_snapshots stores a JSON blob in `snapshot` (not flat columns)
    Queries and aggregations updated accordingly.
"""
import os
import sys
import json
import math
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Set, Optional

# Optional .env load so DASHBOARD_* can live alongside the bot's config.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# ── Configuration ─────────────────────────────────────────────
HOST  = os.environ.get("DASHBOARD_HOST", "127.0.0.1").strip()
PORT  = int(os.environ.get("DASHBOARD_PORT", "8000"))
TOKEN = os.environ.get("DASHBOARD_TOKEN", "").strip()

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", ""}
if HOST not in _LOCAL_HOSTS and not TOKEN:
    raise SystemExit(
        f"[Dashboard] Refusing to start: DASHBOARD_HOST is non-local ('{HOST}') "
        "but DASHBOARD_TOKEN is not set.\n"
        "Set a token to expose the dashboard on your network "
        "(DASHBOARD_TOKEN=...), or bind to 127.0.0.1."
    )

app = FastAPI(title="AI Trading Dashboard", version="3.0")

_default_cors = [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:5173", "http://localhost:5173",
]
_cors_env = os.environ.get("DASHBOARD_CORS", "").strip()
allow_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env else _default_cors
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────
def require_token(request: Request) -> None:
    """FastAPI dependency. No-op when DASHBOARD_TOKEN is unset (local mode)."""
    if not TOKEN:
        return
    supplied = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied:
        supplied = request.query_params.get("token", "")
    if supplied != TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing token")


def _ws_token_ok(ws: WebSocket) -> bool:
    if not TOKEN:
        return True
    return ws.query_params.get("token", "") == TOKEN


# ── WebSocket connection manager ──────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)


manager = ConnectionManager()


# ── Database access ───────────────────────────────────────────
def get_db_path() -> Optional[str]:
    for p in [
        os.path.join(ROOT, "data", "trading.db"),
        os.path.join(ROOT, "trading.db"),
    ]:
        if os.path.exists(p):
            return p
    return None


def qdb(sql: str, params=()) -> list:
    path = get_db_path()
    if not path:
        return []
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[Dashboard][DB] {e}")
        return []
    finally:
        conn.close()


def _latest_snapshot() -> dict:
    rows = qdb("SELECT snapshot FROM portfolio_snapshots ORDER BY id DESC LIMIT 1")
    if not rows:
        return {}
    try:
        return json.loads(rows[0]["snapshot"])
    except Exception:
        return {}


def _period_cutoff(period: str) -> str:
    now = datetime.utcnow()
    if period == "week":
        return (now - timedelta(days=7)).isoformat()
    if period == "month":
        return (now - timedelta(days=30)).isoformat()
    return "2000-01-01"


# ── Data builders (no auth; reused by REST + WebSocket) ───────
def _overview_data() -> dict:
    snap = _latest_snapshot()
    pf = {
        "total_value":    snap.get("total_value", 0),
        "cash":           snap.get("cash", 0),
        "total_pnl":      snap.get("total_pnl", 0),
        "total_pnl_pct":  snap.get("total_pnl_pct", 0),
        "drawdown_pct":   snap.get("drawdown_pct", 0),
        "open_positions": snap.get("open_positions", 0),
    }

    sigs = qdb("SELECT symbol, side FROM signals ORDER BY created_at DESC LIMIT 30")
    syms: dict = {}
    for s in sigs:
        sym = s["symbol"]
        syms.setdefault(sym, {"buy": 0, "sell": 0})
        if s.get("side") in ("buy", "sell"):
            syms[sym][s["side"]] += 1
    regimes = {}
    for sym, cnt in syms.items():
        total = cnt["buy"] + cnt["sell"] or 1
        r = cnt["buy"] / total
        regimes[sym] = ("trending_up" if r > 0.6
                        else "trending_down" if r < 0.4 else "sideways")

    perf = qdb(
        "SELECT COUNT(*) AS c, "
        "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS w "
        "FROM orders WHERE side = 'sell'"
    )
    total_orders = perf[0]["c"] if perf else 0
    total_wins   = (perf[0]["w"] if perf else 0) or 0
    return {
        "portfolio":    pf,
        "regimes":      regimes,
        "total_orders": total_orders,
        "total_wins":   total_wins,
        "win_rate":     round(total_wins / (total_orders or 1) * 100, 1),
    }


def _performance_data(period: str = "all") -> dict:
    cutoff = _period_cutoff(period)
    orders = qdb(
        "SELECT * FROM orders WHERE side = 'sell' AND created_at >= ? "
        "ORDER BY created_at ASC",
        (cutoff,),
    )
    pnls   = [o["pnl"] for o in orders if o.get("pnl") is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate  = len(wins) / len(pnls) * 100 if pnls else 0
    pf        = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0.0
    mean      = sum(pnls) / len(pnls) if pnls else 0
    variance  = sum((x - mean) ** 2 for x in pnls) / len(pnls) if pnls else 0
    std       = math.sqrt(variance) if variance > 0 else 0
    sharpe    = (mean / std * math.sqrt(252)) if std > 0 else 0

    # Real equity curve from portfolio snapshots.
    snap_rows = qdb(
        "SELECT snapshot, saved_at FROM portfolio_snapshots ORDER BY id ASC"
    )
    curve = []
    max_dd = 0.0
    for r in snap_rows:
        try:
            s = json.loads(r["snapshot"])
        except Exception:
            continue
        curve.append({"t": (r["saved_at"] or "")[:16], "v": s.get("total_value", 0)})
        max_dd = max(max_dd, float(s.get("drawdown_pct", 0) or 0))

    by_strat: dict = {}
    for o in orders:
        s = o.get("strategy") or "unknown"
        by_strat.setdefault(s, {"trades": 0, "pnl": 0.0, "wins": 0})
        by_strat[s]["trades"] += 1
        by_strat[s]["pnl"]    += o.get("pnl") or 0
        if (o.get("pnl") or 0) > 0:
            by_strat[s]["wins"] += 1

    return {
        "metrics": {
            "total_trades":  len(orders),
            "win_rate":      round(win_rate, 1),
            "total_pnl":     round(total_pnl, 2),
            "avg_win":       round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss":      round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(pf, 2),
            "sharpe_ratio":  round(sharpe, 2),
            "max_drawdown":  round(max_dd, 2),
            "best_trade":    round(max(pnls), 2) if pnls else 0,
            "worst_trade":   round(min(pnls), 2) if pnls else 0,
        },
        "equity_curve": curve,
        "by_strategy":  by_strat,
        "trades":       orders[-100:],
    }


# ── REST endpoints ────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ts": datetime.utcnow().isoformat(),
        "db": bool(get_db_path()),
        "auth": bool(TOKEN),
    }


@app.get("/api/overview")
async def get_overview(_: None = Depends(require_token)):
    return _overview_data()


@app.get("/api/signals")
async def get_signals(
    limit: int = 200,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    _: None = Depends(require_token),
):
    sql, params, where = "SELECT * FROM signals", [], []
    if symbol:
        where.append("symbol = ?"); params.append(symbol)
    if side:
        where.append("side = ?"); params.append(side)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return qdb(sql, params)


@app.get("/api/orders")
async def get_orders(limit: int = 200, _: None = Depends(require_token)):
    return qdb("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))


@app.get("/api/performance")
async def get_performance(period: str = "all", _: None = Depends(require_token)):
    return _performance_data(period)


@app.get("/api/equity-curve")
async def get_equity_curve(_: None = Depends(require_token)):
    rows = qdb("SELECT snapshot, saved_at FROM portfolio_snapshots ORDER BY id ASC")
    out = []
    for r in rows:
        try:
            s = json.loads(r["snapshot"])
        except Exception:
            continue
        out.append({
            "timestamp":     r["saved_at"],
            "total_value":   s.get("total_value", 0),
            "total_pnl_pct": s.get("total_pnl_pct", 0),
            "drawdown_pct":  s.get("drawdown_pct", 0),
        })
    return out


# ── WebSocket ─────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _ws_token_ok(ws):
        await ws.close(code=1008)   # policy violation
        return
    await manager.connect(ws)
    try:
        await ws.send_json({"type": "snapshot", "data": _overview_data()})
        while True:
            await asyncio.sleep(3)
            sigs = qdb("SELECT * FROM signals ORDER BY created_at DESC LIMIT 5")
            ords = qdb("SELECT * FROM orders  ORDER BY created_at DESC LIMIT 3")
            await ws.send_json({"type": "tick", "data": {
                "portfolio":      _latest_snapshot(),
                "latest_signals": sigs,
                "latest_orders":  ords,
                "ts":             datetime.utcnow().isoformat(),
            }})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ── Serve built frontend (static shell carries no data; data is token-gated) ──
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_DIST):
    _ASSETS = os.path.join(_DIST, "assets")
    if os.path.exists(_ASSETS):
        app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        return FileResponse(os.path.join(_DIST, "index.html"))


if __name__ == "__main__":
    mode = "LOCAL ONLY" if HOST in _LOCAL_HOSTS else f"NETWORK (token {'set' if TOKEN else 'MISSING'})"
    print(f"[Dashboard] starting on {HOST}:{PORT} | {mode}")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="info")