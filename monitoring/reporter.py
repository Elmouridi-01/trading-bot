import csv
import os
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path


REPORT_DIR  = "logs/reports"
TRADES_FILE = f"{REPORT_DIR}/trades.csv"
DAILY_FILE  = f"{REPORT_DIR}/daily.csv"

MAX_TRADE_ROWS = 100_000
MAX_DAILY_ROWS = 3_650


@dataclass
class TradeRecord:
    timestamp:    str
    symbol:       str
    side:         str
    entry_price:  float
    exit_price:   float
    quantity:     float
    pnl:          float
    pnl_pct:      float
    strategy:     str
    regime:       str
    holding_time: str


class TradingReporter:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        self._init_files()
        self._open_trades: dict[str, dict] = {}

    def _init_files(self) -> None:
        if not os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side",
                    "entry_price", "exit_price", "quantity",
                    "pnl", "pnl_pct", "strategy",
                    "regime", "holding_time",
                ])

        if not os.path.exists(DAILY_FILE):
            with open(DAILY_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "date", "total_trades", "winning_trades",
                    "win_rate", "total_pnl", "best_trade",
                    "worst_trade", "total_signals",
                ])

    def _rotate_if_needed(self, filepath: str, max_rows: int) -> None:
        try:
            path = Path(filepath)
            if not path.exists():
                return

            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) <= max_rows + 1:
                return

            header     = lines[0]
            kept_lines = lines[-(max_rows):]

            backup = filepath.replace(".csv", "_archive.csv")
            with open(backup, "a", encoding="utf-8") as f:
                f.writelines(lines[1:-(max_rows)])

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header)
                f.writelines(kept_lines)

        except Exception as e:
            print(f"[Reporter] ⚠️  تعذر تدوير الملف {filepath}: {e}")

    def record_buy(self, symbol: str, price: float,
                   quantity: float, strategy: str,
                   regime: str = "unknown") -> None:
        self._open_trades[symbol] = {
            "entry_price": price,
            "quantity":    quantity,
            "strategy":    strategy,
            "regime":      regime,
            "entry_time":  datetime.utcnow(),
        }

    def record_sell(self, symbol: str, price: float,
                    quantity: float) -> None:
        if symbol not in self._open_trades:
            return

        trade      = self._open_trades.pop(symbol)
        entry      = trade["entry_price"]
        entry_time = trade["entry_time"]
        exit_time  = datetime.utcnow()

        pnl         = (price - entry) * quantity
        pnl_pct     = (price - entry) / entry * 100
        holding     = exit_time - entry_time
        holding_str = f"{int(holding.total_seconds() // 60)}m"

        record = TradeRecord(
            timestamp=exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            side="SELL",
            entry_price=round(entry, 4),
            exit_price=round(price, 4),
            quantity=round(quantity, 6),
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 3),
            strategy=trade["strategy"],
            regime=trade["regime"],
            holding_time=holding_str,
        )

        self._rotate_if_needed(TRADES_FILE, MAX_TRADE_ROWS)
        self._save_trade(record)

        emoji = "✅" if pnl > 0 else "❌"
        print(
            f"[Reporter] {emoji} {symbol} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"Hold: {holding_str} | "
            f"Strategy: {trade['strategy']}"
        )

    def _save_trade(self, record: TradeRecord) -> None:
        with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp, record.symbol, record.side,
                record.entry_price, record.exit_price, record.quantity,
                record.pnl, record.pnl_pct, record.strategy,
                record.regime, record.holding_time,
            ])

    def save_daily_summary(self, summary: dict,
                           total_signals: int) -> None:
        self._rotate_if_needed(DAILY_FILE, MAX_DAILY_ROWS)

        today    = datetime.utcnow().strftime("%Y-%m-%d")
        trades   = summary.get("total_trades", 0)
        win_rate = summary.get("win_rate", 0)
        wins     = int(trades * win_rate / 100) if trades > 0 else 0

        with open(DAILY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                today, trades, wins,
                f"{win_rate:.1f}%",
                summary.get("total_pnl", 0),
                summary.get("best_trade", 0),
                summary.get("worst_trade", 0),
                total_signals,
            ])
        print(
            f"[Reporter] 📊 Daily summary saved | "
            f"Trades: {trades} | Win Rate: {win_rate:.1f}%"
        )

    def get_performance(self) -> dict:
        if not os.path.exists(TRADES_FILE):
            return {}

        trades = []
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades.append({
                        "symbol":    row["symbol"],
                        "pnl":       float(row["pnl"]),
                        "pnl_pct":   float(row["pnl_pct"]),
                        "strategy":  row["strategy"],
                        "regime":    row["regime"],
                        "timestamp": row["timestamp"],
                    })
                except Exception:
                    continue

        if not trades:
            return {"message": "No trades yet"}

        pnls     = [t["pnl"]     for t in trades]
        returns  = [t["pnl_pct"] for t in trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0

        per_trade_sharpe  = self._calc_per_trade_sharpe(returns)
        annualized_sharpe = self._calc_annualized_sharpe(trades)

        gross_profit  = sum(wins)
        gross_loss    = abs(sum(losses))
        profit_factor = (
            round(gross_profit / gross_loss, 2)
            if gross_loss > 0 else float("inf")
        )

        by_strategy: dict[str, dict] = {}
        for t in trades:
            s = t["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "pnl": 0.0, "wins": 0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["pnl"]    += t["pnl"]
            if t["pnl"] > 0:
                by_strategy[s]["wins"] += 1

        return {
            "total_trades":      len(trades),
            "win_rate":          round(win_rate, 1),
            "total_pnl":         round(sum(pnls), 4),
            "avg_win":           round(sum(wins)   / len(wins),   4) if wins   else 0,
            "avg_loss":          round(sum(losses) / len(losses), 4) if losses else 0,
            "best_trade":        round(max(pnls), 4),
            "worst_trade":       round(min(pnls), 4),
            "sharpe_ratio":      round(per_trade_sharpe, 2),
            "sharpe_annualized": round(annualized_sharpe, 2),
            "profit_factor":     profit_factor,
            "by_strategy":       by_strategy,
        }

    def get_execution_quality(self) -> dict:
        """يعيد ملخص جودة التنفيذ من execution_tracker."""
        from monitoring.execution_tracker import execution_tracker
        return execution_tracker.summary()

    @staticmethod
    def _calc_per_trade_sharpe(returns: list[float]) -> float:
        arr = np.array(returns)
        std = np.std(arr)
        if std == 0 or len(arr) < 2:
            return 0.0
        return float(np.mean(arr) / std)

    @staticmethod
    def _calc_annualized_sharpe(trades: list[dict]) -> float:
        if not trades:
            return 0.0

        daily: dict[str, float] = {}
        for t in trades:
            try:
                day        = t["timestamp"][:10]
                daily[day] = daily.get(day, 0.0) + t["pnl_pct"]
            except Exception:
                continue

        if len(daily) < 5:
            return 0.0

        try:
            dates      = sorted(daily.keys())
            start_date = datetime.strptime(dates[0],  "%Y-%m-%d")
            end_date   = datetime.strptime(dates[-1], "%Y-%m-%d")

            all_returns: list[float] = []
            current = start_date
            while current <= end_date:
                all_returns.append(
                    daily.get(current.strftime("%Y-%m-%d"), 0.0)
                )
                current += timedelta(days=1)
        except Exception:
            return 0.0

        arr = np.array(all_returns)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(252))


reporter = TradingReporter()