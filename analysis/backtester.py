import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from analysis.indicators import add_all_indicators


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    entry_idx: int = 0
    exit_idx: int = 0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    trades: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round(self.winning_trades / self.total_trades * 100, 1)

    @property
    def total_return_pct(self) -> float:
        return round(self.total_pnl / 10000 * 100, 2)

    def sharpe_ratio(self) -> float:
        if not self.trades:
            return 0.0
        pnls = [t.pnl for t in self.trades]
        if np.std(pnls) == 0:
            return 0.0
        return round(np.mean(pnls) / np.std(pnls) * np.sqrt(252), 2)

    def print_report(self) -> None:
        print(f"\n{'='*45}")
        print(f"  Backtest: {self.strategy_name} | {self.symbol}")
        print(f"{'='*45}")
        print(f"  إجمالي الصفقات : {self.total_trades}")
        print(f"  صفقات رابحة    : {self.winning_trades}")
        print(f"  Win Rate       : {self.win_rate}%")
        print(f"  إجمالي الربح   : {self.total_pnl:.2f} USDT")
        print(f"  العائد الكلي   : {self.total_return_pct}%")
        print(f"  Max Drawdown   : {self.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio   : {self.sharpe_ratio()}")
        print(f"{'='*45}")


class Backtester:
    def __init__(self, initial_capital: float = 10000.0,
                 position_pct: float = 0.05,
                 commission: float = 0.001):
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.commission = commission

    def run(self, strategy_name: str, symbol: str,
            df: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        result = BacktestResult(strategy_name, symbol)
        capital = self.initial_capital
        position = None
        peak_capital = capital
        max_dd = 0.0

        for i in range(len(df)):
            price = float(df["close"].iloc[i])
            signal = signals.iloc[i]["side"] if i < len(signals) else None

            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital * 100
            if dd > max_dd:
                max_dd = dd

            if signal == "buy" and position is None:
                qty = (capital * self.position_pct) / price
                cost = qty * price * (1 + self.commission)
                if cost <= capital:
                    position = Trade(symbol, "buy", price,
                                     quantity=qty, entry_idx=i)
                    capital -= cost

            elif signal == "sell" and position is not None:
                proceeds = position.quantity * price * (1 - self.commission)
                pnl = proceeds - (position.quantity * position.entry_price)
                position.exit_price = price
                position.pnl = pnl
                position.exit_idx = i
                capital += proceeds

                result.trades.append(position)
                result.total_trades += 1
                result.total_pnl += pnl
                if pnl > 0:
                    result.winning_trades += 1
                position = None

        result.max_drawdown = round(max_dd, 2)
        return result


def generate_signals_momentum(df: pd.DataFrame,
                               rsi_oversold: float = 30,
                               rsi_overbought: float = 70,
                               ema_period: int = 50) -> pd.DataFrame:
    df = add_all_indicators(df)
    signals = []

    for i in range(1, len(df)):
        rsi_prev = df["rsi"].iloc[i - 1]
        rsi_curr = df["rsi"].iloc[i]
        price    = df["close"].iloc[i]
        ema_val  = df[f"ema_{ema_period}"].iloc[i]

        if rsi_prev < rsi_oversold and rsi_curr >= rsi_oversold and price > ema_val:
            signals.append({"side": "buy"})
        elif rsi_prev > rsi_overbought and rsi_curr <= rsi_overbought and price < ema_val:
            signals.append({"side": "sell"})
        else:
            signals.append({"side": None})

    signals.insert(0, {"side": None})
    return pd.DataFrame(signals)


def generate_signals_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    df = add_all_indicators(df)
    signals = []

    for i in range(len(df)):
        price  = df["close"].iloc[i]
        upper  = df["bb_upper"].iloc[i]
        lower  = df["bb_lower"].iloc[i]
        rsi_v  = df["rsi"].iloc[i]

        if pd.isna(upper):
            signals.append({"side": None})
        elif price <= lower and rsi_v < 40:
            signals.append({"side": "buy"})
        elif price >= upper and rsi_v > 60:
            signals.append({"side": "sell"})
        else:
            signals.append({"side": None})

    return pd.DataFrame(signals)


def walk_forward_test(strategy_name: str,
                      df: pd.DataFrame,
                      signal_func,
                      train_pct: float = 0.7,
                      initial_capital: float = 10000.0) -> None:
    split      = int(len(df) * train_pct)
    train_df   = df.iloc[:split]
    test_df    = df.iloc[split:]

    bt = Backtester(initial_capital=initial_capital)

    train_signals = signal_func(train_df)
    train_result  = bt.run(f"{strategy_name} [Train]",
                           "BTC/USDT", train_df, train_signals)

    test_signals = signal_func(test_df)
    test_result  = bt.run(f"{strategy_name} [Test]",
                          "BTC/USDT", test_df, test_signals)

    print(f"\n{'='*55}")
    print(f"  Walk-Forward: {strategy_name}")
    print(f"  Train: {len(train_df)} شمعة | Test: {len(test_df)} شمعة")
    print(f"{'='*55}")
    print(f"\n  [TRAIN] Win Rate: {train_result.win_rate}% | "
          f"P&L: ${train_result.total_pnl:.2f} | "
          f"Drawdown: {train_result.max_drawdown}%")
    print(f"  [TEST]  Win Rate: {test_result.win_rate}% | "
          f"P&L: ${test_result.total_pnl:.2f} | "
          f"Drawdown: {test_result.max_drawdown}%")

    if test_result.win_rate >= train_result.win_rate * 0.8:
        print(f"\n  ✅ الاستراتيجية مستقرة")
    else:
        print(f"\n  ⚠️  تحذير — overfitting محتمل")

    print(f"{'='*55}")