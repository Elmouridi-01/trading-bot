import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from data.collectors.rest_collector import CryptoRestCollector
from core.events import EventBus
from analysis.regime import RegimeDetector, MarketRegime


async def main():
    bus       = EventBus()
    collector = CryptoRestCollector(bus)

    print("جاري جلب البيانات...")
    raw = await collector.exchange.fetch_ohlcv(
        "BTC/USDT", "15m", limit=1000
    )
    await collector.exchange.close()

    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    detector = RegimeDetector()

    # توزيع الـ Regimes
    summary = detector.summary(df)
    print(f"\nتوزيع حالات السوق (آخر 1000 شمعة 15m):")
    for regime, pct in summary.items():
        bar = "█" * int(pct / 2)
        print(f"  {regime:<20} {pct:>5.1f}%  {bar}")

    # الحالة الحالية
    current = detector.current(df)
    print(f"\nالحالة الحالية: {current.value.upper()}")

    # أداء كل Regime
    print(f"\nمتوسط الحركة في كل حالة:")
    regimes   = detector.detect(df)
    df["regime"]  = regimes.apply(lambda x: x.value)
    df["returns"] = df["close"].pct_change(4) * 100

    for regime in MarketRegime:
        mask = df["regime"] == regime.value
        if mask.sum() > 10:
            avg_ret = df.loc[mask, "returns"].mean()
            std_ret = df.loc[mask, "returns"].std()
            count   = mask.sum()
            print(f"  {regime.value:<20} متوسط: {avg_ret:+.3f}% | "
                  f"تذبذب: {std_ret:.3f}% | عدد: {count}")


asyncio.run(main())