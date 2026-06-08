import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.regime_cache import all_regimes, get_regime_info

regimes = all_regimes()

if not regimes:
    print("Cache is empty — RegimeDetector has not run yet.")
    print("Wait for 3 candle cycles (45 minutes at 15m timeframe).")
else:
    for symbol, regime in regimes.items():
        info = get_regime_info(symbol)
        confirmed = info["confirmed"].value if hasattr(info["confirmed"], "value") else str(info["confirmed"])
        pending   = info["pending"].value   if hasattr(info["pending"],   "value") else str(info["pending"])
        print(
            f"{symbol:<12} confirmed={confirmed:<15} "
            f"pending={pending:<15} "
            f"count={info['count']}/{info['needed']}"
        )