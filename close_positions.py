# close_positions.py
"""
سكريبت مؤقت لإغلاق المراكز القديمة في Binance Testnet.
يُحذف بعد الاستخدام.
"""
import asyncio
import math
import ccxt.async_support as ccxt
from config.settings import settings

MIN_VALUE_USD = 1.0


async def close_all_positions():
    exchange = ccxt.binance({
        "apiKey":          settings.TESTNET_API_KEY,
        "secret":          settings.TESTNET_API_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "spot"},
    })
    exchange.set_sandbox_mode(True)

    try:
        print("جاري تحميل Markets...")
        await exchange.load_markets()

        print("جاري جلب الرصيد...")
        balance    = await exchange.fetch_balance()
        free_coins = {
            k: float(v)
            for k, v in balance.get("free", {}).items()
            if v and float(v) > 0 and k != "USDT"
        }

        target_coins = {"BTC", "ETH", "SOL"}
        sold_any     = False

        for coin in target_coins:
            qty = free_coins.get(coin, 0.0)
            if qty <= 0:
                print(f"{coin}: لا يوجد رصيد — تخطي")
                continue

            symbol = f"{coin}/USDT"
            market = exchange.markets.get(symbol, {})

            # جلب السعر الحالي
            try:
                ticker        = await exchange.fetch_ticker(symbol)
                current_price = float(ticker.get("last", 0))
            except Exception:
                current_price = 0.0

            value_usd = qty * current_price
            print(f"{coin}: {qty:.8f} @ ${current_price:,.2f} = ${value_usd:.2f}")

            if value_usd < MIN_VALUE_USD:
                print(f"  → قيمة ضئيلة جداً ({value_usd:.4f}$) — تخطي")
                continue

            # حساب step_size والحد الأدنى
            limits    = market.get("limits", {})
            amount    = limits.get("amount", {})
            min_qty   = float(amount.get("min", 0) or 0)
            step_size = float(amount.get("step", 0) or 0)

            # تقريب الكمية
            if step_size > 0:
                qty_rounded = math.floor(qty / step_size) * step_size
                # حساب عدد المنازل العشرية من step_size
                decimal_places = max(0, int(round(-math.log10(step_size))))
                qty_rounded    = round(qty_rounded, decimal_places)
            else:
                # fallback: 6 منازل عشرية
                qty_rounded = math.floor(qty * 1_000_000) / 1_000_000

            if qty_rounded <= 0 or qty_rounded < min_qty:
                print(f"  → الكمية {qty_rounded} أقل من الحد الأدنى {min_qty} — تخطي")
                continue

            print(f"  → جاري بيع {qty_rounded} {coin}...")
            try:
                order        = await exchange.create_order(
                    symbol   = symbol.replace("/", ""),
                    type     = "market",
                    side     = "sell",
                    amount   = qty_rounded,
                )
                filled_price = float(order.get("average") or order.get("price") or 0)
                filled_qty   = float(order.get("filled") or qty_rounded)
                print(f"  ✅ {coin}: بيع {filled_qty} @ ${filled_price:,.4f}")
                sold_any = True

            except Exception as e:
                print(f"  ❌ {coin}: فشل البيع — {e}")

        if not sold_any:
            print("\n⚠️  لم يُباع شيء")

        # الرصيد النهائي
        print("\nجاري التحقق من الرصيد النهائي...")
        balance_after = await exchange.fetch_balance()
        usdt          = float(balance_after.get("free", {}).get("USDT", 0))
        print(f"الرصيد النهائي: {usdt:,.2f} USDT")

        remaining = {
            k: float(balance_after.get("free", {}).get(k, 0))
            for k in target_coins
            if float(balance_after.get("free", {}).get(k, 0) or 0) > 0
        }
        if remaining:
            print(f"\n⚠️  تبقى dust: {remaining} — ستُتجاهل تلقائياً")
        else:
            print("\n✅ Exchange نظيف تماماً")

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(close_all_positions())