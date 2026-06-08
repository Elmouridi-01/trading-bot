import logging
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
logging.getLogger("aiosqlite").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

import asyncio
import signal
import sys
from core.events import EventBus
from core.engine import TradingEngine
from config.settings import settings


async def main() -> int:
    bus    = EventBus()
    engine = TradingEngine(bus, settings)

    loop           = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler(sig_name: str):
        print(f"\n[Main] 📡 استُقبلت إشارة {sig_name} — جاري الإيقاف الآمن...")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig.name)
        except (NotImplementedError, OSError):
            pass

    exit_code = 0
    try:
        engine_task   = asyncio.create_task(engine.run(),            name="engine")
        shutdown_task = asyncio.create_task(shutdown_event.wait(),   name="shutdown")

        done, pending = await asyncio.wait(
            [engine_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if engine_task in done:
            exc = engine_task.exception()
            if exc:
                print(f"[Main] 💥 خطأ في الـ engine: {exc}")
                exit_code = 1

    except Exception as e:
        print(f"[Main] 💥 خطأ غير متوقع: {e}")
        exit_code = 1
    finally:
        print("[Main] 🔄 تنظيف الـ resources...")
        try:
            await asyncio.wait_for(engine.stop(), timeout=15.0)
        except asyncio.TimeoutError:
            print("[Main] ⚠️ timeout في engine.stop() — إيقاف قسري")
        except Exception as e:
            print(f"[Main] ⚠️ خطأ في engine.stop(): {e}")

    print(f"[Main] ✅ النظام أُوقف — exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — خروج")
        sys.exit(0)