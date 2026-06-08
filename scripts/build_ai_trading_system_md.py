"""Generate ai_trading_system.md: project tree + full user-written engine source."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "ai_trading_system.md"

# Third-party / runtime / generated — never bundle
EXCLUDE_DIRS = {
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "venv",
    ".venv",
    "dist",
    "build",
    ".next",
    "backups",
    "htmlcov",
    ".mypy_cache",
    "research",
    "logs",
}

# Auto-generated or external lockfiles (not user engine code)
EXCLUDE_FILES = {
    "ai_trading_system.md",
    "PROJECT_FULL_SNAPSHOT.md",
    "all_files.txt",
    "package-lock.json",
}

EXCLUDE_REL_PREFIXES = (
    "dashboard/frontend/dist/",
)

INCLUDE_EXT = {
    ".py",
    ".js",
    ".jsx",
    ".css",
    ".html",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".md",
    ".txt",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
}

SKIP_EXT = {
    ".pkl",
    ".pyc",
    ".db",
    ".nbc",
    ".nbi",
    ".png",
    ".jpg",
    ".ico",
    ".woff",
    ".map",
    ".lock",
}

MAX_FILE_BYTES = 500_000

# Human-readable roles for the structure section (Arabic + English)
LAYER_NOTES: dict[str, str] = {
    "main.py": "نقطة الدخول — EventBus + TradingEngine + إيقاف آمن",
    "test_ai.py": "سكربت تدريب/تقييم الذكاء الاصطناعي (تطوير)",
    "ai/": "تعلم الآلة: ميزات، XGBoost، تدريب، تنبؤ، triple-barrier",
    "analysis/": "مؤشرات فنية، نظام السوق (regime)، دفتر الأوامر",
    "backtesting/": "محاكاة تاريخية، مقاييس، walk-forward",
    "config/": "إعدادات Pydantic (بورصة، رموز، مخاطر، Telegram)",
    "core/": "المحرك: TradingEngine، EventBus، circuit breaker، shared state",
    "data/": "جمع بيانات REST/WebSocket + SQLite",
    "strategy/": "استراتيجيات التداول + StrategyManager",
    "risk/": "بوابة المخاطر، Kelly، وقف الخسارة، فلتر AI",
    "execution/": "محفظة، وسيط ورقي/testnet، مطابقة",
    "monitoring/": "سجلات، مقاييس، Telegram، صحة النظام",
    "dashboard/": "FastAPI + React لوحة المراقبة",
    "scripts/": "تشغيل وصيانة (نسخ احتياطي، إعادة تدريب، health)",
    "tests/": "اختبارات pytest (وحدة + تكامل)",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    if "node_modules" in rel:
        return False
    if any(rel.startswith(p) for p in EXCLUDE_REL_PREFIXES):
        return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.name.startswith(".") and path.name not in {".dockerignore"}:
        return False
    ext = path.suffix.lower()
    if ext in SKIP_EXT:
        return False
    if path.name in {"Dockerfile", ".dockerignore"}:
        return True
    if ext not in INCLUDE_EXT:
        return False
    if path.stat().st_size > MAX_FILE_BYTES:
        return False
    return True


def lang_for_ext(ext: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".css": "css",
        ".html": "html",
        ".sql": "sql",
        ".sh": "bash",
        ".bat": "batch",
        ".ps1": "powershell",
        ".md": "markdown",
        ".txt": "text",
    }.get(ext, "text")


def collect_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                if should_include(p):
                    files.append(p)
            except OSError:
                pass
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def build_tree_lines() -> list[str]:
    """ASCII tree of user-written project layout (no venv/node_modules/logs)."""
    lines: list[str] = [
        "## 1. هيكل المشروع الكامل",
        "",
        "نظام تداول عملات مشفرة بالذكاء الاصطناعي — معمارية **event-driven** غير متزامنة.",
        "المحرك (`TradingEngine`) يربط جمع البيانات → الاستراتيجيات → المخاطر → التنفيذ → المراقبة.",
        "",
        "### العلاقات بين الطبقات",
        "",
        "```",
        "main.py",
        "   └── core/engine.py (TradingEngine)",
        "           ├── core/events.py (EventBus — pub/sub غير متزامن)",
        "           ├── data/collectors/ → core/locks/shared_state.py",
        "           │       └── data/storage/database.py (SQLite)",
        "           ├── analysis/regime.py + regime_cache.py",
        "           ├── strategy/* → strategy/manager.py → إشارات SignalEvent",
        "           ├── ai/predictor.py + ai/trainer.py (فلتر/تدريب XGBoost)",
        "           ├── risk/manager.py (موافقة/رفض + Kelly + regime + AI)",
        "           ├── execution/paper_broker | testnet_broker",
        "           │       └── execution/portfolio.py + reconciler.py",
        "           ├── monitoring/* (Telegram, health, metrics, reports)",
        "           └── core/circuit_breaker/breaker.py",
        "",
        "dashboard/backend/server.py ──قراءة──► trading.db",
        "dashboard/frontend/src/ ──WebSocket/REST──► server.py",
        "```",
        "",
        "### شجرة المجلدات والملفات",
        "",
        "```",
        "Bot Ai Agent/",
        "├── main.py                    ← تشغيل المحرك",
        "├── test_ai.py                 ← تجارب تدريب AI",
        "├── requirements.txt",
        "├── pytest.ini",
        "├── Dockerfile",
        "├── docker-compose.yml",
        "├── .dockerignore",
        "├── .streamlit/config.toml",
        "│",
        "├── ai/                        ← طبقة التعلم الآلي",
        "│   ├── features.py            ميزات RSI/EMA/BB/ATR/MACD",
        "│   ├── model.py               تدريب XGBoost + scaler + PSI",
        "│   ├── predictor.py           استدلال وقت التشغيل",
        "│   ├── trainer.py             إعادة تدريب مجدولة",
        "│   ├── labeling/triple_barrier.py",
        "│   ├── training/pipeline.py",
        "│   ├── training/cross_validation.py  (stub)",
        "│   └── models/metadata.json",
        "│",
        "├── analysis/                  ← تحليل السوق",
        "│   ├── indicators.py",
        "│   ├── regime.py + regime_cache.py",
        "│   ├── orderbook.py",
        "│   ├── backtester.py",
        "│   └── optimizer.py",
        "│",
        "├── backtesting/               ← اختبار تاريخي",
        "│   ├── engine.py, metrics.py, optimizer.py",
        "│   ├── walk_forward.py, report.py",
        "│",
        "├── config/settings.py         ← إعدادات التطبيق",
        "│",
        "├── core/                      ← قلب المحرك",
        "│   ├── engine.py              TradingEngine",
        "│   ├── events.py              EventBus + أنواع الأحداث",
        "│   ├── exceptions.py",
        "│   ├── circuit_breaker/breaker.py",
        "│   └── locks/shared_state.py",
        "│",
        "├── data/                      ← بيانات",
        "│   ├── collectors/            REST, WebSocket, order book",
        "│   ├── quality.py",
        "│   └── storage/               models.py + database.py",
        "│",
        "├── strategy/                  ← إشارات التداول",
        "│   ├── base.py, manager.py",
        "│   ├── mean_reversion.py, momentum.py",
        "│   ├── trend_following.py, vwap_reversion.py",
        "│   └── ai_strategy.py         (stub)",
        "│",
        "├── risk/                      ← إدارة المخاطر",
        "│   ├── manager.py             بوابة مركزية + فلتر AI",
        "│   ├── kelly.py, stop_loss.py",
        "│   └── position_sizer.py      (stub)",
        "│",
        "├── execution/                 ← تنفيذ ومحفظة",
        "│   ├── order.py, portfolio.py",
        "│   ├── paper_broker.py, testnet_broker.py",
        "│   └── reconciler.py",
        "│",
        "├── monitoring/                ← مراقبة وتنبيهات",
        "│   ├── logger.py, metrics.py, alerts.py",
        "│   ├── health.py, reporter.py",
        "│   └── execution_tracker.py",
        "│",
        "├── scripts/                   ← أدوات تشغيل",
        "│   ├── health_check.py, retrain_now.py, backup_db.py",
        "│   └── build_ai_trading_system_md.py",
        "│",
        "├── tests/                     ← pytest",
        "│   ├── unit/, integration/",
        "│   └── conftest.py",
        "│",
        "└── dashboard/                 ← واجهة المستخدم",
        "    ├── start.py",
        "    ├── backend/server.py      FastAPI + WebSocket",
        "    └── frontend/src/          React (Overview, Signals, Orders, Performance)",
        "",
        "[مستبعد من هذا المستند — ليس كود المحرك]",
        "  venv/ · node_modules/ · logs/ · *.pkl · *.db · .env",
        "  dashboard/frontend/dist/ · package-lock.json",
        "  ai_trading_system.md · PROJECT_FULL_SNAPSHOT.md · research/",
        "```",
        "",
        "### وصف الطبقات",
        "",
    ]
    for key, note in LAYER_NOTES.items():
        lines.append(f"- **`{key}`** — {note}")
    lines.extend(["", "---", ""])
    return lines


def main() -> None:
    files = collect_files()
    lines: list[str] = [
        "# AI Trading System — هيكل المشروع + كود المحرك الكامل",
        "",
        f"> **تاريخ التوليد:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "> **المحتوى:** هيكل المشروع ثم **100%** من ملفات الكود التي كتبها المطور (محرك النظام).",
        "> **مستبعد:** مكتبات خارجية (`venv/`, `node_modules/`)، مخرجات مولّدة (`dist/`, snapshots)، أسرار (`.env`)، ثنائيات (`*.pkl`, `*.db`).",
        "",
        "---",
        "",
    ]
    lines.extend(build_tree_lines())
    lines.extend(
        [
            "## 2. محتوى ملفات المحرك (كامل بدون اختصار)",
            "",
            f"**عدد الملفات:** {len(files)}",
            "",
            "---",
            "",
        ]
    )

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        ext = path.suffix.lower()
        lang = lang_for_ext(ext) if ext else "text"
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")

        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append(content.rstrip("\n"))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({len(files)} files, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
