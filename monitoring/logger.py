import logging
import structlog
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import settings


LOG_MAX_BYTES  = 50 * 1024 * 1024   # 50 MB
LOG_BACKUP_CNT = 5

# Third-party loggers whose DEBUG/INFO transport chatter is noise in the
# trading log (httpcore prints every TCP/TLS step, ccxt is verbose, etc.).
_NOISY_LOGGERS = (
    "httpcore", "httpx", "ccxt", "urllib3",
    "websockets", "websockets.client", "websockets.server",
    "asyncio", "sqlalchemy", "aiosqlite",
)


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_CNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        format="%(message)s",
    )

    # FIX (log noise): keep third-party transport/debug chatter out of the
    # trading log regardless of the configured LOG_LEVEL.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)