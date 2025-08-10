import logging
import logging.config
import os
from typing import Literal

from .base_setup import PROJECT_ROOT

'''
import logging
from core import setup_logging
setup_logging()
log = logging.getLogger(__name__)
'''

class LogsFormatter(logging.Formatter):
    def __init__(
        self,
        fmt=None,
        datefmt=None,
        style: Literal["%", "{", "$"] = "%",
        validate=True,  # noqa: FBT002
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        base_record = logging.LogRecord(
            name="",
            level=0,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )
        self._base_attrs = set(base_record.__dict__.keys())

        self._base_attrs.add("message")
        if self.usesTime():
            self._base_attrs.add("asctime")

    def format(self, record) -> str:
        message = super().format(record)
        extra_attrs = {
            k: v for k, v in record.__dict__.items()
            if k not in self._base_attrs and not k.startswith("_")
        }  # fmt: skip

        if extra_attrs:
            extra_str = ", ".join(f"{k}={v}" for k, v in extra_attrs.items())
            message = f"{message} [{extra_str}]"
        return message


def setup_logging() -> None:
    """
    Setup logger, call once for entrypoint.
    """
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "breacher.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": LogsFormatter,
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_file),
                "maxBytes": 1_000_000,  # 1MB?
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": os.getenv("APP_LOG_LEVEL", "DEBUG"),
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)
