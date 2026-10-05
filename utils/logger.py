"""
utils/logger.py
Logger con colores para consola y rotación de archivos.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/system.log", level: str = "INFO") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Consola con colores ANSI
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)

    RESET  = "\033[0m"
    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[1;31m", # bold red
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            color = COLORS.get(record.levelname, RESET)
            record.levelname = f"{color}{record.levelname:<8}{RESET}"
            return super().format(record)

    console.setFormatter(ColorFormatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))

    # ── Archivo con rotación (10 MB, 5 backups)
    try:
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
    except OSError:
        pass  # Si no se puede escribir al disco, solo usamos consola

    logger.addHandler(console)
    return logger
