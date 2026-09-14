import sys
from pathlib import Path

from loguru import logger

LOG_FORMAT = (
    "<level>{time:YYYY-MM-DDTHH:mm:ss} {extra[env]} {level} "
    "[f:{function}] [L:{line}] {message}</level>"
)


def init_logger(config):
    logger.remove()
    logger.configure(extra={"env": "dev"})

    logger.add(
        sys.stdout,
        level=config["log_level"],
        colorize=True,
        format=LOG_FORMAT,
    )

    log_path = Path(__file__).resolve().parent.parent / "log/debug"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level=config["log_level"],
        rotation="10 MB",
        retention=5,
        compression="zip",
        format=LOG_FORMAT,
    )

    return logger
