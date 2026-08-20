import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("optexity")
except PackageNotFoundError:
    __version__ = "0.0.0"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"

class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[2m",  # dim
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }

    def format(self, record):
        line = super().format(record)
        colour = self.COLORS.get(record.levelno)
        return f"{colour}{line}\033[0m" if colour else line


# basicConfig only sets its formatter on handlers that lack one, so this
# leaves the file handler plain.
_stream_handler = logging.StreamHandler(sys.stdout)
if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
    _stream_handler.setFormatter(_ColorFormatter(LOG_FORMAT))

logging.basicConfig(
    level=logging.WARNING,  # Default level for root logger
    format=LOG_FORMAT,
    handlers=[
        _stream_handler,
        logging.FileHandler(Path("/tmp/optexity.log")),
    ],
)
current_module = __name__.split(".")[0]  # top-level module/package
logging.getLogger(current_module).setLevel(logging.DEBUG)
