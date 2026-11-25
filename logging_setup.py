import logging
import os
import shutil
from logging.handlers import BaseRotatingHandler
from pathlib import Path


class CopyTruncateRotatingFileHandler(BaseRotatingHandler):
    """Rotate a bind-mounted log by copying it and truncating the original.

    The log file is mounted into the container as a single file, so renaming
    it (what RotatingFileHandler does) fails with "Device or resource busy".
    """

    def __init__(
        self,
        filename: str | Path,
        *,
        max_bytes: int = 2 * 1024 * 1024,
        backup_count: int = 3,
        encoding: str = "utf-8",
    ):
        super().__init__(filename, mode="a", encoding=encoding)
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.max_bytes <= 0:
            return False
        if self.stream is None:
            self.stream = self._open()
        return self.stream.tell() >= self.max_bytes

    def doRollover(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.backup_count > 0:
            for index in range(self.backup_count - 1, 0, -1):
                source = f"{self.baseFilename}.{index}"
                target = f"{self.baseFilename}.{index + 1}"
                if os.path.exists(source):
                    os.replace(source, target)
            shutil.copyfile(self.baseFilename, f"{self.baseFilename}.1")
            # Truncate in place: the file itself must keep the same inode for
            # the bind mount on the host.
            with open(self.baseFilename, "w", encoding=self.encoding):
                pass
        self.stream = self._open()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    rotating_file = CopyTruncateRotatingFileHandler(
        log_path, max_bytes=2 * 1024 * 1024, backup_count=3
    )
    rotating_file.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, rotating_file], force=True)
