import logging
from pathlib import Path

from logging_setup import CopyTruncateRotatingFileHandler


def test_rollover_keeps_the_inode_and_writes_a_backup(tmp_path: Path) -> None:
    path = tmp_path / "agent.log"
    handler = CopyTruncateRotatingFileHandler(
        path, max_bytes=200, backup_count=1
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "x" * 100, None, None)
    inode_before = None

    for _ in range(6):
        handler.emit(record)
        if inode_before is None and (tmp_path / "agent.log.1").exists():
            inode_before = path.stat().st_ino

    assert (tmp_path / "agent.log.1").exists()
    assert path.stat().st_ino == inode_before
    handler.close()
