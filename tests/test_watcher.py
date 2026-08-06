"""DB-free unit tests for the card file watcher handler."""

import logging

from app.watcher import CardFileHandler


def _handler(tmp_path):
    return CardFileHandler(session_factory=lambda: None, watched_dir=tmp_path)


def test_handle_change_missing_file_does_not_raise(caplog, tmp_path):
    """A file that vanishes between the event and stat() must not crash the handler."""
    handler = _handler(tmp_path)
    missing = str(tmp_path / "missing.json")
    with caplog.at_level(logging.WARNING, logger="app.watcher"):
        handler._handle_change(missing)# must return normally, no exception
    assert any(
        "Failed to stat" in record.getMessage() for record in caplog.records
    )


def test_handle_change_large_file_skipped(caplog, tmp_path):
    """Files larger than the 50 KB cap are still skipped with a warning."""
    handler = _handler(tmp_path)
    large = tmp_path / "large.json"
    large.write_bytes(b"x" * (50 * 1024 + 1))
    with caplog.at_level(logging.WARNING, logger="app.watcher"):
        handler._handle_change(str(large))
    assert any(
        "Skipping large file" in record.getMessage() for record in caplog.records
    )
