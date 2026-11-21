"""Server time-zone check (src/api/clock_check.py; scale plan S7)."""
import logging
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.api import clock_check


def at(hours, minutes=0):
    return datetime(2026, 11, 4, 7, 0, tzinfo=timezone(timedelta(hours=hours, minutes=minutes)))


def test_philippine_time_gives_no_warning():
    assert clock_check.timezone_warning(at(8)) is None


def test_utc_and_other_offsets_are_warned_with_the_fix():
    utc = clock_check.timezone_warning(at(0))
    assert "UTC+00:00" in utc and "up to 8 hours" in utc and "TZ=Asia/Manila" in utc
    assert "UTC-05:00" in clock_check.timezone_warning(at(-5)) and "13 hours" in clock_check.timezone_warning(at(-5))
    assert "UTC+05:30" in clock_check.timezone_warning(at(5, 30))


def test_the_server_logs_the_warning_at_startup_and_still_answers(monkeypatch, caplog):
    from src.api import main
    monkeypatch.setattr(clock_check, "timezone_warning", lambda now=None: "Server clock is UTC+00:00 (test)")
    with caplog.at_level(logging.WARNING, logger="campus_verify"), TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
    assert "Server clock is UTC+00:00 (test)" in caplog.text


def test_no_warning_is_logged_on_philippine_time(monkeypatch, caplog):
    from src.api import main
    monkeypatch.setattr(clock_check, "timezone_warning", lambda now=None: None)
    with caplog.at_level(logging.WARNING, logger="campus_verify"), TestClient(main.app):
        pass
    assert "Server clock" not in caplog.text
