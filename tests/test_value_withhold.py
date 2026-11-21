"""Rule V, the value check, withhold only (results/round2/value_withhold_plan.md)."""
import inspect

import pytest

from src.verification import value_withhold as vw

POST = ("DYCI ADVISORY 2 October 2030 The Foundation Day program tomorrow, October 3, 2030, starts at 8:30 AM. The "
        "intramurals will run for 3 days. A 20% discount on miscellaneous fees applies. The tour goes to Baguio.")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(vw.ENV, raising=False)
    assert vw.enabled() is False


@pytest.mark.parametrize("kind,value,shown", [
    ("time", (13, 0), "1:00 PM"), ("time", (8, 30), "8:30 AM"), ("time", (0, 15), "12:15 AM"), ("time", (12, 0), "12:00 PM"),
    ("duration", (1, "day"), "1 day"), ("duration", (3, "week"), "3 weeks"), ("percent", 20.0, "20%"),
    ("count_of", (3, 5), "3 out of 5"), ("date", (11, 12), "November 12"), ("place", "baguio", "Baguio"),
])
def test_show(kind, value, shown):
    assert vw.show(kind, value) == shown


@pytest.mark.parametrize("message,kind,mine,theirs", [
    ("foundation day program on october 3 at 1pm", "time", "1:00 PM", ["8:30 AM"]),
    ("intramurals run for 5 days", "duration", "5 days", ["3 days"]),
    ("50 percent discount on misc fees", "percent", "50%", ["20%"]),
])
def test_conflicts(message, kind, mine, theirs):
    c = vw.conflict(message, POST, POST)
    assert (c["kind"], c["message_value"], c["post_values"], c["action"]) == (kind, mine, theirs, "withheld")
    assert mine in vw.note(c) and theirs[0] in vw.note(c)


@pytest.mark.parametrize("message", [
    "foundation day program on october 3 at 8:30am", "program on october 3", "intramurals for three days",
    "20% discount on fees", "tour to baguio",
])
def test_no_conflict(message):
    assert vw.conflict(message, POST, POST) is None


def test_layer_only_withholds_supported():
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert 'value_withhold.enabled() and model.verdict == "SUPPORTED"' in src
    assert '"verdict": "INSUFFICIENT_EVIDENCE", "value_withhold": found' in src
    assert src.index("unit_check.enabled()") < src.index("value_withhold.enabled()") < src.index("reason_parts.enabled()")
