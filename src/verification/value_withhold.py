"""
Rule V: the value check, withhold only (results/round2/value_withhold_plan.md). CAMPUS_VERIFY_VALUE_WITHHOLD=1 (default
off). A layer on the final answer of _answer (src/api/main.py), after rule N.

Uses src/verification/value_check.find_conflicts unchanged (dates, clock times, percentages, "N out of M", durations,
places). SUPPORTED + a conflict -> INSUFFICIENT_EVIDENCE. Unlike CAMPUS_VERIFY_VALUE_CHECK (inside the frozen verifier,
rejected 2026-10-10 for a false CONTRADICTED), it never turns INSUFFICIENT_EVIDENCE into CONTRADICTED.
"""
from __future__ import annotations

import os
from typing import Optional

from src.verification.value_check import find_conflicts

ENV = "CAMPUS_VERIFY_VALUE_WITHHOLD"
VERSION = "value-withhold-v1"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def show(kind: str, value) -> str:
    """A value as a person would write it."""
    if kind == "time" and isinstance(value, tuple):
        h, m = value
        return f"{(h % 12) or 12}:{m:02d} {'PM' if h >= 12 else 'AM'}"
    if kind == "duration" and isinstance(value, tuple):
        n, unit = value
        return f"{n} {unit}{'' if n == 1 else 's'}"
    if kind == "percent":
        return f"{value:g}%"
    if kind == "count_of" and isinstance(value, tuple):
        return f"{value[0]} out of {value[1]}"
    if kind == "date" and isinstance(value, tuple) and len(value) == 2 and 1 <= value[0] <= 12:
        return f"{MONTHS[value[0] - 1]} {value[1]}"
    return str(value).title() if kind == "place" else str(value)


def conflict(message: str, premise: str, document: str) -> Optional[dict]:
    found = find_conflicts(message, premise or document, document or premise)
    if not found:
        return None
    kind, value, competing = found[0]
    return {"version": VERSION, "kind": kind, "message_value": show(kind, value),
            "post_values": [show(kind, v) for v in competing], "action": "withheld"}


def note(info: dict) -> str:
    return (f" The DYCI post gives {' / '.join(info['post_values'])}, not {info['message_value']}, so this is not "
            f"confirmed.")
