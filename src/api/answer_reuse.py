"""
Answer reuse for identical messages (CAMPUS_VERIFY_ANSWER_REUSE, default off; owner request 2026-11-16).

In a class demo many students type the same rumor. The checker gives the same answer to the same message on the same day
against the same posts (identity checks: 0 of 156 differ between runs since v2.3.1), so the first answer is kept and
handed to everyone else who sends exactly that message:
- a message already answered today is returned at once, without taking a checking slot;
- a message still being checked makes the others wait for that one check instead of each queueing their own.

What makes two messages "the same": the exact text as typed (no trimming or case folding), the calendar date the date
rules use (src/verification/post_age.today), and the loaded collection (every reload builds a new pipeline, which
empties this store). Only /verify/text answers are kept; links and pictures are never reused.

Not kept: a Tagalog/Taglish answer made while the translator did not answer (it would stay "evidence only" for the rest
of the day after the translator comes back), and anything that raised (busy, too short). Nothing is written to disk;
the store holds at most MAX_ENTRIES answers and is lost on restart.
"""
import os
import threading
from collections import OrderedDict
from typing import Callable, Optional

ENV = "CAMPUS_VERIFY_ANSWER_REUSE"
MAX_ENTRIES = 500
WAIT_SECONDS = 180                     # a waiting request gives up on the first check after this and checks on its own


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


class AnswerStore:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._answers: "OrderedDict[tuple, object]" = OrderedDict()
        self._running = {}                                   # key -> threading.Event, set when that check finishes
        self._owner = None

    def _for(self, owner) -> None:
        """Empty the store when the collection changed (called with the lock held)."""
        if owner is not self._owner:
            self._answers.clear()
            self._owner = owner

    def answer(self, owner, key: tuple, compute: Callable[[], object], keep: Callable[[object], bool],
               copy: Callable[[object], object]):
        """(answer, reused). `compute` runs at most once at a time per key; `keep` decides whether its answer is
        stored; `copy` gives each caller its own object."""
        while True:
            with self._lock:
                self._for(owner)
                if key in self._answers:
                    self._answers.move_to_end(key)
                    return copy(self._answers[key]), True
                running = self._running.get(key)
                if running is None:
                    done = self._running[key] = threading.Event()
                    break
            if running.wait(WAIT_SECONDS):
                with self._lock:
                    if key in self._answers or key in self._running:
                        continue                             # kept (or another check started): look again
            return compute(), False                          # the first check failed, was not kept, or took too long
        try:
            result = compute()
            if keep(result):
                with self._lock:
                    if owner is self._owner:                 # not stored if the collection reloaded meanwhile
                        self._answers[key] = copy(result)
                        while len(self._answers) > self.max_entries:
                            self._answers.popitem(last=False)
            return result, False
        finally:
            with self._lock:
                self._running.pop(key, None)
            done.set()

    def size(self) -> int:
        with self._lock:
            return len(self._answers)


def key_for(claim: str, day) -> tuple:
    return (claim, day.isoformat())


def worth_keeping(claim: str, result) -> bool:
    """False for an answer made while the translator was expected but did not answer."""
    from src.api.verdict_policy import withhold_for_language
    from src.normalization import tagalog_translate
    if not (tagalog_translate.enabled() and withhold_for_language(claim)):
        return True
    return getattr(result, "tagalog_translation", None) is not None or getattr(result, "tagalog_reading", None) is not None
