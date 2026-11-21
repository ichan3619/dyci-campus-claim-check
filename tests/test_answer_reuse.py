"""
Answer reuse for identical messages (src/api/answer_reuse.py, CAMPUS_VERIFY_ANSWER_REUSE, default off).
"""
import datetime as dt
import os
import threading
import time

os.environ["CAMPUS_VERIFY_CLASSIFIER"] = "tfidf_lr"

import pytest
from fastapi.testclient import TestClient

from src.api import answer_reuse
from src.api.answer_reuse import AnswerStore

DAY = dt.date(2026, 11, 16)
SAME = lambda r: r
KEEP = lambda r: True


# ---- the store --------------------------------------------------------------------------------------------------------
def test_second_identical_message_is_reused_and_a_different_one_is_not():
    store, calls = AnswerStore(), []
    compute = lambda text: (lambda: calls.append(text) or f"answer to {text}")
    assert store.answer("posts", ("a", "2026-11-16"), compute("a"), KEEP, SAME) == ("answer to a", False)
    assert store.answer("posts", ("a", "2026-11-16"), compute("a"), KEEP, SAME) == ("answer to a", True)
    assert store.answer("posts", ("b", "2026-11-16"), compute("b"), KEEP, SAME) == ("answer to b", False)
    assert store.answer("posts", ("a", "2026-11-17"), compute("a"), KEEP, SAME)[1] is False    # another day
    assert calls == ["a", "b", "a"]


def test_a_new_collection_empties_the_store():
    store = AnswerStore()
    store.answer("posts v1", ("a", "d"), lambda: 1, KEEP, SAME)
    assert store.answer("posts v2", ("a", "d"), lambda: 2, KEEP, SAME) == (2, False)
    assert store.size() == 1


def test_answers_not_worth_keeping_are_checked_again():
    store = AnswerStore()
    store.answer("p", ("a", "d"), lambda: "no translator", lambda r: False, SAME)
    assert store.answer("p", ("a", "d"), lambda: "translated", KEEP, SAME) == ("translated", False)


def test_store_is_bounded():
    store = AnswerStore(max_entries=3)
    for i in range(10):
        store.answer("p", (str(i), "d"), lambda: i, KEEP, SAME)
    assert store.size() == 3
    assert store.answer("p", ("9", "d"), lambda: "again", KEEP, SAME) == (9, True)
    assert store.answer("p", ("0", "d"), lambda: "again", KEEP, SAME) == ("again", False)


def test_each_caller_gets_its_own_copy():
    store = AnswerStore()
    store.answer("p", ("a", "d"), lambda: {"v": 1}, KEEP, dict)
    got, _ = store.answer("p", ("a", "d"), lambda: None, KEEP, dict)
    got["v"] = 2
    assert store.answer("p", ("a", "d"), lambda: None, KEEP, dict)[0] == {"v": 1}


def test_identical_messages_at_the_same_moment_run_one_check():
    store, calls, results = AnswerStore(), [], []
    barrier = threading.Barrier(8)

    def slow():
        calls.append(1)
        time.sleep(0.3)
        return "answer"

    def one():
        barrier.wait()
        results.append(store.answer("p", ("a", "d"), slow, KEEP, SAME))

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert sorted(r[1] for r in results) == [False] + [True] * 7
    assert {r[0] for r in results} == {"answer"}


def test_when_the_first_check_fails_the_others_check_on_their_own():
    store, calls = AnswerStore(), []
    started, results = threading.Event(), []

    def failing():
        calls.append("first")
        started.set()
        time.sleep(0.2)
        raise RuntimeError("busy")

    def first():
        with pytest.raises(RuntimeError):
            store.answer("p", ("a", "d"), failing, KEEP, SAME)

    t = threading.Thread(target=first)
    t.start()
    started.wait(2)
    results.append(store.answer("p", ("a", "d"), lambda: calls.append("second") or "ok", KEEP, SAME))
    t.join()
    assert calls == ["first", "second"] and results == [("ok", False)]


def test_worth_keeping_refuses_a_tagalog_answer_made_without_the_translator(monkeypatch):
    class R:
        tagalog_translation = None
        tagalog_reading = None
    tagalog = "May pasok ba bukas sa DYCI?"
    monkeypatch.setenv("CAMPUS_VERIFY_TAGALOG_TRANSLATE", "1")
    assert not answer_reuse.worth_keeping(tagalog, R())
    R.tagalog_translation = {"accepted": False}
    assert answer_reuse.worth_keeping(tagalog, R())
    assert answer_reuse.worth_keeping("Are classes suspended tomorrow?", type("E", (), {})())
    monkeypatch.setenv("CAMPUS_VERIFY_TAGALOG_TRANSLATE", "0")
    R.tagalog_translation = None
    assert answer_reuse.worth_keeping(tagalog, R())


# ---- through the API --------------------------------------------------------------------------------------------------
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_RATE_LIMIT_PER_MINUTE", "0")
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_off_by_default_nothing_is_marked(client):
    claim = {"claim": "Are classes suspended tomorrow?"}
    first, second = client.post("/verify/text", json=claim), client.post("/verify/text", json=claim)
    assert "x-answer-reused" not in first.headers and "x-answer-reused" not in second.headers
    assert "answer_reuse" not in client.get("/health").json()


def test_on_the_same_message_gets_the_same_answer_marked_reused(client, monkeypatch):
    monkeypatch.setenv(answer_reuse.ENV, "1")
    claim = {"claim": "Classes are suspended on November 20."}
    first, second = client.post("/verify/text", json=claim), client.post("/verify/text", json=claim)
    other = client.post("/verify/text", json={"claim": "Classes are suspended on November 21."})
    assert first.status_code == second.status_code == 200
    assert "x-answer-reused" not in first.headers and second.headers["x-answer-reused"] == "1"
    assert second.json() == first.json()
    assert "x-answer-reused" not in other.headers
    assert client.get("/health").json()["answer_reuse"] is True
