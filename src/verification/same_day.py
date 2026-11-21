"""
Same-day advisory rule Z (CAMPUS_VERIFY_SAME_DAY_ADVISORY, default off; results/round2/cross_post_conflict_plan.md,
Amendment 2). Wired in src/api/main.py after the class-day guard; English messages only.

The verifier judges one post: the best match among the 3 it retrieves that mentions the message's date. A post written
BEFORE a day ("classes resume on November 12") can then answer for that day while the advisory dated that day
("DYCI ADVISORY November 12, 2026 ... classes are suspended") ranks lower and is never read.

When the message names exactly one explicit date D, the advisories whose header date is D among the retriever's top
TOP_N gate-passing posts (at most MAX_ADVISORIES, rank order) are judged against the message as the verifier judges its
post used (passage selection, classifier, date check, value check; frozen components called, never edited), except
that an advisory whose body names no date is about its header date (Amendment 3):

    same-day labels                      verdict before              new verdict
    >=1 SUPPORTED, 0 CONTRADICTED        any                         SUPPORTED, on the first SUPPORTED advisory
    >=1 SUPPORTED and >=1 CONTRADICTED   any                         INSUFFICIENT_EVIDENCE (the advisories disagree)
    no SUPPORTED                         SUPPORTED or CONTRADICTED   INSUFFICIENT_EVIDENCE (the earlier post may be out of date)
    no SUPPORTED                         INSUFFICIENT_EVIDENCE       unchanged

It never creates CONTRADICTED. It does nothing when no same-day advisory exists or the post used is one of them.
Whenever it applies, the same-day advisories are shown first.
"""
import datetime as _dt
import os
import re
from dataclasses import replace
from typing import List, Optional

from src.verification.date_check import HEADER, MONTH_NAMES, event_dates, extract_dates, header_dates

ENV = "CAMPUS_VERIFY_SAME_DAY_ADVISORY"
VERSION = "same-day-v1"
TOP_N = 10                 # Amendment 1: the smallest of 5/10/20 reaching the 12 November advisory (rank 6)
MAX_ADVISORIES = 2

NOTE_SUPPORTED = " Same-day advisory: DYCI posted an advisory dated {day}, and it backs this message."
NOTE_DISAGREE = (" Same-day advisory: DYCI's advisories dated {day} disagree about this message, so it could not be "
                 "settled. Check the latest announcement.")
NOTE_STALE = (" Same-day advisory: DYCI posted an advisory dated {day}; the earlier post used may be out of date, so "
              "this could not be settled. Check the latest announcement.")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"          # default off (plan Amendment 2)


def _message_day(message: str, today: _dt.date):
    """(month, day, year) when the message names exactly one explicit date, else None."""
    dates = extract_dates(message or "")
    if len(dates) != 1:
        return None
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", message or "")}
    year = years.pop() if len(years) == 1 else today.year
    (month, day), = dates
    return month, day, year


def _header_year(text: str) -> Optional[int]:
    mt = HEADER.match(text or "")
    if not mt:
        return None
    y = re.search(r"\b(\d{4})\b", mt.group("date"))
    return int(y.group(1)) if y else None


def is_same_day_advisory(text: str, month: int, day: int, year: int) -> bool:
    if header_dates(text) != frozenset({(month, day)}):
        return False
    hy = _header_year(text)
    return hy is None or hy == year


def date_checked(claim: str, label: str, text: str) -> str:
    """The verifier's date check (check_verdict), except that an advisory whose body names no date is about its header
    date (Amendment 3: "DYCI ADVISORY November 12, 2026 ... suspended immediately" is about November 12)."""
    from src.verification.date_check import check_verdict
    if event_dates(text):
        return check_verdict(claim, label, text)[0]
    claimed = extract_dates(claim)
    if label == "SUPPORTED" and claimed and not (claimed & header_dates(text)):
        return "INSUFFICIENT_EVIDENCE"
    return label


def judge(verifier, claim: str, document):
    """(label, classification, selected_passage, explanation_part): `document` judged as verify_claim judges its post
    used. Reads the verifier's own components; changes nothing in them."""
    from src.retrieval.passage_selector import PassageSelector
    from src.verification.value_check import check_values
    selector = verifier._passage_selector
    chosen = selector.select(claim, [document]) if selector is not None else []
    selected = chosen[0] if chosen else None
    premise = PassageSelector.premise_from(chosen) if chosen else document.text
    if verifier._premise_mode == "sliding" and selected is not None:
        classification, label, _ = verifier._sliding_verdict(claim, document, selected)
    else:
        classification = verifier._classifier.classify(claim, premise)
        label = classification.label
    verdict = label
    if verifier._date_check:
        verdict = date_checked(claim, label, document.text)
    if verifier._value_check:
        verdict, _ = check_values(claim, verdict, premise, document.text)
    return verdict, classification, selected


def check(message: str, claim: str, result, verifier, today: _dt.date):
    """(result, info); info None when the rule does not apply. `message` is what the safeguards read (dates);
    `claim` is the text the verifier judged (pipeline_text)."""
    found = _message_day(message, today)
    if found is None:
        return result, None
    month, day, year = found
    try:
        retrieved = verifier._retriever.retrieve(claim, top_k=TOP_N)
    except Exception:
        return result, None
    gate = verifier._relevance_threshold
    advisories = [(i, d) for i, d in enumerate(retrieved, 1)
                  if d.score >= gate and is_same_day_advisory(d.text, month, day, year)][:MAX_ADVISORIES]
    if not advisories:
        return result, None
    used = result.used_evidence
    if used is not None and any(d.document_id == used.document_id for _, d in advisories):
        return result, None

    judged = []
    for rank, d in advisories:
        label, classification, selected = judge(verifier, claim, d)
        judged.append({"rank": rank, "document": d, "label": label, "classification": classification,
                       "selected": selected})
    labels = [j["label"] for j in judged]
    day_name = f"{MONTH_NAMES[month - 1]} {day}"
    before = result.verdict
    action = "shown_only"
    new = result
    if "SUPPORTED" in labels and "CONTRADICTED" not in labels:
        j = next(j for j in judged if j["label"] == "SUPPORTED")
        action = "supported"
        new = replace(result, verdict="SUPPORTED", used_evidence=j["document"],
                      classification_result=j["classification"], selected_passage=j["selected"],
                      explanation=result.explanation + NOTE_SUPPORTED.format(day=day_name))
    elif "SUPPORTED" in labels:
        action = "disagree_withheld"
        new = replace(result, verdict="INSUFFICIENT_EVIDENCE",
                      explanation=result.explanation + NOTE_DISAGREE.format(day=day_name))
    elif before in ("SUPPORTED", "CONTRADICTED"):
        action = "stale_withheld"
        new = replace(result, verdict="INSUFFICIENT_EVIDENCE",
                      explanation=result.explanation + NOTE_STALE.format(day=day_name))

    shown_ids = {j["document"].document_id for j in judged}
    new = replace(new, retrieved_evidence=[j["document"] for j in judged]
                  + [d for d in new.retrieved_evidence if d.document_id not in shown_ids])
    info = {"version": VERSION, "message_date": f"{year:04d}-{month:02d}-{day:02d}", "top_n": TOP_N,
            "advisories": [{"document_id": j["document"].document_id, "rank": j["rank"],
                            "score": round(float(j["document"].score), 4), "label": j["label"]} for j in judged],
            "post_used_before": used.document_id if used is not None else None,
            "original_verdict": before, "verdict": new.verdict, "action": action}
    return new, info


# ---------------------------------------------------------------------------------------------------------------------
# Rule S (CAMPUS_VERIFY_SAME_DAY_SHOW, default off; results/round2/same_day_show_plan.md): display only.
SHOW_ENV = "CAMPUS_VERIFY_SAME_DAY_SHOW"
SHOW_VERSION = "same-day-show-v1"


def show_enabled() -> bool:
    return os.environ.get(SHOW_ENV, "0").strip() == "1"     # default off (same_day_show_plan.md)


def show(message: str, claim: str, result, verifier, today: _dt.date):
    """(result, info); info None when nothing is added. Up to MAX_ADVISORIES gate-passing posts from the retriever's
    top TOP_N whose header date is the message's one explicit date, not already shown, are placed first among the shown
    posts. Only `retrieved_evidence` is reordered: verdict, post used, confidence, passage and explanation are kept."""
    found = _message_day(message, today)
    if found is None:
        return result, None
    month, day, year = found
    try:
        retrieved = verifier._retriever.retrieve(claim, top_k=TOP_N)
    except Exception:
        return result, None
    gate = verifier._relevance_threshold
    shown = {d.document_id for d in result.retrieved_evidence}
    if result.used_evidence is not None:
        shown.add(result.used_evidence.document_id)
    added = [(i, d) for i, d in enumerate(retrieved, 1)
             if d.score >= gate and d.document_id not in shown
             and is_same_day_advisory(d.text, month, day, year)][:MAX_ADVISORIES]
    if not added:
        return result, None
    new = replace(result, retrieved_evidence=[d for _, d in added] + list(result.retrieved_evidence))
    info = {"version": SHOW_VERSION, "message_date": f"{year:04d}-{month:02d}-{day:02d}", "top_n": TOP_N,
            "added": [{"document_id": d.document_id, "rank": i, "score": round(float(d.score), 4)} for i, d in added]}
    return new, info


# ---------------------------------------------------------------------------------------------------------------------
# Rule L (CAMPUS_VERIFY_SAME_DAY_LOOKUP, default off; results/round2/same_day_lookup_plan.md): display only. The
# advisories dated the message's day are found by their HEADER DATE, not by wording similarity (rule S failed because
# formally worded advisories score under the relevance gate).
LOOKUP_ENV = "CAMPUS_VERIFY_SAME_DAY_LOOKUP"
LOOKUP_VERSION = "same-day-lookup-v1"
LOOKUP_RANK_N = 50
_INDEX = {"documents": None, "index": {}}


def lookup_enabled() -> bool:
    return os.environ.get(LOOKUP_ENV, "0").strip() == "1"   # default off (same_day_lookup_plan.md)


def served_documents(verifier) -> list:
    """The served posts (dicts with document_id, text, source_type, source_url), read through retriever wrappers."""
    r = verifier._retriever
    while not hasattr(r, "_documents") and hasattr(r, "_base"):
        r = r._base
    docs = getattr(r, "_documents", None)
    return docs if docs is not None else []           # the same list object until a reload (date_index caches on it)


def header_day(text: str, published: str = ""):
    """(year, month, day) from a post's header, the year from the header or else from `published` (YYYY-...)."""
    days = header_dates(text)
    if len(days) != 1:
        return None
    (month, day), = days
    year = _header_year(text)
    if year is None and re.match(r"\d{4}", published or ""):
        year = int(published[:4])
    return (year, month, day) if year is not None else None


def date_index(documents, sources=None) -> dict:
    """(year, month, day) -> [document dicts], in corpus order. Cached per documents list (a reload builds a new one)."""
    if _INDEX["documents"] is documents:
        return _INDEX["index"]
    index: dict = {}
    for d in documents:
        info = (sources or {}).get(d["document_id"])
        key = header_day(d.get("text", ""), getattr(info, "published_date", "") or "")
        if key is not None:
            index.setdefault(key, []).append(d)
    _INDEX.update(documents=documents, index=index)
    return index


def lookup(message: str, claim: str, result, verifier, today: _dt.date, sources=None):
    """(result, info); info None when nothing is added. Up to MAX_ADVISORIES served posts whose header date is the
    message's one explicit date, not already shown, are placed first (retriever's ranking for the message first, then
    corpus order). Only `retrieved_evidence` is reordered."""
    from src.retrieval.retriever import RetrievedDocument
    found = _message_day(message, today)
    if found is None:
        return result, None
    month, day, year = found
    documents = served_documents(verifier)
    candidates = date_index(documents, sources).get((year, month, day), [])
    shown = {d.document_id for d in result.retrieved_evidence}
    if result.used_evidence is not None:
        shown.add(result.used_evidence.document_id)
    candidates = [d for d in candidates if d["document_id"] not in shown]
    if not candidates:
        return result, None
    try:
        ranked = verifier._retriever.retrieve(claim, top_k=LOOKUP_RANK_N)
    except Exception:
        ranked = []
    rank = {d.document_id: (i, d.score) for i, d in enumerate(ranked, 1)}
    order = sorted(range(len(candidates)),
                   key=lambda k: (rank.get(candidates[k]["document_id"], (10 ** 6, 0.0))[0], k))
    chosen = [candidates[k] for k in order][:MAX_ADVISORIES]
    added = [RetrievedDocument(document_id=d["document_id"],
                               score=float(rank.get(d["document_id"], (None, 0.0))[1]),
                               text=d.get("text", ""), source_type=d.get("source_type", ""),
                               source_url=d.get("source_url", "")) for d in chosen]
    new = replace(result, retrieved_evidence=added + list(result.retrieved_evidence))
    info = {"version": LOOKUP_VERSION, "message_date": f"{year:04d}-{month:02d}-{day:02d}",
            "day_name": f"{MONTH_NAMES[month - 1]} {day}, {year}",
            "added": [{"document_id": a.document_id, "rank": rank.get(a.document_id, (None,))[0],
                       "score": round(a.score, 4)} for a in added]}
    return new, info


# ---------------------------------------------------------------------------------------------------------------------
# Rule J, Z v2 (CAMPUS_VERIFY_SAME_DAY_JUDGE, default off; results/round2/same_day_judge_plan.md). Rule Z's decision
# table, with the advisories found by header date (rule L's index) and guard G for moved/postponed events.
JUDGE_ENV = "CAMPUS_VERIFY_SAME_DAY_JUDGE"
JUDGE_VERSION = "same-day-judge-v3"
# Guard G word list: v2's (fixed before test set J) plus v3's phrases (V3-b, fixed before test set J3); not tuned
# afterwards. "withdrawn" is deliberately absent: it also announces a plan restored ("the strike was withdrawn").
CHANGE_WORDS = re.compile(r"\b(?:mov(?:e|ed|es|ing)|postpon\w*|reschedul\w*|defer\w*|cancel\w*|call(?:ed|s|ing)?\s+off|"
                          r"extend\w*|extension|relocat\w*|"
                          r"no\s+longer(?:\s+be)?\s+held|will\s+no\s+longer|will\s+not\s+push\s+through|"
                          r"not\s+push(?:ing)?\s+through|will\s+not\s+be\s+held|will\s+not\s+proceed)\b", re.IGNORECASE)


OWN_DAY_WORDS = re.compile(r"\b(?:today|immediately|this\s+(?:morning|afternoon|evening))\b", re.IGNORECASE)


def about_day(text: str, month: int, day: int) -> bool:
    """V3-a (with Amendment 1): an advisory dated D is about D when its body names no date, names D among its dates,
    or says it concerns its own day ("today", "immediately", "this morning" ...)."""
    body = event_dates(text or "")
    return not body or (month, day) in body or bool(OWN_DAY_WORDS.search(text or ""))


def judge_enabled() -> bool:
    return os.environ.get(JUDGE_ENV, "0").strip() == "1"    # default off (same_day_judge_plan.md)


def guard_g(message: str, advisory_text: str, label: str):
    """(label, acted): a SUPPORTED from an advisory announcing a change (moved, postponed, ...) stands only when the
    message also speaks of a change."""
    if label == "SUPPORTED" and CHANGE_WORDS.search(advisory_text or "") and not CHANGE_WORDS.search(message or ""):
        return "INSUFFICIENT_EVIDENCE", True
    return label, False


def judge_same_day(message: str, claim: str, result, verifier, today: _dt.date, sources=None):
    """(result, info); info None when J does not apply (no advisory dated the message's day, or the post used is one)."""
    from src.retrieval.retriever import RetrievedDocument
    found = _message_day(message, today)
    if found is None:
        return result, None
    month, day, year = found
    candidates = [d for d in date_index(served_documents(verifier), sources).get((year, month, day), [])
                  if about_day(d.get("text", ""), month, day)]                                    # V3-a
    used = result.used_evidence
    if not candidates or (used is not None and any(d["document_id"] == used.document_id for d in candidates)):
        return result, None
    try:
        ranked = verifier._retriever.retrieve(claim, top_k=LOOKUP_RANK_N)
    except Exception:
        ranked = []
    rank = {d.document_id: (i, d.score) for i, d in enumerate(ranked, 1)}
    order = sorted(range(len(candidates)), key=lambda k: (rank.get(candidates[k]["document_id"], (10 ** 6, 0.0))[0], k))
    chosen = [candidates[k] for k in order][:MAX_ADVISORIES]
    judged = []
    for d in chosen:
        doc = RetrievedDocument(document_id=d["document_id"], score=float(rank.get(d["document_id"], (None, 0.0))[1]),
                                text=d.get("text", ""), source_type=d.get("source_type", ""),
                                source_url=d.get("source_url", ""))
        label, classification, selected = judge(verifier, claim, doc)
        label, guarded = guard_g(message, doc.text, label)
        judged.append({"document": doc, "label": label, "guard_g": guarded, "classification": classification,
                       "selected": selected, "rank": rank.get(d["document_id"], (None,))[0]})
    labels = [j["label"] for j in judged]
    day_name = f"{MONTH_NAMES[month - 1]} {day}"
    before, action, new = result.verdict, "shown_only", result
    if "SUPPORTED" in labels and "CONTRADICTED" not in labels:
        j = next(j for j in judged if j["label"] == "SUPPORTED")
        action = "supported"
        new = replace(result, verdict="SUPPORTED", used_evidence=j["document"],
                      classification_result=j["classification"], selected_passage=j["selected"],
                      explanation=result.explanation + NOTE_SUPPORTED.format(day=day_name))
    elif "SUPPORTED" in labels:
        action = "disagree_withheld"
        new = replace(result, verdict="INSUFFICIENT_EVIDENCE",
                      explanation=result.explanation + NOTE_DISAGREE.format(day=day_name))
    elif before in ("SUPPORTED", "CONTRADICTED"):
        action = "stale_withheld"
        new = replace(result, verdict="INSUFFICIENT_EVIDENCE",
                      explanation=result.explanation + NOTE_STALE.format(day=day_name))
    ids = {j["document"].document_id for j in judged}
    new = replace(new, retrieved_evidence=[j["document"] for j in judged]
                  + [d for d in new.retrieved_evidence if d.document_id not in ids])
    info = {"version": JUDGE_VERSION, "message_date": f"{year:04d}-{month:02d}-{day:02d}",
            "advisories": [{"document_id": j["document"].document_id, "rank": j["rank"], "label": j["label"],
                            "guard_g": j["guard_g"]} for j in judged],
            "post_used_before": used.document_id if used is not None else None,
            "original_verdict": before, "verdict": new.verdict, "action": action}
    return new, info
