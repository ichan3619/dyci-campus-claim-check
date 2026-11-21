"""
Image Check: legitimacy levels (results/round2/legitimacy_levels_spec.md, approved by the owner 2026-10-24).

One overall result for an image check: five named levels, chosen by fixed rules over the statement checks and the photo
check, with the reasons in plain words. Not a probability; nothing here is trained or tuned. It only summarises results
that already exist and never changes a verdict or a statement result.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

NAMES = {5: "Matches DYCI", 4: "Mostly matches", 3: "Can't confirm",
         2: "Sketchy: looks official, but the content doesn't fit", 1: "Conflicts with DYCI"}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
EXPLAIN = ("Levels follow fixed rules over two checks: whether the statements in the picture match DYCI's announcements, "
           "and whether the photo appears in a DYCI post. They are not a probability. 5 = every checked statement matches; "
           "4 = some match and none conflict; 3 = nothing could be confirmed or contradicted; 2 = something looks official "
           "but does not fit (a statement may conflict with a DYCI post); "
           "1 = a statement clearly conflicts with the DYCI post it otherwise matches.")
CHECK = "Look for this announcement on DYCI's official Facebook page and compare the date and the details."


def post_key(url: Optional[str]) -> Optional[str]:
    """A comparable key for a Facebook post link (pfbid, fbid, numeric post, reel or video id, story id)."""
    if not url:
        return None
    for pattern in (r"(pfbid\w+)", r"[?&]story_fbid=(\w+)", r"[?&]fbid=(\d+)", r"/posts/(\d+)", r"/reel/(\d+)",
                    r"[?&]v=(\d+)", r"/share/p/(\w+)"):
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return url.split("?")[0].rstrip("/").lower()


def corpus_post_index(documents: Iterable[dict]) -> dict:
    """post key -> document_id for every link a collection document carries."""
    out = {}
    for d in documents:
        for field in ("source_url", "origin_url", "original_url"):
            key = post_key(d.get(field))
            if key:
                out.setdefault(key, d.get("document_id"))
    return out


def _date(evidence: Optional[dict]) -> Optional[str]:
    if not evidence:
        return None
    raw = (evidence.get("original_date") or evidence.get("published_date") or "")[:10]
    try:
        y, m, d = (int(x) for x in raw.split("-"))
        return f"{d} {MONTHS[m - 1]} {y}"
    except (ValueError, IndexError):
        return None


def _long_date(raw: Optional[str]) -> str:
    return _date({"published_date": raw}) or "an earlier date"


def _post(evidence: Optional[dict]) -> str:
    when = _date(evidence)
    return f"DYCI's post of {when}" if when else "a DYCI post"


def _photo_where(photo: dict) -> str:
    """Who posted the photo, named honestly: only DYCI's main page is "DYCI's post"."""
    when, page = _long_date(photo.get("post_date")), photo.get("page") or "DYCI"
    return f"DYCI's post of {when}" if page == "DYCI" else f"a post by {page}{',' if ', ' in page else ''} on {when}"


def assess(response: dict, photo_doc_id: Optional[str] = None) -> dict:
    """The level card for an /verify/image response (a plain dict). `photo_doc_id` is the collection document whose
    post the photo check found, when that post's text is in the collection; None when it is not (or no photo)."""
    basis = response.get("check_basis")
    summary = response.get("sentence_summary") or {}
    checks = response.get("sentence_checks") or []
    whole = response.get("text_check") or {}
    signals = response.get("image_signals") or {}
    photo = signals.get("photo_match")
    kind = summary.get("kind") if basis == "sentences" else None
    k, c, n = summary.get("matches", 0), summary.get("contradictions", 0), summary.get("checked", 0)
    if basis != "sentences":
        k, c, n = 0, 0, 0
        if basis == "whole_text_fallback":
            v = whole.get("verdict")
            k, c, n = int(v == "SUPPORTED"), int(v == "CONTRADICTED"), 1 if v else 0
    used = lambda s: ((s.get("text_check") or {}).get("used_evidence") or {})
    matched = [s for s in checks if s.get("outcome") == "match"]
    conflicting = [s for s in checks if s.get("outcome") == "contradiction"]
    matched_docs = {used(s).get("document_id") for s in matched}
    reasons: List[str] = []

    # Update B' (owner, 2026-10-24): the picture compared with ONLY the post its photo came from. Measured on genuine posts,
    # its "conflicts" are unreliable (OCR noise), so it can only raise Level 3 to 4, never lower a level.
    cross = signals.get("photo_post_check") if photo else None
    cross_ok = bool(cross and cross.get("matches", 0) >= 1 and cross.get("contradictions", 0) == 0)
    # Refinement A (owner, 2026-10-24): the photo is DYCI post X whose text is not in the collection, and the only
    # conflict is an unanchored one against another post -> "can't compare", not "sketchy".
    photo_elsewhere = bool(photo and photo_doc_id is None and kind == "possible_conflict")
    if kind == "conflict" or (basis == "whole_text_fallback" and whole.get("verdict") == "CONTRADICTED"):
        level = 1
    elif photo_elsewhere:
        level = 3
    elif kind == "possible_conflict":
        level = 2
    elif basis == "whole_text_fallback" and whole.get("verdict") == "SUPPORTED":
        level = 4                                     # spec: one undivided piece of text is "mostly matches" at best
    elif k >= 1 and c == 0 and k == n:
        level = 5
    elif k >= 1 and c == 0:
        level = 4
    else:
        level = 3
    if level == 3 and cross_ok:
        level = 4
    # Rule I (CAMPUS_VERIFY_AUTO_PHOTO_INDEX; results/round2/photo_auto_index_plan.md): the picture matches a DYCI picture
    # but a value reads differently (a swap) -> it may have been edited: level 1, with that reason first.
    edited = signals.get("edited_copy") if photo else None
    if edited:
        level = 1
        reasons.append(edited.get("summary") or "This picture matches a DYCI picture, but a value reads differently.")

    for s in conflicting if not photo_elsewhere else []:
        reasons.append(f"“{s.get('text', '')[:140]}” conflicts with {_post(used(s))}."
                       + ("" if level == 1 else " It could not be tied firmly to that post."))
    if basis == "whole_text_fallback" and whole.get("verdict") == "CONTRADICTED":
        reasons.append(f"The text conflicts with {_post(whole.get('used_evidence'))}.")
    if matched:
        posts = sorted({_post(used(s)) for s in matched})
        reasons.append(f"{len(matched)} of {n} checked statement{'s' if n != 1 else ''} match {' and '.join(posts)}.")
    elif basis == "whole_text_fallback" and whole.get("verdict") == "SUPPORTED":
        reasons.append(f"The text, checked as one piece, matches {_post(whole.get('used_evidence'))}.")
    if photo:
        where, when = _photo_where(photo), _long_date(photo.get("post_date"))
        if cross_ok:
            reasons.append(f"The text matches the post this photo came from ({where}): {cross['matches']} of "
                           f"{cross['checked']} statements.")
        elif cross:
            reasons.append(f"The photo is from {where}, but the text could not be confirmed against that post.")
        elif photo_elsewhere:
            reasons.append(f"The photo is from {where}, but that post's text is not in the system, so the text was "
                           "compared with other DYCI posts and could not be confirmed.")
        elif photo_doc_id and photo_doc_id in matched_docs:
            reasons.append(f"The photo is from the same post ({when}).")
        elif photo_doc_id and matched:
            reasons.append(f"The photo is from a different post ({where}) than the one the text matches.")
        else:
            reasons.append(f"The photo appears in {where}. That post's text is not in the system yet, so compare the "
                           "wording yourself.")

    card = {"level": level, "name": NAMES[level], "reasons": reasons, "explain": EXPLAIN,
            "photo_note": signals.get("note"), "why_not_sure": None, "closest": [], "check": None}
    if level == 3:
        card["why_not_sure"] = (
            "The text could not be matched to the post this photo came from; the picture may be hard to read." if cross
            else "The photo's post is not in the system yet, so its text could not be compared." if photo_elsewhere
            else _why_not_sure(response, checks, whole, k, n))
        card["closest"] = [{"document_id": e.get("document_id"), "date": _date(e),
                            "url": e.get("original_url") or e.get("source_url"),
                            "text": " ".join((e.get("text") or "").split())[:160]}
                           for e in (whole.get("retrieved_evidence") or [])[:3]]
        card["check"] = CHECK
    return card


def _why_not_sure(response: dict, checks: list, whole: dict, k: int, n: int) -> str:
    status = response.get("status")
    if status == "no_text":
        return "No readable text was found in the picture."
    if status == "uncertain":
        return "The text in the picture could not be read reliably enough to check."
    texts = [s.get("text_check") or {} for s in checks] + [whole]
    if any(t.get("verdict_withheld") == "language" for t in texts):
        return "Part of the text is in Tagalog or Taglish, which gets evidence only, not a verdict."
    if any((t.get("relative_date_check") or {}).get("withheld") for t in texts):
        return "It says “today” or “tomorrow”, but the matching DYCI post is from an earlier date."
    if any(t.get("scope_notice") for t in texts):
        return "It mentions another school; this checker only covers DYCI announcements."
    if n == 0:
        return "No statement in the picture could be checked."
    return "No DYCI announcement in the system matches or contradicts what the picture says."
