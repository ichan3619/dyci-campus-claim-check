"""
Legitimacy levels (src/image/levels.py; results/round2/legitimacy_levels_spec.md): every rule, on synthetic responses.
"""
import re

import pytest

from src.image import levels as lv

POST_A = {"document_id": "ingested:A", "published_date": "2026-10-01", "source_url": "https://www.facebook.com/YangaCollege/posts/pfbidAAA"}
POST_B = {"document_id": "ingested:B", "published_date": "2026-09-26", "source_url": "https://www.facebook.com/YangaCollege/posts/pfbidBBB"}


def _stmt(text, outcome, post=None, **tc):
    return {"text": text, "status": "checked", "outcome": outcome,
            "text_check": dict({"used_evidence": post}, **tc) if post or tc else None}


def _resp(checks, kind, photo=None, whole=None, basis="sentences", status="checked"):
    k = sum(s["outcome"] == "match" for s in checks)
    c = sum(s["outcome"] == "contradiction" for s in checks)
    return {"check_basis": basis, "status": status, "sentence_checks": checks,
            "sentence_summary": {"checked": len(checks), "matches": k, "contradictions": c, "kind": kind},
            "text_check": whole or {"retrieved_evidence": [POST_A, POST_B]},
            "image_signals": {"photo_match": photo, "note": "In testing it found 44 of 58 …"} if photo is not None or True else None}


PHOTO_A = {"post_id": "001", "post_url": "https://www.facebook.com/YangaCollege/posts/pfbidAAA", "post_date": "2026-10-01"}


def test_level_1_for_an_anchored_conflict_names_the_post():
    card = lv.assess(_resp([_stmt("Held on October 8.", "contradiction", POST_A), _stmt("Opening prayer.", "match", POST_A)], "conflict"))
    assert card["level"] == 1 and card["name"] == "Conflicts with DYCI"
    assert any("conflicts with DYCI's post of 1 October 2026" in r for r in card["reasons"])


def test_level_2_for_a_conflict_that_is_not_anchored():
    card = lv.assess(_resp([_stmt("Held on October 8.", "contradiction", POST_B)], "possible_conflict"))
    assert card["level"] == 2 and "could not be tied firmly" in " ".join(card["reasons"])


def test_update_b_prime_a_photo_from_another_known_post_no_longer_makes_it_sketchy():
    card = lv.assess(_resp([_stmt("Something else.", "match", POST_B)], "all_match", photo=PHOTO_A), photo_doc_id="ingested:A")
    assert card["level"] == 5 and any("from a different post" in r for r in card["reasons"])


def test_a_photo_whose_post_text_is_unknown_never_lowers_the_level():
    card = lv.assess(_resp([_stmt("Opening prayer.", "match", POST_B)], "all_match", photo=PHOTO_A), photo_doc_id=None)
    assert card["level"] == 5 and any("compare the wording yourself" in r for r in card["reasons"])


def test_level_5_with_the_photo_from_the_same_post():
    card = lv.assess(_resp([_stmt("Opening prayer.", "match", POST_A)], "all_match", photo=PHOTO_A), photo_doc_id="ingested:A")
    assert card["level"] == 5 and any("same post" in r for r in card["reasons"])


def test_level_4_when_some_match_and_none_conflict():
    card = lv.assess(_resp([_stmt("Opening prayer.", "match", POST_A), _stmt("Raffle and games.", "inconclusive", POST_A)], "partly_confirmed"))
    assert card["level"] == 4 and "1 of 2 checked statements match" in card["reasons"][0]


@pytest.mark.parametrize("verdict, level", [("SUPPORTED", 4), ("CONTRADICTED", 1), ("INSUFFICIENT_EVIDENCE", 3)])
def test_whole_text_fallback(verdict, level):
    whole = {"verdict": verdict, "used_evidence": POST_A, "retrieved_evidence": [POST_A]}
    assert lv.assess(_resp([], None, whole=whole, basis="whole_text_fallback"))["level"] == level


def test_level_3_says_why_shows_the_closest_posts_and_what_to_check():
    card = lv.assess(_resp([_stmt("Nothing known.", "inconclusive", POST_A)], "not_confirmed"))
    assert card["level"] == 3 and card["why_not_sure"].startswith("No DYCI announcement")
    assert [c["document_id"] for c in card["closest"]] == ["ingested:A", "ingested:B"] and card["closest"][0]["date"] == "1 October 2026"
    assert "official Facebook page" in card["check"]


@pytest.mark.parametrize("change, why", [
    (dict(status="no_text"), "No readable text"),
    (dict(status="uncertain"), "could not be read reliably"),
    (dict(tc={"verdict_withheld": "language"}), "Tagalog or Taglish"),
    (dict(tc={"relative_date_check": {"withheld": True}}), "earlier date"),
    (dict(tc={"scope_notice": {"schools": ["UE"]}}), "another school"),
])
def test_level_3_reasons(change, why):
    checks = [_stmt("Walang pasok bukas.", "inconclusive", POST_A, **change.get("tc", {}))]
    resp = _resp(checks, "not_confirmed", status=change.get("status", "checked"))
    card = lv.assess(resp)
    assert card["level"] == 3 and why in card["why_not_sure"]


def test_post_keys_link_facebook_urls_to_collection_documents():
    assert lv.post_key("https://www.facebook.com/YangaCollege/posts/pfbid02abc?rdid=x") == "pfbid02abc"
    assert lv.post_key("https://www.facebook.com/photo/?fbid=1476536611173734&set=a.1") == "1476536611173734"
    assert lv.post_key("https://www.facebook.com/share/p/1c53sBpted/") == "1c53sBpted"
    index = lv.corpus_post_index([POST_A, {"document_id": "x", "origin_url": "https://www.facebook.com/photo/?fbid=99"}])
    assert index == {"pfbidAAA": "ingested:A", "99": "x"}


def test_no_card_text_claims_fake_real_authentic_or_proof():
    words = " ".join(list(lv.NAMES.values()) + [lv.EXPLAIN, lv.CHECK]).lower()
    assert not re.search(r"\b(fake|real|authentic|proof|proves?)\b", words)


def test_refinement_a_a_photo_from_an_unknown_post_with_a_conflict_elsewhere_is_level_3():
    resp = _resp([_stmt("Classes shift online on October 16.", "contradiction", POST_B)], "possible_conflict", photo=PHOTO_A)
    card = lv.assess(resp, photo_doc_id=None)
    assert card["level"] == 3 and "not in the system yet" in card["why_not_sure"]
    assert not any("conflicts with" in r for r in card["reasons"])
    assert any("compared with other DYCI posts" in r for r in card["reasons"])
    assert lv.assess(resp, photo_doc_id="ingested:A")["level"] == 2       # the photo's post IS known: still sketchy
    anchored = _resp([_stmt("On October 8.", "contradiction", POST_B), _stmt("Prayer.", "match", POST_B)], "conflict", photo=PHOTO_A)
    assert lv.assess(anchored, photo_doc_id=None)["level"] == 1           # an anchored conflict stays Level 1


def test_the_photo_reason_names_who_posted_it():
    aces = dict(PHOTO_A, page="ACES, a DYCI student organisation")
    card = lv.assess(_resp([_stmt("Opening prayer.", "match", POST_B)], "all_match", photo=aces), photo_doc_id=None)
    assert any("a post by ACES, a DYCI student organisation, on 1 October 2026" in r for r in card["reasons"])
    plain = lv.assess(_resp([_stmt("Opening prayer.", "match", POST_B)], "all_match", photo=PHOTO_A), photo_doc_id=None)
    assert any("DYCI's post of 1 October 2026" in r for r in plain["reasons"])



def _with_cross(resp, matches, contradictions, checked=3):
    resp["image_signals"]["photo_post_check"] = {"post_id": "001", "checked": checked, "matches": matches,
                                                 "contradictions": contradictions, "statements": []}
    return resp


def test_update_b_prime_a_matching_cross_check_raises_level_3_to_4():
    resp = _with_cross(_resp([_stmt("Hard to read.", "inconclusive", POST_A)], "not_confirmed", photo=PHOTO_A), 2, 0)
    card = lv.assess(resp)
    assert card["level"] == 4
    assert any("matches the post this photo came from (DYCI's post of 1 October 2026): 2 of 3" in r for r in card["reasons"])


@pytest.mark.parametrize("matches, contradictions", [(0, 1), (1, 1), (0, 0)])
def test_update_b_prime_any_other_cross_check_result_changes_no_level(matches, contradictions):
    resp = _with_cross(_resp([_stmt("Hard to read.", "inconclusive", POST_A)], "not_confirmed", photo=PHOTO_A), matches, contradictions)
    card = lv.assess(resp)
    assert card["level"] == 3 and "could not be matched to the post this photo came from" in card["why_not_sure"]
    assert any("could not be confirmed against that post" in r for r in card["reasons"])


def test_update_b_prime_never_overrides_levels_1_and_2():
    conflict = _with_cross(_resp([_stmt("On October 8.", "contradiction", POST_A), _stmt("Prayer.", "match", POST_A)], "conflict", photo=PHOTO_A), 3, 0)
    assert lv.assess(conflict, photo_doc_id="ingested:A")["level"] == 1
    possible = _with_cross(_resp([_stmt("On October 8.", "contradiction", POST_B)], "possible_conflict", photo=PHOTO_A), 3, 0)
    assert lv.assess(possible, photo_doc_id="ingested:A")["level"] == 2
    assert lv.assess(possible, photo_doc_id=None)["level"] == 4          # refinement A gives 3; the cross-check raises it
