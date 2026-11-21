"""
M12.3 independent-set tooling: adding new announcements (src/ingestion/announcements.py), the router freeze
preconditions (scripts/m12_retrieval/m12_3_freeze_router.py) and the Graph credential hints. No network.
"""
import importlib.util
import os

import pytest

from src.ingestion.announcements import add_announcement, closest_development_document
from src.ingestion.corpus_storage import load_corpus
from src.ingestion.facebook_page import explain_fetch_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = [("evt_001", "Official Notice #101: All classes are suspended tomorrow due to Typhoon Bagyo. Students should stay home.")]
NEW = "The Cashier's Office will accept tuition payments on Saturday, October 18, from 8:00 AM to 12:00 NN only."


def _add(path, **kw):
    args = dict(corpus_path=str(path), announcement_id="N01", text=NEW, supplied_by="classmate C2",
                development_texts=DEV, permission_confirmed=True)
    args.update(kw)
    return add_announcement(**args)


def test_adds_a_new_announcement_as_manual_entry(tmp_path):
    path = tmp_path / "ann.jsonl"
    record, closest = _add(path)
    stored = load_corpus(str(path))
    assert len(stored) == 1 and stored[0].source_type == "manual_entry" and stored[0].source_identifier == "N01"
    assert "classmate C2" in stored[0].acquisition_note and closest[0] == "evt_001" and closest[1] < 0.6


@pytest.mark.parametrize("change, message", [
    ({"permission_confirmed": False}, "permission"),
    ({"announcement_id": "A1"}, "N01..N99"),
    ({"supplied_by": "  "}, "supplied"),
    ({"text": NEW + " Contact juan.delacruz@dyci.edu.ph"}, "personal data"),
    ({"text": NEW + " Call 0917 123 4567."}, "personal data"),
    ({"text": DEV[0][1]}, "too similar"),
])
def test_refuses_what_the_protocol_forbids(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        _add(tmp_path / "ann.jsonl", **change)
    assert not (tmp_path / "ann.jsonl").exists()


def test_an_id_cannot_be_used_twice(tmp_path):
    path = tmp_path / "ann.jsonl"
    _add(path)
    with pytest.raises(ValueError, match="already"):
        _add(path, text="Library hours change to 7:00 AM - 9:00 PM starting November 3.")


def test_closest_development_document():
    assert closest_development_document("anything", []) is None
    assert closest_development_document(DEV[0][1], DEV) == ("evt_001", 1.0)


def _freeze_module():
    spec = importlib.util.spec_from_file_location("freeze", os.path.join(ROOT, "scripts", "m12_retrieval", "m12_3_freeze_router.py"))
    freeze = importlib.util.module_from_spec(spec); spec.loader.exec_module(freeze)
    return freeze


def _freeze_copy(tmp_path, criteria_status, frozen_on, reviewed):
    """Copies of the four freeze files under tmp_path (not a git repo), with the given criteria state and review."""
    import csv, json, shutil
    freeze = _freeze_module()
    for rel in freeze.FILES.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), tmp_path / rel)
    crit_path = tmp_path / freeze.FILES["criteria"]
    crit = json.loads(crit_path.read_text(encoding="utf-8")); crit.update(status=criteria_status, frozen_on=frozen_on)
    crit_path.write_text(json.dumps(crit), encoding="utf-8")
    sheet = tmp_path / freeze.FILES["review_sheet"]
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    gold_categories = {c["id"]: c["category"] for c in json.loads((tmp_path / freeze.FILES["classification"]).read_text(encoding="utf-8"))["cases"]}
    for r in rows:     # a completed review consistent with the gold file, whatever state the real review is in
        r["reviewer_category"] = gold_categories[r["id"]] if reviewed else ""
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    freeze.ROOT = str(tmp_path)
    return freeze


@pytest.mark.parametrize("status, frozen_on, reviewed, expect", [
    ("DRAFT_PENDING_CONFIRMATION_OF_INTERPRETATIONS", None, False, {"criteria", "review"}),
    ("FROZEN", "2026-10-07", False, {"review"}),
    ("FROZEN", None, True, {"criteria"}),
    ("FROZEN", "2026-10-07", True, set()),
])
def test_freeze_gate_requires_frozen_criteria_and_a_completed_review(tmp_path, status, frozen_on, reviewed, expect):
    found = _freeze_copy(tmp_path, status, frozen_on, reviewed).problems()
    assert {p.split(":")[0] for p in found} == expect


def test_freeze_gate_requires_reviewed_categories_to_be_applied(tmp_path):
    import csv
    freeze = _freeze_copy(tmp_path, "FROZEN", "2026-10-07", True)
    sheet = tmp_path / freeze.FILES["review_sheet"]
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    rows[0]["reviewer_category"] = "INFO_SEEKING" if rows[0]["reviewer_category"] == "CHECKABLE_CLAIM" else "CHECKABLE_CLAIM"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert any("not applied" in p for p in freeze.problems())


def test_credential_errors_get_a_hint_and_never_the_token():
    msg = explain_fetch_error("Graph API error for page 1 (code=190): expired ?access_token=abc123", "abc123")
    assert "abc123" not in msg and "expired or invalid" in msg
    assert explain_fetch_error("network timeout") == "network timeout"
