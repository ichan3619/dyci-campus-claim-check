"""Rule W step 2: the website fetch on a recorded fixture of the feeds' shape (no live call). Made-up content."""
import json

import pytest

from src.ingestion import dyci_website as w

ANN = [
    {"announcementID": 7, "announcementTitle": "Suspension of Classes &amp; Office Work", "announcementStatus": "Published",
     "announcementDescription": "<p>Classes at all levels are suspended on <b>July 10, 2030</b>.</p><p>Stay safe.</p>",
     "announcementDate": "2030-07-09", "announcementCategory": "Advisory", "announcementLevel": "All",
     "createdBy": "Staff Person", "updatedBy": "Another Person", "createdAt": "2030-07-09 08:00:00"},
    {"announcementID": 8, "announcementTitle": "Draft notice", "announcementStatus": "Draft",
     "announcementDescription": "not yet", "announcementDate": "2030-07-10", "createdBy": "Staff Person"},
]
ARTICLES = [
    {"articleID": 3, "articleTitle": "DYCI Wins Regional Robotics Cup", "articleStatus": "Published",
     "articleExcerpt": "The DYCI Robotics team won first place on August 2, 2030.", "articlePublishedDate": "2030-08-03",
     "articleAuthor": "Student Writer", "createdBy": "Editor Person"},
]
EVENTS = [
    {"eventID": 5, "eventTitle": "Foundation Day 2030", "eventStatus": "Published", "eventStart": "2030-10-11 08:00:00",
     "eventEnd": "2030-10-11 17:00:00", "eventLocation": "DYCI Gymnasium", "eventDescription": "<p>Program and fair.</p>",
     "eventPublishedDate": "2030-09-30"},
]


def fake_get(pages=None):
    pages = pages if pages is not None else [ARTICLES]
    calls = []

    def get(url):
        calls.append(url)
        if "get-announcements" in url:
            return ANN
        if "get-events" in url:
            return {"mode": "all", "data": EVENTS}
        page = int(url.split("page=")[1].split("&")[0])
        data = pages[page - 1] if page <= len(pages) else []
        return {"page": page, "total": sum(len(p) for p in pages), "data": data}
    get.calls = calls
    return get


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(w.ENV, raising=False)
    assert w.enabled() is False


def test_slugs_follow_the_site():
    assert w.slug_announcement("Suspension of Classes & Office Work!") == "suspension-of-classes-office-work"
    assert w.slug_article("  DYCI Wins: Regional Cup ") == "dyci-wins-regional-cup"


def test_text_of():
    assert w.text_of("<p>Classes <b>are</b> suspended.</p><p>Stay&nbsp;safe.</p>") == "Classes are suspended. Stay safe"


def test_records():
    recs = w.records_from(*w.read_feeds(fake_get()), now="2030-07-09T00:00:00+00:00")
    assert [r["document_id"] for r in recs] == ["website:announcement:7", "website:article:3", "website:event:5"]
    a, art, ev = recs
    assert a["text"].startswith("Suspension of Classes & Office Work. Classes at all levels are suspended on July 10, 2030")
    assert a["source_url"] == "https://dyci.edu.ph/announcements/suspension-of-classes-office-work"
    assert a["published_date"] == "2030-07-09" and a["source_type"] == "official_site" and a["credibility_tier"] == "official"
    assert art["source_url"] == "https://dyci.edu.ph/dycian-news/dyci-wins-regional-robotics-cup"
    assert "Venue: DYCI Gymnasium" in ev["text"] and ev["published_date"] == "2030-10-11"
    blob = json.dumps(recs)
    for name in ("Staff Person", "Another Person", "Student Writer", "Editor Person", "Draft notice"):
        assert name not in blob


def test_article_pages_until_total_or_empty():
    get = fake_get(pages=[ARTICLES, [dict(ARTICLES[0], articleID=4)]])
    _, articles, _ = w.read_feeds(get)
    assert [a["articleID"] for a in articles] == [3, 4]
    assert all("limit=50" in u for u in get.calls if "get-articles" in u)


def test_cutoff():
    recs = [{"published_date": "2026-11-15"}, {"published_date": "2026-11-16"}, {"published_date": "2026-11-20"},
            {"published_date": ""}]
    assert len(w.before_cutoff(recs, "2026-11-16", False)) == 1          # undated dropped too
    assert len(w.before_cutoff(recs, "2026-11-16", True)) == 4


@pytest.mark.parametrize("raw,iso", [("2026-11-12", "2026-11-12"), ("2026-11-12 08:00:00", "2026-11-12"),
                                     ("07062025", "2025-07-06"), ("11202026", "2026-11-20"), ("13012025", ""),
                                     ("", ""), (None, ""), ("Nov 12", "")])
def test_iso_date(raw, iso):
    assert w.iso_date(raw) == iso


def test_refresh_is_stable_and_keeps_the_file_on_failure(tmp_path):
    path = str(tmp_path / "dyci_website.jsonl")
    first = w.refresh(path, fake_get())
    second = w.refresh(path, fake_get())
    assert (first["added"], first["items"]) == (3, 3)
    assert (second["added"], second["edited"], second["removed"]) == (0, 0, 0)
    before = open(path, encoding="utf-8").read()

    def broken(url):
        raise ValueError("https://dyci.edu.ph/api/get-announcements.php answered HTTP 503")
    with pytest.raises(ValueError):
        w.refresh(path, broken)
    assert open(path, encoding="utf-8").read() == before


def test_item_for_link(tmp_path, monkeypatch):
    path = tmp_path / "web.jsonl"
    recs = w.records_from(*w.read_feeds(fake_get()), now="x")
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    monkeypatch.setenv(w.FILE_ENV, str(path))
    assert w.item_for_link("https://dyci.edu.ph/announcements/suspension-of-classes-office-work", "")["document_id"] \
        == "website:announcement:7"
    assert w.item_for_link("https://www.dyci.edu.ph/dycian-news/dyci-wins-regional-robotics-cup/", "") is not None
    assert w.item_for_link("https://dyci.edu.ph/announcements/not-there", "") is None
    assert w.item_for_link("https://dyci.edu.ph/admission", "") is None
    assert w.item_for_link("https://dyci-edu-ph.com/announcements/suspension-of-classes-office-work", "") is None


def test_scheduler_reports_and_never_raises(tmp_path, monkeypatch):
    seen = []
    s = w.Scheduler(str(tmp_path / "web.jsonl"), 60, lambda counts, error: seen.append((counts, error)))
    monkeypatch.setattr(w, "cutoff_settings", lambda: (None, False))
    monkeypatch.setattr(w, "http_json", fake_get())
    monkeypatch.setattr(w.refresh, "__defaults__", (fake_get(), None, False))
    s.run_once()
    assert seen[-1][1] is None and seen[-1][0]["items"] == 3

    def broken(url):
        raise ValueError("HTTP 503")
    monkeypatch.setattr(w.refresh, "__defaults__", (broken, None, False))
    s.run_once()
    assert seen[-1][0] is None and "HTTP 503" in seen[-1][1]


def test_fetch_status_keeps_website_reads_apart():
    from types import SimpleNamespace
    from src.api import fetch_status as fs
    refresh = SimpleNamespace(enabled=True, interval_minutes=10, last_run=None, last_added=0, last_error=None)
    log = [{"at": "2026-11-18T10:00:00Z", "ok": False, "error": "ValueError: HTTP 503", "source": "dyci_website"},
           {"at": "2026-11-18T09:58:00Z", "ok": True}]
    out = fs.build(refresh, log, {"file_readable": True})
    assert out["auto_fetch"]["last_success"] == "2026-11-18T09:58:00Z"
    assert any("website failed" in x for x in out["warnings"]) and out["website_last"]["source"] == "dyci_website"


def test_served_website_posts_are_labelled(tmp_path):
    from src.api.source_info import build_source_info
    recs = w.records_from(*w.read_feeds(fake_get()), now="x")
    p = tmp_path / "web.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    info = build_source_info(str(p), str(tmp_path), str(tmp_path / "none.jsonl"))
    assert info["website:announcement:7"].provenance == "dyci_website"
