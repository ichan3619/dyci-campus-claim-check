"""
Tests for the M11 Facebook ingestion extension: optional pagination /
`since` on MetaGraphAPIAdapter.fetch_posts(), and the redundancy filter in
src/ingestion/dedup.py. `requests.get` is mocked; all text is synthetic.
"""

import unittest
from unittest.mock import Mock, patch

from src.ingestion.adapters import MetaGraphAPIAdapter
from src.ingestion.dedup import filter_redundant, similarity
from src.ingestion.evidence import normalize


def rec(text, ident, date="2026-10-01T00:00:00+0000"):
    return normalize(text=text, source_type="facebook", source_url=f"https://x/{ident}",
                     published_date=date, fetched_at="2026-10-02T00:00:00+00:00",
                     credibility_tier="", source_identifier=ident)


def resp(payload, status=200):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    return r


def post(i, msg):
    return {"id": f"p{i}", "message": msg, "created_time": f"2026-10-0{i}T00:00:00+0000",
            "permalink_url": f"https://fb/p{i}"}


class TestAdapterPagination(unittest.TestCase):
    def setUp(self):
        self.adapter = MetaGraphAPIAdapter(page_id="123", access_token="SECRET")

    @patch("src.ingestion.adapters.requests.get")
    def test_default_is_one_page_and_next_not_followed(self, get):
        get.return_value = resp({"data": [post(1, "Library closed.")],
                                 "paging": {"next": "https://graph/next?access_token=SECRET"}})
        out = self.adapter.fetch_posts()
        self.assertEqual(len(out), 1)
        self.assertEqual(get.call_count, 1)
        self.assertTrue(self.adapter.has_more_pages)

    @patch("src.ingestion.adapters.requests.get")
    def test_max_pages_follows_next(self, get):
        get.side_effect = [
            resp({"data": [post(1, "First notice.")], "paging": {"next": "https://graph/next"}}),
            resp({"data": [post(2, "Second notice.")]}),
        ]
        out = self.adapter.fetch_posts(max_pages=5)
        self.assertEqual([r.text for r in out], ["First notice.", "Second notice."])
        self.assertEqual(get.call_count, 2)
        self.assertNotIn("params", get.call_args_list[1].kwargs)  # next URL used as-is
        self.assertFalse(self.adapter.has_more_pages)

    @patch("src.ingestion.adapters.requests.get")
    def test_stops_at_max_pages_and_reports_more(self, get):
        get.return_value = resp({"data": [post(1, "A notice.")], "paging": {"next": "https://graph/next"}})
        self.adapter.fetch_posts(max_pages=2)
        self.assertEqual(get.call_count, 2)
        self.assertTrue(self.adapter.has_more_pages)

    @patch("src.ingestion.adapters.requests.get")
    def test_since_is_passed_through(self, get):
        get.return_value = resp({"data": []})
        self.adapter.fetch_posts(since="2026-09-01")
        self.assertEqual(get.call_args.kwargs["params"]["since"], "2026-09-01")


class TestDedup(unittest.TestCase):
    def test_same_id_skipped(self):
        a = rec("Library closed Oct 12 for inventory.", "a")
        d = filter_redundant([a], existing=[a])
        self.assertFalse(d[0].kept)

    def test_near_identical_same_numbers_skipped_earliest_kept(self):
        early = rec("Reminder: the library will be closed on October 12 for inventory.", "e", "2026-10-01T00:00:00+0000")
        late = rec("REMINDER: The library will be closed on October 12 for inventory!", "l", "2026-10-03T00:00:00+0000")
        d = filter_redundant([late, early])
        self.assertEqual([(x.record.document_id, x.kept) for x in d], [("ingested:e", True), ("ingested:l", False)])

    def test_existing_wins_over_candidate(self):
        old = rec("Enrollment ends on October 30 at the registrar.", "old")
        new = rec("Enrollment ends on October 30 at the registrar.", "new")
        d = filter_redundant([new], existing=[old])
        self.assertFalse(d[0].kept)
        self.assertEqual(d[0].matched_document_id, "ingested:old")

    def test_different_numbers_kept_and_flagged(self):
        a = rec("The library will be closed on October 12 for inventory.", "a")
        b = rec("The library will be closed on October 20 for inventory.", "b")
        d = filter_redundant([b], existing=[a])
        self.assertTrue(d[0].kept)
        self.assertTrue(d[0].numbers_differ)

    def test_unrelated_kept_unflagged(self):
        a = rec("The library will be closed on October 12.", "a")
        b = rec("Basketball tryouts start Monday at the gym.", "b")
        d = filter_redundant([b], existing=[a])
        self.assertTrue(d[0].kept)
        self.assertFalse(d[0].numbers_differ)

    def test_similarity_bounds(self):
        self.assertEqual(similarity("a b c", "a b c"), 1.0)
        self.assertEqual(similarity("a b", "c d"), 0.0)
        self.assertEqual(similarity("", "a"), 0.0)


if __name__ == "__main__":
    unittest.main()
