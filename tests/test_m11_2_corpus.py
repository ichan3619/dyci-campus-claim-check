"""
Tests for M11.2: configurable API corpus (src/api/corpus_config.py) and the
source map (src/ingestion/source_map.py). Synthetic data only; no network.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.api.corpus_config import load_configured_corpus, CORPUS_ENV_VAR, DEV_CORPUS_LABEL
from src.ingestion.corpus_storage import append_to_corpus
from src.ingestion.evidence import normalize
from src.ingestion.source_map import load_source_map, apply_source_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "dev", "sample_claims.csv")


def rec(text, ident, url="https://mock/1", date="2026-10-05T00:00:00+0000"):
    return normalize(text=text, source_type="facebook", source_url=url, published_date=date,
                     fetched_at="2026-10-06T00:00:00+00:00", credibility_tier="", source_identifier=ident)


class TestCorpusConfig(unittest.TestCase):
    def test_default_is_dev_dataset_unchanged(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(CORPUS_ENV_VAR, None)
            docs, label = load_configured_corpus(ROOT, DEV)
        self.assertEqual(label, DEV_CORPUS_LABEL)
        self.assertGreater(len(docs), 0)

    def test_opt_in_loads_jsonl(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "c.jsonl")
            append_to_corpus(rec("Library closed October 12.", "a"), p)
            with patch.dict(os.environ, {CORPUS_ENV_VAR: p}):
                docs, label = load_configured_corpus(ROOT, DEV)
        self.assertEqual(label, "c.jsonl")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["text"], "Library closed October 12.")

    def test_missing_file_fails_loudly_not_silently(self):
        with patch.dict(os.environ, {CORPUS_ENV_VAR: "no/such/file.jsonl"}):
            with self.assertRaises(RuntimeError):
                load_configured_corpus(ROOT, DEV)

    def test_empty_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "empty.jsonl")
            open(p, "w").close()
            with patch.dict(os.environ, {CORPUS_ENV_VAR: p}):
                with self.assertRaises(RuntimeError):
                    load_configured_corpus(ROOT, DEV)


class TestSourceMap(unittest.TestCase):
    def _write(self, t, text):
        p = os.path.join(t, "map.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def test_applies_original_url_and_date_keeps_ids(self):
        with tempfile.TemporaryDirectory() as t:
            m = load_source_map(self._write(t, "mock_post,original_url,original_date\n1_1,https://dyci/orig,2026-09-15\n"))
        r = rec("Notice.", "1_1")
        out, unmapped = apply_source_map([r], m)
        self.assertEqual((out[0].source_url, out[0].published_date), ("https://dyci/orig", "2026-09-15"))
        self.assertEqual(out[0].document_id, r.document_id)
        self.assertEqual(out[0].source_identifier, "1_1")
        self.assertEqual(unmapped, [])

    def test_blank_cell_keeps_fetched_value_and_unmapped_reported(self):
        with tempfile.TemporaryDirectory() as t:
            m = load_source_map(self._write(t, "mock_post,original_url,original_date\n1_1,,2026-09-15\n"))
        a, b = rec("A.", "1_1"), rec("B.", "1_2")
        out, unmapped = apply_source_map([a, b], m)
        self.assertEqual(out[0].source_url, a.source_url)
        self.assertEqual(out[0].published_date, "2026-09-15")
        self.assertEqual([u.source_identifier for u in unmapped], ["1_2"])

    def test_bad_files_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                load_source_map(self._write(t, "a,b\n1,2\n"))
            with self.assertRaises(ValueError):
                load_source_map(self._write(t, "mock_post,original_url,original_date\n1,x,y\n1,x,y\n"))


if __name__ == "__main__":
    unittest.main()
