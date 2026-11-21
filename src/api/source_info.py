"""
Display metadata for evidence documents: where a notice came from, when it was posted and where the original is.

Kept OUT of the retrieval pipeline on purpose: retrieval receives exactly the documents it always did, and this index
is only used to decorate API responses. It is built from the same corpus sources, keyed by document_id, and rebuilt by
the API whenever the live corpus reloads (a change to data/source_map.csv alone shows after the next corpus reload or
restart).

    provenance      how the system acquired the document (PROVENANCE below)
    published_date  the date on the post / notice (for a test-Page copy: when it was re-posted on the test Page)
    original_url    the original DYCI announcement, when known: the record's own origin_url, else data/source_map.csv
    original_date   the original announcement's date, when known (same sources)
    synthetic       True for the development dataset's fictional notices

PROVENANCE
    test_page_mirror  a real DYCI post re-posted by hand to the project's own test Facebook Page and fetched from there.
                      Every `facebook` record is one: the system has no access to DYCI's own Page (M8.5-B blocked on
                      authorization). If that access ever exists, its records must get a different code.
    manual_entry      a DYCI post entered by hand (scripts/add_announcement.py); source_url is the original post
    submitted_page    the page a user submitted to /verify/url, for that request only
    fictional         a development-dataset notice (served only when a corpus override asks for it)
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.api.corpus_config import source_paths
from src.data.loader import load_dataset
from src.ingestion.corpus_storage import load_corpus
from src.ingestion.facebook_page import DEFAULT_SOURCE_MAP, load_optional_source_map

PROVENANCE_BY_SOURCE_TYPE = {"facebook": "test_page_mirror", "manual_entry": "manual_entry",
                             "official_site": "submitted_page", "synthetic": "fictional"}


@dataclass(frozen=True)
class SourceInfo:
    published_date: str = ""
    original_url: str = ""
    original_date: str = ""
    synthetic: bool = False
    provenance: str = ""


def _origin(record, source_map: Optional[Dict[str, Tuple[str, str]]]) -> Tuple[str, str]:
    """(original_url, original_date): the record's own fields first, then the source map, field by field."""
    url, date = record.origin_url or "", record.origin_date or ""
    if source_map and (not url or not date):
        mapped_url, mapped_date = source_map.get(record.source_identifier) or source_map.get(record.source_url) or ("", "")
        url, date = url or mapped_url, date or mapped_date
    return url, date


def build_source_info(spec: str, project_root: str, dev_dataset_path: str,
                      source_map_path: Optional[str] = None) -> Dict[str, SourceInfo]:
    """{document_id: SourceInfo} for every document the corpus spec loads."""
    info: Dict[str, SourceInfo] = {}
    dev = os.path.normcase(os.path.abspath(dev_dataset_path))
    source_map = load_optional_source_map(source_map_path or os.path.join(project_root, DEFAULT_SOURCE_MAP))
    for path in source_paths(spec, project_root, dev_dataset_path):
        if os.path.normcase(path) == dev:
            for record in load_dataset(dev_dataset_path).records:      # one document per group_id (see build_corpus)
                info.setdefault(record.get("group_id", ""), SourceInfo(published_date=record.get("date", "") or "",
                                                                       synthetic=True, provenance="fictional"))
        else:
            for r in load_corpus(path):
                url, date = _origin(r, source_map)
                provenance = ("dyci_website" if r.document_id.startswith("website:")      # rule W: DYCI's website feed
                              else PROVENANCE_BY_SOURCE_TYPE.get(r.source_type, ""))
                info[r.document_id] = SourceInfo(r.published_date, url, date, r.source_type == "synthetic", provenance)
    return info


def info_for_fetched_page(evidence) -> Dict[str, SourceInfo]:
    """The one-document index for /verify/url (the page fetched for that request)."""
    return {evidence.document_id: SourceInfo(published_date=evidence.published_date or "",
                                             original_url=evidence.origin_url or "", original_date=evidence.origin_date or "",
                                             provenance="submitted_page")}
