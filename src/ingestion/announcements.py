"""
Adding the NEW announcements for the M12.3 independent validation set (docs/12_3/validation_protocol.md, "Material").

The protocol requires 6-10 announcements that were NOT used in any M12 development, NOT written by whoever designed the
rules, with personal data removed, a stable ID (N01..N10) and a record of who supplied each. This module enforces what
can be checked mechanically:
    * ID format N01-N99, and no ID used twice;
    * the supplier is recorded (a role or code such as 'classmate C2', never a full name);
    * no e-mail address or phone number in the text (remove personal data before adding);
    * not a near-copy of any development document: the closest development document is reported, and anything at or
      above DEVELOPMENT_SIMILARITY_LIMIT word overlap is refused.
Records are stored as source_type 'manual_entry' in their own corpus file, so the development pool stays unchanged and
the independent run can list the file explicitly (CAMPUS_VERIFY_CORPUS="dev;<facebook corpus>;<this file>").
"""
import os
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from src.ingestion.corpus_storage import append_to_corpus, load_corpus
from src.ingestion.dedup import similarity
from src.ingestion.evidence import NormalizedEvidence, normalize

DEFAULT_ANNOUNCEMENTS_CORPUS = "data/corpus/independent_announcements.jsonl"
DEVELOPMENT_SIMILARITY_LIMIT = 0.6
ANNOUNCEMENT_ID = re.compile(r"^N\d{2}$")
PERSONAL_DATA = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}|\+?\d[\d\s-]{9,}\d")


def closest_development_document(text: str, development_texts: Iterable[Tuple[str, str]]) -> Optional[Tuple[str, float]]:
    """(document_id, word-overlap similarity) of the most similar development document, or None if there are none."""
    scored = [(doc_id, similarity(text, other)) for doc_id, other in development_texts]
    return max(scored, key=lambda x: x[1]) if scored else None


def add_announcement(corpus_path: str, announcement_id: str, text: str, supplied_by: str,
                     development_texts: List[Tuple[str, str]], source_url: str = "", original_date: str = "",
                     permission_confirmed: bool = False, purpose: str = "M12.3 independent") -> Tuple[NormalizedEvidence, Optional[Tuple[str, float]]]:
    """Validate and append one announcement. Returns (record, closest development document). Raises ValueError."""
    if not permission_confirmed:
        raise ValueError("confirm that you may use this announcement (permission_confirmed) before adding it")
    if not ANNOUNCEMENT_ID.match(announcement_id or ""):
        raise ValueError(f"announcement ID must look like N01..N99, got {announcement_id!r}")
    if not (supplied_by or "").strip():
        raise ValueError("record who supplied the announcement (a role or code, not a full name)")
    if PERSONAL_DATA.search(text or ""):
        raise ValueError("the text contains an e-mail address or phone number; remove personal data first")
    existing = load_corpus(corpus_path) if os.path.exists(corpus_path) else []
    if any(r.source_identifier == announcement_id for r in existing):
        raise ValueError(f"{announcement_id} is already in {corpus_path}")
    closest = closest_development_document(text, development_texts)
    if closest and closest[1] >= DEVELOPMENT_SIMILARITY_LIMIT:
        raise ValueError(f"too similar to development document {closest[0]} (word overlap {closest[1]:.2f}); "
                         "the protocol needs announcements that were not used in development")
    record = normalize(
        text=text, source_type="manual_entry", source_url=source_url, published_date=original_date,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"), credibility_tier="unverified",
        source_identifier=announcement_id,
        acquisition_note=f"{purpose} announcement {announcement_id}; supplied by {supplied_by.strip()}; "
                         "permission confirmed by the researcher",
    )
    append_to_corpus(record, corpus_path)
    return record, closest
