"""
Dataset quality auditing: near-duplicate detection and summary statistics.

This module does NOT modify or delete any data -- per M5's explicit
instruction, problems are REPORTED, not automatically fixed. A human
(the researcher) decides what to do with flagged rows.

Built as a reusable module (not a one-off script) because a 25-row
dataset can be eyeballed by hand, but the target 800-1,500 row thesis
dataset cannot -- this needs to scale.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Tuple


def _word_set(text: str) -> set:
    """Crude but adequate tokenization for similarity checking: lowercase,
    strip basic punctuation, split on whitespace. This does NOT need to be
    as careful as retrieval's TfidfVectorizer -- it's only used to flag
    pairs for human review, not to make automated decisions."""
    cleaned = text.lower()
    for ch in ".,?!;:\"'":
        cleaned = cleaned.replace(ch, "")
    return set(cleaned.split())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity: (shared words) / (all distinct words across both
    texts). Chosen over cosine/TF-IDF similarity for THIS purpose because
    it's simpler to explain, needs no fitted vocabulary, and is adequate
    for a coarse "these two texts are suspiciously similar" flag -- it is
    NOT used anywhere in retrieval or classification, only in this
    audit tool.
    """
    set_a, set_b = _word_set(text_a), _word_set(text_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


@dataclass
class DatasetAuditReport:
    """
    Summary of everything examined -- see build_audit_report() below for
    what populates each field.
    """
    total_records: int = 0
    label_distribution: Counter = field(default_factory=Counter)
    source_type_distribution: Counter = field(default_factory=Counter)
    total_groups: int = 0
    rows_per_group: dict = field(default_factory=dict)
    empty_evidence_rows: List[dict] = field(default_factory=list)
    genuine_neutral_rows: List[dict] = field(default_factory=list)
    cross_group_near_duplicate_claims: List[Tuple[str, str, float, str, str]] = field(default_factory=list)
    duplicated_evidence_texts: dict = field(default_factory=dict)


def build_audit_report(records: List[dict], near_duplicate_threshold: float = 0.5) -> DatasetAuditReport:
    """
    Compute the full audit -- the same checks performed manually in M5
    Step 2, now as reusable code so they can be re-run as the dataset
    grows toward its real thesis size.

    near_duplicate_threshold: Jaccard similarity above which two claims
    from DIFFERENT groups are flagged for human review. 0.5 is a
    starting point, not a validated value -- like the retrieval
    threshold, this is a development-only default that should be
    reviewed as more real data is collected.

    IMPORTANT: within-group pairs (same group_id) are deliberately
    EXCLUDED from the near-duplicate check -- a SUPPORTED and
    CONTRADICTED claim about the SAME real event are EXPECTED to share
    vocabulary. Flagging them would just be noise. We only care about
    similarity ACROSS groups, which would indicate two supposedly
    different real-world events are actually near-duplicates of each
    other -- a genuine leakage risk.
    """
    report = DatasetAuditReport()
    report.total_records = len(records)
    report.label_distribution = Counter(r["label"] for r in records)
    report.source_type_distribution = Counter(r["source_type"] for r in records)

    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    report.total_groups = len(groups)
    report.rows_per_group = {gid: len(rows) for gid, rows in groups.items()}

    report.empty_evidence_rows = [r for r in records if not r["evidence"].strip()]
    report.genuine_neutral_rows = [
        r for r in records
        if r["label"] == "INSUFFICIENT_EVIDENCE" and r["evidence"].strip()
    ]

    evidence_counts = Counter(r["evidence"].strip() for r in records if r["evidence"].strip())
    report.duplicated_evidence_texts = {
        text: count for text, count in evidence_counts.items() if count > 1
    }

    for r1, r2 in combinations(records, 2):
        if r1["group_id"] == r2["group_id"]:
            continue
        sim = jaccard_similarity(r1["claim"], r2["claim"])
        if sim > near_duplicate_threshold:
            report.cross_group_near_duplicate_claims.append(
                (r1["group_id"], r2["group_id"], sim, r1["claim"], r2["claim"])
            )

    return report


def print_audit_report(report: DatasetAuditReport):
    """Human-readable printout of an audit report -- used by scripts/audit_dataset.py."""
    print(f"Total records: {report.total_records}")
    print(f"Total groups: {report.total_groups}")
    print()
    print("Label distribution:")
    for label, count in report.label_distribution.most_common():
        print(f"  {label}: {count}")
    print()
    print("Source type distribution:")
    for source, count in report.source_type_distribution.most_common():
        print(f"  {source}: {count}")
    print()
    print(f"Empty-evidence rows (Case A candidates): {len(report.empty_evidence_rows)}")
    print(f"Genuine neutral rows -- evidence present but unrelated (Case B candidates): "
          f"{len(report.genuine_neutral_rows)}")
    print()
    print(f"Evidence texts reused by >1 row: {len(report.duplicated_evidence_texts)}")
    print("  (Expected when a SUPPORTED and CONTRADICTED claim share one group_id/event -- "
          "only a concern if it occurs ACROSS different group_ids, which is checked separately below.)")
    print()
    print(f"Cross-group near-duplicate claims (Jaccard > threshold): "
          f"{len(report.cross_group_near_duplicate_claims)}")
    if report.cross_group_near_duplicate_claims:
        print("  FLAGGED FOR HUMAN REVIEW -- not automatically removed:")
        for g1, g2, sim, c1, c2 in report.cross_group_near_duplicate_claims:
            print(f"  {g1} vs {g2} (similarity {sim:.2f}): '{c1}' <-> '{c2}'")
    else:
        print("  None found.")
