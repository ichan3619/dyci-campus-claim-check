"""
Dataset filtering and group-based splitting -- SHARED across every model
that will ever be evaluated (TF-IDF+LR today, a pretrained NLI model
later).

ARCHITECTURAL NOTE (Milestone 5): this logic previously lived in
src/classification/dataset.py. It has been MOVED here because it is not
actually classification-specific -- it is dataset/evaluation
methodology that must be IDENTICAL for every model being compared,
per Milestone 5 Step 7's requirement that TF-IDF+LR and a future NLI
model be evaluated "using the same data splits." Leaving it inside
src/classification/ risked a future contributor writing a SECOND,
slightly different split function for the NLI model -- which would
silently invalidate the whole comparison (different models would not
even be tested on the same held-out data).

src/classification/dataset.py now just imports and re-exports these
same two functions, so nothing in Milestones 1-4 broke when this moved
-- see that file for the compatibility shim.
"""

import random
from typing import List


def build_classification_examples(records: List[dict]) -> List[dict]:
    """
    Filter loaded records down to ones any claim+evidence model can
    learn from: both claim and evidence must be non-empty.

    Excludes "no evidence found" INSUFFICIENT_EVIDENCE rows (empty
    evidence) -- those represent a RETRIEVAL-level outcome, not
    something a claim+evidence classifier should be trained or tested
    on (see Milestone 3 Step 3 for the full reasoning).
    """
    return [
        r for r in records
        if r.get("claim", "").strip() and r.get("evidence", "").strip()
    ]


def split_by_group(examples: List[dict], train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split examples into train/val/test, keeping every row that shares a
    group_id on the SAME side of the split (leakage prevention).

    seed is fixed so the split is reproducible -- every model compared
    in Milestone 5+ MUST be trained/evaluated on the exact same split,
    or a performance difference could come from unequal test data
    rather than the model itself.
    """
    group_ids = sorted(set(e["group_id"] for e in examples))
    rng = random.Random(seed)
    shuffled_groups = group_ids[:]
    rng.shuffle(shuffled_groups)

    n = len(shuffled_groups)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])

    train = [e for e in examples if e["group_id"] in train_groups]
    val = [e for e in examples if e["group_id"] in val_groups]
    test = [e for e in examples if e["group_id"] in test_groups]

    return train, val, test
