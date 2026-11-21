"""
Relevance check (CAMPUS_VERIFY_RELEVANCE_CHECK, default off): a small logistic-regression classifier that judges whether
the post an English SUPPORTED/CONTRADICTED verdict used is actually about the claim, and WITHHOLDS the verdict
(-> INSUFFICIENT_EVIDENCE) when P(relevant) < 0.5. It never creates a verdict and never turns SUPPORTED into
CONTRADICTED or back. Pre-registered in results/round2/relevance_check_plan.json (drafted in
results/round2/relevance_check_plan_draft.md, revision 3); everything below is fixed by that plan.

    features      10 signals per (checked text, candidate post) pair (FEATURES, pair_features()); none is a document id
    model         StandardScaler + LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000), stored as plain
                  JSON (scaler means/scales, coefficients, intercept) and checked by SHA-256 when loaded; predictions are
                  computed from that JSON, so the out-of-fold models judged and the shipped model use the same code
    evaluation    make_folds(): leave-one-group-out, groups built from labels only, and every document of a held-out
                  group removed from that fold's training pairs (audit_folds() re-checks it)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval.bm25 import TagalogBM25, word_tokens
from src.routing.router import DIGIT_RE, FILLER_WORDS, FUNCTION_WORDS, MONTH_WORDS, WH_WORDS

VERSION = "relevance-check-v1"
THRESHOLD = 0.5
MODEL_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
FEATURES: Tuple[str, ...] = ("e5_score", "e5_margin", "rank", "above_gate", "bm25_rel", "content_overlap",
                             "has_numbers", "number_match", "has_names", "name_match")
# The existing stop lists: scikit-learn's English list (the TF-IDF retriever's) plus the router's function, filler and
# question words (English and Tagalog).
STOP_WORDS: FrozenSet[str] = frozenset(ENGLISH_STOP_WORDS) | FUNCTION_WORDS | FILLER_WORDS | WH_WORDS
WITHHELD_EXPLANATION = (" Relevance check: the closest post may not address this claim (P(relevant)={p:.2f} < 0.50), "
                        "so the verdict was withheld.")
SC = ("SUPPORTED", "CONTRADICTED")
_RAW_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


# ---- features -------------------------------------------------------------------------------------------------------
def content_words(text: str) -> FrozenSet[str]:
    return frozenset(w for w in word_tokens(text) if w.isalpha() and len(w) >= 3 and w not in STOP_WORDS)


def number_terms(text: str) -> FrozenSet[str]:
    return frozenset(w for w in word_tokens(text) if DIGIT_RE.search(w) or w in MONTH_WORDS)


def name_terms(text: str) -> FrozenSet[str]:
    """Capitalised words other than the first word, lowercased."""
    words = _RAW_WORD_RE.findall(unicodedata.normalize("NFKC", text or ""))
    return frozenset(w.lower() for w in words[1:] if w[:1].isupper())


def _share(terms: FrozenSet[str], post_words: FrozenSet[str]) -> float:
    return sum(t in post_words for t in terms) / len(terms) if terms else 0.0


class FeatureBuilder:
    """Feature rows for (text, candidate) pairs over one corpus (the served documents)."""

    def __init__(self, documents: Sequence[dict]):
        self._index = {d["document_id"]: i for i, d in enumerate(documents)}
        self._words = [frozenset(word_tokens(d["text"])) for d in documents]
        self._bm25 = TagalogBM25([d["text"] for d in documents])

    def pair_features(self, text: str, retrieved: Sequence, relevance_threshold: float) -> List[Dict[str, float]]:
        """One row per retrieved candidate (the verifier's top 3, in order); `retrieved` items have .document_id/.score."""
        scores = [float(d.score) for d in retrieved]
        bm25 = self._bm25.score(text)
        top = max(bm25) if bm25 else 0.0
        content, numbers, names = content_words(text), number_terms(text), name_terms(text)
        rows = []
        for rank, doc in enumerate(retrieved):
            i = self._index[doc.document_id]
            words = self._words[i]
            if rank == 0:
                margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
            else:
                margin = scores[rank] - scores[0]
            rows.append({
                "e5_score": scores[rank],
                "e5_margin": margin,
                "rank": float(rank + 1),
                "above_gate": 1.0 if scores[rank] >= relevance_threshold else 0.0,
                "bm25_rel": bm25[i] / top if top > 0 else 0.0,
                "content_overlap": _share(content, words),
                "has_numbers": 1.0 if numbers else 0.0,
                "number_match": _share(numbers, words),
                "has_names": 1.0 if names else 0.0,
                "name_match": _share(names, words),
            })
        return rows


def vector(row: Dict[str, float]) -> List[float]:
    return [float(row[f]) for f in FEATURES]


# ---- model (JSON) ---------------------------------------------------------------------------------------------------
def fit(X: Sequence[Sequence[float]], y: Sequence[int]) -> dict:
    """StandardScaler + LogisticRegression with the pre-registered parameters -> plain-JSON parameters."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(**MODEL_PARAMS).fit(scaler.transform(X), list(y))
    return {"version": VERSION, "features": list(FEATURES), "threshold": THRESHOLD, "model_params": MODEL_PARAMS,
            "scaler_mean": [float(v) for v in scaler.mean_], "scaler_scale": [float(v) for v in scaler.scale_],
            "coef": [float(v) for v in model.coef_[0]], "intercept": float(model.intercept_[0]),
            "n_pairs": len(y), "n_positive": int(sum(y)), "sklearn_version": sklearn.__version__}


def predict(params: dict, x: Sequence[float]) -> float:
    """P(relevant) from the JSON parameters (the same arithmetic as scaler.transform + predict_proba)."""
    z = params["intercept"] + sum(c * (v - m) / s for c, v, m, s in
                                  zip(params["coef"], x, params["scaler_mean"], params["scaler_scale"]))
    return 1.0 / (1.0 + math.exp(-z))


def save_model(params: dict, path: str) -> str:
    data = (json.dumps(params, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def load_model(path: str, expected_sha256: str) -> dict:
    """The model JSON, refused unless its SHA-256 matches and it was fitted on exactly FEATURES."""
    if not expected_sha256:
        raise RuntimeError("CAMPUS_VERIFY_RELEVANCE_CHECK needs CAMPUS_VERIFY_RELEVANCE_MODEL_SHA256 (no adopted model)")
    with open(path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha256:
        raise RuntimeError(f"relevance model {path} has SHA-256 {sha}, expected {expected_sha256}")
    params = json.loads(data.decode("utf-8"))
    if tuple(params.get("features", ())) != FEATURES or params.get("threshold") != THRESHOLD:
        raise RuntimeError(f"relevance model {path} was not fitted on {VERSION}'s features and threshold")
    return params


# ---- product: withhold only -----------------------------------------------------------------------------------------
class RelevanceChecker:
    def __init__(self, documents: Sequence[dict], params: dict):
        self._features, self._params = FeatureBuilder(documents), params

    def apply(self, result, text: str, relevance_threshold: float):
        """(result, info). Only a SUPPORTED/CONTRADICTED result can change, and only to INSUFFICIENT_EVIDENCE."""
        if result.verdict not in SC or result.used_evidence is None:
            return result, None
        retrieved = list(result.retrieved_evidence)
        ids = [d.document_id for d in retrieved]
        rows = self._features.pair_features(text, retrieved, relevance_threshold)
        p = predict(self._params, vector(rows[ids.index(result.used_evidence.document_id)]))
        info = {"version": VERSION, "p_relevant": round(p, 4), "threshold": THRESHOLD, "withheld": p < THRESHOLD,
                "original_verdict": result.verdict}
        if not info["withheld"]:
            return result, info
        return replace(result, verdict="INSUFFICIENT_EVIDENCE",
                       explanation=result.explanation + WITHHELD_EXPLANATION.format(p=p)), info


# ---- evaluation split: leave-one-group-out with document exclusion --------------------------------------------------
class FoldError(RuntimeError):
    """The split cannot be made as pre-registered; the run refuses instead of loosening it."""


@dataclass(frozen=True)
class Message:
    key: str                                  # "<set>:<id>"
    candidates: Tuple[str, ...]               # the main verifier's E5 top 3 (document ids)
    expected: FrozenSet[str] = frozenset()    # expected announcement(s); empty = no answer
    anchor: Optional[str] = None              # round-2 announcement_shown


@dataclass
class Fold:
    group: str
    test_keys: List[str]
    held_docs: List[str]
    train_pairs: List[Tuple[str, str]]
    test_pairs: List[Tuple[str, str]]
    dropped_pairs: int
    train_positive: int = 0
    train_negative: int = 0


def make_groups(messages: Sequence[Message], merges: Iterable[Tuple[str, str]] = ()) -> Dict[str, str]:
    """message key -> group id: connected components over shared expected announcements, shared round-2
    announcement_shown, and the explicit merges. Built from labels only (never from candidates)."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    keys = {m.key for m in messages}
    for m in messages:
        find("M:" + m.key)
        for d in set(m.expected) | ({m.anchor} if m.anchor else set()):
            union("M:" + m.key, "D:" + d)
    for a, b in merges:
        if a not in keys or b not in keys:
            raise FoldError(f"merge {a} + {b}: unknown message")
        union("M:" + a, "M:" + b)
    roots = {m.key: find("M:" + m.key) for m in messages}
    names = {}
    for m in sorted(messages, key=lambda m: m.key):         # stable, readable group ids
        names.setdefault(roots[m.key], f"G{len(names) + 1:02d}")
    return {k: names[r] for k, r in roots.items()}


def make_folds(messages: Sequence[Message], merges: Iterable[Tuple[str, str]] = (),
               min_positive: int = 10, min_negative: int = 10) -> List[Fold]:
    groups = make_groups(messages, merges)
    by_key = {m.key: m for m in messages}
    folds = []
    for g in sorted(set(groups.values())):
        members = [m for m in messages if groups[m.key] == g]
        held = set()
        for m in members:
            held |= set(m.candidates) | set(m.expected) | ({m.anchor} if m.anchor else set())
        train, dropped = [], 0
        for m in messages:
            if groups[m.key] == g:
                continue
            for d in m.candidates:
                if d in held:
                    dropped += 1
                else:
                    train.append((m.key, d))
        pos = sum(d in by_key[k].expected for k, d in train)
        fold = Fold(g, [m.key for m in members], sorted(held), train,
                    [(m.key, d) for m in members for d in m.candidates], dropped, pos, len(train) - pos)
        if pos < min_positive or fold.train_negative < min_negative:
            raise FoldError(f"fold {g} ({', '.join(fold.test_keys)}) has {pos} positive and {fold.train_negative} "
                            f"negative training pairs (minimum {min_positive}/{min_negative})")
        folds.append(fold)
    audit_folds(folds, messages)
    return folds


def audit_folds(folds: Sequence[Fold], messages: Sequence[Message]) -> None:
    """Checks 1, 2 and 6 of the plan (section 5.3); raises FoldError on any violation."""
    seen = []
    for f in folds:
        train_docs = {d for _, d in f.train_pairs}
        train_keys = {k for k, _ in f.train_pairs}
        leaked = train_docs & set(f.held_docs)
        if leaked:
            raise FoldError(f"fold {f.group}: held-out documents in training pairs: {sorted(leaked)}")
        if train_keys & set(f.test_keys):
            raise FoldError(f"fold {f.group}: messages in both training and held-out pairs")
        if {d for _, d in f.test_pairs} - set(f.held_docs):
            raise FoldError(f"fold {f.group}: a held-out pair's document is missing from held_docs")
        seen.extend(f.test_keys)
    if sorted(seen) != sorted(m.key for m in messages):
        raise FoldError("not every message is held out exactly once")
