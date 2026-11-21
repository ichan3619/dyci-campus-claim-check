"""
Verification orchestration layer.

Responsibility of this module, and ONLY this module:

    Given a claim, coordinate retrieval and classification to produce a
    final verification result.

This module contains NO retrieval logic (no TF-IDF, no cosine
similarity -- it calls src/retrieval/retriever.py for that) and NO
classification logic (no TF-IDF+LR training/inference -- it calls
src/classification/classifier.py for that). It only makes DECISIONS
about how to combine their outputs:

    - which retrieved result(s) to pass to the classifier (Milestone 4
      Step 2 decided: top-1 only, for this milestone)
    - what to do when retrieval finds nothing relevant
    - how to package everything into one structured result

This is intentionally a thin layer. If you find yourself wanting to add
TF-IDF code or model-training code here, it belongs in retrieval/ or
classification/ instead, not here.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.retriever import Retriever, RetrievedDocument, DEVELOPMENT_ONLY_RELEVANCE_HINT_THRESHOLD
from src.classification.classifier import ClaimEvidenceClassifier, ClassificationResult
from src.retrieval.passage_selector import PassageSelector, SelectedPassage   # M12.2 (optional layer)
from src.verification.date_check import check_verdict, prefer_dated_documents   # M12.3 (optional layer)
from src.verification.value_check import check_values                         # round 2 (optional layer)


# --------------------------------------------------------------------------
# EVIDENCE STATUS -- this is the Step 5 distinction, made explicit.
# --------------------------------------------------------------------------
# Two different situations both currently produce the SAME final verdict
# label (INSUFFICIENT_EVIDENCE), but for different reasons, and the
# verification result keeps that reason around instead of throwing it
# away. This matters for error analysis later: if most of your errors
# are NO_RELEVANT_EVIDENCE, the fix is "improve retrieval / grow the
# corpus." If most are RELEVANT_EVIDENCE_FOUND-but-still-insufficient,
# the fix is "improve the classifier." Collapsing both into one
# undifferentiated label would hide which problem you actually have.
NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
RELEVANT_EVIDENCE_FOUND = "RELEVANT_EVIDENCE_FOUND"


@dataclass
class VerificationResult:
    """
    The complete output of verifying one claim.

    Fields:
        claim: the original claim text that was checked.

        verdict: the final label -- SUPPORTED, CONTRADICTED, or
            INSUFFICIENT_EVIDENCE. This is what a user-facing UI would
            show as the headline result.

        evidence_status: NO_RELEVANT_EVIDENCE or RELEVANT_EVIDENCE_FOUND
            (see constants above). Explains WHY the verdict is what it
            is, especially when verdict == INSUFFICIENT_EVIDENCE, where
            this field distinguishes "nothing was found" from "something
            was found but doesn't resolve the claim."

        retrieved_evidence: ALL top-K documents retrieval returned, with
            their similarity scores -- kept for transparency/audit even
            though (per the Step 2 decision) only the top one was
            actually classified. Empty list if retrieval found nothing.

        used_evidence: the ONE document that was actually passed to the
            classifier, or None if evidence_status is
            NO_RELEVANT_EVIDENCE (in which case the classifier was never
            called at all).

        classification_result: the full ClassificationResult (label,
            confidence, label_scores) from the classifier, or None if
            the classifier was never called.

        explanation: a short, human-readable sentence describing why
            this verdict was reached. Intended for showing to a user or
            for quick debugging -- NOT a substitute for inspecting the
            structured fields above.
    """
    claim: str
    verdict: str
    evidence_status: str
    retrieved_evidence: List[RetrievedDocument]
    used_evidence: Optional[RetrievedDocument]
    classification_result: Optional[ClassificationResult]
    explanation: str
    # M12.2: the passage actually handed to the classifier when a PassageSelector is in use.
    # None when no selector is configured (the whole document is the premise, as before).
    selected_passage: Optional[SelectedPassage] = None
    # Sprint 2/3: optional normalization record (original_user_input, normalized_claim, flags, features such as
    # short_claim / has_number / code_switched) attached by the caller; None = no normalization was applied.
    # Retrieval and classification never read it; it only travels with the result (e.g. for a ClaimRouter).
    normalization: Optional[dict] = None
    # M12.3: set when the date check withheld the classifier's verdict ('claimed_date_not_in_evidence' or
    # 'evidence_has_no_date'); classification_result still carries the classifier's own label.
    date_check: Optional[str] = None
    # Round 2: set when the value check changed the verdict ('value_conflict' or 'value_conflict_withheld').
    value_check: Optional[str] = None
    # Round 2: set in premise_mode="sliding": windows judged, non-neutral counts, conflict, the deciding window.
    sliding_windows: Optional[dict] = None


class ClaimVerifier:
    """
    Orchestrates retrieval + classification to verify a claim.

    Usage:
        verifier = ClaimVerifier(retriever, classifier)
        result = verifier.verify_claim("Classes are suspended tomorrow.")

    The verifier is constructed with an ALREADY-BUILT Retriever and an
    ALREADY-TRAINED ClaimEvidenceClassifier (dependency injection,
    rather than this class building or loading them itself). This keeps
    ClaimVerifier agnostic to how the retriever's corpus was built or
    how the classifier was trained -- it only knows how to use them.
    """

    def __init__(
        self,
        retriever: Retriever,
        classifier: ClaimEvidenceClassifier,
        top_k: int = 3,
        relevance_threshold: Optional[float] = None,
        passage_selector: Optional[PassageSelector] = None,
        date_check: bool = False,
        value_check: bool = False,
        premise_mode: str = "window",
    ):
        """
        top_k: how many candidates to request from retrieval. Only the
            top-1 of these is actually classified (Step 2 decision), but
            the rest are kept in the result for transparency.

        relevance_threshold: the minimum retrieval score the top result
            must have before we bother calling the classifier at all.
            Below this, we short-circuit straight to INSUFFICIENT_EVIDENCE
            with evidence_status=NO_RELEVANT_EVIDENCE.

            IMPORTANT / KNOWN LIMITATION: this reuses
            DEVELOPMENT_ONLY_RELEVANCE_HINT_THRESHOLD from retriever.py,
            which was introduced in Milestone 2 purely as a printed hint
            in a CLI script -- it was NOT experimentally validated. Using
            it here as an actual decision gate (not just a hint) is a
            genuine limitation of this milestone, not a hidden one: the
            real threshold must eventually be chosen by sweeping
            candidate values against a labeled retrieval-evaluation set,
            as already documented in retriever.py. This is called out
            explicitly in the M4 report's "Known Limitations" section.

        date_check (M12.3, opt-in, default False = unchanged behaviour): when the claim names an explicit calendar
            date, (a) among candidates that cleared the gate, documents containing that date are preferred, and
            (b) a SUPPORTED / CONTRADICTED verdict the evidence cannot back on the date is withheld as
            INSUFFICIENT_EVIDENCE. See src/verification/date_check.py.

        value_check (round 2, opt-in, default False = unchanged behaviour): after the date check, a claimed value
            (date, time, percentage, 'N out of M', duration, place) that the judged premise contradicts turns
            INSUFFICIENT_EVIDENCE into CONTRADICTED and withholds SUPPORTED. See src/verification/value_check.py.

        premise_mode (round 2, opt-in; default "window" = unchanged behaviour): "sliding" judges EVERY premise window of
            the selected document (same window shape as the passage selector: preceding sentence + sentence) with the
            same language-routed model. Verdict: any SUPPORTED + any CONTRADICTED window -> INSUFFICIENT_EVIDENCE
            regardless of confidence; otherwise the shared non-neutral label, reported from the window with the
            highest confidence (= softmax score of its own label; ties -> earliest window); all neutral ->
            INSUFFICIENT_EVIDENCE with the E5-selected window's result. Softmax scores are uncalibrated and compared
            only among one claim's windows (one model), never across models or claims. Needs a passage selector.
        """
        # M12.2 threshold resolution (backward compatible):
        #   explicit argument  >  retriever.default_relevance_threshold (e.g. E5Retriever)
        #   >  the TF-IDF development constant, exactly as before for the original Retriever.
        if relevance_threshold is None:
            relevance_threshold = getattr(retriever, "default_relevance_threshold", None)
        if relevance_threshold is None:
            relevance_threshold = DEVELOPMENT_ONLY_RELEVANCE_HINT_THRESHOLD
        self._retriever = retriever
        self._classifier = classifier
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold
        # Optional passage-selection stage between retrieval and classification (M12.2).
        self._passage_selector = passage_selector
        self._date_check = date_check
        self._value_check = value_check
        if premise_mode not in ("window", "sliding"):
            raise ValueError(f"premise_mode must be 'window' or 'sliding', got {premise_mode!r}")
        self._premise_mode = premise_mode

    def verify_claim(self, claim: str) -> VerificationResult:
        """
        Run the full retrieval -> classification pipeline for one claim.

        Raises ValueError for an empty claim -- there is nothing
        meaningful to verify, and silently returning some placeholder
        result would hide a caller's mistake rather than surface it.
        (This mirrors ClaimEvidenceClassifier.classify()'s behavior on
        an empty claim, for consistency across the whole pipeline.)
        """
        if not claim or not claim.strip():
            raise ValueError("claim must not be empty")

        # STEPS 1-2: retrieve, apply the relevance gate, select the passage (shared with retrieve_information).
        evidence = self._gather_evidence(claim)
        retrieved = evidence.retrieved
        if evidence.top_evidence is None:
            return VerificationResult(
                claim=claim,
                verdict="INSUFFICIENT_EVIDENCE",
                evidence_status=NO_RELEVANT_EVIDENCE,
                retrieved_evidence=retrieved,
                used_evidence=None,
                classification_result=None,
                explanation=self._no_evidence_explanation(retrieved, "claim"),
            )

        # STEP 3: relevant evidence exists -- classify the top result
        # (Step 2 decision: top-1 strategy, not aggregation).
        # This is the ONLY call into src/classification/ -- verification
        # does not reimplement any part of the TF-IDF/LR logic itself.
        top_evidence, selected_passage, premise = evidence.top_evidence, evidence.selected_passage, evidence.premise
        sliding = None
        if self._premise_mode == "sliding" and self._passage_selector is not None and selected_passage is not None:
            classification, label, sliding = self._sliding_verdict(claim, top_evidence, selected_passage)
        else:
            classification = self._classifier.classify(claim, premise)
            label = classification.label
        verdict, date_reason = label, None
        if self._date_check:
            verdict, date_reason = check_verdict(claim, label, top_evidence.text)
        value_reason = None
        if self._value_check:
            verdict, value_reason = check_values(claim, verdict, premise, top_evidence.text)

        return VerificationResult(
            claim=claim,
            verdict=verdict,
            evidence_status=RELEVANT_EVIDENCE_FOUND,
            retrieved_evidence=retrieved,
            used_evidence=top_evidence,
            classification_result=classification,
            explanation=(
                f"Evidence '{top_evidence.document_id}' (retrieval score="
                f"{top_evidence.score:.3f}) was classified as {classification.label} "
                f"with confidence {classification.confidence:.3f}."
                + (f" Premise was the selected passage #{selected_passage.passage_index} "
                   f"(passage score={selected_passage.score:.3f})." if selected_passage else "")
                + (f" The verdict was withheld ({date_reason}): the evidence does not back the date in the claim."
                   if date_reason else "")
                + (f" Value check ({value_reason}): the evidence states a different date, time, number or place."
                   if value_reason else "")
                + ((" Sliding windows: SUPPORTED and CONTRADICTED windows disagreed, so the verdict was withheld."
                    if sliding["conflict"] else
                    f" Sliding windows: the deciding window was #{sliding['chosen_index']} of {sliding['windows']}."
                    if sliding["chosen_index"] is not None else
                    f" Sliding windows: all {sliding['windows']} windows were neutral.") if sliding else "")
            ),
            selected_passage=selected_passage,
            date_check=date_reason,
            value_check=value_reason,
            sliding_windows=sliding,
        )

    def _sliding_verdict(self, claim: str, document: RetrievedDocument, selected: SelectedPassage):
        """(classification shown, verdict label, detail) for premise_mode="sliding" (see __init__)."""
        windows = self._passage_selector.windows_of(document)
        results = [self._classifier.classify(claim, w) for w in windows]
        non_neutral = [(i, r) for i, r in enumerate(results) if r.label in ("SUPPORTED", "CONTRADICTED")]
        labels = sorted({r.label for _, r in non_neutral})
        base_index = selected.passage_index if 0 <= selected.passage_index < len(results) else 0
        detail = {"windows": len(windows), "non_neutral": {lab: sum(r.label == lab for _, r in non_neutral) for lab in labels},
                  "conflict": len(labels) > 1, "chosen_index": None, "chosen_premise": None, "e5_window_index": base_index}
        if len(labels) > 1:                                      # SUPPORTED and CONTRADICTED both present
            return results[base_index], "INSUFFICIENT_EVIDENCE", detail
        if non_neutral:
            index, chosen = max(non_neutral, key=lambda x: (x[1].confidence, -x[0]))   # ties -> earliest window
            detail.update(chosen_index=index, chosen_premise=windows[index])
            return chosen, chosen.label, detail
        return results[base_index], "INSUFFICIENT_EVIDENCE", detail

    def retrieve_information(self, query: str) -> "InformationResult":
        """
        M12.3 INFO_SEEKING path: the SAME retrieval, relevance gate and passage selection as verify_claim(), but the
        classifier is never called and no verdict is produced. Used when the router reads an input as a request for
        information ("kailan ang enrollment?") rather than a claim, so it is never given SUPPORTED / CONTRADICTED /
        INSUFFICIENT_EVIDENCE.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        evidence = self._gather_evidence(query)
        if evidence.top_evidence is None:
            return InformationResult(query=query, evidence_status=NO_RELEVANT_EVIDENCE,
                                     retrieved_evidence=evidence.retrieved, used_evidence=None,
                                     explanation=self._no_evidence_explanation(evidence.retrieved, "question"))
        top, passage = evidence.top_evidence, evidence.selected_passage
        return InformationResult(
            query=query, evidence_status=RELEVANT_EVIDENCE_FOUND, retrieved_evidence=evidence.retrieved,
            used_evidence=top, selected_passage=passage,
            explanation=(f"The most relevant evidence is '{top.document_id}' (retrieval score={top.score:.3f})."
                         + (f" Shown passage #{passage.passage_index} (passage score={passage.score:.3f})."
                            if passage else "")
                         + " No truth verdict is given for a request for information."),
        )

    # -- shared evidence step ------------------------------------------------------------------------------------
    def _gather_evidence(self, text: str) -> "_Evidence":
        # STEP 1: retrieve candidate evidence. This is the ONLY call
        # into src/retrieval/ -- verification does not reimplement any
        # part of the similarity search itself.
        retrieved = self._retriever.retrieve(text, top_k=self._top_k)

        # STEP 2: decide whether retrieval found anything usable.
        # Two ways to land in "no relevant evidence":
        #   (a) retrieval returned nothing at all (empty corpus, or
        #       nothing matched), or
        #   (b) retrieval returned something, but even its BEST result
        #       is below our relevance threshold.
        if not retrieved or retrieved[0].score < self._relevance_threshold:
            return _Evidence(retrieved=retrieved)

        # Only candidates that themselves clear the relevance gate may supply evidence,
        # so a below-threshold document can never become the NLI premise.
        candidates = [d for d in retrieved if d.score >= self._relevance_threshold]
        if self._date_check:
            candidates = prefer_dated_documents(text, candidates)
        top_evidence = candidates[0]
        selected_passage: Optional[SelectedPassage] = None
        premise = top_evidence.text
        if self._passage_selector is not None:
            chosen = self._passage_selector.select(text, candidates)
            if chosen:
                selected_passage = chosen[0]
                premise = PassageSelector.premise_from(chosen)
                top_evidence = next(d for d in candidates if d.document_id == selected_passage.document_id)
        return _Evidence(retrieved=retrieved, top_evidence=top_evidence, selected_passage=selected_passage,
                         premise=premise)

    def _no_evidence_explanation(self, retrieved: List[RetrievedDocument], noun: str) -> str:
        if not retrieved:
            return f"No sufficiently relevant campus evidence was found for this {noun}."
        return (f"The most relevant evidence found (score={retrieved[0].score:.3f}) "
                f"was below the relevance threshold ({self._relevance_threshold}).")


@dataclass
class _Evidence:
    """Output of the shared retrieval + gate + passage-selection step. top_evidence is None below the gate."""
    retrieved: List[RetrievedDocument]
    top_evidence: Optional[RetrievedDocument] = None
    selected_passage: Optional[SelectedPassage] = None
    premise: Optional[str] = None


@dataclass
class InformationResult:
    """
    Output of ClaimVerifier.retrieve_information(): evidence for a request for information, with NO verdict field.
    used_evidence / selected_passage are None when nothing cleared the relevance gate.
    """
    query: str
    evidence_status: str
    retrieved_evidence: List[RetrievedDocument]
    used_evidence: Optional[RetrievedDocument]
    explanation: str
    selected_passage: Optional[SelectedPassage] = None
