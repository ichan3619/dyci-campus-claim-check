"""
Pretrained NLI classifier (Milestone 6).

Responsibility of this module, and ONLY this module: given a CLAIM and a
piece of EVIDENCE, use a pretrained Natural Language Inference model
(cross-encoder/nli-deberta-v3-small by default) to determine their
relationship, and map the model's native NLI labels onto this project's
three labels.

This module does NOT retrieve evidence, does NOT search a corpus, does
NOT know about the Retriever, and does NOT fetch URLs or access
Facebook. It only ever answers "given these two specific texts, what is
their relationship?" -- exactly mirroring
src/classification/classifier.py's TF-IDF+LR implementation, which this
module is designed to be a drop-in alternative to (both return the same
ClassificationResult and expose the same classify(claim, evidence)
method -- see src/classification/result.py).

============================================================================
ZERO-SHOT, NOT FINE-TUNED (Milestone 6 scope)
============================================================================
This classifier uses the pretrained model AS-IS. It does not fine-tune on
our campus dataset. What the pretrained model already learned: general
English entailment/contradiction/neutral relationships, from the SNLI and
MultiNLI datasets (general-domain sentence pairs -- news, fiction,
government reports, etc. -- not campus announcements specifically).
"Zero-shot" here means: the model has never seen a single example from
our dataset, campus-related or otherwise, and is applied directly.

Fine-tuning (training this model further on OUR labeled claim/evidence
pairs) would let it adapt to campus-specific vocabulary and phrasing
patterns, but requires a meaningfully larger labeled dataset than
currently exists (Milestone 5 established the dev dataset has only 21
classifier-usable examples across 13 groups -- far too few to fine-tune
a transformer responsibly without overfitting). Fine-tuning is
explicitly POSTPONED, not attempted here, and is a natural candidate for
a future milestone once the real dataset exists.

============================================================================
WHAT THE CONFIDENCE SCORE MEANS (and does not mean)
============================================================================
The model outputs three raw scores (logits), one per NLI class, which
are converted to probabilities via softmax -- the same mathematical
operation used for label_scores in the TF-IDF+LR baseline (see
classifier.py). This is a genuine probability distribution over the
model's own three NLI classes: it sums to 1.0, and reflects how strongly
the model's internal representation favors one relationship over the
others.

It is NOT a calibrated real-world probability, and it is absolutely NOT
"the probability that the claim is true." A score of
{SUPPORTED: 0.91, ...} means the model's contextual representation of
the evidence and claim strongly resembles entailment pairs it was
trained on -- nothing more. This is the same distinction already
documented for the baseline classifier and enforced project-wide (see
src/classification/result.py, src/verification/verifier.py).
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.classification.result import ClassificationResult


DEFAULT_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Maps the NLI model's OWN label vocabulary onto this project's three
# labels. Keys are lowercase, stripped versions of whatever string the
# model's config uses -- normalized this way because different NLI model
# families capitalize differently (e.g. "entailment" vs "ENTAILMENT",
# confirmed during Step 2's model survey: typeform/distilbert-base-
# uncased-mnli uses "ENTAILMENT"/"NEUTRAL"/"CONTRADICTION", while the
# cross-encoder/nli-deberta-v3 family uses lowercase). This mapping is
# from LABEL NAME, never from a hardcoded index -- see __init__ below
# for why that distinction is the entire point of this section.
NLI_LABEL_TO_PROJECT_LABEL = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT_EVIDENCE",
}


def build_label_id_mapping(raw_id2label: dict) -> dict:
    """
    Given a model's raw config.id2label (e.g. {0: "contradiction",
    1: "entailment", 2: "neutral"}), return a mapping from integer class
    ID to this project's label strings (e.g. {0: "CONTRADICTED",
    1: "SUPPORTED", 2: "INSUFFICIENT_EVIDENCE"}).

    Pulled out as a standalone, pure function (no torch/transformers
    dependency, just plain dicts) SPECIFICALLY so it can be unit tested
    directly against various id2label orderings without downloading any
    real model weights -- this is the exact piece of logic Milestone 6
    Step 3 requires to be correct and verified, so it deserves to be
    testable in isolation rather than buried inside __init__.

    Raises ValueError if any label isn't recognized, or if the three
    project labels aren't all reachable after mapping (e.g. a
    malformed/non-standard config).
    """
    id_to_project_label = {}
    for raw_id, raw_label in raw_id2label.items():
        normalized = str(raw_label).strip().lower()
        if normalized not in NLI_LABEL_TO_PROJECT_LABEL:
            raise ValueError(
                f"Unrecognized NLI label '{raw_label}' in id2label={raw_id2label}. "
                f"Expected only {sorted(NLI_LABEL_TO_PROJECT_LABEL.keys())} -- this "
                "model may not be a standard 3-class NLI model, and should not be "
                "used with this classifier without reviewing the mapping above."
            )
        id_to_project_label[int(raw_id)] = NLI_LABEL_TO_PROJECT_LABEL[normalized]

    if set(id_to_project_label.values()) != {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError(
            f"id2label={raw_id2label} does not map to all three expected project "
            f"labels. Resulting mapping: {id_to_project_label}"
        )

    return id_to_project_label


class NLIClassifier:
    """
    A pretrained NLI model wrapped to satisfy this project's classifier
    interface: classify(claim, evidence) -> ClassificationResult.

    Usage:
        classifier = NLIClassifier()   # downloads/loads the pretrained model
        result = classifier.classify("claim text", "evidence text")

    Unlike ClaimEvidenceClassifier (the TF-IDF+LR baseline), there is NO
    fit() method here -- the model is pretrained and used as-is (see
    module docstring, "ZERO-SHOT, NOT FINE-TUNED"). Loading happens once,
    in __init__, exactly like the baseline's __init__ builds its
    TfidfVectorizer -- the expensive one-time setup is separated from the
    cheap, repeatable classify() call.
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL_NAME, device: Optional[str] = None):
        """
        model_name: any Hugging Face model repo implementing 3-class NLI
            (entailment/contradiction/neutral). Defaults to
            cross-encoder/nli-deberta-v3-small (see Milestone 6 Step 2
            for why this model was selected over the alternatives
            considered: cross-encoder/nli-deberta-v3-xsmall,
            facebook/bart-large-mnli, typeform/distilbert-base-uncased-mnli).

        device: "cpu", "cuda", or None. If None (the default), CUDA is
            used automatically WHEN AVAILABLE, and CPU is used otherwise
            -- CPU is never a fallback that has to be manually requested,
            satisfying the "CPU is the default, GPU is optional" M6
            requirement structurally, not just as documentation.
        """
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()  # inference mode -- disables dropout etc., we are never training this model

        # --------------------------------------------------------------
        # THE CORE M6 REQUIREMENT: read the model's ACTUAL label mapping
        # from its own config, rather than assuming a fixed index order.
        # --------------------------------------------------------------
        # Step 2's model survey found REAL, confirmed inconsistency across
        # NLI model families:
        #   cross-encoder/nli-deberta-v3-* : {0: contradiction, 1: entailment, 2: neutral}
        #   typeform/distilbert-base-uncased-mnli: {0: ENTAILMENT, 1: NEUTRAL, 2: CONTRADICTION}
        # Hardcoding index positions would silently produce WRONG labels
        # if this classifier were ever pointed at a different checkpoint.
        # Reading config.id2label instead (via build_label_id_mapping,
        # tested independently in tests/test_nli_classifier.py) makes the
        # mapping correct regardless of which specific model_name is used.
        self._id_to_project_label = build_label_id_mapping(self._model.config.id2label)

    def classify(self, claim: str, evidence: str) -> ClassificationResult:
        """
        Classify a single (claim, evidence) pair using the pretrained
        NLI model.

        Input contract matches ClaimEvidenceClassifier.classify() exactly
        (same empty-input validation, same return type) so the rest of
        the pipeline (ClaimVerifier, the M5 evaluation harness) can use
        either classifier interchangeably.

        NLI framing (per Milestone 6 spec): evidence = premise,
        claim = hypothesis. We are asking "does the evidence entail,
        contradict, or say nothing about the claim" -- not the reverse.
        """
        if not claim or not claim.strip():
            raise ValueError("claim must not be empty")

        if not evidence or not evidence.strip():
            raise ValueError(
                "evidence must not be empty -- an empty-evidence case should be "
                "handled as INSUFFICIENT_EVIDENCE by the retrieval stage, before "
                "the classifier is ever called (same contract as "
                "ClaimEvidenceClassifier.classify() -- see classifier.py)"
            )

        # premise = evidence, hypothesis = claim (see module docstring's
        # INPUT CONTRACT discussion and the Milestone 6 spec's examples).
        inputs = self._tokenizer(
            evidence, claim,
            truncation=True, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=0)

        label_scores = {}
        for raw_id, project_label in self._id_to_project_label.items():
            label_scores[project_label] = float(probabilities[raw_id])

        predicted_label = max(label_scores, key=label_scores.get)
        confidence = label_scores[predicted_label]

        return ClassificationResult(
            label=predicted_label,
            confidence=confidence,
            label_scores=label_scores,
        )
