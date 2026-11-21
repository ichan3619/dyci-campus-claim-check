"""
Shared evaluation metrics (Milestone 5, Step 7).

ONE metrics implementation, used for every model comparison
(TF-IDF+LR today, pretrained NLI later). If each model had its own
metrics code, a subtle difference (e.g. one script computing macro F1
slightly differently) could make two models look different when the
actual difference was in the measurement, not the model.
"""

from dataclasses import dataclass
from typing import List, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

from src.data.schema import VALID_LABELS

LABELS_SORTED = sorted(VALID_LABELS)


@dataclass
class EvaluationMetrics:
    """
    Structured metrics for one evaluation run.

    accuracy: fraction of examples where predicted label == true label.
        Reported, but NOT treated as the primary metric -- see macro_f1.

    macro_f1: F1 score computed PER CLASS, then averaged UNWEIGHTED
        across classes. This is the PRIMARY metric for this project
        (see module docstring in Milestone 3's train_classifier.py for
        the full reasoning): our three classes are not guaranteed to be
        balanced, and a model that ignores a rare class entirely can
        still score well on plain accuracy. Macro F1 does not let that
        happen -- every class counts equally regardless of how many
        examples it has.

    per_class_f1: F1 broken down by individual label, so we can see
        WHICH class a model struggles with, not just an aggregate.

    per_class_precision / per_class_recall: same idea, for precision
        and recall individually -- useful for Step 9's error analysis
        (e.g. "low recall on CONTRADICTED" points at a different
        problem than "low precision on CONTRADICTED").

    confusion_matrix: rows = true label, columns = predicted label,
        in the fixed order given by `labels` (always LABELS_SORTED),
        so matrices from different model runs are directly comparable.

    n_examples: how many examples this was computed over -- always
        report this alongside the numbers above, since a metric from
        3 examples means something very different from one computed
        over 300.
    """
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: List[List[int]]
    labels: List[str]
    n_examples: int


def compute_metrics(true_labels: List[str], predicted_labels: List[str]) -> EvaluationMetrics:
    """
    Compute the full metrics set for one set of predictions.

    Both lists must be the same length and use labels from VALID_LABELS.
    Uses `labels=LABELS_SORTED` and `zero_division=0` throughout so that
    a class with zero predicted/true examples in a tiny dataset produces
    a defined 0.0 instead of crashing or printing a warning -- this
    matters a lot for our small development dataset, where a class can
    easily be entirely absent from a given split.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must be the same length")
    if len(true_labels) == 0:
        raise ValueError("cannot compute metrics over zero examples")

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels, predicted_labels, average="macro", labels=LABELS_SORTED, zero_division=0
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=LABELS_SORTED, zero_division=0
    )

    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS_SORTED)

    return EvaluationMetrics(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        per_class_f1={label: float(score) for label, score in zip(LABELS_SORTED, f1)},
        per_class_precision={label: float(score) for label, score in zip(LABELS_SORTED, precision)},
        per_class_recall={label: float(score) for label, score in zip(LABELS_SORTED, recall)},
        confusion_matrix=matrix.tolist(),
        labels=LABELS_SORTED,
        n_examples=len(true_labels),
    )


def format_metrics_report(metrics: EvaluationMetrics, title: str = "") -> str:
    """
    Human-readable rendering of an EvaluationMetrics -- used by
    train_classifier.py-style scripts so every model prints results in
    an identical, directly-comparable format.
    """
    lines = []
    if title:
        lines.append(f"--- {title} ({metrics.n_examples} examples) ---")
    lines.append(f"Accuracy: {metrics.accuracy:.2%}")
    lines.append(f"Macro F1: {metrics.macro_f1:.4f}")
    lines.append("")
    lines.append("Per-class F1 / Precision / Recall:")
    for label in metrics.labels:
        lines.append(
            f"  {label:24} F1={metrics.per_class_f1[label]:.3f}  "
            f"P={metrics.per_class_precision[label]:.3f}  "
            f"R={metrics.per_class_recall[label]:.3f}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=predicted):")
    header = "                        " + "  ".join(f"{l[:12]:>12}" for l in metrics.labels)
    lines.append(header)
    for label, row in zip(metrics.labels, metrics.confusion_matrix):
        lines.append(f"{label[:22]:>22}  " + "  ".join(f"{v:>12}" for v in row))
    return "\n".join(lines)
