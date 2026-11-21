"""
Serialization of ComparisonReport to disk (Milestone 7 Step 2).

Responsibility of this module, and ONLY this module: take an ALREADY-
COMPUTED ComparisonReport (produced by compare_models(), completely
unmodified) and write it to disk as JSON, alongside enough
reproducibility metadata (dataset path, split seed, split sizes) to know
exactly how it was produced.

This module does NOT compute anything -- no metrics, no model calls, no
comparison logic. It only serializes existing dataclasses
(EvaluationMetrics and PairedPrediction are both plain dataclasses with
JSON-friendly field types, so this is a thin wrapper around
dataclasses.asdict(), not a reimplementation of anything). This keeps
compare_models.py, evaluate.py, and metrics.py completely untouched, per
the Milestone 7 Step 2 constraints.

IMPORTANT: any JSON file written by this module is a PIPELINE VALIDATION
ARTIFACT, not a thesis model-results artifact, whenever the underlying
test set is small (report.is_small_test_set). This label is baked
directly into the JSON content itself (the "run_type" and "warning"
fields), not left to the reader to infer from surrounding context -- so
the file stays self-explanatory even if it's later separated from this
codebase (e.g. attached standalone to a thesis appendix).
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone

from src.evaluation.compare_models import ComparisonReport


def build_report_dict(report: ComparisonReport, *, dataset_path: str, split_seed: int,
                       train_size: int, val_size: int) -> dict:
    """
    Convert a ComparisonReport plus reproducibility metadata into a
    plain, JSON-serializable dict.

    Kept separate from the file-writing step (save_comparison_report,
    below) specifically so this conversion logic can be unit tested
    without touching the filesystem at all.
    """
    run_type = "PIPELINE_VALIDATION_RUN" if report.is_small_test_set else "COMPARISON_RUN"

    warning = (
        "This is a PIPELINE VALIDATION artifact, not a thesis model-results "
        "artifact, while the held-out test set remains small. Re-run this same "
        "script once the real thesis dataset provides a sufficiently large "
        "held-out test set -- THAT run is the meaningful comparison, not this one."
    ) if report.is_small_test_set else (
        "Test set size is above the small-test-set warning threshold; still "
        "review class balance and dataset provenance before treating this as "
        "a final result."
    )

    return {
        "run_type": run_type,
        "warning": warning,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "dataset_path": dataset_path,
            "split_seed": split_seed,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": report.n_test_examples,
            "is_small_test_set": report.is_small_test_set,
        },
        "baseline_metrics": asdict(report.baseline_metrics),
        "nli_metrics": asdict(report.nli_metrics),
        "per_example_predictions": [asdict(p) for p in report.paired_predictions],
        "classifier_selection": {
            "status": "DEFERRED",
            "reason": (
                "The current held-out test set is too small to support a "
                "selection decision -- see is_small_test_set / warning above."
            ),
            "decision_rule_for_future_runs": (
                "Once the real dataset provides a sufficiently large held-out "
                "test set, select the classifier with the higher macro F1 "
                "(this project's established primary metric, per Milestone 5's "
                "methodology -- see docs/methodology.md), using accuracy and "
                "per-class precision/recall/F1 as supporting evidence, not as "
                "the primary basis for selection."
            ),
        },
    }


def save_comparison_report(report: ComparisonReport, output_path: str, *, dataset_path: str,
                            split_seed: int, train_size: int, val_size: int) -> None:
    """
    Write a ComparisonReport to `output_path` as JSON (pretty-printed,
    UTF-8). This function only serializes and writes -- it does not
    compute, re-run, or alter anything about the report it's given.
    """
    report_dict = build_report_dict(
        report,
        dataset_path=dataset_path,
        split_seed=split_seed,
        train_size=train_size,
        val_size=val_size,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
