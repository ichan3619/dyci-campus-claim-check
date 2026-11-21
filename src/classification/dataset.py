"""
Compatibility shim (Milestone 5).

The actual filtering/splitting logic MOVED to src/evaluation/splitting.py
in Milestone 5, because it is shared dataset methodology, not something
specific to the TF-IDF+LR classifier -- a future NLI model must use the
exact same split. This file re-exports the same two functions so that
every import written in Milestones 1-4
(`from src.classification.dataset import ...`) keeps working unchanged.

New code (Milestone 5 onward) should import directly from
src.evaluation.splitting instead.
"""

from src.evaluation.splitting import build_classification_examples, split_by_group

__all__ = ["build_classification_examples", "split_by_group"]
