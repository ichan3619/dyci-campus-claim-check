"""
LanguageRoutedClassifier -- picks the NLI model by the language of the CLAIM (M12.3, opt-in).

    English claim            -> the evaluated English model (cross-encoder/nli-deberta-v3-small)
    Tagalog / Taglish claim  -> a multilingual NLI model

WHY
    deberta-v3-small is English-only: on the 7 Taglish development claims it was right 0 times and gave 3 false
    CONTRADICTED verdicts. With MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 for those claims only:
    2/7 right, 1 false CONTRADICTED; the 20 English development claims and the 26 canonical claims are unchanged by
    construction, because they never reach the second model (results/m12_3/m12_3_multilingual_nli_dev.json).
    Neither model card lists Tagalog, the sample is small and written in-house, and the multilingual model is about
    10x slower on CPU (~660 ms vs ~66 ms per claim), which only Taglish claims pay. Confirm on the independent set.

The detector is the router's closed Tagalog word list (src.routing.router.is_tagalog_or_taglish), so the same inputs
the router flags 'tagalog_or_taglish' go to the multilingual model.
"""
from dataclasses import replace
from typing import Callable

from src.classification.result import ClassificationResult
from src.routing.router import is_tagalog_or_taglish


class LanguageRoutedClassifier:
    def __init__(self, english, tagalog, english_name: str, tagalog_name: str,
                 detector: Callable[[str], bool] = is_tagalog_or_taglish):
        self.english, self.tagalog = english, tagalog
        self.english_name, self.tagalog_name = english_name, tagalog_name
        self._detector = detector

    def classify(self, claim: str, evidence: str) -> ClassificationResult:
        use_tagalog = self._detector(claim)
        result = (self.tagalog if use_tagalog else self.english).classify(claim, evidence)
        return replace(result, model=self.tagalog_name if use_tagalog else self.english_name)
