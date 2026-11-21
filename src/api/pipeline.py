"""
Builds and maintains the verification pipeline the API serves. No FastAPI imports, so it is testable on its own.

SETTINGS (environment variables; the defaults are the configuration chosen on the M12.3 development data, see
docs/12_3/m12_3_decisions.md)
    CAMPUS_VERIFY_CLASSIFIER         nli (default) | tfidf_lr
    CAMPUS_VERIFY_RETRIEVER          e5 (default: M12.2 configuration) | tfidf
    CAMPUS_VERIFY_CORPUS             default: real DYCI announcements only (SERVED_CORPUS_SPEC: the test-Page posts plus
                                     the hand-entered announcements); 'dev;data/corpus/mock_facebook_corpus.jsonl' is the
                                     pool the M12 experiments used -- see src/api/corpus_config.py
    CAMPUS_VERIFY_ROUTER             1 (default) | 0
    CAMPUS_VERIFY_CLAIM_REWRITE      optionB (default) | none
    CAMPUS_VERIFY_DATE_CHECK         1 (default) | 0
    CAMPUS_VERIFY_VALUE_CHECK        0 (default) | 1: value check after the date check (src/verification/value_check.py)
    CAMPUS_VERIFY_RERANKER           0 (default) | 1: reorder E5's top 10 with bge-reranker-v2-m3 (src/retrieval/reranker.py)
    CAMPUS_VERIFY_PREMISE_MODE       window (default) | sliding: judge every window of the selected document (verifier.py)
    CAMPUS_VERIFY_EN_QUESTION_REWRITE 1 (default since v1.0; adopted 2026-10-10) | 0: English yes/no question -> statement
                                     (src/normalization/english_question.py)
    CAMPUS_VERIFY_TAGLISH_HYBRID     1 (default since v1.1; adopted 2026-10-10) | 0: evidence-only Tagalog/Taglish responses
                                     rank posts with E5 + TagalogBM25
                                     (RRF, HybridRetriever defaults); English requests and every verdict are unaffected
    CAMPUS_VERIFY_RELEVANCE_CHECK    0 (default) | 1: withhold an English SUPPORTED/CONTRADICTED verdict whose post looks
                                     off-topic (src/verification/relevance_check.py); needs the model file
                                     (CAMPUS_VERIFY_RELEVANCE_MODEL, default models/relevance_check_v1.json) and its
                                     SHA-256 in CAMPUS_VERIFY_RELEVANCE_MODEL_SHA256
    CAMPUS_VERIFY_RELATIVE_DATE_CHECK 1 (default since v1.3; adopted 2026-10-13) | 0: withhold SUPPORTED/CONTRADICTED on a
                                     message with a relative day word
                                     ("tomorrow", "bukas" ...) unless the post used was made today or yesterday
                                     (src/verification/relative_date.py)
    CAMPUS_VERIFY_OTHER_SCHOOL_CHECK 1 (default since v1.2; adopted 2026-10-12) | 0: withhold SUPPORTED/CONTRADICTED on a
                                     message naming another school
                                     when the post used does not mention it (src/verification/other_school.py)
    CAMPUS_VERIFY_UNICODE_FOLD       0 (default) | 1: fold styled letters/digits (src/normalization/fold.py) in post text and in a
                                     processing copy of the message; the original text is still displayed
    CAMPUS_VERIFY_IMAGE              0 (default) | 1: POST /verify/image (Image Check Phase 1): in-memory intake + OCR, the
                                     text read is checked by the unchanged /verify/text logic; no image-authenticity signals
    CAMPUS_VERIFY_VERDICT_LANGUAGES  english (default) | all: which claims may receive SUPPORTED/CONTRADICTED; with
                                     'english', Tagalog/Taglish claims get the closest announcements and no verdict
    CAMPUS_VERIFY_TAGALOG_NLI_MODEL  path or Hugging Face id of the multilingual NLI model for Tagalog/Taglish claims;
                                     default: the pinned model in DEFAULT_TAGALOG_MODEL_DIR if it has been downloaded
                                     (scripts/download_models.py), otherwise none; 'none' turns it off
    CAMPUS_VERIFY_RELOAD_SECONDS     10 (default): how often a request may check the corpus files for changes;
                                     0 = every request, negative = never reload
    CAMPUS_VERIFY_FB_REFRESH_MINUTES 0 (default, off): fetch new posts from the mock Facebook Page this often
                                     (needs META_PAGE_ID and META_ACCESS_TOKEN); new posts are appended to
    CAMPUS_VERIFY_FB_CORPUS          data/corpus/mock_facebook_corpus.jsonl (must be one of the corpus sources)

NEW EVIDENCE WITHOUT A RESTART
    PipelineManager.current() re-reads the corpus when any source file changed (size or modification time), at most
    once per CAMPUS_VERIFY_RELOAD_SECONDS. Models are loaded once and reused; e5 passage vectors are cached in the
    encoder, so a rebuild embeds only the new passages. A failed rebuild (e.g. a half-written file) keeps serving the
    previous pipeline and is reported on /health; it is retried on the next check.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from src.api.corpus_config import CORPUS_ENV_VAR, CorpusSource, load_sources, source_paths
from src.verification.verifier import ClaimVerifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dev", "sample_claims.csv")
FACEBOOK_CORPUS = os.path.join("data", "corpus", "mock_facebook_corpus.jsonl")
# The relevance check's model (written only if its pre-registered criteria are met; see relevance_check.py).
RELEVANCE_MODEL_PATH = "models/relevance_check_v1.json"
# The pool the M12 experiments were run on: the fictional development notices plus the test-Page posts. Tools that
# need "the development data" (add_announcement's near-copy check, the M12.3 rehearsal) use this, never the default.
DEVELOPMENT_CORPUS_SPEC = "dev;" + FACEBOOK_CORPUS.replace(os.sep, "/")
# What the app serves by default: real DYCI announcements only -- the posts mirrored on the project's test Page and
# the ones entered by hand with their original link and date. The fictional development notices are not served,
# because mixing them with real posts produced wrong-document verdicts (docs/12_3/m12_3_decisions.md, 2026-10-07).
SERVED_CORPUS_SPEC = ";".join([FACEBOOK_CORPUS.replace(os.sep, "/"), "data/corpus/independent_announcements.jsonl",
                               "data/corpus/dev2_announcements.jsonl"])
DEFAULT_CORPUS_SPEC = SERVED_CORPUS_SPEC
DEFAULT_TAGALOG_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "campus_verify_models",
                                         "mdeberta-v3-base-xnli-multilingual-nli-2mil7@b5113eb3")
E5_TOP_K, E5_CONTEXT_BEFORE, E5_SCORE_MODE = 3, 1, "passage"      # M12.2 frozen configuration


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip() != "0"


def _tagalog_model_from_env(classifier: str) -> str:
    value = os.environ.get("CAMPUS_VERIFY_TAGALOG_NLI_MODEL", "").strip()
    if value.lower() in ("none", "off", "0"):
        return ""
    if value:
        return value
    return DEFAULT_TAGALOG_MODEL_DIR if classifier == "nli" and os.path.isdir(DEFAULT_TAGALOG_MODEL_DIR) else ""


@dataclass(frozen=True)
class Settings:
    classifier: str = "nli"
    retriever: str = "e5"
    corpus: str = DEFAULT_CORPUS_SPEC
    router: bool = True
    claim_rewrite: str = "optionB"
    date_check: bool = True
    value_check: bool = False
    reranker: bool = False
    premise_mode: str = "window"
    verdict_languages: str = "english"
    en_question_rewrite: bool = True
    taglish_hybrid: bool = True
    relevance_check: bool = False
    relative_date_check: bool = True
    other_school_check: bool = True
    unicode_fold: bool = False
    image_check: bool = False
    relevance_model: str = RELEVANCE_MODEL_PATH
    relevance_model_sha256: str = ""
    tagalog_nli_model: str = ""
    reload_seconds: float = 10.0
    fb_refresh_minutes: float = 0.0
    fb_corpus: str = FACEBOOK_CORPUS

    @classmethod
    def from_env(cls) -> "Settings":
        classifier = os.environ.get("CAMPUS_VERIFY_CLASSIFIER", "nli")
        return cls(
            classifier=classifier,
            retriever=os.environ.get("CAMPUS_VERIFY_RETRIEVER", "e5"),
            corpus=os.environ.get(CORPUS_ENV_VAR, "").strip() or DEFAULT_CORPUS_SPEC,
            router=_flag("CAMPUS_VERIFY_ROUTER", True),
            claim_rewrite=os.environ.get("CAMPUS_VERIFY_CLAIM_REWRITE", "optionB"),
            date_check=_flag("CAMPUS_VERIFY_DATE_CHECK", True),
            value_check=_flag("CAMPUS_VERIFY_VALUE_CHECK", False),
            reranker=_flag("CAMPUS_VERIFY_RERANKER", False),
            premise_mode=os.environ.get("CAMPUS_VERIFY_PREMISE_MODE", "window").strip() or "window",
            verdict_languages=os.environ.get("CAMPUS_VERIFY_VERDICT_LANGUAGES", "english").strip() or "english",
            en_question_rewrite=_flag("CAMPUS_VERIFY_EN_QUESTION_REWRITE", True),
            taglish_hybrid=_flag("CAMPUS_VERIFY_TAGLISH_HYBRID", True),
            relevance_check=_flag("CAMPUS_VERIFY_RELEVANCE_CHECK", False),
            relative_date_check=_flag("CAMPUS_VERIFY_RELATIVE_DATE_CHECK", True),
            other_school_check=_flag("CAMPUS_VERIFY_OTHER_SCHOOL_CHECK", True),
            unicode_fold=_flag("CAMPUS_VERIFY_UNICODE_FOLD", False),
            image_check=_flag("CAMPUS_VERIFY_IMAGE", False),
            relevance_model=os.environ.get("CAMPUS_VERIFY_RELEVANCE_MODEL", "").strip() or RELEVANCE_MODEL_PATH,
            relevance_model_sha256=os.environ.get("CAMPUS_VERIFY_RELEVANCE_MODEL_SHA256", "").strip(),
            tagalog_nli_model=_tagalog_model_from_env(classifier),
            reload_seconds=float(os.environ.get("CAMPUS_VERIFY_RELOAD_SECONDS", "10")),
            fb_refresh_minutes=float(os.environ.get("CAMPUS_VERIFY_FB_REFRESH_MINUTES", "0")),
            fb_corpus=os.environ.get("CAMPUS_VERIFY_FB_CORPUS", FACEBOOK_CORPUS),
        )

    def validate(self) -> None:
        if self.classifier not in ("nli", "tfidf_lr"):
            raise RuntimeError(f"Unknown CAMPUS_VERIFY_CLASSIFIER value: {self.classifier!r} (expected 'nli' or 'tfidf_lr')")
        if self.retriever not in ("e5", "tfidf"):
            raise RuntimeError(f"Unknown CAMPUS_VERIFY_RETRIEVER value: {self.retriever!r} (expected 'e5' or 'tfidf')")
        if self.claim_rewrite not in ("none", "optionB"):
            raise RuntimeError(f"Unknown CAMPUS_VERIFY_CLAIM_REWRITE value: {self.claim_rewrite!r} (expected 'none' or 'optionB')")
        if self.verdict_languages not in ("english", "all"):
            raise RuntimeError(f"Unknown CAMPUS_VERIFY_VERDICT_LANGUAGES value: {self.verdict_languages!r} (expected 'english' or 'all')")
        if self.premise_mode not in ("window", "sliding"):
            raise RuntimeError(f"Unknown CAMPUS_VERIFY_PREMISE_MODE value: {self.premise_mode!r} (expected 'window' or 'sliding')")
        if self.reranker and self.retriever != "e5":
            raise RuntimeError("CAMPUS_VERIFY_RERANKER requires CAMPUS_VERIFY_RETRIEVER=e5")
        if self.relevance_check and not self.relevance_model_sha256:
            raise RuntimeError("CAMPUS_VERIFY_RELEVANCE_CHECK=1 needs CAMPUS_VERIFY_RELEVANCE_MODEL_SHA256 (the adopted "
                               "model's SHA-256); no relevance model has been adopted by default")
        if self.tagalog_nli_model and self.classifier != "nli":
            raise RuntimeError("CAMPUS_VERIFY_TAGALOG_NLI_MODEL requires CAMPUS_VERIFY_CLASSIFIER=nli")
        if self.claim_rewrite == "optionB":
            from src.normalization.question_form import rewrite_question
            rewrite_question("ok?")          # fails startup loudly if the frozen Option B file was changed
        if self.fb_refresh_minutes > 0:
            target = os.path.normcase(os.path.abspath(os.path.join(PROJECT_ROOT, self.fb_corpus)))
            watched = [os.path.normcase(p) for p in source_paths(self.corpus, PROJECT_ROOT, DEV_DATASET_PATH)]
            if target not in watched:
                raise RuntimeError(f"CAMPUS_VERIFY_FB_CORPUS {self.fb_corpus!r} is not one of the corpus sources "
                                   f"({self.corpus!r}); fetched posts would never be searched.")

    @property
    def tagalog_model_label(self) -> str:
        return os.path.basename(self.tagalog_nli_model.rstrip("/\\")) if self.tagalog_nli_model else "none"


# ---- one-time builders (models) -------------------------------------------------------------------------------------
def build_classifier(settings: Settings):
    """The classifier, loaded once per process and reused across corpus reloads."""
    if settings.classifier == "nli":
        from src.classification.nli_classifier import DEFAULT_NLI_MODEL_NAME, NLIClassifier
        if settings.tagalog_nli_model:
            from src.classification.language_routed import LanguageRoutedClassifier
            return LanguageRoutedClassifier(NLIClassifier(), NLIClassifier(settings.tagalog_nli_model),
                                            english_name=DEFAULT_NLI_MODEL_NAME, tagalog_name=settings.tagalog_model_label)
        return NLIClassifier()
    # tfidf_lr: fit on the dev dataset exactly as scripts/verify_claim.py does (no trained model is persisted).
    from src.classification.classifier import ClaimEvidenceClassifier
    from src.classification.dataset import build_classification_examples, split_by_group
    from src.data.loader import load_dataset
    load_result = load_dataset(DEV_DATASET_PATH)
    if load_result.errors:
        raise RuntimeError(f"Dev dataset failed to load cleanly, cannot fit ClaimEvidenceClassifier: {load_result.errors}")
    train, _val, _test = split_by_group(build_classification_examples(load_result.records))
    classifier = ClaimEvidenceClassifier()
    classifier.fit([e["claim"] for e in train], [e["evidence"] for e in train], [e["label"] for e in train])
    return classifier


def build_encoder(settings: Settings):
    if settings.retriever != "e5":
        return None
    from src.retrieval.e5_retriever import E5Encoder
    return E5Encoder()


def build_reranker(settings: Settings):
    """The cross-encoder reranker, loaded once, or None when CAMPUS_VERIFY_RERANKER is off."""
    if not settings.reranker:
        return None
    from src.retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


def build_verifier(documents: List[dict], classifier, encoder, settings: Settings,
                   reranker=None) -> Tuple[object, ClaimVerifier]:
    """(retriever, verifier) over `documents`. Cheap when the models are already loaded. With a reranker, the E5
    retriever is wrapped so that only the order of its results changes (src/retrieval/reranker.py)."""
    if encoder is None:
        from src.retrieval.retriever import Retriever
        retriever = Retriever(documents)
        return retriever, ClaimVerifier(retriever, classifier, date_check=settings.date_check,
                                        value_check=settings.value_check)
    from src.retrieval.e5_retriever import E5Retriever
    from src.retrieval.passage_selector import PassageSelector
    retriever = E5Retriever(documents, encoder=encoder, score_mode=E5_SCORE_MODE)
    if reranker is not None:
        from src.retrieval.reranker import RerankingRetriever
        retriever = RerankingRetriever(retriever, reranker)
    selector = PassageSelector(encoder, passage_source=retriever, context_before=E5_CONTEXT_BEFORE)
    return retriever, ClaimVerifier(retriever, classifier, top_k=E5_TOP_K, passage_selector=selector,
                                    date_check=settings.date_check, value_check=settings.value_check,
                                    premise_mode=settings.premise_mode)


def build_evidence_verifier(documents: List[dict], classifier, encoder, settings: Settings) -> ClaimVerifier:
    """The verifier behind EVIDENCE-ONLY Tagalog/Taglish responses when CAMPUS_VERIFY_TAGLISH_HYBRID is on: the same base
    retriever build_verifier makes, wrapped in HybridRetriever with all its existing defaults (TagalogBM25 defaults,
    reciprocal rank fusion, gate_protect). The API only ever calls retrieve_information() on it, never verify_claim()."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    base, base_verifier = build_verifier(documents, classifier, encoder, settings)
    # Same relevance threshold as the main verifier (E5: 0.833), so the 'closest announcement' / 'low match' label rule
    # is today's rule.
    hybrid = HybridRetriever(documents, dense=base, relevance_threshold=base_verifier._relevance_threshold)
    if encoder is None:
        return ClaimVerifier(hybrid, classifier, date_check=settings.date_check)
    from src.retrieval.passage_selector import PassageSelector
    selector = PassageSelector(encoder, passage_source=hybrid, context_before=E5_CONTEXT_BEFORE)
    return ClaimVerifier(hybrid, classifier, top_k=E5_TOP_K, passage_selector=selector, date_check=settings.date_check)


# ---- the live pipeline ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pipeline:
    documents: List[dict]
    sources: List[CorpusSource]
    retriever: object
    verifier: ClaimVerifier
    loaded_at: str
    # Evidence-only verifier for Tagalog/Taglish responses (CAMPUS_VERIFY_TAGLISH_HYBRID); None when the flag is off.
    evidence_verifier: Optional[ClaimVerifier] = None
    # Relevance check for English verdicts (CAMPUS_VERIFY_RELEVANCE_CHECK); None when the flag is off.
    relevance_checker: Optional[object] = None
    # CAMPUS_VERIFY_UNICODE_FOLD: {document_id: original text} for display; None when the flag is off (documents are
    # then the stored text, unchanged). When on, `documents` hold the folded text used for all processing.
    original_texts: Optional[dict] = None

    @property
    def corpus_label(self) -> str:
        return "+".join(s.label for s in self.sources)


@dataclass
class RefreshStatus:
    enabled: bool = False
    interval_minutes: float = 0.0
    last_run: Optional[str] = None
    last_added: Optional[int] = None
    last_error: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PipelineManager:
    def __init__(self, settings: Settings, classifier, encoder, project_root: str = PROJECT_ROOT,
                 dev_dataset_path: str = DEV_DATASET_PATH, clock=time.monotonic, reranker=None):
        self.settings, self.classifier, self.encoder, self.reranker = settings, classifier, encoder, reranker
        self.relevance_model = None
        if settings.relevance_check:                    # loaded once; refused unless its SHA-256 matches
            from src.verification.relevance_check import load_model
            path = settings.relevance_model if os.path.isabs(settings.relevance_model) else                 os.path.join(project_root, settings.relevance_model)
            self.relevance_model = load_model(path, settings.relevance_model_sha256)
        self._root, self._dev, self._clock = project_root, dev_dataset_path, clock
        self._paths = source_paths(settings.corpus, project_root, dev_dataset_path)
        self._reload_lock, self._refresh_lock = threading.Lock(), threading.Lock()
        self._pipeline: Optional[Pipeline] = None
        self._fingerprint = None
        self._last_check = float("-inf")
        self.reload_error: Optional[str] = None
        self.refresh = RefreshStatus(enabled=settings.fb_refresh_minutes > 0, interval_minutes=settings.fb_refresh_minutes)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reload()                                   # the first build must succeed: fail startup loudly otherwise

    # -- corpus ------------------------------------------------------------------------------------------------
    def current(self) -> Pipeline:
        """The pipeline to serve this request, rebuilt first if a corpus file changed since the last check."""
        if self.settings.reload_seconds >= 0 and self._clock() - self._last_check >= self.settings.reload_seconds:
            if self._reload_lock.acquire(blocking=False):      # another request is already checking: serve current
                try:
                    self._last_check = self._clock()
                    if self._fingerprint_now() != self._fingerprint:
                        self._reload_locked()
                finally:
                    self._reload_lock.release()
        return self._pipeline

    def reload(self) -> bool:
        with self._reload_lock:
            return self._reload_locked()

    def _reload_locked(self) -> bool:
        fingerprint = self._fingerprint_now()
        try:
            documents, sources = load_sources(self.settings.corpus, self._root, self._dev)
            originals = None
            if self.settings.unicode_fold:                # fold for processing; keep the stored text for display
                from src.normalization.fold import fold
                originals = {d["document_id"]: d["text"] for d in documents}
                documents = [dict(d, text=fold(d["text"])) for d in documents]
            retriever, verifier = build_verifier(documents, self.classifier, self.encoder, self.settings, self.reranker)
            evidence = (build_evidence_verifier(documents, self.classifier, self.encoder, self.settings)
                        if self.settings.taglish_hybrid else None)
            relevance = None
            if self.relevance_model is not None:
                from src.verification.relevance_check import RelevanceChecker
                relevance = RelevanceChecker(documents, self.relevance_model)
        except Exception as e:                           # keep serving the previous pipeline; retried on next check
            if self._pipeline is None:
                raise
            self.reload_error = f"{_now()}: {e}"
            logger.warning("Corpus reload failed, still serving the previous corpus: %s", e)
            return False
        self._pipeline = Pipeline(documents, sources, retriever, verifier, _now(), evidence, relevance, originals)
        self._fingerprint, self.reload_error = fingerprint, None
        return True

    def _fingerprint_now(self):
        out = []
        for p in self._paths:
            try:
                st = os.stat(p)
                out.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                out.append((p, None, None))
        return tuple(out)

    def temp_verifier(self, documents: List[dict]) -> ClaimVerifier:
        """A verifier over request-scoped documents (/verify/url), with the same models and settings."""
        return build_verifier(documents, self.classifier, self.encoder, self.settings, self.reranker)[1]

    # -- Facebook ----------------------------------------------------------------------------------------------
    def refresh_facebook(self) -> RefreshStatus:
        """Fetch the mock Page once, append new posts to the Facebook corpus file, and reload if anything was added."""
        from src.ingestion.facebook_page import credentials_from_env, explain_fetch_error, refresh_from_mock_page
        with self._refresh_lock:
            creds = credentials_from_env()
            self.refresh.last_run = _now()
            if not creds:
                self.refresh.last_added, self.refresh.last_error = None, "META_PAGE_ID / META_ACCESS_TOKEN not set"
                return self.refresh
            try:
                result, _adapter = refresh_from_mock_page(os.path.join(self._root, self.settings.fb_corpus), *creds)
            except Exception as e:                       # never let a fetch problem take the API down
                self.refresh.last_added, self.refresh.last_error = None, explain_fetch_error(str(e), creds[1])
                return self.refresh
            self.refresh.last_added, self.refresh.last_error = result.added, None
        if result.added:
            self.reload()
        return self.refresh

    def start_facebook_refresh(self) -> None:
        if not self.refresh.enabled or self._thread is not None:
            return

        def loop():
            while not self._stop.is_set():
                self.refresh_facebook()
                self._stop.wait(self.settings.fb_refresh_minutes * 60)

        self._thread = threading.Thread(target=loop, name="facebook-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
