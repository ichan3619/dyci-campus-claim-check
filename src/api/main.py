"""
Campus Verify HTTP API (M10.1 foundation; M12.3 input handling and live corpus).

Responsibility of this module, and ONLY this module: expose the existing pipeline over HTTP. It contains no retrieval,
classification or verdict logic -- those live in src/retrieval/, src/classification/, src/verification/ -- and no
pipeline construction, which lives in src/api/pipeline.py (settings, model loading, live corpus reload, optional
Facebook refresh). This module only:

    1. routes the raw input (CHECKABLE_CLAIM -> verify, INFO_SEEKING -> evidence without a verdict),
    2. translates requests into pipeline calls and results into response models, and
    3. translates pipeline exceptions into HTTP status codes, never into a false verdict.

Run locally with:
    uvicorn src.api.main:app --reload
"""

import contextvars
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.api.pipeline import (DEV_DATASET_PATH, PROJECT_ROOT, PipelineManager, Settings, build_classifier, build_encoder,
                              build_reranker)
from src.api.protection import Busy, ConcurrencyGate, ProtectionSettings, RateLimiter, api_key_ok, client_key
from src.api import answer_reuse, school_scope
from starlette.concurrency import run_in_threadpool
from src.normalization.fold import fold
from src.api.date_notice import date_notice
from src.verification import other_school, relative_date
from src.api.source_info import build_source_info, info_for_fetched_page
from src.api.schemas import (
    MAX_CLAIM_CHARS, TOO_SHORT_MESSAGE, ImageCheckResponse, OCRModel, SentenceCheckModel, SentenceSummaryModel,
    has_enough_letters,
    ClassificationResultModel, CorpusSourceModel, FacebookRefreshModel, HealthResponse, RetrievedDocumentModel,
    SelectedPassageModel, VerificationResultModel, VerifyTextRequest, VerifyUrlRequest,
    AccountActiveRequest, DeleteAccountRequest, GoogleSignInRequest, HistoryPreferenceRequest,
    SourceAddRequest, SourceCutoffRequest, SourceTokenRequest, SourceUpdateRequest,
)
from src.api.ssrf_guard import UnsafeUrlError, check_redirect_chain_is_safe
from src.ingestion.adapters import WebsiteFetchError, WebsiteSourceAdapter
from src.ingestion.evidence import NormalizedEvidence
from src.routing import INFO_SEEKING, ROUTER_VERSION, ClaimRouter
from src.api.verdict_policy import withhold_for_language as _withhold_for_language
from src.normalization import english_question


def withhold_for_language(text: str) -> bool:
    """The verdict gate's language check, with rule R-e while the picture check runs a poster's text with rule R on
    (src/image/poster_cleanup.py): one stray extra-list word in a longer English text does not withhold the verdict."""
    if not _withhold_for_language(text):
        return False
    from src.image import poster_language     # rule G (CAMPUS_VERIFY_POSTER_LANGUAGE, picture check only)
    if poster_language.released_by_rule(text):
        return False
    from src.image.poster_cleanup import POSTER_ACTIVE, language_ok
    return not (POSTER_ACTIVE.get() and language_ok(text))
from src.verification.verifier import ClaimVerifier, InformationResult, VerificationResult

# Surfaced on every response and on /health so a client never mistakes this for a validated, production evidence base.
DATA_STATUS = "development"

# Read once at import; tests replace them with dataclasses.replace(...) before starting the app.
SETTINGS = Settings.from_env()
PROTECTION = ProtectionSettings.from_env()      # rate limits, concurrency cap, body size, optional API key

# CORS (M10.2): the local frontend dev origins only, never "*". CORS controls which BROWSER origins may read responses;
# it is not access control (there is no authentication in this prototype).
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CAMPUS_VERIFY_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

# Report-only Taglish normalizer output (Sprint 2), off by default; it never changes a verdict.
NORMALIZE_REPORT_ENABLED = os.environ.get("CAMPUS_VERIFY_NORMALIZE", "0") == "1"

state: dict = {}
_router = ClaimRouter()


def _build_manager(settings, classifier) -> PipelineManager:
    """The corpus manager. With both scale settings off (the default) this is exactly the v1.6 construction.
    Scale plan S1 (docs/plans/scale_plan.md):
      CAMPUS_VERIFY_VECTOR_STORE=<folder>  saved passage vectors (src/retrieval/vector_store.py); relative to the project
      CAMPUS_VERIFY_BACKGROUND_RELOAD=1    rebuilds after a corpus change run in a background thread
                                           (src/api/background_reload.py), never inside a request"""
    from src.retrieval import vector_store
    from src.api.background_reload import BACKGROUND_ENV
    store = os.environ.get(vector_store.ENV, "").strip()
    background = os.environ.get(BACKGROUND_ENV, "0").strip() == "1"
    if not store and not background:
        return PipelineManager(settings, classifier, build_encoder(settings), reranker=build_reranker(settings))
    from src.api.background_reload import ScalablePipelineManager
    if settings.retriever != "e5":
        encoder = build_encoder(settings)               # None: TF-IDF has no vectors to save
    elif store:
        encoder = vector_store.PersistentE5Encoder(store if os.path.isabs(store) else os.path.join(PROJECT_ROOT, store))
    else:
        encoder = vector_store.LockedE5Encoder()
    return ScalablePipelineManager(settings, classifier, encoder, reranker=build_reranker(settings), background=background)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = SETTINGS
    settings.validate()
    from src.api.clock_check import timezone_warning
    clock_warning = timezone_warning()                       # scale plan S7: the date rules expect Philippine time
    if clock_warning:
        import logging
        logging.getLogger("campus_verify").warning(clock_warning)
    classifier = build_classifier(settings)                 # models are loaded once and survive corpus reloads
    manager = _build_manager(settings, classifier)
    manager.start_facebook_refresh()                         # no-op unless CAMPUS_VERIFY_FB_REFRESH_MINUTES > 0
    scheduler = _sources_scheduler()                         # no-op unless CAMPUS_VERIFY_SOURCES_FETCH_MINUTES > 0
    website = _website_scheduler()                           # no-op unless CAMPUS_VERIFY_WEBSITE=1 (rule W)
    protection = PROTECTION
    state.update(settings=settings, manager=manager, normalizer=None, protection=protection,
                 limiters={"text": RateLimiter(protection.rate_per_minute), "url": RateLimiter(protection.url_rate_per_minute),
                           "image": RateLimiter(protection.image_rate_per_minute)},
                 gate=ConcurrencyGate(protection.max_concurrent, protection.queue_seconds))
    if settings.image_check:
        _photo_index()                                       # build the photo index now, not on the first upload
    yield
    if website is not None:
        website.stop()
    if scheduler is not None:
        scheduler.stop()
    manager.stop()
    state.clear()


app = FastAPI(
    title="Campus Verify API",
    description=(
        "Thin HTTP layer over the campus-verify pipeline (routing -> retrieval -> classification -> verification). "
        "DEVELOPMENT DATA -- see /health and each response's data_status field."
    ),
    lifespan=lifespan,
)
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}
# CAMPUS_VERIFY_FRAME_ANCESTORS (default empty = no site may frame the page): space-separated origins allowed to show the
# page in a frame, e.g. "https://huggingface.co" (Hugging Face Spaces shows apps inside a frame). When set, the
# X-Frame-Options DENY header is replaced by Content-Security-Policy frame-ancestors 'self' <origins>.
FRAME_ENV = "CAMPUS_VERIFY_FRAME_ANCESTORS"
# CAMPUS_VERIFY_REQUEST_LOG=1 (default off): one log line per request with method, path, status and milliseconds only.
# Never the client address, the message, the link, the picture or any query string (use --no-access-log with uvicorn).
REQUEST_LOG_ENV = "CAMPUS_VERIFY_REQUEST_LOG"
# CAMPUS_VERIFY_HARDENING=1 (default off; on in start_public.ps1; docs/evaluation/system_design_review.md priority 2):
#   - Content-Security-Policy: scripts, styles, frames and connections only from this site and Google's sign-in
#     (accounts.google.com/gsi), fonts and images from this site (plus data:/blob: previews), no plugins, no framing
#   - Strict-Transport-Security on HTTPS (and any non-local host, which the public tunnel always serves over HTTPS)
#   - a state-changing request (POST/PUT/PATCH/DELETE) whose Origin header names another site is refused (HTTP 403).
#     Allowed: this request's own host, CAMPUS_VERIFY_PUBLIC_ORIGIN (the public address, because a tunnel may not pass
#     it on) and the CORS origins. Requests without an Origin header (scripts, tests) are not affected.
HARDENING_ENV = "CAMPUS_VERIFY_HARDENING"
PUBLIC_ORIGIN_ENV = "CAMPUS_VERIFY_PUBLIC_ORIGIN"
CSP = ("default-src 'self'; script-src 'self' https://accounts.google.com/gsi/client; "
       "style-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/style; "
       "img-src 'self' data: blob: https://accounts.google.com https://*.googleusercontent.com; font-src 'self'; "
       "connect-src 'self' https://accounts.google.com/gsi/; frame-src https://accounts.google.com/gsi/; "
       "object-src 'none'; base-uri 'self'; form-action 'self'")
HSTS = "max-age=15552000"


def _request_logger() -> logging.Logger:
    log = logging.getLogger("campus_verify.requests")
    if not log.handlers:                                   # uvicorn configures its own loggers only
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def _hardening() -> bool:
    return os.environ.get(HARDENING_ENV, "0").strip() == "1"


def _security_headers() -> dict:
    origins = os.environ.get(FRAME_ENV, "").split()
    headers = dict(SECURITY_HEADERS)
    ancestors = None
    if origins:
        headers.pop("X-Frame-Options", None)
        ancestors = "frame-ancestors 'self' " + " ".join(origins)
    if _hardening():
        headers["Content-Security-Policy"] = CSP + "; " + (ancestors or "frame-ancestors 'none'")
    elif ancestors:
        headers["Content-Security-Policy"] = ancestors
    return headers


def _origin_allowed(request: Request) -> bool:
    """CAMPUS_VERIFY_HARDENING: a browser's cross-site POST is refused. No Origin header (not a browser form) passes."""
    origin = (request.headers.get("origin") or "").rstrip("/")
    if not origin:
        return True
    host = request.headers.get("host", "")
    allowed = {f"https://{host}", f"http://{host}", *ALLOWED_ORIGINS}
    public = os.environ.get(PUBLIC_ORIGIN_ENV, "").strip().rstrip("/")
    if public:
        allowed.add(public)
    return origin in allowed


# Public deployment (pre-deployment review L3, owner 2026-10-25): CAMPUS_VERIFY_PUBLIC=1 hides the API documentation
# pages and trims /health to what the page needs. Default 0 (development): everything is shown, as before. Read per
# request, so it can be tested without re-importing the app.
PUBLIC_ENV = "CAMPUS_VERIFY_PUBLIC"
DEV_PAGES = ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")


def _public() -> bool:
    return os.environ.get(PUBLIC_ENV, "0").strip() == "1"


def _refuse(status: int, detail: str, headers=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)


STATS_KINDS = {"/verify/text": "text", "/verify/url": "link", "/verify/image": "picture"}


def _stats_enabled() -> bool:
    from src.api import stats
    return stats.enabled()


def _stats_counter():
    """The dashboard's counts (CAMPUS_VERIFY_STATS), created on first use: in the SQLite database when CAMPUS_VERIFY_DB_FILE
    is set (importing CAMPUS_VERIFY_STATS_FILE once), otherwise in CAMPUS_VERIFY_STATS_FILE if set."""
    from src.api import db, stats
    if state.get("stats") is None:
        json_file = os.environ.get(stats.FILE_ENV, "").strip() or None
        if _database() is not None:
            state["stats"] = stats.SqliteCounter(_database(), import_json=json_file)
        else:
            state["stats"] = stats.Counter(json_file)
    return state["stats"]


def _database():
    """The app's SQLite database (CAMPUS_VERIFY_DB_FILE), opened on first use; None when it is not set."""
    from src.api import db
    if state.get("db") is None and db.path_from_env():
        state["db"] = db.Database(db.path_from_env())
    return state.get("db")


def _current_user(request: Request):
    """The signed-in DYCI account for this request (CAMPUS_VERIFY_AUTH=google), or None."""
    from src.api import auth
    if not auth.enabled() or _database() is None:
        return None
    return auth.user_for_token(_database(), request.cookies.get(auth.COOKIE))


@app.middleware("http")
async def protect(request: Request, call_next):
    """API key, request size and per-client rate limits for /verify/*; security headers on every response.
    Registered BEFORE the CORS middleware, so CORS wraps it and refusals (401/411/413/429) stay readable by the page."""
    protection, path = state.get("protection"), request.url.path
    started = time.perf_counter()
    if path in DEV_PAGES and _public():
        return _refuse(404, "Not Found")
    if _hardening() and request.method in ("POST", "PUT", "PATCH", "DELETE") and not _origin_allowed(request):
        return _refuse(403, "This request came from another website and was refused.")
    user = _current_user(request) if request.method == "POST" and path.startswith("/verify/") else None
    if request.method == "POST" and path.startswith("/verify/"):
        from src.api import auth
        if auth.required() and user is None:
            return _refuse(401, f"Please sign in with your DYCI Google account (@{auth.domain()}) to use the checker.")
    if protection is not None and request.method == "POST" and path.startswith("/verify/"):
        if not api_key_ok(protection.api_key, request.headers.get("x-api-key")):
            return _refuse(401, "Missing or wrong API key (header X-API-Key).")
        length = request.headers.get("content-length")
        # Image Check: its own body limit and rate bucket, ONLY while CAMPUS_VERIFY_IMAGE is on; otherwise /verify/image
        # is treated exactly as in v1.3 (16 KB limit, text bucket, then 404).
        image_route = path == "/verify/image" and getattr(state.get("settings"), "image_check", False)
        limit = protection.max_image_bytes if image_route else protection.max_body_bytes
        if limit > 0:
            if length is None or not length.isdigit():
                return _refuse(411, "Send the request with a Content-Length header.")
            if int(length) > limit:
                return _refuse(413, f"Request body too large (limit {limit} bytes).")
        bucket = "image" if image_route else ("url" if path == "/verify/url" else "text")
        # A signed-in DYCI account gets its own budget (system design review, priority 1); others share by address.
        who = f"user:{user['id']}" if user else client_key({k.lower(): v for k, v in request.headers.items()},
                                                          request.client.host if request.client else None,
                                                          protection.trusted_proxies)
        allowed, retry = state["limiters"][bucket].check(f"{bucket}:{who}")
        if not allowed:
            return _refuse(429, f"Too many requests. Try again in {retry} seconds.", {"Retry-After": str(retry)})
    from src.api import history
    # CAMPUS_VERIFY_HISTORY (default off): "My checks", only for a signed-in account that turned history on.
    keep = user is not None and bool(user["history_enabled"]) and history.enabled() and path in STATS_KINDS
    sent = None
    if keep and path != "/verify/image":
        try:
            sent = json.loads(await request.body())                # Starlette replays the body to the route
        except ValueError:
            sent = None
    response = await call_next(request)
    if request.method == "POST" and path in STATS_KINDS and response.status_code in (200, 502):
        from src.api import stats
        if stats.enabled() or keep:
            # CAMPUS_VERIFY_STATS (default off): count the kind and outcome only; the body is passed on unchanged.
            body = b"".join([chunk async for chunk in response.body_iterator])
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
            counted = stats.outcome_for(STATS_KINDS[path], response.status_code, parsed)
            if counted is not None and stats.enabled():
                _stats_counter().record(STATS_KINDS[path], *counted)
            if counted is not None and keep:
                entry = history.entry_for(STATS_KINDS[path], counted[0], sent if isinstance(sent, dict) else None,
                                          parsed if isinstance(parsed, dict) else None)
                if entry is not None:
                    history.save(_database(), user["id"], entry)
            response = Response(content=body, status_code=response.status_code, headers=dict(response.headers),
                                media_type=response.media_type)
    for name, value in _security_headers().items():
        response.headers.setdefault(name, value)
    if _hardening() and _https(request):
        response.headers.setdefault("Strict-Transport-Security", HSTS)
    if os.environ.get(REQUEST_LOG_ENV, "0").strip() == "1":
        _request_logger().info("%s %s %s %.0fms", request.method, path, response.status_code,
                               (time.perf_counter() - started) * 1000)
    if path.startswith("/verify/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.method == "GET" and (path.endswith((".html", ".css", ".js")) or path.endswith("/")):
        # The page's own files: the browser asks again on each visit (an unchanged file comes back as a short "not
        # modified"), so a student never runs an old page against a new server after an update.
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["Retry-After"],
)


def _gated(fn, *args, **kwargs):
    """Run model work in one of the limited slots; HTTP 503 (never a verdict) when none frees up in time."""
    try:
        with state["gate"].slot():
            return fn(*args, **kwargs)
    except Busy:
        raise HTTPException(status_code=503, detail="The checker is busy. Try again in a few seconds.",
                            headers={"Retry-After": "5"})


# ---- conversion helpers (pure re-shaping) -------------------------------------------------------------------------
def _retrieved_document_to_model(doc, sources=None) -> RetrievedDocumentModel:
    info = (sources or {}).get(doc.document_id)
    extra = {} if info is None else {"published_date": info.published_date or None, "original_url": info.original_url or None,
                                     "original_date": info.original_date or None, "synthetic": info.synthetic,
                                     "provenance": info.provenance or None}
    return RetrievedDocumentModel(document_id=doc.document_id, score=doc.score, text=doc.text,
                                  source_type=doc.source_type, source_url=doc.source_url, **extra)


def _classification_result_to_model(result) -> Optional[ClassificationResultModel]:
    if result is None:
        return None
    return ClassificationResultModel(label=result.label, confidence=result.confidence,
                                     label_scores=result.label_scores, model=getattr(result, "model", None))


def _selected_passage_to_model(passage) -> Optional[SelectedPassageModel]:
    if passage is None:
        return None
    return SelectedPassageModel(document_id=passage.document_id, text=passage.text, score=passage.score,
                                passage_index=passage.passage_index, context_before_text=passage.context_before_text)


def _verification_result_to_model(result: VerificationResult, claim: str, route=None, claim_rewrite=None,
                                  normalization=None, sources=None, relevance_check=None,
                                  relative_date_check=None, other_school_check=None,
                                  notice_message: Optional[str] = None, post_age_notice=None,
                                  wrong_post_check=None, tense_guard_check=None,
                                  relative_resolution=None, tagalog_reading=None,
                                  tagalog_topic_check=None, tagalog_translation=None, holiday_notice=None,
                                  holiday_check=None, class_day_check=None,
                                  question_focus=None, same_day_advisory=None,
                                  same_day_shown=None, same_day_lookup=None,
                                  header_date_check=None, same_day_judge=None) -> VerificationResultModel:
    message = claim if notice_message is None else notice_message      # folded copy when CAMPUS_VERIFY_UNICODE_FOLD
    return VerificationResultModel(
        claim=claim,
        verdict=result.verdict,
        evidence_status=result.evidence_status,
        retrieved_evidence=[_retrieved_document_to_model(d, sources) for d in result.retrieved_evidence],
        used_evidence=_retrieved_document_to_model(result.used_evidence, sources) if result.used_evidence else None,
        classification_result=_classification_result_to_model(result.classification_result),
        explanation=result.explanation,
        data_status=DATA_STATUS,
        normalization=normalization,
        route=route.to_dict() if route is not None else None,
        selected_passage=_selected_passage_to_model(result.selected_passage),
        claim_rewrite=claim_rewrite,
        date_check=result.date_check,
        value_check=getattr(result, "value_check", None),
        sliding_windows=getattr(result, "sliding_windows", None),
        relevance_check=relevance_check,
        date_notice=date_notice(message, [*result.retrieved_evidence, result.used_evidence]),
        relative_date_notice=relative_date.notice(message, [*result.retrieved_evidence, result.used_evidence], sources),
        relative_date_check=relative_date_check,
        other_school_check=other_school_check,
        wrong_post_check=wrong_post_check,
        tense_guard_check=tense_guard_check,
        relative_resolution=relative_resolution,
        tagalog_reading=tagalog_reading,
        tagalog_topic_check=tagalog_topic_check,
        tagalog_translation=tagalog_translation,
        holiday_notice=holiday_notice,
        holiday_check=holiday_check,
        class_day_check=class_day_check,
        question_focus=question_focus,
        same_day_advisory=same_day_advisory,
        same_day_shown=same_day_shown,
        same_day_lookup=same_day_lookup,
        header_date_check=header_date_check,
        same_day_judge=same_day_judge,
        post_age_notice=post_age_notice,
        hearsay_notice=_hearsay_notice(claim if (tagalog_reading or tagalog_translation) else message),   # "daw" is typed
    )


def _hearsay_notice(message: str):
    """CAMPUS_VERIFY_HEARSAY_NOTE (default on since v1.9; scenario 7): warn-only; the field is left out of the response when None."""
    from src.verification import hearsay
    return hearsay.notice(message) if hearsay.enabled() else None


# Verdict policy (owner decision 2026-10-10): with CAMPUS_VERIFY_VERDICT_LANGUAGES=english, a claim the router flags as
# Tagalog/Taglish gets the closest announcements and no truth verdict. In testing, Taglish verdicts were wrong too often
# (7 of 11 shown verdicts right on the round-2 development messages) to show safely.
LANGUAGE_GUIDANCE = ("Verdicts are given for English claims only for now: checks of Tagalog and Taglish messages were not "
                     "reliable enough in testing. The closest campus announcements are shown so you can check them yourself.")


def _information_result_to_model(info: InformationResult, route, normalization=None, sources=None,
                                 claim: Optional[str] = None, verdict_withheld: Optional[str] = None,
                                 notice_message: Optional[str] = None, relative_resolution=None,
                                 tagalog_reading=None, tagalog_translation=None,
                                 holiday_notice=None) -> VerificationResultModel:
    """INFO_SEEKING response (or a claim whose verdict is withheld by the language policy): same shape, verdict and
    classification_result null, explanation led by the guidance."""
    guidance = LANGUAGE_GUIDANCE if verdict_withheld == "language" else route.guidance
    return VerificationResultModel(
        claim=claim if claim is not None else info.query,
        verdict_withheld=verdict_withheld,
        verdict=None,
        evidence_status=info.evidence_status,
        retrieved_evidence=[_retrieved_document_to_model(d, sources) for d in info.retrieved_evidence],
        used_evidence=_retrieved_document_to_model(info.used_evidence, sources) if info.used_evidence else None,
        classification_result=None,
        explanation=(guidance + " " + info.explanation).strip(),
        data_status=DATA_STATUS,
        normalization=normalization,
        route=route.to_dict(),
        selected_passage=_selected_passage_to_model(info.selected_passage),
        # The raw message: for a withheld claim `claim` is the text as typed; for a question info.query is.
        date_notice=date_notice(notice_message or (claim if claim is not None else info.query),
                                [*info.retrieved_evidence, info.used_evidence]),
        relative_date_notice=relative_date.notice(notice_message or (claim if claim is not None else info.query),
                                                  [*info.retrieved_evidence, info.used_evidence], sources),
        hearsay_notice=_hearsay_notice(claim if (tagalog_reading or tagalog_translation) else
                                       (notice_message or (claim if claim is not None else info.query))),
        relative_resolution=relative_resolution,
        tagalog_reading=tagalog_reading,
        tagalog_translation=tagalog_translation,
        holiday_notice=holiday_notice,
    )


# ---- request handling ------------------------------------------------------------------------------------------------
def _normalization_info(claim: str, documents):
    if not NORMALIZE_REPORT_ENABLED:
        return None
    from src.normalization import TaglishNormalizer
    from src.normalization.report import normalization_summary
    from src.retrieval.bm25 import word_tokens
    if state.get("normalizer") is None:   # the corpus vocabulary is the English evidence for the code_switched feature
        vocab = {w for d in documents for w in word_tokens(d["text"])}
        state["normalizer"] = TaglishNormalizer(extra_vocab=vocab, english_vocab=vocab)
    return normalization_summary(state["normalizer"].normalize(claim))


def _source_index(pipeline):
    """Dates and original links for the live corpus, rebuilt when the corpus reloads. Display only: a failure here never
    blocks a check (the fields are simply left empty)."""
    # Keyed on the pipeline OBJECT (every reload builds a new one), not on loaded_at, which only has one-second
    # resolution: two reloads within the same second would otherwise keep the stale index.
    if state.get("source_info_for") is not pipeline:
        try:
            state["source_info"] = build_source_info(state["settings"].corpus, PROJECT_ROOT, DEV_DATASET_PATH)
        except Exception:                                  # e.g. a file being rewritten right now; retried next time
            state.setdefault("source_info", {})
            return state["source_info"]
        state["source_info_for"] = pipeline
    return state["source_info"]


def _answer(verifier: ClaimVerifier, text: str, normalization=None, sources=None,
            evidence_verifier: Optional[ClaimVerifier] = None, relevance_checker=None,
            text_route: bool = False) -> VerificationResultModel:
    """Route the raw input, then verify it (CHECKABLE_CLAIM / router off) or retrieve information (INFO_SEEKING).

    evidence_verifier (CAMPUS_VERIFY_TAGLISH_HYBRID): used ONLY for evidence-only responses to Tagalog/Taglish input
    (withhold_for_language), i.e. INFO_SEEKING questions and claims withheld by the language policy, and only through
    retrieve_information(). Every verdict and every English request goes through `verifier`, unchanged.

    relevance_checker (CAMPUS_VERIFY_RELEVANCE_CHECK, /verify/text only): after verify_claim() on an English claim, may
    change SUPPORTED/CONTRADICTED to INSUFFICIENT_EVIDENCE when the post used looks off-topic; nothing else.

    CAMPUS_VERIFY_RELATIVE_DATE_CHECK: after verify_claim(), a SUPPORTED/CONTRADICTED verdict on a message with a relative
    day word ("tomorrow", "bukas" ...) becomes INSUFFICIENT_EVIDENCE unless the post used was made today or yesterday.

    CAMPUS_VERIFY_OTHER_SCHOOL_CHECK (/verify/text only, text_route=True): a SUPPORTED/CONTRADICTED verdict on a message
    naming another school becomes INSUFFICIENT_EVIDENCE when the post used does not mention that school.

    CAMPUS_VERIFY_UNICODE_FOLD: every step below reads `message`, the folded processing copy of `text`; the response's
    `claim` stays `text` as typed. With the flag off, `message` is `text`."""
    settings = state["settings"]
    message = fold(text) if settings.unicode_fold else text
    from src.normalization import input_cleanup
    # CAMPUS_VERIFY_ACADEMIC_YEAR / CAMPUS_VERIFY_LOOKALIKE (default off; results/round2/input_hardening_plan.md):
    # "AY 2026-2027" is not read as Tagalog "ay"; Cyrillic/Greek lookalike letters are read as Latin.
    message, cleanup = input_cleanup.clean(message)
    from src.normalization import tagalog_phrases
    phrase = None
    if tagalog_phrases.enabled() and withhold_for_language(message):
        # CAMPUS_VERIFY_TAGALOG_PHRASES (default off; results/round2/tagalog_phrases_plan.md): a Tagalog/Taglish message
        # that fits a fixed frame with every word understood is checked as its English reading; otherwise unchanged.
        phrase = tagalog_phrases.read(message, tagalog_phrases.vocabulary_for(verifier))
        if phrase is not None and not withhold_for_language(phrase["reading"]):
            message = phrase["reading"]
        else:
            phrase = None
    from src.normalization import tagalog_translate
    translation = None
    if phrase is None and tagalog_translate.enabled() and withhold_for_language(message):
        # CAMPUS_VERIFY_TAGALOG_TRANSLATE (default on since v2.3; results/round2/tagalog_translate_v2_report.md): the local SEA-LION
        # translation replaces the message only when its checks C1-C3 pass; otherwise the response is today's.
        translation = tagalog_translate.translate(message, withhold_for_language)
        if translation is not None and translation["accepted"]:
            message = translation["translation"]
    from src.verification import relative_resolve
    resolution = None
    if relative_resolve.enabled() and not withhold_for_language(message):
        # CAMPUS_VERIFY_RESOLVE_RELATIVE (default on since v2.0; results/round2/relative_resolve_v2_report.md): "tomorrow" etc. become
        # the calendar date in Philippine time, so every step below (and the frozen date check) reads a real date.
        message, resolution = relative_resolve.resolve(message)
    read_as = phrase or (translation if translation is not None and translation["accepted"] else None)
    from src.normalization import class_wording
    wording = None
    if class_wording.enabled():
        # CAMPUS_VERIFY_CLASS_WORDING (default off; results/round2/class_wording_plan.md): "no class" -> "no classes" and a
        # leading date moved to the end, so the classifier reads the clipped forms students type the standard way.
        wording = class_wording.rewrite(message)
        if wording is not None:
            message = wording["rewritten"]
    from src.verification import holidays
    holiday_note = holidays.note(message, resolution, relative_resolve.today()) if holidays.enabled() else None
    notice = message if (settings.unicode_fold or resolution or read_as or wording or cleanup) else None
    shown_claim = text if (settings.unicode_fold or resolution or read_as or wording or cleanup) else None
    route = _router.route(message) if settings.router else None
    taglish = withhold_for_language(message)
    evidence = evidence_verifier if (evidence_verifier is not None and taglish) else verifier
    if route is not None and route.status == INFO_SEEKING:
        return _information_result_to_model(evidence.retrieve_information(message), route, normalization, sources,
                                            claim=shown_claim, notice_message=notice, relative_resolution=resolution,
                                            tagalog_reading=phrase, tagalog_translation=translation,
                                            holiday_notice=holiday_note)
    from src.normalization import question_focus
    checked, focus = message, None
    if question_focus.enabled() and not taglish:
        # CAMPUS_VERIFY_QUESTION_FOCUS (default on since v2.3; results/round2/question_focus_v2_report.md): 2+ sentences with exactly one
        # question -> that question is what gets rewritten and judged; every safeguard still reads the whole message.
        focus = question_focus.pick(message)
        if focus is not None:
            checked = focus["checked"]
    pipeline_text, rewrite = checked, None
    if settings.claim_rewrite == "optionB":
        from src.normalization.question_form import rewrite_question
        r = rewrite_question(checked)
        pipeline_text, rewrite = r.rewritten, r.to_dict()
    if settings.verdict_languages == "english" and taglish:
        # Same retrieval text as a claim gets, but no truth verdict: the closest announcements are shown.
        return _information_result_to_model(evidence.retrieve_information(pipeline_text), route, normalization, sources,
                                            claim=text, verdict_withheld="language", notice_message=notice,
                                            relative_resolution=resolution, tagalog_translation=translation,
                                            holiday_notice=holiday_note)
    if settings.en_question_rewrite and not (rewrite and rewrite.get("changed")) and not taglish:
        # English yes/no question that Option B left unchanged: reorder it into a statement (words unchanged).
        statement, refused = english_question.rewrite_english_question(checked)
        if refused is None:
            pipeline_text = statement
            rewrite = {"original": checked, "rewritten": statement, "changed": True, "rules_applied": ["EN_Q_aux_inversion"],
                       "refused": None, "version": english_question.VERSION, "rules_sha256": english_question.RULES_SHA256}
        elif refused == "no_subject_boundary":
            # CAMPUS_VERIFY_EN_BEEN_RULE (R1, default on since v1.8; results/round2/english_fixes_report.md): "Has/Have <long
            # subject> been ...?" -> statement, words only reordered.
            from src.normalization import english_been
            been = english_been.rewrite(checked) if english_been.enabled() else None
            if been is not None:
                pipeline_text = been
                rewrite = {"original": checked, "rewritten": been, "changed": True, "rules_applied": ["EN_been"],
                           "refused": None, "version": english_been.VERSION, "rules_sha256": english_been.RULES_SHA256}
    result = verifier.verify_claim(pipeline_text)
    relevance = None
    if relevance_checker is not None and not taglish:
        result, relevance = relevance_checker.apply(result, pipeline_text, verifier._relevance_threshold)
    from src.verification import name_align
    if name_align.enabled() and not taglish:
        # CAMPUS_VERIFY_NAME_ALIGN (default on since v1.4; results/round2/name_align_plan.md): check again with the post's own name.
        result, _ = name_align.apply(pipeline_text, result, verifier.verify_claim)
    from src.verification import header_date
    header_check = None
    if header_date.enabled() and not taglish:
        # CAMPUS_VERIFY_HEADER_DATE (default off; results/round2/header_date_plan.md): a SUPPORTED withheld only because an
        # advisory gives its date in the header ("DYCI ADVISORY November 12, 2026 ... suspended immediately") is
        # restored when that header date is the message's date; every later safeguard still runs.
        result, header_check = header_date.check(message, result, relative_resolve.today(), sources)
    relative = None
    if settings.relative_date_check:
        result, relative = relative_date.check(message, result, sources)
    other = None
    if settings.other_school_check and text_route:
        result, other = other_school.check(message, result)
    from src.verification import tagalog_topic
    topic = None
    if tagalog_topic.enabled() and taglish:
        # CAMPUS_VERIFY_TAGALOG_TOPIC_GUARD (default off; results/round2/tagalog_verdicts_plan.md): a Tagalog/Taglish
        # SUPPORTED/CONTRADICTED needs the message to share a topic word with the post used.
        result, topic = tagalog_topic.check(message, result)
    from src.verification import wrong_post
    wrong = None
    if wrong_post.enabled() and not taglish:
        # CAMPUS_VERIFY_WRONG_POST (default on since v1.7; results/round2/wrong_post_report.md): a CONTRADICTED verdict
        # resting on a post that is probably about something else becomes INSUFFICIENT_EVIDENCE.
        result, wrong = wrong_post.check(message, result, result.retrieved_evidence)
    if resolution is not None:
        # CAMPUS_VERIFY_RESOLVE_RELATIVE v2 (results/round2/relative_resolve_v2_plan.md): a CONTRADICTED for a resolved
        # message needs a post date within 7 days of the resolved date.
        result, near = relative_resolve.guard(result, resolution)
        if near is not None:
            resolution = dict(resolution, guard=near)
    from src.verification import tense_guard
    tense = None
    if tense_guard.enabled() and not taglish:
        # CAMPUS_VERIFY_TENSE_GUARD (R2, results/round2/english_fixes_plan.md), default off: a SUPPORTED that rests on
        # a passage saying the event WILL happen, for a statement saying it HAS happened, becomes INSUFFICIENT_EVIDENCE.
        result, tense = tense_guard.check(pipeline_text, result)
    holiday = None
    if holidays.enabled() and not taglish:
        # CAMPUS_VERIFY_HOLIDAYS H1 (default on since v2.1; results/round2/holidays_report.md): only an INSUFFICIENT_EVIDENCE left by
        # every other step may be settled by the national holiday list, always with its warning.
        result, far = holidays.far_post_guard(message, result, resolution, relative_resolve.today())   # amendment H-A1
        result, holiday = holidays.check(pipeline_text, message, result, resolution, relative_resolve.today(),
                                         verifier._classifier.classify)
        if far is not None:
            holiday = dict(holiday or {}, far_post_guard=far)
    from src.verification import class_day
    class_check = None
    if class_day.enabled() and not taglish:
        # CAMPUS_VERIFY_CLASS_DAY_GUARD (default on since v2.2; results/round2/class_day_guard_v2_report.md): a class-day message with an
        # explicit date cannot be settled by a DYCI post about a different day.
        result, class_check = class_day.check(message, result, relative_resolve.today())
    from src.verification import same_day
    same_day_check = None
    if same_day.enabled() and not taglish:
        # CAMPUS_VERIFY_SAME_DAY_ADVISORY (default off; results/round2/cross_post_conflict_plan.md, Amendment 2): an
        # advisory dated the message's day, ranked below the posts shown, is judged too and decides as the plan's table.
        result, same_day_check = same_day.check(message, pipeline_text, result, verifier, relative_resolve.today())
    same_day_judge = None
    if same_day.judge_enabled() and same_day_check is None and not taglish:
        # CAMPUS_VERIFY_SAME_DAY_JUDGE (Z v2, default off; results/round2/same_day_judge_plan.md): advisories dated the
        # message's day, found by header date, are judged too and decide as the plan's table; guard G for moved events.
        result, same_day_judge = same_day.judge_same_day(message, pipeline_text, result, verifier,
                                                         relative_resolve.today(), sources)
    same_day_shown = None
    if (same_day.show_enabled() and same_day_check is None and same_day_judge is None and not same_day.enabled()
            and not taglish):
        # CAMPUS_VERIFY_SAME_DAY_SHOW (default off; results/round2/same_day_show_plan.md): display only. An advisory dated
        # the message's day, ranked below the posts shown, is shown first; the verdict is never changed.
        result, same_day_shown = same_day.show(message, pipeline_text, result, verifier, relative_resolve.today())
    same_day_lookup = None
    if (same_day.lookup_enabled() and same_day_check is None and same_day_judge is None and same_day_shown is None
            and not taglish):
        # CAMPUS_VERIFY_SAME_DAY_LOOKUP (default off; results/round2/same_day_lookup_plan.md): display only. Advisories
        # dated the message's day are found by header date and shown first; the verdict is never changed.
        result, same_day_lookup = same_day.lookup(message, pipeline_text, result, verifier, relative_resolve.today(),
                                                  sources)
    from src.verification import post_age
    age_notice = None
    if post_age.enabled() and not taglish:
        # CAMPUS_VERIFY_POST_AGE (default on since v1.4; results/round2/post_age_plan.md; owner 2026-10-25: warn, never
        # withhold): the final verdict is kept, with a warning when its post is old for a message about now.
        age_notice = post_age.notice(message, result, sources, v2=post_age.v2_enabled())   # v2: post_age_v2_plan.md
    model = _verification_result_to_model(result, claim=text, route=route, claim_rewrite=rewrite,
                                          normalization=normalization, sources=sources, relevance_check=relevance,
                                          relative_date_check=relative, other_school_check=other, notice_message=notice,
                                          post_age_notice=age_notice, wrong_post_check=wrong, tense_guard_check=tense,
                                          relative_resolution=resolution, tagalog_reading=phrase,
                                          tagalog_topic_check=topic, tagalog_translation=translation,
                                          holiday_notice=holiday_note, holiday_check=holiday,
                                          class_day_check=class_check, question_focus=focus,
                                          same_day_advisory=same_day_check, same_day_shown=same_day_shown,
                                          same_day_lookup=same_day_lookup, header_date_check=header_check,
                                          same_day_judge=same_day_judge)
    if wording is not None:                     # rule C: what was checked, shown under the message
        model = model.model_copy(update={"class_wording": wording})
    if cleanup is not None:                     # input hardening: what was rewritten before the check
        model = model.model_copy(update={"input_cleanup": cleanup})
    from src.verification import unit_check
    if unit_check.enabled() and model.verdict == "SUPPORTED" and model.used_evidence is not None:
        # CAMPUS_VERIFY_UNIT_CHECK (default off; results/round2/unit_check_plan.md): a message naming another office,
        # level, grade or venue than the post used is not confirmed; never creates CONTRADICTED or SUPPORTED.
        used, passage = model.used_evidence, model.selected_passage
        premise = passage.text if passage is not None and passage.document_id == used.document_id else used.text
        found = unit_check.conflict(message, premise, used.text)
        if found is not None:
            model = model.model_copy(update={"verdict": "INSUFFICIENT_EVIDENCE", "unit_check": found,
                                             "explanation": (model.explanation or "") + unit_check.note(found)})
    from src.verification import value_withhold
    if value_withhold.enabled() and model.verdict == "SUPPORTED" and model.used_evidence is not None:
        # CAMPUS_VERIFY_VALUE_WITHHOLD (default off; results/round2/value_withhold_plan.md): a message whose time, duration,
        # percentage or place differs from the post used is not confirmed; never creates CONTRADICTED or SUPPORTED.
        used, passage = model.used_evidence, model.selected_passage
        premise = passage.text if passage is not None and passage.document_id == used.document_id else used.text
        found = value_withhold.conflict(message, premise, used.text)
        if found is not None:
            model = model.model_copy(update={"verdict": "INSUFFICIENT_EVIDENCE", "value_withhold": found,
                                             "explanation": (model.explanation or "") + value_withhold.note(found)})
    from src.verification import reason_parts
    if reason_parts.enabled() and not taglish and not _REASON_PARTS_ACTIVE.get():
        # CAMPUS_VERIFY_REASON_PARTS (default off; results/round2/reason_parts_plan.md): "<main claim> due to <reason>" is
        # checked part by part; SUPPORTED only when both parts are backed; never creates CONTRADICTED.
        model = _apply_reason_parts(model, text, message, verifier, normalization, sources, evidence_verifier,
                                    relevance_checker, text_route)
    from src.verification import main_point
    if main_point.enabled() and model.verdict == "SUPPORTED" and not _MAIN_POINT_ACTIVE.get():
        # CAMPUS_VERIFY_MAIN_POINT (default off; results/round2/main_point_v2_plan.md): a SUPPORTED message with a
        # reason marker has its main point (the part before it) checked alone; not SUPPORTED alone -> Not confirmed.
        model = _apply_main_point(model, message, verifier, normalization, sources, evidence_verifier,
                                  relevance_checker, text_route)
    from src.verification import contra_relevance
    if contra_relevance.enabled() and model.verdict == "CONTRADICTED" and model.used_evidence is not None:
        # CAMPUS_VERIFY_CONTRA_RELEVANCE (default off; results/round2/contra_relevance_plan.md): a CONTRADICTED answer
        # whose post shares less than a third of the message's content words becomes Not confirmed; the post stays shown.
        found = contra_relevance.check(message, model.verdict, model.used_evidence.text)
        if found is not None:
            found["post"] = model.used_evidence.document_id
            model = model.model_copy(update={"verdict": "INSUFFICIENT_EVIDENCE", "contra_relevance": found,
                                             "explanation": (model.explanation or "") + contra_relevance.NOTE})
    return model


_REASON_PARTS_ACTIVE = contextvars.ContextVar("reason_parts_active", default=False)
_MAIN_POINT_ACTIVE = contextvars.ContextVar("main_point_active", default=False)


def _apply_main_point(model, message, verifier, normalization, sources, evidence_verifier, relevance_checker,
                      text_route):
    """Rule M v2: check the main point (before the first reason marker) through the full pipeline, M off inside;
    never CONTRADICTED."""
    from src.verification import main_point
    parts = main_point.split(message)
    if parts is None or not has_enough_letters(parts[0]):
        return model
    main_text, phrase = parts
    token = _MAIN_POINT_ACTIVE.set(True)
    try:
        inner = _answer(verifier, main_text, normalization=normalization, sources=sources,
                        evidence_verifier=evidence_verifier, relevance_checker=relevance_checker, text_route=text_route)
    finally:
        _MAIN_POINT_ACTIVE.reset(token)
    outer_post = model.used_evidence.document_id if model.used_evidence is not None else None
    inner_post = inner.used_evidence.document_id if inner.used_evidence is not None else None
    outcome = main_point.decide(inner.verdict)
    info = {"version": main_point.VERSION, "main": main_text, "reason_phrase": phrase, "inner_verdict": inner.verdict,
            "inner_post": inner_post, "outer_post": outer_post, "outcome": outcome}
    if outcome == "main_not_supported":
        return model.model_copy(update={"verdict": "INSUFFICIENT_EVIDENCE", "main_point": info,
                                        "explanation": (model.explanation or "") + main_point.NOTE[outcome]})
    return model.model_copy(update={"main_point": info})


def _apply_reason_parts(model, text, message, verifier, normalization, sources, evidence_verifier, relevance_checker,
                        text_route):
    """Rule P: check the main claim through the full pipeline (P off inside), then the reason against the main claim's
    post. Returns the response to send."""
    from src.verification import reason_parts
    parts = reason_parts.split(message)
    if parts is None:
        return model
    main_text, reason = parts
    token = _REASON_PARTS_ACTIVE.set(True)
    try:
        main = _answer(verifier, main_text, normalization=normalization, sources=sources,
                       evidence_verifier=evidence_verifier, relevance_checker=relevance_checker, text_route=text_route)
    finally:
        _REASON_PARTS_ACTIVE.reset(token)
    label = piece = None
    if main.verdict == "SUPPORTED" and main.used_evidence is not None:
        label, piece = reason_parts.check_reason(verifier._classifier.classify, reason, main.used_evidence.text)
    verdict, action = reason_parts.decide(model.verdict, main.verdict, label)
    info = {"version": reason_parts.VERSION, "main_claim": main_text, "reason": reason,
            "main_verdict": main.verdict,
            "main_post": main.used_evidence.document_id if main.used_evidence is not None else None,
            "reason_label": label, "reason_piece": piece, "original_verdict": model.verdict, "verdict": verdict,
            "action": action}
    if action in ("both_backed", "reason_not_stated", "reason_different"):
        note = {"both_backed": " Parts check: the main claim and its reason are both backed by the DYCI post.",
                "reason_not_stated": " Parts check: the main claim is backed, but the DYCI post does not state this reason.",
                "reason_different": " Parts check: the main claim is backed, but the DYCI post gives a different reason."}[action]
        return main.model_copy(update={"claim": model.claim, "verdict": verdict,
                                       "explanation": (main.explanation or "") + note, "reason_parts": info})
    return model.model_copy(update={"reason_parts": info})


def _restore_original_text(result: VerificationResultModel, originals: dict, folded: dict) -> None:
    """CAMPUS_VERIFY_UNICODE_FOLD: display the stored original text of every shown post and passage. Folding keeps
    lengths, so a passage found at position i of the folded document is the original document's text at i."""
    def original_span(doc_id, span):
        f, o = folded.get(doc_id), originals.get(doc_id)
        if not span or f is None or o is None or f == o:
            return span
        i = f.find(span)
        return o[i:i + len(span)] if i >= 0 else span
    for d in [*result.retrieved_evidence, *([result.used_evidence] if result.used_evidence else [])]:
        if d.document_id in originals:
            d.text = originals[d.document_id]
    sp = result.selected_passage
    if sp is not None:
        sp.text = original_span(sp.document_id, sp.text)
        sp.context_before_text = original_span(sp.document_id, sp.context_before_text)


# ---- routes ----------------------------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Full configuration in development; with CAMPUS_VERIFY_PUBLIC=1 only what the page reads (status, image check)."""
    full = _full_health()
    if _public():
        page = {"status": full.status, "image_check": full.image_check, "image_signals": full.image_signals}
        if full.stats:
            page["stats"] = True
        if full.ui_extras:
            page["ui_extras"] = True                          # CAMPUS_VERIFY_UI_EXTRAS
        if full.auth:
            page["auth"] = full.auth                          # CAMPUS_VERIFY_AUTH: the page must show sign-in
        if full.tour:
            page["tour"] = True                               # CAMPUS_VERIFY_TOUR
        if full.history:
            page["history"] = True                            # CAMPUS_VERIFY_HISTORY
        if full.sources:
            page["sources"] = True                            # CAMPUS_VERIFY_SOURCES (the page shows Sources to admins)
        return JSONResponse(page)
    return full


# ---- Sign in with Google, DYCI accounts only (src/api/auth.py; CAMPUS_VERIFY_AUTH=google, default off) ---------------
AUTH_STATUS = {"not_configured": 503, "keys_unavailable": 503, "wrong_domain": 403, "disabled": 403, "self": 400,
               "not_found": 404}


def _auth_ready():
    from src.api import auth
    if not auth.enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    if _database() is None:
        raise HTTPException(status_code=503, detail="Sign-in needs the app database (CAMPUS_VERIFY_DB_FILE).")
    return auth


def _signed_in(request: Request, admin: bool = False):
    user = _current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in with your DYCI Google account.")
    if admin and not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Only administrators can open this page.")
    return user


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _https(request: Request) -> bool:
    """Whether the sign-in cookie is marked Secure. Everywhere except a page opened on the PC itself: the public address
    (a tunnel such as Tailscale Funnel or Cloudflare) is always HTTPS, and proxies do not all say so in a header."""
    return request.url.scheme == "https" or (request.url.hostname or "").lower() not in LOCAL_HOSTS


@app.get("/auth/config")
def auth_config():
    """What the page needs to show Google's button. The client ID is public by design."""
    auth = _auth_ready()
    return {"client_id": auth.client_id(), "domain": auth.domain(), "required": auth.required()}


@app.post("/auth/google")
def auth_google(request: Request, body: GoogleSignInRequest):
    auth = _auth_ready()
    try:
        claims = auth.verify_google_token(body.credential)
        user, token = auth.sign_in(_database(), claims)
    except auth.AuthError as e:
        raise HTTPException(status_code=AUTH_STATUS.get(e.reason, 401), detail=str(e))
    response = JSONResponse({"user": auth.public_user(user)})
    response.set_cookie(auth.COOKIE, token, max_age=auth.session_days() * 86400, httponly=True, samesite="lax",
                        secure=_https(request), path="/")
    return response


@app.post("/auth/logout")
def auth_logout(request: Request):
    auth = _auth_ready()
    auth.sign_out(_database(), request.cookies.get(auth.COOKIE))
    response = JSONResponse({"signed_out": True})
    response.delete_cookie(auth.COOKIE, path="/")
    return response


@app.get("/auth/me")
def auth_me(request: Request):
    auth = _auth_ready()
    return {"user": auth.public_user(_signed_in(request))}


@app.post("/account/history")
def account_history(request: Request, body: HistoryPreferenceRequest):
    """The student's own choice to keep a history of their checks (CAMPUS_VERIFY_HISTORY saves them). Turning it off
    with delete_saved also erases what was saved."""
    from src.api import history
    auth = _auth_ready()
    user = _signed_in(request)
    if not body.history_enabled and body.delete_saved:
        history.clear(_database(), user["id"])
    return {"user": auth.public_user(auth.set_history(_database(), user["id"], body.history_enabled))}


def _history_ready(request: Request):
    """The signed-in account, when "My checks" is on (CAMPUS_VERIFY_HISTORY); 404 otherwise, like any switched-off part."""
    from src.api import history
    _auth_ready()
    if not history.enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    return history, _signed_in(request)


@app.get("/account/checks")
def account_checks(request: Request):
    """The signed-in student's own saved checks, newest first. There is no way to read another account's."""
    history, user = _history_ready(request)
    return {"checks": history.list_for(_database(), user["id"]), "history_enabled": user["history_enabled"],
            "keep_last": history.KEEP_LAST, "keep_days": history.KEEP_DAYS}


@app.post("/account/checks/{check_id}/delete")
def account_check_delete(request: Request, check_id: int):
    history, user = _history_ready(request)
    if not history.delete_one(_database(), user["id"], check_id):
        raise HTTPException(status_code=404, detail="No such saved check.")
    return {"deleted": True}


@app.post("/account/checks/clear")
def account_checks_clear(request: Request):
    history, user = _history_ready(request)
    return {"deleted": history.clear(_database(), user["id"])}


@app.post("/account/tour-seen")
def account_tour_seen(request: Request):
    """The guided tour (CAMPUS_VERIFY_TOUR) was finished or skipped: do not start it by itself again for this account."""
    auth = _auth_ready()
    user = _signed_in(request)
    return {"user": auth.public_user(auth.set_tour_seen(_database(), user["id"]))}


@app.post("/account/delete")
def account_delete(request: Request, body: DeleteAccountRequest):
    """Erase the signed-in account (privacy notice: the right to erasure). Verified twice: the student types DELETE,
    and confirms with Google again (the same DYCI account, within REAUTH_SECONDS). Signing in later makes a new account."""
    auth = _auth_ready()
    user = _signed_in(request)
    if body.confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE to confirm.")
    try:
        auth.check_reauth(user, auth.verify_google_token(body.credential))
    except auth.AuthError as e:
        raise HTTPException(status_code=AUTH_STATUS.get(e.reason, 401), detail=str(e))
    auth.delete_user(_database(), user["id"])
    response = JSONResponse({"deleted": True})
    response.delete_cookie(auth.COOKIE, path="/")
    return response


# ---- Sources: Facebook pages managed by administrators (src/api/sources.py; CAMPUS_VERIFY_SOURCES, default off) -------
def _sources_ready(request: Request):
    """(sources module, secret box, admin) when the Sources page is on; 404 when off, like any switched-off part."""
    from src.api import secret_box, sources
    _auth_ready()
    if not sources.enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    admin = _signed_in(request, admin=True)
    try:
        box = _secret_box()
    except secret_box.SecretBoxError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return sources, box, admin


def _secret_box():
    from src.api.secret_box import SecretBox
    if state.get("secret_box") is None:
        state["secret_box"] = SecretBox()
    return state["secret_box"]


def _sources_scheduler():
    from src.api import auth, sources
    if not (sources.enabled() and auth.enabled() and _database() is not None
            and (sources.fetch_minutes() > 0 or sources.backfill_minutes() > 0)):
        return None
    scheduler = sources.Scheduler(_database, _secret_box, sources.fetch_minutes(), sources.backfill_minutes())
    scheduler.start()
    return scheduler


def _website_scheduler():
    """Rule W (CAMPUS_VERIFY_WEBSITE; results/round2/website_source_plan.md): read DYCI's website feeds now and every
    CAMPUS_VERIFY_WEBSITE_MINUTES (default 60; 0 = never) into the website file, which start_public.ps1 serves."""
    from src.ingestion import dyci_website
    from src.ingestion.facebook_page import record_website_fetch
    if not dyci_website.enabled() or dyci_website.minutes() <= 0:
        return None
    s = dyci_website.Scheduler(dyci_website.file_path(PROJECT_ROOT), dyci_website.minutes(), record_website_fetch)
    s.start()
    return s


def _sources_call(fn, *args, **kwargs):
    from src.api import sources
    try:
        return fn(*args, **kwargs)
    except sources.SourcesError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/fetch-status")
def admin_fetch_status(request: Request):
    """Fix B (CAMPUS_VERIFY_FETCH_STATUS, default off): the live test-page fetch for admins: last fetch, recent fetches,
    what is served, warnings (expired token, no recent success, picture-only posts skipped). Display only."""
    from src.api import fetch_status
    from src.ingestion.facebook_page import fetch_log
    if not fetch_status.enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    _auth_ready()
    _signed_in(request, admin=True)
    manager: Optional[PipelineManager] = state.get("manager")
    if manager is None:
        raise HTTPException(status_code=503, detail="The checker is still starting.")
    path = manager.settings.fb_corpus
    path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    return fetch_status.build(manager.refresh, fetch_log(), fetch_status.served(path))


@app.get("/admin/sources")
def admin_sources(request: Request):
    """Every source page with its token's status (never the token), the D5 cutoff, the schedule and recent activity."""
    sources, box, _admin = _sources_ready(request)
    rows = sources.list_sources(_database())
    return {"sources": rows, "needs_attention": [s["key"] for s in sources.needs_attention(rows)],
            "cutoff": sources.cutoff_day(), "frozen": sources.cutoff_info(), "fetch_minutes": sources.fetch_minutes(),
            "backfill_minutes": sources.backfill_minutes(), "backfill_from": sources.backfill_from().date().isoformat(),
            "encryption": box.method, "tiers": list(sources.TIERS), "activity": sources.activity(_database())}


@app.post("/admin/sources")
def admin_sources_add(request: Request, body: SourceAddRequest):
    sources, box, admin = _sources_ready(request)
    return {"source": _sources_call(sources.add, _database(), box, admin, body.page_id, body.tier, body.token)}


@app.post("/admin/sources/cutoff")
def admin_sources_cutoff(request: Request, body: SourceCutoffRequest):
    """Record the final-evaluation cutoff once and freeze the checker as a Git tag (src/api/sources.py record_cutoff)."""
    sources, _box, admin = _sources_ready(request)
    return {"frozen": _sources_call(sources.record_cutoff, _database(), admin, body.day)}


@app.post("/admin/sources/{key}")
def admin_sources_update(request: Request, key: str, body: SourceUpdateRequest):
    sources, _box, admin = _sources_ready(request)
    return {"source": _sources_call(sources.update, _database(), admin, key, include=body.include, tier=body.tier)}


@app.post("/admin/sources/{key}/token")
def admin_sources_token(request: Request, key: str, body: SourceTokenRequest):
    sources, box, admin = _sources_ready(request)
    return {"source": _sources_call(sources.replace_token, _database(), box, admin, key, body.token)}


@app.post("/admin/sources/{key}/check")
def admin_sources_check(request: Request, key: str):
    sources, box, admin = _sources_ready(request)
    return {"source": _sources_call(sources.check, _database(), box, admin, key)}


@app.post("/admin/sources/{key}/fetch")
def admin_sources_fetch(request: Request, key: str):
    sources, box, admin = _sources_ready(request)
    source = _sources_call(sources.get, _database(), key)
    if not source["include"]:
        raise HTTPException(status_code=400, detail="This page is turned off. Turn it on before fetching.")
    return {"result": _sources_call(sources.fetch, _database(), box, admin, only=[key])}


@app.post("/admin/sources/{key}/remove")
def admin_sources_remove(request: Request, key: str):
    sources, _box, admin = _sources_ready(request)
    _sources_call(sources.remove, _database(), admin, key)
    return {"removed": True}


@app.get("/admin/users")
def admin_users(request: Request):
    auth = _auth_ready()
    _signed_in(request, admin=True)
    return {"users": [{k: u[k] for k in ("id", "email", "name", "created_at", "last_login_at", "is_admin", "active",
                                          "history_enabled")} for u in auth.list_users(_database())]}


@app.post("/admin/users/{user_id}/active")
def admin_set_active(request: Request, user_id: int, body: AccountActiveRequest):
    auth = _auth_ready()
    admin = _signed_in(request, admin=True)
    try:
        user = auth.set_active(_database(), admin, user_id, body.active)
    except auth.AuthError as e:
        raise HTTPException(status_code=AUTH_STATUS.get(e.reason, 400), detail=str(e))
    return {"user": {k: user[k] for k in ("id", "email", "name", "active")}}


def _stats_ready(request: Request, days: int) -> int:
    """The dashboard's guard: 404 while CAMPUS_VERIFY_STATS is off; administrators only while sign-in is on. Returns the
    period in days, kept within 1..KEEP_DAYS."""
    from src.api import auth, stats
    if not stats.enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    if auth.enabled():
        _auth_ready()
        _signed_in(request, admin=True)
    return max(1, min(int(days), stats.KEEP_DAYS))


@app.get("/stats")
def usage_stats(request: Request, days: int = 7):
    """Dashboard counts (CAMPUS_VERIFY_STATS, default off): kinds and outcomes per day, never message text. With sign-in
    on (CAMPUS_VERIFY_AUTH=google), administrators only (panel request 2026-10-28)."""
    return _stats_counter().snapshot(days=_stats_ready(request, days))


@app.get("/stats.csv")
def usage_stats_csv(request: Request, days: int = 7):
    """The dashboard's counts as a spreadsheet (panel request 2026-10-29): one row per day and kind, the same counts as
    /stats, same access rule. Opens in Excel or Google Sheets."""
    from src.api import stats
    days = _stats_ready(request, days)
    name = f"dyci-claim-check-report-{stats.today().isoformat()}-last-{days}-days.csv"
    return Response(content=_stats_counter().csv(days=days), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


def _sources_health() -> bool:
    from src.api import auth, sources
    return sources.enabled() and auth.enabled()


def _history_health() -> bool:
    from src.api import auth, history
    return history.enabled() and auth.enabled()


def _auth_health():
    from src.api import auth
    return {"provider": "google", "domain": auth.domain(), "required": auth.required()} if auth.enabled() else None


def _full_health() -> HealthResponse:
    manager: Optional[PipelineManager] = state.get("manager")
    if manager is None:
        return HealthResponse(status="starting", data_status=DATA_STATUS, classifier="not_loaded",
                              corpus_source="not_loaded")
    settings, pipeline = manager.settings, manager.current()
    r = manager.refresh
    return HealthResponse(
        status="ok",
        data_status=DATA_STATUS,
        classifier=type(manager.classifier).__name__,
        corpus_source=pipeline.corpus_label,
        corpus_size=len(pipeline.documents),
        corpus_sources=[CorpusSourceModel(label=s.label, documents=s.documents) for s in pipeline.sources],
        corpus_loaded_at=pipeline.loaded_at,
        corpus_reload_error=manager.reload_error,
        retriever=settings.retriever,
        router=ROUTER_VERSION if settings.router else "off",
        claim_rewrite=settings.claim_rewrite,
        date_check=settings.date_check,
        value_check=settings.value_check,
        reranker=manager.reranker.label if manager.reranker is not None else "off",
        premise_mode=settings.premise_mode,
        verdict_languages=settings.verdict_languages,
        en_question_rewrite=settings.en_question_rewrite,
        taglish_hybrid=settings.taglish_hybrid,
        relevance_check=settings.relevance_check,
        relative_date_check=settings.relative_date_check,
        other_school_check=settings.other_school_check,
        unicode_fold=settings.unicode_fold,
        image_check=settings.image_check,
        image_signals=settings.image_check and _photo_index_path() is not None and not state.get("photo_index_error"),
        stats=True if _stats_enabled() else None,
        # CAMPUS_VERIFY_UI_EXTRAS (default off; display only): example messages, copy/share and next steps on the page.
        ui_extras=True if os.environ.get("CAMPUS_VERIFY_UI_EXTRAS", "0").strip() == "1" else None,
        # CAMPUS_VERIFY_AUTH=google: the page shows "Sign in with Google" (and requires it when AUTH_REQUIRED=1).
        auth=_auth_health(),
        # CAMPUS_VERIFY_TOUR (default off; display only): the guided first-visit tour and the "Take the tour" menu item.
        tour=True if os.environ.get("CAMPUS_VERIFY_TOUR", "0").strip() == "1" else None,
        # CAMPUS_VERIFY_HISTORY (default off): "My checks" on the Account page, for accounts that turn history on.
        history=True if _history_health() else None,
        # CAMPUS_VERIFY_SOURCES (default off): the Sources page for administrators.
        sources=True if _sources_health() else None,
        # CAMPUS_VERIFY_ANSWER_REUSE (default off): identical messages get the first answer at once.
        answer_reuse=True if answer_reuse.enabled() else None,
        tagalog_nli_model=settings.tagalog_model_label,
        facebook_refresh=FacebookRefreshModel(enabled=r.enabled, interval_minutes=r.interval_minutes, last_run=r.last_run,
                                              last_added=r.last_added, last_error=r.last_error),
    )


@app.post("/verify/text", response_model=VerificationResultModel)
def verify_text(request: VerifyTextRequest, response: Response = None) -> VerificationResultModel:
    """Route and verify against the live corpus (reloaded automatically when its files change)."""
    if not has_enough_letters(request.claim):           # input guard: nothing to check (no verdict, no evidence)
        raise HTTPException(status_code=400, detail=TOO_SHORT_MESSAGE)
    pipeline = state["manager"].current()
    folding = state["settings"].unicode_fold and pipeline.original_texts is not None

    def check():
        result = _gated(_answer, pipeline.verifier, request.claim,
                        normalization=_normalization_info(request.claim, pipeline.documents), sources=_source_index(pipeline),
                        evidence_verifier=pipeline.evidence_verifier, relevance_checker=pipeline.relevance_checker,
                        text_route=True)
        # Display only (C1): the message names another school that no shown post mentions. Text route only: the posts
        # here are DYCI's (a /verify/url page is not).
        result.scope_notice = school_scope.notice(fold(request.claim) if folding else request.claim,
                                                  [*result.retrieved_evidence, result.used_evidence])
        if folding:
            _restore_original_text(result, pipeline.original_texts, {d["document_id"]: d["text"] for d in pipeline.documents})
        from src.verification import org_pages
        if org_pages.enabled():
            result = _apply_org_pages(result, request.claim, pipeline)
        from src.ingestion import dyci_website
        if dyci_website.mode() == "2":
            result = _apply_website_confirmer(result, request.claim)
        return result

    try:
        if response is None or not answer_reuse.enabled():   # the picture check calls this directly: never reused
            return check()
        # CAMPUS_VERIFY_ANSWER_REUSE (default off): the same message, the same day, the same posts -> the same answer,
        # given at once (src/api/answer_reuse.py). The header only marks it; the answer itself is identical.
        from src.verification import post_age
        result, reused = state.setdefault("answers", answer_reuse.AnswerStore()).answer(
            pipeline, answer_reuse.key_for(request.claim, post_age.today()), check,
            keep=lambda r: answer_reuse.worth_keeping(request.claim, r), copy=lambda r: r.model_copy(deep=True))
        if reused:
            response.headers["X-Answer-Reused"] = "1"
        return result
    except ValueError as e:
        # The verifier raises ValueError for an empty claim; pydantic already rejects those (defense in depth).
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verify/url", response_model=VerificationResultModel)
def verify_url(request: VerifyUrlRequest) -> VerificationResultModel:
    """
    EPHEMERAL URL verification (M10.1): the submitted page becomes a one-document corpus for THIS request only, checked
    with the same models and settings as /verify/text. Nothing is stored; the live corpus is never touched.

    The URL and its full redirect chain are validated first (M10.2 SSRF guard): a refused destination is HTTP 400. A
    fetch that fails, or yields no usable text, is HTTP 502 -- never turned into INSUFFICIENT_EVIDENCE, so a caller can
    tell "we couldn't read your URL" apart from "the page doesn't settle your claim".
    """
    if not has_enough_letters(request.claim):
        raise HTTPException(status_code=400, detail=TOO_SHORT_MESSAGE)
    try:
        safe_url = check_redirect_chain_is_safe(request.url)
    except UnsafeUrlError as e:
        raise HTTPException(status_code=400, detail=f"Refusing to fetch this URL: {e}")
    from src.ingestion import dyci_website
    if dyci_website.enabled():
        # Rule W-d: dyci.edu.ph is a JavaScript app whose pages are empty shells; an item link is answered from the
        # stored website feed item instead of the page.
        item = dyci_website.item_for_link(safe_url, PROJECT_ROOT)
        if item is not None:
            return _check_page_record(item, request.claim)
    from src.api import link_source
    if link_source.enabled() and not link_source.is_official(safe_url):
        # Rule U (results/round2/link_source_plan.md; CAMPUS_VERIFY_LINK_SOURCE, default off): a link that is not DYCI's
        # never decides the verdict. The answer is the /verify/text answer from DYCI's own posts; what the page says is
        # reported beside it in link_source.
        dyci = verify_text(VerifyTextRequest(claim=request.claim))
        try:
            page_result = _check_fetched_page(safe_url, request.claim)
            note = link_source.info(safe_url, page_result)
        except HTTPException as e:
            from src.ingestion.facebook_page import scrub
            note = link_source.info(safe_url, None, scrub(str(e.detail)))
        return dyci.model_copy(update={"link_source": note})
    return _check_fetched_page(safe_url, request.claim)


def _website_supported(result, item: dict, found: dict, why: str):
    """Rule W v2: SUPPORTED by a stored DYCI website item through the rule-based match (never CONTRADICTED)."""
    evidence = RetrievedDocumentModel(document_id=item["document_id"], score=found["coverage"], text=item["text"],
                                      source_type=item.get("source_type") or "official_site",
                                      source_url=item.get("source_url") or "",
                                      published_date=item.get("published_date") or None, provenance="dyci_website")
    return result.model_copy(update={
        "verdict": "SUPPORTED", "used_evidence": evidence, "selected_passage": None, "classification_result": None,
        "date_check": None, "value_check": None, "unit_check": None, "value_withhold": None,
        "website_match": {"version": "website-v2", "document_id": item["document_id"], "match": found, "why": why,
                          "dyci_verdict": result.verdict},
        "explanation": "Supported by DYCI's website (dyci.edu.ph): every key word, number and name in the message "
                       "matches the website's announcement for that date."})


def _apply_website_confirmer(result, claim: str):
    """Rule W v2 V2-b (CAMPUS_VERIFY_WEBSITE=2; results/round2/website_source_v2_plan.md): when the served posts give
    INSUFFICIENT_EVIDENCE, or CONTRADICTED by a post naming none of the message's dates, the stored website items that
    name one of the message's dates are tried, newest first, with rule O v3.1's match. Never CONTRADICTED."""
    from src.ingestion import dyci_website
    from src.normalization.fold import fold
    from src.verification import org_pages
    if getattr(result, "verdict_withheld", None) or result.verdict not in ("INSUFFICIENT_EVIDENCE", "CONTRADICTED"):
        return result
    wanted = org_pages.dates_in(claim)
    if not wanted:
        return result
    if result.verdict == "CONTRADICTED":
        used = result.used_evidence
        if used is None or org_pages.dates_in(used.text) & wanted:
            return result                       # the contradicting post is about this day: it stands
    items = sorted(dyci_website.stored_items(PROJECT_ROOT), key=lambda r: r.get("published_date") or "", reverse=True)
    for item in items:
        text = org_pages.clean(fold(item.get("text") or ""))
        if not org_pages.dates_in(text) & wanted:
            continue
        found = org_pages.match(claim, text, strict_numbers=True)
        if found is not None:
            return _website_supported(result, item, found, "confirmer")
    return result


def _check_page_record(record: dict, claim: str) -> VerificationResultModel:
    """Rule W-d: the claim judged against one stored DYCI website item, as the link check judges a fetched page."""
    from src.api.source_info import SourceInfo
    page = {k: record.get(k, "") for k in ("document_id", "text", "source_type", "source_url")}   # the retriever's shape
    sources = {page["document_id"]: SourceInfo(published_date=record.get("published_date") or "",
                                               provenance="dyci_website")}
    try:
        if not state["settings"].unicode_fold:
            result = _gated(lambda: _answer(state["manager"].temp_verifier([page]), claim, sources=sources))
        else:
            folded = dict(page, text=fold(page["text"]))
            result = _gated(lambda: _answer(state["manager"].temp_verifier([folded]), claim, sources=sources))
            _restore_original_text(result, {page["document_id"]: page["text"]}, {page["document_id"]: folded["text"]})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    from src.ingestion import dyci_website
    if dyci_website.mode() == "2" and result.verdict == "INSUFFICIENT_EVIDENCE" \
            and not getattr(result, "verdict_withheld", None):
        # Rule W v2 V2-c: the link's own item, tried with the rule-based match.
        from src.verification import org_pages
        found = org_pages.match(claim, org_pages.clean(fold(record.get("text") or "")), strict_numbers=True)
        if found is not None:
            return _website_supported(result, record, found, "link")
    return result


def _check_fetched_page(safe_url: str, claim: str) -> VerificationResultModel:
    """The link check as it was before rule U: the fetched page is the only evidence (HTTP 502 when it cannot be read)."""
    try:
        # follow_redirects=False: fetch exactly the URL the guard checked; a redirect now is refused, not followed.
        evidence: NormalizedEvidence = WebsiteSourceAdapter(url=safe_url, source_type="official_site",
                                                            follow_redirects=False).fetch()
    except WebsiteFetchError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch the submitted URL: {e}")
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Fetched URL produced no usable evidence: {e}")
    from src.api import login_wall
    if login_wall.enabled() and login_wall.detect(evidence.text):
        # CAMPUS_VERIFY_LOGIN_WALL (default on since v1.9; scenario 10): a login page is a page that could not be read.
        raise HTTPException(status_code=502, detail=login_wall.MESSAGE)

    try:
        page = evidence.to_retriever_document()
        if not state["settings"].unicode_fold:
            return _gated(lambda: _answer(state["manager"].temp_verifier([page]), claim,
                                          sources=info_for_fetched_page(evidence)))
        folded_page = dict(page, text=fold(page["text"]))         # processing copy; the page text is displayed as fetched
        result = _gated(lambda: _answer(state["manager"].temp_verifier([folded_page]), claim,
                                        sources=info_for_fetched_page(evidence)))
        _restore_original_text(result, {page["document_id"]: page["text"]}, {page["document_id"]: folded_page["text"]})
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---- Image Check, Phase 1 (CAMPUS_VERIFY_IMAGE, default off) ----------------------------------------------------------
NO_TEXT_NOTE = "No readable text was found in the image, so there is nothing to check against the announcements."
TEXT_NOTE = ("The text read from the image was checked against the campus announcements. This checks what the image "
             "says, not whether the image itself was edited.")
UNCERTAIN_NOTE = ("The text read from the image could not be checked reliably: a tiny difference in how it was read "
                  "changed the result, so no verdict is shown. Type the message in the text box to check it.")


PHOTO_INDEX_ENV = "CAMPUS_VERIFY_PHOTO_INDEX"      # folder holding index.json; "none" switches the photo check off
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")


def _photo_index_path():
    """The photo index folder (default data/photo_index), or None when there is no index or it is switched off."""
    value = os.environ.get(PHOTO_INDEX_ENV, "").strip() or os.path.join(PROJECT_ROOT, "data", "photo_index")
    if value.lower() == "none" or not os.path.isfile(os.path.join(value, "index.json")):
        return None
    return value


def _photo_index():
    """Photo check (Phase 2A, owner decision 2026-10-24): the index of DYCI's posted photos, loaded on the first image
    request; None when there is none."""
    path = _photo_index_path()
    if path is None:
        return None
    from src.image import auto_index
    auto_folder, auto_stamp = None, None
    if auto_index.enabled() and state.get("manager") is not None:
        # Rule I (CAMPUS_VERIFY_AUTO_PHOTO_INDEX): the pictures the live fetch stored are added; reloaded when they change.
        fb = state["manager"].settings.fb_corpus
        auto_folder = auto_index.folder_for(fb if os.path.isabs(fb) else os.path.join(PROJECT_ROOT, fb))
        listing = os.path.join(auto_folder, "index.json")
        auto_stamp = os.path.getmtime(listing) if os.path.isfile(listing) else None
    key = (path, auto_folder, auto_stamp)
    if state.get("photo_index_path") != key:
        from dataclasses import replace
        from src.image.photo_match import load_index
        try:
            index, readings = load_index(path), {}
            if auto_folder is not None:
                entries, pages, readings = auto_index.load(auto_folder)
                index = replace(index, entries=[*index.entries, *entries], pages={**(index.pages or {}), **pages})
            state["photo_index"], state["photo_readings"], state["photo_index_error"] = index, readings, None
        except Exception as e:                    # a broken index switches the photo check off; it never breaks a check
            state["photo_index"], state["photo_readings"] = None, {}
            state["photo_index_error"] = f"{type(e).__name__}: {e}"[:300]
        state["photo_index_path"] = key
    return state["photo_index"]


def _edited_copy(found, upload_text: str) -> Optional[dict]:
    """Rule I-c: the upload matched a stored DYCI picture; a swapped value means it may have been edited. Display only."""
    from src.image import auto_index
    reading = (state.get("photo_readings") or {}).get(found.photo_id) if found is not None else None
    if not auto_index.enabled() or not reading:
        return None
    swap = auto_index.edited(reading, upload_text)
    if swap is None:
        return None
    try:
        y, mo, d = (int(x) for x in found.post_date.split("-"))
        when = f"{d} {MONTHS[mo - 1]} {y}"
    except (ValueError, IndexError):
        when = "an earlier date"
    swap["summary"] = (f"This picture matches DYCI's picture of {when}, but it reads {', '.join(swap['picture_values'])} "
                       f"where DYCI's picture reads {', '.join(swap['dyci_values'])}. It may have been edited.")
    return swap


def _photo_signals(index, found):
    """Display-only photo-check result with its measured error. Never changes a verdict; never says fake/real/proof."""
    if index is None:
        return None
    m = index.measured
    note = (f"The photo check compares the picture with {len(index.entries)} photos posted by DYCI and DYCI-related "
            f"pages. In testing it "
            f"found {m['found']} of {m['queries']} reused DYCI photos ({round(100 * m['found'] / m['queries'])}%) and "
            f"wrongly matched {m['false_matches']} of {m['negatives']} photos that DYCI never posted. No match does not "
            "mean the photo is new: the check only knows the photos in its collection, and it misses some screenshots "
            "and phone photos of posts.")
    if found is None:
        return {"photo_match": None, "summary": "This picture was not found among the DYCI photos the check knows.",
                "note": note, "measured": m}
    try:
        y, mo, d = (int(x) for x in found.post_date.split("-"))
        when = f"{d} {MONTHS[mo - 1]} {y}"
    except (ValueError, IndexError):
        when = "an earlier date"
    page = (index.pages or {}).get(found.photo_id) or "DYCI"
    where = f"DYCI's post of {when}" if page == "DYCI" else f"a post by {page}{',' if ', ' in page else ''} on {when}"
    return {"photo_match": {"post_id": found.post_id, "post_url": found.post_url, "post_date": found.post_date,
                            "page": page, "matched_by": found.stage},
            "summary": f"This photo also appears in {where}.", "note": note, "measured": m}


def _photo_post_check(found: dict, statements: List[str]) -> Optional[dict]:
    """Option B' (owner, 2026-10-24): compare the picture's statements with ONLY the post its photo came from, using that
    post's checked caption from the photo index. The shared text collection is not touched; display only."""
    index = state.get("photo_index")
    text = (index.texts or {}).get(found.get("post_id")) if index is not None else None
    if not text or not statements:
        return None
    page = {"document_id": f"photo-post:{found['post_id']}", "text": text, "source_type": "manual_entry",
            "source_url": found.get("post_url") or ""}
    verifier = state["manager"].temp_verifier([page])
    rows = [{"text": s[:200], "verdict": verifier.verify_claim(s).verdict} for s in statements]
    return {"post_id": found["post_id"], "checked": len(rows),
            "matches": sum(r["verdict"] == "SUPPORTED" for r in rows),
            "contradictions": sum(r["verdict"] == "CONTRADICTED" for r in rows), "statements": rows}


def _florence_reader():
    """Image Check option B (CAMPUS_VERIFY_IMAGE_READER=florence2; results/round2/image_check_b_plan.md)."""
    if state.get("florence") is None:                 # created on the first image request only
        from src.image.florence_reader import FlorenceReader
        state["florence"] = FlorenceReader()
    return state["florence"]


def _ocr_engine():
    if state.get("ocr") is None:                      # created on the first image request only
        from src.image.ocr import OCREngine
        state["ocr"] = OCREngine()
    return state["ocr"]


# Fix A (CAMPUS_VERIFY_POST_IMAGES): the live fetch reads post pictures with this same engine (one EasyOCR reader).
from src.ingestion import post_images as _post_images  # noqa: E402
_post_images.ENGINE_PROVIDER = _ocr_engine


async def _read_capped(request: Request, limit: int) -> bytes:
    """The request body, refused (413) as soon as it exceeds `limit`, whatever Content-Length said. Memory only."""
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if limit > 0 and total > limit:
            raise HTTPException(status_code=413, detail=f"Request body too large (limit {limit} bytes).")
        chunks.append(chunk)
    return b"".join(chunks)


def _org_verifier():
    """Rule O: (verifier, {document_id: (key, name)}, sources) over the named organisation pages, cached until an archive
    file changes; None when no page is named or none has posts."""
    from src.api.source_info import SourceInfo
    from src.verification import org_pages
    docs, owner, stamp = org_pages.load(PROJECT_ROOT)
    key = (tuple(sorted(org_pages.sources().items())), stamp)
    if org_pages.mode() == "verdict2":
        # Rule O v2: only posts from the last 90 days, with emoji and hashtags removed (the shown text keeps them).
        from src.verification import post_age
        today = post_age.today()
        state["org_originals"] = {d["document_id"]: d["text"] for d in docs}
        docs = [dict(d, text=org_pages.clean(d["text"])) for d in docs if org_pages.recent(d, today)]
        state["org_cleaned"] = {d["document_id"]: d["text"] for d in docs}
        key = key + ("v2", today.isoformat())
    if state.get("org_key") != key:
        state["org_key"], state["org"] = key, None
        if docs:
            sources = {d["document_id"]: SourceInfo(published_date=d.get("published_date") or "",
                                                    original_url=d.get("origin_url") or "",
                                                    original_date=d.get("origin_date") or "", provenance="org_page")
                       for d in docs}
            state["org"] = (state["manager"].temp_verifier(docs), owner, sources)
    return state["org"]


def _apply_org_pages(result, claim: str, pipeline):
    """Rule O (CAMPUS_VERIFY_ORG_PAGES; results/round2/org_pages_plan.md): DYCI first; when DYCI's posts give
    INSUFFICIENT_EVIDENCE, the organisation's own posts may decide, labelled as theirs, except on school-wide topics."""
    from src.verification import org_pages
    if org_pages.mode() in ("verdict3", "verdict31"):
        out = _apply_org_pages_v3(result, claim)
        if org_pages.related_fallback() and out.verdict == "INSUFFICIENT_EVIDENCE" \
                and not getattr(out, "verdict_withheld", None) and not getattr(out, "org_related", None):
            return _org_related_search(out, claim, pipeline)
        return out
    if result.verdict != "INSUFFICIENT_EVIDENCE" or getattr(result, "verdict_withheld", None):
        return result
    org = _org_verifier()
    if org is None:
        return result
    verifier, owner, sources = org
    org_result = _gated(_answer, verifier, claim, sources=sources, relevance_checker=pipeline.relevance_checker,
                        text_route=True)
    v2 = org_pages.mode() == "verdict2"
    if v2:                                  # the post is shown as posted, not as cleaned for the comparison
        _restore_original_text(org_result, state.get("org_originals") or {}, state.get("org_cleaned") or {})
    rel = org_pages.related(org_result, owner)
    wide = org_pages.school_wide_v2(claim) if v2 else org_pages.school_wide(claim)
    if org_pages.mode() == "related" or wide \
            or org_result.verdict not in ("SUPPORTED", "CONTRADICTED") \
            or org_result.used_evidence is None:
        return result.model_copy(update={"org_related": rel}) if rel else result
    key, name = owner.get(org_result.used_evidence.document_id, ("", ""))
    info = {"version": org_pages.VERSION, "key": key, "name": name, "dyci_verdict": result.verdict,
            "document_id": org_result.used_evidence.document_id, "post_url": org_result.used_evidence.source_url or "",
            "post_date": (org_result.used_evidence.published_date or "")[:10]}
    if v2:
        info.update({"version": "org-pages-v2", "label": org_pages.LABEL, "recent_days": org_pages.RECENT_DAYS})
    return org_result.model_copy(update={"org_source": info,
                                         "explanation": (org_result.explanation or "")
                                         + org_pages.note(name, org_result.verdict, v2)})


def _org_related_search(result, claim: str, pipeline):
    """CAMPUS_VERIFY_ORG_RELATED_FALLBACK: the page's closest post by meaning (all its posts, as in "related" mode),
    shown as related; display only, the verdict is never changed."""
    from src.verification import org_pages
    org = _org_verifier()
    if org is None:
        return result
    verifier, owner, sources = org
    org_result = _gated(_answer, verifier, claim, sources=sources, relevance_checker=pipeline.relevance_checker,
                        text_route=True)
    rel = org_pages.related(org_result, owner)
    return result.model_copy(update={"org_related": rel}) if rel else result


def _apply_org_pages_v3(result, claim: str):
    """Rule O v3 (CAMPUS_VERIFY_ORG_PAGES=3; results/round2/org_pages_v3_plan.md): a page post confirms by a rule-based
    match (same date, 85% of the content words, no value or unit conflict); never CONTRADICTED. Consulted when DYCI's
    answer is INSUFFICIENT_EVIDENCE, or CONTRADICTED by a DYCI post that names none of the message's dates."""
    from src.verification import org_pages, post_age
    if getattr(result, "verdict_withheld", None) or result.verdict not in ("INSUFFICIENT_EVIDENCE", "CONTRADICTED"):
        return result
    wanted = org_pages.dates_in(claim)
    if result.verdict == "CONTRADICTED":
        used = result.used_evidence
        if not wanted or used is None or org_pages.dates_in(used.text) & wanted:
            return result                       # DYCI's post is about this day: its answer stands
    posts, owner = org_pages.recent_clean_posts(PROJECT_ROOT, post_age.today())
    candidates = [p for p in posts if wanted and org_pages.dates_in(p["text"]) & wanted]

    def shown(p):
        key, name = owner.get(p["document_id"], ("", ""))
        return {"key": key, "name": name, "document_id": p["document_id"], "text": p["shown_text"][:300],
                "post_url": p.get("source_url") or "", "post_date": (p.get("published_date") or "")[:10]}
    wide = org_pages.school_wide_v2(claim)
    # Rule O-SW (CAMPUS_VERIFY_ORG_SCHOOL_WIDE, default off; results/round2/org_school_wide_plan.md): a school-wide
    # message may also be confirmed by a page, with the department note; otherwise it is never the page's to decide.
    if (wide and not org_pages.school_wide_enabled()) or not candidates:
        if result.verdict == "INSUFFICIENT_EVIDENCE" and candidates:
            return result.model_copy(update={"org_related": shown(candidates[0])})
        return result
    strict = org_pages.mode() == "verdict31"    # rule O v3.1: every number must appear in the post
    for p in candidates:                        # newest first
        found = org_pages.match(claim, p["text"], strict_numbers=strict)
        if found is None:
            continue
        info = shown(p)
        info.update({"version": "org-pages-v3.1" if strict else "org-pages-v3", "label": org_pages.LABEL,
                     "recent_days": org_pages.RECENT_DAYS,
                     "dyci_verdict": result.verdict, "match": found})
        info.pop("text")
        if wide:                                # rule O-SW: a school-wide message, the department's word
            info.update({"version": info["version"] + "-sw", "label": None, "school_wide": True,
                         "note": org_pages.school_wide_note(info["name"])})
        evidence = RetrievedDocumentModel(document_id=p["document_id"], score=found["coverage"], text=p["shown_text"],
                                          source_type=p.get("source_type") or "facebook",
                                          source_url=p.get("source_url") or "",
                                          published_date=p.get("published_date") or None, provenance="org_page")
        return result.model_copy(update={
            "verdict": "SUPPORTED", "used_evidence": evidence, "selected_passage": None, "classification_result": None,
            "date_check": None, "value_check": None, "unit_check": None, "value_withhold": None, "org_source": info,
            "explanation": info["note"] if wide else org_pages.note(info["name"], "SUPPORTED", True).strip()})
    if result.verdict == "INSUFFICIENT_EVIDENCE":
        return result.model_copy(update={"org_related": shown(candidates[0])})
    return result


def _plain_verify_text(req):
    return verify_text(req)


def _poster_verify_text(req):
    """verify_text with rule R's language gate (R-e) active; set inside the worker thread that runs the check."""
    from src.image.poster_cleanup import POSTER_ACTIVE
    token = POSTER_ACTIVE.set(True)
    try:
        return verify_text(req)
    finally:
        POSTER_ACTIVE.reset(token)


def _language_verify_text(check):
    """Rule G (CAMPUS_VERIFY_POSTER_LANGUAGE; results/round2/poster_language_plan.md): `check` run with G active in the
    worker thread; when G released the verdict, the answer carries poster_language."""
    def run(req):
        from src.image import poster_language
        notes = []
        active, released = poster_language.ACTIVE.set(True), poster_language.RELEASED.set(notes)
        try:
            result = check(req)
        finally:
            poster_language.ACTIVE.reset(active)
            poster_language.RELEASED.reset(released)
        if notes and hasattr(result, "model_copy"):
            result = result.model_copy(update={"poster_language": notes[0]})
        return result
    return run


@app.post("/verify/image", response_model=ImageCheckResponse)
async def verify_image(request: Request) -> ImageCheckResponse:
    """Image Check Phase 1: read the text in an uploaded image (in memory only) and check it with the unchanged
    /verify/text logic. The image is never written to disk or logged, and neither is the text read from it."""
    if not state["settings"].image_check:
        raise HTTPException(status_code=404, detail="Not Found")          # the standard 404 body, as in v1.3
    import time as _time
    from src.image.intake import ImageRejected, load
    timings = {}
    t0 = _time.perf_counter()
    data = await _read_capped(request, state["protection"].max_image_bytes)
    content_type = request.headers.get("content-type")
    facts = {}

    def intake_and_ocr():
        t = _time.perf_counter()
        image = load(data, content_type)
        timings["intake"] = round((_time.perf_counter() - t) * 1000, 1)
        from src.image import provenance
        if provenance.enabled():          # Phase 2B P1 (results/round2/ai_image_plan_draft.md), default off; display only
            t = _time.perf_counter()
            from src.image.intake import media_type
            facts["provenance"] = provenance.check(data, media_type(content_type))
            timings["provenance"] = round((_time.perf_counter() - t) * 1000, 1)
        t = _time.perf_counter()
        from src.image import florence_reader, ui_filter
        if florence_reader.reader_name() == florence_reader.FLORENCE:   # option B, with option A's interface rules
            result, filtered = _florence_reader().read_filtered(image)
        elif ui_filter.enabled():         # Image Check v2 option A (results/round2/image_check_v2_plan.md), default off
            result, filtered = ui_filter.read_filtered(_ocr_engine(), image)
        else:
            result, filtered = _ocr_engine().read(image), None
        timings["ocr"] = round((_time.perf_counter() - t) * 1000, 1)
        index, found = _photo_index(), None
        if index is not None:                                         # same in-memory image; nothing is stored
            from src.image.photo_match import match as photo_match
            t = _time.perf_counter()
            found = photo_match(image, index.entries, index.max_hamming, index.min_inliers)
            timings["photo_match"] = round((_time.perf_counter() - t) * 1000, 1)
        signals = _photo_signals(index, found)
        if signals is not None and found is not None:
            edited = _edited_copy(found, result.text)          # rule I-c, display only (CAMPUS_VERIFY_AUTO_PHOTO_INDEX)
            if edited is not None:
                signals["edited_copy"] = edited
        return (result, filtered), signals

    try:
        (ocr, filtered), photo_signals = await run_in_threadpool(_gated, intake_and_ocr)   # decode, OCR, photo check
    except ImageRejected as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    finally:
        del data                                                          # the upload is not kept past this point
    from src.image.normalize import agreement_variant, normalize_ocr_text
    cleaned = normalize_ocr_text(filtered.text if filtered is not None else ocr.text)   # every change is reported
    if filtered is not None:                                              # ocr.text stays the full reading
        cleaned.changes[:0] = [f"interface filter ({r['reason']}): {r['text']}" for r in filtered.removed]
    text, truncated = cleaned.text, False
    # Rule D (results/round2/poster_dates_plan.md; CAMPUS_VERIFY_POSTER_DATES, default off): letter-spaced dates
    # ("NOVEMBER 3 0, 20 26") are rejoined before the check; every change is reported.
    from src.image import poster_dates
    dates_info = None
    if poster_dates.enabled():
        text, date_changes = poster_dates.rejoin(text)
        if date_changes:
            cleaned.changes.extend(f"date: {c}" for c in date_changes)
            dates_info = {"version": poster_dates.VERSION, "changes": date_changes}
    # Rule E (results/round2/poster_core_plan.md; CAMPUS_VERIFY_POSTER_CORE, default off): the poster's seal, school-name
    # header, label, issue date and footer are set aside; the core is checked. Every removal is reported.
    from src.image import poster_core
    core_info = None
    if poster_core.enabled():
        core_text, core_info = poster_core.core(text)
        if core_info and has_enough_letters(core_text):
            text = core_text
            cleaned.changes.extend(f"poster: set aside “{r[:80]}”" for r in core_info["removed"])
        else:
            core_info = None
    # Rule R (results/round2/poster_cleanup_plan.md; CAMPUS_VERIFY_POSTER_CLEANUP, default off): a poster's seal, school
    # name, label, issue date and footer are set aside before the check; every change is reported.
    from src.image import poster_cleanup
    poster = None
    if poster_cleanup.enabled():
        pc = poster_cleanup.clean(text)
        if pc.changes and has_enough_letters(pc.text):
            text = pc.text
            cleaned.changes.extend(f"poster: {c}" for c in pc.changes)
            poster = pc.to_dict()
    check_text = _poster_verify_text if poster_cleanup.enabled() else _plain_verify_text
    from src.image import poster_language
    if poster_language.enabled():
        check_text = _language_verify_text(check_text)
    if len(text) > MAX_CLAIM_CHARS:
        text, truncated = text[:MAX_CLAIM_CHARS].rsplit(" ", 1)[0], True
    text_check, note, status, agreement = None, NO_TEXT_NOTE, "no_text", None
    # Source rule (results/round2/image_source_rule_plan.md; CAMPUS_VERIFY_IMAGE_SOURCE_RULE, default off): a screenshot
    # whose Facebook header names another institution is not checked; the closest DYCI announcements are shown instead.
    from src.image import source_rule
    source = None
    if filtered is not None and source_rule.enabled():
        source = source_rule.judge([r["text"] for r in filtered.removed if r["reason"].startswith("post header")])
        if source["status"] == "other_page" and has_enough_letters(text):
            pipeline = state["manager"].current()
            info = await run_in_threadpool(_gated, pipeline.verifier.retrieve_information, text)
            sources = _source_index(pipeline)
            source["closest"] = [_retrieved_document_to_model(d, sources).model_dump() for d in info.retrieved_evidence[:3]]
            note, status = source["note"], "other_page"
    if has_enough_letters(text) and status != "other_page":
        t = _time.perf_counter()
        # Exactly the /verify/text route logic (input guard, gate, verdict, notices), called after the image slot was
        # released so a slot is never held twice.
        text_check = await run_in_threadpool(check_text, VerifyTextRequest(claim=text))
        note, status = TEXT_NOTE, "checked"
        variant = agreement_variant(text) if text_check.verdict in ("SUPPORTED", "CONTRADICTED") else None
        if variant is not None:
            # Agreement check (no threshold): the same text with its final dot toggled, which OCR often gets wrong.
            # If the verdict changes, the reading is too fragile to support a verdict, so none is shown.
            other = await run_in_threadpool(check_text, VerifyTextRequest(claim=variant))
            agreement = {"rule": "final_full_stop_toggled", "agreed": other.verdict == text_check.verdict}
            if not agreement["agreed"]:
                text_check, note, status = None, UNCERTAIN_NOTE, "uncertain"
        timings["text_check"] = round((_time.perf_counter() - t) * 1000, 1)
    # Part B (decision record 2026-10-15): sentence-level checks decide the displayed result when any sentence qualifies;
    # the whole-text check above is kept unchanged and never overrides or upgrades them.
    from src.image import sentences as sent
    plan = sent.plan_sentences(text) if status not in ("no_text", "other_page") else []
    basis = sent.select_basis(text, plan) if status != "other_page" else "none"
    sentence_checks, summary, whole_text_agreement = [], None, None
    if basis == "sentences":
        t = _time.perf_counter()
        # One /verify/text call per statement (plus its agreement check), each in its own gate slot; any 503 fails the
        # whole request, so a partial result is never returned.
        results = await run_in_threadpool(sent.check_statements, plan,
                                          lambda s: check_text(VerifyTextRequest(claim=s)))
        timings["sentence_checks"] = round((_time.perf_counter() - t) * 1000, 1)
        counts = sent.count_outcomes(plan, results)
        headline = sent.aggregate(counts, results)
        by_index = {r.index: r for r in results}
        for seg in plan:
            r = by_index.get(seg.index)
            sentence_checks.append(SentenceCheckModel(
                index=seg.index, text=seg.text, status=r.status if r else seg.status,
                outcome=r.outcome if r else None, verdict=r.verdict if r else None,
                agreement=r.agreement if r else None, text_check=r.result if r else None))
        summary = SentenceSummaryModel(segments=len(plan), checked=counts.n, matches=counts.k,
                                       contradictions=counts.c, inconclusive=counts.i,
                                       not_checked_short=counts.not_checked_short,
                                       not_checked_limit=counts.not_checked_limit, kind=headline["kind"],
                                       label=headline["label"], anchored_posts=headline.get("anchored_posts", []),
                                       option_a_kind=headline.get("option_a_kind"))
        whole = text_check.verdict if text_check is not None else ("uncertain" if status == "uncertain" else None)
        disagrees = ((whole == "CONTRADICTED" and counts.c == 0)
                     or (whole == "SUPPORTED" and (counts.k == 0 or counts.c > 0)))
        whole_text_agreement = {"whole_text_verdict": whole, "sentence_kind": headline["kind"], "disagrees": disagrees}
    elif basis == "whole_text_fallback":
        sentence_checks = [SentenceCheckModel(index=s.index, text=s.text, status=s.status) for s in plan]
    found = (photo_signals or {}).get("photo_match")
    if found:
        statements = ([s.text for s in plan if s.status == "to_check"] if basis == "sentences"
                      else [text] if basis == "whole_text_fallback" else [])
        t = _time.perf_counter()
        cross = await run_in_threadpool(_gated, lambda: _photo_post_check(found, statements))
        timings["photo_post_check"] = round((_time.perf_counter() - t) * 1000, 1)
        if cross is not None:
            photo_signals["photo_post_check"] = cross
    timings["total"] = round((_time.perf_counter() - t0) * 1000, 1)
    response = ImageCheckResponse(ocr=OCRModel(text=ocr.text, checked_text=text, normalization=cleaned.changes,
                                               confidence=ocr.confidence, lines=ocr.lines, truncated=truncated),
                                  check_basis=basis, sentence_checks=sentence_checks, sentence_summary=summary,
                                  whole_text_agreement=whole_text_agreement,
                                  status=status, text_check=text_check, agreement_check=agreement, note=note,
                                  image_signals=photo_signals, timings_ms=timings, source_check=source,
                                  provenance=facts.get("provenance"), poster_cleanup=poster, poster_dates=dates_info,
                                  poster_core=core_info)
    # Legitimacy level (results/round2/legitimacy_levels_spec.md): a summary of the results above; it changes none of them.
    from src.image.levels import assess, corpus_post_index, post_key
    found = (photo_signals or {}).get("photo_match")
    photo_doc_id = None
    if found:
        photo_doc_id = corpus_post_index(state["manager"].current().documents).get(post_key(found.get("post_url")))
    response.assessment = assess(response.model_dump(), photo_doc_id)
    # Rule F (results/round2/changed_date_plan.md; CAMPUS_VERIFY_CHANGED_DATE, default off): a picture that closely
    # matches the post the whole-text check used, but gives a date up to 7 days off, is put at Level 2 with the reason.
    from src.image import changed_date
    if changed_date.enabled():
        info = changed_date.check(text, text_check, response.assessment.get("level"))
        if info:
            response.assessment = changed_date.apply(response.assessment, info)
            response.changed_date = info
    # Option B (results/round2/text_conflict_level_plan.md; CAMPUS_VERIFY_TEXT_CONFLICT_LEVEL, default off): a picture
    # whose whole text is CONTRADICTED by a post it closely matches goes from Level 2-4 to Level 1; no verdict changes.
    from src.image import text_conflict_level
    if text_conflict_level.enabled():
        info = text_conflict_level.check(text, text_check, response.assessment.get("level"))
        if info:
            response.assessment = text_conflict_level.apply(response.assessment, info)
            response.text_conflict_level = info
    # Option A (results/round2/picture_text_line_plan.md; CAMPUS_VERIFY_PICTURE_TEXT_LINE, default off): the level card
    # also shows the verdict the picture's words get as typed text; no verdict or level is changed.
    from src.image import text_line
    if text_line.enabled():
        response.text_line = text_line.build(text_check, response.assessment.get("level"))
    return response


# ---- hosting (docs/deploy.md) -----------------------------------------------------------------------------------------
# CAMPUS_VERIFY_SERVE_FRONTEND=1 (default off): the API also serves the page from frontend/, so one process and one
# address host everything. js/config.js is answered here with an empty API_BASE_URL, so the page calls this same
# origin; the file on disk (local development, http://127.0.0.1:8000) is left as it is. Read once at import.
FRONTEND_ENV = "CAMPUS_VERIFY_SERVE_FRONTEND"
SAME_ORIGIN_CONFIG = "window.CAMPUS_VERIFY_CONFIG = { API_BASE_URL: \"\", REQUEST_TIMEOUT_MS: 120000 };\n"

if os.environ.get(FRONTEND_ENV, "0").strip() == "1":
    from fastapi.staticfiles import StaticFiles

    @app.get("/js/config.js", include_in_schema=False)
    def same_origin_config():
        return Response(SAME_ORIGIN_CONFIG, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=os.path.join(PROJECT_ROOT, "frontend"), html=True), name="frontend")
