"""
Pydantic request/response models for the M10.1 API.

Responsibility of this module, and ONLY this module: define the HTTP-facing
shapes and validate raw request bodies. These models deliberately MIRROR
the existing dataclasses in src/verification/verifier.py,
src/classification/result.py, and src/retrieval/retriever.py field-for-
field -- this file does not introduce any new concept, label, or piece of
data that doesn't already exist in the underlying pipeline. Converting a
VerificationResult into a VerificationResultModel (done in
src/api/main.py) is a pure, lossless re-shaping for JSON serialization,
not a reinterpretation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _reject_blank(value: str) -> str:
    """
    Shared validator: mirrors ClaimVerifier.verify_claim()'s own
    `if not claim or not claim.strip(): raise ValueError(...)` check, so
    a blank/whitespace-only claim is rejected at the API boundary with a
    clear 422 rather than being passed through to the pipeline and
    surfacing as a less obvious error later.
    """
    if not value.strip():
        raise ValueError("must not be empty or whitespace-only")
    return value


# Longest accepted input. Real messages are a sentence or two; anything longer is rejected (HTTP 422) before any model
# runs, which also bounds the work one request can cause.
MAX_CLAIM_CHARS = 1000
# Input guard (2026-10-13): a message with fewer letters than this cannot be a claim or question ("a", emoji only).
MIN_CLAIM_LETTERS = 3
TOO_SHORT_MESSAGE = ("Please type a claim or question about a DYCI announcement, for example \u201cWalang pasok sa "
                     "Sept 28?\u201d or \u201cThe midterms are on September 28 to 30.\u201d")


def has_enough_letters(text: str) -> bool:
    return sum(ch.isalpha() for ch in text or "") >= MIN_CLAIM_LETTERS
MAX_URL_CHARS = 2048


class VerifyTextRequest(BaseModel):
    """Request body for POST /verify/text."""
    claim: str = Field(..., min_length=1, max_length=MAX_CLAIM_CHARS, description="The claim text to verify.")

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, v: str) -> str:
        return _reject_blank(v)


class VerifyUrlRequest(BaseModel):
    """
    Request body for POST /verify/url.

    NOTE: `url` is validated as a non-empty string, not Pydantic's
    stricter `HttpUrl` type. WebsiteSourceAdapter's own `_fetch_html()`
    is what actually validates reachability and content -- duplicating a
    second, separate URL-syntax validator here would just be a second
    place for that logic to disagree with the one that actually matters.
    A malformed URL still fails cleanly, just as a WebsiteFetchError
    from the adapter (mapped to HTTP 502 in main.py), not as a 422.
    """
    claim: str = Field(..., min_length=1, max_length=MAX_CLAIM_CHARS, description="The claim text to verify.")
    url: str = Field(..., min_length=1, max_length=MAX_URL_CHARS, description="The URL to fetch as evidence for this one request.")

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, v: str) -> str:
        return _reject_blank(v)

    @field_validator("url")
    @classmethod
    def url_not_blank(cls, v: str) -> str:
        return _reject_blank(v)


class RetrievedDocumentModel(BaseModel):
    """Mirrors src.retrieval.retriever.RetrievedDocument field-for-field."""
    document_id: str
    score: float = Field(..., description="Topical similarity score. Higher = more relevant. NOT a truth signal.")
    text: str
    source_type: str
    source_url: str
    # Display metadata (src/api/source_info.py); not used by retrieval or the verdict. Null when unknown.
    published_date: Optional[str] = Field(default=None, description="Date on the post/notice (for a mock post: when it was posted on the test Page).")
    original_url: Optional[str] = Field(default=None, description="The original announcement, when known (source map).")
    original_date: Optional[str] = Field(default=None, description="The original announcement's date, when known.")
    synthetic: bool = Field(default=False, description="True for the development dataset's fictional notices.")
    provenance: Optional[str] = Field(default=None, description="How the document was acquired: test_page_mirror (a DYCI post re-posted to the project's test Facebook Page; not DYCI's own Page), manual_entry, submitted_page or fictional.")


class ClassificationResultModel(BaseModel):
    """Mirrors src.classification.result.ClassificationResult field-for-field."""
    label: str
    confidence: float = Field(..., description="The model's own score for `label`. NOT a calibrated real-world probability.")
    label_scores: Dict[str, float]
    model: Optional[str] = Field(default=None, description="Which NLI model judged the claim, when language routing is on.")


class NormalizationInfoModel(BaseModel):
    """
    Sprint 2/3 addition (NOT a mirror of a pipeline dataclass): what the Taglish normalizer saw and did, plus the
    features the claim router needs. Informational only -- it never changes the verdict. Populated only when the
    server runs with CAMPUS_VERIFY_NORMALIZE=1; otherwise VerificationResultModel.normalization is null.
    Keep field names in sync with src/normalization/report.py (enforced by tests/test_normalization_report.py).
    """
    original_user_input: str
    normalized_claim: str
    changed: bool
    edit_rules: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    short_claim: bool
    has_number: bool
    code_switched: bool
    token_count: int
    content_token_count: int
    normalizer_version: str
    lexicon_sha256: str


class RouteInfoModel(BaseModel):
    """
    M12.3 addition: what ClaimRouter decided about the RAW input before verification (see src/routing/router.py).
    CHECKABLE_CLAIM inputs are verified; INFO_SEEKING inputs get evidence and no verdict. Null only when the router is
    switched off (CAMPUS_VERIFY_ROUTER=0).
    Keep field names in sync with ROUTE_DECISION_KEYS in src/routing/router.py (enforced by tests/test_claim_router.py).
    """
    status: str = Field(..., description="CHECKABLE_CLAIM or INFO_SEEKING.")
    reasons: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    guidance: str = ""
    signals: Dict[str, Any] = Field(default_factory=dict)
    router_version: str


class SelectedPassageModel(BaseModel):
    """Mirrors src.retrieval.passage_selector.SelectedPassage: the sentence actually used (M12.2 passage selection)."""
    document_id: str
    text: str
    score: float = Field(..., description="Passage similarity to the input. Ranking use only; NOT a truth signal.")
    passage_index: int
    context_before_text: str = ""


class ClaimRewriteModel(BaseModel):
    """
    M12.3 opt-in (CAMPUS_VERIFY_CLAIM_REWRITE=optionB): the yes/no-question-to-statement rewrite that was checked
    instead of the claim as typed. `original` is always the user's input, unchanged.
    Keep field names in sync with CLAIM_REWRITE_KEYS in src/normalization/question_form.py (tests enforce it).
    """
    original: str
    rewritten: str
    changed: bool
    rules_applied: List[str] = Field(default_factory=list)
    refused: Optional[str] = None
    version: str
    rules_sha256: str


class VerificationResultModel(BaseModel):
    """
    Mirrors src.verification.verifier.VerificationResult field-for-field,
    plus one addition: `data_status`, so every response is self-describing
    about the maturity of the evidence/classifier it was produced from,
    rather than a client having to separately call /health to find out.

    M12.3: when route.status is INFO_SEEKING the input was not verified -- `verdict` and `classification_result` are
    null and the evidence fields show what retrieval found for the question.
    """
    claim: str
    verdict: Optional[str] = Field(
        ...,
        description="SUPPORTED, CONTRADICTED or INSUFFICIENT_EVIDENCE for a checkable claim; null when the router read "
                    "the input as INFO_SEEKING (no truth verdict is given for a request for information).",
    )
    evidence_status: str
    retrieved_evidence: List[RetrievedDocumentModel]
    used_evidence: Optional[RetrievedDocumentModel]
    classification_result: Optional[ClassificationResultModel]
    explanation: str
    data_status: str = Field(
        ...,
        description="'development' -- this API currently runs on the existing "
                    "small development dataset/corpus, not a production-scale, "
                    "fully validated evidence base.",
    )
    normalization: Optional[NormalizationInfoModel] = Field(
        default=None,
        description="Taglish normalization report and router features; null unless CAMPUS_VERIFY_NORMALIZE=1. "
                    "Informational only: the verdict is computed from the claim exactly as submitted.",
    )
    route: Optional[RouteInfoModel] = Field(
        default=None,
        description="ClaimRouter decision (CHECKABLE_CLAIM / INFO_SEEKING) on the raw input; null only when the router "
                    "is switched off (CAMPUS_VERIFY_ROUTER=0).",
    )
    selected_passage: Optional[SelectedPassageModel] = Field(
        default=None,
        description="The passage used, when passage selection is configured (CAMPUS_VERIFY_RETRIEVER=e5); otherwise null.",
    )
    claim_rewrite: Optional[ClaimRewriteModel] = Field(
        default=None,
        description="The rewritten claim that was checked, when CAMPUS_VERIFY_CLAIM_REWRITE=optionB; otherwise null.",
    )
    date_check: Optional[str] = Field(
        default=None,
        description="Set when the date check withheld the classifier's verdict: 'claimed_date_not_in_evidence' or "
                    "'evidence_has_no_date'. classification_result still shows the classifier's own label.",
    )
    value_check: Optional[str] = Field(
        default=None,
        description="Set when the value check changed the verdict: 'value_conflict' (INSUFFICIENT_EVIDENCE -> "
                    "CONTRADICTED) or 'value_conflict_withheld' (SUPPORTED -> INSUFFICIENT_EVIDENCE).",
    )
    verdict_withheld: Optional[str] = Field(
        default=None,
        description="'language' when the claim was flagged Tagalog/Taglish and CAMPUS_VERIFY_VERDICT_LANGUAGES=english: "
                    "the closest announcements are returned and no truth verdict is given.",
    )
    sliding_windows: Optional[dict] = Field(
        default=None,
        description="Set when CAMPUS_VERIFY_PREMISE_MODE=sliding: windows judged, non-neutral counts, whether "
                    "SUPPORTED and CONTRADICTED windows conflicted, and the deciding window.",
    )
    date_notice: Optional[dict] = Field(
        default=None,
        description="Display only. Set when the message names an explicit date (date_check.extract_dates) that a shown "
                    "post does not contain anywhere in its text: message_dates, and per document_id the missing dates "
                    "and the sentence shown ('This post does not mention September 27.'). Never affects a decision.",
    )
    scope_notice: Optional[dict] = Field(
        default=None,
        description="Display only, /verify/text only. Set when the message names a school other than DYCI that no shown "
                    "post mentions: schools, and the sentence shown ('This checker only covers DYCI announcements. The "
                    "posts shown are DYCI's, not BulSU's.'). Never affects a decision.",
    )
    relative_date_notice: Optional[dict] = Field(
        default=None,
        description="Display only. Set when the message has a relative day word ('tomorrow', 'bukas' ...): words, and per "
                    "shown document_id the post's recorded date (or null) and the sentence shown. Never affects a decision.",
    )
    post_age_notice: Optional[dict] = Field(
        default=None,
        description="Display only (CAMPUS_VERIFY_POST_AGE, default on). Set when a SUPPORTED/CONTRADICTED message about "
                    "now (no date named) rests on a post older than max_days days or undated: document_id, post_date, "
                    "post_age_days (null when undated), today, max_days, and the warning shown. Never affects a decision.",
    )
    relative_date_check: Optional[dict] = Field(
        default=None,
        description="Set when CAMPUS_VERIFY_RELATIVE_DATE_CHECK judged a SUPPORTED/CONTRADICTED verdict on a message with a "
                    "relative day word: words, post_date, today, original_verdict, and withheld (true when the verdict "
                    "was changed to INSUFFICIENT_EVIDENCE because the post used is not from today or yesterday).",
    )
    other_school_check: Optional[dict] = Field(
        default=None,
        description="Set when CAMPUS_VERIFY_OTHER_SCHOOL_CHECK judged a SUPPORTED/CONTRADICTED verdict on a message naming "
                    "another school: schools, missing_in_post, original_verdict, and withheld (true when the verdict was "
                    "changed to INSUFFICIENT_EVIDENCE because the post used does not mention the school).",
    )
    hearsay_notice: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_HEARSAY_NOTE (default off): words, text. Present only when the message is phrased as a "
                    "prediction or hearsay; warn-only, never changes a verdict. Left out of the response when not set.",
    )
    relative_resolution: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_RESOLVE_RELATIVE (default off; results/round2/relative_resolve_plan.md): words, today, "
                    "dates, read_as, checked_as. Set when today/tomorrow/yesterday were replaced by Philippine-time "
                    "calendar dates before checking. Left out of the response when not set.",
    )
    holiday_notice: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_HOLIDAYS H3 (default off; results/round2/holidays_plan.md): version, dates, text. Set "
                    "when the message names a date on the 2026 national holiday list (Proclamation No. 1006, s. 2025). "
                    "Display only. Left out of the response when not set.",
    )
    question_focus: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_QUESTION_FOCUS (default on since v2.3; results/round2/question_focus_v2_report.md): version, "
                    "sentences, question. Set when a message of 2+ sentences with exactly one question was judged by "
                    "that question. Left out of the response when not set.",
    )
    class_day_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_CLASS_DAY_GUARD (default off; results/round2/class_day_guard_plan.md): version, "
                    "message_dates, post_used, nearest_post_date_days, near_days, withheld, original_verdict. Set when "
                    "a class-day message with an explicit date got a SUPPORTED/CONTRADICTED from a DYCI post. Left out "
                    "of the response when not set.",
    )
    same_day_advisory: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_SAME_DAY_ADVISORY (default off; results/round2/cross_post_conflict_plan.md, Amendment 2): "
                    "version, message_date, top_n, advisories (document_id, rank, score, label), post_used_before, "
                    "original_verdict, verdict, action (supported / disagree_withheld / stale_withheld / shown_only). "
                    "Set when an advisory dated the message's day was found among the top posts. Left out of the "
                    "response when not set.",
    )
    same_day_shown: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_SAME_DAY_SHOW (default off; results/round2/same_day_show_plan.md): display only. version, "
                    "message_date, top_n, added (document_id, rank, score). Set when an advisory dated the message's day "
                    "was placed first among the shown posts. Left out of the response when not set.",
    )
    same_day_lookup: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_SAME_DAY_LOOKUP (default off; results/round2/same_day_lookup_plan.md): display only. "
                    "version, message_date, added (document_id, rank or null, score). Set when advisories dated the "
                    "message's day were found by header date and placed first. Left out of the response when not set.",
    )
    reason_parts: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_REASON_PARTS (default off; results/round2/reason_parts_plan.md): version, main_claim, "
                    "reason, main_verdict, main_post, reason_label (backed / different / not_stated), reason_piece, "
                    "original_verdict, verdict, action. Set when a message with a reason marker ('due to', 'because "
                    "of' ...) was checked part by part. Left out of the response when not set.",
    )
    main_point: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_MAIN_POINT (default off; results/round2/main_point_v2_plan.md): version, main, "
                    "reason_phrase, inner_verdict, inner_post, outer_post, outcome (main_not_supported / "
                    "main_supported). Set when a SUPPORTED message with a reason marker had its main point checked "
                    "alone. Left out of the response when not set.",
    )
    unit_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_UNIT_CHECK (default off; results/round2/unit_check_plan.md): set when a SUPPORTED was "
                    "withheld because the message names another office, level, grade or venue than the post used: "
                    "version, kind, message_unit, post_units, action. Left out of the response when not set.",
    )
    website_match: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_WEBSITE=2 (default off; results/round2/website_source_v2_plan.md): set when a DYCI "
                    "website item confirmed the message by the rule-based match: version, document_id, match, why, "
                    "dyci_verdict.",
    )
    contra_relevance: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_CONTRA_RELEVANCE (default off; results/round2/contra_relevance_plan.md): set when a "
                    "CONTRADICTED answer became Not confirmed because its post shares less than a third of the "
                    "message's content words: version, share, words, found, post.",
    )
    poster_language: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_POSTER_LANGUAGE (default off; results/round2/poster_language_plan.md): set on a "
                    "picture's text check when rule G released the verdict: version, released, ignored (the school's "
                    "name tokens), stray (the one extra-list word).",
    )
    input_cleanup: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_ACADEMIC_YEAR / CAMPUS_VERIFY_LOOKALIKE (default off; results/round2/"
                    "input_hardening_plan.md): set when the message was checked after rewriting 'AY <years>' as "
                    "'A.Y.' or mapping lookalike letters: version, academic_year (count), lookalike (count).",
    )
    class_wording: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_CLASS_WORDING (default off; results/round2/class_wording_plan.md): set when the message "
                    "was checked in the standard class wording: version, original, rewritten, rules (plural, date_first).",
    )
    org_source: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_ORG_PAGES (default off; results/round2/org_pages_plan.md): set when DYCI's posts gave "
                    "no answer and a named student organisation's own post decided: version, key, name, dyci_verdict, "
                    "document_id, post_url, post_date. The verdict is theirs, not an official DYCI announcement.",
    )
    org_related: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_ORG_PAGES: the organisation's closest relevant post, shown as related only (no verdict "
                    "from it): key, name, document_id, text, post_url, post_date.",
    )
    value_withhold: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_VALUE_WITHHOLD (default off; results/round2/value_withhold_plan.md): set when a "
                    "SUPPORTED was withheld because the message gives another time, duration, percentage, date or place "
                    "than the post used: version, kind, message_value, post_values, action. Left out when not set.",
    )
    link_source: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_LINK_SOURCE (default off; results/round2/link_source_plan.md): set when a submitted "
                    "link is not DYCI's. The verdict above then comes from DYCI's own posts; this field holds the host, "
                    "official=false, whether the page was read, and what it says (page_verdict, page_says, "
                    "page_sentence) or why it could not be read. Left out of the response when not set.",
    )
    same_day_judge: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_SAME_DAY_JUDGE (Z v2, default off; results/round2/same_day_judge_plan.md): version, "
                    "message_date, advisories (document_id, rank, label, guard_g), post_used_before, original_verdict, "
                    "verdict, action (supported / disagree_withheld / stale_withheld / shown_only). Left out of the "
                    "response when not set.",
    )
    header_date_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_HEADER_DATE (default off; results/round2/header_date_plan.md): version, post_used, "
                    "header_date, original_verdict, verdict. Set when a SUPPORTED withheld only because the advisory "
                    "gives its date in the header was restored. Left out of the response when not set.",
    )
    holiday_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_HOLIDAYS H1: date, holiday, kind, statement, source, label, confidence, applied, "
                    "original_verdict, warning. Set when an INSUFFICIENT_EVIDENCE answer was judged against the holiday "
                    "list; it changed the verdict only when applied. Left out of the response when not set.",
    )
    tagalog_translation: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_TAGALOG_TRANSLATE (default on since v2.3; results/round2/tagalog_translate_v2_report.md): version, "
                    "model, model_sha256, instruction_sha256, translation, checks, accepted, seconds. Set when a "
                    "Tagalog/Taglish message was sent to the local translator; it replaced the message only when "
                    "accepted. Left out of the response when not set.",
    )
    tagalog_topic_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_TAGALOG_TOPIC_GUARD (default off; results/round2/tagalog_verdicts_plan.md): version, "
                    "post_used, shared_topic_words, withheld, original_verdict. Set when a Tagalog/Taglish SUPPORTED or "
                    "CONTRADICTED was checked for a shared topic word with its post. Left out of the response when not set.",
    )
    tagalog_reading: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_TAGALOG_PHRASES (default off; results/round2/tagalog_phrases_plan.md): version, frame, "
                    "reading, vocabulary_sha256, vocabulary_size. Set when a Tagalog/Taglish message fitted a fixed frame "
                    "and was checked as its English reading. Left out of the response when not set.",
    )
    tense_guard_check: Optional[dict] = Field(
        default=None, exclude_if=lambda v: v is None,
        description="CAMPUS_VERIFY_TENSE_GUARD (default off; results/round2/english_fixes_plan.md): past, future, "
                    "original_verdict, withheld. Set when a SUPPORTED was withheld because the post only says the event "
                    "will happen. Left out of the response when not set.",
    )
    wrong_post_check: Optional[dict] = Field(
        default=None,
        description="Set when CAMPUS_VERIFY_WRONG_POST judged a CONTRADICTED verdict on a message naming a date the post "
                    "used does not mention (results/round2/wrong_post_plan.md): message_dates, post_used, "
                    "better_dated_post, shared_topic_words, reason, original_verdict, and withheld (true when the verdict "
                    "was changed to INSUFFICIENT_EVIDENCE because the post is probably about something else).",
    )
    relevance_check: Optional[dict] = Field(
        default=None,
        description="Set when CAMPUS_VERIFY_RELEVANCE_CHECK judged an English SUPPORTED/CONTRADICTED verdict: "
                    "p_relevant, threshold, original_verdict, and withheld (true when the verdict was changed to "
                    "INSUFFICIENT_EVIDENCE because the closest post may not address the claim).",
    )


class CorpusSourceModel(BaseModel):
    """One configured evidence source (src.api.corpus_config.CorpusSource)."""
    label: str = Field(..., description="'dev_dataset' or the corpus file name.")
    documents: int


class FacebookRefreshModel(BaseModel):
    """Background refresh from the mock Facebook Page (CAMPUS_VERIFY_FB_REFRESH_MINUTES); errors never contain the token."""
    enabled: bool = False
    interval_minutes: float = 0.0
    last_run: Optional[str] = None
    last_added: Optional[int] = None
    last_error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    data_status: str
    classifier: str
    # M11.2: which evidence pool the API is running on (added with defaults,
    # so existing clients that ignore these fields are unaffected).
    corpus_source: str = "dev_dataset"
    corpus_size: int = 0
    corpus_sources: List[CorpusSourceModel] = Field(default_factory=list)
    corpus_loaded_at: Optional[str] = None
    corpus_reload_error: Optional[str] = None
    # M12.3: how inputs are handled (defaults keep older clients working).
    retriever: str = "tfidf"
    router: str = "off"
    claim_rewrite: str = "none"
    date_check: bool = False
    value_check: bool = False
    reranker: str = "off"
    premise_mode: str = "window"
    verdict_languages: str = "all"
    en_question_rewrite: bool = False
    taglish_hybrid: bool = False
    relevance_check: bool = False
    relative_date_check: bool = False
    other_school_check: bool = False
    unicode_fold: bool = False
    image_check: bool = False
    image_signals: bool = False        # the photo check (Phase 2A) is on: image check on and a photo index present
    stats: Optional[bool] = Field(None, exclude_if=lambda v: v is None)   # CAMPUS_VERIFY_STATS on; absent when off
    ui_extras: Optional[bool] = Field(None, exclude_if=lambda v: v is None)   # CAMPUS_VERIFY_UI_EXTRAS on; absent when off
    auth: Optional[dict] = Field(None, exclude_if=lambda v: v is None)        # CAMPUS_VERIFY_AUTH=google; absent when off
    tour: Optional[bool] = Field(None, exclude_if=lambda v: v is None)        # CAMPUS_VERIFY_TOUR on; absent when off
    history: Optional[bool] = Field(None, exclude_if=lambda v: v is None)     # CAMPUS_VERIFY_HISTORY on; absent when off
    sources: Optional[bool] = Field(None, exclude_if=lambda v: v is None)     # CAMPUS_VERIFY_SOURCES on; absent when off
    answer_reuse: Optional[bool] = Field(None, exclude_if=lambda v: v is None)  # CAMPUS_VERIFY_ANSWER_REUSE on; absent when off
    tagalog_nli_model: str = "none"
    facebook_refresh: Optional[FacebookRefreshModel] = None


class OCRModel(BaseModel):
    """What OCR read from the uploaded image. Returned to the user only; never logged or stored."""
    text: str                                    # as read by OCR
    checked_text: str = ""                       # after src.image.normalize (layout noise only) and truncation
    normalization: List[str] = Field(default_factory=list)   # the clean-up rules that changed something
    confidence: Optional[float] = None           # the OCR engine's own score; not evidence that the text is correct
    lines: int = 0
    truncated: bool = False


class SentenceCheckModel(BaseModel):
    """One segment of the checked text (Part B, src/image/sentences.py). Checked segments carry the unchanged
    /verify/text response for that statement; skipped segments carry none."""
    index: int
    text: str
    status: str                                  # "checked" | "uncertain" | "not_checked_short" | "not_checked_limit"
    outcome: Optional[str] = None                # "match" | "contradiction" | "inconclusive" (checked segments only)
    verdict: Optional[str] = None                # the statement's three-way verdict, unchanged (or null)
    agreement: Optional[dict] = None             # the full-stop agreement check, when applied
    text_check: Optional[VerificationResultModel] = None


class SentenceSummaryModel(BaseModel):
    """Counts and the headline for sentence-level results. n = k + c + i; skipped segments are never counted."""
    segments: int = 0
    checked: int = 0                             # n
    matches: int = 0                             # k
    contradictions: int = 0                      # c
    inconclusive: int = 0                        # i
    not_checked_short: int = 0
    not_checked_limit: int = 0
    kind: Optional[str] = None                   # not_confirmed | partly_confirmed | all_match | conflict | possible_conflict
    label: Optional[str] = None
    anchored_posts: List[str] = Field(default_factory=list)
    option_a_kind: Optional[str] = None          # what C2 option A would show (owner's revert check)


class ImageCheckResponse(BaseModel):
    """POST /verify/image (Image Check Phase 1). `text_check` is exactly what /verify/text returns for
    `ocr.checked_text` (the whole-text check). It is null when no readable text was found (status "no_text"), or when the
    agreement check found the verdict fragile (status "uncertain"). Image signals (photo check, provenance facts) are display only.

    Part B (additive): `check_basis` says what decides the displayed result. "sentences": `sentence_summary` (from
    `sentence_checks`) decides; the whole-text result never overrides or upgrades it, and `whole_text_agreement` records any
    disagreement. "whole_text_fallback": no sentence qualified, so the whole-text check is the only check. "none": no
    readable text."""
    ocr: OCRModel
    check_basis: str = "none"
    sentence_checks: List[SentenceCheckModel] = Field(default_factory=list)
    sentence_summary: Optional[SentenceSummaryModel] = None
    whole_text_agreement: Optional[dict] = None
    status: str = "checked"                      # "checked" | "no_text" | "uncertain" | "other_page" (source rule)
    text_check: Optional[VerificationResultModel] = None
    agreement_check: Optional[dict] = None
    note: str
    image_signals: Optional[dict] = Field(None, description="The photo check (Phase 2A): whether the picture matches a photo DYCI has posted, with the measured error. Display only; never changes a verdict. Null when there is no photo index.")
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    source_check: Optional[dict] = Field(None, description="Source rule (CAMPUS_VERIFY_IMAGE_SOURCE_RULE, Florence-2 reader): "
                                         "status other_page | header_read | no_header, page_name, institutions found in "
                                         "the header, note, and for other_page the closest DYCI announcements. Null "
                                         "when the rule is off. It only ever withholds a verdict.")
    provenance: Optional[dict] = Field(None, description="Provenance facts (Phase 2B P1, CAMPUS_VERIFY_IMAGE_PROVENANCE): "
                                       "Content Credentials (C2PA), the file's own AI label and camera details, with a "
                                       "summary. Display only; never changes a verdict or the level. Null when off.")
    changed_date: Optional[dict] = Field(None, description="Rule F (CAMPUS_VERIFY_CHANGED_DATE): the picture closely "
                                         "matches the post the whole-text check used but gives a date up to 7 days off "
                                         "(Level 2): version, picture_date, post_date, post_dated, post, post_published, "
                                         "overlap. "
                                         "Null when off or not found.")
    text_conflict_level: Optional[dict] = Field(None, description="Option B (CAMPUS_VERIFY_TEXT_CONFLICT_LEVEL): the "
                                                "whole text is CONTRADICTED by a post the picture closely matches, so "
                                                "the picture was put from Level 2-4 at Level 1: version, from_level, "
                                                "post, post_dated, post_name, overlap. Null when off or not found.")
    text_line: Optional[dict] = Field(None, description="Option A v2 (CAMPUS_VERIFY_PICTURE_TEXT_LINE): the verdict "
                                      "the picture's words get checked as one piece of text (text_check), shown on "
                                      "the level card: version, verdict, post, post_dated (the post's header date), "
                                      "post_date, post_url, rule_g (rule G released the verdict), line, differs (the "
                                      "line and the level point different ways). Display only. Null when off, or when "
                                      "text_check has no verdict or it was withheld.")
    poster_core: Optional[dict] = Field(None, description="Rule E (CAMPUS_VERIFY_POSTER_CORE): the poster's seal, "
                                        "school-name header, label, issue date and footer set aside before the check: "
                                        "version, removed, poster_date. Null when off or nothing was set aside.")
    poster_dates: Optional[dict] = Field(None, description="Rule D (CAMPUS_VERIFY_POSTER_DATES): letter-spaced dates "
                                         "rejoined before the check, with every change listed. Null when off or unchanged.")
    poster_cleanup: Optional[dict] = Field(None, description="Rule R (CAMPUS_VERIFY_POSTER_CLEANUP): the poster's seal, "
                                           "school name, label, issue date (poster_date) and footer set aside before the "
                                           "check, with every change listed. Null when off or nothing was changed.")
    assessment: Optional[dict] = Field(None, description="Legitimacy level 1-5 with its reasons (results/round2/"
                                       "legitimacy_levels_spec.md): fixed rules over the results above, not a probability; "
                                       "it changes no verdict.")
    data_status: str = "development"


class GoogleSignInRequest(BaseModel):
    """POST /auth/google: the ID token Google Identity Services gave the page (CAMPUS_VERIFY_AUTH=google)."""
    credential: str = Field(..., min_length=20, max_length=8192)


class SourceAddRequest(BaseModel):
    """POST /admin/sources: a Facebook page to collect from. The token is checked with Meta, then stored encrypted."""
    page_id: str = Field(..., min_length=1, max_length=40)
    tier: str = Field(..., min_length=1, max_length=20)
    token: str = Field(..., min_length=1, max_length=2000)


class SourceCutoffRequest(BaseModel):
    """POST /admin/sources/cutoff: the final-evaluation cutoff day, YYYY-MM-DD (recorded once)."""
    day: str = Field(..., max_length=10)


class SourceTokenRequest(BaseModel):
    """POST /admin/sources/{key}/token: a replacement token for that page."""
    token: str = Field(..., min_length=1, max_length=2000)


class SourceUpdateRequest(BaseModel):
    """POST /admin/sources/{key}: turn fetching on or off, or change the tier."""
    include: Optional[bool] = None
    tier: Optional[str] = Field(None, max_length=20)


class HistoryPreferenceRequest(BaseModel):
    """POST /account/history: the signed-in student's choice to keep a history of their checks."""
    history_enabled: bool
    delete_saved: bool = False          # with history_enabled false: also erase the saved checks


class AccountActiveRequest(BaseModel):
    """POST /admin/users/{id}/active (administrators only): enable or disable a DYCI account."""
    active: bool


class DeleteAccountRequest(BaseModel):
    """POST /account/delete: the typed confirmation and a fresh Google ID token from the same DYCI account."""
    confirm: str = Field(..., max_length=20)
    credential: str = Field(..., min_length=20, max_length=8192)
