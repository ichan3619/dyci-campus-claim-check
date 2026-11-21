"""
Source adapters (Milestone 8).

Responsibility of this module, and ONLY this module: fetch RAW content
from a specific kind of source, and hand it to
src.ingestion.evidence.normalize() to become a NormalizedEvidence.

============================================================================
WHAT "SOURCE ADAPTER" MEANS HERE, AND WHY THERE ARE THREE OF THEM
============================================================================
A source adapter's only job is answering "how do I get raw content OUT
of this particular kind of source" -- a local file, a live website, a
Facebook page. It does NOT normalize, validate, or clean anything itself
(that's evidence.normalize()'s job) -- keeping "how do I fetch" and "how
do I standardize what I fetched" as two separate concerns is exactly
what lets a future WebsiteSourceAdapter or MetaGraphAPIAdapter slot
in later without touching evidence.py or retrieval at all.

Four adapters exist in this file, in very different states of
completeness -- this is deliberate, not an oversight:

    LocalFileSourceAdapter   -- FULLY IMPLEMENTED AND TESTED. Reads
        content from a local file. This is the "controlled/local
        fixture" adapter -- it proves the adapter -> normalize ->
        Retriever pipeline works end-to-end without depending on
        network access.

    WebsiteSourceAdapter     -- IMPLEMENTED as of Milestone 8.2. Fetches
        a live webpage over HTTP (`requests`) and extracts its content
        (`src/ingestion/html_extraction.py`, using BeautifulSoup).
        IMPORTANT: "implemented" here means the CODE PATH works and is
        tested against controlled local HTML fixtures -- it does NOT
        mean any real official campus website has actually been
        ingested through it yet. That validation is Milestone 8.4/8.6's
        job, not this one's. See docs/methodology.md's Milestone 8.2
        section for the exact scope and limitations.

    MetaGraphAPIAdapter      -- IMPLEMENTED as of Milestone 8.5, in the
        SAME "code path works, validated only against controlled
        fixtures" sense as WebsiteSourceAdapter above -- NOT the same as
        real DYCI Facebook Page validation. It intentionally does NOT
        implement SourceAdapter's single-document fetch() (see its own
        docstring for why -- a Facebook Page's /posts endpoint naturally
        returns many documents per call, not one); it exposes
        fetch_posts() -> List[NormalizedEvidence] instead. This project
        has no Meta Graph API credentials and no authorized access to
        the real DYCI Facebook Page -- every test of this adapter uses a
        mocked HTTP response built from Meta's documented response
        shape, never a live call. Real DYCI Facebook ingestion is
        Milestone 8.5-B / 8.6's job, not this one's. Direct Facebook
        scraping remains disallowed under any circumstance, regardless
        of implementation status.

IMPORTANT: implementing and testing an adapter against CONTROLLED
fixtures is not the same as real-world source validation. See the
Milestone 8.2 report for the explicit "what's actually working"
statement.
"""

import os
import re
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional, Protocol, Iterable

from src.ingestion.evidence import NormalizedEvidence, normalize
from src.ingestion.html_extraction import extract_content

# Kept conservative and explicit, per Step 2's "reasonable timeout"
# requirement -- prevents a slow/unresponsive site from hanging the
# ingestion process indefinitely. A single constant (not per-call
# configuration) because this adapter fetches individual pages, not a
# crawler with varying per-request needs.
DEFAULT_FETCH_TIMEOUT_SECONDS = 10

# Descriptive, per Step 2 -- identifies this traffic honestly rather
# than pretending to be a browser.
USER_AGENT = "CampusVerifyThesisBot/1.0 (educational research project; source ingestion)"


class WebsiteFetchError(Exception):
    """
    Raised when a live webpage fetch fails -- network error, timeout, or
    a non-success HTTP status. A distinct exception type (same pattern
    as CorpusStorageError in corpus_storage.py) so a caller can
    specifically catch "the live fetch failed" without conflating it
    with normalize()'s ValueError (bad/empty content) or
    FileNotFoundError (LocalFileSourceAdapter's failure mode) -- these
    are three different failure categories from three different
    adapters, kept distinguishable on purpose.
    """
    pass


class SourceAdapter(Protocol):
    """
    The interface every source adapter must implement: produce ONE
    NormalizedEvidence from whatever this adapter's source is.

    Structural typing (Protocol), same pattern already used for
    src.evaluation.evaluate.ClaimEvidenceModel -- a future adapter
    doesn't need to inherit from anything, it just needs a matching
    fetch() method.
    """
    def fetch(self) -> NormalizedEvidence:
        ...


@dataclass
class LocalFileSourceAdapter:
    """
    Reads evidence content from a local text file.

    This represents content that has ALREADY been retrieved and saved
    locally (e.g. a researcher manually saved a copy of an announcement
    page's text) -- it performs no network access itself. This is the
    ONE fully working, fully tested adapter in Milestone 8.

    Usage:
        adapter = LocalFileSourceAdapter(
            file_path="tests/fixtures/sample_announcement.txt",
            source_type="official_site",
            source_url="https://university.example.edu/announcements/123",
        )
        evidence = adapter.fetch()
    """
    file_path: str
    source_type: str
    source_url: str = ""
    published_date: str = ""
    credibility_tier: str = ""
    source_identifier: str = ""

    def fetch(self) -> NormalizedEvidence:
        """
        Read the file and normalize its content.

        Raises FileNotFoundError if the path doesn't exist (a real,
        useful error -- not swallowed or converted into empty evidence),
        and ValueError (via normalize()) if the file's content is empty
        or the configured source_type/credibility_tier is invalid.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        fetched_at = datetime.now(timezone.utc).isoformat()

        return normalize(
            text=raw_text,
            source_type=self.source_type,
            source_url=self.source_url,
            published_date=self.published_date,
            fetched_at=fetched_at,
            credibility_tier=self.credibility_tier,
            source_identifier=self.source_identifier or self.file_path,
        )


@dataclass
class WebsiteSourceAdapter:
    """
    Fetches a live webpage over HTTP and converts it into a
    NormalizedEvidence.

    Usage:
        adapter = WebsiteSourceAdapter(
            url="https://university.example.edu/announcements/123",
            source_type="official_site",
        )
        evidence = adapter.fetch()

    Implementation notes:
        - Fetching uses `requests` (a GET request, explicit timeout,
          descriptive User-Agent -- see module-level constants).
        - Extraction uses src.ingestion.html_extraction.extract_content()
          -- a separate, pure, network-free module, tested independently
          of this class.
        - The extracted title and body are combined into ONE text string
          before normalization: NormalizedEvidence has no separate
          `title` field (adding one would mean changing the existing
          evidence representation, which this milestone deliberately
          avoids -- see docs/methodology.md's Milestone 8.2 section for
          the reasoning). Folding title into text is a simple, explicit
          design choice, not an oversight.
        - document_id is derived entirely by the EXISTING normalize()
          logic from source_url -- no second identity scheme is
          introduced here, so the same URL fetched twice produces the
          same document_id, integrating naturally with M8.3's
          duplicate-skip behavior in append_to_corpus().

    IMPORTANT SCOPE LIMITATION: this class is tested against controlled
    local HTML fixtures (tests/test_website_adapter.py), NOT against any
    real live website. It has the technical ability to fetch and parse
    arbitrary HTML, but reliable extraction is not guaranteed for
    arbitrary real-world page structures -- see the module-level
    docstring of html_extraction.py and docs/methodology.md.
    """
    url: str
    source_type: str = "official_site"
    credibility_tier: str = ""
    timeout_seconds: int = DEFAULT_FETCH_TIMEOUT_SECONDS
    # False for user-submitted URLs (POST /verify/url): the SSRF guard has already walked and checked the redirect
    # chain, so this fetch must read exactly the checked URL. A redirect at this point is refused (WebsiteFetchError),
    # never followed: a server could answer the check with 200 and this fetch with a redirect to an internal address.
    follow_redirects: bool = True

    def fetch(self) -> NormalizedEvidence:
        """
        Fetch self.url, extract its content, and normalize it.

        Raises:
            WebsiteFetchError: network failure, timeout, or non-success
                HTTP status. Failures are NEVER silently converted into
                empty evidence -- per Step 2's explicit requirement, a
                failed fetch is a loud error, not a quiet empty result.
            ValueError: (via normalize()) if the extracted content is
                empty/whitespace-only, or if source_type/credibility_tier
                is invalid -- the SAME validation every other adapter's
                output goes through, not a separate check reimplemented
                here.
        """
        html = self._fetch_html()
        extracted = extract_content(html)

        if extracted.title:
            combined_text = f"{extracted.title}\n\n{extracted.body_text}"
        else:
            combined_text = extracted.body_text

        fetched_at = datetime.now(timezone.utc).isoformat()

        return normalize(
            text=combined_text,
            source_type=self.source_type,
            source_url=self.url,
            published_date=extracted.published_date,
            fetched_at=fetched_at,
            credibility_tier=self.credibility_tier,
        )

    def _fetch_html(self) -> str:
        """
        Perform the actual HTTP GET, translating every failure mode into
        a single, clearly-named WebsiteFetchError rather than letting
        assorted requests exceptions leak out uncategorized, or -- the
        one thing this must never do -- swallowing a failure into an
        empty string that would later masquerade as "the page had no
        content."
        """
        try:
            response = requests.get(
                self.url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": USER_AGENT},
                stream=True,          # read in chunks so an oversized page can be stopped (MAX_PAGE_BYTES)
                allow_redirects=self.follow_redirects,
            )
        except requests.exceptions.Timeout as e:
            raise WebsiteFetchError(f"timed out fetching {self.url} (limit={self.timeout_seconds}s): {e}")
        except requests.exceptions.ConnectionError as e:
            raise WebsiteFetchError(f"connection error fetching {self.url}: {e}")
        except requests.exceptions.RequestException as e:
            raise WebsiteFetchError(f"request failed for {self.url}: {e}")

        if not self.follow_redirects and 300 <= response.status_code < 400:
            response.close()
            raise WebsiteFetchError(
                f"the page redirected (HTTP {response.status_code}) after the safety check; the redirect was not "
                f"followed: {self.url}"
            )
        if response.status_code != 200:
            raise WebsiteFetchError(
                f"unexpected HTTP status {response.status_code} fetching {self.url}"
            )
        self._read_capped(response)

        # --------------------------------------------------------------
        # ENCODING FIX (discovered against the real DYCI site during
        # Milestone 8.4's live validation -- confirmed root cause, not
        # a guess): `requests` determines response.encoding from the
        # Content-Type header via get_encoding_from_headers(), which
        # defaults ANY "text/*" content type LACKING AN EXPLICIT
        # charset PARAMETER to ISO-8859-1 (a longstanding requests
        # behavior, inherited from an old RFC 2616 default). Most real
        # servers -- including DYCI's -- send `Content-Type: text/html`
        # with no charset parameter at all, relying on the page's own
        # `<meta charset="utf-8">` tag instead. requests never looks at
        # that tag, so `response.text` alone silently mis-decodes real
        # UTF-8 bytes (e.g. a curly apostrophe) as Latin-1, producing
        # mojibake like "Yangaâ€™s" instead of "Yanga's" -- confirmed
        # against the real page's actual response during Milestone 8.4.
        #
        # response.apparent_encoding sniffs the ACTUAL byte content
        # (via charset-normalizer/chardet, a real dependency of
        # `requests` itself -- no new dependency added here) rather
        # than trusting a header that may be absent or wrong. Setting
        # it before reading .text is the standard, correct fix -- not a
        # downstream string-replacement patch on already-corrupted text.
        response.encoding = response.apparent_encoding
        return response.text

    def _read_capped(self, response) -> None:
        """
        Read the body in chunks, refusing anything over MAX_PAGE_BYTES (a declared Content-Length over the limit is
        refused before reading). Without this, a URL pointing at a huge file would be read into memory whole -- the
        machine running the models has little to spare. The bytes are stored on the response, so .apparent_encoding
        and .text below work exactly as they did before.
        """
        declared = (response.headers or {}).get("Content-Length")
        if isinstance(declared, str) and declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
            response.close()
            raise WebsiteFetchError(f"page too large ({int(declared)} bytes, limit {MAX_PAGE_BYTES}) at {self.url}")
        body = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > MAX_PAGE_BYTES:
                    response.close()
                    raise WebsiteFetchError(f"page larger than the {MAX_PAGE_BYTES}-byte limit at {self.url}")
        except requests.exceptions.RequestException as e:
            raise WebsiteFetchError(f"reading {self.url} failed: {e}")
        response._content = bytes(body)     # requests' own cache of the body; .text/.apparent_encoding decode it


# Largest page /verify/url will read (bytes). Campus announcement pages are far smaller.
MAX_PAGE_BYTES = int(os.environ.get("CAMPUS_VERIFY_MAX_PAGE_BYTES", str(2 * 1024 * 1024)))


def _scrub_token(text: str, token: str) -> str:
    """
    Milestone 8.5-A fix: redacts an access token from arbitrary text
    BEFORE it is ever placed into a GraphAPIFetchError message (and
    therefore before it could end up in a log, a caller's exception
    handler, or test/console output). Two independent passes, since
    either alone can miss a real case:

      1. Exact-token replacement -- catches the token appearing verbatim
         anywhere in the text (the common case: `requests`/`urllib3`
         exception messages frequently embed the full failed request URL,
         including its query string, so a ConnectionError or Timeout's
         own str() can literally contain `...&access_token=<the token>`).
      2. A regex over the `access_token=...` query-parameter shape --
         catches the case where the token was partially re-encoded or
         truncated by an underlying HTTP library, so an exact string
         match might not fire, but the parameter shape still would.

    This function does not change what is raised (still GraphAPIFetchError
    with the same diagnostic content otherwise -- page_id, failure kind,
    HTTP details) or when it is raised -- only removes the token from the
    text of the message itself.
    """
    if not text:
        return text
    if token:
        text = text.replace(token, "[REDACTED]")
    return re.sub(r"(access_token=)[^&\s'\"]+", r"\1[REDACTED]", text)


class GraphAPIFetchError(Exception):
    """
    Raised when a Meta Graph API request fails -- network error, timeout,
    a non-success HTTP status, or an API-level error object in the
    response body. Same pattern as WebsiteFetchError above: a distinct
    exception type so a caller can specifically catch "the Graph API
    call failed" without conflating it with normalize()'s ValueError
    (a single post's content being rejected -- see fetch_posts()'s
    per-post skip handling, which is deliberately NOT this exception)
    or a JSON-decoding failure.
    """
    pass


# Graph API version pinned to a recent stable release at the time this
# adapter was written. Meta version-deprecates old releases on a rolling
# schedule -- per project policy (see docs/methodology.md's Milestone
# 8.5 section), this MUST be re-verified against Meta's current official
# documentation before any real (Milestone 8.5-B) Page is ever connected,
# not assumed to still be correct.
DEFAULT_GRAPH_API_VERSION = "v21.0"

# Read-only field set: an id (for document_id/duplicate detection), the
# post's own text, its publication time, and a permalink for
# provenance/source_url. Deliberately excludes anything relating to
# engagement metrics, insights, or any write/manage capability -- this
# adapter's whole job is ingesting Page post TEXT as evidence, not
# analytics and not Page management (see the module-level scope note
# and docs/methodology.md's Milestone 8.5 section on minimum-permission
# reasoning).
GRAPH_API_POST_FIELDS = "id,message,created_time,permalink_url"


@dataclass
class MetaGraphAPIAdapter:
    """
    Fetches a Facebook Page's recent posts via the official Meta Graph
    API and converts each into a NormalizedEvidence.

    Usage:
        adapter = MetaGraphAPIAdapter(
            page_id="123456789",
            access_token="<a real Page access token>",
        )
        evidence_list = adapter.fetch_posts()

    ============================================================================
    WHY fetch_posts() -> List[...], NOT fetch() -> NormalizedEvidence
    ============================================================================
    SourceAdapter's fetch() contract (one document per call) fits a local
    file or a single webpage. It does NOT fit a Facebook Page: the Graph
    API's /{page-id}/posts endpoint naturally returns MANY posts in one
    response. Rather than distort that existing single-document contract
    (which LocalFileSourceAdapter and WebsiteSourceAdapter both rely on,
    unmodified, elsewhere in this codebase) or generalize SourceAdapter
    itself to always return a list (which would touch two already-working,
    already-tested adapters just to accommodate a third), this adapter
    intentionally does NOT implement SourceAdapter's fetch() at all. It
    exposes its own fetch_posts() -> List[NormalizedEvidence] instead.
    This is a deliberate, discussed architectural decision for Milestone
    8.5, not an oversight or a protocol violation.

    ============================================================================
    SCOPE: READ-ONLY, ONE PAGE OF RESULTS, MOCK-TESTED ONLY
    ============================================================================
    - Read-only: only ever performs a GET against /{page-id}/posts. No
      publish/edit/delete/moderate capability exists here or is planned
      here -- this thesis's Facebook integration is read-only source
      ingestion (see docs/methodology.md's Milestone 8.5 section).
    - ONE page of results per fetch_posts() call: `paging.next` (if
      present in the response) is deliberately NOT followed in this
      Milestone 8.5 implementation. This is an explicit, documented
      scope limitation, not partial/broken pagination support -- see
      fetch_posts()'s own docstring below.
    - Mock-tested only: this project has no Meta Graph API credentials
      and no authorized access to the real DYCI Facebook Page. Every
      test of this class mocks `requests.get` against a fixture built
      from Meta's documented response shape (see
      tests/fixtures/facebook_graph_api_posts_response.json and
      tests/test_facebook_adapter.py) -- there is no live call anywhere
      in this class's test coverage, and none of that fixture content is
      real DYCI data. Real-Page validation is Milestone 8.5-B / 8.6.
    """
    page_id: str
    access_token: str
    api_version: str = DEFAULT_GRAPH_API_VERSION
    timeout_seconds: int = DEFAULT_FETCH_TIMEOUT_SECONDS
    source_type: str = "facebook"
    credibility_tier: str = ""
    # M8.5-C: copied onto every record this adapter produces (see
    # NormalizedEvidence.acquisition_note). Lets a caller state, on the
    # record itself, that e.g. the Page is a MOCK/TEST mirror.
    acquisition_note: str = ""
    # Fix A (CAMPUS_VERIFY_POST_IMAGES; results/round2/post_images_plan.md): when set, `full_picture` is also requested
    # and every post's text (new AND known, so the edit/deletion sync compares like with like) is post_text(post): the
    # caption plus the text read from the post's picture, or None when the post has no usable text. None = unchanged.
    post_text: Optional[Callable[[dict], Optional[str]]] = None
    # Rule I (CAMPUS_VERIFY_AUTO_PHOTO_INDEX; results/round2/photo_auto_index_plan.md): when set, `full_picture` is also
    # requested and on_post(post) is called for every fetched post (it stores the picture); the post's text is unchanged.
    on_post: Optional[Callable[[dict], None]] = None

    def __post_init__(self):
        # Populated fresh by every fetch_posts() call (see there): the
        # deterministic, testable record of which posts in the LAST
        # fetch_posts() call were skipped and why. This is the "make the
        # skip observable, without a logging framework" mechanism --
        # plain data on the adapter instance, inspectable by a caller or
        # a test, rather than a print()/log statement that could be
        # missed, or a silently swallowed skip that could not.
        self.skipped_posts: List[dict] = []
        self.has_more_pages: bool = False
        # M8.5-C: Graph post IDs seen in the last fetch that the caller
        # already knew (see fetch_posts known_post_ids), and whether the
        # last fetch stopped early at a page of only known posts.
        self.known_skipped: List[str] = []
        self.stopped_at_known_page: bool = False
        # Edits/deletions (src/ingestion/refresh.py sync): the CURRENT message of every already-known post seen in the
        # last fetch (None when it has no text now), and the created_time of every post the fetch returned, which
        # tells the caller which date range the fetch covered.
        self.known_current: dict = {}
        self.fetched_created_times: List[str] = []
        self.fetched_ids: List[str] = []

    def _get_json(self, url: str, params: Optional[dict]) -> dict:
        """
        One GET + the shared error handling (network, non-JSON, API-level
        error object, non-200). `params` is None when following a
        `paging.next` URL, which already embeds every query parameter.
        Every error message is scrubbed of the access token.
        """
        try:
            if params is None:
                response = requests.get(url, timeout=self.timeout_seconds)
            else:
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
        except requests.exceptions.Timeout as e:
            raise GraphAPIFetchError(_scrub_token(
                f"timed out fetching posts for page {self.page_id} (limit={self.timeout_seconds}s): {e}",
                self.access_token,
            ))
        except requests.exceptions.ConnectionError as e:
            raise GraphAPIFetchError(_scrub_token(
                f"connection error fetching posts for page {self.page_id}: {e}", self.access_token
            ))
        except requests.exceptions.RequestException as e:
            raise GraphAPIFetchError(_scrub_token(
                f"request failed fetching posts for page {self.page_id}: {e}", self.access_token
            ))

        try:
            payload = response.json()
        except ValueError as e:
            raise GraphAPIFetchError(
                f"Graph API response for page {self.page_id} was not valid JSON: {e}"
            )

        if isinstance(payload, dict) and "error" in payload:
            error_info = payload.get("error") or {}
            raise GraphAPIFetchError(
                f"Graph API error for page {self.page_id} "
                f"(code={error_info.get('code', '?')}): "
                f"{error_info.get('message', '(no message provided)')}"
            )

        if response.status_code != 200:
            raise GraphAPIFetchError(
                f"unexpected HTTP status {response.status_code} fetching posts for page {self.page_id}"
            )
        return payload if isinstance(payload, dict) else {}

    def fetch_posts(
        self, limit: int = 25, *, max_pages: int = 1, since: Optional[str] = None,
        known_post_ids: Optional[Iterable[str]] = None, stop_at_known_page: bool = False,
        until: Optional[str] = None,
    ) -> List[NormalizedEvidence]:
        """
        Fetch up to `max_pages` pages (each up to `limit` posts) from this
        Page's /posts endpoint and normalize each usable post into a
        NormalizedEvidence.

        Returns only the successfully normalized posts. A post that
        cannot be turned into evidence (missing/empty `message`, or
        rejected by normalize() for any other reason) is SKIPPED, not
        included in the returned list, and NEVER crashes the rest of the
        batch -- but the skip is not silent: it is recorded in
        self.skipped_posts as {"post_id": ..., "reason": ...}, reset at
        the start of every fetch_posts() call.

        PAGINATION (M11 extension): the default max_pages=1 keeps the
        original M8.5 behavior exactly: ONE page, `paging.next` not
        followed. Passing max_pages > 1 follows `paging.next` up to that
        many pages. After the call, self.has_more_pages is True if the
        API still offered a next page that was NOT fetched (so a caller
        can say so instead of silently missing posts).

        `since`: optional value passed straight to the Graph API's
        `since` parameter (a Unix timestamp or a date string) to fetch
        only posts newer than that. `until` (2026-11-16, past-posts fetching): the same for the Graph API's
        `until` parameter, only posts up to that time.

        `known_post_ids` (M8.5-C): Graph post IDs the caller already has
        (for the corpus, each record's source_identifier). A fetched post
        whose ID is in this set is NOT normalized: it is recorded in
        self.known_skipped and left out of the result. The Graph API has
        no "exclude these IDs" filter, so the post still travels over the
        network; what is avoided is re-processing it.

        `stop_at_known_page` (M8.5-C, OPT-IN, default False): stop
        following `paging.next` once a fetched page consists ENTIRELY of
        known posts, and set self.stopped_at_known_page. Pages arrive
        newest-first, so this avoids paging through the whole history.
        It is off by default because it can MISS a new post whose
        created_time is older than an already-known page (e.g. a
        backdated post). Correctness first: the default reads up to
        `max_pages` pages.

        Raises:
            GraphAPIFetchError: network failure, timeout, non-200 HTTP
                status, an API-level {"error": {...}} object in the
                response body, or a response body that isn't valid JSON
                at all. Same "loud, never silently empty" principle as
                WebsiteSourceAdapter's WebsiteFetchError.
        """
        self.skipped_posts = []
        self.has_more_pages = False
        self.known_skipped = []
        self.stopped_at_known_page = False
        self.known_current = {}
        self.fetched_created_times = []
        self.fetched_ids = []
        known = set(known_post_ids or ())

        url = f"https://graph.facebook.com/{self.api_version}/{self.page_id}/posts"
        params = {
            "fields": GRAPH_API_POST_FIELDS + (",full_picture" if self.post_text is not None or self.on_post is not None
                                               else ""),
            "limit": limit,
            "access_token": self.access_token,
        }
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        posts: list = []
        next_url: Optional[str] = None
        for page_number in range(max(1, max_pages)):
            if page_number == 0:
                payload = self._get_json(url, params)
            else:
                payload = self._get_json(next_url, None)
            page_posts = payload.get("data", [])
            posts.extend(page_posts)
            paging = payload.get("paging", {})
            next_url = paging.get("next") if isinstance(paging, dict) else None
            if (stop_at_known_page and known and page_posts and next_url
                    and all(isinstance(p, dict) and p.get("id") in known for p in page_posts)):
                self.stopped_at_known_page = True
                break
            if not next_url:
                break
        self.has_more_pages = bool(next_url)

        fetched_at = datetime.now(timezone.utc).isoformat()

        evidence_list: List[NormalizedEvidence] = []
        for post in posts:
            post_id = post.get("id", "") if isinstance(post, dict) else ""
            message = post.get("message") if isinstance(post, dict) else None
            if self.on_post is not None and isinstance(post, dict):
                self.on_post(post)                      # rule I: store the post's picture (text unchanged)
            if self.post_text is not None and isinstance(post, dict):
                message = self.post_text(post)          # fix A: caption + picture text (None when neither has text)

            if post_id:
                self.fetched_ids.append(post_id)
                if post.get("created_time"):
                    self.fetched_created_times.append(post["created_time"])
            if post_id and post_id in known:
                self.known_skipped.append(post_id)
                self.known_current[post_id] = {
                    "message": message if isinstance(message, str) and message.strip() else None,
                    "created_time": post.get("created_time", "") or "",
                }
                continue

            if not isinstance(message, str) or not message.strip():
                self.skipped_posts.append({
                    "post_id": post_id or "(missing id)",
                    "reason": "missing or empty 'message' field",
                })
                continue

            try:
                evidence = normalize(
                    text=message,
                    source_type=self.source_type,
                    source_url=post.get("permalink_url", "") or "",
                    published_date=post.get("created_time", "") or "",
                    fetched_at=fetched_at,
                    credibility_tier=self.credibility_tier,
                    source_identifier=post_id,
                    acquisition_note=self.acquisition_note,
                )
            except ValueError as e:
                self.skipped_posts.append({
                    "post_id": post_id or "(missing id)",
                    "reason": f"rejected by normalize(): {e}",
                })
                continue

            evidence_list.append(evidence)

        return evidence_list
