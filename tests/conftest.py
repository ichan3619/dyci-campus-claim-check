"""
Test-wide configuration. The API's defaults load e5, two NLI models and the dev + Facebook corpus; the test suite pins
the fast, offline configuration instead (TF-IDF retrieval, the TF-IDF+LR classifier fitted on the dev CSV, dev corpus,
no Tagalog model), whatever the developer's shell has set. Set here, before any test module imports src.api.main,
because the API reads its settings at import time. Tests that need another setting use the `api_settings` fixture.
"""
import os
from dataclasses import replace

import pytest

os.environ.update({
    "CAMPUS_VERIFY_CLASSIFIER": "tfidf_lr",
    "CAMPUS_VERIFY_RETRIEVER": "tfidf",
    "CAMPUS_VERIFY_CORPUS": "dev",
    "CAMPUS_VERIFY_TAGALOG_NLI_MODEL": "none",
    "CAMPUS_VERIFY_FB_REFRESH_MINUTES": "0",
    # The suite sends hundreds of requests from one test client; rate limits are tested on their own
    # (tests/test_api_protection.py) through the `api_protection` fixture.
    "CAMPUS_VERIFY_RATE_LIMIT_PER_MINUTE": "0",
    "CAMPUS_VERIFY_URL_RATE_LIMIT_PER_MINUTE": "0",
    # Existing API tests check verdicts on Tagalog/Taglish claims; the language policy has its own tests.
    "CAMPUS_VERIFY_VERDICT_LANGUAGES": "all",
    # The photo check (data/photo_index) is tested on its own (tests/test_photo_check_api.py) with a synthetic index.
    "CAMPUS_VERIFY_PHOTO_INDEX": "none",
    # Name alignment and the post-age rule (on by default since v1.4) are tested on their own
    # (tests/test_name_align.py, tests/test_post_age.py); the rest of the suite keeps the configuration it was written for.
    "CAMPUS_VERIFY_NAME_ALIGN": "0",
    "CAMPUS_VERIFY_POST_AGE": "0",
    # Question focus and the local Tagalog translator (on by default since v2.3) are tested on their own
    # (tests/test_question_focus.py, tests/test_tagalog_translate.py): the suite never calls a running llama.cpp server.
    "CAMPUS_VERIFY_QUESTION_FOCUS": "0",
    "CAMPUS_VERIFY_TAGALOG_TRANSLATE": "0",
})
os.environ.pop("CAMPUS_VERIFY_API_KEY", None)
os.environ.pop("CAMPUS_VERIFY_PUBLIC", None)
os.environ.pop("CAMPUS_VERIFY_POST_AGE_V2", None)
os.environ.pop("CAMPUS_VERIFY_CLAIM_REWRITE", None)
os.environ.pop("CAMPUS_VERIFY_ROUTER", None)
os.environ.pop("CAMPUS_VERIFY_DATE_CHECK", None)


@pytest.fixture()
def api_settings(monkeypatch):
    """api_settings(router=False, claim_rewrite='none', ...) -> the src.api.main module with those settings in place."""
    from src.api import main as api_main

    def use(**changes):
        monkeypatch.setattr(api_main, "SETTINGS", replace(api_main.SETTINGS, **changes))
        return api_main

    return use


@pytest.fixture()
def api_protection(monkeypatch):
    """api_protection(rate_per_minute=3, ...) -> the src.api.main module with those protection settings in place."""
    from src.api import main as api_main

    def use(**changes):
        monkeypatch.setattr(api_main, "PROTECTION", replace(api_main.PROTECTION, **changes))
        return api_main

    return use
