"""
Fix A: read the picture in each fetched post (results/round2/post_images_plan.md). CAMPUS_VERIFY_POST_IMAGES=1 (default
off). Used by the live test-page fetch (src/ingestion/facebook_page.py) through MetaGraphAPIAdapter.post_text.

A post's text becomes "<caption>\n\nText in the post's picture: <reading>". The picture is read once per post ID and the
reading is cached next to the served file (<served file>.image_text.json), so later fetches, and the edit/deletion sync,
see exactly the text the post was stored with. A download or read failure leaves the post as it would be without the fix.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional
from urllib.parse import urlsplit

ENV = "CAMPUS_VERIFY_POST_IMAGES"
VERSION = "post-images-v1"
MARK = "Text in the post's picture:"
MAX_BYTES = 10 * 1024 * 1024
TIMEOUT_SECONDS = 20
ALLOWED_HOST_SUFFIXES = ("fbcdn.net", "facebook.com")

# The text reader: main.py registers the picture check's engine so one EasyOCR reader is shared; otherwise one is made.
ENGINE_PROVIDER: Optional[Callable[[], object]] = None
_ENGINE = None


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _engine():
    global _ENGINE
    if ENGINE_PROVIDER is not None:
        return ENGINE_PROVIDER()
    if _ENGINE is None:
        from src.image.ocr import OCREngine
        _ENGINE = OCREngine()
    return _ENGINE


def _letters(text: str) -> int:
    return len(re.findall(r"[^\W\d_]", text or ""))


def allowed_url(url: str) -> bool:
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    return parts.scheme == "https" and any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)


def download(url: str) -> tuple:
    """(bytes, content type) of a post's picture; raises ValueError for a refused or failed download."""
    import requests
    if not allowed_url(url):
        raise ValueError("picture address refused (https on fbcdn.net or facebook.com only)")
    with requests.get(url, timeout=TIMEOUT_SECONDS, stream=True, allow_redirects=False) as r:
        if r.status_code != 200:
            raise ValueError(f"picture download failed (HTTP {r.status_code})")
        data = b""
        for chunk in r.iter_content(64 * 1024):
            data += chunk
            if len(data) > MAX_BYTES:
                raise ValueError("picture larger than 10 MB")
        return data, r.headers.get("content-type", "")


def compose(caption: Optional[str], reading: Optional[str]) -> Optional[str]:
    caption = (caption or "").strip()
    reading = (reading or "").strip()
    if _letters(reading) < 3:
        reading = ""
    if caption and reading:
        return f"{caption}\n\n{MARK} {reading}"
    return caption or (f"{MARK} {reading}" if reading else None)


class PostImageReader:
    """post -> stored text, for MetaGraphAPIAdapter.post_text. Readings are cached by post ID in `cache_path`."""

    def __init__(self, cache_path: str, downloader: Callable[[str], tuple] = download,
                 engine: Optional[Callable[[], object]] = None):
        self.cache_path, self.downloader, self.engine = cache_path, downloader, engine or _engine
        self.problems: list = []          # [{"post_id", "reason"}] for this fetch (fix B's log shows them)
        self.read_now: list = []          # post IDs read during this fetch
        try:
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            self.cache = {}

    def _save(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.cache_path)

    def reading(self, post: dict) -> Optional[str]:
        post_id, url = str(post.get("id") or ""), post.get("full_picture")
        if not post_id or not url:
            return None
        if post_id in self.cache:
            return self.cache[post_id].get("text")
        try:
            from src.image.intake import load
            from src.image.normalize import normalize_ocr_text
            data, content_type = self.downloader(url)
            image = load(data, content_type)
            text = normalize_ocr_text(self.engine().read(image).text).text
        except Exception as e:             # never stops the fetch: the post stays as it would be without the fix
            self.problems.append({"post_id": post_id, "reason": f"picture not read: {type(e).__name__}: {str(e)[:160]}"})
            return None
        self.cache[post_id] = {"text": text, "version": VERSION}
        self.read_now.append(post_id)
        self._save()
        return text

    def __call__(self, post: dict) -> Optional[str]:
        return compose(post.get("message") if isinstance(post.get("message"), str) else None, self.reading(post))
