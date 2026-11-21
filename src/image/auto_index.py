"""
Rule I: DYCI's own pictures as the reference (results/round2/photo_auto_index_plan.md, Amendment 1).
CAMPUS_VERIFY_AUTO_PHOTO_INDEX=1 (default off).

- Collector (I-a): the live fetch hands every post to PictureCollector; a post's picture is downloaded once (fix A's
  guarded download), read once by the text reader, and stored in <served file>.photos/ with index.json (post ID, link,
  date, page, caption, reading). Post text and the evidence are not changed.
- Index (I-b): load() gives these pictures as photo-index entries, with the hand-made index's frozen thresholds.
- Edited copies (I-c): edited() compares the values read from an upload with the values read from the DYCI picture it
  matched. A kind is flagged when each reading has a value of that kind the other lacks (a swap).
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Callable, Optional

ENV = "CAMPUS_VERIFY_AUTO_PHOTO_INDEX"
VERSION = "auto-photo-index-v1"
KINDS = ("date", "time", "percent", "duration")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def page_label(url: str) -> str:
    """Who posted it, as the hand-made index words it (scripts/round2/photo_index_build.py): only DYCI's main page is
    "DYCI"; the test page and any other page are "a Facebook page linked to DYCI"."""
    m = re.search(r"facebook\.com/([^/?]+)", url or "")
    return "DYCI" if m and m.group(1) == "YangaCollege" else "a Facebook page linked to DYCI"


def folder_for(served_file: str) -> str:
    return served_file + ".photos"


def _safe(post_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", post_id)[:80]


class PictureCollector:
    """MetaGraphAPIAdapter.on_post: store each post's picture and its reading once (see the module doc)."""

    def __init__(self, folder: str, downloader: Optional[Callable[[str], tuple]] = None,
                 engine: Optional[Callable[[], object]] = None):
        from src.ingestion import post_images
        self.folder = folder
        self.downloader = downloader or post_images.download
        self.engine = engine or post_images._engine
        self.problems: list = []
        self.saved: list = []
        os.makedirs(folder, exist_ok=True)
        try:
            with open(os.path.join(folder, "index.json"), encoding="utf-8") as f:
                self.record = json.load(f)
        except (OSError, ValueError):
            self.record = {"version": VERSION, "photos": []}
        self.known = {p["post_id"] for p in self.record["photos"]}

    def _save(self) -> None:
        path = os.path.join(self.folder, "index.json")
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.record, f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)

    def __call__(self, post: dict) -> None:
        post_id, url = str(post.get("id") or ""), post.get("full_picture")
        if not post_id or not url or post_id in self.known:
            return
        try:
            from src.image.intake import load
            from src.image.normalize import normalize_ocr_text
            data, content_type = self.downloader(url)
            image = load(data, content_type)
            reading = normalize_ocr_text(self.engine().read(image).text).text
            name = _safe(post_id) + ".jpg"
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=92)
            with open(os.path.join(self.folder, name), "wb") as f:
                f.write(buf.getvalue())
        except Exception as e:             # never stops the fetch; nothing is stored for this post
            self.problems.append({"post_id": post_id, "reason": f"picture not stored: {type(e).__name__}: {str(e)[:160]}"})
            return
        self.record["photos"].append({"file": name, "post_id": post_id, "post_url": post.get("permalink_url") or "",
                                      "post_date": (post.get("created_time") or "")[:10],
                                      "page": page_label(post.get("permalink_url") or ""),
                                      "caption": post.get("message") or "", "reading": reading})
        self.known.add(post_id)
        self.saved.append(post_id)
        self._save()


def load(folder: str):
    """(entries, pages, readings) for the stored pictures: photo-index entries, photo id -> page, photo id -> reading."""
    from PIL import Image
    from src.image.photo_match import index_entry
    path = os.path.join(folder, "index.json")
    if not os.path.isfile(path):
        return [], {}, {}
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    entries, pages, readings = [], {}, {}
    for p in record.get("photos", []):
        photo_id = "auto/" + p["file"]
        try:
            with Image.open(os.path.join(folder, p["file"])) as im:
                entries.append(index_entry(photo_id, im.convert("RGB"), p["post_id"], p["post_url"], p["post_date"]))
        except OSError:
            continue
        pages[photo_id] = p.get("page") or "a Facebook page linked to DYCI"
        readings[photo_id] = p.get("reading") or ""
    return entries, pages, readings


def _values(text: str) -> dict:
    from src.verification.unit_check import units
    from src.verification.value_check import extract_values
    found = {k: set(v) for k, v in extract_values(text or "").items() if k in KINDS}
    for kind, names in units(text or "").items():
        found["unit:" + kind] = set(names)
    return found


def edited(original_reading: str, upload_text: str) -> Optional[dict]:
    """The first swapped kind between DYCI's picture and the upload, or None (Amendment 1)."""
    from src.verification.value_withhold import show
    a, b = _values(original_reading), _values(upload_text)
    for kind in [*KINDS, "unit:office", "unit:level", "unit:grade", "unit:venue"]:
        if kind in a and kind in b and (a[kind] - b[kind]) and (b[kind] - a[kind]):
            base = kind.split(":")[-1]
            shown = lambda vs: sorted(show(base, v) for v in vs)
            return {"version": VERSION, "kind": base, "dyci_values": shown(a[kind] - b[kind]),
                    "picture_values": shown(b[kind] - a[kind])}
    return None
