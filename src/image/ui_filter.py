"""
Image Check v2, option A: remove Facebook interface text before checking (results/round2/image_check_v2_plan.md,
approved 2026-10-26). Setting CAMPUS_VERIFY_IMAGE_UI_FILTER, default off. Built from development screenshots only
(data/photo_eval/files/*_S.png and the image_eval development items); the 20 final posts are never used to write it.

read_boxes() reads an image exactly as OCREngine.read() (src/image/ocr.py, frozen, not edited) does, with the same
reader, settings and reading order, but keeps every box's position and confidence. Joined, its text is byte-identical
to OCREngine.read()'s (tested).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.image.ocr import OCREngine, order_boxes

ENV = "CAMPUS_VERIFY_IMAGE_UI_FILTER"


@dataclass(frozen=True)
class Box:
    text: str
    confidence: float
    top: float
    bottom: float
    left: float
    right: float


def read_boxes(engine: OCREngine, image) -> Tuple[List[Box], Tuple[int, int]]:
    """(boxes in OCREngine.read()'s reading order, (width, height) of the image the boxes refer to)."""
    if max(image.size) > engine.max_side:
        image = image.copy()
        image.thumbnail((engine.max_side, engine.max_side))
    arr = np.asarray(image)
    with engine._lock:
        reader = engine._get()
        previous = None
        if engine.threads > 0:
            import torch
            previous = torch.get_num_threads()
            torch.set_num_threads(engine.threads)
        try:
            results = reader.readtext(arr, detail=1, paragraph=False, canvas_size=engine.canvas_size)
        finally:
            if previous is not None:
                import torch
                torch.set_num_threads(previous)
    boxes = []
    for points, text, conf in order_boxes(results):
        if not text.strip():
            continue
        ys, xs = [p[1] for p in points], [p[0] for p in points]
        boxes.append(Box(text.strip(), float(conf), float(min(ys)), float(max(ys)), float(min(xs)), float(max(xs))))
    return boxes, image.size


def joined(boxes: List[Box]) -> str:
    """The text exactly as OCREngine.read() would return it."""
    return " ".join(b.text for b in boxes)


def group_lines(boxes: List[Box]) -> List[List[Box]]:
    """Consecutive boxes (already in reading order) that overlap vertically form one visual line."""
    lines: List[List[Box]] = []
    for b in boxes:
        if lines:
            last = lines[-1]
            top, bottom = min(x.top for x in last), max(x.bottom for x in last)
            overlap = min(bottom, b.bottom) - max(top, b.top)
            if overlap >= 0.5 * min(bottom - top, b.bottom - b.top):
                last.append(b)
                continue
        lines.append([b])
    return lines


# ---- the rules (results/round2/image_check_v2_plan.md section 3) ------------------------------------------------------
import re  # noqa: E402

_MONTH = (r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|"
          r"nov(ember)?|dec(ember)?)")
_TIME = r"\d{1,2}\s?[:.]\s?\d{2}\s?[ap]\.?\s?m\.?"
# A whole line made only of Facebook interface (after lower-casing and dropping the separators "·", "•", "|").
UI_LINE = re.compile("|".join([
    # Facebook timestamps: "Jul", "September 26", "September 26 at 11:03 PM", "July 13, 2025 at 4:12 PM". A date WITH a
    # year counts only when followed by a time: a bare "July 13, 2026" line can be the announcement's own date.
    rf"{_MONTH}\.?(\s\d{{1,2}})?(\sat\s{_TIME})?",
    rf"{_MONTH}\.?\s\d{{1,2}},?\s\d{{4}}\sat\s{_TIME}",
    rf"(yesterday|today)(\sat\s{_TIME})?",                                    # "Yesterday at 5:40 PM"
    r"\d{1,2}\s?(m|h|d|w|min|mins|hr|hrs|hour|hours|day|days|wk|wks)",       # "2h", "3 days"
    r"(see\s(more|less|translation|original)|follow(ing)?|like|comment|share|send|reply|edited|author|top\sfan|"
    r"most\srelevant|all\sreactions|write\sa\s(comment|public\scomment)\W*|view\s(more|previous)\scomments|"
    r"like\s+comment(\s+share)?(\s+send)?|comment\s+share)",
    r"\d+(\.\d+)?\s?k?(\s(comments?|shares?|reactions?|views?))?",             # "1.2K", "34 comments"
]))
INLINE_CUT = re.compile(r"^see\s(more|less)\W*$")                               # a box after which the line is not caption
_SEPARATORS = re.compile(r"[·•|]+")
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)                               # candidates, chosen on development only
# Chosen 2026-10-26 by the rule fixed before measuring (scripts/round2/image_v2_dev.py, results/round2/image_v2_dev.json):
# the highest candidate losing at most 5% of announcement words on the development images (0.20: 2.2% lost, 36% of
# the other lines removed; 0.30 lost 7.2%).
MIN_LINE_CONFIDENCE = 0.20


def enabled() -> bool:
    import os
    return os.environ.get(ENV, "0").strip() == "1"


def read_filtered(engine: OCREngine, image):
    """(OCRResult exactly as OCREngine.read() would return it, FilterResult). Used only with the setting on."""
    from src.image.ocr import OCRResult
    boxes, _ = read_boxes(engine, image)
    confs = [b.confidence for b in boxes]
    raw = OCRResult(text=joined(boxes), confidence=round(sum(confs) / len(confs), 4) if confs else None, lines=len(boxes))
    return raw, filter_boxes(boxes, MIN_LINE_CONFIDENCE)


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", _SEPARATORS.sub(" ", text.lower())).strip(" .,:;!-")


def line_confidence(line: List[Box]) -> float:
    """Mean box confidence, weighted by the number of characters in each box."""
    weights = [max(1, len(b.text)) for b in line]
    return sum(b.confidence * w for b, w in zip(line, weights)) / sum(weights)


@dataclass
class FilterResult:
    text: str                                    # what is checked: the kept boxes, joined as OCREngine.read() joins
    removed: List[dict]                          # {"text", "reason"} for every removed part, in reading order
    lines_in: int
    lines_kept: int


def filter_boxes(boxes: List[Box], min_line_confidence: Optional[float]) -> FilterResult:
    """Remove interface lines, everything after an inline "See more"/"See less", and (when min_line_confidence is set)
    lines whose mean confidence is below it. With no rule matching, the text equals joined(boxes)."""
    kept: List[Box] = []
    removed: List[dict] = []
    lines = group_lines(boxes)
    for line in lines:
        text = " ".join(b.text for b in line)
        if UI_LINE.fullmatch(_canon(text) or "-"):
            removed.append({"text": text, "reason": "interface line"})
            continue
        cut = next((i for i, b in enumerate(line) if INLINE_CUT.match(_canon(b.text))), None)
        if cut is not None:
            removed.append({"text": " ".join(b.text for b in line[cut:]), "reason": "after See more / See less"})
            line = line[:cut]
            if not line:
                continue
        if min_line_confidence is not None and line_confidence(line) < min_line_confidence:
            removed.append({"text": " ".join(b.text for b in line), "reason": "low reading confidence"})
            continue
        kept.extend(line)
    return FilterResult(joined(kept), removed, len(lines), len(group_lines(kept)))
