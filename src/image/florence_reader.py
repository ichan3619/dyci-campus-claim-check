"""
Image Check option B: Florence-2 base as the reader (results/round2/image_check_b_plan.md, approved 2026-10-26).
Setting CAMPUS_VERIFY_IMAGE_READER: "easyocr" (default, today's reader, unchanged) or "florence2".

Florence-2 base (florence-community/Florence-2-base, MIT licence, pinned revision below) runs locally on the CPU with
the installed libraries; nothing leaves the server. Task <OCR_WITH_REGION> gives one text per line with its position;
decoding is greedy (num_beams=1). Development measurement (results/round2/image_b_feasibility.md): 93% of caption
words found against EasyOCR's 72%, 1.7 GB of memory.

The lines are put in reading order by the same function EasyOCR's boxes go through (src/image/ocr.py order_boxes),
joined with spaces as EasyOCR's text is, and option A's interface rules (src/image/ui_filter.py, unchanged) are
applied. Its confidence rule is not: Florence-2 reports no per-line confidence.

HEADER RULE (added 2026-10-26 on development images only, before the freeze; owner-approved): Facebook's post header
puts the page name and the timestamp on ONE line ("School News 2.0 October 4 at 2:19PM", "Follow August 20"), so
option A's whole-line rule misses it and the header is glued to the first statement (development route check: two
genuine phone photos contradicted that way). A line above the first long line (more than HEADER_MAX_WORDS words)
that has at most HEADER_MAX_WORDS words and contains a Facebook timestamp with a time ("October 4 at 2:19PM"),
a relative time ("2h", "Yesterday at 5:40 PM", "Just now") or the word "Follow" is removed.

This is text reading only; it does not judge whether a picture is genuine.
"""
import os
import re
import threading
from typing import List, Optional, Tuple

from src.image.ocr import OCRResult, order_boxes
from src.image.ui_filter import _MONTH, _TIME, Box, FilterResult, _canon, filter_boxes, group_lines, joined

ENV = "CAMPUS_VERIFY_IMAGE_READER"
EASYOCR, FLORENCE = "easyocr", "florence2"
REPO = "florence-community/Florence-2-base"
REVISION = "00921df66db728a9ceb750f5eca43e5c203a2051"
TASK = "<OCR_WITH_REGION>"
MAX_NEW_TOKENS = 1024
HEADER_MAX_WORDS = 10
# A date alone does NOT mark a header: "Classes are suspended on September 26." is a short announcement line. Only a
# timestamp WITH a time, a relative time, or "Follow" does (caught by a unit test before the freeze).
HEADER_MARK = re.compile(rf"\b{_MONTH}\.?\s\d{{1,2}}(,?\s\d{{4}})?\sat\s{_TIME}|\b(yesterday|today)\sat\s{_TIME}|"
                         rf"\bjust\snow\b|\b\d{{1,2}}\s?(m|h|d|w|min|mins|hr|hrs)\b|\bfollow\b")


def remove_header(boxes: List[Box]) -> Tuple[List[Box], List[dict]]:
    """Drop short lines that carry a Facebook timestamp or "Follow", above the first long line (see HEADER RULE)."""
    kept, removed = [], []
    in_header = True
    for line in group_lines(boxes):
        text = " ".join(b.text for b in line)
        n_words = len(text.split())
        if in_header and n_words > HEADER_MAX_WORDS:
            in_header = False
        if in_header and HEADER_MARK.search(_canon(text)):
            removed.append({"text": text, "reason": "post header (page name and time)"})
            continue
        kept.extend(line)
    return kept, removed


def reader_name() -> str:
    value = os.environ.get(ENV, EASYOCR).strip().lower() or EASYOCR
    if value not in (EASYOCR, FLORENCE):
        raise ValueError(f"{ENV} must be {EASYOCR!r} or {FLORENCE!r}, not {value!r}")
    return value


def boxes_from_regions(quad_boxes, labels) -> List[Box]:
    """Florence-2's (quad box, text) pairs in reading order, as ui_filter.Box values (confidence unknown: 1.0)."""
    results = []
    for quad, label in zip(quad_boxes, labels):
        text = (label or "").replace("</s>", "").replace("<s>", "").strip()
        if not text:
            continue
        points = [[float(quad[i]), float(quad[i + 1])] for i in range(0, 8, 2)]
        results.append((points, text, 1.0))
    out = []
    for points, text, conf in order_boxes(results):
        ys, xs = [p[1] for p in points], [p[0] for p in points]
        out.append(Box(text, conf, min(ys), max(ys), min(xs), max(xs)))
    return out


class FlorenceReader:
    """Loads the model once (on first use) and reads one image at a time."""

    def __init__(self, repo: str = REPO, revision: str = REVISION):
        self.repo, self.revision = repo, revision
        self._lock = threading.Lock()
        self._model = self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
            self._model = Florence2ForConditionalGeneration.from_pretrained(
                self.repo, revision=self.revision, torch_dtype=torch.float32).eval()
        return self._processor, self._model

    def read_boxes(self, image) -> List[Box]:
        import torch
        with self._lock:
            processor, model = self._load()
            with torch.inference_mode():
                inputs = processor(text=TASK, images=image, return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=1, do_sample=False)
            raw = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw, task=TASK, image_size=image.size)[TASK]
        return boxes_from_regions(parsed.get("quad_boxes", []), parsed.get("labels", []))

    def read_filtered(self, image) -> Tuple[OCRResult, FilterResult]:
        """(the full reading, as OCRResult; the reading after option A's interface rules)."""
        boxes = self.read_boxes(image)
        raw = OCRResult(text=joined(boxes), confidence=None, lines=len(boxes))
        kept, header = remove_header(boxes)
        filtered = filter_boxes(kept, None)
        filtered.removed[:0] = header
        filtered.lines_in = len(group_lines(boxes))
        return raw, filtered
