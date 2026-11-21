"""
OCR for Image Check Phase 1: EasyOCR 1.7.2, English + Tagalog, CPU, offline.

The two model files were downloaded once in the approved install step (2026-10-13) to EasyOCR's standard folder
(~/.EasyOCR/model). At run time nothing is downloaded (download_enabled=False) and the files are refused unless their
SHA-256 matches MODEL_FILES. The reader is created on the first image request only (lazy), so text checks never pay for
it. Images are passed as in-memory numpy arrays; nothing is written to disk. Text and boxes are returned to the caller
and never logged.
"""
import hashlib
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

LANGUAGES = ["en", "tl"]
# Optimisation experiment settings (results/round2/image_ocr_optimization_protocol.md). The defaults reproduce v0 exactly:
#   CAMPUS_VERIFY_IMAGE_OCR_MAX_SIDE  2048  long side of the image handed to OCR (intake already caps it at 2048)
#   CAMPUS_VERIFY_IMAGE_OCR_CANVAS    2560  EasyOCR canvas_size (its own default)
#   CAMPUS_VERIFY_IMAGE_OCR_THREADS   0     CPU threads during the OCR call only (0 = leave torch's default)
DEFAULT_MAX_SIDE, DEFAULT_CANVAS, DEFAULT_THREADS = 2048, 2560, 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
MODEL_FILES = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",   # text detection
    "latin_g2.pth": "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55",        # Latin recognition
}


def order_boxes(results):
    """EasyOCR results ([box points], text, confidence) in reading order: grouped into lines, lines top to bottom, boxes
    in a line left to right (Part A, results/round2/image_partA_protocol.md).

    A box joins a line when its vertical extent overlaps the line's extent (mean top, mean bottom of its boxes so far) by
    at least half the smaller of the two heights; boxes are taken by top edge, and a box that qualifies for several lines
    joins the one with the largest overlap. Sorting by top edge alone misordered words of one line whose boxes start a few
    pixels apart (taller letters, descenders)."""
    lines = []                                       # each: {"top": [..], "bottom": [..], "items": [(left, result)]}
    for r in sorted(results, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0]))):
        top, bottom, left = min(p[1] for p in r[0]), max(p[1] for p in r[0]), min(p[0] for p in r[0])
        best, best_overlap = None, 0.0
        for line in lines:
            lt, lb = sum(line["top"]) / len(line["top"]), sum(line["bottom"]) / len(line["bottom"])
            overlap = min(bottom, lb) - max(top, lt)
            if overlap >= 0.5 * min(bottom - top, lb - lt) and overlap > best_overlap:
                best, best_overlap = line, overlap
        if best is None:
            best = {"top": [], "bottom": [], "items": []}
            lines.append(best)
        best["top"].append(top)
        best["bottom"].append(bottom)
        best["items"].append((left, r))
    lines.sort(key=lambda l: (sum(l["top"]) / len(l["top"]) + sum(l["bottom"]) / len(l["bottom"])) / 2)
    return [r for line in lines for _, r in sorted(line["items"], key=lambda x: x[0])]


@dataclass
class OCRResult:
    text: str
    confidence: Optional[float]
    lines: int


class OCREngine:
    def __init__(self, model_dir: str = MODEL_DIR, max_side: int = None, canvas_size: int = None, threads: int = None):
        self.model_dir = model_dir
        self.max_side = max_side if max_side is not None else _env_int("CAMPUS_VERIFY_IMAGE_OCR_MAX_SIDE", DEFAULT_MAX_SIDE)
        self.canvas_size = canvas_size if canvas_size is not None else _env_int("CAMPUS_VERIFY_IMAGE_OCR_CANVAS", DEFAULT_CANVAS)
        self.threads = threads if threads is not None else _env_int("CAMPUS_VERIFY_IMAGE_OCR_THREADS", DEFAULT_THREADS)
        self._reader = None
        self._lock = threading.Lock()

    def settings(self) -> dict:
        return {"max_side": self.max_side, "canvas_size": self.canvas_size, "threads": self.threads}

    def _verify_models(self) -> None:
        for name, expected in MODEL_FILES.items():
            path = os.path.join(self.model_dir, name)
            if not os.path.isfile(path):
                raise RuntimeError(f"OCR model file missing: {name} (run the approved install step)")
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != expected:
                    raise RuntimeError(f"OCR model file {name} does not match its recorded SHA-256")

    def _get(self):
        if self._reader is None:
            self._verify_models()
            import easyocr
            self._reader = easyocr.Reader(LANGUAGES, gpu=False, verbose=False, download_enabled=False,
                                          model_storage_directory=self.model_dir)
        return self._reader

    def read(self, image) -> OCRResult:
        """Text read from a PIL image, in reading order (order_boxes: lines top to bottom, left to right in a line)."""
        if max(image.size) > self.max_side:               # experiment factor 1 (default 2048: no change after intake)
            image = image.copy()
            image.thumbnail((self.max_side, self.max_side))
        arr = np.asarray(image)
        with self._lock:                                  # one inference at a time on this reader
            reader = self._get()
            previous = None
            if self.threads > 0:                          # experiment factor 3, restored right after the call
                import torch
                previous = torch.get_num_threads()
                torch.set_num_threads(self.threads)
            try:
                results = reader.readtext(arr, detail=1, paragraph=False, canvas_size=self.canvas_size)   # factor 2
            finally:
                if previous is not None:
                    import torch
                    torch.set_num_threads(previous)
        results = order_boxes(results)
        texts: List[str] = [r[1].strip() for r in results if r[1].strip()]
        confs = [float(r[2]) for r in results if r[1].strip()]
        return OCRResult(text=" ".join(texts), confidence=round(sum(confs) / len(confs), 4) if confs else None,
                         lines=len(texts))
