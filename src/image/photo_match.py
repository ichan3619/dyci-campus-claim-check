"""
Image Check Phase 2A: photo-reuse matching (results/round2/image_phase2a_plan.md, approved 2026-10-24).

Finds whether an uploaded image contains a photo that DYCI has already posted, so the page can say "This photo also
appears in DYCI's post of <date>". Display only: it never decides or changes a verdict, a statement result or
text_check, and "no match" means nothing. It never identifies people (no face recognition).

    Stage 1  perceptual hash (64-bit DCT pHash) of the whole image; a match needs Hamming distance <= max_hamming.
             Catches resized and recompressed copies.
    Stage 2  ORB features and a RANSAC homography; a match needs >= min_inliers inliers. Catches a photo inside a
             screenshot, crops and photos of a screen.

max_hamming and min_inliers are chosen on the dev split only and then frozen (plan section 2). The defaults below are
development placeholders, not the frozen values. Nothing here reads or writes files; callers pass PIL images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.fft import dct

DEV_MAX_HAMMING = 10        # placeholder until the dev split is tuned
DEV_MIN_INLIERS = 25        # placeholder until the dev split is tuned
ORB_FEATURES = 1500
ORB_MAX_SIDE = 1024
RATIO = 0.75                # Lowe's ratio test
# Amendment 2 (plan section 5b): a stage-2 match must map the index photo onto the query as a plausible picture.
GEOM_MIN_AREA = 0.02        # projected photo covers at least 2% of the query image ...
GEOM_MAX_AREA = 4.0         # ... and at most 400%
GEOM_MAX_ASPECT = 2.0       # aspect ratio kept within a factor of 2


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash: 32x32 grey, 2-D DCT, the 8x8 low-frequency block against its median (DC excluded)."""
    grey = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    coeffs = dct(dct(grey, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = coeffs.flatten()
    median = np.median(flat[1:])
    bits = 0
    for value in flat:
        bits = (bits << 1) | int(value > median)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _features(img: Image.Image):
    """(points, descriptors, size) of ORB on the grey image scaled to at most ORB_MAX_SIDE; size is that scaled size."""
    import cv2
    grey = img.convert("L")
    scale = min(1.0, ORB_MAX_SIDE / max(grey.size))
    if scale < 1.0:
        grey = grey.resize((max(1, round(grey.width * scale)), max(1, round(grey.height * scale))), Image.LANCZOS)
    keypoints, descriptors = cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(np.asarray(grey, dtype=np.uint8), None)
    points = np.float32([k.pt for k in keypoints]) if keypoints else np.zeros((0, 2), np.float32)
    return points, descriptors, grey.size


def _plausible(homography, src_size, dst_size) -> bool:
    """Amendment 2: the index photo's corners, projected into the query, form a convex, same-orientation quadrilateral
    covering 2%-400% of the query, with the aspect ratio kept within a factor of 2."""
    import cv2
    w, h = src_size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, homography).reshape(4, 2).astype(np.float64)
    if not np.all(np.isfinite(q)):
        return False
    crosses = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not all(x > 0 for x in crosses):            # image corners in this order turn the same way: convex, not mirrored
        return False
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)))
    if not GEOM_MIN_AREA <= area / float(dst_size[0] * dst_size[1]) <= GEOM_MAX_AREA:
        return False
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if height <= 0:
        return False
    ratio = (width / height) / (w / h)
    return 1 / GEOM_MAX_ASPECT <= ratio <= GEOM_MAX_ASPECT


@dataclass(frozen=True)
class IndexEntry:
    photo_id: str
    post_id: str
    post_url: str
    post_date: str
    hash: int
    points: np.ndarray
    descriptors: Optional[np.ndarray]
    size: tuple = (0, 0)        # the scaled size the features were taken from


@dataclass(frozen=True)
class PhotoMatch:
    photo_id: str
    post_id: str
    post_url: str
    post_date: str
    stage: str                  # "hash" or "features"
    hamming: Optional[int] = None
    inliers: Optional[int] = None


def index_entry(photo_id: str, img: Image.Image, post_id: str, post_url: str = "", post_date: str = "") -> IndexEntry:
    points, descriptors, size = _features(img)
    return IndexEntry(photo_id, post_id, post_url, post_date, phash(img), points, descriptors, size)


def _inliers(query_points, query_des, entry: IndexEntry, query_size) -> int:
    import cv2
    if query_des is None or entry.descriptors is None or len(query_des) < 2 or len(entry.descriptors) < 2:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(entry.descriptors, query_des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 4:
        return 0
    src = np.float32([entry.points[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([query_points[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None or not _plausible(homography, entry.size, query_size):
        return 0
    return int(mask.sum())


def match(img: Image.Image, index: Sequence[IndexEntry], max_hamming: int = DEV_MAX_HAMMING,
          min_inliers: int = DEV_MIN_INLIERS) -> Optional[PhotoMatch]:
    """The best match of `img` against `index`, or None. Stage 1 first; stage 2 only when stage 1 finds nothing."""
    if not index:
        return None
    h = phash(img)
    best = min(index, key=lambda e: hamming(h, e.hash))
    distance = hamming(h, best.hash)
    if distance <= max_hamming:
        return PhotoMatch(best.photo_id, best.post_id, best.post_url, best.post_date, "hash", hamming=distance)
    points, descriptors, size = _features(img)
    scored: List[tuple] = [(_inliers(points, descriptors, e, size), e) for e in index]
    inliers, entry = max(scored, key=lambda t: t[0])
    if inliers >= min_inliers:
        return PhotoMatch(entry.photo_id, entry.post_id, entry.post_url, entry.post_date, "features", inliers=inliers)
    return None


@dataclass(frozen=True)
class PhotoIndex:
    entries: List[IndexEntry]
    max_hamming: int
    min_inliers: int
    measured: dict
    pages: dict = None          # photo id -> who posted it ("DYCI", "ACES, a DYCI student organisation", ...)
    texts: dict = None          # post id -> the post's checked caption (only for the cross-check with that one post)


def load_index(folder: str) -> PhotoIndex:
    """The app's photo index (data/photo_index/index.json, built by scripts/round2/photo_index_build.py): every DYCI photo
    with its post link and date, the frozen thresholds and the measured rates. Reads files only."""
    import json
    import os
    with open(os.path.join(folder, "index.json"), encoding="utf-8") as f:
        record = json.load(f)
    entries = []
    for p in record["photos"]:
        with Image.open(os.path.join(folder, p["file"])) as im:
            entries.append(index_entry(p["file"], im.convert("RGB"), p["post_id"], p["post_url"], p["post_date"]))
    pages = {p["file"]: p.get("page") or "DYCI" for p in record["photos"]}
    texts = {p["post_id"]: p["text"] for p in record["photos"] if p.get("text")}
    return PhotoIndex(entries, record["max_hamming"], record["min_inliers"], record["measured"], pages, texts)
