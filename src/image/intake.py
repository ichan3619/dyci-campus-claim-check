"""
Secure in-memory image intake (results/round2/image_check_plan_draft.md revision 2, section 2.5).

Order, cheapest first, nothing expensive before the cheap checks pass:
    1. declared Content-Type must be image/jpeg, image/png or image/webp                       -> 415
    2. the first bytes must match that type (JPEG FF D8 FF, PNG signature, WebP RIFF....WEBP)  -> 422
    3. width x height read from the header only (Image.open is lazy) must be <= MAX_PIXELS      -> 422
    4. decode, apply the EXIF orientation, convert to RGB, downscale so the long side <= MAX_SIDE
Bytes are only ever held in memory (io.BytesIO); nothing is written to disk and nothing is logged. Error messages are
generic and never include image content.
"""
import io
from typing import Optional

from PIL import Image, ImageOps

ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_PIXELS = 40_000_000
MAX_SIDE = 2048
Image.MAX_IMAGE_PIXELS = MAX_PIXELS          # Pillow's own decompression-bomb guard, set explicitly

UNSUPPORTED = "Unsupported image type. Use a JPEG, PNG or WebP image."
UNREADABLE = "The image could not be read."
TOO_LARGE_IMAGE = "The image is too large to process."


class ImageRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def media_type(content_type: Optional[str]) -> str:
    return (content_type or "").split(";")[0].strip().lower()


def _signature_ok(kind: str, data: bytes) -> bool:
    if kind == "image/jpeg":
        return data[:3] == b"\xff\xd8\xff"
    if kind == "image/png":
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    if kind == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def load(data: bytes, content_type: Optional[str]) -> Image.Image:
    """The checked, decoded, oriented, RGB, downscaled image. Raises ImageRejected with a generic message."""
    kind = media_type(content_type)
    if kind not in ALLOWED_TYPES:
        raise ImageRejected(415, UNSUPPORTED)
    if not data or not _signature_ok(kind, data):
        raise ImageRejected(422, UNREADABLE)
    try:
        img = Image.open(io.BytesIO(data))               # reads the header only
        w, h = img.size
        if w <= 0 or h <= 0 or w * h > MAX_PIXELS:
            raise ImageRejected(422, TOO_LARGE_IMAGE)
        expected = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[kind]
        if img.format != expected:
            raise ImageRejected(422, UNREADABLE)
        img.load()                                       # full decode only after every cheap check passed
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        return img
    except ImageRejected:
        raise
    except Image.DecompressionBombError:                 # Pillow's header-stage guard (above 2 x MAX_PIXELS)
        raise ImageRejected(422, TOO_LARGE_IMAGE)
    except Exception:                                  # truncated, corrupt, bomb, decoder errors: one generic answer
        raise ImageRejected(422, UNREADABLE)
