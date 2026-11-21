"""
Image Check Phase 2B, part P1: provenance facts (results/round2/ai_image_plan_draft.md section 1; approved 2026-10-26).
Setting CAMPUS_VERIFY_IMAGE_PROVENANCE, default off. Display only: it never changes a verdict, a sentence check or the
legitimacy level.

It reports what the uploaded FILE says about itself:
    Content Credentials (C2PA)  signed record of who made the file and how; "made with AI" when its actions say
                                trainedAlgorithmicMedia (made) or compositeWithTrainedAlgorithmicMedia (edited)
    the file's own AI label     the IPTC digital source type in the XMP packet; NOT signed, anyone can add or remove it
    camera details              EXIF make, model, software and date taken only
It never says a picture is real because a record exists, or fake because none does: Facebook and Messenger remove all
three, so "none found" is the normal result for a shared picture.

Privacy and safety:
- Remote manifests are never fetched (verify.remote_manifest_fetch false; tests/test_provenance.py proves no request is
  made), so an upload can never make the server contact an address taken from the picture. OCSP fetch is off too.
- Only the fields above are returned, each cut to MAX_FIELD characters. GPS, serial numbers, owner and author names in
  EXIF or in the C2PA record are never read out.
- Signatures are checked, but the signer is NOT checked against a trust list (none is shipped), so a valid record is
  described as "signed by <name>", never as trusted.
- Any failure gives status "unreadable"; it never fails the image check.
"""
import io
import json
import os
import re
from typing import Optional

ENV = "CAMPUS_VERIFY_IMAGE_PROVENANCE"
VERSION = "provenance-v1"
MAX_FIELD = 80
_IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
AI_MADE = "trainedAlgorithmicMedia"
AI_EDITED = "compositeWithTrainedAlgorithmicMedia"
_XMP = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)
_SOURCE_TYPE = re.compile(r"digitalsourcetype/(\w+)", re.I)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

NONE_FOUND = ("No Content Credentials or camera details were found. This is normal for pictures shared on Facebook or "
              "Messenger, which remove them, so it says nothing about whether the picture is real.")
NOTE = ("These are facts recorded in the file. Content Credentials show where a file came from and how it was made, "
        "not whether what it shows is true. Camera details and unsigned labels can be edited by anyone.")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _clip(value) -> Optional[str]:
    if value is None:
        return None
    text = _CONTROL.sub(" ", str(value)).strip()
    return text[:MAX_FIELD] or None


_context = None


def _reader_context():
    global _context
    if _context is None:
        import c2pa
        _context = c2pa.Context(c2pa.Settings.from_dict(
            {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}))
    return _context


def _source_types(manifest: dict):
    for assertion in manifest.get("assertions") or []:
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions") or []:
                kind = str(action.get("digitalSourceType") or "")
                if kind:
                    yield kind.rsplit("/", 1)[-1]


def _ai(kinds) -> Optional[str]:
    kinds = set(kinds)
    return "made" if AI_MADE in kinds else "edited" if AI_EDITED in kinds else None


def content_credentials(data: bytes, media_type: str) -> dict:
    """The C2PA record of the file: status none | signed | invalid | remote_not_fetched | unreadable."""
    try:
        import c2pa
        try:
            reader = c2pa.Reader.try_create(media_type, io.BytesIO(data), context=_reader_context())
        except c2pa.C2paError as e:
            if "remote" in str(e).lower():
                return {"status": "remote_not_fetched"}
            return {"status": "unreadable"}
        if reader is None:
            return {"status": "none"}
        try:
            store = json.loads(reader.json())
            state = reader.get_validation_state()
        finally:
            reader.close()
        manifest = (store.get("manifests") or {}).get(store.get("active_manifest")) or {}
        generator = manifest.get("claim_generator_info") or [{}]
        generator = " ".join(x for x in (generator[0].get("name"), generator[0].get("version")) if x) or \
            manifest.get("claim_generator")
        signature = manifest.get("signature_info") or {}
        return {"status": "invalid" if state == "Invalid" else "signed",
                "ai": _ai(_source_types(manifest)),
                "generator": _clip(generator),
                "signed_by": _clip(signature.get("issuer") or signature.get("common_name")),
                "signed_on": _clip(signature.get("time")),
                "trust_list_checked": False}
    except Exception:                                   # never fails the image check; nothing about the image is logged
        return {"status": "unreadable"}


def file_label(data: bytes) -> Optional[dict]:
    """The IPTC digital source type in the XMP packet, if any (unsigned)."""
    match = _XMP.search(data)
    if not match:
        return None
    kinds = _SOURCE_TYPE.findall(match.group(0).decode("utf-8", "replace"))
    if not kinds:
        return None
    return {"source_type": _clip(kinds[0]), "ai": _ai(kinds)}


def camera(data: bytes) -> Optional[dict]:
    """EXIF make, model, software and date taken only (never GPS, serial numbers or owner names)."""
    try:
        from PIL import Image
        exif = Image.open(io.BytesIO(data)).getexif()
        found = {"make": _clip(exif.get(0x010F)), "model": _clip(exif.get(0x0110)),
                 "software": _clip(exif.get(0x0131)),
                 "taken": _clip(exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132))}
    except Exception:
        return None
    return found if any(found.values()) else None


def summary(credentials: dict, label: Optional[dict], cam: Optional[dict]) -> str:
    status = credentials["status"]
    parts = []
    if status == "signed":
        made = {"made": "made with an AI tool", "edited": "edited with an AI tool"}.get(credentials.get("ai"))
        who = f"signed by “{credentials['signed_by']}”" if credentials.get("signed_by") else "signed"
        tool = f" ({credentials['generator']})" if credentials.get("generator") else ""
        if made:
            parts.append(f"This picture carries Content Credentials saying it was {made}{tool}, {who}.")
        else:
            parts.append(f"This picture carries Content Credentials{tool}, {who}. The signer was not checked "
                         "against a list of trusted signers.")
    elif status == "invalid":
        parts.append("This picture carries Content Credentials that do not match it: the file was changed after it "
                     "was signed, or the record is damaged.")
    elif status == "remote_not_fetched":
        parts.append("This picture points to Content Credentials stored online. They were not fetched, to protect "
                     "your privacy.")
    if label and label.get("ai"):
        parts.append("The file's own label says it was " + ("made" if label["ai"] == "made" else "edited")
                     + " with AI. This label is not signed, so anyone can add or remove it.")
    if cam:
        what = " ".join(x for x in (cam.get("make"), cam.get("model")) if x)
        bits = [b for b in (what and f"camera {what}", cam.get("software") and f"software {cam['software']}",
                            cam.get("taken") and f"date {cam['taken']}") if b]
        parts.append("Camera details in the file: " + ", ".join(bits) + ".")
    if not parts:
        return NONE_FOUND if status == "none" else "The file's details could not be read."
    return " ".join(parts)


def check(data: bytes, media_type: str) -> dict:
    """Provenance facts for an upload that already passed intake (src/image/intake.py load)."""
    credentials = content_credentials(data, media_type)
    label, cam = file_label(data), camera(data)
    return {"version": VERSION, "content_credentials": credentials, "file_label": label, "camera": cam,
            "summary": summary(credentials, label, cam), "note": NOTE}
