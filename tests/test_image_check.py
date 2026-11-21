"""Image Check Phase 1 (CAMPUS_VERIFY_IMAGE, default off): secure in-memory intake, OCR, and the unchanged /verify/text
pipeline for the text read (results/round2/image_check_plan_draft.md revision 2, sections 2.3-2.5)."""
import asyncio
import builtins
import io
import logging
import os
import struct
import tempfile
import zlib
from dataclasses import replace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from src.image import intake
from src.image.intake import ImageRejected, load
from src.image.normalize import agreement_variant, normalize_ocr_text
from src.image.ocr import MODEL_FILES, OCREngine, OCRResult, order_boxes

FONT = r"C:\Windows\Fonts\arial.ttf"


def _png(img):
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


def _jpeg(img, exif=None):
    buf = io.BytesIO(); img.save(buf, "JPEG", **({"exif": exif} if exif else {})); return buf.getvalue()


def _text_image(text, size=(1080, 400)):
    img = Image.new("RGB", size, "white")
    font = ImageFont.truetype(FONT, 56) if os.path.exists(FONT) else ImageFont.load_default()
    ImageDraw.Draw(img).text((40, 150), text, fill="black", font=font)
    return img


def _chunk(kind, payload):
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _fake_png_header(width, height):
    """A structurally valid PNG whose header claims width x height (no real pixel data): tests the header-only check."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")


# ---- intake ---------------------------------------------------------------------------------------------------------
def test_accepts_the_three_types_and_downscales():
    big = Image.new("RGB", (4000, 1000), "white")
    for data, ctype in ((_png(big), "image/png"), (_jpeg(big), "image/jpeg; charset=x")):
        img = load(data, ctype)
        assert img.mode == "RGB" and max(img.size) == intake.MAX_SIDE
    buf = io.BytesIO(); Image.new("RGB", (50, 40)).save(buf, "WEBP")
    assert load(buf.getvalue(), "image/webp").size == (50, 40)


def test_exif_orientation_is_applied():
    exif = Image.Exif(); exif[0x0112] = 6                                     # rotate 90 degrees
    assert load(_jpeg(Image.new("RGB", (300, 100), "white"), exif.tobytes()), "image/jpeg").size == (100, 300)


@pytest.mark.parametrize("ctype", [None, "", "image/gif", "application/octet-stream", "text/plain"])
def test_unsupported_type_is_415(ctype):
    with pytest.raises(ImageRejected) as e:
        load(_png(Image.new("RGB", (10, 10))), ctype)
    assert e.value.status == 415 and "Unsupported" in e.value.message


@pytest.mark.parametrize("data, ctype", [
    (lambda: _png(Image.new("RGB", (10, 10))), "image/jpeg"),                 # declared JPEG, bytes are PNG
    (lambda: b"GIF89a" + b"\x00" * 50, "image/png"),
    (lambda: _png(Image.new("RGB", (64, 64)))[:40], "image/png"),             # truncated
    (lambda: b"", "image/png"),
])
def test_mismatched_or_corrupt_is_422_generic(data, ctype):
    with pytest.raises(ImageRejected) as e:
        load(data(), ctype)
    assert e.value.status == 422 and e.value.message == intake.UNREADABLE


@pytest.mark.parametrize("side", [20000, 7000])        # 400 MP: Pillow's own guard; 49 MP: our header check
def test_decompression_bomb_is_refused_from_the_header_before_decoding(side):
    with pytest.raises(ImageRejected) as e:
        load(_fake_png_header(side, side), "image/png")
    assert e.value.status == 422 and e.value.message == intake.TOO_LARGE_IMAGE


# ---- OCR engine -----------------------------------------------------------------------------------------------------
def test_model_hash_mismatch_is_refused(tmp_path):
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b"not the model")
    with pytest.raises(RuntimeError, match="SHA-256"):
        OCREngine(str(tmp_path)).read(Image.new("RGB", (20, 20), "white"))


def test_missing_model_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        OCREngine(str(tmp_path)).read(Image.new("RGB", (20, 20), "white"))


@pytest.mark.skipif(not os.path.exists(FONT), reason="needs a TrueType font")
def test_real_ocr_reads_rendered_text():
    result = OCREngine().read(_text_image("Seminar September 23"))
    assert "Seminar" in result.text and "23" in result.text and result.lines >= 1


# ---- API: setting off behaves as v1.3 ---------------------------------------------------------------------------------
def test_off_is_the_standard_404_and_the_16kb_limit_still_applies(api_settings):
    api_main = api_settings()
    with TestClient(api_main.app) as c:
        h = c.get("/health").json()
        r = c.post("/verify/image", content=_png(Image.new("RGB", (20, 20))), headers={"Content-Type": "image/png"})
        big = c.post("/verify/image", content=b"\x89PNG" + b"0" * 20000, headers={"Content-Type": "image/png"})
    assert (h["image_check"], h["image_signals"]) == (False, False)
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}
    assert big.status_code == 413 and "16384" in big.json()["detail"]
    assert api_main.state.get("ocr") is None                                  # no OCR engine created


# ---- API: setting on ------------------------------------------------------------------------------------------------
class _FakeOCR:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def read(self, image):
        self.calls += 1
        return OCRResult(text=self.text, confidence=0.9, lines=1)


def _on(api_settings, text="Classes are suspended tomorrow due to a typhoon."):
    api_main = api_settings(image_check=True)
    api_main.state["ocr"] = _FakeOCR(text)
    return api_main


def _post_image(c, data=None, ctype="image/png"):
    return c.post("/verify/image", content=data if data is not None else _png(_text_image("x")), headers={"Content-Type": ctype})


def test_text_check_is_exactly_the_verify_text_response(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("Classes are suspended tomorrow due to a typhoon.")
        img = _post_image(c).json()
        txt = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow due to a typhoon."}).json()
    assert img["text_check"] == txt
    assert img["ocr"]["text"] == "Classes are suspended tomorrow due to a typhoon." and img["image_signals"] is None
    assert "not whether the image itself was edited" in img["note"]


def test_no_readable_text_gives_no_verdict(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("  ")
        body = _post_image(c).json()
    assert body["text_check"] is None and "No readable text" in body["note"] and body["status"] == "no_text"


def test_long_text_is_truncated_to_the_claim_limit(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("Classes resume on Monday. " * 80)
        body = _post_image(c).json()
    assert body["ocr"]["truncated"] is True and len(body["ocr"]["checked_text"]) <= 1000 and body["text_check"] is not None
    assert body["ocr"]["text"] == "Classes resume on Monday. " * 80               # the raw reading is kept as read


def test_errors_are_generic(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("x")
        assert _post_image(c, ctype="image/gif").status_code == 415
        bad = _post_image(c, data=_png(Image.new("RGB", (10, 10)))[:30])
    assert bad.status_code == 422 and bad.json() == {"detail": intake.UNREADABLE}


def test_image_limit_applies_only_on_this_route_and_text_route_keeps_16kb(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("x")
        ok_size = _post_image(c, data=_png(Image.effect_noise((300, 300), 64).convert("RGB")))   # > 16 KB, < 5 MB
        too_big = _post_image(c, data=b"\x89PNG" + b"0" * (5 * 1024 * 1024 + 1))
        text_big = c.post("/verify/text", content=b'{"claim": "' + b"a" * 20000 + b'"}', headers={"Content-Type": "application/json"})
    assert ok_size.status_code == 200
    assert too_big.status_code == 413 and str(5 * 1024 * 1024) in too_big.json()["detail"]
    assert text_big.status_code == 413 and "16384" in text_big.json()["detail"]


def test_missing_content_length_is_411(api_settings):
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        r = c.post("/verify/image", content=iter([_png(Image.new("RGB", (10, 10)))]), headers={"Content-Type": "image/png"})
    assert r.status_code == 411


def test_body_read_is_capped_even_if_content_length_lies():
    from src.api.main import _read_capped

    class _Req:
        async def stream(self):
            for _ in range(3):
                yield b"x" * 100
    with pytest.raises(HTTPException) as e:
        asyncio.run(_read_capped(_Req(), 250))
    assert e.value.status_code == 413


def test_image_rate_limit_bucket(api_settings, api_protection, monkeypatch):
    api_main = _on(api_settings)
    monkeypatch.setattr(api_main, "PROTECTION", replace(api_main.PROTECTION, image_rate_per_minute=1))
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("x")
        first, second = _post_image(c), _post_image(c)
        text_ok = c.post("/verify/text", json={"claim": "may pasok?"})
    assert first.status_code == 200 and second.status_code == 429 and text_ok.status_code == 200


def test_busy_gate_gives_503_and_no_ocr(api_settings):
    from src.api.protection import Busy
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        fake = _FakeOCR("x"); api_main.state["ocr"] = fake

        class _Gate:
            def slot(self):
                raise Busy()
        api_main.state["gate"] = _Gate()
        r = _post_image(c)
    assert r.status_code == 503 and fake.calls == 0


def test_health_reports_the_setting(api_settings):
    with TestClient(_on(api_settings).app) as c:
        h = c.get("/health").json()
    assert (h["image_check"], h["image_signals"]) == (True, False)


# ---- privacy: nothing written, nothing logged (real OCR) -------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(FONT), reason="needs a TrueType font")
def test_no_disk_writes_and_no_ocr_text_in_logs(api_settings, monkeypatch, caplog):
    api_main = api_settings(image_check=True)
    secret = "Seminar September 23"
    data = _png(_text_image(secret))
    writes = []
    real_open, real_mkstemp, real_ntf = builtins.open, tempfile.mkstemp, tempfile.NamedTemporaryFile

    def spy_open(file, mode="r", *a, **k):
        if any(ch in mode for ch in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    tmp_before = set(os.listdir(tempfile.gettempdir()))
    with TestClient(api_main.app) as c:
        c.post("/verify/image", content=_png(_text_image("warm up")), headers={"Content-Type": "image/png"})   # loads OCR
        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(tempfile, "mkstemp", lambda *a, **k: writes.append("mkstemp") or real_mkstemp(*a, **k))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *a, **k: writes.append("ntf") or real_ntf(*a, **k))
        caplog.set_level(logging.DEBUG)
        body = c.post("/verify/image", content=data, headers={"Content-Type": "image/png"}).json()
        monkeypatch.setattr(builtins, "open", real_open)
    assert "Seminar" in body["ocr"]["text"]
    assert writes == []
    assert set(os.listdir(tempfile.gettempdir())) - tmp_before == set()
    assert all("Seminar" not in r.getMessage() for r in caplog.records)


# ---- harmless OCR variation (reliability investigation) -------------------------------------------------------------
@pytest.mark.parametrize("raw, cleaned, rules", [
    ("Venue: Elida Convention Center . Late registration", "Venue: Elida Convention Center. Late registration",
     ["space_before_punctuation"]),
    ("No classes\non August 21 ,  2026", "No classes on August 21, 2026", ["whitespace", "space_before_punctuation"]),
    ("Venue : Elida", "Venue: Elida", ["space_before_punctuation"]),
    ("Classes end at 1.00 PM today", "Classes end at 1:00 PM today", ["time_separator"]),
    ("Classes end at 1.00pm today", "Classes end at 1:00pm today", ["time_separator"]),
    ("Starts 10.30 a.m. sharp", "Starts 10:30 a.m. sharp", ["time_separator"]),
])
def test_harmless_ocr_noise_is_tidied_and_reported(raw, cleaned, rules):
    result = normalize_ocr_text(raw)
    assert (result.text, result.changes) == (cleaned, rules)


@pytest.mark.parametrize("text", [
    "Registration fee: PHP 1.00 only",           # an amount, not a time
    "Classes end at 13.00 PM",                   # not a 12-hour time
    "Classes end at 1.5 PM",                     # not hh.mm
    "Pi is 3.14 and the fee is PHP .50 each",    # decimals
    "Classes on September 29, not September 28", # dates and negations are never touched
    "Classes end at 2:00 PM",                    # already a time; never rewritten to another value
    "NO CLASSES ON AUGUST 21",                   # case is kept
    "Wait ... what?",                            # an ellipsis is not a stray space
])
def test_substantive_text_is_never_reinterpreted(text):
    result = normalize_ocr_text(text)
    assert (result.text, result.changes) == (text, [])


def test_agreement_variant_toggles_only_a_final_full_stop():
    assert agreement_variant("Classes are suspended.") == "Classes are suspended"
    assert agreement_variant("Classes are suspended") == "Classes are suspended."
    assert agreement_variant("Totoo ba?") is None and agreement_variant("Ingat!") is None


Q12_CLEAN = ("Registration fee: PHP 250. Deadline: October 3, 2026. Venue: Elida Convention Center. "
             "Late registration is not allowed.")


def test_space_before_a_full_stop_does_not_change_the_verdict(api_settings):
    # The Q12 regression: V2 read "Center ." and the raw text got CONTRADICTED; the clean text gets INSUFFICIENT_EVIDENCE.
    api_main = _on(api_settings, Q12_CLEAN.replace("Center.", "Center ."))
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
        clean = c.post("/verify/text", json={"claim": Q12_CLEAN}).json()
    assert body["ocr"]["checked_text"] == Q12_CLEAN and body["ocr"]["normalization"] == ["space_before_punctuation"]
    assert body["status"] == "checked" and body["text_check"] == clean
    assert body["text_check"]["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_time_read_with_a_dot_is_checked_as_the_time(api_settings):
    typed = "Classes on September 10, 2026 will shift to online modality starting 1:00 PM."
    api_main = _on(api_settings, typed.replace("1:00", "1.00"))
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
        direct = c.post("/verify/text", json={"claim": typed}).json()
        raw = c.post("/verify/text", json={"claim": typed.replace("1:00", "1.00")}).json()
    assert body["ocr"]["text"].endswith("1.00 PM.") and body["ocr"]["checked_text"] == typed
    assert body["ocr"]["normalization"] == ["time_separator"]
    assert body["text_check"] == direct and direct["verdict"] == raw["verdict"]


class _StubCheck:
    """Stands in for verify_text in the image route: a fixed verdict per exact text (the pytest suite runs TF-IDF, so the
    release-model behaviour is checked by scripts/round2/image_ocr_variation_probe.py instead)."""
    def __init__(self, verdicts, default=None, posts=None, fail_on=None):
        self.verdicts, self.calls, self.default, self.posts, self.fail_on = verdicts, [], default, posts or {}, fail_on

    def __call__(self, req):
        from src.api.schemas import RetrievedDocumentModel, VerificationResultModel
        self.calls.append(req.claim)
        if req.claim == self.fail_on:
            raise HTTPException(status_code=503, detail="The checker is busy. Try again in a few seconds.")
        verdict = self.verdicts[req.claim] if self.default is None else self.verdicts.get(req.claim, self.default)
        post = self.posts.get(req.claim.rstrip("."))
        used = RetrievedDocumentModel(document_id=post, score=0.9, text="", source_type="manual_entry",
                                      source_url="") if post else None
        return VerificationResultModel(claim=req.claim, verdict=verdict,
                                       evidence_status="RELEVANT_EVIDENCE_FOUND", retrieved_evidence=[], used_evidence=used,
                                       classification_result=None, explanation="stub", data_status="development")


def test_fragile_verdict_is_withheld_not_shown(api_settings, monkeypatch):
    # Release models: Q12 without its final full stop is CONTRADICTED, with it INSUFFICIENT_EVIDENCE.
    api_main = _on(api_settings, Q12_CLEAN[:-1])
    stub = _StubCheck({Q12_CLEAN[:-1]: "CONTRADICTED", Q12_CLEAN: "INSUFFICIENT_EVIDENCE"}, default="INSUFFICIENT_EVIDENCE")
    monkeypatch.setattr(api_main, "verify_text", stub)
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
    assert stub.calls[:2] == [Q12_CLEAN[:-1], Q12_CLEAN]                  # the whole-text check and its agreement check first
    assert body["status"] == "uncertain" and body["text_check"] is None
    assert body["agreement_check"] == {"rule": "final_full_stop_toggled", "agreed": False}
    assert "could not be checked reliably" in body["note"]


def test_stable_verdict_is_shown(api_settings, monkeypatch):
    claim = "There will be no classes at all levels on August 21, 2026 in observance of Ninoy Aquino Day."
    api_main = _on(api_settings, claim)
    stub = _StubCheck({claim: "SUPPORTED", claim[:-1]: "SUPPORTED"})
    monkeypatch.setattr(api_main, "verify_text", stub)
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
    assert body["status"] == "checked" and body["text_check"]["verdict"] == "SUPPORTED"
    assert body["agreement_check"] == {"rule": "final_full_stop_toggled", "agreed": True}


def test_insufficient_evidence_needs_no_agreement_check(api_settings, monkeypatch):
    api_main = _on(api_settings, Q12_CLEAN)
    stub = _StubCheck({Q12_CLEAN: "INSUFFICIENT_EVIDENCE"}, default="INSUFFICIENT_EVIDENCE")
    monkeypatch.setattr(api_main, "verify_text", stub)
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
    assert stub.calls[0] == Q12_CLEAN and Q12_CLEAN[:-1] not in stub.calls   # no agreement check on the whole text
    assert body["status"] == "checked" and body["agreement_check"] is None


# ---- reading order (Part A) -------------------------------------------------------------------------------------------
def _box(text, x, y, w, h):
    return ([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], text, 0.9)


def test_words_of_one_line_are_read_left_to_right_even_if_their_tops_differ():
    # "tuloy" starts 6 px higher (taller letters) than "hindi" on the same line: sorting by top edge put it first
    boxes = [_box("hindi", 100, 210, 120, 40), _box("tuloy", 240, 204, 130, 46), _box("Totoo ba na", 0, 208, 90, 42)]
    assert [b[1] for b in order_boxes(boxes)] == ["Totoo ba na", "hindi", "tuloy"]


def test_lines_are_read_top_to_bottom():
    boxes = [_box("second", 0, 300, 100, 40), _box("first b", 150, 102, 100, 40), _box("first a", 0, 100, 100, 40),
             _box("third", 0, 500, 100, 40)]
    assert [b[1] for b in order_boxes(boxes)] == ["first a", "first b", "second", "third"]


def test_boxes_that_barely_touch_vertically_are_separate_lines():
    # overlap 10 px < half of the smaller height (40 / 2): two lines, the higher one first
    boxes = [_box("lower", 0, 130, 100, 40), _box("upper", 200, 100, 100, 40)]
    assert [b[1] for b in order_boxes(boxes)] == ["upper", "lower"]


def test_a_box_joins_the_line_it_overlaps_most():
    boxes = [_box("a1", 0, 100, 50, 40), _box("b1", 0, 150, 50, 40), _box("x", 100, 140, 50, 40)]
    assert [b[1] for b in order_boxes(boxes)] == ["a1", "b1", "x"]   # x overlaps b1's line (30 px) more than a1's (0)


def test_reading_order_keeps_every_box_and_handles_empty_input():
    boxes = [_box(str(i), (i * 37) % 300, (i * 53) % 400, 40, 30) for i in range(25)]
    assert sorted(b[1] for b in order_boxes(boxes)) == sorted(b[1] for b in boxes)
    assert order_boxes([]) == []


# ---- Part B Stage 2: sentence-level results in /verify/image (stubbed checker) --------------------------------------------
S1, S2, S3 = "Classes resume on Monday next week.", "The fee is PHP 250 for everyone.", "The seminar is at the gym."
THREE = f"{S1} {S2} {S3}"


def _image(api_settings, monkeypatch, text, stub):
    api_main = _on(api_settings, text)
    monkeypatch.setattr(api_main, "verify_text", stub)
    with TestClient(api_main.app) as c:
        return _post_image(c)


def test_sentence_results_decide_and_the_whole_text_check_is_kept(api_settings, monkeypatch):
    stub = _StubCheck({S1: "SUPPORTED", S1[:-1]: "SUPPORTED"}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, THREE, stub).json()
    assert body["check_basis"] == "sentences"
    assert [(s["index"], s["status"], s["outcome"]) for s in body["sentence_checks"]] == [
        (0, "checked", "match"), (1, "checked", "inconclusive"), (2, "checked", "inconclusive")]
    summary = body["sentence_summary"]
    assert (summary["checked"], summary["matches"], summary["contradictions"], summary["inconclusive"]) == (3, 1, 0, 2)
    assert summary["kind"] == "partly_confirmed" and summary["label"] == "Partly confirmed: 1 of 3 checked statements match"
    assert body["text_check"]["claim"] == THREE and body["text_check"]["verdict"] == "INSUFFICIENT_EVIDENCE"   # kept
    assert body["sentence_checks"][0]["text_check"]["claim"] == S1                                      # per statement
    assert body["whole_text_agreement"] == {"whole_text_verdict": "INSUFFICIENT_EVIDENCE",
                                            "sentence_kind": "partly_confirmed", "disagrees": False}
    assert "sentence_checks" in body["timings_ms"]
    assert stub.calls[0] == THREE                                                       # whole-text check first


def test_whole_text_contradiction_never_overrides_the_sentence_results(api_settings, monkeypatch):
    stub = _StubCheck({THREE: "CONTRADICTED", THREE[:-1]: "CONTRADICTED"}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, THREE, stub).json()
    assert body["text_check"]["verdict"] == "CONTRADICTED"                             # still returned by the API
    assert body["sentence_summary"]["kind"] == "not_confirmed" and body["sentence_summary"]["contradictions"] == 0
    assert body["whole_text_agreement"]["disagrees"] is True


def test_whole_text_support_never_upgrades_the_sentence_results(api_settings, monkeypatch):
    stub = _StubCheck({THREE: "SUPPORTED", THREE[:-1]: "SUPPORTED"}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, THREE, stub).json()
    assert body["sentence_summary"]["kind"] == "not_confirmed" and body["whole_text_agreement"]["disagrees"] is True


def test_no_eligible_sentence_uses_the_whole_text_fallback(api_settings, monkeypatch):
    stub = _StubCheck({}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, "Walang pasok bukas", stub).json()
    assert body["check_basis"] == "whole_text_fallback" and body["sentence_summary"] is None
    assert [(s["status"], s["text_check"]) for s in body["sentence_checks"]] == [("not_checked_short", None)]
    assert stub.calls == ["Walang pasok bukas"] and body["text_check"]["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_no_readable_text_has_no_basis(api_settings, monkeypatch):
    stub = _StubCheck({}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, "  ", stub).json()
    assert body["check_basis"] == "none" and body["sentence_checks"] == [] and stub.calls == []


def test_anchored_contradiction_gets_the_strong_headline(api_settings, monkeypatch):
    stub = _StubCheck({S1: "SUPPORTED", S1[:-1]: "SUPPORTED", S2: "CONTRADICTED", S2[:-1]: "CONTRADICTED"},
                      default="INSUFFICIENT_EVIDENCE", posts={S1[:-1]: "N20", S2[:-1]: "N20"})
    summary = _image(api_settings, monkeypatch, THREE, stub).json()["sentence_summary"]
    assert summary["kind"] == "conflict" and summary["label"] == "A statement conflicts with an announcement"
    assert summary["anchored_posts"] == ["N20"] and summary["option_a_kind"] == "conflict"


def test_unanchored_contradiction_gets_the_soft_headline(api_settings, monkeypatch):
    stub = _StubCheck({S1: "SUPPORTED", S1[:-1]: "SUPPORTED", S2: "CONTRADICTED", S2[:-1]: "CONTRADICTED"},
                      default="INSUFFICIENT_EVIDENCE", posts={S1[:-1]: "N20", S2[:-1]: "N16"})
    body = _image(api_settings, monkeypatch, THREE, stub).json()
    summary = body["sentence_summary"]
    assert summary["kind"] == "possible_conflict"
    assert summary["label"] == "A statement may conflict with an announcement. Check the post."
    assert summary["option_a_kind"] == "conflict" and body["sentence_checks"][1]["outcome"] == "contradiction"


def test_a_busy_sentence_check_fails_the_whole_request(api_settings, monkeypatch):
    stub = _StubCheck({}, default="INSUFFICIENT_EVIDENCE", fail_on=S2)
    r = _image(api_settings, monkeypatch, THREE, stub)
    assert r.status_code == 503 and "sentence_checks" not in r.json()                  # no partial result


def test_short_segments_and_the_limit_of_twelve_are_reported(api_settings, monkeypatch):
    text = "DYCI ADVISORY. " + " ".join(f"Statement number {i} is here." for i in range(14))
    stub = _StubCheck({}, default="INSUFFICIENT_EVIDENCE")
    body = _image(api_settings, monkeypatch, text, stub).json()
    summary = body["sentence_summary"]
    assert (summary["segments"], summary["checked"], summary["not_checked_short"], summary["not_checked_limit"]) == (15, 12, 1, 2)
    assert [s["status"] for s in body["sentence_checks"]][-2:] == ["not_checked_limit", "not_checked_limit"]
