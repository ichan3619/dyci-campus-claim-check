"""
Image Check Phase 2B P1, provenance facts (src/image/provenance.py; results/round2/ai_image_plan_draft.md). Every file
is made here: a throw-away test certificate signs synthetic pictures, so no real or downloaded image is used.
"""
import datetime
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import c2pa
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient
from PIL import Image

from src.image import provenance as pv
from tests.test_image_check import _on, _post_image

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


@pytest.fixture(scope="module")
def signer_pems():
    now = datetime.datetime.now(datetime.timezone.utc)

    def name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                          x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org")])

    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name("Test Root")).issuer_name(name("Test Root"))
          .public_key(ca_key.public_key()).serial_number(1).not_valid_before(now - datetime.timedelta(days=1))
          .not_valid_after(now + datetime.timedelta(days=30))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(name("Test Signer")).issuer_name(ca.subject)
            .public_key(key.public_key()).serial_number(2).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    chain = leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM)
    return chain, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())


def _jpeg(color=(120, 30, 200), **save):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "JPEG", **save)
    return buf.getvalue()


def _sign(pems, source_type, data=None, remote_url=None):
    manifest = {"claim_generator_info": [{"name": "TestGen", "version": "1.0"}], "title": "t.jpg",
                "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [
                    {"action": "c2pa.created", "digitalSourceType": IPTC + source_type}]}}]}
    builder = c2pa.Builder(manifest)
    if remote_url:
        builder.set_no_embed()
        builder.set_remote_url(remote_url)
    out = io.BytesIO()
    builder.sign(c2pa.Signer.from_info(c2pa.C2paSignerInfo(b"es256", pems[0], pems[1], None)), "image/jpeg",
                 io.BytesIO(data or _jpeg()), out)
    return out.getvalue()


def test_ai_made_content_credentials_are_reported(signer_pems):
    facts = pv.check(_sign(signer_pems, "trainedAlgorithmicMedia"), "image/jpeg")
    cc = facts["content_credentials"]
    assert (cc["status"], cc["ai"], cc["generator"], cc["signed_by"]) == ("signed", "made", "TestGen 1.0", "Test Org")
    assert cc["trust_list_checked"] is False
    assert "saying it was made with an AI tool (TestGen 1.0), signed by “Test Org”" in facts["summary"]


def test_ai_edited_and_camera_records(signer_pems):
    assert pv.check(_sign(signer_pems, "compositeWithTrainedAlgorithmicMedia"), "image/jpeg")[
        "content_credentials"]["ai"] == "edited"
    facts = pv.check(_sign(signer_pems, "digitalCapture"), "image/jpeg")
    assert facts["content_credentials"]["ai"] is None
    assert "not checked against a list of trusted signers" in facts["summary"] and "AI" not in facts["summary"]


def test_a_picture_changed_after_signing_is_invalid(signer_pems):
    signed = bytearray(_sign(signer_pems, "digitalCapture"))
    signed[-20] ^= 0xFF
    facts = pv.check(bytes(signed), "image/jpeg")
    assert facts["content_credentials"]["status"] == "invalid" and "changed after it was signed" in facts["summary"]


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_no_record_is_normal_and_never_called_fake_or_real(fmt, mime):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, fmt)
    facts = pv.check(buf.getvalue(), mime)
    assert facts["content_credentials"] == {"status": "none"} and facts["summary"] == pv.NONE_FOUND
    assert "says nothing about whether the picture is real" in facts["summary"]
    assert not re.search(r"(?i)\b(fake|authentic|genuine)\b", facts["summary"] + facts["note"])


def test_a_remote_manifest_is_never_fetched(signer_pems):
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/manifest.c2pa"
        image = _sign(signer_pems, "digitalCapture", remote_url=url)
        with pytest.raises(c2pa.C2paError):                  # the test can see a fetch: default settings do fetch
            c2pa.Reader("image/jpeg", io.BytesIO(image), context=c2pa.Context())
        assert len(hits) == 1
        facts = pv.check(image, "image/jpeg")
    finally:
        server.shutdown()
    assert len(hits) == 1, "provenance.check contacted the address in the picture"
    assert facts["content_credentials"]["status"] == "remote_not_fetched" and "not fetched" in facts["summary"]


def test_camera_details_never_include_gps_serials_or_names():
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0131] = "Canon", "EOS R6", "Lightroom"
    exif[0x013B], exif[0x8298] = "Juan Dela Cruz", "Copyright Juan Dela Cruz"           # Artist, Copyright
    exif.get_ifd(0x8769)[0x9003] = "2026:10:01 09:30:00"
    exif.get_ifd(0x8769)[0xA431] = "SERIAL123456"                                       # BodySerialNumber
    exif.get_ifd(0x8825)[1], exif.get_ifd(0x8825)[2] = "N", (14.0, 48.0, 0.0)           # GPS
    facts = pv.check(_jpeg(exif=exif), "image/jpeg")
    assert facts["camera"] == {"make": "Canon", "model": "EOS R6", "software": "Lightroom",
                               "taken": "2026:10:01 09:30:00"}
    out = json.dumps(facts)
    assert not any(s in out for s in ("Juan", "SERIAL", "14.0", "48.0"))
    assert "Camera details in the file: camera Canon EOS R6, software Lightroom, date 2026:10:01 09:30:00." \
        in facts["summary"]


def test_the_unsigned_file_label_is_reported_as_unsigned():
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           b'<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
           b'</rdf:RDF></x:xmpmeta>')
    facts = pv.check(_jpeg(xmp=xmp), "image/jpeg")
    assert facts["file_label"] == {"source_type": "trainedAlgorithmicMedia", "ai": "made"}
    assert "The file's own label says it was made with AI. This label is not signed" in facts["summary"]


def test_unreadable_input_never_raises():
    facts = pv.check(b"\xff\xd8\xff" + b"\x00" * 50, "image/jpeg")
    assert facts["content_credentials"]["status"] in ("none", "unreadable") and facts["camera"] is None


def test_fields_are_clipped_and_cleaned():
    assert pv._clip("a\x00b\nc" + "x" * 200) == ("a b c" + "x" * 200)[:pv.MAX_FIELD]
    assert pv._clip("   ") is None and pv._clip(None) is None


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(pv.ENV, raising=False)
    assert pv.enabled() is False


# ---- the image route ------------------------------------------------------------------------------------------------
def _route(api_settings, monkeypatch, data, on):
    monkeypatch.setenv(pv.ENV, "1" if on else "0")
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        return _post_image(c, data=data, ctype="image/jpeg").json()


def test_the_route_adds_facts_and_changes_nothing_else(api_settings, monkeypatch, signer_pems):
    image = _sign(signer_pems, "trainedAlgorithmicMedia")
    off, on = _route(api_settings, monkeypatch, image, False), _route(api_settings, monkeypatch, image, True)
    assert off["provenance"] is None and on["provenance"]["content_credentials"]["ai"] == "made"
    assert "provenance" in on["timings_ms"] and "provenance" not in off["timings_ms"]
    for body in (off, on):
        body.pop("timings_ms")
    on.pop("provenance"), off.pop("provenance")
    assert on == off                                             # verdicts, statements and level are untouched


def test_the_page_shows_the_summary_and_never_sets_a_verdict():
    import os
    app = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "js", "app.js"),
               encoding="utf-8").read()
    block = app[app.index("// Provenance facts (Phase 2B P1)"):app.index("// Legitimacy level (results/round2")]
    assert "p.summary" in block and "p.note" in block and "textContent" not in block and "innerHTML" not in block
    assert not re.search(r"verdict\s*=(?!=)|level\s*=(?!=)", block)
