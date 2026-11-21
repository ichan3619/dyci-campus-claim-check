"""
Download the models the API uses by default, once, so later runs work offline.

    python scripts/download_models.py

  intfloat/multilingual-e5-small            retrieval (Hugging Face cache)
  cross-encoder/nli-deberta-v3-small        verdicts for English claims (Hugging Face cache)
  MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 @ b5113eb3   verdicts for Tagalog/Taglish claims,
      pinned to the revision evaluated in M12.3, saved as plain files in src/api/pipeline.DEFAULT_TAGALOG_MODEL_DIR
      (no symlinks, which Windows refuses without developer mode). The API picks it up automatically from there.

About 1.2 GB in total. Already-downloaded models are skipped.

    python scripts/download_models.py --with-ocr

also downloads the two EasyOCR files for Image Check (CAMPUS_VERIFY_IMAGE) into ~/.EasyOCR/model and checks each against
the SHA-256 in src/image/ocr.MODEL_FILES; a mismatch stops with an error (the API refuses such files anyway).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pipeline import DEFAULT_TAGALOG_MODEL_DIR  # noqa: E402

TAGALOG_REPO = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TAGALOG_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"


def main() -> int:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.classification.nli_classifier import DEFAULT_NLI_MODEL_NAME
    from src.retrieval.e5_retriever import DEFAULT_E5_MODEL_NAME

    print(f"[1/3] {DEFAULT_E5_MODEL_NAME}"); SentenceTransformer(DEFAULT_E5_MODEL_NAME)
    print(f"[2/3] {DEFAULT_NLI_MODEL_NAME}")
    AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL_NAME); AutoModelForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL_NAME)
    print(f"[3/3] {TAGALOG_REPO} @ {TAGALOG_REVISION[:8]} -> {DEFAULT_TAGALOG_MODEL_DIR}")
    snapshot_download(TAGALOG_REPO, revision=TAGALOG_REVISION, local_dir=DEFAULT_TAGALOG_MODEL_DIR,
                      allow_patterns=["*.json", "model.safetensors", "*.model"])
    if "--with-ocr" in sys.argv:
        import hashlib
        import easyocr
        from src.image.ocr import LANGUAGES, MODEL_DIR, MODEL_FILES
        print(f"[4/4] EasyOCR {', '.join(LANGUAGES)} -> {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        easyocr.Reader(LANGUAGES, gpu=False, verbose=False, download_enabled=True, model_storage_directory=MODEL_DIR)
        for name, expected in MODEL_FILES.items():
            with open(os.path.join(MODEL_DIR, name), "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != expected:
                    raise SystemExit(f"SHA-256 mismatch for {name}: refusing it")
        print("      SHA-256 checked:", ", ".join(MODEL_FILES))
    print("done. Start the API with: uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
