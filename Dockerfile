# DYCI Campus Claim Check: one image with the API and the page (docs/deploy.md).
# Target: Hugging Face Spaces (Docker, CPU basic: 2 vCPU, 16 GB RAM), port 7860, user 1000. Not yet built and run on
# that host (Docker is not installed on the development machine); see docs/deploy.md, "First build".
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Manila
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only PyTorch first (the default Linux wheels pull several GB of GPU libraries), then the exact tested versions.
RUN pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.1
COPY --chown=user requirements-lock.txt .
RUN pip install --no-cache-dir --user -r requirements-lock.txt

# Models are fetched once at build time (pinned where the project pins them; OCR files checked by SHA-256), so the
# running app works offline.
COPY --chown=user src ./src
COPY --chown=user scripts/download_models.py ./scripts/download_models.py
# The frozen Option B question rewrite: the app checks this file's SHA-256 at startup (src/normalization/question_form.py).
COPY --chown=user scripts/m12_retrieval/m12_3_normalizer.py ./scripts/m12_retrieval/m12_3_normalizer.py
RUN python scripts/download_models.py --with-ocr

# Only what the app reads: the page, the served announcements, the photo index and the source map. Nothing private
# (participant messages, labelling sheets, evaluation images) is copied; .dockerignore enforces the same list.
COPY --chown=user frontend ./frontend
COPY --chown=user data/corpus/mock_facebook_corpus.jsonl data/corpus/independent_announcements.jsonl data/corpus/dev2_announcements.jsonl ./data/corpus/
COPY --chown=user data/source_map.csv ./data/source_map.csv
COPY --chown=user data/dev/sample_claims.csv ./data/dev/sample_claims.csv

ENV HF_HUB_OFFLINE=1 \
    CAMPUS_VERIFY_PUBLIC=1 \
    CAMPUS_VERIFY_SERVE_FRONTEND=1 \
    CAMPUS_VERIFY_IMAGE=1 \
    CAMPUS_VERIFY_REQUEST_LOG=1 \
    CAMPUS_VERIFY_TRUST_PROXY=1 \
    CAMPUS_VERIFY_FRAME_ANCESTORS="https://huggingface.co"

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
