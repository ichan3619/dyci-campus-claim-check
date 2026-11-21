# DYCI Campus Claim Check

**Check a campus rumor against the school's own announcements before passing it on.**

DYCI Campus Claim Check is a web app for students of Dr. Yanga's Colleges, Inc. (Bocaue, Bulacan, Philippines). A
student types something they heard ("no classes tomorrow?"), pastes a link, or uploads a picture of a poster. The app
finds the matching DYCI announcement and answers:

| Answer | Meaning |
|---|---|
| **Supported** | A DYCI announcement backs it. The announcement is shown, with its date and link. |
| **Contradicted** | A DYCI announcement says otherwise. |
| **Not confirmed** | DYCI's announcements do not settle it. This does not mean the claim is false. |

Pictures get a **level from 1 (conflicts with DYCI) to 5 (matches DYCI)**, with the reasons, and the file's Content
Credentials when it has them.

The app never stamps anything as true. Every answer names the announcement it used, so a student can check it in one
click, and when the announcements do not settle a claim it says so instead of guessing.

> Built as an undergraduate thesis project. It reads DYCI's public Facebook announcements and is not an official DYCI
> service.

---

## How it works

```
 message / link / picture
          │
          ▼
 ┌─────────────────┐   picture → EasyOCR text reader → statements
 │  Normalize      │   dates ("tomorrow" → a real date), Tagalog → English (local translator),
 │                 │   lookalike letters, class wording
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Retrieve       │   E5 multilingual embeddings find the closest DYCI announcements
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Judge (NLI)    │   a frozen DeBERTa-v3 NLI model: does the announcement support or contradict it?
 └────────┬────────┘
          ▼
 ┌─────────────────┐   dates and holidays, times and values, offices and venues, stated reasons,
 │  Rules          │   same-day advisories, department pages, picture levels; each one behind its own switch
 └────────┬────────┘
          ▼
   answer + the announcement used + notes on why
```

**Why not a chatbot?** A chatbot can invent an answer and cannot always show why it decided. Here the verdict comes
from a fixed NLI model and written rules, and every answer points to its source. Typed instructions ("ignore previous
instructions…") are just part of the text being checked.

## Features

- **Text, link and picture checks**, with answers in English. Messages may be in English, Tagalog or Taglish.
- **Picture levels 1-5**: the text read from a poster or screenshot is split into statements, each statement is
  checked, and the level sums them up. A poster whose date was moved by up to 7 days is flagged as possibly edited.
- **Content Credentials (C2PA)**: a picture's signed origin record is read and shown, for example "changed after it
  was signed". It is display only and never changes a verdict.
- **Cautious by design**: a verdict is withheld when values conflict, when Tagalog cannot be read safely, or when a
  picture's reading is fragile.
- **Live sources**: DYCI's Facebook announcements are fetched on a schedule. Department and organisation pages
  (for example ACES) can confirm their own matters, always labelled as theirs.
- **Accounts**: Google sign-in limited to the school's domain, an idle time-out, and an optional "My checks" history.
- **Admin pages**: sources and fetch status, and a dashboard of counts only.

## Tech stack

| Area | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic |
| Retrieval | `intfloat/multilingual-e5-small` (sentence-transformers) |
| Verdicts | `cross-encoder/nli-deberta-v3-small` (English), `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Tagalog/Taglish) |
| Translation | SEA-LION v3 8B (GGUF) run locally with llama.cpp: optional, nothing leaves the machine |
| Pictures | EasyOCR, Pillow, c2pa-python |
| Front end | Plain HTML, CSS and JavaScript (no framework, no build step) |
| Storage | SQLite |
| Tests | pytest |

## Getting started

Requirements: Python 3.12 and about 1.5 GB of disk space for the models. A GPU is not needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
python scripts/download_models.py --with-ocr      # once: about 1.2 GB of models

# serve the API and the page on http://127.0.0.1:8000
# (PowerShell: $env:CAMPUS_VERIFY_SERVE_FRONTEND = "1")
CAMPUS_VERIFY_SERVE_FRONTEND=1 python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

A `Dockerfile` is included for container hosting (CPU only, port 7860).

### Sample data

The announcements in `data/corpus/` are **made-up samples**. Each is marked "SAMPLE" and dated 2030, so the app runs
and the tests pass without republishing the school's posts. The deployed version reads DYCI's own announcements from
its Facebook pages. To use real posts, put them in the same JSON Lines format (one object per line with
`document_id`, `text`, `source_type`, `source_url` and `published_date`) and point `CAMPUS_VERIFY_CORPUS` at the
files.

### Configuration

Every behaviour beyond the base checker sits behind a `CAMPUS_VERIFY_*` environment variable, **off by default**.
[`config/public.env.example`](config/public.env.example) lists the settings of the deployed version, including the
picture check, the rules and the live sources. Sign-in needs your own Google OAuth client ID.

## How it was tested

Each rule went through the same steps before it could be switched on:

1. A written plan with pass rules, committed first.
2. A test set written by a separate writer who never saw the code, committed before any code.
3. The rule's code behind its own switch, off by default.
4. Exactly one scored run against the pre-set pass rules.

Rules that failed stayed off, and their reports were kept.

| Result | Value |
|---|---|
| Macro F1, NLI vs baselines (development set) | **0.500**, vs 0.390 (TF-IDF + logistic regression) and 0.212 (majority class) |
| Red-team: 60 attacks (prompt injection, pasted posts plus a false line, lookalike letters, odd input) | 46 as expected, 6 cautious, 8 flagged; prompt injection 5 of 5 resisted |
| A false fact asked six different ways | Supported in 0 of 24 wordings |
| Automated tests | **2,031 passing** in this repository (2,295 in the full private suite) |

### Running the tests

```bash
pytest -q
```

2,031 tests pass in about a minute (the models must be downloaded first). The full suite has 2,295 tests; the rest
read the evaluation datasets (real student messages and labelling sheets), which are private and not published here.
Those test files are left out, and 18 single tests are deselected in `pytest.ini` with the reason.

## Known limitations

- **Copied wording can carry a changed headline.** A poster that keeps an announcement's long text but changes its
  short headline can still match. Two rules built to catch this failed their tests and stay off.
- **Short claims can pick the wrong announcement**, which can give a wrong "Contradicted".
- **"Do not…" instructions** can be misread as a conflict by the NLI model.
- **Tagalog/Taglish** gets no verdict unless the local translator can read it safely.
- **Pixel-level AI edits are not detected.** The app checks what a picture *says*, and shows Content Credentials when
  present.

## Project structure

```
src/
  api/            FastAPI app, settings, sign-in, history, sources
  retrieval/      E5 retriever, passage selection
  classification/ NLI classifiers (English, multilingual)
  verification/   verifier, date check and the rules
  normalization/  dates, Tagalog, class wording, input clean-up
  image/          OCR, statements, picture levels, Content Credentials
  ingestion/      Facebook and website readers
  routing/        claim router (claim / question / Tagalog)
frontend/         the page: index.html, css/, js/
data/corpus/      made-up SAMPLE announcements (replace with your own posts)
scripts/          model download
tests/            pytest suite
```

## License

Copyright © 2026. All rights reserved. The source is published for viewing only; no permission is granted to copy,
modify or redistribute it. See [LICENSE](LICENSE).
