# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this project does
Classifies ~14,139 U.S. DC fast-charging (DCFC) EV stations into four **overhead-cover**
classes from top-down NAIP aerial imagery, using Gemini (Google Vertex AI) as the VLM:
`no_shade`, `shade_structure`, `shade_solar_pv`, `in_garage` (+ `uncertain` escape hatch).

Each site has a **primary** chip (48 m, charger centered) and a **context** chip (96 m), both
384×384 px PNGs. See [README.md](README.md) for onboarding and [PLAN.md](PLAN.md) for methodology/findings.

## Environment
- Always use the **`ev-charger` conda env** (Python 3.11), not `base`:
  `conda activate ev-charger` (or call `...\envs\ev-charger\python.exe` directly).
- Install deps: `pip install -r requirements.txt`.
- The **labeling app uses only the standard library** — it runs with any Python 3, no installs.

## Google Vertex AI (for model/classification only)
- Project `hai-gcp-a-model`, region `us-central1`, model `gemini-2.5-flash` (full run) /
  `gemini-2.5-pro` (fallback). Configured in `.env` (copy from `.env.example`).
- Auth = gcloud Application Default Credentials (`gcloud auth application-default login`).
  ADC is global, so it works from any env. No API key is stored.

## Common commands
```bash
# Manual labeling (no cloud needed) — opens http://127.0.0.1:8000, autosaves to gold/gold_labels.csv
python gold/label_app.py

# One-pair smoke test of the classifier
python -m src.classify

# Evaluate model vs. human gold labels (accuracy + confusion matrix + per-class P/R)
python gold/evaluate.py

# Full dataset run (concurrent, resumable), then merge to parquet
python -m src.run_batch                 # options: --limit N, --model gemini-2.5-pro
python -m src.run_batch --finalize
```

## Architecture
- `src/config.py` — paths, model, concurrency, taxonomy. `python-dotenv` is imported optionally
  so the stdlib-only labeling app can import it without deps.
- `src/prompt.py` — **versioned** system prompt + JSON `response_schema`. Bump `PROMPT_VERSION`
  on any prompt change so predictions stay traceable.
- `src/data.py` — loads `data-ev/data/final/afdc_dcfc_sites_with_imagery.csv`; image paths in the
  CSV point at the original authoring machine (`D:\Ryan\...`) and are remapped locally **by basename**.
- `src/classify.py` — Gemini Vertex call: one primary+context pair → structured JSON, with retries.
- `src/run_batch.py` — ThreadPool over the worklist; **resumable** via `outputs/predictions_checkpoint.jsonl`.
- `gold/` — `sample_gold.py` (stratified 250-site sample), `label_app.py` (manual labeler),
  `prelabel.py` (model pre-labels for review), `evaluate.py` (scoring).

## Data (not in git)
- `data-ev/` (~4 GB) is **gitignored** — obtain separately and place at repo root
  (`data-ev/data/images/{primary,context}/*.png`, `data-ev/data/final/*.csv`).
- `outputs/` and `.env` are also gitignored.

## Key conventions & gotchas
- **Judge the image CENTER** = the charger's *recorded* coordinate.
- **Geocoding offset**: the recorded coordinate sometimes lands on a nearby commercial/retail
  **building roof** instead of the stall (e.g. station 46668). Prompt v2 is hardened so a flat
  featureless commercial roof is NOT `in_garage`; `in_garage` is reserved for real parking decks.
  This is mitigated, not eliminated.
- **Trees are not structures** — vegetation-only shade is `no_shade`.
- **Gold set is the held-out test set** — never use gold-set sites as few-shot examples (leakage).
- Don't commit `data-ev/`; verify with `git status --short` before committing.
