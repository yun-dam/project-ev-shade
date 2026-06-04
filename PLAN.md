# EV Charger Shade Classification — Project Plan

Classify every U.S. DC fast-charging (DCFC) site into one of four overhead-cover
classes from top-down NAIP aerial imagery, using a vision-language model (Gemini on Vertex AI).

## Classes
| Class | Definition | NAIP signature |
|---|---|---|
| `no_shade` | Open parking lot, exposed to sky | Bare pavement, painted stalls, cars visible from above |
| `shade_structure` | Solid man-made canopy / carport | Uniform gray/white roof over stalls, clean straight shadow, no panel texture |
| `shade_solar_pv` | Solar PV canopy over stalls | Dark blue/black fine-grid panel texture |
| `in_garage` | Inside a parking structure / deck | Multi-level deck (cars/ramps on roof) or enclosed garage |
| `uncertain` | (escape hatch) | Ambiguous, blur, off-center, commercial roof with unclear lot |

## Data
- **14,139** sites with complete `primary` (48 m, 384px) + `context` (96 m, 384px) image pairs.
- Source: NAIP aerial orthoimagery, mostly 2022–2023, top-down ~0.6 m native.
- Metadata: `data-ev/data/final/afdc_dcfc_sites_with_imagery.csv` (44 cols: id, lat/lon,
  network, ports, power, NAIP year, image paths). Image paths in the CSV point to the original
  authoring machine (`D:\Ryan\...`) and are remapped locally by basename in `src/data.py`.

## Tech stack
- **Model:** `gemini-2.5-flash` (full run) / `gemini-2.5-pro` (low-confidence fallback).
- **Backend:** Vertex AI, project `hai-gcp-a-model`, region `us-central1`, ADC auth (no API key stored).
- **Both images** sent per site (primary for detail, context to confirm structure extent).
- Structured JSON output: `{classification, confidence, evidence}` enforced via `response_schema`.

## Pipeline (code in `src/`)
- `config.py` — paths, model, concurrency, taxonomy.
- `prompt.py` — versioned system prompt + response schema (currently **v2**).
- `data.py` — load CSV, remap image paths, build/verify worklist. ✅ all 14,139 pairs resolve.
- `classify.py` — Gemini client, per-pair call with retries/backoff. ✅ verified.
- `run_batch.py` — concurrent (ThreadPool), **resumable** (JSONL checkpoint), `--finalize`
  merges to `outputs/predictions.parquet` + `.csv` with metadata. ✅ verified on 8 sites.

## Gold-set validation (code in `gold/`)
- `sample_gold.py` — stratified-by-network sample of 250 → `gold_sample.csv` + HTML contact sheet. ✅
- Human labels go in `gold/gold_labels.csv` (`station_id,label`).
- `evaluate.py` — runs current model on gold set → overall accuracy + confusion matrix +
  per-class precision/recall. Target ≥85% overall; watch garage/solar confusion.
- **Top-up for rare classes:** solar PV and garages are sparse; after a cheap screening pass
  we add model-flagged candidates of rare classes to the gold set so per-class recall is measurable.

## Key finding — geocoding offset (handled in prompt v2)
AFDC coordinates sometimes land on a **nearby commercial/retail building roof** rather than the
actual charging stall (e.g. site 46668: center on a store roof, charger really in the lot in front).
A naive prompt over-labels these `in_garage`. **v2** instructs the model that a flat, featureless
commercial roof is NOT a garage, reserves `in_garage` for true parking decks, and routes the visible
adjacent lot to `no_shade`/`uncertain`. Verified: the three retail-roof test cases flipped from
`in_garage` → `no_shade`. The gold set will quantify residual error.

## Cost & throughput
- ~1.1k input + ~150 output tokens/site → ~16M/2M tokens over 14k sites ≈ **$10–15 with Flash**.
- ~12 concurrent requests → full run in a couple hours.

## Status / next steps
1. ✅ Setup, data prep, classifier, batch runner, gold tooling — all built and verified.
2. ⏳ **Label the gold set** (250 sites) — owner TBD (analyst hand-label vs. model-prelabel + review).
3. ⏳ Evaluate v2 on gold, iterate prompt to target accuracy.
4. ⏳ Full 14k run → `predictions.parquet`.
5. ⏳ QA: confidence histogram, route low-confidence to Pro, class-distribution report
   (per state / per network), HTML review contact sheet.

## Limitations
- Ground-floor / enclosed garages are physically invisible from top-down → expect lower `in_garage` recall.
- Tree shade vs. built canopy handled in prompt but a known confusion source.
- Geocoding offset (above) is mitigated, not eliminated.
