# EV Charger Shade Classification

Classify U.S. DC fast-charging (DCFC) stations into four **overhead-cover** classes from
top-down NAIP aerial imagery, using a vision-language model (Gemini on Google Vertex AI).

| Class | Meaning |
|---|---|
| `no_shade` | Open parking lot, exposed to sky |
| `shade_structure` | Covered by a solid man-made canopy / carport |
| `shade_solar_pv` | Covered by a solar-panel canopy |
| `in_garage` | Inside a parking structure / deck |
| `uncertain` | Ambiguous (escape hatch — not a real category) |

Each site has a **primary** chip (48 m, charger centered) and a **context** chip (96 m),
both 384×384 px PNGs.

> See [PLAN.md](PLAN.md) for the full project plan, methodology, and findings.

---

## Two ways to use this repo

- **Labeling** (build the gold-standard ground-truth set) → needs only **Python 3.x** + the image data. **No Google Cloud, no pip installs.** Jump to [Labeling](#labeling-no-cloud-needed).
- **Classification / evaluation** (run the model) → needs the Python packages + **Google Vertex AI access**. See [Running the model](#running-the-model-needs-vertex-ai).

---

## 1. Get the code and data

```bash
# place the project somewhere, then from its root:
cd ev-charger
```

The imagery + metadata live under `data-ev/` (~2 GB) and are **not** included in version
control (see `.gitignore`). Get the `data-ev/` folder from the project owner and place it at
the repo root so the structure is:

```
ev-charger/
├── data-ev/data/images/primary/*.png
├── data-ev/data/images/context/*.png
└── data-ev/data/final/afdc_dcfc_sites_with_imagery.csv
```

Paths in `gold/gold_sample.csv` are **relative to the repo root**, so as long as `data-ev/`
sits here, everything resolves on any machine.

---

## Labeling (no cloud needed)

This is all a labeler needs. Requires only a Python 3 install — uses the standard library.

```bash
python gold/label_app.py
```

It opens **http://127.0.0.1:8000**. For each of the 250 gold sites you see the primary (48 m)
and context (96 m) images. Choose a class:

| Key | Class | &nbsp; | Key | Action |
|---|---|---|---|---|
| `1` | no_shade | | `←` / `→` | prev / next |
| `2` | shade_structure | | `u` | next unlabeled |
| `3` | shade_solar_pv | | | |
| `4` | in_garage | | | |
| `5` | uncertain | | | |

- Every choice **autosaves** to `gold/gold_labels.csv` and advances.
- Click the same class again to clear it.
- **Resumable** — stop (Ctrl-C) and re-run anytime; it resumes at your first unlabeled site.
- No model predictions are shown, so labels stay unbiased.

**Labeling tips**
- Judge the **center** of the images (that's the charger's recorded location).
- *Trees are not structures* — a spot shaded only by vegetation is `no_shade`.
- Solar PV = dark-blue fine **grid** texture; plain canopy = smooth uniform roof.
- The recorded coordinate occasionally lands on an adjacent **building roof** instead of the
  stall. If the center is a roof but the real charger lot is obviously beside it, use
  `uncertain`. (See the geocoding note in [PLAN.md](PLAN.md).)

---

## Running the model (needs Vertex AI)

### Setup

```bash
# 1. Create and activate an environment (conda shown; venv works too)
conda create -n ev-charger python=3.11 -y
conda activate ev-charger

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Vertex AI project settings
cp .env.example .env          # Windows: copy .env.example .env
#   then edit .env to set your GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION

# 4. Authenticate (one time). Uses Application Default Credentials — no API key stored.
gcloud auth application-default login
#   You need access to a GCP project with the Vertex AI API enabled.
```

### Smoke test (one image pair)

```bash
python -m src.classify        # should print a JSON classification
```

### Evaluate against the gold set

After `gold/gold_labels.csv` has human labels:

```bash
python gold/evaluate.py       # accuracy + confusion matrix + per-class precision/recall
```

### Full run (all ~14k sites)

```bash
python -m src.run_batch                 # concurrent, resumable, checkpoints as it goes
python -m src.run_batch --finalize      # merge checkpoint -> outputs/predictions.parquet (+ .csv)
# options: --limit N (test on N sites), --model gemini-2.5-pro
```

The full run costs roughly **$10–15** with `gemini-2.5-flash` and is **resumable** — re-running
skips sites already in `outputs/predictions_checkpoint.jsonl`.

---

## Repository layout

```
ev-charger/
├── README.md                 # this file
├── PLAN.md                   # full plan, methodology, findings
├── requirements.txt          # pinned deps (model side only; labeling needs none)
├── .env.example              # Vertex AI project settings template (no secrets)
├── src/
│   ├── config.py             # paths, model, concurrency, taxonomy
│   ├── prompt.py             # versioned prompt + JSON output schema
│   ├── data.py               # load metadata, resolve local image paths
│   ├── classify.py           # Gemini (Vertex) call: one pair -> structured JSON
│   └── run_batch.py          # concurrent, resumable full-dataset runner
├── gold/
│   ├── sample_gold.py        # build the stratified 250-site gold sample
│   ├── label_app.py          # ← manual labeling web app (stdlib only)
│   ├── evaluate.py           # score model vs. human labels
│   ├── gold_sample.csv       # the 250-site worklist (committed)
│   └── gold_labels.csv       # human labels (filled by label_app.py)
└── data-ev/                  # imagery + metadata (NOT in repo; obtain separately)
```

## Notes / known limitations

- **ADC auth is global** (gcloud), so it works from any Python env once you've logged in.
- Ground-floor / enclosed garages are largely invisible from top-down → expect lower
  `in_garage` recall.
- The recorded charger coordinate is sometimes offset onto a nearby building roof; the model
  prompt (`src/prompt.py`, v2) is hardened against this, but it is mitigated, not eliminated.
