"""Central configuration: paths, model, and run parameters."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: lets the stdlib-only labeling app import config without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- Repo / data layout -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data-ev" / "data"

FINAL_CSV = DATA / "final" / "afdc_dcfc_sites_with_imagery.csv"
IMAGES_PRIMARY = DATA / "images" / "primary"
IMAGES_CONTEXT = DATA / "images" / "context"

OUTPUT_DIR = ROOT / "outputs"
GOLD_DIR = ROOT / "gold"
OUTPUT_DIR.mkdir(exist_ok=True)
GOLD_DIR.mkdir(exist_ok=True)

PREDICTIONS_PARQUET = OUTPUT_DIR / "predictions.parquet"
CHECKPOINT_JSONL = OUTPUT_DIR / "predictions_checkpoint.jsonl"

# --- Vertex AI / model ------------------------------------------------------
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "hai-gcp-a-model")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-pro")

# --- Run parameters ---------------------------------------------------------
CONCURRENCY = int(os.getenv("CONCURRENCY", "12"))
MAX_RETRIES = 4

# --- Taxonomy ---------------------------------------------------------------
CLASSES = ["no_shade", "shade_structure", "shade_solar_pv", "in_garage", "uncertain"]
