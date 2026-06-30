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

# --- Google Maps Platform (ground-level image retrieval) --------------------
# SEPARATE from the Vertex AI ADC above: this is an API key for the Places API
# (New) + Street View Static API. Enable both on the key in the Cloud console.
MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

GROUND_IMAGES_DIR = OUTPUT_DIR / "ground_images"

# Nearby Search: strict proximity so we only match the charger at the coordinate.
NEARBY_RADIUS_M = float(os.getenv("NEARBY_RADIUS_M", "20.0"))
# Place Photos: cap resolution to control bandwidth/storage.
PHOTO_MAX_PX = int(os.getenv("PHOTO_MAX_PX", "1600"))
# Place Photos: how many photos to pull per site (1st may not show the charger,
# so grabbing 2-3 gives the VLM more candidates).
PHOTO_MAX_COUNT = int(os.getenv("PHOTO_MAX_COUNT", "3"))
# Street View Static: ambient roadside capture parameters.
STREETVIEW_SIZE = os.getenv("STREETVIEW_SIZE", "640x480")
STREETVIEW_FOV = int(os.getenv("STREETVIEW_FOV", "90"))

# --- Taxonomy ---------------------------------------------------------------
CLASSES = ["no_shade", "shade_structure", "shade_solar_pv", "in_garage", "uncertain"]
