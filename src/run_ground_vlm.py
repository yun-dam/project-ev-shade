"""Concurrent, resumable VLM classification over the ground-level images.

For every station that has a downloaded Google image, sends its 1-3 photos to
Gemini with the two-stage ground prompt and records the structured result. Writes
one JSON line per station to a checkpoint as it goes (crash/Ctrl-C safe, resumable).
Finalize with --finalize to merge into a results CSV joined with site metadata.

Needs Vertex ADC (gcloud auth application-default login) + project hai-gcp-a-model.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import config
from .classify_ground import classify_site

SITES_CSV = config.ROOT / "afdc_dcfc_sites_with_google_images.csv"
VLM_CHECKPOINT_JSONL = config.OUTPUT_DIR / "ground_vlm_checkpoint.jsonl"
VLM_RESULTS_CSV = config.OUTPUT_DIR / "ground_vlm_results.csv"
VLM_RESULTS_PARQUET = config.OUTPUT_DIR / "ground_vlm_results.parquet"

_write_lock = threading.Lock()


def _done_ids() -> set[int]:
    done: set[int] = set()
    if VLM_CHECKPOINT_JSONL.exists():
        with open(VLM_CHECKPOINT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("error") is None:  # only successes count as done
                        done.add(int(r["station_id"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


def _append(record: dict) -> None:
    with _write_lock, open(VLM_CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _sites_with_images() -> pd.DataFrame:
    df = pd.read_csv(SITES_CSV, low_memory=False)
    return df[df["google_has_image"] == True][  # noqa: E712
        ["station_id", "google_image_files"]
    ].copy()


def run(limit: int | None = None, model: str | None = None) -> None:
    sites = _sites_with_images()
    done = _done_ids()
    todo = [r for r in sites.itertuples(index=False) if int(r.station_id) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[vlm] sites={len(sites)} done={len(done)} todo={len(todo)} "
          f"model={model or config.MODEL}")

    def task(r):
        files = [config.GROUND_IMAGES_DIR / f
                 for f in str(r.google_image_files).split(";") if f]
        files = [f for f in files if f.exists()]
        res = classify_site(files, model=model)
        res["station_id"] = int(r.station_id)
        return res

    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futures = {ex.submit(task, r): r for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classify"):
            _append(fut.result())


def finalize() -> None:
    """Merge checkpoint into a results CSV joined with site metadata."""
    rows = []
    with open(VLM_CHECKPOINT_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    preds = pd.DataFrame(rows).drop_duplicates("station_id", keep="last")

    sites = pd.read_csv(SITES_CSV, low_memory=False)
    keep = [c for c in ("station_id", "station_name", "street_address",
                        "latitude", "longitude", "google_image_status",
                        "google_image_files") if c in sites.columns]
    merged = sites[keep].merge(preds, on="station_id", how="right")
    merged.to_parquet(VLM_RESULTS_PARQUET, index=False)
    merged.to_csv(VLM_RESULTS_CSV, index=False)
    print(f"[finalize] wrote {len(merged)} rows -> {VLM_RESULTS_CSV.name}")
    print("\n[classification]")
    print(merged["classification"].value_counts(dropna=False).to_string())
    print("\n[charger_present]")
    print(merged["charger_present"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="classify only N sites")
    ap.add_argument("--model", type=str, default=None, help="override model (e.g. gemini-2.5-pro)")
    ap.add_argument("--finalize", action="store_true", help="merge checkpoint -> CSV")
    args = ap.parse_args()

    if args.finalize:
        finalize()
    else:
        run(limit=args.limit, model=args.model)
