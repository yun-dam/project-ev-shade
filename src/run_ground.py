"""Concurrent, resumable batch ground-image retrieval over the dataset.

For every site it runs the dual-source pipeline in ``retrieve_ground`` and writes
one JSON line per completed site to a checkpoint, so a crash or Ctrl-C never loses
work and re-running skips sites already done. Finalize with --finalize to merge the
checkpoint into a parquet/CSV manifest (joined with site metadata).

Needs GOOGLE_MAPS_API_KEY (see .env.example) — separate from the Vertex AI ADC.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from . import config
from .data import load_sites
from .retrieve_ground import retrieve_with_retries

GROUND_CHECKPOINT_JSONL = config.OUTPUT_DIR / "ground_checkpoint.jsonl"
GROUND_MANIFEST_PARQUET = config.OUTPUT_DIR / "ground_manifest.parquet"

_write_lock = threading.Lock()


def _done_ids() -> set[int]:
    done: set[int] = set()
    if GROUND_CHECKPOINT_JSONL.exists():
        with open(GROUND_CHECKPOINT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["station_id"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


def _append(record: dict) -> None:
    with _write_lock, open(GROUND_CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run(limit: int | None = None) -> None:
    if not config.MAPS_API_KEY:
        raise SystemExit(
            "GOOGLE_MAPS_API_KEY is not set. Add it to .env (see .env.example) and "
            "enable Places API (New) + Street View Static API on the key."
        )

    sites = load_sites()[["station_id", "latitude", "longitude"]].dropna()
    done = _done_ids()
    todo = [r for r in sites.itertuples(index=False) if int(r.station_id) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[ground] total={len(sites)} done={len(done)} todo={len(todo)}")

    def task(r):
        res = retrieve_with_retries(
            float(r.latitude), float(r.longitude), int(r.station_id)
        )
        return asdict(res)

    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futures = {ex.submit(task, r): r for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="retrieve"):
            _append(fut.result())


def finalize() -> None:
    """Merge the checkpoint into a manifest joined with site metadata.

    Left-joins onto the FULL site list so every site appears exactly once, even
    sites not yet retrieved. Adds explicit, unambiguous flags for the downstream
    VLM step:
      - ``charger_found``: a nearby EV charger was matched at the coordinate.
      - ``has_image``: a ground-level image was actually downloaded.
      - ``status``: human-readable outcome (see below).
    """
    rows = []
    with open(GROUND_CHECKPOINT_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    man = pd.DataFrame(rows).drop_duplicates("station_id", keep="last")

    sites = load_sites()
    keep = [c for c in ("station_id", "station_name", "street_address",
                        "latitude", "longitude") if c in sites.columns]
    # Left-join onto ALL sites so unprocessed sites are visibly blank, not missing.
    merged = sites[keep].merge(man, on="station_id", how="left")

    # --- Explicit, non-confusing flags ------------------------------------
    merged["charger_found"] = merged["place_id"].notna()
    merged["has_image"] = merged["image_path"].notna()

    def _status(r) -> str:
        if pd.isna(r["source"]):
            return "not_processed"          # never run through retrieval
        if r["has_image"]:
            return f"ok_{r['source']}"       # ok_place_photo / ok_street_view
        if r["charger_found"]:
            return "charger_no_image"        # charger exists, no photo & no street view
        return "no_charger_found"            # nearby search returned nothing

    merged["status"] = merged.apply(_status, axis=1)

    # Clear column order for human + downstream consumption.
    col_order = [
        "station_id", "station_name", "street_address", "latitude", "longitude",
        "status", "charger_found", "has_image", "source", "image_path",
        "display_name", "resolved_lat", "resolved_lon", "place_id", "error",
    ]
    merged = merged[[c for c in col_order if c in merged.columns]]

    merged.to_parquet(GROUND_MANIFEST_PARQUET, index=False)
    merged.to_csv(config.OUTPUT_DIR / "ground_manifest.csv", index=False)
    print(f"[finalize] wrote {len(merged)} rows -> {GROUND_MANIFEST_PARQUET}")
    print("\n[status breakdown]")
    print(merged["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="retrieve only N sites (testing)")
    ap.add_argument("--finalize", action="store_true", help="merge checkpoint -> parquet")
    args = ap.parse_args()

    if args.finalize:
        finalize()
    else:
        run(limit=args.limit)
