"""Augment place-photo sites with multiple candidate photos (1..N).

The first Place photo sometimes shows a storefront/sign rather than the EV
charger itself. For every site already classified ``ok_place_photo`` in the
ground manifest, this re-fetches the place's photo list via Place Details (New)
using the stored ``place_id`` (no proximity re-search) and downloads up to
``config.PHOTO_MAX_COUNT`` photos named ``{station_id}_place_photo_{i}.jpg``.

This gives the downstream VLM several candidates per site. Resumable via its own
checkpoint; re-running skips sites already done.

Needs GOOGLE_MAPS_API_KEY (Place Details (New) + Place Photos enabled).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from . import config
from .retrieve_ground import download_place_photo, place_photo_names
from .run_ground import GROUND_CHECKPOINT_JSONL, finalize

EXTRA_CHECKPOINT_JSONL = config.OUTPUT_DIR / "extra_photos_checkpoint.jsonl"
GROUND_MANIFEST_CSV = config.OUTPUT_DIR / "ground_manifest.csv"

# Place Details + 3 photo fetches per site bursts the Places API rate limit, so
# keep this well below the classifier's concurrency and back off hard on 429.
EXTRA_CONCURRENCY = int(os.getenv("EXTRA_CONCURRENCY", "5"))

_write_lock = threading.Lock()


def _retry(fn, max_retries: int = 6):
    """Call ``fn`` with exponential backoff, honoring Retry-After on HTTP 429."""
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as e:
            last = e
            resp = e.response
            if resp is not None and resp.status_code == 429:
                wait = resp.headers.get("Retry-After")
                time.sleep(float(wait) if wait else min(2**attempt, 30))
            elif resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise  # non-throttle client error: don't retry
            else:
                time.sleep(min(2**attempt, 30))
        except requests.RequestException as e:
            last = e
            time.sleep(min(2**attempt, 30))
    raise last


def _done_ids() -> set[int]:
    """Only SUCCESSFUL sites count as done, so prior failures get retried."""
    done: set[int] = set()
    if EXTRA_CHECKPOINT_JSONL.exists():
        with open(EXTRA_CHECKPOINT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("n_photos", 0) > 0:
                        done.add(int(r["station_id"]))
                except Exception:  # noqa: BLE001
                    continue
    return done


def _append(record: dict) -> None:
    with _write_lock, open(EXTRA_CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _photo_sites() -> pd.DataFrame:
    """Sites that previously yielded a place photo (have a usable place_id)."""
    df = pd.read_csv(GROUND_MANIFEST_CSV)
    return df[(df["status"] == "ok_place_photo") & df["place_id"].notna()][
        ["station_id", "place_id"]
    ].copy()


def fetch_one(station_id: int, place_id: str, max_count: int) -> dict:
    """Download up to ``max_count`` photos for one place. Returns a record."""
    out_dir = config.GROUND_IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        names = _retry(lambda: place_photo_names(place_id))[:max_count]
        paths = []
        for i, name in enumerate(names, start=1):
            dest = out_dir / f"{station_id}_place_photo_{i}.jpg"
            _retry(lambda n=name, d=dest: download_place_photo(n, d))
            paths.append(dest.name)
        return {
            "station_id": int(station_id),
            "n_photos": len(paths),
            "photo_files": paths,
            "error": None if paths else "Place Details returned no photos",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "station_id": int(station_id),
            "n_photos": 0,
            "photo_files": [],
            "error": str(e)[:200],
        }


def run(limit: int | None = None, max_count: int | None = None) -> None:
    if not config.MAPS_API_KEY:
        raise SystemExit("GOOGLE_MAPS_API_KEY is not set (see .env.example).")
    if not GROUND_CHECKPOINT_JSONL.exists():
        raise SystemExit("Run `python -m src.run_ground` first to build the manifest.")

    max_count = config.PHOTO_MAX_COUNT if max_count is None else max_count
    sites = _photo_sites()
    done = _done_ids()
    todo = [r for r in sites.itertuples(index=False) if int(r.station_id) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[extra] place_photo sites={len(sites)} done={len(done)} "
          f"todo={len(todo)} max_count={max_count} workers={EXTRA_CONCURRENCY}")

    with ThreadPoolExecutor(max_workers=EXTRA_CONCURRENCY) as ex:
        futures = {
            ex.submit(fetch_one, int(r.station_id), r.place_id, max_count): r
            for r in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="photos"):
            _append(fut.result())


def merge_into_manifest() -> None:
    """Add n_photos / photo_files columns to the ground manifest."""
    rows = []
    with open(EXTRA_CHECKPOINT_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    extra = pd.DataFrame(rows).drop_duplicates("station_id", keep="last")
    extra["photo_files"] = extra["photo_files"].apply(lambda xs: ";".join(xs))

    finalize()  # rebuild base manifest first
    man = pd.read_csv(GROUND_MANIFEST_CSV)
    man = man.merge(extra[["station_id", "n_photos", "photo_files"]],
                    on="station_id", how="left")
    man.to_parquet(config.OUTPUT_DIR / "ground_manifest.parquet", index=False)
    man.to_csv(GROUND_MANIFEST_CSV, index=False)
    n = int((man["n_photos"].fillna(0) > 1).sum())
    print(f"[merge] manifest updated; {n} sites now have >1 photo")
    print(man["n_photos"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N sites (testing)")
    ap.add_argument("--max-count", type=int, default=None, help="photos per site (default cfg)")
    ap.add_argument("--merge", action="store_true", help="merge checkpoint -> manifest")
    args = ap.parse_args()

    if args.merge:
        merge_into_manifest()
    else:
        run(limit=args.limit, max_count=args.max_count)
