"""Concurrent, resumable batch classifier over the full worklist.

Writes one JSON line per completed site to a checkpoint file as it goes, so a crash
or Ctrl-C never loses work. Re-running skips sites already in the checkpoint.
Finalize with --finalize to merge the checkpoint into predictions.parquet (joined
with site metadata).
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import config
from .classify import classify_pair
from .data import build_worklist, load_sites

_write_lock = threading.Lock()


def _done_ids() -> set[int]:
    done: set[int] = set()
    if config.CHECKPOINT_JSONL.exists():
        with open(config.CHECKPOINT_JSONL, encoding="utf-8") as f:
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
    with _write_lock, open(config.CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run(limit: int | None = None, model: str | None = None) -> None:
    work = build_worklist()
    done = _done_ids()
    todo = [w for w in work if w.station_id not in done]
    if limit:
        todo = todo[:limit]
    print(f"[run] total={len(work)} done={len(done)} todo={len(todo)} model={model or config.MODEL}")

    def task(item):
        res = classify_pair(item.primary_path, item.context_path, model=model)
        res["station_id"] = item.station_id
        return res

    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futures = {ex.submit(task, w): w for w in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classify"):
            _append(fut.result())


def finalize() -> None:
    """Merge checkpoint JSONL with site metadata into predictions.parquet."""
    rows = []
    with open(config.CHECKPOINT_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    preds = pd.DataFrame(rows).drop_duplicates("station_id", keep="last")

    sites = load_sites()
    keep = [
        "station_id", "station_name", "state", "city", "latitude", "longitude",
        "ev_network", "facility_type", "primary_naip_year",
    ]
    merged = preds.merge(sites[keep], on="station_id", how="left")
    merged.to_parquet(config.PREDICTIONS_PARQUET, index=False)
    merged.to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)
    print(f"[finalize] wrote {len(merged)} rows -> {config.PREDICTIONS_PARQUET}")
    print(merged["classification"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="classify only N sites (testing)")
    ap.add_argument("--model", type=str, default=None, help="override model")
    ap.add_argument("--finalize", action="store_true", help="merge checkpoint -> parquet")
    args = ap.parse_args()

    if args.finalize:
        finalize()
    else:
        run(limit=args.limit, model=args.model)
