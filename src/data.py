"""Load site metadata and build the local image worklist.

The CSV stores image paths from the original authoring machine (D:\\Ryan\\...).
We ignore those absolute paths and resolve images locally by basename.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass
class WorkItem:
    station_id: int
    primary_path: Path
    context_path: Path


def _basename(p) -> str | None:
    if not isinstance(p, str) or not p:
        return None
    return os.path.basename(p.replace("\\", "/"))


def load_sites() -> pd.DataFrame:
    """Return sites that have a complete primary+context image pair."""
    df = pd.read_csv(config.FINAL_CSV, low_memory=False)
    df = df[df["imagery_complete"] == True].copy()  # noqa: E712
    return df


def build_worklist(df: pd.DataFrame | None = None) -> list[WorkItem]:
    """Resolve local image paths for every complete site; verify existence."""
    if df is None:
        df = load_sites()

    items: list[WorkItem] = []
    missing: list[int] = []
    for row in df.itertuples(index=False):
        pb = _basename(getattr(row, "primary_image_path"))
        cb = _basename(getattr(row, "context_image_path"))
        if not pb or not cb:
            missing.append(int(row.station_id))
            continue
        pp = config.IMAGES_PRIMARY / pb
        cp = config.IMAGES_CONTEXT / cb
        if not pp.exists() or not cp.exists():
            missing.append(int(row.station_id))
            continue
        items.append(WorkItem(int(row.station_id), pp, cp))

    if missing:
        print(f"[data] WARNING: {len(missing)} sites missing local images (skipped).")
    return items


if __name__ == "__main__":
    sites = load_sites()
    print(f"[data] complete-imagery sites in CSV: {len(sites)}")
    work = build_worklist(sites)
    print(f"[data] resolvable local image pairs:  {len(work)}")
    print(f"[data] example: {work[0]}")
