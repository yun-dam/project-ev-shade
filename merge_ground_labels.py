"""Map manual ground-image labels onto the master site CSV.

Reads the labels produced by ``label_ground_app.py`` (ground_labels.csv) and
left-joins them onto the master site table by ``station_id``, adding a
``human_label`` column. Every site is kept; sites not yet labeled get a blank.

Run (from the repo root):
    python merge_ground_labels.py

Output: afdc_dcfc_sites_with_labels.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def merge(master_csv: Path, labels_csv: Path, out_csv: Path) -> None:
    master = pd.read_csv(master_csv, low_memory=False)
    if "station_id" not in master.columns:
        raise SystemExit(f"{master_csv.name} has no station_id column.")

    if not labels_csv.exists():
        raise SystemExit(
            f"{labels_csv.name} not found. Have your colleague run "
            f"`python label_ground_app.py` and label some sites first."
        )

    lab = pd.read_csv(labels_csv)
    lab = lab[lab["label"].notna() & (lab["label"].astype(str).str.strip() != "")]
    lab = (lab[["station_id", "label"]]
           .rename(columns={"label": "human_label"})
           .drop_duplicates("station_id", keep="last"))

    out = master.merge(lab, on="station_id", how="left")
    out.to_csv(out_csv, index=False)

    n_label = int(out["human_label"].notna().sum())
    print(f"[merge] master sites: {len(master)}  |  labeled: {n_label}")
    print(f"[merge] wrote {out_csv.name}")
    if n_label:
        print("\n[human_label distribution]")
        print(out["human_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=ROOT / "afdc_dcfc_sites_with_google_images.csv")
    ap.add_argument("--labels", type=Path, default=ROOT / "ground_labels.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "afdc_dcfc_sites_with_labels.csv")
    args = ap.parse_args()
    merge(args.master, args.labels, args.out)
