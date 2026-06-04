"""Evaluate model predictions against the human-labeled gold set.

Reads gold/gold_labels.csv (station_id,label), runs the current model on those
sites (or reuses predictions if present), and reports overall accuracy plus a
per-class confusion matrix and precision/recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.classify import classify_pair  # noqa: E402
from src.data import build_worklist  # noqa: E402

LABELS_CSV = config.GOLD_DIR / "gold_labels.csv"


def main() -> None:
    gold = pd.read_csv(LABELS_CSV)
    gold = gold[gold["label"].notna() & (gold["label"] != "")]
    work = {w.station_id: w for w in build_worklist()}

    preds = []
    for sid in gold["station_id"]:
        w = work[int(sid)]
        r = classify_pair(w.primary_path, w.context_path)
        preds.append({"station_id": int(sid), "pred": r["classification"], "conf": r["confidence"]})
    pred_df = pd.DataFrame(preds)

    m = gold.merge(pred_df, on="station_id")
    acc = (m["label"] == m["pred"]).mean()
    print(f"\nOverall accuracy: {acc:.1%}  (n={len(m)}, model={config.MODEL}, prompt={__import__('src.prompt', fromlist=['PROMPT_VERSION']).PROMPT_VERSION})\n")

    print("Confusion matrix (rows=truth, cols=pred):")
    cm = pd.crosstab(m["label"], m["pred"], dropna=False)
    print(cm.to_string(), "\n")

    for cls in config.CLASSES:
        tp = ((m["label"] == cls) & (m["pred"] == cls)).sum()
        fp = ((m["label"] != cls) & (m["pred"] == cls)).sum()
        fn = ((m["label"] == cls) & (m["pred"] != cls)).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {cls:16s} precision={prec:.2f} recall={rec:.2f} (support={tp+fn})")

    m.to_csv(config.GOLD_DIR / "gold_eval_detail.csv", index=False)
    print("\nWrote gold/gold_eval_detail.csv")


if __name__ == "__main__":
    main()
