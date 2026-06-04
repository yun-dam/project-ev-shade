"""Pre-label the gold sample with the current model for human review.

Runs the v2 classifier on every site in gold/gold_sample.csv and writes:
  gold/gold_labels.csv       - station_id, pred, confidence, evidence, label (pre-filled = pred)
  gold/gold_review.html      - both images + predicted label/confidence/evidence per site

REVIEW WORKFLOW: open gold_review.html, scan each cell, and FIX the `label` column in
gold_labels.csv wherever the prediction is wrong. Then run `python gold/evaluate.py`.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.classify import classify_pair  # noqa: E402

CLASS_COLOR = {
    "no_shade": "#4a90d9", "shade_structure": "#e0a030", "shade_solar_pv": "#7b4fc0",
    "in_garage": "#cc5555", "uncertain": "#888", "error": "#000",
}


def main() -> None:
    sample = pd.read_csv(config.GOLD_DIR / "gold_sample.csv")

    def task(row):
        r = classify_pair(config.ROOT / row.primary_path, config.ROOT / row.context_path)
        return {
            "station_id": row.station_id, "ev_network": row.ev_network, "state": row.state,
            "primary_path": row.primary_path, "context_path": row.context_path,
            "pred": r["classification"], "confidence": round(r["confidence"], 2),
            "evidence": r["evidence"],
        }

    results = []
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futs = [ex.submit(task, row) for row in sample.itertuples(index=False)]
        for f in tqdm(as_completed(futs), total=len(futs), desc="prelabel"):
            results.append(f.result())

    df = pd.DataFrame(results).sort_values("station_id")
    df["label"] = df["pred"]  # pre-fill; human corrects wrong ones
    cols = ["station_id", "ev_network", "state", "pred", "confidence", "label",
            "evidence", "primary_path", "context_path"]
    df[cols].to_csv(config.GOLD_DIR / "gold_labels.csv", index=False)
    _review_html(df)
    print(f"\n[prelabel] wrote gold/gold_labels.csv + gold/gold_review.html ({len(df)} sites)")
    print(df["pred"].value_counts().to_string())


def _review_html(df: pd.DataFrame) -> None:
    cells = []
    for r in df.itertuples(index=False):
        p = (config.ROOT / r.primary_path).as_uri()
        c = (config.ROOT / r.context_path).as_uri()
        col = CLASS_COLOR.get(r.pred, "#888")
        cells.append(f"""
        <div class="cell">
          <div class="hd">{r.station_id} &mdash; {r.ev_network} ({r.state})</div>
          <img src="{p}"><img src="{c}">
          <div class="pred" style="background:{col}">{r.pred} &middot; {r.confidence}</div>
          <div class="ev">{r.evidence}</div>
        </div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>Gold review</title>
    <style>body{{font-family:sans-serif;background:#1b1b1b;color:#eee;margin:12px}}
    .grid{{display:flex;flex-wrap:wrap;gap:10px}}
    .cell{{background:#2b2b2b;padding:6px;border-radius:6px;width:404px}}
    .cell img{{width:196px;height:196px;image-rendering:pixelated;vertical-align:top}}
    .hd{{font-size:12px;color:#bbb;margin-bottom:4px}}
    .pred{{font-size:13px;font-weight:bold;color:#fff;padding:3px 6px;border-radius:4px;margin-top:4px;display:inline-block}}
    .ev{{font-size:11px;color:#ccc;margin-top:4px;line-height:1.3}}</style>
    <h2>Gold review &mdash; left=primary(48m), right=context(96m). Fix wrong labels in gold_labels.csv.</h2>
    <div class=grid>{''.join(cells)}</div>"""
    (config.GOLD_DIR / "gold_review.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
