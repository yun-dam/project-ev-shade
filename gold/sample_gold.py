"""Draw a stratified gold-set sample for human labeling.

Strategy: stratified random sample across ev_network (so the major operators are
represented), with a fixed seed for reproducibility. Rare classes (solar PV,
garages) are sparse, so use `screen_gold.py` afterwards to top up the sample with
model-screened candidates of under-represented classes before labeling.

Outputs:
  gold/gold_sample.csv      - station_id + image paths + blank `label` column to fill
  gold/gold_contactsheet.html - both images per site, side by side, for fast labeling
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data import build_worklist, load_sites  # noqa: E402

N = 250
SEED = 42


def main() -> None:
    sites = load_sites()
    work = {w.station_id: w for w in build_worklist(sites)}
    sites = sites[sites["station_id"].isin(work)].copy()

    # Stratified by network: proportional allocation, min 1 per group.
    sites["_net"] = sites["ev_network"].fillna("Unknown")
    frac = N / len(sites)
    sample = (
        sites.groupby("_net", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED), include_groups=False)
        .sample(frac=1, random_state=SEED)
    )
    sample = sample.head(N)

    rows = []
    for r in sample.itertuples(index=False):
        w = work[r.station_id]
        rows.append({
            "station_id": r.station_id,
            "station_name": r.station_name,
            "state": r.state,
            "ev_network": r.ev_network,
            "primary_path": str(w.primary_path.relative_to(config.ROOT)),
            "context_path": str(w.context_path.relative_to(config.ROOT)),
            "label": "",  # human fills: no_shade | shade_structure | shade_solar_pv | in_garage | uncertain
        })
    out = pd.DataFrame(rows)
    out.to_csv(config.GOLD_DIR / "gold_sample.csv", index=False)
    _write_contactsheet(out)
    print(f"[gold] wrote {len(out)} sites -> gold/gold_sample.csv + gold/gold_contactsheet.html")


def _write_contactsheet(df: pd.DataFrame) -> None:
    cells = []
    for r in df.itertuples(index=False):
        p = (config.ROOT / r.primary_path).as_uri()
        c = (config.ROOT / r.context_path).as_uri()
        cells.append(f"""
        <div class="cell">
          <div class="hd">{r.station_id} &mdash; {r.ev_network} ({r.state})</div>
          <img src="{p}"><img src="{c}">
          <div class="lbl">label: <b>{r.label or '____'}</b></div>
        </div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>Gold contact sheet</title>
    <style>body{{font-family:sans-serif;background:#222;color:#eee}}
    .grid{{display:flex;flex-wrap:wrap;gap:10px}}
    .cell{{background:#333;padding:6px;border-radius:6px;width:404px}}
    .cell img{{width:196px;height:196px;image-rendering:pixelated}}
    .hd{{font-size:12px;margin-bottom:4px}} .lbl{{font-size:12px;margin-top:4px}}</style>
    <h2>Gold set &mdash; left = primary (48m), right = context (96m)</h2>
    <div class=grid>{''.join(cells)}</div>"""
    (config.GOLD_DIR / "gold_contactsheet.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
