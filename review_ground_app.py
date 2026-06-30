"""Visual review of VLM predictions on the ground-level images.

Shows every classified station as a card: its 1-3 Google photos + the predicted
class (color-coded), confidence, charger_present, and the model's evidence. You
can filter by predicted class and mark each prediction correct (v) or wrong (x)
to get a quick visual accuracy estimate. Verdicts autosave to ground_vlm_review.csv.

Run (from the repo root) — needs only Python 3:
    python review_ground_app.py
Opens http://127.0.0.1:8002.
"""
from __future__ import annotations

import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "outputs" / "ground_vlm_results.csv"
IMAGES_DIR = ROOT / "outputs" / "ground_images"
REVIEW_CSV = ROOT / "ground_vlm_review.csv"
PORT = 8002
CLASSES = ["no_shade", "shade_structure", "shade_solar_pv", "in_garage", "uncertain"]

_lock = threading.Lock()
ITEMS: list[dict] = []
REVIEW: dict[str, str] = {}   # station_id -> "correct" | "wrong"


def _load() -> None:
    global ITEMS, REVIEW
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing {RESULTS_CSV}. Run `python -m src.run_ground_vlm` first.")
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            files = [x for x in (r.get("google_image_files", "") or "").split(";") if x]
            ITEMS.append({
                "station_id": str(r["station_id"]),
                "station_name": r.get("station_name", ""),
                "files": files,
                "charger_present": r.get("charger_present", ""),
                "classification": r.get("classification", ""),
                "confidence": r.get("confidence", ""),
                "evidence": r.get("evidence", ""),
            })
    if REVIEW_CSV.exists():
        with open(REVIEW_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("verdict"):
                    REVIEW[str(r["station_id"])] = r["verdict"]
        print(f"[review] resumed {len(REVIEW)} verdicts.")


def _save() -> None:
    tmp = REVIEW_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "classification", "verdict"])
        for it in ITEMS:
            sid = it["station_id"]
            if sid in REVIEW:
                w.writerow([sid, it["classification"], REVIEW[sid]])
    tmp.replace(REVIEW_CSV)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            data = {"classes": CLASSES, "items": [
                {**{k: it[k] for k in ("station_id", "station_name", "charger_present",
                                       "classification", "confidence", "evidence")},
                 "n": len(it["files"]), "verdict": REVIEW.get(it["station_id"], "")}
                for it in ITEMS]}
            self._send(200, json.dumps(data).encode("utf-8"))
        elif u.path == "/img":
            q = parse_qs(u.query)
            sid, idx = q.get("sid", [""])[0], int(q.get("idx", ["0"])[0])
            it = next((x for x in ITEMS if x["station_id"] == sid), None)
            if not it or idx < 0 or idx >= len(it["files"]):
                self._send(404, b"{}"); return
            p = IMAGES_DIR / it["files"][idx]
            if not p.exists():
                self._send(404, b"{}"); return
            self._send(200, p.read_bytes(), "image/jpeg")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/review":
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n) or b"{}")
            sid, verdict = str(p.get("station_id", "")), p.get("verdict", "")
            with _lock:
                if verdict in ("correct", "wrong"):
                    REVIEW[sid] = verdict
                else:
                    REVIEW.pop(sid, None)
                _save()
            self._send(200, json.dumps({"ok": True, "n": len(REVIEW)}).encode())
        else:
            self._send(404, b"{}")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>VLM prediction review</title>
<style>
 body{font-family:system-ui,sans-serif;background:#161616;color:#eee;margin:0}
 #bar{position:sticky;top:0;z-index:5;background:#0d0d0d;padding:10px 16px;display:flex;
   flex-wrap:wrap;align-items:center;gap:8px;font-size:13px}
 .flt{padding:6px 10px;border:none;border-radius:6px;cursor:pointer;color:#fff;font-weight:600;font-size:12px;opacity:.55}
 .flt.on{opacity:1;outline:2px solid #fff}
 .c_no_shade{background:#4a90d9}.c_shade_structure{background:#e0a030}
 .c_shade_solar_pv{background:#7b4fc0}.c_in_garage{background:#cc5555}.c_uncertain{background:#777}
 #stat{margin-left:auto;font-size:13px;color:#ccc}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:16px}
 .card{background:#1f1f1f;border:2px solid #333;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
 .card.correct{border-color:#3a3}.card.wrong{border-color:#d44}
 .imgs{display:flex;gap:2px;background:#000;height:170px}
 .imgs img{flex:1;height:100%;object-fit:cover;cursor:zoom-in;min-width:0}
 .body{padding:8px 10px}
 .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700;color:#fff}
 .meta{font-size:11px;color:#999;margin:4px 0}
 .ev{font-size:12px;color:#cbd;line-height:1.35;margin-top:4px}
 .acts{display:flex;gap:6px;margin-top:8px}
 .acts button{flex:1;border:none;border-radius:6px;padding:7px;cursor:pointer;font-weight:700;font-size:13px}
 .ok{background:#2c5}.no{background:#c44;color:#fff}
 .acts button.sel{outline:3px solid #fff}
 kbd{background:#333;border-radius:3px;padding:0 5px}
</style></head><body>
<div id=bar>
  <b>VLM prediction review</b>
  <span id=filters></span>
  <span id=stat></span>
</div>
<div id=grid></div>
<script>
let S=null, filter='all';
async function load(){ S=await (await fetch('/api/state')).json(); renderFilters(); render(); }
function counts(){ const c={all:S.items.length}; S.classes.forEach(k=>c[k]=0);
  S.items.forEach(it=>c[it.classification]=(c[it.classification]||0)+1); return c; }
function renderFilters(){ const d=document.getElementById('filters'); const c=counts(); d.innerHTML='';
  const mk=(k,label)=>{ const b=document.createElement('button');
    b.className='flt '+(k==='all'?'':'c_'+k)+(filter===k?' on':''); if(k==='all')b.style.background='#445';
    b.textContent=label+' ('+(c[k]||0)+')'; b.onclick=()=>{filter=k;renderFilters();render();}; d.appendChild(b); };
  mk('all','ALL'); S.classes.forEach(k=>mk(k,k));
}
function render(){ const g=document.getElementById('grid'); g.innerHTML='';
  const items=S.items.filter(it=>filter==='all'||it.classification===filter);
  items.forEach(it=>{ const card=document.createElement('div');
    card.className='card'+(it.verdict?(' '+it.verdict):'');
    let imgs=''; for(let k=0;k<it.n;k++) imgs+=`<img src="/img?sid=${it.station_id}&idx=${k}" onclick="window.open(this.src,'_blank')">`;
    const cls=it.classification;
    card.innerHTML=`<div class=imgs>${imgs}</div><div class=body>
      <span class="badge c_${cls}">${cls}</span>
      <span class=meta>conf ${it.confidence} &middot; charger ${it.charger_present} &middot; #${it.station_id} &middot; ${it.n} img</span>
      <div class=ev>${(it.evidence||'').replace(/</g,'&lt;')}</div>
      <div class=acts>
        <button class="ok ${it.verdict==='correct'?'sel':''}" onclick="mark('${it.station_id}','correct')">&#10003; correct</button>
        <button class="no ${it.verdict==='wrong'?'sel':''}" onclick="mark('${it.station_id}','wrong')">&#10007; wrong</button>
      </div></div>`;
    g.appendChild(card); });
  stat();
}
function stat(){ const r=S.items.filter(x=>x.verdict); const ok=r.filter(x=>x.verdict==='correct').length;
  const acc=r.length? (100*ok/r.length).toFixed(1)+'%':'-';
  document.getElementById('stat').textContent=`reviewed ${r.length}/${S.items.length} | correct ${ok} | agreement ${acc}`;
}
async function mark(sid,v){ const it=S.items.find(x=>x.station_id===sid);
  it.verdict = (it.verdict===v)?'':v;
  await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({station_id:sid,verdict:it.verdict})});
  render();
}
load();
</script></body></html>"""


def main() -> None:
    _load()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[review] {len(ITEMS)} predictions loaded. Open {url}")
    print(f"[review] verdicts autosave to {REVIEW_CSV.name}. Ctrl-C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] stopped. Verdicts saved.")


if __name__ == "__main__":
    main()
