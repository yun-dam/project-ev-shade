"""Manual labeling app for the gold set (label it yourself, blank slate).

Run (from the repo root, env activated):
    python gold/label_app.py

Opens a local web page (http://127.0.0.1:8000). For each site you see the primary
(48 m) and context (96 m) image. Pick a class with the number keys or buttons:

    1 = no_shade        2 = shade_structure   3 = shade_solar_pv
    4 = in_garage       5 = uncertain
    <- / -> = previous / next      u = jump to next unlabeled

Every choice autosaves to gold/gold_labels.csv immediately, so you can stop and
re-run anytime to resume. No model predictions are shown (unbiased labeling).
"""
from __future__ import annotations

import csv
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SAMPLE_CSV = config.GOLD_DIR / "gold_sample.csv"
LABELS_CSV = config.GOLD_DIR / "gold_labels.csv"
PRELABEL_BACKUP = config.GOLD_DIR / "gold_prelabels.csv"
PORT = 8000
CLASSES = ["no_shade", "shade_structure", "shade_solar_pv", "in_garage", "uncertain"]

_lock = threading.Lock()
ITEMS: list[dict] = []          # ordered worklist from gold_sample.csv
LABELS: dict[str, str] = {}     # station_id -> label


def _load() -> None:
    """Load the worklist and any existing manual labels (for resume)."""
    global ITEMS, LABELS
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        ITEMS = list(csv.DictReader(f))

    # If gold_labels.csv is actually the model PRE-labels, preserve it once and start fresh.
    if LABELS_CSV.exists():
        with open(LABELS_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        is_prelabel = bool(rows) and "pred" in rows[0]
        if is_prelabel and not PRELABEL_BACKUP.exists():
            LABELS_CSV.replace(PRELABEL_BACKUP)
            print(f"[label] preserved model pre-labels -> {PRELABEL_BACKUP.name}; starting blank.")
        elif not is_prelabel:
            for r in rows:
                if r.get("label"):
                    LABELS[str(r["station_id"])] = r["label"]
            print(f"[label] resumed {len(LABELS)} existing manual labels.")


def _save() -> None:
    """Rewrite gold_labels.csv from the in-memory state (atomic-ish)."""
    tmp = LABELS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "ev_network", "state", "label", "primary_path", "context_path"])
        for it in ITEMS:
            sid = str(it["station_id"])
            w.writerow([sid, it.get("ev_network", ""), it.get("state", ""),
                        LABELS.get(sid, ""), it["primary_path"], it["context_path"]])
    tmp.replace(LABELS_CSV)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
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
            data = {
                "classes": CLASSES,
                "items": [{"station_id": str(it["station_id"]),
                           "ev_network": it.get("ev_network", ""),
                           "state": it.get("state", ""),
                           "label": LABELS.get(str(it["station_id"]), "")} for it in ITEMS],
            }
            self._send(200, json.dumps(data).encode("utf-8"))
        elif u.path == "/img":
            q = parse_qs(u.query)
            sid, kind = q.get("sid", [""])[0], q.get("kind", ["primary"])[0]
            it = next((x for x in ITEMS if str(x["station_id"]) == sid), None)
            if not it:
                self._send(404, b"{}")
                return
            rel = it["primary_path"] if kind == "primary" else it["context_path"]
            img = (config.ROOT / rel).read_bytes()
            self._send(200, img, "image/png")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/label":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            sid, label = str(payload.get("station_id", "")), payload.get("label", "")
            with _lock:
                if label in CLASSES:
                    LABELS[sid] = label
                elif label == "":
                    LABELS.pop(sid, None)
                _save()
            self._send(200, json.dumps({"ok": True, "labeled": len(LABELS)}).encode())
        else:
            self._send(404, b"{}")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Gold labeler</title>
<style>
 body{font-family:system-ui,sans-serif;background:#161616;color:#eee;margin:0;height:100vh;
   display:flex;flex-direction:column;align-items:center}
 #bar{width:100%;background:#0d0d0d;padding:8px 16px;box-sizing:border-box;display:flex;
   align-items:center;gap:16px;font-size:14px}
 #prog{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden}
 #progfill{height:100%;background:#4a90d9;width:0}
 #imgs{display:flex;gap:16px;margin:16px}
 .imw{text-align:center}
 .imw img{width:420px;height:420px;image-rendering:pixelated;background:#000;border:2px solid #333}
 .cap{font-size:13px;color:#aaa;margin-top:4px}
 #btns{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:4px}
 .cb{padding:12px 18px;font-size:15px;border:none;border-radius:8px;cursor:pointer;color:#fff;font-weight:600}
 .c0{background:#4a90d9}.c1{background:#e0a030}.c2{background:#7b4fc0}.c3{background:#cc5555}.c4{background:#777}
 .cb.sel{outline:4px solid #fff}
 #nav{display:flex;gap:14px;align-items:center;margin:10px;font-size:14px}
 button.nb{background:#333;color:#eee;border:none;padding:8px 14px;border-radius:6px;cursor:pointer}
 #meta{font-size:14px;color:#ccc}
 kbd{background:#333;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body>
<div id=bar>
  <b>EV charger overhead-cover labeling</b>
  <span id=meta></span>
  <div id=prog><div id=progfill></div></div>
  <span id=count></span>
</div>
<div id=imgs>
  <div class=imw><img id=primary><div class=cap>PRIMARY &mdash; 48 m (charger at center)</div></div>
  <div class=imw><img id=context><div class=cap>CONTEXT &mdash; 96 m</div></div>
</div>
<div id=btns></div>
<div id=nav>
  <button class=nb id=prev>&larr; Prev</button>
  <button class=nb id=next>Next &rarr;</button>
  <button class=nb id=unl>Next unlabeled (u)</button>
  <span style="color:#888">keys: 1-5 = class &nbsp; &larr;/&rarr; = nav &nbsp; u = next unlabeled</span>
</div>
<script>
let S=null, i=0;
const LABELHINT={no_shade:"1",shade_structure:"2",shade_solar_pv:"3",in_garage:"4",uncertain:"5"};
async function load(){ S=await (await fetch('/api/state')).json();
  // start at first unlabeled
  const u=S.items.findIndex(x=>!x.label); i=u<0?0:u; renderBtns(); render(); }
function renderBtns(){ const d=document.getElementById('btns'); d.innerHTML='';
  S.classes.forEach((c,k)=>{ const b=document.createElement('button');
    b.className='cb c'+k; b.id='b_'+c; b.textContent=(k+1)+'. '+c;
    b.onclick=()=>setLabel(c); d.appendChild(b); }); }
function render(){ const it=S.items[i];
  document.getElementById('primary').src='/img?kind=primary&sid='+it.station_id+'&_='+i;
  document.getElementById('context').src='/img?kind=context&sid='+it.station_id+'&_='+i;
  document.getElementById('meta').textContent=`#${i+1}/${S.items.length}  |  station ${it.station_id}  |  ${it.ev_network} (${it.state})`;
  const done=S.items.filter(x=>x.label).length;
  document.getElementById('count').textContent=done+' / '+S.items.length+' labeled';
  document.getElementById('progfill').style.width=(100*done/S.items.length)+'%';
  S.classes.forEach(c=>document.getElementById('b_'+c).classList.toggle('sel', it.label===c));
}
async function setLabel(c){ const it=S.items[i];
  it.label = (it.label===c)? '' : c;  // click again to clear
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({station_id:it.station_id,label:it.label})});
  render(); if(it.label){ setTimeout(()=>{ if(i<S.items.length-1){i++;render();} },120); }
}
function go(d){ i=Math.max(0,Math.min(S.items.length-1,i+d)); render(); }
function nextUnlabeled(){ const n=S.items.findIndex((x,k)=>k>i&&!x.label);
  i = n<0 ? (S.items.findIndex(x=>!x.label)) : n; if(i<0)i=S.items.length-1; render(); }
document.getElementById('prev').onclick=()=>go(-1);
document.getElementById('next').onclick=()=>go(1);
document.getElementById('unl').onclick=nextUnlabeled;
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='5'){ setLabel(S.classes[+e.key-1]); }
  else if(e.key==='ArrowLeft'){ go(-1); }
  else if(e.key==='ArrowRight'){ go(1); }
  else if(e.key==='u'||e.key==='U'){ nextUnlabeled(); }
});
load();
</script></body></html>"""


def main() -> None:
    _load()
    _save()  # ensure file exists with the full worklist
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[label] {len(ITEMS)} sites loaded. Open {url}")
    print("[label] autosaves to gold/gold_labels.csv after every choice. Ctrl-C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[label] stopped. Progress saved.")


if __name__ == "__main__":
    main()
