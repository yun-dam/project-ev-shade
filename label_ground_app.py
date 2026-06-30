"""Manual labeling app for the Google ground-level images.

Label every station that has a downloaded Google image (user place-photos or a
Street View capture) into the project's overhead-cover classes. All available
images for a station are shown side by side, so the charger is visible even when
it is not in the first photo.

Run (from the repo root) — needs ONLY Python 3, no packages to install:

    python label_ground_app.py

Opens http://127.0.0.1:8001. Pick a class with the number keys or buttons:

    1 = no_shade        2 = shade_structure   3 = shade_solar_pv
    4 = in_garage       5 = uncertain
    <- / -> = previous / next       u = jump to next unlabeled
    click an image     = open full size in a new tab

Every choice autosaves to ground_labels.csv immediately, so you can stop and
re-run anytime to resume.
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
SITES_CSV = ROOT / "afdc_dcfc_sites_with_google_images.csv"
IMAGES_DIR = ROOT / "outputs" / "ground_images"
LABELS_CSV = ROOT / "ground_labels.csv"
PORT = 8001
CLASSES = ["no_shade", "shade_structure", "shade_solar_pv", "in_garage", "uncertain"]

_lock = threading.Lock()
ITEMS: list[dict] = []          # worklist: sites that have >=1 Google image
LABELS: dict[str, str] = {}     # station_id -> label


def _load() -> None:
    """Build the worklist from sites with images; resume existing labels."""
    global ITEMS, LABELS
    if not SITES_CSV.exists():
        raise SystemExit(f"Missing {SITES_CSV.name}. Generate it first.")

    with open(SITES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("google_has_image", "")).strip().lower() != "true":
                continue
            files = [x for x in (r.get("google_image_files", "") or "").split(";") if x]
            if not files:
                continue
            ITEMS.append({
                "station_id": str(r["station_id"]),
                "station_name": r.get("station_name", ""),
                "street_address": r.get("street_address", ""),
                "status": r.get("google_image_status", ""),
                "files": files,
            })

    if LABELS_CSV.exists():
        with open(LABELS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("label"):
                    LABELS[str(r["station_id"])] = r["label"]
        print(f"[label] resumed {len(LABELS)} existing labels.")


def _save() -> None:
    """Rewrite ground_labels.csv from in-memory state (atomic-ish)."""
    tmp = LABELS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "station_name", "status", "label", "image_files"])
        for it in ITEMS:
            sid = it["station_id"]
            w.writerow([sid, it["station_name"], it["status"],
                        LABELS.get(sid, ""), ";".join(it["files"])])
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
                "items": [{"station_id": it["station_id"],
                           "station_name": it["station_name"],
                           "street_address": it["street_address"],
                           "status": it["status"],
                           "n": len(it["files"]),
                           "label": LABELS.get(it["station_id"], "")} for it in ITEMS],
            }
            self._send(200, json.dumps(data).encode("utf-8"))
        elif u.path == "/img":
            q = parse_qs(u.query)
            sid = q.get("sid", [""])[0]
            idx = int(q.get("idx", ["0"])[0])
            it = next((x for x in ITEMS if x["station_id"] == sid), None)
            if not it or idx < 0 or idx >= len(it["files"]):
                self._send(404, b"{}")
                return
            path = IMAGES_DIR / it["files"][idx]
            if not path.exists():
                self._send(404, b"{}")
                return
            self._send(200, path.read_bytes(), "image/jpeg")
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


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Ground-image labeler</title>
<style>
 body{font-family:system-ui,sans-serif;background:#161616;color:#eee;margin:0;min-height:100vh;
   display:flex;flex-direction:column;align-items:center}
 #bar{width:100%;background:#0d0d0d;padding:8px 16px;box-sizing:border-box;display:flex;
   align-items:center;gap:16px;font-size:14px;position:sticky;top:0;z-index:5}
 #prog{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden}
 #progfill{height:100%;background:#4a90d9;width:0}
 #imgs{display:flex;gap:14px;margin:16px;flex-wrap:wrap;justify-content:center}
 .imw{text-align:center}
 .imw img{max-width:440px;max-height:460px;background:#000;border:2px solid #333;cursor:zoom-in;
   object-fit:contain}
 .cap{font-size:12px;color:#aaa;margin-top:4px}
 #btns{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:4px}
 .cb{padding:12px 18px;font-size:15px;border:none;border-radius:8px;cursor:pointer;color:#fff;font-weight:600}
 .c0{background:#4a90d9}.c1{background:#e0a030}.c2{background:#7b4fc0}.c3{background:#cc5555}.c4{background:#777}
 .cb.sel{outline:4px solid #fff}
 #nav{display:flex;gap:14px;align-items:center;margin:10px;font-size:14px}
 button.nb{background:#333;color:#eee;border:none;padding:8px 14px;border-radius:6px;cursor:pointer}
 #meta{font-size:14px;color:#ccc}
 #lookup{background:#222;border:1px solid #444;color:#eee;border-radius:6px;padding:6px 8px;
   font-size:13px;width:120px}
 #lookupmsg{font-size:12px;color:#e0a030}
 #sub{font-size:12px;color:#888;margin:-6px 0 4px}
 .tag{font-size:11px;padding:2px 7px;border-radius:10px;background:#283}
 .tag.sv{background:#356}
 kbd{background:#333;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body>
<div id=bar>
  <b>EV charger &mdash; ground image labeling</b>
  <input id=lookup placeholder="station_id" autocomplete=off>
  <button class=nb id=golookup>Go (id)</button>
  <span id=lookupmsg></span>
  <span id=meta></span>
  <div id=prog><div id=progfill></div></div>
  <span id=count></span>
</div>
<div id=sub></div>
<div id=imgs></div>
<div id=btns></div>
<div id=nav>
  <button class=nb id=prev>&larr; Prev</button>
  <button class=nb id=next>Next &rarr;</button>
  <button class=nb id=unl>Next unlabeled (u)</button>
  <span style="color:#888">keys: <kbd>1</kbd>-<kbd>5</kbd> class &nbsp; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> nav &nbsp; <kbd>u</kbd> next unlabeled</span>
</div>
<script>
let S=null, i=0;
async function load(){ S=await (await fetch('/api/state')).json();
  const u=S.items.findIndex(x=>!x.label); i=u<0?0:u; renderBtns(); render(); }
function renderBtns(){ const d=document.getElementById('btns'); d.innerHTML='';
  S.classes.forEach((c,k)=>{ const b=document.createElement('button');
    b.className='cb c'+k; b.id='b_'+c; b.textContent=(k+1)+'. '+c;
    b.onclick=()=>setLabel(c); d.appendChild(b); }); }
function render(){ const it=S.items[i];
  const box=document.getElementById('imgs'); box.innerHTML='';
  for(let k=0;k<it.n;k++){ const w=document.createElement('div'); w.className='imw';
    const im=document.createElement('img');
    im.src='/img?sid='+it.station_id+'&idx='+k+'&_='+i;
    im.onclick=()=>window.open(im.src,'_blank');
    const cap=document.createElement('div'); cap.className='cap'; cap.textContent='image '+(k+1)+' / '+it.n;
    w.appendChild(im); w.appendChild(cap); box.appendChild(w); }
  const tag = it.status==='ok_street_view' ? '<span class="tag sv">Street View</span>'
                                           : '<span class="tag">user photo</span>';
  document.getElementById('meta').innerHTML=`#${i+1}/${S.items.length} &nbsp; station ${it.station_id} &nbsp; ${tag}`;
  document.getElementById('sub').textContent=[it.station_name,it.street_address].filter(Boolean).join('  —  ');
  const done=S.items.filter(x=>x.label).length;
  document.getElementById('count').textContent=done+' / '+S.items.length+' labeled';
  document.getElementById('progfill').style.width=(100*done/S.items.length)+'%';
  S.classes.forEach(c=>document.getElementById('b_'+c).classList.toggle('sel', it.label===c));
}
async function setLabel(c){ const it=S.items[i];
  it.label = (it.label===c)? '' : c;  // click same again to clear
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({station_id:it.station_id,label:it.label})});
  render(); if(it.label){ setTimeout(()=>{ if(i<S.items.length-1){i++;render();} },120); }
}
function go(d){ i=Math.max(0,Math.min(S.items.length-1,i+d)); render(); }
function nextUnlabeled(){ let n=S.items.findIndex((x,k)=>k>i&&!x.label);
  if(n<0) n=S.items.findIndex(x=>!x.label); if(n<0)n=S.items.length-1; i=n; render(); }
function jumpToId(){ const v=document.getElementById('lookup').value.trim();
  const msg=document.getElementById('lookupmsg'); if(!v){ msg.textContent=''; return; }
  const n=S.items.findIndex(x=>x.station_id===v);
  if(n<0){ msg.textContent='id '+v+' not found (no image, or not in this set)'; return; }
  msg.textContent=''; i=n; render(); }
document.getElementById('prev').onclick=()=>go(-1);
document.getElementById('next').onclick=()=>go(1);
document.getElementById('unl').onclick=nextUnlabeled;
document.getElementById('golookup').onclick=jumpToId;
document.getElementById('lookup').addEventListener('keydown',e=>{
  if(e.key==='Enter'){ e.preventDefault(); jumpToId(); } e.stopPropagation(); });
document.addEventListener('keydown',e=>{
  if(e.target && e.target.tagName==='INPUT') return;   // don't trigger while typing an id
  if(e.key>='1'&&e.key<='5'){ setLabel(S.classes[+e.key-1]); }
  else if(e.key==='ArrowLeft'){ go(-1); }
  else if(e.key==='ArrowRight'){ go(1); }
  else if(e.key==='u'||e.key==='U'){ nextUnlabeled(); }
});
load();
</script></body></html>"""


def main() -> None:
    _load()
    _save()  # ensure the CSV exists with the full worklist
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[label] {len(ITEMS)} sites with images loaded. Open {url}")
    print(f"[label] autosaves to {LABELS_CSV.name} after every choice. Ctrl-C to stop.")
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
