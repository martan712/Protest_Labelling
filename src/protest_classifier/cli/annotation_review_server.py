#!/usr/bin/env python3
"""Small, local CSV annotation review tool.

Usage: .venv/bin/annotation-review-server
Then open http://127.0.0.1:8765.  The server only exposes CSV files below
the selected annotations directory and writes updates atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation review</title><style>
:root{font:14px system-ui,sans-serif;color:#202124;background:#f6f7f9}body{margin:0}header{position:sticky;top:0;z-index:3;background:#26354a;color:white;padding:12px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0 16px 0 0}select,input,textarea,button{font:inherit;border:1px solid #b9c2ce;border-radius:4px;padding:6px;background:white}button{cursor:pointer;background:#e8eef8}button:disabled{opacity:.5}.layout{padding:14px 18px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.bar input{min-width:300px}.status{margin-left:auto}.ok{color:#b9f2c9}.error{color:#ffb4b4}.table-wrap{overflow:auto;max-height:calc(100vh - 105px);background:white;border:1px solid #d4dae2}table{border-collapse:collapse;width:100%;min-width:1150px}th,td{border-bottom:1px solid #e2e6eb;padding:7px;vertical-align:top;text-align:left}th{position:sticky;top:0;background:#edf1f6;z-index:1}th:nth-child(1){width:125px}th:nth-child(2){width:31%}th:nth-child(3){width:180px}th:nth-child(4){width:180px}th:nth-child(5){width:31%}th:nth-child(6){width:76px}.source{white-space:pre-wrap;line-height:1.35;max-height:120px;overflow:auto;color:#303944}.readonly{white-space:pre-wrap;color:#596575;max-width:220px}.edit{width:100%;box-sizing:border-box;min-height:32px;resize:vertical}.evidence{min-height:70px}.dirty{background:#fffbe8}.saved{color:#14783c;font-size:12px}.muted{color:#66717f}
</style></head><body><header><h1>Annotation review</h1><label>File <select id="file"></select></label><button id="reload">Reload</button><button id="saveAll">Save all changes</button></header><main class="layout"><div class="bar"><input id="search" placeholder="Filter by text, ID, or label"><label><input type="checkbox" id="changed"> unsaved only</label><span id="count" class="muted"></span><span id="status"></span></div><div class="table-wrap"><table><thead><tr><th>Event</th><th>Source text</th><th>Primary label</th><th>Alternative labels</th><th>Evidence</th></tr></thead><tbody id="body"></tbody></table></div></main><datalist id="labels"></datalist><script>
const $=id=>document.getElementById(id);let file='',rows=[],columns=[],dirty=new Set();const edit=['primary_label','alternative_labels','evidence'];
async function api(url,opts){let r=await fetch(url,opts),j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function files(){let j=await api('/api/files');$('file').innerHTML='';j.files.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;$('file').appendChild(o)});if(j.files.length)await load()}
async function load(){file=$('file').value;let j=await api('/api/data?file='+encodeURIComponent(file));rows=j.rows;columns=j.columns;dirty.clear();render()}
function text(r){return r.notes||r.description||r.text||''}function matches(r){let q=$('search').value.toLowerCase();return(!q||Object.values(r).some(v=>String(v??'').toLowerCase().includes(q)))}
function labelOptions(){return [...new Set(rows.flatMap(r=>[r.primary_label,r.alternative_labels]).filter(Boolean))].sort()}
function render(){let body=$('body');body.innerHTML='';let shown=0,labels=labelOptions();rows.forEach((r,i)=>{if(!matches(r)||($('changed').checked&&!dirty.has(i)))return;shown++;let tr=document.createElement('tr');tr.dataset.i=i;if(dirty.has(i))tr.classList.add('dirty');let event=document.createElement('td');event.textContent=r.event_id_cnty||`row ${i+1}`;tr.appendChild(event);let source=document.createElement('td');source.className='source';source.textContent=text(r);tr.appendChild(source);edit.forEach(c=>{let td=document.createElement('td'),el;if(c==='primary_label'||c==='alternative_labels'){el=document.createElement('select');el.className='edit';let opts=['',...labels];if(r[c]&&!opts.includes(r[c]))opts.push(r[c]);opts.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v||'(none)';el.appendChild(o)});el.value=r[c]||''}else{el=document.createElement('textarea');el.className='edit evidence';el.value=r[c]||''}el.addEventListener('change',()=>{r[c]=el.value;dirty.add(i);tr.classList.add('dirty');updateSaveState()});if(c==='evidence')el.addEventListener('input',()=>{r[c]=el.value;dirty.add(i);tr.classList.add('dirty');updateSaveState()});td.appendChild(el);tr.appendChild(td)});body.appendChild(tr)});$('count').textContent=`${shown} of ${rows.length} records`;updateSaveState()}
function updateSaveState(){$('saveAll').disabled=dirty.size===0;$('saveAll').textContent=dirty.size?`Save all changes (${dirty.size})`:'Save all changes'}
async function saveAll(){if(!dirty.size)return;let b=$('saveAll');b.disabled=true;try{let j=await api('/api/document',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({file,rows})});rows=j.rows;dirty.clear();$('status').textContent='All changes saved';$('status').className='ok';render();setTimeout(()=>{$('status').textContent=''},1400)}catch(e){$('status').textContent=e.message;$('status').className='error';b.disabled=false;updateSaveState()}}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
$('file').onchange=load;$('reload').onclick=load;$('saveAll').onclick=saveAll;$('search').oninput=render;$('changed').onchange=render;files().catch(e=>{$('status').textContent=e.message;$('status').className='error'});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    root: Path

    def _json(self, value: object, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if parsed.path == "/api/files":
            # Chunk queues are intermediate inputs; review only assembled/final
            # annotation files unless a different directory is supplied.
            files = [p.relative_to(self.root).as_posix() for p in self.root.rglob("*.csv")
                     if p.is_file() and not any("chunk" in part.lower()
                                                for part in p.relative_to(self.root).parts)]
            self._json({"files": sorted(files)}); return
        if parsed.path == "/api/data":
            rel = parse_qs(parsed.query).get("file", [""])[0]
            try: path = self._safe(rel)
            except ValueError as e: self._json({"error": str(e)}, 400); return
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh); rows = list(reader); columns = reader.fieldnames or []
            self._json({"columns": columns, "rows": rows}); return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        endpoint = urlparse(self.path).path
        if endpoint == "/api/document":
            self._save_document()
            return
        if endpoint != "/api/row": self._json({"error": "Not found"}, 404); return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            path = self._safe(body["file"]); row_index = int(body["index"]); new_row = body["row"]
            if row_index < 0: raise ValueError("Invalid row index")
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh); columns = reader.fieldnames or []; rows = list(reader)
            if row_index >= len(rows): raise ValueError("Row no longer exists; reload the file")
            if set(new_row) - set(columns): raise ValueError("Unknown column in update")
            original = rows[row_index]
            # The browser can only edit the three review fields.  Read-only
            # source/provenance fields always come from the file on disk.
            editable = {c for c in ("primary_label", "alternative_labels", "evidence") if c in columns}
            updated = dict(original)
            for c in editable:
                if str(new_row.get(c, "")) != original.get(c, ""):
                    updated[c] = str(new_row.get(c, ""))
            if any(updated.get(c, "") != original.get(c, "") for c in editable) and "annotator" in columns:
                updated["annotator"] = "manual"
            rows[row_index] = {c: updated.get(c, "") for c in columns}
            self._atomic_write(path, columns, rows)
            self._json({"row": rows[row_index]})
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as e: self._json({"error": str(e)}, 400)

    def _save_document(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            path = self._safe(body["file"]); incoming = body["rows"]
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh); columns = reader.fieldnames or []; original = list(reader)
            if not isinstance(incoming, list) or len(incoming) != len(original):
                raise ValueError("The file changed on disk; reload before saving")
            editable = {c for c in ("primary_label", "alternative_labels", "evidence") if c in columns}
            updated_rows = []
            for old, new in zip(original, incoming):
                if not isinstance(new, dict) or any(str(new.get(c, "")) != old.get(c, "") for c in ("event_id_cnty",)):
                    raise ValueError("The row order or event IDs changed; reload before saving")
                updated = dict(old); changed = False
                for c in editable:
                    value = str(new.get(c, ""))
                    if value != old.get(c, ""):
                        updated[c] = value; changed = True
                if changed and "annotator" in columns: updated["annotator"] = "manual"
                updated_rows.append({c: updated.get(c, "") for c in columns})
            self._atomic_write(path, columns, updated_rows)
            self._json({"rows": updated_rows})
        except (ValueError, KeyError, json.JSONDecodeError, OSError, TypeError) as e:
            self._json({"error": str(e)}, 400)

    def _safe(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if path.parent == self.root or self.root in path.parents:
            if path.suffix.lower() == ".csv" and path.exists(): return path
        raise ValueError("Invalid annotation file")

    @staticmethod
    def _atomic_write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns); writer.writeheader(); writer.writerows(rows); fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def log_message(self, *_args: object) -> None: pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); root = args.annotations.resolve()
    if not root.is_dir(): parser.error(f"annotation directory does not exist: {root}")
    Handler.root = root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Review annotations at http://{args.host}:{args.port} (root: {root})")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
