"""A local page for labelling answers as faithful or not. Keys: F, U, and the arrow keys.

Usage: python -m evals.label    # then open http://127.0.0.1:8765
Each answer is saved as you go, so you can stop and resume at any time.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from evals.harness.judge import judge_agreement

QUEUE = Path("evals/golden/judge_human_queue_v1.jsonl")
LABELS = Path("evals/golden/judge_human_labels_v1.jsonl")
JUDGE_SET = Path("evals/golden/judge_v1.jsonl")
HOST, PORT = "127.0.0.1", 8765
MAX_BODY_BYTES = 4096

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Label answers</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--box:#f6f6f6;--good:#17693a;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#a0a0a0;--line:#383838;--box:#222;--good:#6fcf97;--bad:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,system-ui,sans-serif;max-width:760px;margin:2.5rem auto;padding:0 1rem}
h1{font-size:1.2rem;margin:0}.bar{height:6px;background:var(--line);border-radius:3px;margin:.8rem 0 1.6rem}
.fill{height:100%;background:var(--good);border-radius:3px;transition:width .2s}
.label{color:var(--muted);font-size:.85rem;margin:1.1rem 0 .25rem}.box{background:var(--box);border-radius:8px;padding:.7rem .9rem;white-space:pre-wrap}
.actions{display:flex;gap:.7rem;margin-top:1.5rem;flex-wrap:wrap}
button{font:inherit;padding:.55rem 1rem;border-radius:8px;border:1px solid var(--line);background:var(--box);color:var(--fg);cursor:pointer}
button.chosen{border-color:var(--good);box-shadow:0 0 0 1px var(--good)}
.rule,.status{color:var(--muted);font-size:.9rem}.status{margin-top:1rem;min-height:1.4em}
</style></head><body>
<h1>Is this answer fully supported by the source?</h1>
<div class="bar"><div class="fill" id="fill" style="width:0"></div></div>
<main id="card"></main>
<div class="actions">
<button id="yes">Faithful (F)</button><button id="no">Unfaithful (U)</button>
<button id="back">Back (&larr;)</button><button id="next">Next (&rarr;)</button></div>
<p class="rule">Faithful means every claim is in the source. Numbers, names and yes/no conclusions must match. Extra details make it unfaithful.</p>
<p class="status" id="status" role="status"></p>
<script>
let items=[],labels={},at=0;
const $=id=>document.getElementById(id);
function field(name,text){const t=document.createElement('div');t.className='label';t.textContent=name;
 const b=document.createElement('div');b.className='box';b.textContent=text;return[t,b]}
function draw(){const done=Object.keys(labels).length;$('fill').style.width=(100*done/items.length)+'%';
 if(at>=items.length){$('card').textContent='All '+items.length+' items are labelled. You can close this page.';$('status').textContent='';
  $('yes').classList.remove('chosen');$('no').classList.remove('chosen');return}
 const item=items[at],card=$('card');card.replaceChildren(...field('Question',item.input.question),...field('Source',item.input.context),...field('Answer',item.input.answer));
 const value=labels[item.id];$('yes').classList.toggle('chosen',value===true);$('no').classList.toggle('chosen',value===false);
 $('status').textContent='Item '+(at+1)+' of '+items.length+', '+done+' labelled'}
async function choose(value){if(at>=items.length)return;const id=items[at].id;
 const r=await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,faithful:value})});
 if(!r.ok){$('status').textContent='Not saved. Is the server still running?';return}
 labels[id]=value;at+=1;draw()}
function move(step){at=Math.max(0,Math.min(items.length,at+step));draw()}
$('yes').onclick=()=>choose(true);$('no').onclick=()=>choose(false);$('back').onclick=()=>move(-1);$('next').onclick=()=>move(1);
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k==='f')choose(true);else if(k==='u')choose(false);
 else if(k==='arrowleft')move(-1);else if(k==='arrowright')move(1)});
fetch('/state').then(r=>r.json()).then(s=>{items=s.items;labels=s.labels;at=s.position;draw()});
</script></body></html>
"""


class Labels:
    def __init__(self, queue_path: Path = QUEUE, labels_path: Path = LABELS) -> None:
        self.queue_path = queue_path
        self.labels_path = labels_path
        self.items = [json.loads(line) for line in queue_path.read_text("utf-8").splitlines()]
        self._ids = {item["id"] for item in self.items}

    def labels(self) -> dict[str, bool]:
        """The label file is append-only, and the latest line for an id wins."""
        if not self.labels_path.exists():
            return {}
        rows = (json.loads(line) for line in self.labels_path.read_text("utf-8").splitlines())
        return {row["id"]: row["faithful"] for row in rows}

    def save(self, item_id: str, faithful: bool) -> None:
        if item_id not in self._ids:
            raise KeyError(item_id)
        if not isinstance(faithful, bool):
            raise TypeError("faithful must be true or false")
        with self.labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": item_id, "faithful": faithful}) + "\n")

    def state(self) -> dict[str, Any]:
        labels = self.labels()
        position = next(
            (i for i, item in enumerate(self.items) if item["id"] not in labels), len(self.items)
        )
        return {"items": self.items, "labels": labels, "position": position, "done": len(labels)}


def agreement_report(labels: Labels, seeded: dict[str, bool]) -> dict[str, Any]:
    """Your labels against the seeded truth. Judge-against-human uses the same function."""
    human = labels.labels()
    sources = {item["id"]: item["source_id"] for item in labels.items}
    pairs = [(seeded[sources[i]], label) for i, label in human.items() if sources[i] in seeded]
    return judge_agreement([truth for truth, _ in pairs], [label for _, label in pairs])


def make_handler(labels: Labels) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._json(HTTPStatus.OK, labels.state())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            # Another website can post text/plain here, but not JSON without a CORS preflight,
            # and this server never approves one.
            if self.headers.get_content_type() != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON only"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if self.path != "/label" or not 0 < length <= MAX_BODY_BYTES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad request"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                labels.save(body["id"], body["faithful"])
            except (json.JSONDecodeError, KeyError, TypeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad label"})
                return
            self._json(HTTPStatus.OK, {"ok": True})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", action="store_true", help="Print agreement and exit.")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    labels = Labels()
    if args.agreement:
        rows = (json.loads(line) for line in JUDGE_SET.read_text("utf-8").splitlines())
        seeded = {row["id"]: row["expected"]["faithful"] for row in rows}
        print(agreement_report(labels, seeded))
        return
    print(
        f"Open http://{HOST}:{args.port}  ({labels.state()['done']}/{len(labels.items)} labelled)"
    )
    ThreadingHTTPServer((HOST, args.port), make_handler(labels)).serve_forever()


if __name__ == "__main__":
    main()
