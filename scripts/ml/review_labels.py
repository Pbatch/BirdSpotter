"""Review classifier images in the browser: B marks bird, N marks no_bird, Z undoes."""

import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "classifier_birds"

PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Label review</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #eee; font: 15px system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; display: flex; gap: 24px; align-items: baseline; }
  #progress { font-weight: 600; }
  #meta { color: #aaa; }
  main { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
  img { max-width: 100%; max-height: 100%; object-fit: contain; }
  footer { padding: 10px 16px; color: #aaa; }
  kbd { background: #333; border-radius: 4px; padding: 1px 6px; color: #fff; }
  #flash { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
           font-size: 64px; font-weight: 700; opacity: 0; transition: opacity .25s; }
</style>
<header><span id="progress"></span><span id="meta"></span></header>
<main><img id="img" alt=""></main>
<footer><kbd>B</kbd> bird &nbsp; <kbd>N</kbd> no bird &nbsp; <kbd>Z</kbd> undo</footer>
<div id="flash"></div>
<script>
let items = [], done = {}, history = [], i = 0;
const $ = (id) => document.getElementById(id);

function next() {
  while (i < items.length && done[items[i].image]) i++;
  if (i >= items.length) {
    $("img").removeAttribute("src");
    $("progress").textContent = `All ${items.length} reviewed`;
    $("meta").textContent = "";
    return;
  }
  const it = items[i];
  $("img").src = "/image/" + encodeURIComponent(it.image);
  $("progress").textContent = `${Object.keys(done).length} / ${items.length}`;
  $("meta").textContent = `given: ${it.given} · p_bird ${it.p_bird} · ${it.source} · ${it.image}`;
  const ahead = items[i + 1];
  if (ahead) new Image().src = "/image/" + encodeURIComponent(ahead.image);
}

function flash(text, colour) {
  const f = $("flash");
  f.textContent = text; f.style.color = colour; f.style.opacity = 1;
  setTimeout(() => (f.style.opacity = 0), 250);
}

async function label(decision) {
  const it = items[i];
  if (!it) return;
  await fetch("/label", { method: "POST", body: JSON.stringify({ image: it.image, decision }) });
  done[it.image] = decision;
  history.push(i);
  flash(decision === "bird" ? "BIRD" : "NO BIRD", decision === "bird" ? "#4ade80" : "#f87171");
  next();
}

async function undo() {
  const last = history.pop();
  if (last === undefined) return;
  await fetch("/undo", { method: "POST", body: JSON.stringify({ image: items[last].image }) });
  delete done[items[last].image];
  i = last;
  flash("UNDO", "#facc15");
  next();
}

document.addEventListener("keydown", (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey) return;
  const k = e.key.toLowerCase();
  if (k === "b") label("bird");
  else if (k === "n") label("no_bird");
  else if (k === "z") undo();
});

fetch("/state").then((r) => r.json()).then((s) => { items = s.items; done = s.done; next(); });
</script>
"""


def load_decisions(path: Path) -> dict[str, str]:
    """Replay the append-only log, so later lines (and undos) win."""
    decisions: dict[str, str] = {}
    if path.exists():
        for line in path.open():
            record = json.loads(line)
            if record["decision"] is None:
                decisions.pop(record["image"], None)
            else:
                decisions[record["image"]] = record["decision"]
    return decisions


def make_handler(
    items: list[dict], output: Path, image_root: Path = DATASET
) -> type[BaseHTTPRequestHandler]:
    allowed = {item["image"] for item in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                state = {"items": items, "done": load_decisions(output)}
                self.send(json.dumps(state).encode(), "application/json")
            elif path.startswith("/image/") and (name := unquote(path[7:])) in allowed:
                file = image_root / name
                self.send(file.read_bytes(), mimetypes.guess_type(file)[0] or "image/jpeg")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["image"] not in allowed:
                self.send(b"unknown image", "text/plain", 400)
                return
            decision = body.get("decision") if self.path == "/label" else None
            with output.open("a") as f:
                f.write(json.dumps({"image": body["image"], "decision": decision}) + "\n")
            self.send(b"ok", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issues", type=Path, help="JSONL with image, given, p_bird and source")
    parser.add_argument(
        "--output", type=Path, help="Decision log (default: <issues>.decisions.jsonl)"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DATASET,
        help="Directory that item image paths are relative to (default: %(default)s)",
    )
    args = parser.parse_args()

    items = [json.loads(line) for line in args.issues.open()]
    output = args.output or args.issues.with_suffix(".decisions.jsonl")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(items, output, args.image_root)
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reviewing {len(items)} images at {url}; decisions go to {output}")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
