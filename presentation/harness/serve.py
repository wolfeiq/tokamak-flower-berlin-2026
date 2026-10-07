"""Serve the deck and its optional live, synthetic Python gateway on loopback."""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "flower-app"))
os.environ.pop("FUSION_THERMAL_FIXTURES", None)


def make_server(port=8790, ledger=None):
    from fusion_agent.thermal.core import Gateway

    gateway = Gateway(ledger or ROOT / ".state" / "presentation.sqlite3")
    files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/slides.css": ("slides.css", "text/css; charset=utf-8"),
        "/slides.js": ("slides.js", "text/javascript; charset=utf-8"),
        "/evidence.js": ("evidence.js", "text/javascript; charset=utf-8"),
        "/speech_for_harness.txt": ("speech_for_harness.txt", "text/plain; charset=utf-8"),
        "/presentation.pdf": ("presentation.pdf", "application/pdf"),
    }
    for font in (ROOT / "assets").glob("*.ttf"):
        files["/assets/" + font.name] = ("assets/" + font.name, "font/ttf")

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind="application/json; charset=utf-8"):
            raw = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/status":
                return self.send(200, {"mode": "live-python-gateway", "case_id": gateway.case_id})
            if path in files:
                name, kind = files[path]
                target = ROOT / name
                if target.is_file():
                    return self.send(200, target.read_bytes(), kind)
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/request":
                return self.send(404, {"error": "Not found"})
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if self.headers.get("Origin") not in allowed:
                return self.send(403, {"error": "Use the presentation's local origin"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError()
                request = json.loads(self.rfile.read(length))
                if set(request) != {"site", "kind"}:
                    raise ValueError()
                if request["site"] not in ("A", "B", "C") or request["kind"] not in ("context", "balance", "source_check", "raw_logs"):
                    raise ValueError()
                result = gateway.request(request["site"], request["kind"])
                self.send(200, {"request": request, "result": result})
            except (ValueError, TypeError, KeyError):
                self.send(400, {"error": "Invalid evidence request"})
            except Exception:
                self.send(500, {"error": "Gateway request failed; inspect the local server"})
                raise

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Open http://127.0.0.1:{server.server_port} — synthetic gateway, no LLM.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
