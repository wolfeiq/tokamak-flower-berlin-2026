"""Read-only presentation server. No Flower, experiment, or dashboard endpoints."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def make_server(port=8788):
    pages = {
        "/": (ROOT / "index.html", "text/html"),
        "/deck.js": (ROOT / "deck.js", "text/javascript"),
        "/style.css": (ROOT / "style.css", "text/css"),
    }
    pages.update({
        f"/visuals/{path.name}": (path, "text/html")
        for path in (ROOT / "visuals").glob("*.html")
    })

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.headers.get("Host") not in {
                f"localhost:{self.server.server_port}",
                f"127.0.0.1:{self.server.server_port}",
            }:
                self.send_error(403)
                return
            item = pages.get(self.path)
            if item is None:
                self.send_error(404)
                return
            path, mime = item
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Fusion presentation: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
