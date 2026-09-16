"""Local analysis boundary. Serve behind HTTPS for connections across facilities."""

from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(site, token: str, host="127.0.0.1", port=8765):
    if not token or len(token) < 24:
        raise ValueError("Site token must contain at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Never write request bodies or authorization headers to logs.
            pass

        def send_json(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            if not hmac.compare_digest(auth.encode(), f"Bearer {token}".encode()):
                self.send_json(401, {"error": "Unauthorized"})
                return
            if self.path != "/analysis":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("Invalid body size")
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(size))
                if set(body) != {"operation", "arguments"}:
                    raise ValueError("Unexpected request fields")
                if not isinstance(body["arguments"], dict):
                    raise TypeError("Arguments must be an object")
                result = site.call(body["operation"], body["arguments"])
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError, OSError):
                self.send_json(
                    400, {"error": "Invalid analysis request or unavailable evidence"}
                )

    return ThreadingHTTPServer((host, port), Handler)
