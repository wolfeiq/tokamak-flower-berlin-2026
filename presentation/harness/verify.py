"""Exercise the presentation server, then export and check the live browser deck."""
import gc
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from serve import ROOT, make_server


def request(base, product, origin=True):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = base
    req = Request(base + "/api/request", data=json.dumps(product).encode(), headers=headers)
    with urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    with tempfile.TemporaryDirectory(prefix="fusion-deck-check-") as directory:
        server = make_server(0, Path(directory) / "gateway.sqlite3")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for route in ("/.venv/pyvenv.cfg", "/.state/presentation.sqlite3", "/speaker_deep_dive.txt", "/../flower-app/pyproject.toml"):
                try:
                    urlopen(base + route)
                    raise AssertionError("Private path was served")
                except HTTPError as exc:
                    assert exc.code == 404
            try:
                request(base, {"site": "A", "kind": "context"}, origin=False)
                raise AssertionError("Cross-origin mutation was accepted")
            except HTTPError as exc:
                assert exc.code == 403
            # Denied requests should not spend units or create a release.
            assert request(base, {"site": "B", "kind": "source_check"})["result"]["reason"] == "prerequisite-missing"
            result = subprocess.run([shutil.which("node") or "node", str(ROOT / "export.mjs"), "--url=" + base], check=False)
            assert result.returncode == 0, "Browser verification/export failed"
            # A fresh Gateway object must see the debits of the browser's calls.
            from fusion_agent.thermal.core import Gateway
            restarted = Gateway(Path(directory) / "gateway.sqlite3")
            cached = restarted.request("B", "source_check")
            assert cached["cached"] and cached["spent"] == 5
            assert restarted.request("C", "balance")["reason"] == "analogy-not-applicable"
            print("PASS: local server isolation, same-origin requests, prerequisites and ledger persistence.")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)
            gc.collect()


if __name__ == "__main__":
    main()
