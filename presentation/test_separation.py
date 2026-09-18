"""Presentation navigation must not expose the executable research application."""

import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from serve import ROOT, make_server


def test_presentation_has_visuals_but_no_execution_api():
    server = make_server(0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for path in ("/", "/deck.js", "/visuals/atlas.html", "/visuals/diiid_like.html",
                     "/visuals/marl-architecture.png", "/visuals/flower-agent-architecture.svg"):
            with urlopen(base + path) as response:
                assert response.status == 200
        for path in ("/api/run", "/api/thermal", "/../flower-app/dashboard.py", "/serve.py"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/run", data=b"{}", method="POST"))
        assert error.value.code == 501
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base, headers={"Host": "untrusted.example"}))
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_deck_does_not_load_application_code():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert "visuals/atlas.html" in html
    for executable in ("/app.js", "/thermal.js", "/physics.js", "run-form", "/api/"):
        assert executable not in html
