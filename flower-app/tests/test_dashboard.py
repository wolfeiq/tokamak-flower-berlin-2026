"""Exercise the frontend boundary without submitting paid hosted runs."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import dashboard as dashboard_module
from dashboard import Dashboard, decode_logs, make_server


@pytest.fixture
def dashboard(tmp_path):
    app = Dashboard(tmp_path)
    server = make_server(app, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def post(base, path, data, origin=None):
    return urlopen(
        Request(
            base + path,
            method="POST",
            data=json.dumps(data).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Fusion-UI": "1",
                "Origin": origin or base,
            },
        ),
        timeout=3,
    )


def test_frontend_and_live_tools(dashboard):
    _, base = dashboard
    with urlopen(base) as response:
        html = response.read()
        assert b"Fusion Investigator" in html
        assert b'class="application"' in html
        assert b'class="slide"' not in html
        assert b'/deck.js' not in html
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/deck.js")
    assert error.value.code == 404
    with urlopen(base + "/api/evidence") as response:
        evidence = json.load(response)["diagnoses"]
    assert len(evidence) == 3
    assert evidence[2]["ranked"][0]["hypothesis"] == "increased_heat_loss"
    for scale, passed in [(1.25, True), (1.43, False)]:
        with post(
            base, "/api/validate", {"site": "demo-a", "power_scale": scale}
        ) as response:
            result = json.load(response)
        assert result["improves_all_scenarios"] is passed
        assert result["rollouts"] == 6


def test_invalid_requests_and_cross_origin_cannot_launch(dashboard):
    app, base = dashboard
    with pytest.raises(HTTPError) as exc:
        post(base, "/api/run", {"prompt": "Investigate"}, "https://another.example")
    assert exc.value.code == 403
    assert not app.active
    for path, data in [
        ("/api/run", {"prompt": ""}),
        ("/api/validate", {"site": "demo-a", "power_scale": 99}),
    ]:
        with pytest.raises(HTTPError) as exc:
            post(base, path, data)
        assert exc.value.code == 400
    with pytest.raises(HTTPError) as exc:
        urlopen(Request(base, headers={"Host": "untrusted.example"}))
    assert exc.value.code == 403
    with pytest.raises(HTTPError):
        urlopen(base + "/../.dashboard/run.json")


def test_worker_uses_flower_and_preserves_multiline_prompt(tmp_path):
    import tomllib

    app = Dashboard(tmp_path)
    calls = []

    def cli(args):
        calls.append(args)
        if args[0] == "run":
            config = tomllib.loads((tmp_path / "request.toml").read_text())
            assert config["agent.input"] == 'Compare sites.\nQuote: "test"'
            return json.dumps({"success": True, "run-id": "123"})
        if args[0] == "ls":
            return json.dumps({"runs": [{"status": "finished:completed"}]})
        return '# Finding\nEvidence\nFUSION_TOOL_AUDIT [{"tool":"list_sites"}]'

    app.cli = cli
    app.active = True
    app.worker('Compare sites.\nQuote: "test"')
    assert [c[0] for c in calls] == ["run", "ls", "log"]
    # No federation flag unless the operator sets one: a hardcoded federation
    # made every other presenter's submission fail with no run ID.
    assert "--federation" not in calls[0]
    assert app.snapshot()["status"] == "finished:completed"
    assert app.snapshot()["run_id"] == "123"
    assert not app.active
    assert app.snapshot()["audit"] == [{"tool": "list_sites"}]
    assert Dashboard(tmp_path).snapshot()["report"] == "# Finding\nEvidence"


def test_worker_passes_operator_federation_override(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_module, "FEDERATION", "@team/shared")
    app = Dashboard(tmp_path)
    calls = []

    def cli(args):
        calls.append(args)
        if args[0] == "run":
            return json.dumps({"success": True, "run-id": "123"})
        if args[0] == "ls":
            return json.dumps({"runs": [{"status": "finished:completed"}]})
        return "report"

    app.cli = cli
    app.active = True
    app.worker("Investigate")
    run_call = calls[0]
    assert run_call[run_call.index("--federation") + 1] == "@team/shared"


def test_submission_failure_is_visible(tmp_path):
    app = Dashboard(tmp_path)

    def fail(_):
        raise RuntimeError("Flower login expired")

    app.cli = fail
    app.active = True
    app.worker("Investigate")
    assert app.snapshot()["status"] == "connection error"
    assert "login expired" in app.snapshot()["error"]
    assert not app.active


def test_decode_logs_keeps_report_separate_from_audit():
    parsed = decode_logs("Runtime preamble\n# Report\nHello\nFUSION_TOOL_AUDIT []\n")
    assert parsed["report"] == "# Report\nHello"
    assert parsed["audit"] == []


def test_rehearsal_replay_and_manual_disclosure_use_real_gateway(dashboard):
    app, base = dashboard
    with post(
        base, "/api/release", {"site": "facility-a", "kind": "balance"}
    ) as response:
        assert json.load(response)["events"][-1]["reason"] == "prerequisite-missing"
    with post(base, "/api/replay", {}) as response:
        first = json.load(response)
    assert first["mode"] == "local-rehearsal"
    assert len(first["events"]) == 10
    assert first["events"][4]["reason"] == "analogy-not-applicable"
    assert first["events"][8]["reason"] == "unsupported-request"
    assert first["events"][-1]["cached"]
    with post(base, "/api/replay", {}) as response:
        second = json.load(response)
    assert all(e["cached"] for e in second["events"] if e["status"] == "released")
    assert not app.active  # Rehearsal cannot launch a hosted run.
    restored = Dashboard(app.state_dir)
    assert restored.thermal_snapshot() == second
