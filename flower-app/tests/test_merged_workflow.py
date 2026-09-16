import copy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_agent.orchestrator import investigate
from fusion_agent.service import make_server
from fusion_agent.sites import RemoteSite, ThermalSite
from fusion_agent.steward import FacilitySteward
from fusion_agent.thermal.core import Gateway
from fusion_agent.thermal.torax_extract import electron_source, fraction_to_action
from fusion_agent.tools import Toolbox


class Item(SimpleNamespace):
    def to_dict(self):
        return vars(self)


def test_steward_cannot_bypass_prerequisites_or_export_prose(tmp_path):
    site = ThermalSite("facility-a", "A", tmp_path)
    messages = []

    def create(**kwargs):
        messages.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output=[
                Item(type="function_call", name="release_requested", arguments="{}")
            ],
        )

    steward = FacilitySteward(
        SimpleNamespace(responses=SimpleNamespace(create=create)), "test"
    )
    box = Toolbox({site.site_id: site}, steward=steward)
    assert (
        box.execute("request_evidence", {"site": site.site_id, "kind": "balance"})[
            "reason"
        ]
        == "prerequisite-missing"
    )
    result = box.execute("request_evidence", {"site": site.site_id, "kind": "context"})
    assert result["status"] == "released"
    assert "chi_true" not in json.dumps(messages)

    steward.client.responses.create = lambda **_: SimpleNamespace(
        status="completed", output=[Item(type="message", text="PRIVATE MARKER")]
    )
    withheld = box.execute(
        "request_evidence", {"site": site.site_id, "kind": "balance"}
    )
    assert withheld["status"] == "withheld"
    assert "PRIVATE MARKER" not in json.dumps(box.audit)
    assert len(site.call("evidence_history", {})["events"]) == 2


def test_full_investigation_uses_separate_steward_contexts(tmp_path):
    site = ThermalSite("facility-b", "B", tmp_path)
    requests = []
    next_product = iter(["context", "balance", "source_check"])

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        if kwargs.get("stream"):
            return iter(
                [
                    Item(
                        type="response.output_text.delta",
                        delta="B has an audit; A still needs a measurement.",
                    ),
                    Item(type="response.completed"),
                ]
            )
        if kwargs["tools"][0]["name"] == "release_requested":
            return SimpleNamespace(
                status="completed",
                output=[
                    Item(type="function_call", name="release_requested", arguments="{}")
                ],
            )
        product = next(next_product, None)
        calls = (
            []
            if product is None
            else [
                Item(
                    type="function_call",
                    name="request_evidence",
                    call_id=product,
                    arguments=json.dumps({"site": "facility-b", "kind": product}),
                )
            ]
        )
        return SimpleNamespace(status="completed", output=calls)

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    box = Toolbox({site.site_id: site}, steward=FacilitySteward(client, "test"))
    answer = investigate(
        client,
        SimpleNamespace(emit=lambda _: None),
        box,
        "test",
        [{"role": "user", "content": "Investigate"}],
    )
    assert "measurement" in answer
    assert [a["result"]["spent"] for a in box.audit] == [1, 3, 5]
    assert box.audit[-1]["result"]["finding"]["corrected_transport"] == "near-reference"
    steward_requests = [
        r for r in requests if r["tools"][0]["name"] == "release_requested"
    ]
    assert len(steward_requests) == 3
    assert all(isinstance(r["input"], str) for r in steward_requests)
    assert "chi_true" not in json.dumps(requests)


def test_remote_site_enforces_same_gateway(tmp_path, monkeypatch):
    site = ThermalSite("facility-c", "C", tmp_path)
    token = "remote-test-token-at-least-24-chars"
    monkeypatch.setenv("THERMAL_TEST_TOKEN", token)
    server = make_server(site, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = RemoteSite(
            "facility-c", f"http://127.0.0.1:{server.server_port}", "THERMAL_TEST_TOKEN"
        )
        assert (
            remote.call("request_evidence", {"kind": "context"})["status"] == "released"
        )
        assert (
            remote.call("request_evidence", {"kind": "balance"})["reason"]
            == "analogy-not-applicable"
        )
        assert remote.call("evidence_history", {})["events"][0]["site"] == "facility-c"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_evidence_cache_version_changes_with_inputs(tmp_path, monkeypatch):
    from fusion_agent.thermal import core

    gate = Gateway(tmp_path / "ledger.sqlite3")
    before = gate.request("A", "context")
    monkeypatch.setattr(core, "CASE", "different-case")
    changed = Gateway(tmp_path / "ledger.sqlite3")
    assert changed.events() == []
    after = changed.request("A", "context")
    assert not after["cached"]
    assert before["evidence_id"] != after["evidence_id"]


def test_fraction_conversion_prevents_half_power_bug():
    assert (fraction_to_action(0.001) + 1) / 2 == pytest.approx(0.001)
    assert fraction_to_action(0) == -1
    assert fraction_to_action(1) == 1
    with pytest.raises(ValueError):
        fraction_to_action(float("nan"))


def test_unverified_torax_snapshot_cannot_release_balance(tmp_path, monkeypatch):
    from fusion_agent.thermal.core import _devices
    from fusion_agent.thermal.profiles import KEV_TO_J, simulate

    devices = _devices()
    case = simulate(devices["A"])
    fixture = {
        "A": {
            "device": devices["A"]["name"],
            "provenance": "torax-1.4.3-test",
            "r": case["r"].tolist(),
            "n_e": case["n_e"].tolist(),
            "T_e_keV": (case["T_e"] / KEV_TO_J).tolist(),
            "source": case["source"].tolist(),
            "steady_state_verified": False,
        }
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(fixture))
    monkeypatch.setenv("FUSION_THERMAL_FIXTURES", str(path))
    gateway = Gateway(tmp_path / "ledger.sqlite3")
    context = gateway.request("A", "context")
    assert context["finding"]["balance_assumption"] == "unverified-steady-state"
    assert gateway.request("A", "balance")["reason"] == "balance-assumption-unverified"


def test_torax_source_dictionary_and_exchange_are_read():
    state = SimpleNamespace(
        core_sources=SimpleNamespace(
            T_e={"aux": np.array([2.0, 3.0]), "sink": np.array([-0.2, -0.3])},
            qei=SimpleNamespace(qei_coef=np.array([0.5, 0.5])),
        ),
        core_profiles=SimpleNamespace(
            T_e=SimpleNamespace(value=np.array([1.0, 2.0])),
            T_i=SimpleNamespace(value=np.array([2.0, 4.0])),
        ),
    )
    np.testing.assert_allclose(electron_source(state), [2.3, 3.7])
    state.core_sources.T_e["aux"] = np.array([1.0])
    with pytest.raises(ValueError, match="inconsistent"):
        electron_source(state)


@pytest.mark.parametrize("terminal", ["response.incomplete", "response.completed"])
def test_empty_or_incomplete_report_is_not_success(terminal):
    def create(**kwargs):
        if kwargs.get("stream"):
            return iter([Item(type=terminal)])
        return SimpleNamespace(status="completed", output=[])

    with pytest.raises(RuntimeError):
        investigate(
            SimpleNamespace(responses=SimpleNamespace(create=create)),
            SimpleNamespace(emit=lambda _: None),
            Toolbox({}),
            "test",
            [],
        )


def test_terminal_text_is_recovered_without_deltas():
    def create(**kwargs):
        if kwargs.get("stream"):
            return iter(
                [
                    Item(
                        type="response.completed",
                        response={
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "Recovered report",
                                        }
                                    ]
                                }
                            ]
                        },
                    )
                ]
            )
        return SimpleNamespace(status="completed", output=[])

    answer = investigate(
        SimpleNamespace(responses=SimpleNamespace(create=create)),
        SimpleNamespace(emit=lambda _: None),
        Toolbox({}),
        "test",
        [],
    )
    assert answer == "Recovered report"
