import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion_agent import evidence, heating
from fusion_agent.orchestrator import history, investigate
from fusion_agent.service import make_server
from fusion_agent.sites import LocalSite, RemoteSite, load_sites
from fusion_agent.tools import Toolbox


def checkpoint():
    def run(seed, errors):
        return {
            "seed": seed,
            "shots": [
                {
                    "shot": i,
                    "steps": 30,
                    "beta_error": err,
                    "is_evaluation": True,
                    "evaluation_eligible": True,
                    "violations": [],
                    "terminated_early": False,
                }
                for i, err in enumerate(errors)
            ],
        }

    return {
        "manifest": {
            "metric": "shots_to_joint_competence_evaluated",
            "seeds": 2,
            "join": 4,
            "expected_steps": 30,
            "joiners": ["device"],
            "arms": ["scratch", "federated_uniform"],
            "task": "test",
            "tolerances": {"device": 0.1},
            "status": "running",
        },
        "runs": {
            "device:scratch": [run(0, [0.2, 0.05, 0.05, 0.05]), run(1, [0.2] * 4)],
            "device:federated_uniform": [run(0, [0.05] * 4)],
        },
        "run_metadata": {
            "device:scratch": {
                str(s): {
                    "joiner_probe_shots": 0,
                    "joiner_selection_shots": 0,
                    "source_shots": 0,
                }
                for s in (0, 1)
            },
            "device:federated_uniform": {
                "0": {
                    "joiner_probe_shots": 0,
                    "joiner_selection_shots": 2,
                    "source_shots": 240,
                    "handover_adopted": True,
                }
            },
        },
    }


def test_partial_coverage_censoring_costs_and_selection():
    result = evidence.summary(checkpoint(), "abc", "study")
    assert not result["coverage_complete"]
    scratch, fed = result["rows"]
    assert scratch["censored"] == 1
    assert scratch["median_joiner_shots_among_reached"] == 3
    assert fed["completed_seeds"] == 1
    assert fed["median_joiner_shots_among_reached"] == 2
    assert fed["independent_confirmation_median"] == 4
    assert fed["median_source_shots"] == 240
    assert result["paired_against_scratch"][0]["faster"] == 1
    assert result["paired_against_scratch"][1]["slower"] == 1
    assert "shots" not in result


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 29},
        {"terminated_early": True},
        {"violations": ["beta"]},
        {"beta_error": None},
        {"beta_error": float("nan")},
    ],
)
def test_joint_success_requires_all_conditions(changes):
    p = checkpoint()
    run = p["runs"]["device:federated_uniform"][0]
    run["shots"][1].update(changes)
    assert evidence.competence(run, 0.1, 30) == 4


def test_rejected_trial_cannot_confirm():
    run = checkpoint()["runs"]["device:federated_uniform"][0]
    run["shots"][0]["evaluation_eligible"] = False
    assert evidence.competence(run, 0.1, 30) == 3


def test_missing_metadata_does_not_invent_sensitivity():
    p = checkpoint()
    p["run_metadata"] = {}
    row = evidence.summary(p, "abc", "study")["rows"][1]
    assert row["independent_confirmation_median"] is None
    assert not row["metadata_complete"]


def test_duplicate_seeds_rejected():
    p = checkpoint()
    p["runs"]["device:scratch"].append(p["runs"]["device:scratch"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        evidence.summary(p, "abc", "study")


@pytest.mark.parametrize("name", ["../secret", "C:/secret", "a/b", "..", ""])
def test_study_paths_confined(tmp_path, name):
    with pytest.raises(ValueError):
        evidence.inspect_study(tmp_path, name)


def test_checkpoint_read_and_provenance(tmp_path):
    (tmp_path / "study").mkdir()
    (tmp_path / "study/checkpoint.json").write_text(json.dumps(checkpoint()))
    listed = evidence.list_studies(tmp_path)["studies"][0]
    result = evidence.inspect_study(tmp_path, "study")
    assert listed["completed_runs"] == 3
    assert result["evidence_id"] == "checkpoint:" + listed["sha256"]


def test_demo_distinguishes_partner_hypotheses_and_does_not_export_traces():
    a = heating.diagnose("demo-a", "heating-response")
    c = heating.diagnose("demo-c", "heating-response")
    assert a["ranked"][0]["hypothesis"] == "reduced_heating_effectiveness"
    assert c["ranked"][0]["hypothesis"] == "increased_heat_loss"
    assert not a["raw_traces_exported"]
    assert "samples" not in a or isinstance(a["samples"], int)
    assert "synthetic" in a["kind"]


def test_candidate_validation_and_limits():
    result = heating.validate("demo-a", "heating-response", 1.25)
    assert result["improves_all_scenarios"]
    assert result["physical_shots"] == 0
    assert not result["deployment_authorized"]
    assert not heating.validate("demo-a", "heating-response", 1.4)[
        "improves_all_scenarios"
    ]
    assert not heating.validate("demo-a", "heating-response", 0.5)[
        "improves_all_scenarios"
    ]


@pytest.mark.parametrize(
    "scale,steps", [(float("nan"), 40), (2, 40), (1, 100000), (True, 40), (1, True)]
)
def test_validation_budget(scale, steps):
    with pytest.raises(ValueError):
        heating.validate("demo-a", "heating-response", scale, steps)


def test_tool_allowlist_and_budget():
    box = Toolbox(load_sites(), max_calls=2)
    assert "error" in box.execute("shell", {"command": "echo danger"})
    assert "error" in box.execute("list_cases", {"site": "invented"})
    assert box.execute("list_sites", {})["error"] == "Tool budget exhausted"
    assert box.used == 2


def test_malformed_numeric_arguments_do_not_break_shared_audit():
    box = Toolbox(load_sites())
    result = box.execute(
        "validate_heating",
        {
            "site": "demo-a",
            "case_id": "heating-response",
            "steps": 40,
            "power_scale": float("nan"),
        },
    )
    assert "error" in result
    json.dumps(box.audit, allow_nan=False)


@pytest.fixture
def service(monkeypatch):
    token = "test-token-for-local-service-only"
    monkeypatch.setenv("TEST_SITE_TOKEN", token)
    server = make_server(LocalSite("demo-a", demo=True), token, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def test_remote_site_end_to_end(service):
    remote = RemoteSite("demo-a", service, "TEST_SITE_TOKEN")
    result = remote.call("diagnose_heating", {"case_id": "heating-response"})
    assert result["site"] == "demo-a"
    assert result["ranked"][0]["efficiency"] == 0.7
    assert "samples" not in result or isinstance(result["samples"], int)


def test_remote_auth_and_identity(service, monkeypatch):
    remote = RemoteSite("wrong-site", service, "TEST_SITE_TOKEN")
    with pytest.raises(ValueError, match="identity"):
        remote.call("list_cases", {})
    monkeypatch.setenv("TEST_SITE_TOKEN", "wrong-token")
    with pytest.raises(ValueError, match="rejected"):
        remote.call("list_cases", {})


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "file:///secret",
        "https://user:password@example.com",
        "https://example.com/?token=abc",
    ],
)
def test_remote_url_policy(url):
    with pytest.raises(ValueError):
        RemoteSite("site", url, "TOKEN")


def test_history_ignores_failed_runs_tool_content_and_replays_current_once():
    trace = [
        {"run_id": 1, "event": "message", "data": {"role": "user", "content": "old"}},
        {
            "run_id": 1,
            "event": "response.completed",
            "data": {
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "answer"}],
                        }
                    ]
                }
            },
        },
        {
            "run_id": 2,
            "event": "message",
            "data": {"role": "user", "content": "failed"},
        },
        {
            "run_id": 2,
            "event": "response.output_text.delta",
            "data": {"delta": "partial"},
        },
        {
            "run_id": 3,
            "event": "message",
            "data": {"role": "user", "content": "current"},
        },
    ]
    assert [m["content"] for m in history(trace, 3, "current")] == [
        "old",
        "answer",
        "current",
    ]


class Event:
    def __init__(self, kind, **data):
        self.type = kind
        self.__dict__.update(data)

    def to_dict(self):
        return dict(self.__dict__)


def test_model_loop_executes_real_tools_and_forces_final_answer():
    requests, published = [], []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        if kwargs.get("stream"):
            return iter(
                [
                    Event("response.output_text.delta", delta="Synthetic evidence."),
                    Event("response.completed"),
                ]
            )
        return SimpleNamespace(
            status="completed",
            output=[
                Event(
                    "function_call",
                    name="diagnose_heating",
                    call_id="call1",
                    arguments=json.dumps(
                        {"site": "demo-a", "case_id": "heating-response"}
                    ),
                )
            ],
        )

    box = Toolbox(load_sites(), max_calls=1)
    answer = investigate(
        SimpleNamespace(responses=SimpleNamespace(create=create)),
        SimpleNamespace(emit=published.append),
        box,
        "test-model",
        [{"role": "user", "content": "Investigate"}],
    )
    assert answer == "Synthetic evidence."
    assert len(requests) == 2
    assert "tools" not in requests[-1]
    output = json.loads(requests[-1]["input"][-1]["output"])
    assert output["ranked"][0]["efficiency"] == 0.7
    assert published[-1]["type"] == "response.completed"


def test_incomplete_stream_fails():
    def create(**kwargs):
        if kwargs.get("stream"):
            return iter([Event("response.output_text.delta", delta="partial")])
        return SimpleNamespace(status="completed", output=[])

    with pytest.raises(RuntimeError, match="without a completed"):
        investigate(
            SimpleNamespace(responses=SimpleNamespace(create=create)),
            SimpleNamespace(emit=lambda e: None),
            Toolbox(load_sites()),
            "test-model",
            [{"role": "user", "content": "Question"}],
        )


def test_flower_entrypoint_import_and_state(monkeypatch):
    from flwr.agentapp import AgentApp
    from flwr.app import Context, RecordDict

    from fusion_agent import agent_app

    assert isinstance(agent_app.app, AgentApp)
    monkeypatch.setenv("FLWR_RUNTIME_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("FLWR_RUNTIME_API_KEY", "test-runtime-only")
    monkeypatch.delenv("FUSION_SITES_CONFIG", raising=False)

    def execute(client, events, box, model, messages, max_turns):
        box.execute("list_sites", {})
        return "Completed local integration test"

    monkeypatch.setattr(agent_app, "investigate", execute)
    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={"agent.input": "Investigate", "agent.model": "test"},
    )
    agent_app.main(SimpleNamespace(events=SimpleNamespace(get_trace=list)), context)
    state = context.state["fusion.last_investigation"]
    assert state["tool_calls"] == 1
    assert json.loads(state["audit"])[0]["tool"] == "list_sites"


def test_current_checkpoint_matches_existing_metric_if_available():
    # An independent oracle against the repo's existing metric, read-only.
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    paths = sorted((root / "results/coldstart").glob("*/checkpoint.json"))
    if not paths:
        pytest.skip("No local experiment checkpoints")
    from hfmarl.metrics.curves import shots_to_joint_competence_evaluated
    from hfmarl.metrics.log import RunLog

    payload, digest = evidence.read_checkpoint(paths[-1])
    result = evidence.summary(payload, digest, paths[-1].parent.name)
    for row in result["rows"]:
        key = f"{row['device']}:{row['arm']}"
        logs = [RunLog.from_dict(r) for r in payload["runs"].get(key, [])]
        if not logs:
            continue
        expected = shots_to_joint_competence_evaluated(
            logs,
            payload["manifest"]["tolerances"][row["device"]],
            expected_steps=payload["manifest"]["expected_steps"],
        )
        assert row["reached"] == expected.n_reached
        if expected.n_reached:
            assert row["median_joiner_shots_among_reached"] == expected.median
