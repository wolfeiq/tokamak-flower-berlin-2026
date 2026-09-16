from types import SimpleNamespace
import json
from thermal_investigation.agent_app import facility_turn, investigate, MAX_CALLS
from thermal_investigation.core import Gateway


class Call:
    type = 'function_call'
    call_id = 'test-call'
    def __init__(self, name, args):
        self.name, self.arguments = name, json.dumps(args)
    def model_dump(self, **kwargs):
        return dict(type=self.type, name=self.name, arguments=self.arguments, call_id=self.call_id)


class FakeClient:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []
        self.responses = self
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output=self.calls)


def test_facility_cannot_export_free_text(tmp_path):
    gate = Gateway(tmp_path / 'db')
    client = FakeClient([SimpleNamespace(type='message', text='SECRET')])
    out = facility_turn(client, 'model', gate, 'A', 'context')
    assert out['status'] == 'withheld'
    assert 'SECRET' not in json.dumps(out)


def test_facility_request_runs_gateway(tmp_path):
    gate = Gateway(tmp_path / 'db')
    client = FakeClient([Call('release_requested', {})])
    assert facility_turn(client, 'model', gate, 'A', 'balance')['status'] == 'denied'
    assert facility_turn(client, 'model', gate, 'A', 'context')['status'] == 'released'
    assert 'chi_true' not in json.dumps(client.requests)


def test_unbounded_model_calls_get_bounded(tmp_path):
    gate = Gateway(tmp_path / 'db')
    client = FakeClient([Call('unknown', {}) for _ in range(MAX_CALLS + 5)])
    _, items = investigate(client, 'model', gate, 'test')
    assert len(client.requests) == 1
    assert any('tool-budget-exhausted' in x.get('output', '') for x in items)
    assert not gate.events()
