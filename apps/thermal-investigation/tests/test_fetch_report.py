"""The live-run recovery path: log fencing must survive a real log stream."""
import json

import pytest

from thermal_investigation.agent_app import REPORT_BEGIN, REPORT_END, status_of, text_from
from thermal_investigation.fetch_report import extract

REPORT = {"mode": "live-flower", "case": "thermal-demo-v1", "events": [], "narrative": "n"}


def fenced(report):
    return f"{REPORT_BEGIN}\n{json.dumps(report)}\n{REPORT_END}\n"


def test_extract_ignores_surrounding_log_noise():
    log = "INFO:  Installing application dependencies...\n" + fenced(REPORT) + "INFO:\n"
    assert extract(log) == REPORT


def test_extract_strips_ansi_colour_codes():
    assert extract("\x1b[92mINFO \x1b[0m: start\n" + fenced(REPORT)) == REPORT


def test_extract_prefers_the_last_report_in_a_retried_log():
    first = {**REPORT, "mode": "live-flower-incomplete"}
    assert extract(fenced(first) + fenced(REPORT))["mode"] == "live-flower"


def test_extract_refuses_a_log_without_a_report():
    # A failed or still-running run must not yield a silently empty dashboard.
    with pytest.raises(SystemExit):
        extract("INFO:  Start `flwr-agentapp` process\n")


def test_text_from_recovers_output_when_no_deltas_streamed():
    final = {"response": {"output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "advisory"}]},
    ]}}
    assert text_from(final) == "advisory"


def test_text_from_ignores_reasoning_only_output():
    assert text_from({"response": {"output": [{"type": "reasoning", "summary": []}]}}) == ""


def test_status_of_reports_truncation_reason():
    final = {"response": {"status": "incomplete",
                          "incomplete_details": {"reason": "max_output_tokens"}}}
    assert status_of(final) == "incomplete/max_output_tokens"


def test_status_of_handles_a_stream_that_ended_without_a_terminal_event():
    assert status_of(None) == "unknown/no-reason"
