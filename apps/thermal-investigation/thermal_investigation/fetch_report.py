"""Recover a live run's report from the Flower log stream into a local dashboard.

Runs on the operator's machine, not inside the FAB. SuperGrid deployments need
not provide an artifact provider, so `flwr pull` can fail while the run itself
succeeded. The AgentApp fences its sanitized report into the run log; this reads
it back and writes the same investigation-output that the replay produces.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .agent_app import REPORT_BEGIN, REPORT_END

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def extract(log: str) -> dict:
    """Return the fenced report, preferring the last one the run emitted."""
    text = ANSI.sub("", log)
    if REPORT_BEGIN not in text or REPORT_END not in text:
        raise SystemExit(
            "No fenced report in this run's log. The run may predate this feature, "
            "may still be running, or may have failed before writing a report."
        )
    body = text.rsplit(REPORT_BEGIN, 1)[1].split(REPORT_END, 1)[0]
    return json.loads(body.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("superlink", nargs="?", default="supergrid")
    parser.add_argument("--output", type=Path, default=Path("investigation-output"))
    args = parser.parse_args()

    proc = subprocess.run(
        ["flwr", "log", args.run_id, args.superlink, "--show"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"flwr log failed:\n{proc.stdout}{proc.stderr}")

    report = extract(proc.stdout + proc.stderr)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    dashboard = Path(__file__).with_name("dashboard.html")
    if dashboard.exists():
        shutil.copyfile(dashboard, args.output / "index.html")

    released = sum(1 for e in report["events"] if e.get("status") == "released")
    denied = sum(1 for e in report["events"] if e.get("status") == "denied")
    print(
        f"Recovered {report['mode']} run {args.run_id} to {args.output}: "
        f"{len(report['events'])} events, {released} released, {denied} denied.\n"
        f"Serve ONLY this folder: python -m http.server 8765 --bind 127.0.0.1 "
        f"--directory {args.output}"
    )


if __name__ == "__main__":
    main()
