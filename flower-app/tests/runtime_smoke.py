"""Exercise real SuperLink/FAB execution with a scripted Responses test server.

No API key, external provider, or real LLM is involved. This tests the Flower
transport contract, actual tool execution and streamed events, not model quality.
Run directly with the app's Python environment. All services are loopback-only.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "smoke-output"
    output.mkdir(exist_ok=True)
    control_port, fleet_port = free_port(), free_port()
    observations = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items = body.get("input", [])
            tool_outputs = [
                x
                for x in items
                if isinstance(x, dict) and x.get("type") == "function_call_output"
            ]
            observations.append({"stream": body.get("stream"), "outputs": tool_outputs})
            if not body.get("stream"):
                calls = (
                    []
                    if tool_outputs
                    else [
                        {
                            "type": "function_call",
                            "id": "fc_smoke",
                            "call_id": "smoke_call",
                            "name": "diagnose_heating",
                            "status": "completed",
                            "arguments": json.dumps(
                                {"site": "demo-a", "case_id": "heating-response"}
                            ),
                        }
                    ]
                )
                response = {
                    "id": "resp_smoke",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "smoke-model",
                    "output": calls,
                }
                if not body.get("tools"):
                    response["output"] = [
                        {
                            "id": "msg_title",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Synthetic heating investigation",
                                    "annotations": [],
                                }
                            ],
                        }
                    ]
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            assert tool_outputs, "Agent never supplied tool evidence"
            result = json.loads(tool_outputs[-1]["output"])
            assert result["ranked"][0]["efficiency"] == 0.7
            text = "SMOKE_OK: real Flower runtime executed synthetic site analysis."
            message = {
                "id": "msg_smoke",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            response = {
                "id": "resp_final",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "smoke-model",
                "output": [message],
            }
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**response, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_smoke",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                },
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": response,
                },
            ]
            encoded = "".join(
                "data: " + json.dumps(e) + "\n\n" for e in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = output / "flower-home"
    home.mkdir(exist_ok=True)
    (home / "config.toml").write_text(
        f'[superlink.smoke]\naddress = "127.0.0.1:{control_port}"\ninsecure = true\n'
    )
    env = dict(os.environ)
    env.update(
        FLWR_HOME=str(home),
        PYTHONUTF8="1",
        FLWR_TELEMETRY_ENABLED="0",
        FLWR_MODEL_API_ENDPOINT=f"http://127.0.0.1:{server.server_port}/v1/responses",
        FLWR_MODEL_API_KEY="",
        NO_PROXY="127.0.0.1,localhost",
    )
    env.pop("FUSION_SITES_CONFIG", None)
    bin_dir = Path(sys.executable).parent
    for key in list(env):
        if key.upper() == "PATH":
            env.pop(key)
    env["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    suffix = ".exe" if os.name == "nt" else ""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = None
    try:
        with (output / "superlink.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(bin_dir / ("flower-superlink" + suffix)),
                    "--insecure",
                    "--disable-runtime-dependency-installation",
                    "--port",
                    str(control_port),
                    "--fleet-api-address",
                    f"127.0.0.1:{fleet_port}",
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "SuperLink exited; inspect smoke-output/superlink.log"
                    )
                try:
                    with socket.create_connection(
                        ("127.0.0.1", control_port), timeout=0.5
                    ):
                        break
                except OSError:
                    time.sleep(0.2)
            run = subprocess.run(
                [
                    str(bin_dir / ("flwr" + suffix)),
                    "run",
                    ".",
                    "smoke",
                    "--stream",
                    "--run-config",
                    'agent.model="smoke-model"',
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                creationflags=flags,
                check=False,
            )
            (output / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            print(run.stdout, run.stderr)
            if run.returncode or "SMOKE_OK" not in run.stdout + run.stderr:
                raise RuntimeError(
                    "Flower runtime smoke failed; inspect smoke-output logs"
                )
            (output / "verification.json").write_text(
                json.dumps(
                    {
                        "harness": "Flower SuperLink 1.37",
                        "provider": "scripted test double",
                        "real_llm": False,
                        "requests": len(observations),
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=flags,
                    check=False,
                )
            else:
                process.terminate()
            process.wait(timeout=15)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
