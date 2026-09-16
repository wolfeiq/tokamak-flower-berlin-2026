"""Model-free demo, local evidence inspection, and facility service launcher."""

import argparse
import json
import os

from .service import make_server
from .sites import load_sites
from .tools import Toolbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Operator-owned site configuration JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Deterministic synthetic tool demo; no LLM")
    commands.add_parser("sites")
    tool = commands.add_parser("tool")
    tool.add_argument("name")
    tool.add_argument("arguments", help="JSON object")
    serve = commands.add_parser("serve")
    serve.add_argument("--site", required=True)
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token-env", default="FUSION_SITE_TOKEN")
    args = parser.parse_args()
    sites = load_sites(args.config or os.environ.get("FUSION_SITES_CONFIG"))
    box = Toolbox(sites)
    if args.command == "serve":
        site = sites[args.site]
        if site.describe()["transport"] != "local":
            parser.error("Serve a local adapter, not another remote endpoint")
        server = make_server(site, os.environ.get(args.token_env, ""), port=args.port)
        print(
            f"Analysis service for {args.site} on loopback port {args.port}", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    if args.command == "demo":
        result = {
            "mode": "deterministic synthetic tool demo; no LLM or Flower run",
            "evidence": [
                box.execute(
                    "diagnose_heating", {"site": s, "case_id": "heating-response"}
                )
                for s in sites
                if "diagnose_heating" in sites[s].describe()["capabilities"]
            ],
            "validation": box.execute(
                "validate_heating",
                {
                    "site": "demo-a",
                    "case_id": "heating-response",
                    "power_scale": 1.4,
                    "steps": 40,
                },
            ),
        }
    elif args.command == "sites":
        result = box.execute("list_sites", {})
    else:
        result = box.execute(args.name, json.loads(args.arguments))
    print(json.dumps(result, indent=2, allow_nan=False))
    if "error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
