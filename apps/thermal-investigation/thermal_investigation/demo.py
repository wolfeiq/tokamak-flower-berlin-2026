"""Offline scripted replay for UI/policy rehearsal; never labelled a live agent."""
import argparse
from pathlib import Path
from .core import Gateway, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("investigation-output"))
    args = parser.parse_args()
    gateway = Gateway(args.output.parent / "investigation-state" / "disclosure.sqlite")
    for site, kind in [("A", "context"), ("A", "balance"), ("B", "context"),
                       ("C", "context"), ("C", "balance"), ("B", "balance"),
                       ("B", "source_check"), ("A", "source_check"),
                       ("B", "raw_logs"), ("B", "source_check")]:
        gateway.request(site, kind)
    save_report(gateway, args.output, "scripted-replay", "A: elevated apparent transport under an assumed source does not establish a transport fault. B: independent source audit found lower delivery; correction returned transport near reference. This supplies a useful investigative precedent, not A's diagnosis. C: flat profile is an invalid transport analogy. Next: obtain an independently calibrated absorbed-heating estimate at A, with uncertainty. Engineer review required.")
    print(f"Replay written to {args.output}. Serve ONLY this folder: python -m http.server 8765 --bind 127.0.0.1 --directory {args.output}")


if __name__ == "__main__":
    main()
