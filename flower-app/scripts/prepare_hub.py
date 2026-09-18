"""Stage the explicit public AgentApp sources; never collect local run state."""

from pathlib import Path
from shutil import copyfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "fusion-investigator"
    if output.exists():
        raise SystemExit(f"Stage already exists; inspect it before replacing: {output}")
    output.mkdir(parents=True)
    files = [
        *sorted((root / "fusion_agent").rglob("*.py")),
        root / "fusion_agent" / "thermal" / "_devices.json",
        *(root / name for name in ("pyproject.toml", "LICENSE.md", "THERMAL_LICENSE.md", "MERGE.md", ".gitignore")),
    ]
    for source in files:
        target = output / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)
    copyfile(root / "scripts" / "HUB_README.md", output / "README.md")
    print(f"Staged {len(files) + 1} public files: {output}")


if __name__ == "__main__":
    main()
