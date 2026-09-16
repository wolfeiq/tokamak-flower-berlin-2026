"""Compatibility launcher for the merged Fusion app's optional TORAX exporter."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "flower-app/scripts/generate_thermal_fixtures.py"), run_name="__main__")
