"""Bundle the current repo's pure-NumPy closure module into the companion FAB."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
source = root / "hfmarl/identification/closure.py"
target = root / "apps/thermal-investigation/thermal_investigation/_closure.py"
data = source.read_bytes()
target.write_bytes(data)
print(f"Prepared {target.relative_to(root)} from repo closure.py; sha256={hashlib.sha256(data).hexdigest()}")
