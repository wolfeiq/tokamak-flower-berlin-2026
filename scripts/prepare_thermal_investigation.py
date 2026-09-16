"""Bundle the current repo's pure-NumPy closure module into the companion FAB."""
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

root = Path(__file__).resolve().parents[1]
source = root / "hfmarl/identification/closure.py"
target = root / "apps/thermal-investigation/thermal_investigation/_closure.py"
data = source.read_bytes()
target.write_bytes(data)
print(f"Prepared {target.relative_to(root)} from repo closure.py; sha256={hashlib.sha256(data).hexdigest()}")

# The three facilities are real devices from this repo's registry, not three
# tunings of one fixture. Copying the parameters here keeps hfmarl (and TORAX)
# out of the agent environment while leaving the numbers traceable to source.
import json
import math

from hfmarl.devices.registry import DEVICES, OPERATING_POINTS
from hfmarl.envs.torax_config import pedestal_T_keV

SITES = {"A": "diiid_like", "B": "sparc_like", "C": "tcv_like"}

devices = {}
for site, name in SITES.items():
    d = DEVICES[name]
    op = OPERATING_POINTS[name]
    devices[site] = {
        "name": d.name,
        "R_major": d.R_major,
        "a_minor": d.a_minor,
        "B_0": d.B_0,
        "elongation": d.elongation,
        "Ip_nominal": d.Ip_nominal,
        "T_e_keV": op["T_e_keV"],
        "T_i_keV": op["T_i_keV"],
        "n_e": op["n_e"],
        "n_rho": 101,
        # Total auxiliary heating the device can command, and the plasma volume
        # it lands in. Together these set a physical W/m^3, so each facility is
        # heated like itself rather than by a shared magic number.
        "P_aux": max(a.hi for a in d.actuators if a.cluster == "thermal"),
        "volume": 2.0 * math.pi**2 * d.R_major * d.a_minor**2 * d.elongation,
        # Core-transport solves stop at the pedestal top; the H-mode pedestal
        # is a separate physics problem this model does not attempt. Taking the
        # boundary from the repo's own helper keeps the four devices comparable
        # in normalised pressure the same way build_config does.
        "T_pedestal_keV": pedestal_T_keV(d),
        "rho_boundary": 0.85,
    }

devices_path = root / "apps/thermal-investigation/thermal_investigation/_devices.json"
devices_path.write_text(json.dumps(devices, indent=2, sort_keys=True))
print(f"Prepared {devices_path.relative_to(root)} for sites " + ", ".join(
    f"{s}={v['name']}" for s, v in sorted(devices.items())))
