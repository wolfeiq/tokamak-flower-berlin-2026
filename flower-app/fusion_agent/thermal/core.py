"""Facility-controlled release boundary. No arbitrary record or text export.

This is data minimisation, NOT differential privacy. The synthetic demo hosts
all sites in one process; production sites must own their service and ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

EVIDENCE = {
    "context": (1, None),
    "balance": (2, "context"),
    "source_check": (2, "balance"),
}
FACILITIES = ("A", "B", "C")
CASE = "thermal-reduced-v2"


def _closure():
    # Always use the packaged snapshot: local and hosted execution must agree.
    from ._closure import chi_from_power_balance

    return chi_from_power_balance


def _devices():
    """Machine parameters copied from hfmarl/devices/registry.py at build time."""
    path = Path(__file__).with_name("_devices.json")
    if not path.exists():
        raise RuntimeError(
            "_devices.json is missing. Run scripts/prepare_thermal_investigation.py "
            "from the repository root before building or running this app."
        )
    return json.loads(path.read_text())


# Per-site scenario. These are investigation circumstances -- how hard the
# machine was heated, and whether anyone independently measured the delivered
# power -- NOT physics fudge factors. The profiles themselves come out of the
# solve in profiles.py on each device's real parameters.
SCENARIOS = {
    # Heated normally; apparent transport is elevated only because the
    # commanded power is assumed delivered.
    "A": {"power_fraction": 1.0, "source_audit": False},
    # Same observable ambiguity, but here an independent audit of delivered
    # power exists, so the ambiguity is resolvable at B and only at B.
    "B": {"power_fraction": 1.0, "source_audit": True},
    # Barely heated: the profile never reaches the critical gradient, so it
    # carries no information about transport at all.
    "C": {"power_fraction": 0.001, "source_audit": False},
}

# The commanded source is overestimated by this factor relative to what the
# plasma actually absorbed. This is the whole diagnostic ambiguity: assuming it
# away turns a heating-delivery fault into an apparent transport anomaly.
ASSUMED_SOURCE_FACTOR = 1.6
# Below the critical normalised gradient a profile is too flat to identify
# transport, whatever estimator is used.
USABLE_R_OVER_LT = 4.0
_WINDOW = slice(10, 90)


def _band(ratio: float) -> str:
    """Coarse, fixed bins. Requesters cannot move these thresholds."""
    return (
        "above-reference"
        if ratio > 1.25
        else ("below-reference" if ratio < 0.8 else "near-reference")
    )


def _torax_fixture(site: str):
    """Real TORAX profiles, when scripts/generate_thermal_fixtures.py has run.

    Absent by default: TORAX needs jax>=0.10.0, which publishes no macOS x86_64
    wheel, so the reduced solve in profiles.py is what most machines get. The
    two paths are deliberately interchangeable and always distinguishable by the
    provenance string that travels with every released finding.

    chi_true is ground truth the reduced solver knows and TORAX output does not
    carry, so the reference here is the transport recovered from the DELIVERED
    source -- the same quantity, obtained the way an experimentalist would.
    """
    configured = os.environ.get("FUSION_THERMAL_FIXTURES")
    if not configured:
        return None
    path = Path(configured)
    blob = json.loads(path.read_text())
    if site not in blob:
        raise ValueError("Configured fixture set is incomplete")
    fixture = blob[site]
    r = np.asarray(fixture["r"], dtype=float)
    n_e = np.asarray(fixture["n_e"], dtype=float)
    T_e = np.asarray(fixture["T_e_keV"], dtype=float) * 1.0e3 * 1.602176634e-19
    source = np.asarray(fixture["source"], dtype=float)
    if (
        r.ndim != 1
        or len(r) < 20
        or any(
            a.shape != r.shape or not np.all(np.isfinite(a))
            for a in (r, n_e, T_e, source)
        )
        or np.any(np.diff(r) <= 0)
        or np.any(n_e <= 0)
        or np.any(T_e <= 0)
    ):
        raise ValueError("Invalid simulation profile grid or values")
    if fixture.get("device") != _devices()[site]["name"]:
        raise ValueError("Fixture device does not match configured facility")
    chi_reference = _closure()(r, n_e, T_e, source).chi
    return {
        "r": r,
        "n_e": n_e,
        "T_e": T_e,
        "source": source,
        "chi_true": chi_reference,
        "provenance": fixture["provenance"],
        "steady_state_verified": fixture.get("steady_state_verified") is True,
    }


def analyse_site(site: str) -> dict:
    """Local computation. Raw arrays never leave this function.

    Profiles are solved, not manufactured: see profiles.py. Each facility is a
    different real machine from the repo's device registry, so the profiles
    differ because the machines differ. Ground truth (chi_true, the delivered
    source) stays inside this function and is never returned.
    """
    if site not in FACILITIES:
        raise ValueError("Unknown site")
    from . import profiles as _profiles

    estimate = _closure()
    device = _devices()[site]
    scenario = SCENARIOS[site]
    case = _torax_fixture(site) or _profiles.simulate(
        device, scenario["power_fraction"]
    )

    r, n_e, T_e = case["r"], case["n_e"], case["T_e"]
    R_major = float(device["R_major"])
    r_over_lt = R_major * np.abs(np.gradient(T_e, r)) / np.maximum(T_e, 1e-30)
    interior = slice(max(1, len(r) // 10), max(2, 9 * len(r) // 10))
    usable = bool(np.median(r_over_lt[interior]) >= USABLE_R_OVER_LT)

    # What an analyst sees: the COMMANDED power, which overstates delivery.
    assumed_source = case["source"] * ASSUMED_SOURCE_FACTOR
    apparent = estimate(r, n_e, T_e, assumed_source)
    reference = float(np.nanmedian(case["chi_true"][interior]))
    apparent_ratio = float(np.nanmedian(apparent.chi)) / max(reference, 1e-30)

    context = {
        "balance_assumption": (
            "verified-steady-state"
            if not case["provenance"].startswith("torax-")
            or case.get("steady_state_verified")
            else "unverified-steady-state"
        ),
        "regime": "simulation-snapshot"
        if case["provenance"].startswith("torax-")
        else "solved-steady-state",
        "geometry": "cylindrical",
        "subsystem": "thermal-transport",
        # Coarse machine class, not the device name: enough to judge whether an
        # analogy is even worth testing, without identifying the facility.
        "field_class": "high-field"
        if float(device["B_0"]) >= 5.0
        else "moderate-field",
        "size_class": "compact" if float(device["a_minor"]) < 0.6 else "medium",
        "gradient_quality": "usable" if usable else "insufficient",
        "symptom": "weak-temperature-response"
        if usable
        else "flat-temperature-profile",
    }
    balance = {
        "apparent_transport": _band(apparent_ratio) if usable else "unidentifiable",
        "source_basis": "assumed-command",
        "source_verified": False,
        "causal_conclusion": "not-identifiable-from-this-evidence",
        "profile_shape": [
            round(float(x), 1)
            for x in np.interp(np.linspace(r[0], r[-1], 11), r, T_e)
            / max(float(T_e.max()), 1e-30)
        ],
        "profile_radius": [
            round(float(x), 3) for x in np.linspace(r[0], r[-1], 11) / device["a_minor"]
        ],
        "profile_units": "normalised-temperature",
    }
    source_check = {
        "independent_audit_available": bool(scenario["source_audit"]),
        "finding": "delivered-below-assumed" if scenario["source_audit"] else "unknown",
        "corrected_transport": "not-computed",
    }
    if scenario["source_audit"]:
        # Re-run the same estimator against the audited delivered power.
        corrected = estimate(r, n_e, T_e, case["source"])
        corrected_ratio = float(np.nanmedian(corrected.chi)) / max(reference, 1e-30)
        source_check["corrected_transport"] = _band(corrected_ratio)
    return {
        "context": context,
        "balance": balance,
        "source_check": source_check,
        "provenance": case.get("provenance", "reduced-1d-transport-v1"),
    }


class Gateway:
    """Durable, transaction-protected per-case/site disclosure accounting.

    Case identity and allowance belong to the service, not the caller. Repeated
    identical releases return cached evidence; new sessions do not reset budget.
    Only fixed evidence codes can be requested, no filters or free-form output.
    """

    def __init__(self, ledger: Path, allowance: int = 5):
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.allowance = allowance
        # Invalidate cached evidence when fixtures, solver or scenarios change.
        digest = hashlib.sha256(CASE.encode())
        for name in ("core.py", "profiles.py", "_closure.py", "_devices.json"):
            digest.update(Path(__file__).with_name(name).read_bytes())
        configured = os.environ.get("FUSION_THERMAL_FIXTURES")
        if configured:
            digest.update(Path(configured).read_bytes())
        self.case_id = CASE + ":" + digest.hexdigest()[:16]
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS releases (site TEXT, case_id TEXT, kind TEXT, cost INTEGER, payload TEXT, PRIMARY KEY(site,case_id,kind))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, event TEXT)"
            )

    def connect(self):
        return sqlite3.connect(self.ledger, timeout=20)

    def request(self, site, kind):
        # Unknown input is never echoed into shared logs (could contain secrets).
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                type(site) is not str
                or site not in FACILITIES
                or type(kind) is not str
                or kind not in EVIDENCE
            ):
                result = {"status": "denied", "reason": "unsupported-request"}
            else:
                rows = db.execute(
                    "SELECT kind,cost,payload FROM releases WHERE site=? AND case_id=?",
                    (site, self.case_id),
                ).fetchall()
                prior = {r[0]: json.loads(r[2]) for r in rows}
                spent = sum(r[1] for r in rows)
                cost, prerequisite = EVIDENCE[kind]
                base = {"site": site, "evidence": kind}
                if kind in prior:
                    result = {**prior[kind], "cached": True, "spent": spent}
                elif prerequisite and prerequisite not in prior:
                    result = {
                        **base,
                        "status": "denied",
                        "reason": "prerequisite-missing",
                        "spent": spent,
                    }
                elif (
                    kind != "context"
                    and prior["context"]["finding"]["gradient_quality"]
                    == "insufficient"
                ):
                    result = {
                        **base,
                        "status": "denied",
                        "reason": "analogy-not-applicable",
                        "spent": spent,
                    }
                elif (
                    kind != "context"
                    and prior["context"]["finding"].get("balance_assumption")
                    == "unverified-steady-state"
                ):
                    result = {
                        **base,
                        "status": "denied",
                        "reason": "balance-assumption-unverified",
                        "spent": spent,
                    }
                elif spent + cost > self.allowance:
                    result = {
                        **base,
                        "status": "denied",
                        "reason": "budget-exhausted",
                        "spent": spent,
                    }
                else:
                    analysis = analyse_site(site)
                    finding = analysis[kind]
                    provenance = analysis["provenance"]
                    evidence_id = hashlib.sha256(
                        f"{self.case_id}:{site}:{kind}:{json.dumps(finding, sort_keys=True)}".encode()
                    ).hexdigest()[:16]
                    result = {
                        **base,
                        "status": "released",
                        "finding": finding,
                        "evidence_id": evidence_id,
                        "cost": cost,
                        "spent": spent + cost,
                        "allowance": self.allowance,
                        "cached": False,
                        "provenance": provenance,
                    }
                    db.execute(
                        "INSERT INTO releases VALUES (?,?,?,?,?)",
                        (site, self.case_id, kind, cost, json.dumps(result)),
                    )
            result["case_id"] = self.case_id
            db.execute(
                "INSERT INTO audit(event) VALUES (?)",
                (json.dumps(result, allow_nan=False),),
            )
        return result

    def events(self):
        with self.connect() as db:
            return [
                event
                for r in db.execute("SELECT event FROM audit ORDER BY id")
                if (event := json.loads(r[0])).get("case_id") == self.case_id
            ]


def save_report(gateway, output, mode, narrative=""):
    """Only released evidence goes into the dashboard; no raw fixtures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": mode,
        "case": CASE,
        "events": gateway.events(),
        "narrative": narrative,
        "limitation": "Synthetic steady-state cases; no real reactor diagnosis or formal privacy guarantee.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
