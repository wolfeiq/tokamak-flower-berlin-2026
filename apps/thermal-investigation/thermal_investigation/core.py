"""Facility-controlled release boundary. No arbitrary record or text export.

This is data minimisation, NOT differential privacy. The synthetic demo hosts
all sites in one process; production sites must own their service and ledger.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

EVIDENCE = {
    "context": (1, None),
    "balance": (2, "context"),
    "source_check": (2, "balance"),
}
FACILITIES = ("A", "B", "C")
CASE = "thermal-demo-v1"


def _closure():
    # The build helper copies the repo's actual pure-NumPy implementation.
    # In a checkout, import it directly without requiring TORAX installation.
    try:
        from hfmarl.identification.closure import manufactured_case, chi_from_power_balance
    except ImportError:
        from ._closure import manufactured_case, chi_from_power_balance
    return manufactured_case, chi_from_power_balance


def analyse_site(site: str) -> dict:
    """Local computation. Raw arrays never leave this function.

    Manufactured steady-state fixtures, not TORAX runs or real plant incidents.
    Comparing commanded with independently verified source is an explicit
    synthetic measurement assumption. No hidden injected labels are returned.
    """
    if site not in FACILITIES:
        raise ValueError("Unknown site")
    manufacture, estimate = _closure()
    case = manufacture(lambda r: np.ones_like(r), T_axis=5.0 if site != "C" else 0.2)
    # At B only, an independent source audit exists in the fixture.
    # A has the SAME observable ambiguity but no source audit.
    assumed_source = case["source"] * 1.6
    result = estimate(case["rho"], case["n_e"], case["T_e"], assumed_source)
    valid = result.chi[np.isfinite(result.chi)]
    usable = bool(valid.size > 10)
    context = {
        "regime": "manufactured-steady-state",
        "geometry": "cylindrical",
        "subsystem": "thermal-transport",
        "gradient_quality": "usable" if usable else "insufficient",
        "symptom": "weak-temperature-response" if usable else "flat-temperature-profile",
    }
    # Coarse bins are fixed; requesters cannot change thresholds or query slices.
    band = "unidentifiable" if not usable else (
        "above-reference" if float(np.median(valid)) > 1.2 else "near-reference"
    )
    balance = {
        "apparent_transport": band,
        "source_basis": "assumed-command",
        "source_verified": False,
        "causal_conclusion": "not-identifiable-from-this-evidence",
        "profile_shape": [round(float(x), 1) for x in
                          (case["T_e"][::10] / max(float(case["T_e"].max()), 1e-9))],
        "profile_units": "normalised-temperature",
    }
    source_check = {
        "independent_audit_available": site == "B",
        "finding": "delivered-below-assumed" if site == "B" else "unknown",
        "corrected_transport": "near-reference" if site == "B" else "not-computed",
    }
    if site == "B":
        corrected = estimate(case["rho"], case["n_e"], case["T_e"], case["source"])
        median = float(np.nanmedian(corrected.chi))
        source_check["corrected_transport"] = (
            "near-reference" if 0.8 <= median <= 1.2 else "above-reference"
        )
    return {"context": context, "balance": balance, "source_check": source_check}


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
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS releases (site TEXT, case_id TEXT, kind TEXT, cost INTEGER, payload TEXT, PRIMARY KEY(site,case_id,kind))")
            db.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, event TEXT)")

    def connect(self):
        return sqlite3.connect(self.ledger, timeout=20)

    def request(self, site, kind):
        # Unknown input is never echoed into shared logs (could contain secrets).
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if type(site) is not str or site not in FACILITIES or type(kind) is not str or kind not in EVIDENCE:
                result = {"status": "denied", "reason": "unsupported-request"}
            else:
                rows = db.execute("SELECT kind,cost,payload FROM releases WHERE site=? AND case_id=?", (site, CASE)).fetchall()
                prior = {r[0]: json.loads(r[2]) for r in rows}
                spent = sum(r[1] for r in rows)
                cost, prerequisite = EVIDENCE[kind]
                base = {"site": site, "evidence": kind}
                if kind in prior:
                    result = {**prior[kind], "cached": True, "spent": spent}
                elif prerequisite and prerequisite not in prior:
                    result = {**base, "status": "denied", "reason": "prerequisite-missing", "spent": spent}
                elif kind != "context" and prior["context"]["finding"]["gradient_quality"] == "insufficient":
                    result = {**base, "status": "denied", "reason": "analogy-not-applicable", "spent": spent}
                elif spent + cost > self.allowance:
                    result = {**base, "status": "denied", "reason": "budget-exhausted", "spent": spent}
                else:
                    finding = analyse_site(site)[kind]
                    evidence_id = hashlib.sha256(f"{CASE}:{site}:{kind}".encode()).hexdigest()[:12]
                    result = {**base, "status": "released", "finding": finding,
                              "evidence_id": evidence_id, "cost": cost, "spent": spent + cost,
                              "allowance": self.allowance, "cached": False,
                              "provenance": "synthetic-manufactured-v1"}
                    db.execute("INSERT INTO releases VALUES (?,?,?,?,?)", (site, CASE, kind, cost, json.dumps(result)))
            db.execute("INSERT INTO audit(event) VALUES (?)", (json.dumps(result, allow_nan=False),))
        return result

    def events(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT event FROM audit ORDER BY id")]


def save_report(gateway, output, mode, narrative=""):
    """Only released evidence goes into the dashboard; no raw fixtures."""
    import shutil
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"mode": mode, "case": CASE, "events": gateway.events(),
              "narrative": narrative, "limitation": "Synthetic steady-state cases; no real reactor diagnosis or formal privacy guarantee."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    dashboard = Path(__file__).with_name("dashboard.html")
    # Flower FABs do not include HTML assets. The local checkout hosts the UI.
    if dashboard.exists():
        shutil.copyfile(dashboard, output / "index.html")
    return report
