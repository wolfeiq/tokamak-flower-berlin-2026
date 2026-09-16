"""Bounded toy heat-balance investigation; never a plasma-control interface.

All values are dimensionless. The simulator is explicitly not TORAX or an MHD
model. Raw local traces are reduced to diagnostic scores before leaving a site.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import mean

PROFILES = {
    "demo-a": (0.70, 0.16, 0.0),
    "demo-b": (0.75, 0.16, 0.0),
    "demo-c": (1.00, 0.25, 0.0),
}


def case(site: str) -> dict:
    efficiency, loss, bias = PROFILES[site]
    temperature, samples = 1.0, []
    for i in range(48):
        power = 0.20 + 0.06 * math.sin(i * 0.7) + 0.03 * math.cos(i * 0.23)
        following = temperature + 0.25 * (efficiency * power - loss * temperature)
        samples.append(
            {
                "temperature": temperature + bias,
                "power": power,
                "next_temperature": following + bias,
            }
        )
        temperature = following
    return {"case_id": "heating-response", "dt": 0.25, "samples": samples}


def identity(site: str) -> dict:
    digest = hashlib.sha256(json.dumps(case(site), sort_keys=True).encode()).hexdigest()
    return {
        "site": site,
        "case_id": "heating-response",
        "evidence_id": f"synthetic:{digest}",
        "kind": "synthetic dimensionless heat-balance demo; not a real tokamak",
        "model": "T_next = T + dt * (efficiency * power - loss * T)",
    }


def diagnose(site: str, case_id: str) -> dict:
    if case_id != "heating-response":
        raise ValueError("Unknown case")
    data = case(site)
    # Fit on one portion, score on withheld samples. This ranks explanations,
    # not causal identifiability; richer measurements are needed in a plant.
    train, heldout = data["samples"][:32], data["samples"][32:]

    def mse(samples, efficiency, loss):
        return mean(
            (
                s["temperature"]
                + data["dt"] * (efficiency * s["power"] - loss * s["temperature"])
                - s["next_temperature"]
            )
            ** 2
            for s in samples
        )

    hypotheses = {
        "nominal_response": [(1.0, 0.16)],
        "reduced_heating_effectiveness": [(x / 100, 0.16) for x in range(40, 101, 5)],
        "increased_heat_loss": [(1.0, x / 100) for x in range(16, 33)],
    }
    scores = []
    for name, grid in hypotheses.items():
        efficiency, loss = min(grid, key=lambda p: mse(train, *p))
        scores.append(
            {
                "hypothesis": name,
                "efficiency": efficiency,
                "loss": loss,
                "heldout_rmse": math.sqrt(mse(heldout, efficiency, loss)),
            }
        )
    scores.sort(key=lambda s: s["heldout_rmse"])
    return {
        **identity(site),
        "samples": len(data["samples"]),
        "ranked": scores,
        "raw_traces_exported": False,
        "limitations": [
            "Small, noiseless synthetic case with a restricted model family.",
            "A ranking is not a confirmed physical cause; check diagnostics.",
            "No tearing, disruption, or real actuator dynamics modeled.",
        ],
        "next_diagnostic": "Independently check delivered heating power and thermometer calibration.",
    }


def validate(site: str, case_id: str, power_scale: float, steps: int = 40) -> dict:
    if (
        type(power_scale) not in (float, int)
        or not math.isfinite(power_scale)
        or not 0.5 <= power_scale <= 1.5
    ):
        raise ValueError("power_scale must be finite and between 0.5 and 1.5")
    if type(steps) is not int or not 10 <= steps <= 80:
        raise ValueError("steps must be an integer between 10 and 80")
    diagnostic = diagnose(site, case_id)
    fitted = diagnostic["ranked"][0]

    def rollout(scale, efficiency, loss):
        temperature, errors, violations = 1.0, [], 0
        for _ in range(steps):
            # A fixed proportional controller used only in this toy sandbox.
            power = min(0.4, max(0, (0.16 + 0.5 * (1 - temperature)) * scale))
            temperature += 0.25 * (efficiency * power - loss * temperature)
            errors.append(abs(temperature - 1))
            violations += int(not 0.6 <= temperature <= 1.4)
        return {"mean_tracking_error": mean(errors), "limit_exceedances": violations}

    scenarios = [
        (fitted["efficiency"] * e, fitted["loss"] * loss)
        for e, loss in ((0.9, 1.1), (1, 1), (1.1, 0.9))
    ]
    baseline = [rollout(1, *p) for p in scenarios]
    candidate = [rollout(power_scale, *p) for p in scenarios]
    return {
        **identity(site),
        "validation": "bounded synthetic model sensitivity study",
        "power_scale": power_scale,
        "steps_per_rollout": steps,
        "rollouts": 6,
        "physical_shots": 0,
        "baseline": baseline,
        "candidate": candidate,
        "improves_all_scenarios": all(
            c["mean_tracking_error"] < b["mean_tracking_error"]
            and c["limit_exceedances"] == 0
            for b, c in zip(baseline, candidate)
        ),
        "deployment_authorized": False,
        "limitations": "Uses the fitted toy model; not independent physical validation.",
    }
