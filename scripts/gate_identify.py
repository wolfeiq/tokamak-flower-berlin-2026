#!/usr/bin/env python
"""Gate: is inverse identification possible in THIS TORAX install?

Three questions, all of which must be answered before the identification
experiments mean anything. Cheap to run; each failure has a different fix.

  1. Does `prescribed` transport accept chi as a PROFILE over rho?
     The closure-free story (identify chi(rho) rather than a scalar multiplier)
     depends on it. In TORAX `main`, PrescribedTransportModel declares chi_i,
     chi_e, D_e, V_e as TimeVaryingArray and only `model_name` is JAX_STATIC --
     so they are traced leaves and differentiable. Whether 1.4.3 exposes them
     identically is exactly what this checks.

  2. Do gradients flow to each candidate unknown?
     A zero gradient means either the parameter does not affect the
     observations (unidentifiable) or it is not being traced (a plumbing bug).
     Those look identical from the outside and need opposite fixes.

  3. Is the Fisher matrix well conditioned over the candidate set?
     Several candidates are degenerate with each other. Fitting a degenerate
     set yields a confident answer at an arbitrary point of the degenerate
     direction -- not an error, a wrong number.

    uv run python scripts/gate_identify.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.identification.crlb import crlb, jacobian_by_finite_differences  # noqa: E402
from hfmarl.identification.torax_inverse import (  # noqa: E402
    CANDIDATE_UNKNOWNS,
    build_loss,
    check_gradient_flow,
)
from hfmarl.util.report import record  # noqa: E402


def probe_prescribed_transport(say) -> bool:
    """Can we set chi_i as a radial profile and have TORAX accept it?

    THE NAME WAS THE PROBLEM. This only ever tried ``model_name='prescribed'``,
    which does not exist in TORAX 1.4.3 -- the transport tags are exactly
    ``qlknn``, ``constant`` and ``combined``. So it reported that the
    closure-free loop could not close, when in fact ``ConstantTransportModel``
    already declares ``chi_i``/``chi_e``/``D_e``/``V_e`` as ``TimeVaryingArray``
    and accepts a full chi(rho) profile. A gate that answers "impossible" when
    the answer is "yes, spelled differently" is worse than no gate, so the
    ``constant`` form is now tried FIRST and the others are kept only to detect
    version drift.
    """
    import torax

    say("1. transport accepting a chi(rho) profile")
    n_rho = 25
    rho = np.linspace(0, 1, n_rho)
    chi_profile = 1.0 + 2.0 * np.exp(-(((rho - 0.5) / 0.15) ** 2))

    # v1.4.3 flat form first, then main's registry form.
    forms = {
        # 1.4.3, and the one that actually works: `constant` is not "a constant
        # chi", it is "chi supplied by the config rather than by a model", and
        # the config may supply a profile.
        "1.4.3: transport={'model_name':'constant', 'chi_i': {rho: chi}}": lambda d: {
            "model_name": "constant",
            "chi_i": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
            "chi_e": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
            "D_e": {0.0: {float(r): 0.5 for r in rho}},
            "V_e": {0.0: {float(r): -0.1 for r in rho}},
        },
        "flat (other): transport={'model_name':'prescribed', 'chi_i': {...}}": lambda d: {
            "model_name": "prescribed",
            "chi_i": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
            "chi_e": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
            "D_e": 0.5,
            "V_e": -0.1,
        },
        "registry (main): transport={'core_transport_models': {...}}": lambda d: {
            "core_transport_models": {
                "prescribed": {
                    "model_name": "prescribed",
                    "chi_i": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
                    "chi_e": {0.0: {float(r): float(c) for r, c in zip(rho, chi_profile)}},
                }
            }
        },
    }

    device = get("iter_like")
    ok_any = False
    for label, make in forms.items():
        cfg = build_config(device, n_rho=n_rho)
        cfg["transport"] = make(device)
        try:
            torax.ToraxConfig.from_dict(cfg)
            say(f"   OK   {label}")
            ok_any = True
        except Exception as e:
            say(f"   no   {label}")
            say(f"        {type(e).__name__}: {str(e)[:150]}")
    if ok_any:
        say("   -> the closure-free loop CAN close: identify chi(rho), write it")
        say("      back as a transport profile, re-solve, compare.")
    else:
        say("   FAIL: no transport form accepted a chi(rho) profile.")
        say("   The closure-free chi identification cannot feed its result back")
        say("   into TORAX. The identification itself still works (it only needs")
        say("   profiles and sources), but you lose the forward-validation loop.")
        say("   Check the installed version's transport tags before believing")
        say("   this: 1.4.3 offers exactly qlknn / constant / combined.")
    say()
    return ok_any


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    try:
        import torax  # noqa: F401
    except ImportError as e:
        print(f"FAIL: TORAX not importable ({e}). See SETUP.md.")
        return 1

    prescribed_ok = probe_prescribed_transport(say)

    # --- 2. gradient flow ----------------------------------------------
    say("2. gradient flow to each candidate unknown")
    device = get(args.device)
    names = list(CANDIDATE_UNKNOWNS)
    truth = np.array([CANDIDATE_UNKNOWNS[n][0] for n in names])
    lo = np.array([CANDIDATE_UNKNOWNS[n][1] for n in names])
    hi = np.array([CANDIDATE_UNKNOWNS[n][2] for n in names])

    cfg = build_config(device, t_final=args.delta_t * (args.steps + 2))
    try:
        # Synthetic "observations" from the nominal parameters.
        loss_and_grad, simulate = build_loss(
            cfg, names, {"T_e": np.zeros((args.steps, 25))}, args.delta_t, args.steps
        )
        obs = {k: np.asarray(v) for k, v in simulate(truth).items()}
        loss_and_grad, _ = build_loss(cfg, names, obs, args.delta_t, args.steps)
        flow = check_gradient_flow(loss_and_grad, truth * 1.1, names)
    except Exception as e:
        say(f"   FAIL: could not build the differentiable loss: {type(e).__name__}: {e}")
        say("   This is the first contact between the inverse harness and TORAX;")
        say("   the traceback below is the useful output.")
        traceback.print_exc()
        if not args.no_record:
            record("Gate identify (FAILED)", "```\n" + "\n".join(lines) + "\n```")
        return 1

    dead = []
    for n in names:
        g = flow[n]
        mark = "OK  " if g > 1e-12 else "DEAD"
        if g <= 1e-12:
            dead.append(n)
        say(f"   {mark} |dL/d({n})| = {g:.4e}")
    if dead:
        say(f"   {len(dead)} parameter(s) receive NO gradient. Either the")
        say("   observations do not depend on them, or they are not traced.")
        say("   The CRLB below distinguishes the two.")
    say()

    # --- 3. identifiability --------------------------------------------
    say("3. identifiability over the candidate set (CRLB)")
    live = [n for n in names if n not in dead] or names
    idx = [names.index(n) for n in live]

    def forward(theta_sub):
        full = truth.copy()
        full[idx] = theta_sub
        return np.concatenate([np.asarray(v).ravel() for v in simulate(full).values()])

    J = jacobian_by_finite_differences(forward, truth[idx])
    sigma = 0.02 * float(np.std(np.concatenate(
        [np.asarray(v).ravel() for v in obs.values()])))
    rep = crlb(J, max(sigma, 1e-12), live, truth[idx])
    for ln in rep.summary().splitlines():
        say("   " + ln)
    say()

    usable = [n for n, r in zip(live, rep.relative_crlb) if r < 0.1]
    say(f"VERDICT: {len(usable)}/{len(names)} candidates are identifiable at 10%:")
    for n in usable:
        say(f"   {n}")
    say(f"prescribed chi(rho) profile supported: {'YES' if prescribed_ok else 'NO'}")
    say()
    say("Use the identifiable set in scripts/exp_recover_scalars.py.")

    status = "PASSED" if usable else "FAILED"
    if not args.no_record:
        p = record(f"Gate identify -- {device.name} ({status})",
                   "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return 0 if usable else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
