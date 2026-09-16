#!/usr/bin/env python
"""Gate 0d -- can the actuators actually reach the setpoint?

THE FAILURE THIS CATCHES
------------------------
A setpoint outside a device's reachable band is not a hard control problem, it
is an absent one: the policy saturates at one rail and every parameter vector
scores identically. Federating across such a device contributes noise at best.

`envs/task.validate_against_limits` already refuses a setpoint too HIGH for the
safety envelope. Nothing checked the other three ways this goes wrong, all of
which are silent:

  * the setpoint is ABOVE what full command can produce  -> saturated, no
    interior optimum, gate 1 can never pass;
  * the setpoint is BELOW what zero command produces     -> same, other rail;
  * the device starts ALREADY IN VIOLATION               -> every episode
    terminates on step 1 regardless of action, the reward is exactly constant
    and no gradient exists at all.

WHAT IT FOUND, AND WHAT CHANGED
-------------------------------
On the first run against live TORAX this failed 3 of 4 devices. `task.py` held
one absolute setpoint (beta_N = 2.0) and one absolute tolerance (0.15) for a
device set whose reachable band spanned a factor of 50, because density was
normalised as a Greenwald fraction and temperature was not. `tcv_like` began
every episode already past the beta limit and scored exactly -550.00 for every
policy; `iter_like` saturated 4x short of target; `sparc_like` had 0.012 of
authority because its real 25 MW of ICRF was switched off.

Three fixes, in `torax_config.py`, `devices/registry.py` and `envs/task.py`:
pedestal temperature scaled as a*B_0 so every device sits at comparable
normalised pressure; the primary auxiliary heat channel renamed `aux_heat` and
given each device's real power; and setpoint AND tolerance expressed as
fractions of the device's measured band. All four devices now pass, and pass
Phase 1.

The tolerance half matters as much as the setpoint half: 0.15 in beta_N was
7.5% of DIII-D's control authority and 197% of SPARC's, so even a reachable
setpoint would have made the task a different difficulty on every device --
and shots-to-threshold would then measure device calibration rather than
learning, which is the quantity SPEC.md 5 compares across the federation.

PASS, per device, requires all three:
  1. the setpoint lies strictly inside the reachable band;
  2. the band is wider than the task tolerance -- otherwise the actuators
     cannot move beta_N far enough to matter even where they do bracket it;
  3. the shot at zero command survives without an immediate limit violation.

A tolerance below what a diagnostic could resolve is reported as a WARNING, not
a failure: it is a well-posed simulation task that would not transfer.

RE-RUN THIS AFTER ANY PLANT CHANGE and paste the bands into
`devices/registry.BETA_N_BANDS`. They are keyed by task as well as device,
because the transport model moves them -- `iter_like` has 0.444 of authority
under `constant` and 0.011 under `qlknn`.

    uv run python scripts/gate_authority.py --task easy --plot
    uv run python scripts/gate_authority.py --task brutal
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.task import PRESETS  # noqa: E402
from hfmarl.envs.torax_env import ToraxDeviceEnv, ToraxUnavailableError  # noqa: E402
from hfmarl.util.report import peak_rss_mb, record  # noqa: E402

# Command levels to probe, as a fraction of each actuator's envelope. The ends
# bracket the band; the interior points show whether the response is monotone,
# because a non-monotone one means the band is not an interval and the
# endpoints alone would misreport it.
LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Smallest beta_N tolerance worth calling a control objective.
#
# Normalising the tolerance to each device's control authority is what makes the
# task equally hard everywhere, but it has a floor that normalisation cannot see:
# a target of beta_N = 0.808 +- 0.0011 (iter_like on `hard`) is a perfectly
# well-posed simulation problem and a meaningless experimental one, because no
# diagnostic resolves beta_N to a part in a thousand. Reported as a WARNING
# rather than a failure -- the run is still valid, its transfer to a real
# machine is not.
MIN_MEANINGFUL_TOLERANCE: float = 0.01


def tracking_floor(device: str, task: str, rows: list[dict]) -> dict:
    """Mean tracking error under perfect static feedforward, no feedback.

    NOT A FLOOR, despite what this function was called and what its
    docstring claimed. It measures ONE controller: open-loop, commanding
    at each step the level the sweep says holds the current target in
    steady state. Feedback can beat it, and measured on `moderate` a
    plain PI controller does -- tcv_like reaches 0.0373 against this
    quantity's 0.1189, three times better, because feedback corrects the
    lag that open-loop cannot see.

    It remains useful as a REFERENCE: it is what the plant gives you for
    free from a sweep, so a controller that cannot beat it is adding
    nothing. It bounds nothing from below.

    WHY THIS IS A SEPARATE QUESTION FROM AUTHORITY
    ----------------------------------------------
    The sweep above answers a STATIC question: does the actuator envelope
    bracket the setpoint. A moving setpoint adds a dynamic one it cannot
    answer -- can the plant keep up? A first-order plant with confinement
    time tau chasing a ramp of slope s carries a lag error of tau*s however
    good the controller is, and a tolerance below that floor is
    unsatisfiable by construction.

    That failure looks EXACTLY like a learning failure, which is the same
    reason this gate exists at all -- it just cost 1200 shots on 'moderate'
    to find out, because the static check passed.

    Measured, not estimated. The plant is driven with a perfect static
    feedforward: at each step, command the level that the sweep says holds
    the current target in steady state. No controller can beat perfect
    knowledge of the static map, so what remains is lag plus the unavoidable
    startup transient -- neither of which a feedback controller has to
    accept in full.
    """
    lv = np.array([2.0 * r["level"] - 1.0 for r in rows], float)
    bn = np.array([r["beta_N"] for r in rows], float)
    ok = np.isfinite(bn)
    if ok.sum() < 2:
        return {"available": False}
    # Invert the static map: command needed to hold a given beta_N.
    coef = np.polyfit(bn[ok], lv[ok], 1)

    # This one needs the REAL target -- it measures what a perfect static
    # feedforward attains against it -- so it cannot use the placeholder
    # above, and it cannot run before the band has been recorded. On a
    # brand-new task, say so and carry on: the sweep above is what the
    # run was for, and the floor is available on the next pass.
    try:
        env = ToraxDeviceEnv(get(device), task=task,
                             strict_task_check=False)
    except KeyError as exc:
        return {"available": False, "reason": str(exc)}
    env.reset()
    for _ in range(env.task.steps_per_shot):
        tgt = float(env.target_beta_N)
        cmd = float(np.clip(np.polyval(coef, tgt), -1.0, 1.0))
        # Drive EVERY actuator at the commanded level, because that is how
        # `probe` built the static map above. Inverting a map measured with
        # both actuators and then driving one of them reports a lag floor
        # that is really a missing heating source.
        action = np.full(env.n_actions, cmd)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    errs = np.array([abs(st.scalars["beta_N"] - st.target)
                     for st in env.trajectory
                     if st.ok and "beta_N" in st.scalars], float)
    if errs.size == 0:
        return {"available": False}
    tol = float(env.task.tolerance)
    return {
        "available": True,
        "mean": float(errs.mean()),
        "max": float(errs.max()),
        "settled_mean": float(errs[len(errs) // 3:].mean()),
        "tolerance": tol,
        "fraction_of_budget": float(errs.mean() / tol) if tol > 0 else float("inf"),
    }


def probe(device, task: str, level: float) -> dict:
    """One open-loop shot at a fixed command. Returns what the limits see."""
    # strict_task_check is off deliberately: this gate must be able to
    # characterise a device whose task is already known to be unsatisfiable,
    # which is the whole point of running it.
    # A PLACEHOLDER SETPOINT, so a task with no measured band yet can be
    # measured at all. `resolve_for` needs the band, the band comes from
    # this script, and this script built an env -- so a new preset could
    # never be bootstrapped, and the KeyError told you to run the tool
    # that could not run. A command sweep never reads the target, so an
    # absolute placeholder costs nothing and `resolve_for` is a no-op.
    spec = PRESETS[task].for_band_measurement()
    env = ToraxDeviceEnv(get(device), task=spec, strict_task_check=False)
    env.reset()
    action = np.full(env.n_actions, 2.0 * level - 1.0)  # [-1,1] <- fraction

    violated_at = None
    for i in range(env.task.steps_per_shot):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated and violated_at is None:
            violated_at = i + 1
        if terminated or truncated:
            break

    last = env.trajectory[-1] if env.trajectory else None
    s = last.scalars if last is not None else {}
    return {
        "level": level,
        "P_aux_MW": s.get("P_aux_total", float("nan")) / 1e6,
        "beta_N": s.get("beta_N", float("nan")),
        "q95": s.get("q95", float("nan")),
        "fgw": s.get("fgw_n_e_line_avg", float("nan")),
        "steps_survived": len(env.trajectory),
        "steps_possible": env.task.steps_per_shot,
        "violated_at": violated_at,
        "violations": list(last.violations) if last is not None else [],
    }


def assess(device: str, rows: list[dict], task: str) -> tuple[bool, list[str]]:
    """Verdict and reasons for one device.

    The task is resolved against this device first: presets hold fractions of
    the measured band, so comparing a raw preset's `base` (0.5) against a beta_N
    band would be comparing a fraction to a physical quantity.
    """
    spec = PRESETS[task] if isinstance(task, str) else task
    try:
        spec = spec.resolve_for(get(device))
    except (KeyError, ValueError):
        # No measured band yet -- which is the situation this gate exists to
        # get us out of. Fall back to the unresolved spec so the gate can still
        # report the band it just measured.
        pass
    t_final = spec.episode_length
    target_lo = min(spec.setpoint.target(t, t_final) for t in np.linspace(0, t_final, 200))
    target_hi = max(spec.setpoint.target(t, t_final) for t in np.linspace(0, t_final, 200))

    betas = np.array([r["beta_N"] for r in rows], dtype=float)
    betas = betas[np.isfinite(betas)]
    reasons: list[str] = []
    if betas.size == 0:
        return False, ["no finite beta_N at any command level -- the simulation failed"]

    lo, hi = float(betas.min()), float(betas.max())
    ok = True

    if target_hi >= hi:
        ok = False
        reasons.append(
            f"setpoint reaches {target_hi:.2f} but full command only produces "
            f"beta_N = {hi:.2f}. The policy saturates at maximum power and every "
            "parameter vector scores alike; there is no control problem here."
        )
    if target_lo <= lo:
        ok = False
        reasons.append(
            f"setpoint falls to {target_lo:.2f} but ZERO command already gives "
            f"beta_N = {lo:.2f}. The actuators can only push further away."
        )
    if (hi - lo) < spec.tolerance:
        ok = False
        reasons.append(
            f"actuator authority is {hi - lo:.3f} in beta_N, narrower than the "
            f"task tolerance {spec.tolerance:.3f}. Full-scale command barely "
            "moves the controlled variable."
        )

    if spec.tolerance < MIN_MEANINGFUL_TOLERANCE:
        reasons.append(
            f"WARNING (not a failure): the resolved tolerance is "
            f"{spec.tolerance:.4f} in beta_N, below the "
            f"{MIN_MEANINGFUL_TOLERANCE:g} a real diagnostic could resolve. "
            "The task is well-posed in simulation and not transferable; this "
            "device has too little control authority for this preset."
        )

    zero = next((r for r in rows if r["level"] == 0.0), None)
    if zero is not None and zero["violated_at"] == 1:
        ok = False
        reasons.append(
            f"the shot at ZERO command violates {zero['violations']} on step 1. "
            "Every episode terminates immediately whatever the policy does, so "
            "the reward is exactly constant and carries no gradient."
        )
    return ok, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=sorted(DEVICES))
    ap.add_argument("--task", default="easy", choices=sorted(PRESETS))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="results/authority")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    spec = PRESETS[args.task]
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"task     : {args.task}  (mode {spec.setpoint_mode})")
    say(f"setpoint : {spec.setpoint.kind}, base {spec.setpoint.base:g}, "
        f"amplitude {spec.setpoint.amplitude:g}, tolerance {spec.tolerance:g}")
    if spec.setpoint_mode == "band_fraction":
        say("           (fractions of each device's measured band -- resolved "
            "per device below)")
    say(f"episode  : {spec.episode_length:g}s in {spec.delta_t_a:g}s steps")
    say()

    results: dict[str, list[dict]] = {}
    verdicts: dict[str, tuple[bool, list[str]]] = {}
    floors: dict[str, dict] = {}

    for name in args.devices:
        say(f"{name}")
        rows = []
        for level in LEVELS:
            try:
                r = probe(name, args.task, level)
            except ToraxUnavailableError as e:
                print(f"FAIL: {e}")
                return 1
            rows.append(r)
            flag = "" if r["violated_at"] is None else f"  VIOLATED@{r['violated_at']}"
            say(f"  cmd {level:4.0%}  P_aux {r['P_aux_MW']:7.2f} MW   "
                f"beta_N {r['beta_N']:8.3f}   q95 {r['q95']:6.2f}   "
                f"fgw {r['fgw']:5.3f}   {r['steps_survived']:2d}/"
                f"{r['steps_possible']:2d} steps{flag}")
        results[name] = rows
        try:
            rs = spec.resolve_for(get(name))
            say(f"  resolved target beta_N {rs.setpoint.base:.3f} "
                f"+-{rs.tolerance:.4f}")
        except (KeyError, ValueError) as e:
            say(f"  (no measured band recorded yet: {e})")
        verdicts[name] = assess(name, rows, args.task)
        ok, reasons = verdicts[name]

        if spec.setpoint.kind != "constant":
            fl = tracking_floor(name, args.task, rows)
            floors[name] = fl
            if fl.get("available"):
                say(f"  tracking floor (perfect static feedforward): "
                    f"mean |err| {fl['mean']:.4f}, after settling "
                    f"{fl['settled_mean']:.4f}, tolerance {fl['tolerance']:.4f}")
                say(f"    -> {fl['fraction_of_budget']:.0%} of the error "
                    f"budget is spent on lag before the controller acts")
                if fl["settled_mean"] > fl["tolerance"]:
                    ok = False
                    reasons = reasons + [
                        "the setpoint moves faster than OPEN-LOOP control "
                        f"can follow: perfect static feedforward leaves "
                        f"{fl['settled_mean']:.4f} of mean error after "
                        f"settling, against a tolerance of "
                        f"{fl['tolerance']:.4f}. Feedback may still close "
                        "this -- measured, a PI controller beats the "
                        "open-loop number on some devices -- so treat it "
                        "as a warning about the setpoint rate, not proof "
                        "that the task is impossible."]
                    verdicts[name] = (ok, reasons)
        say(f"  -> {'PASS' if ok else 'FAIL'}")
        for reason in reasons:
            say(f"     {reason}")
        say()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"authority_{args.task}.json").write_text(
        json.dumps({"task": args.task, "levels": list(LEVELS), "results": results},
                   indent=1),
        encoding="utf-8",
    )
    say(f"wrote {out / f'authority_{args.task}.json'}")

    if args.plot:
        path = plot(results, args.task, out)
        say(f"wrote {path}")

    say()
    failed = [d for d, (ok, _) in verdicts.items() if not ok]
    say(f"summary: {len(verdicts) - len(failed)}/{len(verdicts)} devices can "
        f"reach the '{args.task}' setpoint")
    say(f"peak RSS: {peak_rss_mb():.0f} MB")
    say()
    if failed:
        say(f"FAIL: {', '.join(failed)} cannot be controlled to this setpoint.")
        say("Gate 1 cannot pass on these devices and the Phase 5 matrix would")
        say("spend its budget on clients that produce no learning signal. The")
        say("setpoint is a per-device quantity, not a constant -- set it from")
        say("the measured band above (the repo already does exactly this for")
        say("density, which is specified as a Greenwald fraction rather than in")
        say("m^-3, so that four devices start in comparable states).")
        status = "FAILED"
        rc = 1
    else:
        say("PASS: every device's actuators bracket the setpoint with margin.")
        status = "PASSED"
        rc = 0

    if not args.no_record:
        p = record(f"Gate 0d -- actuator authority, task '{args.task}' ({status})",
                   "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return rc


def plot(results: dict[str, list[dict]], task: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hfmarl.envs.limits import DEFAULT_LIMITS

    spec = PRESETS[task]
    beta_hard = next(l.hard for l in DEFAULT_LIMITS if l.name == "beta_N")
    beta_soft = next(l.soft for l in DEFAULT_LIMITS if l.name == "beta_N")
    names = list(results)

    all_beta = np.array(
        [r["beta_N"] for name in names for r in results[name]], dtype=float
    )
    all_beta = all_beta[np.isfinite(all_beta)]
    top = float(all_beta.max()) * 1.7 if all_beta.size else 10.0

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhspan(beta_hard, top, color="#d62728", alpha=0.10, lw=0)
    ax.axhline(beta_hard, color="#d62728", lw=1.2,
               label=f"beta_N hard limit ({beta_hard:g})")
    ax.axhline(beta_soft, color="#ff7f0e", lw=1.0, ls=":",
               label=f"soft edge ({beta_soft:g})")

    for i, name in enumerate(names):
        betas = np.array([r["beta_N"] for r in results[name]], dtype=float)
        good = np.isfinite(betas)
        if not good.any():
            continue
        lo, hi = betas[good].min(), betas[good].max()

        # The setpoint is per device now: presets carry fractions of this
        # device's own band, so there is no single line to draw across the plot.
        try:
            rs = spec.resolve_for(get(name))
            target, tol = rs.setpoint.base, rs.tolerance
            reaches = lo < target < hi
        except (KeyError, ValueError):
            target = tol = None
            reaches = False

        colour = "#2ca02c" if reaches else "#d62728"
        ax.plot([i, i], [lo, hi], color=colour, lw=11, solid_capstyle="round",
                alpha=0.35, zorder=2)
        ax.plot([i] * int(good.sum()), betas[good], "o", color=colour, ms=5,
                zorder=3, mec="white", mew=0.8)
        if target is not None:
            ax.plot([i - 0.28, i + 0.28], [target, target], color="#1f77b4",
                    lw=2.4, zorder=5,
                    label="resolved setpoint" if i == 0 else None)
            ax.fill_between([i - 0.28, i + 0.28], target - tol, target + tol,
                            color="#1f77b4", alpha=0.22, lw=0, zorder=4,
                            label="tolerance" if i == 0 else None)
        ax.annotate(f"{lo:.2f}\u2013{hi:.2f}", (i, hi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=colour)

    ax.set_yscale("symlog", linthresh=1.0, linscale=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$\beta_N$ reachable at end of shot")
    ax.set_title(
        f"Actuator authority vs the '{task}' setpoint\n"
        "setpoint and tolerance are fractions of each device's OWN measured "
        "band, so the task is equally hard everywhere",
        fontsize=10.5)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0.0, top)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()

    path = out / f"authority_{task}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
