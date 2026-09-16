"""Run one experimental condition across the device set.

This is the loop SPEC.md Phase 5 describes and nothing had yet executed: local
search on each device, updates into the asynchronous buffer, a personalised
aggregate back into each policy.

WHAT IS AND IS NOT BEING COMPARED
---------------------------------
Every condition uses the same local search, the same common initialisation, the
same seeds and the same per-device shot budget. The only thing that varies is
what happens between rounds. Anything else varying would make a difference in
shots-to-competence uninterpretable.

Adoption costs a real shot. When a client installs an aggregate it fires one
episode to find out what it received, and that episode is logged like any
other. Not counting it would let a federated condition import performance for
free and report a speedup that a real machine could not reproduce.
"""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from hfmarl.agents.policy import make_policy
from hfmarl.agents.search import HillClimber, fire_shot
from hfmarl.devices.registry import DEVICES, encoded_states, get as get_device
from hfmarl.envs.task import get as get_task
from hfmarl.envs.torax_env import ToraxDeviceEnv
from hfmarl.experiments.conditions import (
    CENTRALISED,
    CONDITIONS,
    FEDAVG_NAIVE,
    FEDBUFF_SIMILARITY,
    FEDBUFF_UNIFORM,
    ISOLATED,
)
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import (
    ClientUpdate,
    region_from_state,
    region_from_states,
    suggest_bandwidth,
)
from hfmarl.metrics.log import RunLog
from hfmarl.physics.torax_adapter import mean_state, state_from_env

# Every agent in this build is the thermal one: it owns `aux_heat` and `ecrh`.
# Particle and current agents arrive with Phase 2, and until they do there is
# exactly one federation channel.
ONLY_CLUSTER = "thermal"

# PROTOCOL.md 3. Each rung adds exactly ONE ingredient to the one above it,
# so a gap between adjacent rungs is attributable to that ingredient and to
# nothing else. The previous design -- scratch, uniform, similarity -- could
# not separate pretraining from multiple sources from repeated exchange,
# because every federated arm enjoyed all three at once.
#
#   scratch               its own shots only
#   single_source         + one pretrained source, chosen on design info
#   handover_merge        + all sources, trained alone, merged once
#   federated_uniform     + exchange DURING training, equal weights
#   federated_similarity  + physics weighting
#
# `conventional` is the sixth rung and lives in scripts/exp_conventional.py:
# it is not a parameter vector, so it does not share this code path.
COLD_START_ARMS: tuple[str, ...] = (
    "scratch", "single_source", "handover_merge",
    "federated_uniform", "federated_similarity",
)


def _resolve_task(task):
    """Accept a preset name or a TaskSpec. The catastrophe experiment needs
    band-restricted variants that are not presets.
    """
    return get_task(task) if isinstance(task, str) else task

def _region_or_point(states, fallback):
    """A measured region, or the nominal point when nothing was usable.

    `region_from_states` raises when every state is non-finite, because the
    caller has to choose between a nominal fallback and skipping the round.
    Here the choice is the fallback, and it is a DEGRADED mode: the nominal
    table is the one measurement showed wrong by 2x in beta_N and 7x in nu*,
    so a run that leans on it has a weaker claim than one that does not.
    Counted, so the write-up can say how often it happened.
    """
    if states:
        try:
            return region_from_states(states), False
        except ValueError:
            pass
    return region_from_state(fallback), True


class RoleControlUnavailable(RuntimeError):
    """The role-blind negative control cannot be run with one cluster."""


def run_condition(
    condition: str,
    device_names: list[str],
    task_name: str,
    seed: int,
    shots: int,
    local_shots: int = 25,
    stagger: dict[str, int] | None = None,
    bandwidth: float | None = None,
    rule: str = "geomedian",
    align: bool = True,
    clip_factor: float | None = 2.0,
    accept_if_better: bool = False,
    eval_every: int = 10,
    use_safety: bool = True,
    scramble: dict[str, str] | None = None,
    verbose_every: int = 0,
    say=print,
) -> tuple[list[RunLog], dict, FedBuffServer]:
    """Train every device under one condition.

    Returns the logs, a record of how the federation behaved (the only
    evidence that it did), and the server itself -- a device joining later
    inherits that buffer, and rebuilding it would be training the same
    peers twice.
    """
    spec = CONDITIONS[condition]

    if condition == FEDAVG_NAIVE:
        raise RoleControlUnavailable(
            "naive FedAvg is the ROLE-BLIND negative control: it aggregates "
            "thermal with particle and current agents to show that role "
            "matching is load-bearing. This build has one cluster "
            f"({ONLY_CLUSTER!r}), so role-blind and role-matched aggregation "
            "are the same computation and running it would report a duplicate "
            "of fedbuff_uniform as if it were a control. It becomes "
            "measurable when Phase 2 adds the particle and current agents."
        )

    task = _resolve_task(task_name)
    devices = [get_device(n) for n in device_names]
    states = encoded_states()
    envs = {
        d.name: ToraxDeviceEnv(d, task=task, seed=seed) for d in devices
    }

    # One initialisation, shared. See the note in federation/server.py: this is
    # a precondition for weight averaging to mean anything, not a convenience.
    probe = envs[devices[0].name]
    theta0 = make_policy(probe._observe().shape[0], probe.n_actions,
                         seed=seed).get_flat().copy()

    policies = {
        d.name: make_policy(envs[d.name]._observe().shape[0],
                            envs[d.name].n_actions, seed=seed)
        for d in devices
    }
    climbers = {}
    for d in devices:
        # A source must see the same proposals alone, in a merge arm, and
        # in any leave-one-out fold. Its position in the supplied subset
        # is not part of the experiment's random seed.
        i = sorted(DEVICES).index(d.name)
        c = HillClimber(n_params=theta0.size, seed=seed * 1000 + i)
        c.seed_from(theta0)
        climbers[d.name] = c

    logs = {d.name: RunLog(condition=condition, device=d.name, seed=seed)
            for d in devices}
    # Where each device ACTUALLY operated during the last block of shots.
    # Measured, per shot, because a device's own excursion across its
    # command envelope is larger than its distance to its nearest peer for
    # two of the four devices here -- so "the distance between device A and
    # device B" is not a well-posed question about static points.
    visited = {d.name: [] for d in devices}
    spent = {d.name: 0 for d in devices}

    if bandwidth is None:
        bandwidth = suggest_bandwidth([states[d.name] for d in devices])
    ref_policy = policies[devices[0].name]
    server = FedBuffServer(
        bandwidth=bandwidth, use_similarity=spec.use_similarity,
        rule=rule, align=align, clip_factor=clip_factor,
        use_safety=use_safety,
        shape=(ref_policy.obs_dim, ref_policy.hidden, ref_policy.act_dim),
    )

    # The centralised upper bound is one policy trained on every device's
    # shots. AUDIT: it used to share a single climber across the per-device
    # loop, so a candidate evaluated on device B was compared against a best
    # return earned on device A -- an improvement from -10 to -1 on TCV was
    # rejected because ITER had once returned -0.01. That optimises no
    # defined objective and is not an upper bound on anything.
    #
    # A candidate is now scored on the WHOLE device set, and every one of
    # those shots is charged. Pooling data is supposed to be expensive in
    # shots and cheap in nothing; the condition exists to bound what perfect
    # data sharing could buy, not to buy it at a discount.
    shared = HillClimber(n_params=theta0.size, seed=seed * 1000 + 977)
    shared.seed_from(theta0)

    history: list[dict] = []
    rejected = 0
    unbarred = 0  # adoptions made with no unbiased incumbent estimate
    # SELECTION COSTS SHOTS, and the cost is not shared evenly across the
    # ladder. An arm that exchanges has to measure its incumbent before it
    # can refuse an aggregate, so its sources spend shots on that; an arm
    # that never exchanges spends none. At 80 pretraining shots with an
    # exchange every 20 that is three or four shots per client, four to five
    # percent of the budget -- small, real, and invisible unless counted.
    acceptance_shots = 0
    fallbacks = 0
    rnd = 0
    while any(spent[d.name] < shots for d in devices):
        rnd += 1

        if condition == CENTRALISED:
            # One candidate, scored on every device, all shots charged.
            for _ in range(local_shots):
                if all(spent[d.name] >= shots for d in devices):
                    break
                cand = shared.propose()
                total = 0.0
                scored = 0
                for d in devices:
                    if spent[d.name] >= shots:
                        continue
                    rec, r = fire_shot(envs[d.name], policies[d.name], cand,
                                       spent[d.name])
                    logs[d.name].add(rec)
                    spent[d.name] += 1
                    total += r
                    scored += 1
                    try:
                        visited[d.name].append(state_from_env(envs[d.name]))
                    except Exception:
                        pass
                if scored:
                    # The joint objective is the MEAN across the devices
                    # scored, so a candidate is never rewarded for the
                    # accident of how many machines happened to have budget.
                    shared.observe(cand, total / scored)
            continue

        for d in devices:
            name = d.name
            env, pol = envs[name], policies[name]
            climber = climbers[name]
            for _ in range(min(local_shots, shots - spent[name])):
                # One shot in every `eval_every` fires the incumbent
                # unperturbed. It costs budget like any other shot --
                # which is the point: certifying a controller on the
                # exploration shots is not something a real machine
                # gets to do either.
                if (eval_every and spent[name] % eval_every == 0
                        and np.isfinite(climber.best_return)):
                    rec, ev_total = fire_shot(env, pol, climber.best,
                                              spent[name], evaluation=True)
                    logs[name].add(rec)
                    spent[name] += 1
                    # This is an UNPERTURBED draw of the current incumbent,
                    # which is exactly what reject-if-worse needs as its bar.
                    # It was being fired, logged, charged -- and thrown away,
                    # while the acceptance test used the running maximum
                    # instead.
                    climber.note_evaluation(ev_total)
                    continue
                cand = climber.propose()
                rec, total = fire_shot(env, pol, cand, spent[name])
                logs[name].add(rec)
                spent[name] += 1
                climber.observe(cand, total)
                try:
                    visited[name].append(state_from_env(env))
                except Exception:
                    pass  # a failed solve has no state to encode

        if verbose_every and rnd % verbose_every == 0:
            mean_best = (shared.best_return if condition == CENTRALISED
                         else np.mean([climbers[n].best_return
                                       for n in [d.name for d in devices]]))
            say(f"    round {rnd:4d}  shots/device {min(spent.values()):5d}  "
                f"mean best {mean_best:9.3f}")

        if not spec.federated:
            continue

        # --- publish ----------------------------------------------------
        for d in devices:
            period = (stagger or {}).get(d.name, 1)
            if period > 1 and rnd % period:
                continue  # this machine is not running this round
            # How this update was earned, not just how big it is. A peer
            # whose recent shots kept crossing limits is teaching that.
            recent = logs[d.name].shots[-local_shots:]
            viol = (sum(1 for r in recent if r.violated) / len(recent)
                    if recent else 0.0)
            # SCRAMBLE CONTROL: publish under another device's measured
            # coordinates. If the weighting still works, it was never
            # the physics doing the work.
            src = (scramble or {}).get(d.name, d.name)
            # A leave-one-out permutation can name the held-out device,
            # which has no source-training trajectory. Use its nominal
            # coordinates in that case, and count the fallback below.
            seen = visited.get(src, [])[-local_shots:]
            here = mean_state(seen) if seen else states[src]
            # The REGION these shots covered, not just its centre.
            region, degraded = _region_or_point(seen, states[src])
            fallbacks += int(degraded)
            server.publish(ClientUpdate(
                device=d.name,
                cluster=ONLY_CLUSTER,
                weights={"flat": climbers[d.name].best.copy()},
                state=here,
                region=region,
                n_samples=local_shots,
                round_produced=rnd,
                violation_rate=viol,
            ))

        # --- aggregate and adopt ----------------------------------------
        # `reference` is what makes alignment and clipping work: peers are
        # permuted into THIS client's basis and clipped around its current
        # model, rather than around an arbitrary peer's.
        aggregates = {
            d.name: server.aggregate_for(
                d.name,
                (mean_state(visited[d.name][-local_shots:])
                 if visited[d.name] else states[d.name]),
                ONLY_CLUSTER, rnd, reference=climbers[d.name].best,
                target_region=_region_or_point(
                    visited[d.name][-local_shots:], states[d.name])[0])
            for d in devices
        }
        for d in devices:
            name = d.name
            agg = aggregates[name]
            if agg is None or spent[name] >= shots:
                continue
            # TRUE SELECTION, as opposed to the robust rules above: those
            # bound how much a bad peer can move the aggregate, but nothing
            # is ever refused. This refuses -- and unlike the robust rules it
            # can reject an aggregate that every peer agreed on and that is
            # still wrong HERE.
            #
            # ONE DRAW AGAINST ONE DRAW. The bar used to be `best_return`, a
            # running maximum over every shot fired, which rises with the
            # shot count whatever the controller does -- so a genuinely
            # better aggregate was adopted 76% of the time after one shot and
            # 8% after a hundred. See FINDINGS.md.
            #
            # The incumbent's bar has to be CURRENT. Candidates fire
            # continuously and every improvement installs a new incumbent,
            # which invalidates the last evaluation -- so by the time an
            # exchange comes round there is usually no unbiased estimate, and
            # adopting blind there would make --select quietly inert. One
            # shot buys the comparison; it is logged and charged like any
            # other.
            bar, source = climbers[name].acceptance_bar()
            if (accept_if_better and source == "none"
                    and spent[name] < shots):
                rec_inc, inc_total = fire_shot(
                    envs[name], policies[name], climbers[name].best,
                    spent[name], evaluation=True)
                logs[name].add(rec_inc)
                spent[name] += 1
                acceptance_shots += 1
                climbers[name].note_evaluation(inc_total)
                bar, source = climbers[name].acceptance_bar()
            if spent[name] >= shots:
                continue
            rec, total = fire_shot(envs[name], policies[name], agg, spent[name])
            logs[name].add(rec)
            spent[name] += 1
            if accept_if_better and source == "none":
                # Budget ran out before a bar could be bought. Refusing here
                # would be the old bias in another form.
                unbarred += 1
                climbers[name].adopt(agg, total)
            elif accept_if_better and total < bar:
                rejected += 1
            else:
                climbers[name].adopt(agg, total)

        history.append({"round": rnd, "weights": server.weight_table()})

    for name, log in logs.items():
        log.notes = (f"condition={condition} task={task_name} "
                     f"local_shots={local_shots} bandwidth={bandwidth:.3f}")

    return list(logs.values()), {
        "condition": condition,
        "bandwidth": float(bandwidth),
        "rounds": rnd,
        "local_shots": local_shots,
        "shots_by_device": {name: len(log) for name, log in logs.items()},
        "total_shots": sum(len(log) for log in logs.values()),
        "federated": spec.federated,
        "rule": rule,
        "align": align,
        "clip_factor": clip_factor,
        "accept_if_better": accept_if_better,
        "aggregates_rejected": rejected,
        "aggregates_unbarred": unbarred,
        "acceptance_shots": acceptance_shots,
        "nominal_region_fallbacks": fallbacks,
        "stagger": stagger or {},
        # The measured operating region each source actually covered.
        # Without this the handover-merge arm would weight its sources by
        # NOMINAL coordinates while the federated arms use measured ones,
        # and the gap between those two rungs would conflate "exchange
        # during training" with "better coordinates" -- which is exactly the
        # confound the ladder exists to remove.
        "final_states": {
            d.name: (mean_state(visited[d.name][-local_shots:])
                     if visited[d.name] else states[d.name])
            for d in devices
        },
        "final_regions": {
            d.name: _region_or_point(visited[d.name][-local_shots:],
                                    states[d.name])[0]
            for d in devices
        },
        "final_params": {
            d.name: (shared.best.copy() if condition == CENTRALISED
                     else climbers[d.name].best.copy())
            for d in devices
        },
        # ... and on the same basis, how often each source crossed a limit
        # over its last `local_shots`. The federated arms publish this and
        # are downweighted by it; handover-merge published nothing, so it
        # escaped the safety rule entirely and its merge was weighted
        # differently from the rung above it for a second reason.
        "final_violation_rates": {
            d.name: (sum(1 for r in logs[d.name].shots[-local_shots:]
                         if r.violated)
                     / max(1, len(logs[d.name].shots[-local_shots:])))
            for d in devices
        },
        "weight_history": history[-1] if history else None,
    }, server



def run_catastrophe(
    target: str,
    peers: list[str],
    peer_task,
    safe_task,
    unseen_task,
    seed: int,
    peer_shots: int,
    target_shots: int,
    eval_shots: int,
    local_shots: int = 25,
    withheld_beta_N: float | None = None,
    calibration_shots: int = 5,
    eval_disturbance: float = 0.05,
    federated: bool = True,
    rule: str = "geomedian",
    align: bool = True,
    clip_factor: float | None = 2.0,
    accept_if_better: bool = False,
    say=print,
) -> tuple[RunLog, RunLog, dict]:
    """SPEC.md 1, second half -- the claim the spec calls the one that matters.

    'Large tokamaks cannot generate training disruptions, because they cannot
    tolerate them. A device must therefore learn its limits from devices that
    have already crossed them.'

    So: the peers train ACROSS the full band, including the high-beta_N region
    where limits bite. The target trains only in a restricted safe region --
    it never goes near the limit, exactly as a machine that cannot afford to.
    Then the target is evaluated in the region it has never entered.

    The evaluation uses the FROZEN final policy and does no learning. The
    claim is about behaviour in an unseen regime, and a target still
    exploring during evaluation would be learning the regime rather than
    transferring knowledge of it.

    This experiment needs no headroom and no speedup. Whether isolated
    training is fast is irrelevant to whether it is SAFE somewhere it has
    never been.
    """
    states = encoded_states()
    safe = _resolve_task(safe_task)
    unseen = _resolve_task(unseen_task)

    server = None
    peer_rounds = 0
    bandwidth = suggest_bandwidth([states[n] for n in ([target] + peers)])
    if federated:
        _, meta, server = run_condition(
            FEDBUFF_SIMILARITY, list(peers), peer_task, seed=seed,
            shots=peer_shots, local_shots=local_shots, bandwidth=bandwidth,
            rule=rule, align=align, clip_factor=clip_factor,
            accept_if_better=accept_if_better, say=say,
        )
        peer_rounds = meta["rounds"]

    device = get_device(target)

    # --- AUDIT #4: make the holdout real --------------------------------
    # Lowering the setpoint does not withhold a region. The search still
    # explores the whole actuator envelope, so the target visits the states
    # it is supposed never to have seen -- and the measured training
    # violation rate proved it did. The withheld region is now enforced by
    # an interlock above the controller, which is where a real machine puts
    # it, and the coverage is recorded so the holdout can be CHECKED rather
    # than asserted.
    cap, calib = None, []
    if withheld_beta_N is not None:
        cal_env = ToraxDeviceEnv(device, task=safe, seed=seed,
                                 strict_task_check=False)
        levels = np.linspace(-1.0, 1.0, max(2, calibration_shots))
        for lv in levels:
            cal_env.reset()
            for _ in range(cal_env.task.steps_per_shot):
                _, _, term, trunc, _ = cal_env.step(
                    np.full(cal_env.n_actions, lv))
                if term or trunc:
                    break
            reached = [st.scalars['beta_N'] for st in cal_env.trajectory
                       if st.ok and 'beta_N' in st.scalars]
            calib.append((float(lv), max(reached) if reached else float('nan')))
        good = [(lv, b) for lv, b in calib if np.isfinite(b)]
        if len(good) >= 2:
            lv_a = np.array([g[0] for g in good])
            bn_a = np.array([g[1] for g in good])
            # Highest command whose worst-case beta_N stays under the floor,
            # with one sweep step of margin. Conservative on purpose: a cap
            # that is slightly too low costs authority, one that is slightly
            # too high destroys the experiment.
            under = lv_a[bn_a < withheld_beta_N]
            step = float(levels[1] - levels[0]) if len(levels) > 1 else 0.0
            if not under.size:
                # Even minimum command reaches the withheld region, so there
                # is no safe region to train in and the configuration is
                # wrong. Raising beats pinning the actuators and reporting a
                # controller that was never allowed to control.
                raise ValueError(
                    f"{target}: every command level reaches beta_N >= "
                    f"{withheld_beta_N:.3f}, so the 'safe' training "
                    "region does not exist. Raise --unseen, or pick a "
                    "target whose band leaves room below it.")
            # Clamped into the actuator range. A cap below -1 is not a
            # conservative interlock, it is the actuators held shut: the
            # target would train pinned at minimum power and the experiment
            # would measure an idle controller.
            cap = float(np.clip(under.max() - step, -1.0, 1.0))
            if cap <= -1.0 + 1e-9:
                say(f"    WARNING: interlock leaves {target} almost no "
                    "authority -- the cap sits at the envelope floor. The "
                    "safe region is a sliver, and results will reflect "
                    "that rather than the method.")

    env = ToraxDeviceEnv(device, task=safe, seed=seed, strict_task_check=False,
                         action_cap=cap)
    policy = make_policy(env._observe().shape[0], env.n_actions, seed=seed)
    theta0 = policy.get_flat().copy()
    climber = HillClimber(n_params=theta0.size, seed=seed * 1000 + 31)
    climber.seed_from(theta0)

    label = "federated" if federated else "isolated"
    train_log = RunLog(condition=label + "_safe_training", device=target,
                       seed=seed)
    spent = 0
    adopted = 0
    while spent < target_shots:
        for _ in range(min(local_shots, target_shots - spent)):
            cand = climber.propose()
            rec, total = fire_shot(env, policy, cand, spent)
            train_log.add(rec)
            spent += 1
            climber.observe(cand, total)
        if server is not None and spent < target_shots:
            agg = server.aggregate_for(
                target, states[target], ONLY_CLUSTER, peer_rounds + 1,
                reference=climber.best)
            if agg is not None:
                # ONE DRAW AGAINST ONE DRAW, as in run_condition. This loop
                # fires no evaluation shots of its own, so the incumbent has
                # no unbiased estimate and one is bought here -- a shot,
                # charged to the target like every other.
                if accept_if_better and spent < target_shots:
                    rec_inc, inc_total = fire_shot(
                        env, policy, climber.best, spent, evaluation=True)
                    train_log.add(rec_inc)
                    spent += 1
                    climber.note_evaluation(inc_total)
                rec, total = fire_shot(env, policy, agg, spent)
                train_log.add(rec)
                spent += 1
                bar, source = climber.acceptance_bar()
                if accept_if_better and source != "none" and total < bar:
                    pass
                else:
                    climber.adopt(agg, total)
                    adopted += 1

    # Did the interlock hold? Checked at action-step granularity, which is
    # the finest the env exposes -- TORAX takes several internal substeps per
    # action and those are not visible here, so a brief excursion inside one
    # window would not be caught. Stated rather than assumed.
    max_seen = max((st.scalars.get('beta_N', float('-inf'))
                    for st in env.trajectory if st.ok), default=float('nan'))

    # --- evaluation in the unseen regime, frozen policy ------------------
    # AUDIT: the default `easy` task has no disturbances, so repeating a
    # frozen controller from the same reset yields the SAME trajectory every
    # time. Sixty identical repeats are one trial reported sixty times. Each
    # evaluation now gets its own seed and a non-zero disturbance, so the
    # scenarios are independent and the count means what it says.
    eval_unseen = replace(unseen, disturbance_std=max(
        float(getattr(unseen, 'disturbance_std', 0.0)), eval_disturbance))
    eval_log = RunLog(condition=label, device=target, seed=seed)
    frozen = climber.best.copy()
    for i in range(eval_shots):
        eval_env = ToraxDeviceEnv(device, task=eval_unseen,
                                  seed=seed * 10_000 + 7919 + i,
                                  strict_task_check=False)
        rec, _ = fire_shot(eval_env, policy, frozen, i, evaluation=True)
        eval_log.add(rec)

    train_log.notes = f"{label} safe-region training, {spent} shots"
    eval_log.notes = (f"{label} frozen-policy evaluation in the unseen "
                      f"regime, {eval_shots} shots")
    return train_log, eval_log, {
        "target": target,
        "withheld_beta_N": withheld_beta_N,
        "action_cap": cap,
        "calibration": calib,
        "calibration_shots": len(calib),
        "max_beta_N_in_training": float(max_seen),
        "holdout_breached": bool(
            withheld_beta_N is not None and np.isfinite(max_seen)
            and max_seen >= withheld_beta_N),
        "eval_disturbance": eval_disturbance,
        "peers": list(peers),
        "federated": federated,
        "peer_rounds": peer_rounds,
        "aggregates_adopted": adopted,
        "bandwidth": float(bandwidth),
        "received_weights": (server.weight_table().get(target, {})
                             if server is not None else {}),
    }


def derangements(names, n, seed=0):
    """N permutations in which NO device keeps its own coordinates.

    The scramble control: publish each client under another device's measured
    coordinates and see whether the similarity weighting still appears to
    work. If it does, it was never the physics doing it.

    A plain shuffle can leave devices in place, which would make the control
    partly not a control. With four devices there are nine derangements; more
    than that is sampling with replacement, and this says so by repeating
    rather than pretending to have more.

    Lived in scripts/exp_federation.py until the cold-start experiment needed
    it too -- a control that only one script can run is a control that does
    not get run.
    """
    import itertools
    import random

    rng = random.Random(seed)
    all_d = [p for p in itertools.permutations(names)
             if all(a != b for a, b in zip(p, names))]
    if not all_d:
        return []
    picks = (rng.sample(all_d, n) if n <= len(all_d)
             else [rng.choice(all_d) for _ in range(n)])
    return [dict(zip(names, p)) for p in picks]


def nearest_by_design(joiner: str, candidates: list[str]) -> str:
    """Pick a source using DESIGN information only.

    PROTOCOL.md 2 lists geometry and field as free -- they are known from
    drawings before the machine runs -- while anything measured from the
    newcomer's own discharges is charged or refused. Selecting the source by
    dimensionless distance would need the joiner's calibration, so the rule
    here uses aspect ratio, field and current, each normalised by the spread
    across the candidate set so no single quantity dominates by unit alone.

    Deterministic, and fixed before any run: a source chosen after seeing
    which one transfers best would make the single-source arm the best of
    several, and the ladder would stop being a ladder.

    Features, and why each is in logs or not. a*B_0 sets the gyroradius
    scale that rho* is built on, and I_p sets the current scale; across this
    set they span factors of 30 and 42, so they are compared in the log --
    the same reason `similarity_distance` log-scales rho* and nu*. Aspect
    ratio and q_cyl are O(1) and comparable linearly. Each is then z-scored
    over the candidate set so no quantity dominates through its units.

    A first version z-scored a*B_0 and I_p linearly and picked iter_like as
    the nearest design match for tcv_like -- the largest machine in the set
    for the smallest -- because a factor-of-30 spread makes a linear z-score
    a report on the biggest device.
    """
    names = [joiner] + list(candidates)
    devs = {n: get_device(n) for n in names}
    feats = {
        n: np.array([
            np.log10(d.a_minor * d.B_0),
            np.log10(d.Ip_nominal / 1e6),
            d.R_major / d.a_minor,
            d.q_cylindrical,
        ], dtype=float)
        for n, d in devs.items()
    }
    stack = np.stack([feats[n] for n in names])
    scale = stack.std(axis=0)
    scale[scale <= 0] = 1.0
    ref = feats[joiner] / scale
    return min(candidates,
               key=lambda c: float(np.linalg.norm(feats[c] / scale - ref)))


def _merge_finals(finals: dict, regions: dict, server: FedBuffServer,
                  states: dict, joiner: str, joiner_state, joiner_region,
                  reference, cluster: str = ONLY_CLUSTER, *,
                  violation_rates: dict | None = None):
    """Publish already-trained models once and aggregate them once.

    The handover-merge arm. Its sources trained in isolation and never
    exchanged anything, so a gap between this and the federated arms is
    attributable to exchanging DURING training rather than to combining
    models at all -- which is the one thing the previous three-arm design
    could not separate.

    THAT ATTRIBUTION WAS FALSE UNTIL NOW, in two ways at once. This arm's
    server was built with `use_similarity=True` while the rung directly above
    it (`federated_uniform`) weights uniformly, so climbing the ladder
    REMOVED an ingredient instead of adding one. And nothing published here
    carried a violation rate, so the safety downweighting that every
    federated update pays did not apply. Measured weights for a tcv_like
    joiner showed both: handover-merge gave diiid_like 0.558 (pure
    similarity, no safety penalty) where federated_uniform gave it 0.245.
    A gap between those two rungs was three changes, not one.
    """
    # NO SILENT FALLBACK. `violation_rates=None` would merge with every
    # source treated as clean, which is precisely the behaviour this argument
    # was added to remove -- and it would look identical in the output. A
    # missing safety input is a bug in the caller, so it raises.
    if violation_rates is None:
        raise ValueError(
            "_merge_finals needs each source's violation rate; without it "
            "the handover merge escapes the safety downweighting the "
            "federated rungs pay and the two stop being comparable. "
            "run_condition returns them as meta['final_violation_rates'].")
    missing = sorted(set(finals) - set(violation_rates))
    if missing:
        raise ValueError(
            f"no violation rate for {missing}; every merged source must "
            "carry one for the same reason.")
    rates = violation_rates
    for name, vec in finals.items():
        server.publish(ClientUpdate(
            device=name, cluster=cluster, weights={"flat": vec.copy()},
            state=states[name], region=regions.get(name),
            n_samples=1, round_produced=1,
            violation_rate=float(rates.get(name, 0.0)),
        ))
    return server.aggregate_for(joiner, joiner_state, cluster, 2,
                                reference=reference,
                                target_region=joiner_region)


def run_cold_start(
    joiner: str,
    incumbents: list[str],
    task_name: str,
    seed: int,
    pretrain_shots: int,
    join_shots: int,
    local_shots: int = 25,
    arm: str | None = None,
    inherit: bool = True,  # legacy; `arm` supersedes it
    use_similarity: bool = True,  # legacy; `arm` supersedes it
    rule: str = "geomedian",
    align: bool = True,
    clip_factor: float | None = 2.0,
    accept_if_better: bool = False,
    eval_every: int = 10,
    probe_shots: int = 5,
    stagger: dict[str, int] | None = None,
    scramble: dict[str, str] | None = None,
    say=print,
) -> tuple[RunLog, dict]:
    """A brand-new machine joins a federation that has already been running.

    THIS IS METRICS.md METRIC 4, AND IT IS THE STRONGEST NUMBER IN THE SET.
    It does not need the headroom gate to pass. Headroom is about whether an
    ASYMPTOTIC speedup is bigger than campaign noise; cold start is about the
    first shots on a machine that has none, where the gap is largest and
    where a fusion audience actually feels it -- at 20-40 shots/day, 40 shots
    versus 240 is a week of commissioning versus a month.

    The joiner's budget counts only ITS shots. The incumbents' pre-training
    is not charged to it, which is the whole point: those shots were fired on
    other machines, in other countries, on other campaigns.

    WHAT THIS CAN FALSIFY
    ---------------------
    Similarity theory makes a prediction here, and it is the reason to run
    leave-one-out rather than one arbitrary joiner: a machine CLOSE to the
    federation in (rho*, nu*, beta_N, q95) should inherit more than a distant
    one. sparc_like sits 0.595 from iter_like; tcv_like sits 1.669. If both
    gain the same amount, the weighting is decoration and SPEC.md 4b earns
    nothing -- which is a result, and a publishable one.
    """
    if arm is None:
        arm = ("federated_similarity" if use_similarity
               else "federated_uniform") if inherit else "scratch"
    if arm not in COLD_START_ARMS:
        raise ValueError(
            f"unknown arm {arm!r}; expected one of {COLD_START_ARMS}. "
            "conventional lives in scripts/exp_conventional.py.")

    condition = (FEDBUFF_SIMILARITY if arm == "federated_similarity"
                 else FEDBUFF_UNIFORM)
    states = encoded_states()

    # The kernel bandwidth is computed over the WHOLE device set including
    # the joiner. Letting it float with the incumbent subset would change
    # the metric between held-out folds, and the folds would stop being
    # comparable to each other.
    bandwidth = suggest_bandwidth(
        [states[n] for n in ([joiner] + list(incumbents))])

    # Source training. Every arm that has sources gives each of them the
    # SAME shot budget, so a difference between arms is never a difference
    # in how much source compute they were handed.
    meta, server, handover, source_used = {"rounds": 0}, None, None, []

    if arm == "scratch":
        # No sources at all. Pre-training a federation and discarding it
        # would spend a third of this experiment's compute on nothing.
        pass

    elif arm == "single_source":
        # One source, chosen on DESIGN information before the run. Isolates
        # what a plain warm start is worth, separately from having several
        # sources or from federating them.
        source_used = [nearest_by_design(joiner, list(incumbents))]
        _, meta, _ = run_condition(
            ISOLATED, source_used, task_name, seed=seed,
            shots=pretrain_shots, local_shots=local_shots,
            bandwidth=bandwidth, eval_every=eval_every, scramble=scramble,
            say=say,
        )
        handover = meta["final_params"][source_used[0]].copy()

    elif arm == "handover_merge":
        # All sources, each trained ALONE, merged once at handover. A gap
        # between this and the federated arms is attributable to exchanging
        # during training rather than to combining models at all.
        source_used = list(incumbents)
        _, meta, _ = run_condition(
            ISOLATED, source_used, task_name, seed=seed,
            shots=pretrain_shots, local_shots=local_shots,
            bandwidth=bandwidth, eval_every=eval_every, scramble=scramble,
            say=say,
        )
        # UNIFORM, to match the rung above. The ladder's whole claim is that
        # a gap between adjacent rungs is attributable to the one ingredient
        # that differs; `federated_uniform` weights uniformly, so the merge
        # it is compared against must too, or the comparison measures the
        # weighting rule and the exchange schedule at once.
        server = FedBuffServer(
            bandwidth=bandwidth, use_similarity=False, rule=rule,
            align=align, clip_factor=clip_factor,
        )

    else:  # federated_uniform | federated_similarity
        source_used = list(incumbents)
        _, meta, server = run_condition(
            condition, source_used, task_name, seed=seed,
            shots=pretrain_shots, local_shots=local_shots, stagger=stagger,
            bandwidth=bandwidth, rule=rule, align=align,
            clip_factor=clip_factor, accept_if_better=accept_if_better,
            eval_every=eval_every, scramble=scramble, say=say,
        )

    device = get_device(joiner)
    task = get_task(task_name)
    env = ToraxDeviceEnv(device, task=task, seed=seed)
    policy = make_policy(env._observe().shape[0], env.n_actions, seed=seed)
    theta0 = make_policy(env._observe().shape[0], env.n_actions,
                         seed=seed).get_flat().copy()

    climber = HillClimber(n_params=theta0.size, seed=seed * 1000 + 999)
    climber.seed_from(theta0)

    # The ARM, not the legacy booleans. Those now default to True/True
    # whatever arm is running, so every log -- scratch included -- was
    # written with condition "cold_inherit_similarity". The script keyed its
    # own dict correctly, so the analysis was right, but `runs.json` carried
    # the wrong label on every record. AUDIT #10 exists so artifacts support
    # reanalysis, and a reanalysis reading the RunLog's own condition field
    # would have attributed all five arms to the similarity arm.
    label = arm
    log = RunLog(condition=label, device=joiner, seed=seed)
    spent = 0
    # Shots the JOINER spent measuring its own incumbent so it could judge a
    # handover. `scratch` has no handover and spends none, so it gets one
    # more training shot out of 120 than every inheriting arm. Under a
    # percent, and the sort of asymmetry that turns into a mystery if it is
    # not written down.
    joiner_acceptance_shots = 0
    received: dict[str, float] = {}

    # COMMISSIONING. A joiner that has fired nothing has no measured region,
    # so weighting it falls back to the nominal table -- the one measurement
    # showed wrong by 2x in beta_N and 7x in nu*. A real machine runs
    # characterisation discharges before anyone hands it a controller. These
    # are charged to its budget like every other shot, and they also give it
    # an incumbent, without which reject-if-worse cannot fire on the first
    # adoption.
    probe_states = []
    for _ in range(min(probe_shots, join_shots)):
        cand = climber.propose()
        rec, total = fire_shot(env, policy, cand, spent)
        log.add(rec)
        spent += 1
        climber.observe(cand, total)
        try:
            probe_states.append(state_from_env(env))
        except Exception:
            pass

    commissioning_shots = spent
    joiner_state = mean_state(probe_states) if probe_states else states[joiner]
    joiner_region = _region_or_point(probe_states, states[joiner])[0]

    agg = None
    handover_adopted: bool | None = None
    handover_return = float("nan")
    incumbent_return = float("nan")
    handover_trial_shots = 0
    if arm == "single_source":
        # No aggregation: one model, handed over as it stands. Merging a
        # single model with itself would put the robust rules in the path of
        # an arm whose whole point is that they are not involved.
        agg = handover
    elif arm == "handover_merge":
        # The SCRAMBLE reaches this rung too, or it would be the one arm
        # publishing true coordinates while the federated rungs published
        # permuted ones -- a control that changes two things.
        # Alignment needs the network shape in this arm too. Without it,
        # align=True silently skipped the permutation matching that the
        # federated arms perform, confounding the repeated-exchange control.
        server.shape = (policy.obs_dim, policy.hidden, policy.act_dim)
        # Use the same measured source window as a buffered update. Keep
        # nominal states available for a scramble that names the held-out
        # device, which has no source-training trajectory.
        handover_states = {**states, **meta.get("final_states", {})}
        pub_states = ({d: handover_states[scramble.get(d, d)] for d in states}
                      if scramble else handover_states)
        pub_regions = ({d: meta["final_regions"].get(
            scramble.get(d, d), region_from_state(pub_states[d]))
                        for d in meta["final_params"]}
                       if scramble else meta["final_regions"])
        agg = _merge_finals(
            meta["final_params"], pub_regions, server, pub_states,
            joiner, joiner_state, joiner_region, reference=climber.best,
            violation_rates=meta.get("final_violation_rates"))
    elif server is not None:
        agg = server.aggregate_for(
            joiner, joiner_state, ONLY_CLUSTER, meta['rounds'] + 1,
            reference=climber.best, target_region=joiner_region)

    if agg is not None and spent < join_shots:
        # AUDIT #2, and a regression of it. This shot IS the zero-adaptation
        # assessment of the inherited controller -- the only observation of
        # it before any local learning. Leaving it unmarked excluded it from
        # every evaluation-based metric, so the handover was measured only
        # after the joiner had already started changing it.
        #
        # The audit caught this once. My own commissioning-shots patch then
        # rewrote the block and dropped the flag again, and nothing noticed
        # because #2 was the one finding I never wrote a test for. It has one
        # now: test_the_handover_shot_is_marked_as_an_evaluation.
        rec, total = fire_shot(env, policy, agg, spent, evaluation=True)
        log.add(rec)
        spent += 1
        handover_trial_shots += 1
        # RECORD THE DECISION, not just its effect. With --select on, an arm
        # that rejects every handover is behaving exactly like scratch, and
        # nothing in the artifacts said which happened -- the same failure
        # as the guard that went quiet: silence meant two opposite things.
        # Inferring it from the shots-to-competence number afterwards is
        # reasoning backwards from the result being explained.
        handover_return = float(total)
        # ONE DRAW AGAINST ONE DRAW. The bar was `best_return`, the best of
        # the joiner's five commissioning shots -- so the handover had to
        # beat the luckiest of five noisy draws on a single draw of its own.
        # With two equally good controllers that is a 17% chance of adoption,
        # and the arm then runs as scratch wearing a federated label.
        #
        # An unbiased bar needs one unperturbed evaluation of the incumbent.
        # The commissioning shots are candidates, not evaluations, so there
        # is none yet -- and it is worth one shot out of the joiner's budget
        # to have the comparison be a comparison.
        bar, source = climber.acceptance_bar()
        rec_inc = None
        if accept_if_better and source == "none" and spent < join_shots:
            rec_inc, inc_total = fire_shot(env, policy, climber.best, spent,
                                           evaluation=True)
            log.add(rec_inc)
            spent += 1
            joiner_acceptance_shots += 1
            climber.note_evaluation(inc_total)
            bar, source = climber.acceptance_bar()
        incumbent_return = float(bar)
        handover_adopted = not (accept_if_better and source != "none"
                                and total < bar)
        # Keep both paid-for trials in the diagnostics, but only certify
        # the controller installed by this decision. Otherwise a rejected
        # handover can provide a false successful evaluation, or the two
        # trials can confirm competence across two different controllers.
        rec.evaluation_eligible = handover_adopted
        if rec_inc is not None:
            rec_inc.evaluation_eligible = not handover_adopted
        if handover_adopted:
            climber.adopt(agg, total)
            received = (server.weight_table().get(joiner, {})
                        if server is not None else {})

    while spent < join_shots:
        if (eval_every and spent % eval_every == 0
                and np.isfinite(climber.best_return)):
            rec, ev_total = fire_shot(env, policy, climber.best, spent,
                                      evaluation=True)
            log.add(rec)
            spent += 1
            climber.note_evaluation(ev_total)
            continue
        cand = climber.propose()
        rec, total = fire_shot(env, policy, cand, spent)
        log.add(rec)
        spent += 1
        climber.observe(cand, total)

    log.notes = (f"{label} joiner={joiner} incumbents={list(incumbents)} "
                 f"pretrain={pretrain_shots} bandwidth={bandwidth:.3f}")
    return log, {
        "joiner": joiner,
        "arm": arm,
        "sources_used": source_used,
        "source_shots_by_device": dict(meta.get("shots_by_device", {})),
        "source_shots": int(meta.get("total_shots", 0)),
        "joiner_shots": len(log),
        "joiner_probe_shots": commissioning_shots,
        "joiner_selection_shots": handover_trial_shots + joiner_acceptance_shots,
        "joiner_evaluation_shots": (sum(r.is_evaluation for r in log.shots)
                                    - handover_trial_shots
                                    - joiner_acceptance_shots),
        "joiner_training_shots": (sum(not r.is_evaluation for r in log.shots)
                                  - commissioning_shots),
        # This upstream sweep is cached in the registry, shared by all
        # arms, and was not executed again by this run. Report its cost
        # separately from the shots actually fired above (PROTOCOL.md 1).
        "shared_calibration_shots": (5 if task.setpoint_mode == "band_fraction"
                                     else 0),
        "calibration_source": "cached device-band sweep (PROTOCOL.md 1)",
        "incumbents": list(incumbents),
        "inherit": arm != "scratch",
        "use_similarity": arm == "federated_similarity",
        "rule": rule,
        "align": align,
        "clip_factor": clip_factor,
        "clipped": int(server.last_clipped) if server is not None else 0,
        # WHY THE AGGREGATE WAS EMPTY, when it was. The server has recorded
        # `last_rejected_all` and `last_error` since the NaN-region bug --
        # where a non-finite state silently disabled federation and the arm
        # ran as scratch under a federated label -- but nothing outside the
        # server has ever read them. A flag nobody reads is not an
        # observation; it is a comment. If every round rejects, that is the
        # explanation for an arm that looks identical to scratch, and it must
        # appear in the run rather than be inferred afterwards.
        "aggregate_rejected_all": (bool(server.last_rejected_all)
                                   if server is not None else False),
        "aggregate_error": (str(server.last_error)
                            if server is not None else ""),
        # ... and over the whole run, not just the last call. The `last_`
        # flags describe one aggregation; an arm that federated nothing for
        # twenty rounds and then succeeded once would report clean.
        "rejected_rounds": (int(server.rejected_rounds)
                            if server is not None else 0),
        "error_rounds": (int(server.error_rounds)
                         if server is not None else 0),
        # Rounds where the similarity kernel underflowed for every peer and
        # the weights fell back to uniform. When that happens the similarity
        # arm IS the uniform arm, and the two would otherwise be reported as
        # separate rungs producing the same answer for no stated reason.
        "uniform_fallback_rounds": (int(server.uniform_fallback_rounds)
                                    if server is not None else 0),
        "source_rejections": int(meta.get("aggregates_rejected", 0)),
        "source_acceptance_shots": int(meta.get("acceptance_shots", 0)),
        "joiner_acceptance_shots": int(joiner_acceptance_shots),
        "bandwidth": float(bandwidth),
        "pretrain_rounds": meta["rounds"],
        # None means no handover was offered at all (scratch, or every peer
        # rejected); True/False is a decision that was actually taken.
        "handover_adopted": handover_adopted,
        "handover_return": handover_return,
        "incumbent_return_at_handover": incumbent_return,
        "accept_if_better": accept_if_better,
        "received_weights": received,
        # Recorded per run, because a scrambled fold that looks like a real
        # one in the artifacts is worse than no control at all.
        "scramble": dict(scramble) if scramble else {},
    }


def isolated_reference(logs_by_condition: dict[str, list[RunLog]]) -> list[RunLog]:
    """The denominator of every ratio, fetched by name so no caller guesses."""
    return logs_by_condition[ISOLATED]
