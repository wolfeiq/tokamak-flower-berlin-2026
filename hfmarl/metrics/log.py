"""Shot accounting.

THE CURRENCY IS SHOTS
---------------------
Not wall-clock, not gradient steps, not environment transitions. A shot is one
discharge: one episode, one TORAX simulation. It is what costs money and
machine time, and it is the number a fusion audience reads. Every metric in
this package is denominated in shots, and the logging below exists to make that
denomination impossible to fudge.

One `ShotRecord` per episode. One `RunLog` per (condition, device, seed). One
`ExperimentLog` per experiment. Nothing is aggregated at write time -- the raw
per-shot record is kept, because SPEC.md §8 is explicit that cross-device
comparisons cannot be reconstructed afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def _as_float(v) -> float:
    """float(), with JSON's `null` meaning "not a number".

    `save_run` writes non-finite floats as `null`, because bare NaN is not
    valid JSON and an artifact only one tool can read is not an artifact. A
    solver-failed shot records a NaN tracking error, so that null comes back
    here on every reload of a run that had one -- and `float(None)` raises.

    The round trip has to be lossless in the direction that matters: NaN out,
    NaN back.
    """
    return float("nan") if v is None else float(v)


@dataclass
class ShotRecord:
    """One discharge.

    `federated_in` records whether this shot's policy had already received a
    federated update, so a learning curve can be split at the point federation
    actually started influencing behaviour rather than at round boundaries.
    """

    shot: int  # global shot index for this run, from 0
    reward: float  # total episode return
    steps: int  # control steps survived
    violations: tuple[str, ...] = ()  # limit names crossed during this shot
    terminated_early: bool = False  # ended on a violation or solver failure
    beta_error: float = float("nan")  # mean |beta_N - target| over the shot
    federated_in: int = 0  # federated updates received before this shot
    wall_seconds: float = 0.0
    # An EVALUATION shot fires the incumbent policy unperturbed. Every
    # other shot fires a candidate -- a deliberate perturbation -- so a
    # controller that already works still logs errors while it keeps
    # probing. Measuring competence on the candidate stream measures
    # exploration noise; measuring it here measures the controller.
    is_evaluation: bool = False
    # Handover trials remain evaluations for zero-adaptation diagnostics,
    # but a rejected controller must not certify the installed incumbent.
    # Older logs lack this distinction and retain their original behavior.
    evaluation_eligible: bool = True

    def __post_init__(self) -> None:
        """Coerce to plain Python types.

        numpy scalars are not JSON-serialisable, and a `np.int64` leaking in
        from a rollout would make `ExperimentLog.save` raise -- hours into a
        run, after the data it was meant to protect had been collected.
        Coercing at construction moves that failure to the moment of the
        mistake, and makes it impossible to hold an unsaveable record.
        """
        object.__setattr__(self, "shot", int(self.shot))
        object.__setattr__(self, "reward", _as_float(self.reward))
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "violations", tuple(str(v) for v in self.violations))
        object.__setattr__(self, "terminated_early", bool(self.terminated_early))
        object.__setattr__(self, "beta_error", _as_float(self.beta_error))
        object.__setattr__(self, "federated_in", int(self.federated_in))
        object.__setattr__(self, "wall_seconds", _as_float(self.wall_seconds))
        object.__setattr__(self, "is_evaluation", bool(self.is_evaluation))
        object.__setattr__(self, "evaluation_eligible",
                           bool(self.evaluation_eligible))

    @property
    def violated(self) -> bool:
        return bool(self.violations)


@dataclass
class RunLog:
    """One training run: a single (condition, device, seed) triple.

    The triple is mandatory rather than optional metadata. A reward curve with
    no seed attached cannot be pooled, and SPEC.md §8 warns that
    sample-efficiency claims will not survive single runs.
    """

    condition: str
    device: str
    seed: int
    shots: list[ShotRecord] = field(default_factory=list)
    notes: str = ""

    def add(self, record: ShotRecord) -> None:
        self.shots.append(record)

    def __len__(self) -> int:
        return len(self.shots)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.condition, self.device, self.seed)

    # -- vector views -----------------------------------------------------

    def rewards(self) -> np.ndarray:
        return np.array([s.reward for s in self.shots], dtype=float)

    def violation_flags(self) -> np.ndarray:
        return np.array([s.violated for s in self.shots], dtype=bool)

    def cumulative_violations(self) -> np.ndarray:
        """Running total of shots that crossed any limit.

        Counted per shot, not per crossing: a shot that violates three limits
        at once is one unsafe discharge, not three. Counting crossings would
        make a single bad shot look like a trend.
        """
        return np.cumsum(self.violation_flags().astype(int))

    def n_nonfinite(self) -> int:
        """Shots whose return is NaN or inf.

        These do not simply vanish: a NaN reward never clears a threshold, so
        the run looks like a failure to converge and is right-censored -- which
        then feeds the headline speedup ratio. A data-quality problem must not
        be able to masquerade as a learning result, so it is counted and
        surfaced by `ExperimentLog.coverage()`.
        """
        r = self.rewards()
        return int((~np.isfinite(r)).sum()) if r.size else 0

    def beta_errors(self) -> np.ndarray:
        return np.array([s.beta_error for s in self.shots], dtype=float)

    def evaluation_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Tracking error on evaluation shots, and the shot index of each.

        The index matters: shots-to-competence must be reported in TOTAL
        shots consumed, not in evaluations performed. A machine that
        evaluates every tenth shot has still fired all ten.
        """
        errs, idx = [], []
        for s in self.shots:
            if s.is_evaluation:
                errs.append(s.beta_error)
                idx.append(s.shot)
        return (np.array(errs, dtype=float),
                np.array(idx, dtype=int))

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "device": self.device,
            "seed": self.seed,
            "notes": self.notes,
            "shots": [asdict(s) for s in self.shots],
        }

    @staticmethod
    def from_dict(d: dict) -> "RunLog":
        log = RunLog(condition=d["condition"], device=d["device"], seed=int(d["seed"]),
                     notes=d.get("notes", ""))
        for s in d["shots"]:
            s = dict(s)
            s["violations"] = tuple(s.get("violations", ()))
            log.add(ShotRecord(**s))
        return log


class ExperimentLog:
    """Every run in one experiment, addressable by (condition, device, seed)."""

    def __init__(self, name: str = "experiment"):
        self.name = name
        self.runs: dict[tuple[str, str, int], RunLog] = {}

    def add(self, run: RunLog) -> None:
        if run.key in self.runs:
            raise ValueError(
                f"duplicate run {run.key}; refusing to overwrite. "
                "Re-running one cell of the matrix should replace it explicitly."
            )
        self.runs[run.key] = run

    def get(self, condition: str, device: str, seed: int) -> RunLog:
        return self.runs[(condition, device, seed)]

    def select(self, condition: str | None = None, device: str | None = None
               ) -> list[RunLog]:
        """All runs matching the given filters, across seeds."""
        return [
            r for r in self.runs.values()
            if (condition is None or r.condition == condition)
            and (device is None or r.device == device)
        ]

    @property
    def conditions(self) -> list[str]:
        return sorted({r.condition for r in self.runs.values()})

    @property
    def devices(self) -> list[str]:
        return sorted({r.device for r in self.runs.values()})

    @property
    def seeds(self) -> list[int]:
        return sorted({r.seed for r in self.runs.values()})

    def coverage(self) -> str:
        """Which matrix cells exist, and with how many seeds.

        Print this before plotting. A missing cell silently drops a bar from a
        grouped chart, and an unequal seed count between conditions makes a
        comparison unfair without making it look unfair.
        """
        lines = [f"experiment: {self.name}", f"runs: {len(self.runs)}"]
        lines.append(f"  {'condition':28s}" + "".join(f"{d:>14s}" for d in self.devices))
        all_counts: list[int] = []
        for c in self.conditions:
            counts = [len(self.select(c, d)) for d in self.devices]
            lines.append(f"  {c:28s}" + "".join(f"{n:>14d}" for n in counts))
            all_counts.extend(counts)

        # Raggedness is checked across the WHOLE matrix, not row by row. A
        # condition trained on 5 seeds against one trained on 2 is an unfair
        # comparison that looks perfectly tidy if each row is inspected alone --
        # which was the original bug here.
        bad = {r.key: r.n_nonfinite() for r in self.runs.values() if r.n_nonfinite()}
        if bad:
            total = sum(bad.values())
            lines.append(
                f"  WARNING: {total} shot(s) across {len(bad)} run(s) have "
                "non-finite returns. A NaN return never clears a threshold, so "
                "those runs are silently right-censored and drag the speedup "
                "ratio. Investigate before reporting."
            )
            for k, n in sorted(bad.items())[:5]:
                lines.append(f"    {k}: {n}")

        if all_counts and (len(set(all_counts)) > 1 or 0 in all_counts):
            missing = [
                f"{c}/{d}"
                for c in self.conditions
                for d in self.devices
                if not self.select(c, d)
            ]
            lines.append(
                f"  WARNING: ragged matrix -- seed counts range "
                f"{min(all_counts)}-{max(all_counts)} across cells. Conditions "
                "with different seed counts are not fairly comparable, and an "
                "empty cell vanishes silently from grouped plots."
            )
            if missing:
                lines.append(f"  empty cells: {', '.join(missing)}")
        return "\n".join(lines)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the whole experiment as VALID JSON.

        A solver-failed shot records a NaN tracking error, and `json.dumps`
        writes bare `NaN`, which Python reads back and nothing else will.
        `_jsonable` maps non-finite floats to null and `allow_nan=False`
        turns anything that slips past into a loud failure rather than a
        file that only this codebase can open. `_as_float` brings the NaN
        back on load, so the round trip is lossless where it matters.
        """
        from hfmarl.util.artifacts import _jsonable

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            _jsonable({"name": self.name,
                       "runs": [r.to_dict() for r in self.runs.values()]}),
            indent=1, allow_nan=False,
        ), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "ExperimentLog":
        d = json.loads(Path(path).read_text())
        log = ExperimentLog(d.get("name", "experiment"))
        for r in d["runs"]:
            log.add(RunLog.from_dict(r))
        return log
