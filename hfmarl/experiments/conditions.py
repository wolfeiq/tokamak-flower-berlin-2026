"""The five experimental conditions (SPEC.md Phase 5).

Defined once, here, because every plot, every metric and every runner must
agree on the names and on the order they appear in a legend. A condition
spelled two ways is a silently missing bar.
"""

from __future__ import annotations

from dataclasses import dataclass

ISOLATED = "isolated"
FEDBUFF_UNIFORM = "fedbuff_uniform"
FEDBUFF_SIMILARITY = "fedbuff_similarity"
FEDAVG_NAIVE = "fedavg_naive"
CENTRALISED = "centralised"

# Legend order: baselines, then the method, then the negative control, then the
# upper bound. Fixed so figures are comparable across runs.
CONDITION_ORDER: tuple[str, ...] = (
    ISOLATED, FEDBUFF_UNIFORM, FEDBUFF_SIMILARITY, FEDAVG_NAIVE, CENTRALISED,
)

LABELS: dict[str, str] = {
    ISOLATED: "isolated (no federation)",
    FEDBUFF_UNIFORM: "FedBuff, role-matched, uniform",
    FEDBUFF_SIMILARITY: "FedBuff, role-matched, similarity ← method",
    FEDAVG_NAIVE: "naive FedAvg (role-blind)",
    CENTRALISED: "centralised (upper bound)",
}


@dataclass(frozen=True)
class ConditionSpec:
    """How one condition is actually configured."""

    name: str
    federated: bool
    role_matched: bool
    use_similarity: bool
    pooled_data: bool
    role: str  # what it is FOR -- baseline / method / control / bound
    expectation: str  # what it should show if the claim is true

    def aggregation_kwargs(self) -> dict:
        """Arguments for `federation.similarity.aggregation_weights`."""
        return {"use_similarity": self.use_similarity}


CONDITIONS: dict[str, ConditionSpec] = {
    ISOLATED: ConditionSpec(
        ISOLATED, federated=False, role_matched=False, use_similarity=False,
        pooled_data=False, role="baseline",
        expectation="Slowest to threshold. Defines the denominator of every "
                    "speedup ratio, and its shots-to-threshold IS the headroom "
                    "check -- if it converges in a few hundred shots the whole "
                    "experiment has nothing to show.",
    ),
    FEDBUFF_UNIFORM: ConditionSpec(
        FEDBUFF_UNIFORM, federated=True, role_matched=True, use_similarity=False,
        pooled_data=False, role="baseline",
        expectation="Faster than isolated. Isolates the value of federating at "
                    "all, separately from how peers are weighted.",
    ),
    FEDBUFF_SIMILARITY: ConditionSpec(
        FEDBUFF_SIMILARITY, federated=True, role_matched=True, use_similarity=True,
        pooled_data=False, role="method",
        expectation="Fastest of the non-pooled conditions. The gap over "
                    "fedbuff_uniform is the entire contribution of SPEC.md §4b; "
                    "if that gap is zero, the physics weighting earns nothing.",
    ),
    FEDAVG_NAIVE: ConditionSpec(
        FEDAVG_NAIVE, federated=True, role_matched=False, use_similarity=False,
        pooled_data=False, role="negative control",
        expectation="At or BELOW isolated. Averaging thermal with particle and "
                    "current agents mixes incompatible action semantics. This "
                    "is the condition that demonstrates role matching matters; "
                    "if it matches the others, role matching is unsupported.",
    ),
    CENTRALISED: ConditionSpec(
        CENTRALISED, federated=False, role_matched=False, use_similarity=False,
        pooled_data=True, role="upper bound",
        expectation="Best overall, and NOT achievable in practice -- private "
                    "fusion will not pool discharge data (SPEC.md §9). Use it "
                    "to set the threshold and to bound the gap, never as a "
                    "condition the method is expected to beat.",
    ),
}


def get(name: str) -> ConditionSpec:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; have {sorted(CONDITIONS)}")
    return CONDITIONS[name]


def describe() -> str:
    lines = ["The five conditions, all on identical seeds:", ""]
    for i, c in enumerate(CONDITION_ORDER, 1):
        s = CONDITIONS[c]
        # The key is printed alongside the label because it is what gets typed
        # on a command line and stored in a RunLog; a description you cannot
        # act on is half a description.
        lines.append(f"{i}. {LABELS[c]}")
        lines.append(f"     key: {c}   role: {s.role}")
        lines.append(f"     {s.expectation}")
        lines.append("")
    return "\n".join(lines)
