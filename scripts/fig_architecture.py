"""Architecture diagram for hfmarl-fusion, as the code actually stands."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERIFIED = "#2e8b3d"
WRITTEN = "#d98a1f"
SCAFFOLD = "#8b9096"
BLOCKED = "#c8342f"
EXTERNAL = "#3d6ba5"
FACE = {VERIFIED: "#ecf7ed", WRITTEN: "#fdf4e5", SCAFFOLD: "#f2f3f4",
        BLOCKED: "#fdecec", EXTERNAL: "#eaf0f8"}

LINE_H = 2.05
TITLE_H = 3.9
SUB_H = 3.0
PAD_T, PAD_B = 2.4, 2.0

fig = plt.figure(figsize=(19.0, 13.8))
gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.09,
                      left=0.010, right=0.990, top=0.928, bottom=0.020)
ax = fig.add_subplot(gs[0])
axg = fig.add_subplot(gs[1])
for a in (ax, axg):
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.axis("off")


def box(x, top, w, title, lines, colour, sub=None, fs=6.9, tfs=9.0):
    """Draw a box whose TOP edge is at `top`. Returns its bottom edge."""
    h = PAD_T + TITLE_H + (SUB_H if sub else 0) + len(lines) * LINE_H + PAD_B
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.40,rounding_size=1.2",
        linewidth=1.8, edgecolor=colour, facecolor=FACE[colour], zorder=2))
    cy = top - PAD_T
    ax.text(x + w / 2, cy, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", color="#141414", zorder=3)
    cy -= TITLE_H
    if sub:
        ax.text(x + w / 2, cy, sub, ha="center", va="top", fontsize=6.5,
                style="italic", color=colour, zorder=3)
        cy -= SUB_H
    for ln in lines:
        ax.text(x + 1.8, cy, ln, ha="left", va="top", fontsize=fs,
                color="#2b2b2b", zorder=3)
        cy -= LINE_H
    return y


def arrow(a, p, q, colour="#5a5e62", lw=1.7, rad=0.0, ls="-"):
    a.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=colour,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=2, shrinkB=2))


def tag(a, x, y, t, fs=6.4, colour="#5a5e62"):
    a.text(x, y, t, ha="center", va="center", fontsize=fs, color=colour,
           zorder=4, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="none", alpha=0.96))


W = 23.0
X = [1.0, 26.0, 51.0, 76.0]
GAP = 3.4

# ===================== BAND 1 — the plant, left to right ===================
B1 = 93.5
bottoms = []
bottoms.append(box(X[0], B1, W, "devices/registry.py", [
    "4 tokamaks, circular geometry:",
    "  iter_like    sparc_like",
    "  diiid_like   tcv_like",
    "",
    "actuators, same shape everywhere:",
    "  aux_heat -> generic_heat",
    "  ecrh     -> ecrh",
    "  gas_puff, ip;  icrh disabled",
    "",
    "BETA_N_BANDS   measured,",
    "  keyed by device x task",
], VERIFIED, sub="the federation clients"))

bottoms.append(box(X[1], B1, W, "envs/  config, ramps, task", [
    "torax_config.build_config",
    "  T_ped ~ a*B_0    device-relative",
    "  n_e as a Greenwald fraction",
    "  use_pereverzev when stiff",
    "",
    "actuators.ActuatorBank",
    "  two-breakpoint ramps  =  the",
    "  no-recompile contract",
    "",
    "task.TaskSpec.resolve_for(device)",
    "  setpoint + tolerance as",
    "  fractions of the measured band",
], VERIFIED, sub="turns a device into a plant"))

bottoms.append(box(X[2], B1, W, "TORAX 1.4.3", [
    "1-D core transport solver, JAX",
    "",
    "6 ms / step, after 4.4 s compile",
    "negative control 570-600x slower",
    "",
    "transport: constant | qlknn",
    "linear solver, fixed dt",
    "",
    "adaptive_dt must be OFF on the",
    "differentiable path (NaN tangent)",
], EXTERNAL, sub="external, pinned"))

bottoms.append(box(X[3], B1, W, "envs/torax_env.py", [
    "ToraxCore       lifecycle + stepping",
    "ToraxDeviceEnv  Gymnasium-shaped",
    "",
    "obs (9): beta_N, err, target, q95,",
    "   fgw, li3, H98, f_bootstrap, t/T",
    "act (2): aux_heat, ecrh  in [-1,1]",
    "",
    "reward = tracking + limit margin",
    "            + early-exit charge",
    "",
    "limits: greenwald / beta_N / q95",
    "solver failure + NaN = violations",
], VERIFIED, sub="the RL surface"))

for i in range(3):
    arrow(ax, (X[i] + W, 80.0), (X[i + 1], 80.0))
tag(ax, X[0] + W + 1.2, 83.2, "device")
tag(ax, X[1] + W + 1.2, 83.2, "config\ndict")
tag(ax, X[2] + W + 1.2, 83.2, "state +\npost")
arrow(ax, (X[3], 70.0), (X[2] + W, 70.0))
tag(ax, X[2] + W + 1.2, 66.9, "ramps")

# ============ BAND 2 — physics, federation, metrics, agents ================
B2 = min(bottoms) - GAP
bottoms = []
bottoms.append(box(X[0], B2, W, "physics/dimensionless.py", [
    "encode(...) -> rho*, nu*, beta_N,",
    "               q95, mach",
    "",
    "Connor-Taylor: two plasmas at the",
    "same point are the same plasma.",
    "rho* spans a decade across the set.",
], VERIFIED, sub="why transfer should work at all"))

bottoms.append(box(X[1], B2, W, "federation/similarity.py", [
    "distance in (rho*, nu*, beta, q)",
    "   log-scaled, weighted",
    "Gaussian kernel -> weight",
    "staleness: rounds + config epoch",
    "aggregation_weights(...)",
    "   role-matched, by cluster",
], WRITTEN, sub="unit-tested, not yet in a loop"))

bottoms.append(box(X[2], B2, W, "metrics/", [
    "RunLog     shots, never grad steps",
    "curves.py  the six metrics",
    "   censoring kept explicit;",
    "   speedup_ratio refuses a number",
    "   when censoring differs",
    "plots.py   the six figures",
], VERIFIED, sub="the output contract"))

bottoms.append(box(X[3], B2, W, "agents/", [
    "MLPPolicy  9 -> 16 -> 2, tanh",
    "   194 params = 776 bytes",
    "   <-  the federated payload",
    "",
    "cem.train_cem   gradient-free",
    "   proves the env is learnable",
], VERIFIED, sub="policy + trainer"))

# the RL loop: short and vertical, env sits directly above agents
arrow(ax, (X[3] + 6.0, B2 + GAP), (X[3] + 6.0, B2))
tag(ax, X[3] + 6.0, B2 + GAP / 2, "obs, reward")
arrow(ax, (X[3] + 17.0, B2), (X[3] + 17.0, B2 + GAP))
tag(ax, X[3] + 17.0, B2 + GAP / 2, "action")

arrow(ax, (X[0] + W, B2 - 9.0), (X[1], B2 - 9.0))
tag(ax, X[0] + W + 1.2, B2 - 5.8, "operating\npoint")
arrow(ax, (X[3], B2 - 9.0), (X[2] + W, B2 - 9.0))
tag(ax, X[2] + W + 1.2, B2 - 5.8, "shot\nrecords")
B2_BOT = min(bottoms)
arrow(ax, (X[1] + 11.5, B2_BOT), (X[3] + 11.5, B2_BOT), colour="#a8adb1",
      ls="--", rad=0.12)
tag(ax, 62.0, B2_BOT - 2.0, "aggregated weights  (Phase 5, not yet wired)",
    fs=6.3, colour="#a8adb1")

# ================ BAND 3 — identification, what's next, blocked ===========
B3 = B2_BOT - GAP - 1.6
box(X[0], B3, W * 2 + 2.0, "identification/   per-device physics ID", [
    "closure.py        chi(rho) from the INTEGRATED power balance -- conservation imposed exactly,",
    "                  the closure left free.  ~1.3% error on clean data.",
    "torax_inverse.py  differentiate through TORAX in FORWARD mode (jacfwd); reverse mode cannot",
    "                  cross lax.while_loop.  Z_eff recovered to 0.5% clean, 1.0% noisy + sparse.",
    "crlb.py           identifiability screen -- it PREDICTED which parameter would fail:",
    "                  0/4 identifiable at 10%, Z_eff ~ resistivity_multiplier at |r| = 0.998.",
    "pinn.py           inverse PINN, kept as a documented negative result.",
], VERIFIED, sub="couples RL and PINNs rather than stacking them", fs=6.7)

box(X[2], B3, W, "Phases 2-3, 5-6", [
    "thermal cluster (2 agents)",
    "particle + current clusters",
    "the five-condition matrix",
    "catastrophe transfer",
], SCAFFOLD, sub="scaffolded, not implemented")

box(X[3], B3, W, "Phase 7  divertor detachment", [
    "SPEC.md's strongest use case.",
    "TORAX refuses edge/SOL models on",
    "circular geometry, and circular is",
    "the only file-free geometry.  Needs",
    "real equilibria -- and no SPARC",
    "equilibrium exists at all.",
], BLOCKED, sub="blocked, not merely deferred")

# -------------------------------------------------------------- legend -----
legend = [(VERIFIED, "run against live TORAX, passing"),
          (WRITTEN, "implemented + unit-tested, not yet exercised end to end"),
          (SCAFFOLD, "scaffolded, not implemented"),
          (BLOCKED, "blocked by TORAX / missing equilibria"),
          (EXTERNAL, "external dependency")]
for i, (c, t) in enumerate(legend):
    x0 = 1.0 + i * 20.0
    ax.add_patch(FancyBboxPatch((x0, 96.9), 2.0, 1.8,
                 boxstyle="round,pad=0.16,rounding_size=0.4",
                 linewidth=1.5, edgecolor=c, facecolor=FACE[c], zorder=3))
    ax.text(x0 + 2.8, 97.8, t, ha="left", va="center", fontsize=6.9,
            color="#333333")

fig.text(0.5, 0.974,
         "hfmarl-fusion  —  federated MARL for research tokamaks",
         ha="center", fontsize=17, fontweight="bold", color="#121212")
fig.text(0.5, 0.949,
         "branch audit/live-torax-fixes  ·  367 tests  ·  "
         "TORAX 1.4.3 running natively on Windows CPU",
         ha="center", fontsize=9.6, color="#5a5e62")

# ================================ the gates ================================
axg.text(0.6, 99, "the gates  —  each one is a number, and a FAIL is the point",
         ha="left", va="top", fontsize=11.5, fontweight="bold", color="#121212")

gates = [
    ("gate0_env", "does TORAX\nrun here?", "PASS",
     "18/18 scalars present\nfirst run, unmodified", VERIFIED),
    ("gate0_jit", "does JIT survive\nper-step actions?", "PASS",
     "6 ms/step, spread 1.2x\ncontrol 601x slower", VERIFIED),
    ("gate_authority", "is the setpoint\nreachable?", "PASS 4/4",
     "was 1/4 — added by the\naudit, and found the bug", VERIFIED),
    ("gate1_thermal", "tracks setpoint\nunder limits?", "PASS 4/4",
     "on 'moderate'; beats the\nbest baseline by >10x", VERIFIED),
    ("gate_identify", "is anything\nidentifiable?", "RUNS",
     "0/4 at 10%; two params\ncollinear at r = 0.998", VERIFIED),
    ("exp_recover", "recover a hidden\nTORAX config?", "PARTIAL",
     "Z_eff 0.5%, Qei 7.5%\n— the CRLB said so", WRITTEN),
    ("gate_headroom", "is the task\nhard enough?", "FAIL",
     "'easy' converges in 240 shots,\nunder the 1000-shot floor", BLOCKED),
    ("gate0_bench", "CPU or GPU?", "NOT RUN",
     "informational only\nthis venv is CPU-only", SCAFFOLD),
]
w, gap = 11.20, 1.25
for i, (nm, q, verdict, detail, colour) in enumerate(gates):
    x0 = 0.6 + i * (w + gap)
    axg.add_patch(FancyBboxPatch(
        (x0, 16), w, 70, boxstyle="round,pad=0.36,rounding_size=1.0",
        linewidth=1.8, edgecolor=colour, facecolor=FACE[colour], zorder=2))
    axg.text(x0 + w / 2, 81, nm, ha="center", va="top", fontsize=8.2,
             fontweight="bold", color="#141414", zorder=3)
    axg.text(x0 + w / 2, 73, q, ha="center", va="top", fontsize=6.9,
             style="italic", color="#5a5e62", zorder=3)
    axg.text(x0 + w / 2, 50, verdict, ha="center", va="center", fontsize=11,
             fontweight="bold", color=colour, zorder=3)
    axg.text(x0 + w / 2, 37, detail, ha="center", va="top", fontsize=6.8,
             color="#2b2b2b", zorder=3)
    if i < len(gates) - 1:
        arrow(axg, (x0 + w, 51), (x0 + w + gap, 51), colour="#b9bdc0", lw=1.3)

axg.text(0.6, 7,
         "Order matters: a gate's result is meaningless if the one before it failed.  "
         "gate_authority sits ahead of gate1 because an unreachable setpoint looks "
         "exactly like a learning failure — and costs an afternoon to tell apart.",
         ha="left", va="center", fontsize=7.8, color="#5a5e62")

out = Path(__file__).resolve().parents[1] / "results" / "audit"
out.mkdir(parents=True, exist_ok=True)
path = out / "architecture.png"
fig.savefig(path, dpi=135, facecolor="white")
print("wrote", path)
