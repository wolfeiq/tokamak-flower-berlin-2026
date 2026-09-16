"""System design: the three-level stack, and the federation that joins devices.

`fig_architecture.py` draws the code -- which module calls which. This draws the
SYSTEM: who decides what, on which clock, and what crosses between machines.

Every box is coloured by what is true today, not by what SPEC.md asks for.
Two of the three agents are grey because their actuators are disabled; the
numbers in the aggregation panel are real output from `describe_devices.py`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERIFIED = "#2e8b3d"
WRITTEN = "#d98a1f"
SCAFFOLD = "#8b9096"
EXTERNAL = "#3d6ba5"
BLOCKED = "#c8342f"
FACE = {VERIFIED: "#ecf7ed", WRITTEN: "#fdf4e5", SCAFFOLD: "#f2f3f4",
        EXTERNAL: "#eaf0f8", BLOCKED: "#fdecec"}
INK = "#141414"
MUTED = "#5a5e62"

fig = plt.figure(figsize=(19.2, 13.6))
gs = fig.add_gridspec(2, 1, height_ratios=[3.05, 0.82], hspace=0.07,
                      left=0.008, right=0.992, top=0.925, bottom=0.018)
gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.045)
axL = fig.add_subplot(gs_top[0])
axR = fig.add_subplot(gs_top[1])
axB = fig.add_subplot(gs[1])
for a in (axL, axR, axB):
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.axis("off")

# The panels have very different aspect ratios, so a "line" is a different
# number of y-units in each. Every box computes its own height from its
# content in the metrics of the panel it lives on -- boxes sized by eye is
# exactly how text ends up sitting on a border.
METRICS = {
    "tall": dict(pad_t=2.4, title_h=3.4, sub_h=3.0, gap=0.4, line_h=2.5,
                 pad_b=1.8),
    "wide": dict(pad_t=5.0, title_h=7.0, sub_h=6.0, gap=1.0, line_h=5.4,
                 pad_b=4.5),
}


def box(ax, x, top, w, colour, title=None, lines=(), sub=None, m="tall",
        tfs=8.4, fs=6.8, align="center", zorder=2, face=None, min_h=0.0,
        dashed=False):
    """Draw a box anchored by its TOP edge; height follows content.

    Returns the bottom edge, so the next row can be placed from it.
    """
    g = METRICS[m]
    h = (g["pad_t"] + (g["title_h"] if title else 0)
         + (g["sub_h"] if sub else 0)
         + (g["gap"] + len(lines) * g["line_h"] if lines else 0)
         + g["pad_b"])
    h = max(h, min_h)
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.38,rounding_size=1.1",
        linewidth=1.7, edgecolor=colour,
        facecolor=FACE[colour] if face is None else face, zorder=zorder))
    cx = x + w / 2
    cursor = top - g["pad_t"]
    if title:
        ax.text(cx, cursor - g["title_h"] / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=INK, zorder=zorder + 1)
        cursor -= g["title_h"]
    if sub:
        ax.text(cx, cursor - g["sub_h"] / 2, sub, ha="center", va="center",
                fontsize=fs - 0.3, style="italic", color=colour,
                zorder=zorder + 1)
        cursor -= g["sub_h"]
    if lines:
        cursor -= g["gap"]
        tx = cx if align == "center" else x + 2.8
        for ln in lines:
            ax.text(tx, cursor - g["line_h"] / 2, ln, ha=align, va="center",
                    fontsize=fs, color="#2b2b2b", zorder=zorder + 1)
            cursor -= g["line_h"]
    return y


def arrow(ax, p0, p1, colour="#6f7478", lw=1.5, zorder=5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=lw,
        color=colour, zorder=zorder, shrinkA=0, shrinkB=0))


def tag(ax, x, y, text, colour=MUTED, fs=6.3):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=colour,
            zorder=7, bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                                edgecolor="none"))


# =========================== LEFT: one device ===============================
axL.text(0.5, 98.5, "ONE DEVICE  —  a flat team, one agent per cluster",
         ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
axL.text(0.5, 94.6,
         "No head sets targets for anyone. Cross-cluster conflict is "
         "resolved by the shared reward.",
         ha="left", va="top", fontsize=7.6, color=MUTED)

box(axL, 1, 89, 97, BLOCKED, dashed=True,
    title="REMOVED 2026-09-15   ·   cluster heads (level 2) and the "
          "device head (level 3)",
    lines=["The plant cannot exhibit the problem they solve: two live "
           "actuators, one scalar objective, and an isolated flat policy "
           "that reaches threshold in 240 shots.",
           "The one example SPEC.md gave for the device head — thermal "
           "wants more NBI, stability wants less — needs a stability "
           "cluster TORAX 1-D core cannot represent.",
           "Comes back when Phase 2 measures a flat team losing to a "
           "monolith on the same actuator set. Not before."],
    fs=6.6, tfs=8.2)

AGENTS = [
    ("THERMAL AGENT", VERIFIED, "runs today",
     ["actuators: aux_heat, ecrh", "TORAX: ion + electron heat transport"],
     ["aux_heat", "ecrh"]),
    ("PARTICLE AGENT", SCAFFOLD, "actuator disabled",
     ["actuators: gas_puff, pellets", "TORAX: particle transport"],
     ["gas puff", "pellets"]),
    ("CURRENT AGENT", SCAFFOLD, "actuator disabled",
     ["actuators: Ip, ECCD, NBCD", "TORAX: current diffusion"],
     ["Ip", "ECCD", "NBCD"]),
]
lane_w, lane_gap = 30.0, 3.5
lane_x = [1.0 + i * (lane_w + lane_gap) for i in range(3)]
AGENT_TOP, CHIP_TOP, CHIP_H = 70.0, 52.0, 7.0
BUS_Y = 41.0

for x0, (name, colour, state, lines, actuators) in zip(lane_x, AGENTS):
    abot = box(axL, x0, AGENT_TOP, lane_w, colour, title=name, sub=state,
               lines=lines, fs=6.5, tfs=8.0)
    chip_w = (lane_w - (len(actuators) - 1) * 1.4) / len(actuators)
    for j, act in enumerate(actuators):
        cx0 = x0 + j * (chip_w + 1.4)
        ccx = cx0 + chip_w / 2
        axL.add_patch(FancyBboxPatch(
            (cx0, CHIP_TOP - CHIP_H), chip_w, CHIP_H,
            boxstyle="round,pad=0.30,rounding_size=0.9", linewidth=1.5,
            edgecolor=colour, facecolor=FACE[colour], zorder=2))
        axL.text(ccx, CHIP_TOP - CHIP_H / 2, act, ha="center", va="center",
                 fontsize=6.9, fontweight="bold", color=INK, zorder=3)
        arrow(axL, (x0 + lane_w / 2, abot), (ccx, CHIP_TOP), lw=1.1)
        axL.plot([ccx, ccx], [CHIP_TOP - CHIP_H, BUS_Y],
                 color="#b9bdc0", lw=1.1, zorder=1)

axL.plot([6, 92], [BUS_Y, BUS_Y], color="#b9bdc0", lw=1.3, zorder=1)
tag(axL, 49, BUS_Y, "one shared team reward  ·  0.2–15 ms inference (PACMAN)")

dec_bot = box(axL, 4, 36, 42, SCAFFOLD,
              title="DECODER", sub="not written",
              lines=["dimensionless action  →  SI actuator command",
                     "25 MW of ICRF here is 1 MW of ECRH there"], fs=6.5)
enc_bot = box(axL, 54, 36, 42, VERIFIED,
              title="ENCODER", sub="dimensionless.py  ·  unit-tested",
              lines=["SI state  →  ρ*, ν*, β_N, q95, Mach",
                     "but the policy never sees ρ* or ν*"], fs=6.5)

arrow(axL, (25, BUS_Y), (25, 36), lw=2.0)
arrow(axL, (25, dec_bot), (25, 16), lw=2.0)
arrow(axL, (75, 16), (75, enc_bot), lw=2.0)
arrow(axL, (75, 36), (75, BUS_Y), lw=2.0)
tag(axL, 25, 38.6, "action", colour="#3f4346")
tag(axL, 75, 38.6, "state", colour="#3f4346")

box(axL, 13, 16, 74, EXTERNAL,
    title="TORAX 1.4.3   ·   the plant",
    lines=["1-D core transport, JAX, circular geometry",
           "6 ms/step  ·  0.66 s per shot (easy, iter_like, measured)"],
    fs=6.8)

# ======================= RIGHT: across devices ==============================
axR.text(0.5, 98.5, "ACROSS DEVICES  —  role-matched, asynchronous",
         ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
axR.text(0.5, 94.6,
         "Thermal aggregates only with thermal. Weighting is by distance in "
         "dimensionless space, not by sample count.",
         ha="left", va="top", fontsize=7.6, color=MUTED)

CARD_X, CARD_W, CARD_TOP, CARD_BOT = 2.0, 90.0, 90.0, 10.0
for k, (d, label) in enumerate([(6.0, "CURRENT channel"),
                                (3.0, "PARTICLE channel")]):
    axR.add_patch(FancyBboxPatch(
        (CARD_X + d, CARD_BOT - d), CARD_W, CARD_TOP - CARD_BOT,
        boxstyle="round,pad=0.38,rounding_size=1.1",
        linewidth=1.5, edgecolor="#c2c6c9", facecolor="#fbfbfc", zorder=1 + k))
    # Only the sliver BELOW the front card is visible, so the label goes
    # there -- at the top it would sit behind the thermal channel.
    axR.text(CARD_X + d + CARD_W - 2.0, CARD_BOT - d + 1.6, label,
             ha="right", va="center", fontsize=6.6, color="#9aa0a4",
             zorder=2 + k)

axR.add_patch(FancyBboxPatch(
    (CARD_X, CARD_BOT), CARD_W, CARD_TOP - CARD_BOT,
    boxstyle="round,pad=0.38,rounding_size=1.1",
    linewidth=1.9, edgecolor=WRITTEN, facecolor="#fffdf8", zorder=3))
axR.text(4.5, 87.2, "THERMAL channel", ha="left", va="center", fontsize=8.8,
         fontweight="bold", color=WRITTEN, zorder=4)
axR.text(90.0, 87.2, "a thermal update never enters another channel",
         ha="right", va="center", fontsize=6.6, style="italic", color=WRITTEN,
         zorder=4)

INNER_X, INNER_W = 4.5, 85.0

DEVICES = [
    ("iter_like", ["ρ*  1.4e-3", "ν*  0.022", "β_N 1.62"]),
    ("sparc_like", ["ρ*  1.9e-3", "ν*  0.034", "β_N 0.91"]),
    ("diiid_like", ["ρ*  6.0e-3", "ν*  0.034", "β_N 2.25"]),
    ("tcv_like", ["ρ*  1.4e-2", "ν*  0.095", "β_N 1.17"]),
]
dw, dgap, dx0 = 19.6, 2.3, INNER_X
dev_bot = 0.0
for i, (name, lines) in enumerate(DEVICES):
    x = dx0 + i * (dw + dgap)
    dev_bot = box(axR, x, 85.0, dw, VERIFIED, title=name, lines=lines,
                  fs=6.3, tfs=7.8, zorder=4)
    arrow(axR, (x + dw / 2, dev_bot), (x + dw / 2, 65.0), lw=1.4, zorder=5)

buf_bot = box(axR, INNER_X, 65.0, INNER_W, SCAFFOLD,
              title="FedBuff BUFFER   ·   asynchronous, no round barrier",
              lines=["tokamaks fire 20–40 shots/day on different campaigns; "
                     "some are down for months",
                     "updates land when a device happens to be running — "
                     "synchronous rounds are physically impossible"],
              fs=6.5, zorder=4)
arrow(axR, (47.0, buf_bot), (47.0, 47.5), lw=1.8, zorder=5)

agg_bot = box(axR, INNER_X, 47.5, INNER_W, WRITTEN,
              title="aggregation_weights(updates, target=iter_like)",
              sub="similarity.py  ·  unit-tested  ·  nothing consumes it yet",
              lines=["w_k  ∝  exp(−d_k² / 2h²)   ×   (1 + τ_k)^−α   ×   "
                     "epoch_penalty^(config epochs)",
                     "d_k = weighted distance in (ρ*, ν*, β_N, q95)      "
                     "h = 1.176 (median heuristic)      mixed clusters raise"],
              fs=6.5, zorder=4)
arrow(axR, (47.0, agg_bot), (47.0, 27.0), lw=1.8, zorder=5)

con_bot = box(axR, INNER_X, 27.0, INNER_W, VERIFIED,
              title="contribution to iter_like's thermal agent",
              lines=["sparc_like  0.880           diiid_like  0.644"
                     "           tcv_like  0.365",
                     "TCV is the negative control: furthest in dimensionless "
                     "space, and it contributes least — that is the claim",
                     "computed from the hand-written OPERATING_POINTS table, "
                     "not from TORAX output (RUNBOOK §5: re-derive)"],
              fs=6.5, zorder=4)

# The aggregated model goes back to the target device, routed outside the card.
fb_y = (con_bot + 27.0) / 2
fb_in = 77.0
fb_x = 1.0
for a, b in [((INNER_X, fb_y), (fb_x, fb_y)),
             ((fb_x, fb_y), (fb_x, fb_in))]:
    axR.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-", linewidth=1.5, color="#6f7478", zorder=6,
        shrinkA=0, shrinkB=0))
arrow(axR, (fb_x, fb_in), (INNER_X, fb_in), lw=1.5, zorder=6)
axR.text(2.9, (fb_y + fb_in) / 2,
         "personalised weights back to the target device",
         ha="center", va="center", fontsize=6.1, color="#3f4346",
         rotation=90, zorder=7)

axR.text(50.0, 1.2,
         "Three identical channels. Role matching is enforced, not assumed: "
         "mixing clusters raises rather than averaging nonsense.",
         ha="center", va="center", fontsize=7.2, color=MUTED, style="italic")

# =========================== BOTTOM: reality ================================
axB.text(0.6, 96, "WHAT ACTUALLY RUNS TODAY", ha="left", va="top",
         fontsize=11.0, fontweight="bold", color=INK)

REALITY = [
    ("the control path", VERIFIED,
     ["torax_env  obs(9)  →  MLPPolicy 9→16→2  →  aux_heat, ecrh",
      "One thermal agent per device, which as of today IS the",
      "architecture rather than a shortfall against it."]),
    ("the federated payload", WRITTEN,
     ["194 parameters = 776 bytes, pinned by a test — a whole policy's",
      "worth of numbers, typed as a delta and never yet produced.",
      "Bandwidth is not a constraint anyone has to argue about."]),
    ("the federation loop", SCAFFOLD,
     ["Distance, kernel, staleness, role-matched weights: all tested.",
      "Buffer, rounds, Flower, the feedback arrow: none of it exists.",
      "No aggregated weight has ever reached a policy."]),
    ("the blocking gate, now run", BLOCKED,
     ["gate_headroom, easy on iter_like: isolated reaches threshold in",
      "240 shots — under the 1000-shot floor, so any federation speedup",
      "is campaign noise. FAIL (headroom): the task must get harder.",
      "Cost is not the constraint — 0.66 s/shot puts the matrix at ~22 h."]),
]
rw, rgap = 23.4, 1.7
for i, (name, colour, lines) in enumerate(REALITY):
    x = 0.6 + i * (rw + rgap)
    box(axB, x, 84, rw, colour, title=name, lines=lines, m="wide",
        fs=6.6, tfs=8.2, align="left")

fig.text(0.5, 0.973, "hfmarl-fusion  —  system design",
         ha="center", fontsize=17, fontweight="bold", color="#121212")
fig.text(0.5, 0.949,
         "federated MARL for research tokamaks   ·   "
         "green = runs against live TORAX   orange = written and tested, "
         "not yet looped   grey = designed, not built   "
         "red = measured, and failing   blue = external",
         ha="center", fontsize=9.4, color=MUTED)

out = Path(__file__).resolve().parents[1] / "results" / "audit"
out.mkdir(parents=True, exist_ok=True)
path = out / "system_design.png"
fig.savefig(path, dpi=135, facecolor="white")
print("wrote", path)
