# Speaker script

## Block 1 — Flower LLM investigation agents

### Opening

Our project explores collaboration between tokamak facilities in two ways:
sharing evidence for an investigation, and sharing learned policies for control
research. I will start with the Flower LLM-agent application.

### The problem — show the 3D reactor assembly

Imagine a tokamak's plasma is colder than expected. Was less heating power
delivered? Did heat transport increase? Or is a measurement misleading us?
These explanations can look similar but lead to different decisions.

The 3D view helps us locate the discussion in a machine. It is illustrative
geometry; the plasma physics comes from separate models. Our question here is:
can experience from another facility help us choose the next independent
measurement?

### Research background — Princeton and cross-device transfer

AI already has experimental results in fusion. Princeton-led researchers
demonstrated a reinforcement-learning controller on DIII-D that reduced the
likelihood of tearing instabilities while maintaining high-performance plasma
operation. PACMAN provides an architecture for integrating machine-learning
predictors and controllers into DIII-D, from diagnostic processing to actuation.

Sources: [tearing avoidance](https://www.nature.com/articles/s41586-024-07024-9),
[PACMAN](https://arxiv.org/abs/2511.08818).

There is also evidence that knowledge can transfer between machines. One study
adapted a disruption predictor from J-TEXT to EAST using only 20 EAST discharges,
reaching performance comparable to direct training on about 1,900 EAST
discharges. This was prediction and transfer learning. It motivates our
cross-facility question; it does not establish our agent's benefit.

Source: [cross-device transfer](https://www.nature.com/articles/s42005-023-01296-9).

### Our application

Our project uses TORAX, Google DeepMind's tokamak transport simulator, for the
control and cold-start experiments in the second block.

Alongside those experiments, we built an investigation assistant using the
actual Flower AgentApp harness. Its proposed role is between experiments:
requesting evidence, comparing cases and recommending a diagnostic check.

An investigator requests evidence from facility-specific stewards. Each steward
has a separate model context, while a software gateway enforces which evidence
can be released. The workflow records what was requested, shared or refused.

Sources: [TORAX](https://github.com/google-deepmind/torax),
[Flower Agent](https://flower.ai/docs/agent/index.html).

### Why federation

If partners can pool their records, a central agent is a reasonable baseline.
Our proposed use case is collaboration between independently operated
facilities that retain control over their records and release policies.
The investigator can request approved analyses without requiring every partner
to supply its complete raw dataset to a central workspace.

That is the value we are exploring with Flower. Today the facilities are
simulated; real distributed deployment and the benefit to investigators still
need validation.

### Switch to the application — separate tab

Open the research app at `http://127.0.0.1:8787/#thermal`.
The presentation stays open on port 8788.

For this demonstration, the cases come from a reduced one-dimensional thermal
model. They are separate from the TORAX control study.

At Facility A, an apparent transport anomaly remains unresolved because we
lack an independent measurement of delivered heating power.

At Facility B, an independent power audit changes the interpretation of its
own case. It suggests a useful measurement for A; it cannot establish A's cause.

Facility C fails the demonstration's criterion for a comparable profile, so
that analogy is rejected. We finish with a justified next step and an evidence
trail, rather than claiming a diagnosis we cannot establish.

If using **Run local demonstration**, say: “This replay follows a scripted
sequence through the actual gateway. It makes no live LLM call.” A saved report
must be introduced as a recorded run. A fresh hosted run uses account quota
and must be deliberately launched in the application.

### Transition

The agent block asks what evidence we should request from another facility.
The control block asks whether experience from other devices can reduce the
local trials a new controller needs. That brings us to TORAX and HFMARL.

## Block 2 — TORAX / HFMARL introduction and atlas cue

This is the introduction and visualization cue for the second block. Its full
results narration is separate work; no new results are asserted here.

We use TORAX to evolve one-dimensional core plasma profiles in our control
experiments. Our saved cold-start study tests source handover followed by local
learning with policy search. It is not a PPO or SAC experiment.

**Show the federation atlas.** This places our device models at the geographic
locations of the facilities that inspired them. Click a machine to include its
synthetic update; click it again to remove it. Drag to rotate the scene. The
participation weights are computed by our aggregation code.

The coloured channels represent thermal, particle and current control roles.
They illustrate the broader control architecture, not the LLM investigation
workflow and not the scope of the thermal-only cold-start study. These are
synthetic updates at nominal operating points. There are no live reactor
connections, and the animation is not evidence of improved learning.

The experiments must answer that question through comparisons with local
learning, source costs and paired outcomes. TORAX's one-dimensional transport
scope also limits what we can claim about physical-reactor performance.
