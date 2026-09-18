# Speaker script

## Slide 1 — Tokamak collaboration

Our project explores collaboration between tokamak facilities in two ways:
sharing evidence for an investigation, and sharing learned policies for control
research. We built a Flower LLM-agent application alongside a control study
using TORAX, Google DeepMind's tokamak transport simulator.

The opening model is illustrative geometry. The physics experiments are separate.

## Slide 2 — The problem

Each machine has limited local experience, and experience from another tokamak
does not automatically transfer. Geometry, diagnostics and operating regimes differ.

Imagine a plasma is colder than expected. Was less heating power delivered,
did heat transport increase, or is a measurement misleading us? Similar-looking
observations can require different decisions.

Meanwhile, independently operated facilities may need to retain control over
their records. If records can be pooled, a central approach is a reasonable
baseline. We explore collaboration when partners need to decide what to disclose.

## Slide 3 — What already exists

There are four foundations for this work. Princeton-led researchers demonstrated
reinforcement learning on DIII-D for avoiding tearing instabilities while
maintaining high-performance operation. PACMAN connects diagnostic processing,
machine-learning predictors and controllers to DIII-D's control system.

Cross-device transfer has also shown promise: a disruption predictor adapted
from J-TEXT to EAST with 20 EAST discharges achieved performance comparable to
direct training on about 1,900 EAST discharges. That was a prediction study.

Finally, TORAX provides a JAX-based, one-dimensional tokamak transport simulator.
We use TORAX for our control experiments. These studies motivate our project;
they do not establish a benefit from our particular federation design.

Sources: [Princeton-led tearing avoidance](https://www.nature.com/articles/s41586-024-07024-9),
[PACMAN](https://arxiv.org/abs/2511.08818),
[cross-device transfer](https://www.nature.com/articles/s42005-023-01296-9),
[TORAX](https://github.com/google-deepmind/torax).

## Slide 4 — The two things we built

First, we built an investigation assistant with the actual Flower AgentApp
harness. An investigator requests approved evidence from facility stewards,
compares thermal anomalies and recommends an independent diagnostic check.
Its use case is helping a researcher decide what evidence to collect next.
The current demonstration uses synthetic cases and a reduced thermal model.

Second, we developed TORAX-based control and cold-start experiments within the
HFMARL research project. The use case is warming up a new device's controller
with experience from other devices and testing whether it needs fewer local trials.

The completed study uses thermal policy search. The broader architecture includes
multiple control roles, but this study does not validate a full MARL controller
and is not a PPO or SAC experiment. Results are mixed, so we do not claim a
general federation speedup.

Source: [Flower Agent harness](https://flower.ai/docs/agent/index.html).

## Slide 5 — Federation atlas

This interactive map places our device models at the geographic locations of
the facilities that inspired them. Click a machine to include or remove its
synthetic update; drag to rotate the scene. The displayed weights come from our
aggregation code.

The coloured channels represent thermal, particle and current control roles.
They illustrate the broader control architecture. These are synthetic updates
at nominal operating points, with no live reactor connections. The animation
explains the design; the experiments assess whether sharing helps learning.

## Slide 6 — Architecture comparison

On the left is our MARL system-design image: a flat team within a device and
role-matched, asynchronous exchange across devices. It is an earlier audit
snapshot, so its test counts and status labels should not be read as current.
Its central idea is exchanging policy parameters to support controller learning
in TORAX. The current cold-start experiment covers the thermal policy-search path.

On the right is the Flower investigation workflow. The research interface passes
a question to the investigator in the Flower AgentApp runtime. The investigator
consults facility-specific stewards, whose software gateways enforce what local
evidence can be released. Approved findings return to the investigator and are
assembled into an advisory report for the researcher.

The key comparison is what they exchange and what they produce: learned policy
parameters for control research, versus approved findings for an investigation.
The Flower assistant recommends a next diagnostic step; it does not actuate a reactor.

Click either architecture image to inspect it at full resolution.

## Slide 7 — Cold-start results

Cold start is the challenge of learning a reliable controller for a new tokamak
with little local experience, and federation aims to reduce the required trials
by sharing policies learned on other devices.
In our TORAX study, uniform federation lowered SPARC-like's median local shots
to competence from 23.5 to 16 (about 32%), but the six paired seeds split evenly
between faster, tied and slower outcomes, so this suggests a possible
device-specific benefit rather than a proven general solution.

If asked about the analysis: selection trials are excluded from confirmation
but still charged as local shots. Competence requires two consecutive evaluations
that complete, stay within limits and track within tolerance. Source training
used 240 shots beyond these local totals. TCV-like went the other way, from 13.5
to 18.5 median local shots. This is thermal policy search, not full MARL validation.

## Optional application demonstration — separate tab

Open the research app at `http://127.0.0.1:8787/#thermal`; keep the presentation
on port 8788. At Facility A, an apparent transport anomaly remains unresolved
without an independent delivered-power measurement. Facility B's power audit
suggests that measurement for A, but cannot establish A's cause. Facility C
fails the demonstration's comparability criterion, so that analogy is rejected.

If using **Run local demonstration**, introduce it as a scripted replay through
the actual gateway with no live LLM call. Introduce a saved report as a recorded
run. A fresh hosted run consumes account quota and is launched deliberately in
the application. Real distributed deployment and investigator benefit still
need validation.
