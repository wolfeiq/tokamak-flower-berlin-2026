# Setup — Windows + WSL2 + NVIDIA

Target: Ubuntu under WSL2, Python 3.12, TORAX 1.4.3.

Read §1 before installing anything. Two WSL-specific mistakes will cost you an
afternoon each.

---

## 1. Two things that catch everyone on WSL

**The NVIDIA driver goes on Windows, not inside WSL.**
Install the standard NVIDIA Windows driver (or NVIDIA's WSL-CUDA driver) on the
Windows side. Do **not** `apt install nvidia-driver-*` inside Ubuntu — that
overwrites the WSL passthrough stubs and breaks GPU access in a way that is
awkward to undo. The CUDA *libraries* come from the pip wheel, so you do not
need a system CUDA toolkit either.

Check it works from inside WSL before installing anything else:

```bash
nvidia-smi          # must print your GPU. If not, fix Windows-side first.
```

**Keep the repo on the Linux filesystem, not `/mnt/c`.**
Filesystem calls that cross the Windows boundary are roughly an order of
magnitude slower, and this project does a lot of small-file I/O (JAX's
compilation cache especially). Clone into `~`, not `/mnt/c/Users/...`.

```bash
cd ~ && git clone <this repo> hfmarl-fusion && cd hfmarl-fusion
```

---

## 2. Install

```bash
# uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/hfmarl-fusion
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e .            # TORAX 1.4.3 + gymnasium + numpy/scipy
uv pip install -e '.[dev]'     # pytest, ruff
```

TORAX 1.4.3 requires Python ≥ 3.11; 3.12 is what upstream CI uses. Do not use
3.13+ without checking — TORAX `main` has already moved its floor to 3.12 and
the ecosystem around JAX moves fast.

### GPU (optional — read §5 before assuming you want it)

```bash
uv pip install -e '.[gpu]'     # jax[cuda12]
python -c "import jax; print(jax.devices())"    # expect CudaDevice(...)
```

If that prints only `CpuDevice`, the Windows driver is the first thing to check.

---

## 3. Environment variables

Put these in your shell profile. They matter more than they look.

```bash
# Cache compiled kernels across runs. Without this you pay TORAX's compile
# cost on every single script invocation, which dominates short experiments.
export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0

# Leave TORAX error checking OFF during training.
# It injects Python callbacks that JAX cannot serialise, which SILENTLY
# DISABLES the compilation cache above. Turn it on only when debugging a
# specific solver failure.
# export TORAX_ERRORS_ENABLED=True
```

That interaction — error checking disabling the compile cache — is documented
in TORAX's `docs/using_jax.rst` and is easy to lose hours to.

---

## 4. Verify, in this order

Do not skip ahead. Each gate's result is meaningless if the previous one failed.

```bash
# 0. Physics and metric code, no TORAX needed. Should be instant.
uv run pytest                            # expect ~116 passed

# 0'. Device set in dimensionless space. No TORAX needed.
uv run python scripts/describe_devices.py

# 1. Does TORAX run here, and does it report the quantities the limits need?
uv run python scripts/gate0_env.py

# 2. THE gate. Does JIT survive per-step actuator changes?
#    Always pass --negative-control: it proves the test can actually fail.
uv run python scripts/gate0_jit.py --negative-control

# 3. Is the GPU actually faster? (Often not — see §5.)
uv run python scripts/gate0_bench.py

# 4. Phase 1: one thermal agent tracking a setpoint under limits.
uv run python scripts/gate1_thermal.py --device iter_like
```

Every gate appends its numbers to `FINDINGS.md`. That file is the project's
record; commit it.

---

## 5. Do not assume the GPU wins

Two reasons it may lose to the CPU here:

1. **TORAX runs in float64 by default.** Consumer NVIDIA cards (GeForce, most
   RTX) execute f64 at about 1/32 of their f32 rate. Datacentre cards (A100,
   H100) do not have this penalty.
2. **This is a 1-D problem on ~25 radial cells.** There is very little
   arithmetic per kernel, so kernel-launch overhead can dominate entirely.

`gate0_bench.py` measures CPU/GPU × f64/f32 and tells you which to use.

The likely real benefit of this machine is **RAM and core count** — Phase 5
runs 5 baselines × 4 devices × several seeds, which parallelises across
processes far better than it accelerates on one GPU. Plan for parallel CPU
clients unless the benchmark says otherwise.

---

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `gate0_env.py` fails config validation | TORAX is not 1.4.3. `main` replaced the flat `transport: {model_name}` block with a `core_transport_models` registry. Re-pin. |
| Every step takes as long as the first | The no-recompile contract is broken. Check `tests/test_actuators.py` passes, then that no action touches a `STATIC_EXACT` path. |
| Compile cost paid on every script run | `JAX_COMPILATION_CACHE_DIR` unset, or `TORAX_ERRORS_ENABLED=True` is silently disabling the cache. |
| `jax.devices()` shows CPU only | Windows-side NVIDIA driver. Check `nvidia-smi` inside WSL first. |
| Everything is mysteriously slow | Repo is on `/mnt/c`. Move it to `~`. |
| Import errors for `torax` in tests | Expected and fine — only `gate*` scripts need TORAX. `pytest` must pass without it. |
