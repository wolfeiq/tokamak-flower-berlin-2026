"""Append gate results to FINDINGS.md.

Gate scripts print numbers. Numbers that are only printed get lost, and
SPEC.md §8 is explicit that cross-device comparisons cannot be reconstructed
afterwards. So every gate writes a timestamped, environment-stamped block to
FINDINGS.md as well as stdout.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def environment_stamp() -> str:
    """Everything needed to know whether two results are comparable."""
    bits = [f"host={platform.node()}", f"platform={platform.platform()}",
            f"python={platform.python_version()}"]
    try:
        import jax

        bits.append(f"jax={jax.__version__}")
        bits.append(f"devices={[d.platform + ':' + d.device_kind for d in jax.devices()]}")
        bits.append(f"x64={jax.config.read('jax_enable_x64')}")
    except Exception as e:
        bits.append(f"jax=unavailable({type(e).__name__})")
    try:
        import torax

        bits.append(f"torax={getattr(torax, '__version__', 'unknown')}")
    except Exception:
        bits.append("torax=unavailable")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(), capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0:
            bits.append(f"commit={sha.stdout.strip()}")
    except Exception:
        pass
    return "  \n".join(bits)


def record(title: str, body: str, path: Path | None = None) -> Path:
    """Append one result block. Returns the file written."""
    target = path or (repo_root() / "FINDINGS.md")
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n\n---\n\n## {title}\n\n"
        f"*{stamp}*\n\n"
        f"{environment_stamp()}\n\n"
        f"{body}\n"
    )
    # ENCODING IS NOT A DETAIL HERE. `open` without one uses the platform
    # default -- cp1252 on Windows -- so one non-ASCII character in a gate's
    # output writes a byte that is not valid UTF-8 into a file every other
    # tool reads as UTF-8, and the corruption is silent at write time.
    with target.open("a", encoding="utf-8") as fh:
        fh.write(block)
    return target


def peak_rss_mb() -> float:
    """Peak resident set size in MB. Best-effort; 0.0 if unavailable.

    `resource` is POSIX-only, so on Windows this silently returned 0.0 and
    every gate reported "peak RSS: 0 MB" -- a memory figure that looks
    measured and is not. psutil ships as a TORAX dependency, so it is the
    fallback; it reports the current working set rather than the historical
    peak, which is the honest best available on that platform.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3
    except Exception:
        pass
    try:
        import os

        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        # peak_wset is Windows-only and is the true peak; rss is the current
        # working set and is what every other platform falls back to.
        return float(getattr(info, "peak_wset", info.rss)) / 1e6
    except Exception:
        return 0.0
