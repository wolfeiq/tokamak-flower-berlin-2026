#!/usr/bin/env python
"""Kill the newest experiment before the machine starts swapping.

    uv run python scripts/guard_memory.py --floor-mb 900

This runs unattended for hours on a 16 GB laptop that has other things open.
Each TORAX process holds 0.6-2 GB, and the failure mode is not a clean
exception -- it is the whole machine thrashing, which loses every running
experiment at once and takes the desktop with it.

WHY IT KILLS THE NEWEST
-----------------------
The newest process is the one that has produced the least. Killing the oldest
would throw away the most work, and killing by size would preferentially kill
whichever experiment happens to hold the biggest JAX cache -- which is not the
same as the one it is cheapest to lose.

It never kills itself, never kills a process it did not recognise as an
experiment, and writes what it did and why to a log the next check-in reads. A
guard that acts silently is indistinguishable from a crash.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import psutil

# Substrings identifying a runnable experiment. Anything else -- the editor,
# the shell, an unrelated Python -- is left alone.
EXPERIMENT_MARKERS = ("scripts/exp_", "scripts\\exp_", "scripts/gate_",
                      "scripts\\gate_", "scripts/measure_", "scripts\\measure_")


def experiment_processes() -> list[psutil.Process]:
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if any(m in cmd for m in EXPERIMENT_MARKERS):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(out, key=lambda p: p.info["create_time"])


def victim_processes() -> list[psutil.Process]:
    """Everything the guard may kill, oldest first.

    Experiments plus TEST RUNS. The first real kill sacrificed a six-seed
    experiment -- an hour of work -- because a full pytest run in the
    foreground had pushed available memory under the floor. pytest imports
    JAX and TORAX and costs about as much as an experiment, but it was
    invisible to the guard, so the guard protected the machine by killing the
    only thing it could see.

    A test suite takes thirty seconds to re-run. That makes it the cheapest
    thing on the machine to lose, and therefore the right victim whenever it
    is the newest -- which, being something I launch by hand while runs are
    already going, it almost always is. "Kill the newest" was always a proxy
    for "kill what is cheapest to lose"; this makes the proxy less wrong.
    """
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if any(m in cmd for m in EXPERIMENT_MARKERS) or _is_pytest(cmd):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(out, key=lambda p: p.info["create_time"])


# Things that are cheap to lose and that I launch by hand while runs are
# already going: the test suite, and one-off analysis scripts, which live in
# the session scratchpad. Neither is an experiment, so neither was visible to
# the guard; both cost as much memory as one, because both import JAX and
# TORAX. An ad-hoc script that is invisible here does not merely go
# unprotected -- it makes the guard kill an EXPERIMENT to make room for it.
EXPENDABLE_MARKERS = ("pytest", "/scratchpad/",
                      "\\scratchpad\\")


def _is_pytest(cmd: str) -> bool:
    """Named for the first case; covers everything expendable."""
    return any(m in cmd for m in EXPENDABLE_MARKERS)


def _is_guard(cmdline) -> bool:
    """True only for a process actually RUNNING this script.

    Substring-matching the whole command line is not enough: `python -c
    "import guard_memory"` and `grep guard_memory` both contain the name, and
    either would make the interlock below refuse to start a guard when none
    was running. The script has to appear as an argument in its own right.
    """
    return any(str(a).replace("\\", "/").endswith("scripts/guard_memory.py")
               or str(a).replace("\\", "/").endswith("/guard_memory.py")
               or str(a) == "guard_memory.py"
               for a in (cmdline or []))


def hogs(top: int = 4) -> str:
    """The largest python processes right now, tagged by role."""
    rows = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if any(m in cmd for m in EXPERIMENT_MARKERS):
                tag = "exp"
            elif _is_guard(p.info["cmdline"]):
                tag = "guard"
            elif _is_pytest(cmd):
                tag = "pytest"
            else:
                tag = "other"
            rows.append((p.memory_info().rss / 1e6, p.info["pid"], tag))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(reverse=True)
    return ", ".join(f"{tag} pid {pid} {mb:.0f} MB"
                     for mb, pid, tag in rows[:top]) or "no python processes"


def _own_chain() -> set[int]:
    """This process, its ancestors and its children.

    On Windows the venv launcher (`.venv/Scripts/python.exe`) execs a second
    interpreter, so every guard shows up in the process table TWICE with the
    same command line. Without this the interlock below would see its own
    shim and refuse to start.
    """
    me = psutil.Process()
    chain = {me.pid}
    try:
        for a in me.parents():
            chain.add(a.pid)
        for c in me.children(recursive=True):
            chain.add(c.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return chain


def other_guards() -> list[int]:
    """Guard processes that are not this one."""
    mine = _own_chain()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in mine:
                continue
            if "python" not in (p.info["name"] or "").lower():
                continue
            if _is_guard(p.info["cmdline"]):
                out.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-mb", type=float, default=900.0,
                    help="kill the newest experiment when available RAM drops "
                         "below this")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--log", default="results/memory_guard.log")
    ap.add_argument("--heartbeat", type=float, default=600.0,
                    help="seconds between liveness lines in the log")
    ap.add_argument("--force", action="store_true",
                    help="start even if another guard is already running")
    args = ap.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # SINGLE INSTANCE, because three of them were running at once and I did
    # not notice for hours. Restarting the guard after a suspected death is
    # the natural reflex, and `pkill -f guard_memory` does not reliably take
    # the venv shim with it, so each "restart" left the old one alive. Three
    # guards is worse than none in the way that matters: they poll
    # independently, so one dip below the floor kills THREE experiments
    # instead of one, and they append to the same log, interleaving the
    # record of what happened.
    others = other_guards()
    if others and not args.force:
        note(f"refusing to start: guard already running as pid(s) "
             f"{', '.join(str(p) for p in others)}. Use --force to override.")
        return 3

    total = psutil.virtual_memory().total / 1e6
    note(f"guard up: floor {args.floor_mb:.0f} MB, total {total:.0f} MB, "
         f"poll {args.interval:.0f}s")

    # A HEARTBEAT, because this guard died once without saying so. It ran
    # for two hours, was killed from outside -- no traceback, no exit line --
    # and the only evidence was its absence from the process table. The log
    # is what a check-in reads, so the log has to distinguish 'quiet because
    # nothing happened' from 'quiet because nothing is running'. A stale
    # last line now means a dead guard.
    low_streak = 0
    last_beat = 0.0
    while True:
        vm = psutil.virtual_memory()
        avail = vm.available / 1e6
        procs = experiment_processes()
        victims = victim_processes()

        if avail < args.floor_mb and victims:
            # Two consecutive readings, because a momentary dip during a JAX
            # compile is normal and killing on it would make the guard the
            # thing that ends every run.
            low_streak += 1
            if low_streak >= 2:
                victim = victims[-1]
                try:
                    rss = victim.memory_info().rss / 1e6
                    cmd = " ".join(victim.cmdline())[:120]
                    victim.kill()
                    note(f"KILLED pid {victim.pid} ({rss:.0f} MB) -- available "
                         f"{avail:.0f} MB below floor. cmd: {cmd}")
                    # WHO ELSE WAS HOLDING MEMORY. The first real kill was
                    # caused by a pytest run in the foreground, not by either
                    # experiment, and the log said only that memory was low --
                    # so the guard killed an hour of work to make room for a
                    # 30-second test suite and left no way to tell. A kill
                    # line that does not name the pressure is a kill line that
                    # gets misread as "the experiments were too big".
                    note("  memory at kill: " + hogs())
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    note(f"could not kill pid {victim.pid}: {e}")
                low_streak = 0
        else:
            if low_streak:
                note(f"recovered: available {avail:.0f} MB")
            low_streak = 0

        now = time.monotonic()
        if now - last_beat >= args.heartbeat:
            note(f"alive: available {avail:.0f} MB, "
                 f"{len(procs)} experiment process(es)")
            last_beat = now

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
