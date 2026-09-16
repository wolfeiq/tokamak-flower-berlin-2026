"""The memory guard's interlocks.

Written after three guards were found running at once. Each had been started
because the previous one was believed dead; `pkill -f guard_memory` had not
taken the Windows venv shim with it. Three independent guards is strictly
worse than one: a single dip below the floor kills three experiments instead
of one, and all three append to the same log.

The bug was in how the guard was OPERATED rather than in the guard, which is
exactly the kind that comes back -- so the fix is an interlock in the guard
and this is the test that keeps it.
"""
import sys
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import guard_memory as gm  # noqa: E402


class FakeProc:
    def __init__(self, pid, name="python.exe", cmdline=(), create_time=0.0):
        self.pid = pid
        self.info = {"pid": pid, "name": name, "cmdline": list(cmdline),
                     "create_time": create_time}

    def cmdline(self):
        return self.info["cmdline"]


def _patch_table(monkeypatch, procs, own_chain):
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: procs)
    monkeypatch.setattr(gm, "_own_chain", lambda: set(own_chain))


GUARD = ["python", "scripts/guard_memory.py", "--floor-mb", "900"]


def test_own_shim_is_not_mistaken_for_another_guard(monkeypatch):
    """The launcher and the interpreter it execs share a command line.

    If the interlock counted those, no guard could ever start.
    """
    procs = [FakeProc(100, cmdline=GUARD), FakeProc(101, cmdline=GUARD)]
    _patch_table(monkeypatch, procs, own_chain={100, 101})
    assert gm.other_guards() == []


def test_a_guard_from_an_earlier_launch_is_seen(monkeypatch):
    procs = [FakeProc(100, cmdline=GUARD), FakeProc(101, cmdline=GUARD),
             FakeProc(7, cmdline=GUARD)]
    _patch_table(monkeypatch, procs, own_chain={100, 101})
    assert gm.other_guards() == [7]


def test_experiments_are_not_counted_as_guards(monkeypatch):
    procs = [FakeProc(9, cmdline=["python", "scripts/exp_coldstart.py"])]
    _patch_table(monkeypatch, procs, own_chain={100})
    assert gm.other_guards() == []


def test_non_python_processes_are_ignored(monkeypatch):
    procs = [FakeProc(9, name="bash.exe", cmdline=["bash", "guard_memory"])]
    _patch_table(monkeypatch, procs, own_chain={100})
    assert gm.other_guards() == []


def test_dead_process_during_scan_does_not_crash(monkeypatch):
    class Vanishing(FakeProc):
        @property
        def info(self):
            raise psutil.NoSuchProcess(self.pid)

        @info.setter
        def info(self, v):
            pass

    procs = [Vanishing(9), FakeProc(7, cmdline=GUARD)]
    _patch_table(monkeypatch, procs, own_chain={100})
    assert gm.other_guards() == [7]


def test_the_guard_still_kills_the_newest_experiment(monkeypatch):
    """The property the whole thing exists for, restated here.

    Ordered by create_time, so the last element is the youngest -- the one
    that has produced the least and is therefore cheapest to lose.
    """
    procs = [
        FakeProc(1, cmdline=["python", "scripts/exp_coldstart.py"],
                 create_time=10.0),
        FakeProc(2, cmdline=["python", "scripts/exp_federation.py"],
                 create_time=50.0),
        FakeProc(3, cmdline=["python", "-c", "print(1)"], create_time=99.0),
    ]
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: procs)
    monkeypatch.setattr(gm.os, "getpid", lambda: 999)
    found = gm.experiment_processes()
    assert [p.pid for p in found] == [1, 2]
    assert found[-1].pid == 2


@pytest.mark.parametrize("flag", ["--force"])
def test_force_is_available_as_an_escape_hatch(flag):
    """An interlock with no override is a way to be locked out at 3am."""
    src = (Path(gm.__file__)).read_text(encoding="utf-8")
    assert flag in src
    assert "args.force" in src


def test_a_bare_mention_of_the_script_is_not_a_running_guard():
    """`python -c "import guard_memory"` contains the name and is not one.

    Found by writing a one-liner to inspect the guard and watching the guard
    count that one-liner as a second guard.
    """
    assert not gm._is_guard(["python", "-c", "import guard_memory; print(1)"])
    assert not gm._is_guard(["python", "-m", "grep", "guard_memory"])
    assert gm._is_guard(["python", "scripts/guard_memory.py", "--floor-mb"])
    assert gm._is_guard(["python", r"C:\x\scripts\guard_memory.py"])
    assert gm._is_guard(["python", "guard_memory.py"])


def test_a_test_run_is_killed_before_an_experiment(monkeypatch):
    """The cost ordering the guard is really trying to express.

    A pytest process costs thirty seconds to re-run; a six-seed experiment
    costs an hour. When both are on the machine and memory runs out, the
    suite is the one to lose -- and because it is launched by hand while runs
    are already going, it is also the newest.
    """
    procs = [
        FakeProc(1, cmdline=["python", "scripts/exp_coldstart.py"],
                 create_time=10.0),
        FakeProc(2, cmdline=["python", "-m", "pytest", "-q"],
                 create_time=90.0),
    ]
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: procs)
    monkeypatch.setattr(gm.os, "getpid", lambda: 999)
    assert [p.pid for p in gm.victim_processes()] == [1, 2]
    # ... and the heartbeat count is unchanged: a test run is not an
    # experiment, and reporting it as one would misstate what is running.
    assert [p.pid for p in gm.experiment_processes()] == [1]


def test_hogs_names_the_pressure_not_just_the_victim(monkeypatch):
    """The kill line has to say WHAT filled the memory.

    The first real kill blamed nothing; the cause was a foreground test run
    and the log gave no way to see that.
    """
    class Sized(FakeProc):
        def __init__(self, pid, mb, cmdline):
            super().__init__(pid, cmdline=cmdline)
            self._mb = mb

        def memory_info(self):
            class MI:
                rss = self._mb * 10 ** 6
            return MI()

    procs = [
        Sized(1, 600, ["python", "scripts/exp_coldstart.py"]),
        Sized(2, 950, ["python", "-m", "pytest", "-q"]),
        Sized(3, 20, ["python", "scripts/guard_memory.py"]),
    ]
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: procs)
    line = gm.hogs()
    assert line.startswith("pytest pid 2 950 MB")
    assert "exp pid 1 600 MB" in line
    assert "guard pid 3 20 MB" in line


def test_hogs_survives_an_empty_process_table(monkeypatch):
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: [])
    assert gm.hogs() == "no python processes"


def test_a_scratchpad_script_is_expendable():
    """One-off analysis scripts cost as much memory as an experiment.

    They are invisible to EXPERIMENT_MARKERS, so before this the guard would
    kill a six-seed run to make room for a twenty-line probe.
    """
    win = "python C:\\Temp\\claude\\sess\\scratchpad\\probe.py"
    nix = "python /c/Temp/claude/sess/scratchpad/probe.py"
    assert gm._is_pytest(win)
    assert gm._is_pytest(nix)
    assert not gm._is_pytest("python scripts/exp_coldstart.py --seeds 6")
    assert not gm._is_pytest("python scripts/guard_memory.py")


def test_an_experiment_outranks_a_scratchpad_script_of_the_same_age(monkeypatch):
    """Ordering is by age, and the probe is always the newer thing."""
    procs = [
        FakeProc(1, cmdline=["python", "scripts/exp_coldstart.py"],
                 create_time=10.0),
        FakeProc(2, cmdline=["python", "/tmp/scratchpad/probe.py"],
                 create_time=11.0),
    ]
    monkeypatch.setattr(gm.psutil, "process_iter", lambda attrs=None: procs)
    monkeypatch.setattr(gm.os, "getpid", lambda: 999)
    assert gm.victim_processes()[-1].pid == 2
