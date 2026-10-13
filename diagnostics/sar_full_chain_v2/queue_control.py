"""Cancel only an identified, still-waiting SAR chain launcher.

Uses a pidfd to prevent PID reuse races, pauses only that launcher, and checks
its state and descendants again before termination. Never signals Memory,
probe jobs, a process group, or a training child. No SIGKILL fallback exists.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time


WAITING_PHASES = frozenset(("waiting_declared_memory", "waiting_existing_probe"))
FORMATS = frozenset(("SAR_FULL_CHAIN_V1", "SAR_FULL_CHAIN_V2"))


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Invalid SAR queue metadata: " + str(path))
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LinuxProcesses:
    """Standard-library Linux process access; pidfd support is mandatory."""
    def __init__(self, proc_root="/proc"):
        self.proc_root = Path(proc_root)

    def _stat(self, pid):
        try:
            text = (self.proc_root / str(pid) / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            if fields[0] in ("Z", "X"):
                return None
            return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), start_ticks=fields[19])
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, ValueError, IndexError) as error:
            raise RuntimeError("Cannot safely inspect Linux process " + str(pid)) from error

    def identity(self, pid):
        row = self._stat(pid)
        if row is None:
            return None
        try:
            raw = (self.proc_root / str(pid) / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except OSError as error:
            raise RuntimeError("Cannot read queued launcher arguments") from error
        row["argv"] = [value.decode("utf-8", errors="replace") for value in raw.split(b"\0") if value]
        return row

    def descendants(self, pid):
        rows = []
        for folder in self.proc_root.iterdir():
            if folder.name.isdigit():
                row = self._stat(folder.name)
                if row is not None:
                    rows.append(row)
        found, frontier = set(), {int(pid)}
        while frontier:
            next_frontier = {row["pid"] for row in rows
                             if row["ppid"] in frontier and row["pid"] not in found}
            found.update(next_frontier)
            frontier = next_frontier
        return [row for row in rows if row["pid"] in found]

    def open_pidfd(self, pid):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Safe queue cancellation requires Linux pidfd support; no signals sent")
        return os.pidfd_open(pid, 0)

    def send(self, descriptor, signum):
        signal.pidfd_send_signal(descriptor, signum, None, 0)

    def close(self, descriptor):
        os.close(descriptor)


def _metadata(folder):
    launch = _read(folder / "launch.json")
    launcher = _read(folder / "launcher.json")
    state = _read(folder / "chain_state.json")
    if launch.get("format") not in FORMATS or state.get("format") != launch.get("format"):
        raise RuntimeError("Unexpected SAR chain format; queue was not cancelled")
    for record in (launch, launcher):
        value = record.get("root")
        if not isinstance(value, str) or Path(value).resolve() != folder:
            raise RuntimeError("SAR launcher root binding differs")
    pid = state.get("pid")
    if type(pid) is not int or pid <= 1 or launcher.get("pid") != pid:
        raise RuntimeError("SAR queue PID binding is missing or inconsistent")
    tick = state.get("pid_start_ticks")
    if type(tick) not in (str, int) or not re.fullmatch(r"[0-9]+", str(tick)) or int(tick) <= 0:
        raise RuntimeError("SAR queue lacks a trustworthy process start time")
    if "pid_start_ticks" in launcher and str(launcher["pid_start_ticks"]) != str(tick):
        raise RuntimeError("SAR launcher start-time binding differs")
    for key in ("q_root", "soft_root"):
        value = launch.get(key)
        if not isinstance(value, str):
            raise RuntimeError("SAR output binding missing: " + key)
        for record in (state, launcher):
            if key in record and Path(record[key]).resolve() != Path(value).resolve():
                raise RuntimeError("SAR output binding differs: " + key)
    return dict(launch=launch, launcher=launcher, state=state, pid=pid, start_ticks=str(tick))


def _identity_matches(identity, metadata, folder):
    if identity["pid"] != metadata["pid"] or str(identity["start_ticks"]) != metadata["start_ticks"]:
        raise RuntimeError("PID was reused or SAR start-time binding differs; no termination permitted")
    argv = identity.get("argv", [])
    launch = metadata["launch"]
    bindings = launch.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("SAR launch source binding is missing")
    if len(argv) < 5 or argv[1] != "-u" or argv[3] != "--worker-root" or argv.count("--worker-root") != 1:
        raise RuntimeError("Process is not the recorded SAR chain worker")
    if Path(argv[4]).resolve() != folder:
        raise RuntimeError("Process command names a different SAR root")
    script = Path(argv[2]).resolve()
    candidates = {Path(name).resolve(): value for name, value in bindings.items()
                  if isinstance(name, str) and Path(name).name == "chain.py"}
    digest = candidates.get(script)
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not script.is_file() or _sha(script) != digest):
        raise RuntimeError("Process chain.py is not an unchanged bound launcher")
    python = launch.get("python")
    if not isinstance(python, str) or os.path.abspath(argv[0]) != os.path.abspath(python):
        raise RuntimeError("SAR worker interpreter binding differs")


def _own_workflow(root, folder):
    path = Path(root).resolve()
    if not path.is_relative_to(folder):
        return None  # Existing external Q source is never cancelled or altered.
    state_path = path / "workflow_state.json"
    return _read(state_path) if state_path.is_file() else None


def _require_waiting(metadata, folder):
    state, launch = metadata["state"], metadata["launch"]
    if state.get("complete") is not False or state.get("failure"):
        raise RuntimeError("SAR task is not a clean pending queue")
    phase = state.get("phase")
    q = _own_workflow(launch["q_root"], folder)
    soft = _own_workflow(launch["soft_root"], folder)
    if soft is not None and (soft.get("stages") or soft.get("complete") is True):
        raise RuntimeError("SAR+soft stages have started; active workflow was not cancelled")
    if phase in WAITING_PHASES:
        if q is not None and (q.get("stages") or q.get("complete") is True):
            raise RuntimeError("Q stages have started despite a stale waiting phase")
        return
    if (phase == "sar_q" and q is not None and q.get("current_stage") == "waiting_gpu"
            and q.get("stages") == [] and q.get("complete") is False and not q.get("failure")):
        return
    raise RuntimeError("SAR task has started or is not in an approved waiting state; no cancellation")


def cancel_queued(output_parent, root=None, *, _processes=None, _sleep=None, _clock=None):
    """Cancel one pure queue or fail safely. Never stop an active training job.

    Optional underscored dependencies are for independent CPU tests. Production
    calls require Linux pidfds and are bounded to 15 seconds after identification.
    """
    parent = Path(output_parent).expanduser().resolve()
    if root is None:
        latest = parent / "latest.json"
        if not latest.is_file():
            return dict(status="no_active_queue", reason="no_latest_record")
        value = _read(latest).get("root")
        if not isinstance(value, str):
            raise RuntimeError("Invalid latest SAR root; nothing was signalled")
        folder = Path(value).expanduser().resolve()
    else:
        folder = Path(root).expanduser().resolve()
    if folder == parent or not folder.is_relative_to(parent):
        raise RuntimeError("Queued SAR root must be beneath the declared output parent")
    metadata = _metadata(folder)
    processes = _processes if _processes is not None else LinuxProcesses()
    pause = _sleep if _sleep is not None else time.sleep
    clock = _clock if _clock is not None else time.monotonic
    identity = processes.identity(metadata["pid"])
    if identity is None:
        return dict(status="no_active_queue", reason="recorded_launcher_not_alive", root=str(folder))
    _identity_matches(identity, metadata, folder)
    _require_waiting(metadata, folder)
    descriptor = None
    stopped = False
    pending_error = None
    deadline = clock() + 15.0
    try:
        try:
            descriptor = processes.open_pidfd(metadata["pid"])
        except ProcessLookupError:
            return dict(status="no_active_queue", reason="launcher_exited_before_open", root=str(folder))
        identity = processes.identity(metadata["pid"])
        if identity is None:
            return dict(status="no_active_queue", reason="launcher_exited_before_pause", root=str(folder))
        _identity_matches(identity, metadata, folder)
        processes.send(descriptor, signal.SIGSTOP)
        stopped = True
        stop_deadline = min(deadline, clock() + 2.0)
        while True:
            identity = processes.identity(metadata["pid"])
            if identity is None:
                return dict(status="no_active_queue", reason="launcher_exited_during_pause", root=str(folder))
            _identity_matches(identity, metadata, folder)
            if identity.get("state") == "T":
                break
            if clock() >= stop_deadline:
                raise RuntimeError("Could not confirm the exact launcher paused; no termination sent")
            pause(0.05)
        # Parent is stopped, so it cannot start a new stage between this fresh
        # check and SIGTERM. Existing children continue and force a refusal.
        fresh = _metadata(folder)
        _identity_matches(identity, fresh, folder)
        if fresh["pid"] != metadata["pid"] or fresh["start_ticks"] != metadata["start_ticks"]:
            raise RuntimeError("Queue identity metadata changed while pausing")
        _require_waiting(fresh, folder)
        children = processes.descendants(metadata["pid"])
        if children:
            raise RuntimeError("Launcher has active descendants; no termination sent")
        processes.send(descriptor, signal.SIGTERM)
        processes.send(descriptor, signal.SIGCONT)
        stopped = False
        while True:
            identity = processes.identity(metadata["pid"])
            if identity is None or str(identity["start_ticks"]) != metadata["start_ticks"]:
                return dict(status="cancelled", root=str(folder), pid=metadata["pid"],
                            pid_start_ticks=metadata["start_ticks"],
                            signals="SIGSTOP launcher; SIGTERM+SIGCONT launcher",
                            memory_signalled=False, probe_signalled=False, training_children_signalled=False)
            if clock() >= deadline:
                raise RuntimeError("Queued launcher did not exit within 15 seconds; no SIGKILL or new training")
            pause(0.05)
    except ProcessLookupError:
        return dict(status="no_active_queue", reason="launcher_exited_during_cancel", root=str(folder))
    except BaseException as error:
        pending_error = error
        raise
    finally:
        if descriptor is not None:
            try:
                if stopped:
                    try:
                        processes.send(descriptor, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                    except Exception as error:
                        raise RuntimeError("Could not resume the original launcher after refusing cancellation: "
                                           + str(pending_error)) from error
            finally:
                processes.close(descriptor)
