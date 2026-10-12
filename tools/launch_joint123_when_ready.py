"""Linux-only queue for the existing J1/J2/J3 pipeline; never modifies the parent.

Uses only the standard library while waiting. Run with the bright4090 Python.
All model, split, bank and code checks remain in the original run.sh check.
One launcher per output-parent directory is allowed; the advisory lock is kept
through exec into run.sh pipeline, including its training subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # A producer may still be writing. Never load checkpoints here.
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def artifacts_ready(parent):
    """Lightweight completion check, NOT a replacement for strict precheck."""
    summary = read_json(parent / "front" / "summary.json")
    if summary is None:
        return False, "front/summary.json not yet readable"
    if summary.get("smoke") is not False or summary.get("test_accessed") is not False:
        raise ValueError("Parent front must be formal and have no Test access")
    if summary.get("complete") is not True:
        return False, "front training not complete"
    cache = read_json(parent / "train_memory" / "manifest.json")
    if cache is None:
        return False, "Train Memory manifest not yet readable"
    if cache.get("smoke") is not False or cache.get("test_accessed") is not False:
        raise ValueError("Train Memory must be formal and have no Test access")
    if cache.get("complete") is not True:
        return False, "Train Memory extraction not complete"
    if (cache.get("split") != "train" or cache.get("image_count") != 2918
            or not isinstance(cache.get("shards"), list) or len(cache["shards"]) != 2918):
        raise ValueError("Completed cache is not the required full Train2918 cache")
    for relative in ("front/best.pth", "train_memory/bank.pt"):
        path = parent / relative
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"Required nonempty artifact missing: {relative}"
    return True, "front and full Train Memory completion markers present"


def process_identity(pid, proc_root=Path("/proc")):
    """PID + kernel start ticks distinguish a live parent from a reused PID."""
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    if fields[0] in {"Z", "X"}:
        return None
    return pid, fields[19]  # /proc stat field 22: starttime, after state field 3.


def verify_parent_process(pid, parent, proc_root=Path("/proc")):
    identity = process_identity(pid, proc_root)
    if identity is None:
        return None
    try:
        argv = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
        environment = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        if process_identity(pid, proc_root) is None:
            return None
        raise
    if (not any(arg.endswith(b"damageformer_srac_transformer_v1/run.sh") for arg in argv)
            or b"pipeline" not in argv):
        raise ValueError("--parent-pid must identify the original transformer_v1 run.sh pipeline, not a worker")
    env = dict(entry.split(b"=", 1) for entry in environment if b"=" in entry)
    run_root = env.get(b"RUN_ROOT")
    if run_root is None or Path(os.fsdecode(run_root)).resolve() != parent:
        raise ValueError("Parent PID RUN_ROOT does not match --parent-run; refusing to watch another experiment")
    if process_identity(pid, proc_root) != identity:
        raise ValueError("Parent PID changed during inspection; recheck the process")
    return identity


def gpu_processes():
    """Conservative policy: all GPUs reported by nvidia-smi must have no compute jobs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=20,
    )
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            if not value.isdigit():
                raise ValueError(f"Cannot verify GPU-idle status: {value}")
            pids.append(int(value))
    return sorted(set(pids))


def decision(ready, reason, parent_alive, busy_pids):
    if parent_alive:
        return False, f"WAIT parent pipeline is running; artifacts: {reason}"
    if not ready:
        raise ValueError("Parent pipeline exited, but artifacts are incomplete: " + reason
                         + ". Inspect its log; no automatic restart or cache rebuild.")
    if busy_pids:
        return False, "WAIT GPU compute processes: " + ",".join(map(str, busy_pids))
    return True, "READY parent exited, artifacts ready and no GPU compute process detected"


def validate_paths(project, parent, a1, output):
    if not parent.is_dir() or not a1.is_file():
        raise ValueError("Parent directory or original A1 file does not exist")
    if output == parent or parent in output.parents or output in parent.parents:
        raise ValueError("Output must be outside, and must not contain, the parent run")
    if output.exists():
        raise ValueError(f"Output already exists; preserve it and choose a NEW J123_ROOT: {output}")
    script = project / "bda_benchmark/experiments/damageformer_srac_joint123_v1/run.sh"
    if not script.is_file():
        raise ValueError(f"Upload the complete joint123 experiment first: {script}")
    return script


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--parent-run", default=os.environ.get("PARENT_RUN"))
    parser.add_argument("--a1-checkpoint", default=os.environ.get("A1_CKPT"))
    parser.add_argument("--output", default=os.environ.get("J123_ROOT"))
    parser.add_argument("--parent-pid", type=int, required=True,
                        help="PID of the original run.sh pipeline (not train_front/worker PID)")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--timeout-hours", type=float, default=168,
                        help="Maximum waiting time only; does not terminate training (default 7 days)")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run this launcher on the Linux GPU server, not the local Windows mirror")
    if not all((args.parent_run, args.a1_checkpoint, args.output)):
        parser.error("Set PARENT_RUN, A1_CKPT and a NEW J123_ROOT, or pass their arguments")
    if args.parent_pid <= 1 or args.poll_seconds <= 0 or args.timeout_hours <= 0:
        parser.error("PID must exceed 1; polling and timeout must be positive")

    import fcntl

    project, parent, a1, output = [Path(value).resolve() for value in
                                  (args.project, args.parent_run, args.a1_checkpoint, args.output)]
    script = validate_paths(project, parent, a1, output)
    identity = verify_parent_process(args.parent_pid, parent)
    output.parent.mkdir(parents=True, exist_ok=True)  # Do NOT create output itself.
    # Shared lock also prevents two different new output names launching together.
    lock = (output.parent / ".joint123-launch.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValueError("Another launcher/pipeline owns this output-family lock; do not launch twice") from exc
    with lock:
        log(f"PARENT={parent}")
        log(f"OUTPUT={output}; polling={args.poll_seconds}s; timeout={args.timeout_hours}h")
        log("No parent weights/source/results will be modified. GPU check covers ALL GPUs reported by nvidia-smi.")
        if identity is None:
            log("Parent PID is already absent; complete artifacts and strict validation are still required.")
        environment = dict(os.environ, PYTHON=sys.executable, PARENT_RUN=str(parent),
                           A1_CKPT=str(a1), J123_ROOT=str(output), PYTHONUNBUFFERED="1")
        environment.setdefault("SEED", "3407")
        environment.setdefault("DEVICE", "cuda:0")
        environment.setdefault("WORKERS", "4")
        environment.setdefault("AMP", "bf16")
        deadline = time.monotonic() + args.timeout_hours * 3600
        previous = None
        last_print = -float("inf")
        while True:
            if output.exists():
                raise ValueError(f"Output appeared while waiting; refusing to overwrite: {output}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Waiting deadline reached; nothing was trained or deleted by this launcher")
            ready, reason = artifacts_ready(parent)
            alive = identity is not None and process_identity(args.parent_pid) == identity
            busy = gpu_processes() if ready and not alive else []
            launch, status = decision(ready, reason, alive, busy)
            now = time.monotonic()
            if status != previous or now - last_print >= 600:
                log(status)
                previous, last_print = status, now
            if not launch:
                time.sleep(min(args.poll_seconds, max(0, deadline - now)))
                continue
            # Keep existing strict gates. An error is fatal, never a reason to loosen checks.
            for mode in ("preflight", "check"):
                log(f"STRICT {mode}")
                subprocess.run(["bash", str(script), mode], cwd=project, env=environment, check=True)
            validate_paths(project, parent, a1, output)
            if gpu_processes():
                log("WAIT GPU became busy during validation; returning to polling, without launching")
                time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
                continue
            break
        log("START pipeline: metric 2000 (or verified supplied metric) -> probe 20 -> J1/J2/J3 each 5000")
        os.set_inheritable(lock.fileno(), True)
        os.chdir(project)
        # Replace the watcher; lock and exit status survive. No orphaned watcher retry.
        os.execvpe("bash", ["bash", str(script), "pipeline"], environment)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        log(f"STOP: {error}")
        sys.exit(1)
