#!/usr/bin/env python3
"""One original-model continuation: checkpoint150 to2500, Val450 every500, no Test."""
from __future__ import annotations

import argparse
from collections import deque
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import evidence
import preflight
import train_resume

FORMAT = "BRIGHT_SEVERITY_ORIGINAL_RESUME_SUPERVISOR_V1"
MAX_SECONDS = 86400
DEFAULT_SECONDS = 43200
HEARTBEAT_SECONDS = 30
MARKER = ".original_resume_latest.json"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def study_root(parent):
    return Path(parent).expanduser().resolve().parent.parent


def gpu_idle():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError("Cannot check GPU occupancy: " + result.stderr.strip())
    pids = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if pids:
        raise RuntimeError("GPU has existing compute jobs; no recovery started. PIDs=" + ",".join(pids))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_runner(destination):
    """Snapshot exactly the invoked zipapp or this recovery's five source files."""
    destination = Path(destination)
    invoked = Path(sys.argv[0]).resolve()
    if invoked.is_file() and zipfile.is_zipfile(invoked):
        with invoked.open("rb") as source, destination.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
    else:
        here = Path(__file__).resolve().parent
        names = ("__main__.py", "launch.py", "evidence.py", "preflight.py", "train_resume.py")
        for name in names:
            if not (here / name).is_file():
                raise FileNotFoundError("Recovery source missing: " + str(here / name))
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(here / name, name)
    return digest(destination)


def validate_lease(fd, root):
    if fd is None:
        raise ValueError("Use start; internal command requires an inherited lease")
    actual = os.fstat(fd)
    expected = (Path(root) / ".severity_night.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Wrong inherited lease file")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def validate_config(root):
    root = Path(root).resolve()
    config = read_json(root / "config.json")
    if config.get("format") != FORMAT or Path(config["output"]).resolve() != root:
        raise ValueError("Recovery configuration/output mismatch")
    if study_root(config["parent"]) != root.parent or not root.name.startswith("original_resume_"):
        raise ValueError("Recovery output is outside the source experiment family")
    if not 1 <= int(config["timeout_seconds"]) <= MAX_SECONDS:
        raise ValueError("Invalid recovery time limit")
    if digest(root / "runner.pyz") != config["runner_sha256"]:
        raise ValueError("Recovery runner changed after start")
    return config


def select_output(args):
    """An explicitly selected or self-contained run wins over the latest marker."""
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
    else:
        invoked = Path(sys.argv[0]).resolve()
        if invoked.name == "runner.pyz" and (invoked.parent / "config.json").is_file():
            output = invoked.parent
        else:
            root = study_root(args.parent)
            output = Path(read_json(root / MARKER)["output"]).resolve()
            if output.parent != root or not output.name.startswith("original_resume_"):
                raise ValueError("Latest marker points outside this recovery family")
    if not output.is_dir():
        raise FileNotFoundError("Recovery directory missing: " + str(output))
    return output


def check(args):
    preflight.check_inputs(Path(args.package).expanduser().resolve(), Path(args.parent).expanduser().resolve(), Path(args.baseline).expanduser().resolve(), Path(args.plain).expanduser().resolve())
    print("RECOVERY CHECK PASSED: checkpoint150 plus observed/plain windows verified; no GPU job or optimizer update.", flush=True)


def failed_summary(root, error):
    """Keep worker evidence; mark an incomplete or killed recovery as failed."""
    path = Path(root) / "summary.json"
    try:
        summary = read_json(path) if path.is_file() else {}
    except (OSError, ValueError):
        summary = {}
    summary.update(phase="failed", complete=False, recovery_complete=False, validation_complete=False,
                   full_test_evidence=False, full_validation_evidence=False,
                   experiment_complete=False, original_experiment_complete=False,
                   supervisor_error=error)
    if not summary.get("error"):
        summary["error"] = error
    write_json(path, summary)


def disk_headroom(root, parent):
    # 47 checkpoint files (200..2500); reserve twice current payload size plus
    # metadata headroom. This is a launch check, not a guaranteed upper bound.
    size = (Path(parent) / "checkpoints/checkpoint_000150.pt").stat().st_size
    required = 47 * size * 2 + 512 * 2**20
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeError(f"Insufficient disk headroom: free={free}, required={required} bytes")
    return dict(free_bytes=free, required_headroom_bytes=required,
                expected_new_checkpoints=47, guaranteed_upper_bound=False)


def validate_success(result):
    expected = dict(phase="complete", complete=True, training_complete=True,
                    validation_complete=True, full_validation_evidence=True,
                    full_test_evidence=False, test_accessed=False,
                    last_complete_step=2500, new_complete_updates=2350,
                    historical_failure_resolved=False)
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError("Recovery completion evidence differs: " + key)
    if result.get("error") is not None:
        raise ValueError("Recovery completion has an error")


def start(args):
    if args.output is not None:
        raise ValueError("start always creates a new original_resume_* directory; do not pass --output")
    if not 1 <= args.timeout_seconds <= MAX_SECONDS:
        raise ValueError("timeout-seconds must be 1..86400")
    parent = Path(args.parent).expanduser().resolve()
    package = Path(args.package).expanduser().resolve()
    preflight.check_inputs(package, parent, Path(args.baseline).expanduser().resolve(), Path(args.plain).expanduser().resolve())
    root = study_root(parent)
    if not root.is_dir():
        raise FileNotFoundError("Study root does not exist: " + str(root))
    with (root / ".severity_night.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_idle()
        disk = disk_headroom(root, parent)
        output = Path(tempfile.mkdtemp(prefix="original_resume_", dir=root))
        try:
            config = dict(format=FORMAT, parent=str(parent), package=str(package), baseline=str(Path(args.baseline).expanduser().resolve()), plain=str(Path(args.plain).expanduser().resolve()),
                output=str(output), timeout_seconds=args.timeout_seconds, disk_headroom=disk,
                runner_sha256=snapshot_runner(output / "runner.pyz"), started_utc=utc_now())
            write_json(output / "config.json", config)
            write_json(output / "summary.json", dict(phase="recovery_starting", complete=False,
                recovery_complete=False, full_validation_evidence=False, full_test_evidence=False,
                training_complete=False, validation_complete=False, study_complete=False,
                experiment_complete=False, original_experiment_complete=False,
                test_accessed=False, output=str(output)))
            write_json(output / "supervisor.json", dict(format=FORMAT, phase="starting", complete=False,
                recovery_complete=False, output=str(output), retry_count=0, started_utc=utc_now()))
            env = os.environ.copy()
            env.update(CUDA_LAUNCH_BLOCKING="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
            # Both supervisor and worker must use O_APPEND. A supervisor with
            # an independent non-append offset could overwrite child log lines.
            with (output / "train.log").open("xb"):
                pass
            with (output / "train.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-B", "-u", str(output / "runner.pyz"),
                    "supervise", "--output", str(output), "--lease-fd", str(lease.fileno())],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True, pass_fds=(lease.fileno(),), env=env)
            write_json(root / MARKER, dict(output=str(output), pid=process.pid))
        except BaseException as exc:
            failed_summary(output, type(exc).__name__ + ": " + str(exc))
            raise
    print("已后台启动 original 正式恢复：150→2500；每500步验证Val450，终点固定Train诊断。")
    print(f"PID={process.pid}\nOUTPUT={output}\n每次最多{args.timeout_seconds}秒，失败或超时即停止，不自动重试。")
    print(f"日志：{sys.executable} -B {output / 'runner.pyz'} logs")
    print(f"结果：{sys.executable} -B {output / 'runner.pyz'} status")
    return 0


def stop_owned(process):
    """The worker is always launched as its own process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def wait_worker(process, limit, status, root):
    started = time.monotonic()
    deadline = started + limit
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Recovery exceeded time limit; no automatic retry")
        try:
            return process.wait(timeout=min(HEARTBEAT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            status.update(child_pid=process.pid, elapsed_seconds=round(elapsed, 1), updated_utc=utc_now())
            write_json(Path(root) / "supervisor.json", status)
            print("ORIGINAL RESUME HEARTBEAT " + json.dumps(dict(pid=process.pid,
                elapsed_seconds=round(elapsed, 1), limit_seconds=limit)), flush=True)


def supervise(args):
    root = select_output(args)
    config = validate_config(root)
    validate_lease(args.lease_fd, root.parent)
    status = dict(format=FORMAT, phase="preflight", complete=False, recovery_complete=False,
        experiment_complete=False, original_experiment_complete=False, full_test_evidence=False,
        full_validation_evidence=False, test_accessed=False, pid=os.getpid(), retry_count=0,
        output=str(root), started_utc=utc_now())
    child = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError("Supervisor received signal " + str(signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupted)
    try:
        write_json(root / "supervisor.json", status)
        gpu_idle()
        command = [sys.executable, "-B", "-u", str(root / "runner.pyz"), "worker",
                   "--output", str(root), "--lease-fd", str(args.lease_fd)]
        env = os.environ.copy()
        env.update(CUDA_LAUNCH_BLOCKING="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        with (root / "train.log").open("ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                pass_fds=(args.lease_fd,), env=env)
        status.update(phase="recovery_running", child_pid=child.pid, command=command)
        write_json(root / "supervisor.json", status)
        code = wait_worker(child, int(config["timeout_seconds"]), status, root)
        status["returncode"] = code
        result = read_json(root / "summary.json")
        if code != 0:
            raise RuntimeError("Recovery worker failed/incomplete (exit=" + str(code) + "): " +
                               str(result.get("error") or result.get("phase")))
        validate_success(result)
        status.update(phase="complete", complete=True, recovery_complete=True,
                      training_complete=True, validation_complete=True,
                      full_validation_evidence=True, experiment_complete=True,
                      study_complete=False, original_experiment_complete=False, error=None)
        return 0
    except BaseException as exc:
        if child is not None:
            stop_owned(child)
        error = type(exc).__name__ + ": " + str(exc)
        status.update(phase="failed", error=error, recovery_complete=False)
        failed_summary(root, error)
        (root / "supervisor_traceback.txt").write_text(traceback.format_exc())
        print("ORIGINAL RESUME SUPERVISOR FAILED: " + error, flush=True)
        return 2
    finally:
        status.update(job_finished=True, ended_utc=utc_now())
        write_json(root / "supervisor.json", status)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def worker(args):
    root = select_output(args)
    config = validate_config(root)
    validate_lease(args.lease_fd, root.parent)
    args.parent, args.package, args.output = Path(config["parent"]), Path(config["package"]), root
    args.baseline = Path(config["baseline"])
    args.observed = args.baseline
    args.plain = Path(config["plain"])
    train_resume.run_worker(args)
    return 0


def print_status(root):
    path = root / "summary.json"
    if not path.is_file():
        print("尚未生成训练摘要")
        return
    summary = read_json(path)
    fields = ("format", "phase", "complete", "training_complete", "validation_complete",
              "fixed_train_complete", "full_validation_evidence", "test_accessed",
              "restored_step", "last_complete_step", "last_journaled_step", "endpoint_step",
              "new_complete_updates", "last_checkpoint", "current_image", "failure_phase",
              "error", "failed_step_optimizer_execution_unknown", "historical_failure_resolved",
              "original_failed_runs_unchanged", "output", "elapsed_seconds")
    print("训练状态：")
    print(json.dumps({key: summary[key] for key in fields if key in summary},
                     indent=2, ensure_ascii=False))
    for name, report in summary.get("evaluations", {}).items():
        row = dict(evaluation=name, step=report.get("step"), image_count=report.get("image_count"),
                   complete=report.get("complete", False),
                   final_integrity_pending=report.get("final_integrity_pending"),
                   metrics={key: {metric: value.get(metric) for metric in
                       ("miou4_pct", "foreground_miou3_pct", "iou_pct")}
                       for key, value in report.get("metrics", {}).items()})
        print("EVALUATION " + json.dumps(row, ensure_ascii=False))
    path = root / "supervisor.json"
    print("监督状态：")
    print(path.read_text() if path.is_file() else "尚未生成")


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("command", choices=("start", "check", "logs", "status", "worker", "supervise"))
    command.add_argument("--parent", type=Path, default=evidence.PARENT)
    command.add_argument("--package", type=Path, default=evidence.PACKAGE)
    command.add_argument("--baseline", type=Path, default=evidence.BASELINE)
    command.add_argument("--plain", type=Path, default=preflight.PLAIN)
    command.add_argument("--output", type=Path)
    command.add_argument("--timeout-seconds", type=int, default=DEFAULT_SECONDS)
    command.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "check":
        check(args)
        return 0
    if args.command == "start":
        return start(args)
    if args.command == "supervise":
        return supervise(args)
    if args.command == "worker":
        return worker(args)
    root = select_output(args)
    if args.command == "logs":
        print("训练日志：" + str(root / "train.log"))
        with (root / "train.log").open(errors="replace") as stream:
            print("".join(deque(stream, maxlen=70)), end="")
    else:
        print_status(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
