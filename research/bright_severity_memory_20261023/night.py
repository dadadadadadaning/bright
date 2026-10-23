#!/usr/bin/env python3
"""Serial, bounded fresh experiments. Never resumes or changes an older run."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1"
VARIANTS = ("original", "calibration", "severity_memory", "shuffled_memory")


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def gpu_idle():
    """Refuse any visible GPU compute job; never terminate somebody else's job."""
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError("Cannot check GPU occupancy: " + p.stderr.strip())
    jobs = [s.strip() for s in p.stdout.splitlines() if s.strip()]
    if jobs:
        raise RuntimeError("GPU has existing compute jobs; finish them first. PIDs=" + ",".join(jobs))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    # Source snapshots, not dependencies or massive dataset files.
    return {p.name: file_hash(p) for p in sorted(HERE.glob("*.py"))}


def commands(config, root):
    common = ["--source-run", config["source_run"], "--addon-dir", config["addon_dir"],
              "--seed", str(config["seed"]), "--steps", str(config["steps"])]
    for phase in ("smoke", "train"):
        for variant in config["variants"]:
            name = phase + "_" + variant
            yield name, [sys.executable, "-B", "-u", str(HERE / "experiment.py"), phase,
                         *common, "--variant", variant, "--output", str(root / name)]


def start(args):
    import experiment
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate variants")
    config = dict(seed=args.seed, steps=args.steps, variants=args.variants,
                  source_run=args.source_run, addon_dir=args.addon_dir,
                  stage_timeout=args.stage_timeout, source_files=identity())
    if args.stage_timeout < 60 or args.stage_timeout > 24 * 3600:
        raise ValueError("stage-timeout must be 60..86400 seconds")
    # Full source preflight before detaching; no torch or GPU allocations.
    for variant in args.variants:
        opts = experiment.parser().parse_args(["plan", "--variant", variant,
            "--source-run", args.source_run, "--addon-dir", args.addon_dir,
            "--seed", str(args.seed), "--steps", str(args.steps)])
        experiment.prepare_plan(opts)
    gpu_idle()
    parent = Path(args.root).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".severity_night.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(tempfile.mkdtemp(prefix=f"night_{args.seed}_", dir=parent))
        write_json(root / "config.json", config)
        write_json(root / "night_summary.json", dict(phase="starting", complete=False,
                    training_complete=False, validation_complete=False, comparison_complete=False,
                    test_accessed=False, output=str(root), started_utc=datetime.datetime.now(
                    datetime.timezone.utc).isoformat(), stages={}))
        env = os.environ.copy()
        env.update(CUDA_LAUNCH_BLOCKING="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        with (root / "night.log").open("xb") as log:
            proc = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "night.py"),
                "worker", "--run", str(root), "--lease-fd", str(lease.fileno())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                pass_fds=(lease.fileno(),), start_new_session=True, close_fds=True)
        write_json(parent / "latest.json", dict(output=str(root), pid=proc.pid))
    print(f"已在后台排队：所有 smoke 通过后，依次训练 {', '.join(args.variants)}。")
    print("每个分支从头训练到固定终点，任何阶段失败即停止；启动成功不代表训练完成。")
    print(f"PID={proc.pid}\nOUTPUT={root}\nLOG={root / 'night.log'}")
    print(f"查看日志：tail -n 60 -f '{root / 'night.log'}'")
    print(f"查看结果：{sys.executable} -B '{HERE / 'night.py'}' status --run '{root}'")


def stop_owned(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def worker(args):
    root = Path(args.run).resolve()
    config = json.loads((root / "config.json").read_text())
    status = json.loads((root / "night_summary.json").read_text())
    status.update(pid=os.getpid(), phase="checking", error=None)
    # Inherit the actual startup lease; direct manual worker invocation is invalid.
    if args.lease_fd is None:
        raise ValueError("Use start; worker requires the inherited lease")
    lease_stat = os.fstat(args.lease_fd)
    expected = (root.parent / ".severity_night.lock").stat()
    if (lease_stat.st_dev, lease_stat.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Wrong inherited lease")
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    try:
        core_init = None
        for name, command in commands(config, root):
            status.update(phase=name, current_command=command)
            write_json(root / "night_summary.json", status)
            if identity() != config["source_files"]:
                raise ValueError("Delivery source changed after start")
            gpu_idle()
            print("NIGHT STAGE " + name, flush=True)
            started = time.monotonic()
            with (root / (name + ".log")).open("xb") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = int(time.monotonic() - started)
                        print("NIGHT HEARTBEAT " + json.dumps(dict(stage=name, pid=child.pid,
                              elapsed_seconds=elapsed, log=str(root / (name + ".log")))), flush=True)
                        status.update(child_pid=child.pid, elapsed_stage_seconds=elapsed)
                        write_json(root / "night_summary.json", status)
                        if elapsed >= config["stage_timeout"]:
                            stop_owned(child)
                            raise TimeoutError(name + " exceeded the per-stage limit; no automatic retry")
            summary_path = root / name / "summary.json"
            result = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            status["stages"][name] = dict(returncode=code, summary=str(summary_path),
                                         phase=result.get("phase"), error=result.get("error"))
            write_json(root / "night_summary.json", status)
            if code != 0 or result.get("complete") is not True:
                raise RuntimeError(name + " failed; read " + str(root / (name + ".log")))
            if result.get("test_accessed") is not False:
                raise ValueError("Unexpected Test access")
            if name.startswith("train_") and (result.get("last_complete_step") != config["steps"]
                    or result.get("full_validation_evidence") is not True):
                raise ValueError("Training endpoint or complete Val450 evidence missing")
            manifest = json.loads((root / name / "manifest.json").read_text())
            current = manifest["initial_core_head_sha256"]
            if core_init is not None and current != core_init:
                raise ValueError("Fresh shared head initialization differs across variants")
            core_init = current
            print("NIGHT STAGE COMPLETE " + name, flush=True)
        # Generate matched endpoint comparisons only after all requested runs pass.
        status.update(training_complete=True, validation_complete=True, comparison_complete=False)
        from compare_runs import compare
        pairs = [("original", "calibration"), ("calibration", "severity_memory"),
                 ("original", "severity_memory"), ("shuffled_memory", "severity_memory")]
        for before, after in pairs:
            if before not in config["variants"] or after not in config["variants"]:
                continue
            status.update(phase="compare_" + before + "_to_" + after)
            write_json(root / "night_summary.json", status)
            suffix = f"val_step_{config['steps']:06d}/result.json"
            result, _ = compare(root / ("train_" + before) / suffix,
                                root / ("train_" + after) / suffix)
            write_json(root / ("comparison_" + before + "_to_" + after + ".json"), result)
            print("NIGHT COMPARISON " + before + " -> " + after + " " +
                  json.dumps(result["differences"], ensure_ascii=False), flush=True)
        status.update(phase="complete", complete=True, comparison_complete=True, error=None)
    except BaseException as exc:
        if child is not None:
            stop_owned(child)
        status.update(failed_stage=status.get("phase"), phase="failed", complete=False,
                      error=type(exc).__name__ + ": " + str(exc))
        (root / "night_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json(root / "night_summary.json", status)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("start", "worker", "status", "logs"))
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--run", type=Path)
    p.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--steps", type=int, choices=(2500,), default=2500,
                   help="This queue and its comparisons use the fixed2500 protocol")
    p.add_argument("--stage-timeout", type=int, default=10800)
    p.add_argument("--source-run", default="/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_EXTENDED_V2/run_20261019_053508")
    p.add_argument("--addon-dir", default="/root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1")
    return p


def main():
    args = parser().parse_args()
    if args.command == "start":
        return start(args)
    if args.run is None:
        args.run = Path(json.loads((Path(args.root) / "latest.json").read_text())["output"])
    if args.command == "worker":
        return worker(args)
    if args.command == "status":
        print((args.run / "night_summary.json").read_text())
    else:
        from collections import deque
        with (args.run / "night.log").open() as log:
            print("".join(deque(log, maxlen=60)), end="")
        status = json.loads((args.run / "night_summary.json").read_text())
        stage = status.get("failed_stage") or status.get("phase")
        path = args.run / (str(stage) + ".log")
        if path.is_file():
            print("\nCURRENT STAGE LOG: " + str(path))
            with path.open() as log:
                print("".join(deque(log, maxlen=30)), end="")


if __name__ == "__main__":
    main()
