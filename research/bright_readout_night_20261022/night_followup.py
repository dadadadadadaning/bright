#!/usr/bin/env python3
"""Wait for the current pair, then run bounded, sequential BRIGHT follow-ups.

Standard-library controller only. It never imports torch, resumes training,
changes experiment code, accesses Test, or kills a process it did not start.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1"
DEFAULT_PACKAGE = "/root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022"
FORMAT = "BRIGHT_MEMORY_READOUT_DESIGN_V1"
IDENTITY = {
    "experiment.py": "4bbbbfd171b1b39b23e38b7ea7c3c07bcbabdf14a6dca133e92c2448340e71f9",
    "model_adapter.py": "9a81006d0191a3379145286d74a74f7d236d230534a980b0671ff3da4b1e8ba4",
    "supervision.py": "a90c36512fbe84d70510c7c594c9b7d401554a9943209314f58e36810d5be35e",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stop_signal(signum, frame):
    # Raising enters run_child's owned-process cleanup before the controller exits.
    raise KeyboardInterrupt("Controller received signal " + str(signum))


def verify_package(package):
    for name, expected in IDENTITY.items():
        require(sha(package / name) == expected, "Unexpected experiment version: " + name)
    hashes = read(package / "SHA256SUMS.json")
    for name, expected in hashes.items():
        candidate = (package / name).resolve()
        require(package in candidate.parents, "Unsafe package manifest path")
        require(sha(candidate) == expected, "Package manifest mismatch: " + name)
    for name in ("compare_runs.py", "analyze_results.py"):
        require(name in hashes, "Missing verified helper: " + name)


def inspect_run(folder, variant, weight=0.0, smoke=False, reference_plan=None):
    """Return None while unfinished; reject failure/mismatch; validate completion."""
    path = folder / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = read(path)
    except json.JSONDecodeError:
        return None  # A live writer can be between writes; bounded by caller.
    require(summary.get("format") == FORMAT, "Unexpected summary format: " + str(folder))
    require(summary.get("variant") == variant and summary.get("seed") == 3407,
            "Unexpected variant/seed: " + str(folder))
    require(summary.get("smoke") is smoke and summary.get("test_accessed") is False,
            "Unexpected smoke/Test scope: " + str(folder))
    require(summary.get("phase") != "failed" and not summary.get("error"),
            "Existing experiment failed: " + str(folder) + ": " + str(summary.get("error")))
    if summary.get("complete") is not True:
        return None
    endpoint = 3 if smoke else 2500
    require(summary.get("phase") == ("complete_smoke" if smoke else "complete") and
            summary.get("training_complete") is True and
            summary.get("last_complete_step") == endpoint and summary.get("endpoint_step") == endpoint,
            "Incomplete endpoint despite complete flag: " + str(folder))
    plan = read(folder / "plan.json")
    require(plan.get("variant") == variant and plan.get("effective_mode") == variant and
            plan.get("seed") == 3407 and plan.get("endpoint_step") == 2500 and
            plan.get("read_weight") == weight and plan.get("smoke_steps") == 3 and
            plan.get("test_accessed") is False and plan.get("extension_files") == IDENTITY,
            "Run plan differs from requested experiment: " + str(folder))
    if reference_plan is not None:
        for key in ("source", "addon_dir", "addon_sha256", "extension_files", "post500_lr_multiplier",
                    "scheduler_horizon", "warmup_steps", "maximum_lr", "weight_decay", "grad_clip",
                    "optimizer", "endpoint_step", "fixed_train_count", "support_arm"):
            require(key in plan and plan[key] == reference_plan.get(key), "Run plan mismatch: " + key)
    if smoke:
        require(summary.get("full_validation_evidence") is False, "Smoke incorrectly claims full validation")
        return plan
    report_path = folder / "val_step_002500" / "result.json"
    report = read(report_path)
    require(summary.get("full_validation_evidence") is True and
            report.get("format") == FORMAT + "_EVALUATION" and
            report.get("complete") is True and report.get("full_validation_evidence") is True and
            report.get("final_integrity_pending") is False and report.get("smoke") is False and
            report.get("test_accessed") is False and report.get("journal_verified") is True and
            report.get("step") == 2500 and report.get("seed") == 3407 and
            report.get("variant") == variant and report.get("split") == "val" and
            report.get("image_count") == 450,
            "Missing complete endpoint Val450: " + str(folder))
    names = report.get("requested_names", [])
    require(len(names) == len(set(names)) == 450, "Invalid Val450 name list")
    require(read(folder / "manifest.json")["plan"] == plan, "Manifest/plan mismatch")
    require(report_path.with_suffix(".json.sha256").read_text().split()[0] == sha(report_path),
            "Evaluation checksum mismatch")
    return plan


def lock_available(path):
    with path.open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lease, fcntl.LOCK_UN)
        return True


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=15)
    require(result.returncode == 0, "Cannot check GPU occupancy: " + result.stderr.strip())
    lines = [s.strip() for s in result.stdout.splitlines() if s.strip()]
    require(all(s.isdecimal() for s in lines), "Unrecognized GPU process output")
    return sorted(set(map(int, lines)))


def command(package, folder, variant, weight, mode, plan):
    return [sys.executable, "-B", "-u", str(package / "experiment.py"), mode,
            "--variant", variant, "--read-weight", str(weight), "--seed", "3407", "--steps", "2500",
            "--smoke-steps", "3", "--source-run", plan["source"]["source_run"],
            "--addon-dir", plan["addon_dir"], "--post500-lr", str(plan["post500_lr_multiplier"]),
            "--fixed-train", str(plan["fixed_train_count"]), "--output", str(folder)]


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.package = args.package.resolve()
        self.started = time.monotonic()
        self.deadline = self.started + args.max_hours * 3600
        self.output = Path(tempfile.mkdtemp(prefix="night_followup_", dir=self.root))
        self.status = dict(format="BRIGHT_READOUT_NIGHT_FOLLOWUP_V1", phase="starting", complete=False,
                           pid=os.getpid(), output=str(self.output), test_accessed=False, stages=[],
                           include_read_aux=args.include_read_aux, current_stage=None,
                           controller_sha256=sha(__file__), package=str(self.package),
                           wait_hours=args.wait_hours, max_hours=args.max_hours)
        self.save()
        print("NIGHT OUTPUT=" + str(self.output), flush=True)

    def save(self):
        self.status["elapsed_seconds"] = round(time.monotonic() - self.started, 1)
        temporary = self.output / "status.json.tmp"
        temporary.write_text(json.dumps(self.status, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.output / "status.json")

    def check_time(self, deadline=None):
        if time.monotonic() >= min(self.deadline, deadline or self.deadline):
            raise TimeoutError("Night queue time limit reached; existing experiments remain untouched")

    def heartbeat(self, **fields):
        self.save()
        print("NIGHT HEARTBEAT " + json.dumps(dict(stage=self.status["current_stage"], **fields)), flush=True)

    def wait_baseline(self, folder, variant, reference=None):
        self.status.update(phase="waiting", current_stage="wait_" + variant)
        end = time.monotonic() + self.args.wait_hours * 3600
        while True:
            self.check_time(end)
            plan = inspect_run(folder, variant, reference_plan=reference)
            if plan is not None:
                self.status["stages"].append(dict(stage="wait_" + variant, state="complete", output=str(folder)))
                self.save()
                return plan
            self.heartbeat(waiting_for=str(folder))
            time.sleep(min(30, max(0, min(end, self.deadline) - time.monotonic())))

    def wait_gpu(self):
        end = time.monotonic() + self.args.wait_hours * 3600
        while True:
            self.check_time(end)
            unlocked = lock_available(self.root / ".readout_active.lock")
            pids = gpu_pids()
            if unlocked and not pids:
                return
            self.heartbeat(waiting_for_gpu=True, active_lease=not unlocked, gpu_pids=pids)
            time.sleep(min(30, max(0, min(end, self.deadline) - time.monotonic())))

    def run_child(self, label, argv, gpu=False):
        self.check_time()
        self.status.update(phase="running", current_stage=label)
        self.save()
        if gpu:
            self.wait_gpu()
        self.check_time()
        logfile = self.output / (label + ".log")
        entry = dict(stage=label, state="running", log=str(logfile), command=argv)
        self.status["stages"].append(entry)
        print("NIGHT START " + label + " LOG=" + str(logfile), flush=True)
        with logfile.open("x") as stream:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                entry["pid"] = child.pid
                self.save()
                while True:
                    self.check_time()
                    try:
                        code = child.wait(timeout=min(30, max(.01, self.deadline-time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        self.heartbeat(child_pid=child.pid, log=str(logfile))
            except BaseException:
                # This process group was created above; never signal existing GPU jobs.
                if child.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                entry.update(state="interrupted", returncode=child.returncode)
                self.save()
                raise
        entry.update(state="complete" if code == 0 else "failed", returncode=code)
        self.save()
        require(code == 0, label + " failed; inspect " + str(logfile))

    def experiment(self, variant, weight, stem, plan):
        for mode in ("smoke", "train"):
            folder = self.root / (stem + ("_smoke" if mode == "smoke" else ""))
            label = variant + "_" + mode
            if folder.exists():
                valid = inspect_run(folder, variant, weight, mode == "smoke", plan)
                require(valid is not None, "Preserve incomplete existing output; no automatic restart: " + str(folder))
                self.status["stages"].append(dict(stage=label, state="reused_complete", output=str(folder)))
                self.save()
                continue
            self.run_child(label, command(self.package, folder, variant, weight, mode, plan), gpu=True)
            require(inspect_run(folder, variant, weight, mode == "smoke", plan) is not None,
                    "Child exited without complete evidence: " + str(folder))
        return self.root / stem

    def compare(self, label, before, after):
        result = lambda folder: str(folder / "val_step_002500" / "result.json")
        self.run_child(label, [sys.executable, "-B", str(self.package / "compare_runs.py"),
                              "--before", result(before), "--after", result(after),
                              "--output", str(self.output / label)])

    def run(self):
        try:
            verify_package(self.package)
            original, no_reader = self.root / "original_3407_01", self.root / "no_reader_3407_01"
            plan = self.wait_baseline(original, "original")
            self.wait_baseline(no_reader, "no_reader", plan)
            self.compare("comparison_reader", no_reader, original)
            no_label = self.experiment("no_label", 0.0, "no_label_3407_night_01", plan)
            self.compare("comparison_label", no_label, original)
            if self.args.include_read_aux:
                aux = self.experiment("read_aux", .01, "read_aux_001_3407_night_01", plan)
                self.compare("comparison_read_aux", original, aux)
            self.status.update(phase="complete", complete=True, current_stage=None)
            self.save()
            print("NIGHT COMPLETE " + str(self.output), flush=True)
            return 0
        except BaseException as exc:
            self.status.update(phase="failed", complete=False, error=type(exc).__name__ + ": " + str(exc))
            self.save()
            (self.output / "traceback.txt").write_text(traceback.format_exc())
            print("NIGHT STOP " + self.status["error"], flush=True)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--package", type=Path, default=Path(DEFAULT_PACKAGE))
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--include-read-aux", action="store_true", help="Also run exploratory read_aux with weight 0.01")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_signal)
    require(all(math.isfinite(v) and v > 0 for v in (args.wait_hours, args.max_hours)), "Positive finite time limits required")
    require(args.root.is_dir(), "Existing experiment root missing")
    with (args.root / ".night_followup.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("NIGHT STOP another night controller is already active", file=sys.stderr)
            return 2
        return Queue(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
