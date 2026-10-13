#!/usr/bin/env python3
"""Bounded P/R initialization sequencing; no archive edits or relaxed gates.

Exactly one source-faithful full check is followed by either a fresh full
confirmation or a diagnostic first-failure prefix capture. Training is opt-in
and requires both full checks plus the original training entry's own gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import uuid

FORMAT = "BRIGHT_PRT_SEQUENCE_EXPERIMENTS_V1"
CHECK_FORMAT = "PRT_SEQUENCE_CHECK_V1"
FAILED_DEFAULT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34"
NIGHT_DEFAULT = str(Path(__file__).resolve().parent.parent / "region_night")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save_state(root, state):
    temporary = Path(root) / ".workflow_state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(Path(root) / "workflow_state.json")


def file_ledger(paths):
    return {str(Path(path).resolve()): digest(path) for path in sorted(set(paths))}


def verify_ledger(ledger):
    changed = [name for name, expected in ledger.items()
               if not Path(name).is_file() or digest(name) != expected]
    if changed:
        raise ValueError("Immutable source/evidence files changed: " + repr(changed))


def checked_failed_run(root):
    """Restrict this experiment to the already isolated failed initialization."""
    root = Path(root).resolve()
    paths = {key: root / value for key, value in {
        "protocol": "fresh_e0/protocol.json",
        "manifest": "prt_fit/manifest.json",
        "fit_summary": "prt_fit/summary.json",
        "initial_result": "prt_fit/initial_val/result.json",
        "journal": "prt_fit/initial_val/images.jsonl",
    }.items()}
    protocol, manifest, fit, report = [read_json(paths[key]) for key in
                                      ("protocol", "manifest", "fit_summary", "initial_result")]
    if (protocol.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
            or protocol.get("complete") is not True
            or protocol.get("full_validation_evidence") is not True
            or protocol.get("image_count") != 450
            or protocol.get("scope") != "all" or protocol.get("profile") != "legacy"
            or protocol.get("pool_mode") != "native" or protocol.get("smoke") is not False
            or protocol.get("reproduction", {}).get("passed") is not True
            or any(protocol.get(key) is not True for key in
                   ("input_files_unchanged", "code_unchanged", "frozen_evidence_unchanged", "numerical_unchanged"))):
        raise ValueError("Requires the existing passed legacy/native all-path Val450 E0")
    names = protocol.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 must preserve 450 unique Val names in their exact order")
    if (manifest.get("format") != "SRAC_REGION_HEAD_ADAPT_V1"
            or fit.get("format") != "SRAC_REGION_HEAD_ADAPT_V1"
            or report.get("format") != "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
            or any(record.get("smoke") is not False or record.get("test_accessed") is not False
                   for record in (manifest, fit, report))):
        raise ValueError("Requires archived formal, Train/Val-only head initialization evidence")
    if (fit.get("complete") is not False or fit.get("phase") != "initial_Val"
            or fit.get("updates_per_arm") != {arm: 0 for arm in ("p", "r", "t")}
            or report.get("complete") is not False or report.get("initial") is not True
            or report.get("full_validation_evidence") is not False
            or report.get("image_count") != 450 or report.get("initial_checks_passed") is not False
            or report.get("initial_parameters", {}).get("passed") is not True
            or report.get("native_a1_replay_ok") is not True
            or report.get("reproduction", {}).get("passed") is not True
            or report.get("e0_metric_comparison", {}).get("passed") is not True
            or not report.get("immutable") or any(value is not True for value in report["immutable"].values())):
        raise ValueError("Archived failure must be isolated to initial head probabilities, with zero updates")
    checks = report.get("initial_probability_checks", {})
    pool = report.get("pool_guard", {})
    if (set(checks) != {"p", "r", "t"}
            or any(row.get("checked_images") != 450 for row in checks.values())
            or not any(checks[arm].get("passed") is False for arm in ("p", "r"))
            or checks["t"].get("passed") is not True
            or pool.get("restored") is not True or pool.get("failures") != []):
        raise ValueError("Archived failure must be isolated to P/R probabilities, not pool or T failures")
    checksum = digest(paths["protocol"])
    source_checksum = manifest.get("source_sha256")
    if not isinstance(source_checksum, str) or len(source_checksum) != 64:
        raise ValueError("Missing original head source hash")
    if any(record.get("protocol_sha256") != checksum or record.get("source_sha256") != source_checksum
           for record in (manifest, fit, report)):
        raise ValueError("Failed source/protocol identity does not agree across artifacts")
    rows = [json.loads(line) for line in paths["journal"].read_text().splitlines() if line.strip()]
    if ([row.get("image") for row in rows] != names
            or [row.get("index") for row in rows] != list(range(450))):
        raise ValueError("Archived failed Val journal order differs from passed E0")
    return dict(root=str(root), protocol=str(paths["protocol"]), protocol_sha256=checksum,
                original_source_sha256=source_checksum, protocol_record=protocol,
                evidence_sha256=file_ledger(paths.values()))


def load_night_tools(folder):
    """Import only the verified stdlib launcher; do not initialize CUDA."""
    folder = Path(folder).resolve()
    manifest = read_json(folder / "SHA256SUMS.json")
    if not manifest or any(not isinstance(name, str) or Path(name).name != name
                           or not isinstance(checksum, str) or len(checksum) != 64
                           for name, checksum in manifest.items()):
        raise ValueError("Invalid original night tool SHA256 ledger")
    ledger = {str(folder / name): checksum for name, checksum in manifest.items()}
    if "night_run.py" not in manifest or "region_residual.py" not in manifest:
        raise ValueError("Original night tool package is incomplete")
    verify_ledger(ledger)
    ledger[str(folder / "SHA256SUMS.json")] = digest(folder / "SHA256SUMS.json")
    spec = importlib.util.spec_from_file_location("prt_sequence_original_night", folder / "night_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ledger


@contextmanager
def exclusive_lock(parent):
    with (Path(parent) / ".prt_sequence.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another sequence launcher owns this output parent; no duplicate run") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_command(args, evidence, output, mode):
    return [str(args.python), "-u", str(Path(__file__).with_name("initial_sequence.py")),
            "--repo", str(args.repo), "--failed-run", evidence["root"],
            "--protocol", evidence["protocol"], "--output", str(output), "--mode", mode]


def training_command(args, evidence, root, joint):
    return [str(args.python), "-u", str(Path(args.night_tools) / "night_run.py"),
            "--repo", str(args.repo), "--python", str(args.python), "--joint-run", str(joint),
            "--protocol", evidence["protocol"], "--output-parent", str(Path(root) / "formal_night"),
            "--wait-gpu-idle"]


def full_check_admitted(summary, output, evidence):
    """Independently inspect original strict report; never trust exit zero alone."""
    identity = (
        summary.get("format") == CHECK_FORMAT, summary.get("mode") == "full",
        summary.get("complete") is True, summary.get("full_order_passed") is True,
        summary.get("failure_kind") is None, summary.get("diagnostic_only") is True,
        summary.get("training") is False, summary.get("test_accessed") is False,
        summary.get("full_validation_evidence") is False,
        summary.get("protocol_sha256") == evidence["protocol_sha256"],
        summary.get("original_source_sha256") == evidence["original_source_sha256"],
    )
    if not all(identity):
        return False
    path = Path(output) / "initial_val/result.json"
    if not path.is_file():
        return False
    report = read_json(path)
    checks = report.get("initial_probability_checks", {})
    return bool(
        report.get("format") == "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
        and report.get("complete") is True and report.get("full_validation_evidence") is True
        and report.get("image_count") == 450 and report.get("initial") is True
        and report.get("smoke") is False and report.get("training") is False
        and report.get("test_accessed") is False and report.get("initial_checks_passed") is True
        and report.get("initial_parameters", {}).get("passed") is True
        and report.get("reproduction", {}).get("passed") is True
        and report.get("e0_metric_comparison", {}).get("passed") is True
        and report.get("native_a1_replay_ok") is True and report.get("immutable")
        and all(value is True for value in report["immutable"].values())
        and report.get("pool_guard", {}).get("restored") is True
        and not report["pool_guard"].get("failures")
        and report.get("protocol_sha256") == evidence["protocol_sha256"]
        and report.get("source_sha256") == evidence["original_source_sha256"]
        and set(checks) == {"p", "r", "t"}
        and all(row.get("passed") is True and row.get("checked_images") == 450
                and row.get("failed_images") == 0 for row in checks.values())
    )


def run_stage(name, command, root, repo, env, state, runner, immutable):
    verify_ledger(immutable)
    row = dict(name=name, command=command, status="running", returncode=None,
               log=str(Path(root) / "logs" / (name + ".log")))
    state["stages"].append(row)
    state["current_stage"] = name
    save_state(root, state)
    print("\nStarting " + name, flush=True)
    # run_logged opens/creates this log before starting the child.
    rc = runner(command, row["log"], repo, env)
    row.update(returncode=rc, status="passed" if rc == 0 else "failed")
    save_state(root, state)
    verify_ledger(immutable)
    return rc, row


def execute_checks(args, root, repo, env, state, evidence, joint, night,
                   immutable, runner=None, gpu_wait=None):
    runner = runner or night.run_logged
    for number in (1, 2):
        if gpu_wait is not None:
            state["current_stage"] = "waiting_gpu"
            save_state(root, state)
            gpu_wait()
        mode = "full" if number == 1 or state["checks"][0]["full_order_passed"] else "capture"
        output = Path(root) / ("check_%02d" % number)
        name = "check_%02d_%s" % (number, mode)
        command = check_command(args, evidence, output, mode)
        rc, stage = run_stage(name, command, root, repo, env, state, runner, immutable)
        summary_path = output / "check.json"
        if not summary_path.is_file():
            raise RuntimeError("Check process produced no check.json; no automatic retry or training")
        summary = read_json(summary_path)
        if (summary.get("format") != CHECK_FORMAT or summary.get("mode") != mode
                or summary.get("protocol_sha256") != evidence["protocol_sha256"]
                or summary.get("original_source_sha256") != evidence["original_source_sha256"]
                or summary.get("diagnostic_only") is not True
                or summary.get("training") is not False or summary.get("test_accessed") is not False
                or summary.get("full_validation_evidence") is not False):
            raise RuntimeError("Diagnostic identity/scope differs; no automatic retry or training")
        admitted = bool(mode == "full" and rc == 0 and full_check_admitted(summary, output, evidence))
        record = dict(mode=mode, full_order_passed=admitted, failure_kind=summary.get("failure_kind"),
                      check_file=str(summary_path), check_sha256=digest(summary_path),
                      scene_captured=summary.get("scene_captured", False), returncode=rc)
        state["checks"].append(record)
        state["diagnostic_processes_run"] = number
        save_state(root, state)
        if mode == "full" and not admitted:
            state["training_blocked"] = True
            save_state(root, state)
            if rc != 1 or summary.get("failure_kind") != "initial_probability_gate":
                raise RuntimeError("Full check failed outside the isolated initial probability gate; stopped")
            if number == 2:
                break
        elif mode == "capture":
            state["training_blocked"] = True
            state["capture_status"] = ("captured_failure_scene" if summary.get("scene_captured") is True
                                       else "instrumented_prefix_not_reproduced")
            save_state(root, state)
            if rc != 0 or summary.get("complete") is not True:
                raise RuntimeError("Prefix capture failed; preserve report and stop")
    both_passed = len(state["checks"]) == 2 and all(row["full_order_passed"] for row in state["checks"])
    state.update(training_blocked=not both_passed,
                 history_status="history_not_reproduced" if both_passed else "full_order_gate_failed",
                 root_cause_resolved=False, eligible_for_training=both_passed,
                 diagnostic_complete=True, current_stage="diagnostic_complete")
    save_state(root, state)
    if not both_passed:
        print("DIAGNOSTIC COMPLETE; an original full check failed. Training blocked; no retry.", flush=True)
        return 1
    print("Two fixed full checks passed: history_not_reproduced; root cause is not proven resolved.", flush=True)
    if not args.continue_to_training:
        state.update(complete=True, current_stage="complete", training_entry_started=False)
        save_state(root, state)
        print("CHECKS ONLY COMPLETE. No head updates. Training entry's own initial Val450 is still required.", flush=True)
        return 0
    state["training_entry_started"] = True
    save_state(root, state)
    command = training_command(args, evidence, root, joint)
    rc, _stage = run_stage("formal_night", command, root, repo, env, state, runner, immutable)
    if rc != 0:
        raise RuntimeError("Original night entry failed; no retry, resume, fresh E0 or relaxed gates")
    state.update(complete=True, current_stage="complete", training_entry_returncode=rc)
    save_state(root, state)
    print("SEQUENCE COMPLETE. Trained scores are determined by the original independent Val reports.", flush=True)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--python", default="/root/shared-nvme/conda-envs/bright4090/bin/python")
    p.add_argument("--failed-run", default=FAILED_DEFAULT)
    p.add_argument("--night-tools", default=NIGHT_DEFAULT)
    p.add_argument("--output-parent", default="/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_SEQUENCE_EXPERIMENTS_V1")
    p.add_argument("--plan-only", action="store_true", help="Print the bounded conditional plan; no output/GPU/Train/Val")
    p.add_argument("--wait-gpu-idle", action="store_true", help="Wait for target GPU; never kill another job")
    p.add_argument("--gpu-wait-timeout", type=int, default=0)
    options = p.add_mutually_exclusive_group()
    options.add_argument("--continue-to-training", action="store_true",
                         help="Opt in to one original P/R/T+residual sequence only after two full checks pass")
    options.add_argument("--checks-only", action="store_true", help="Default behavior; run at most two diagnostic processes")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    repo, parent = Path(args.repo).resolve(), Path(args.output_parent).resolve()
    args.repo, args.python = str(repo), str(Path(args.python).resolve())
    args.night_tools = str(Path(args.night_tools).resolve())
    if args.gpu_wait_timeout < 0:
        raise ValueError("GPU wait timeout cannot be negative")
    if not (repo / "bda_benchmark").is_dir() or not Path(args.python).is_file():
        raise FileNotFoundError("Original repository or Python executable does not exist")
    if not Path(__file__).with_name("initial_sequence.py").is_file():
        raise FileNotFoundError("Companion initial_sequence.py missing")
    evidence = checked_failed_run(args.failed_run)
    night, night_ledger = load_night_tools(args.night_tools)
    joint = Path(evidence["protocol_record"]["arguments"]["joint_run"])
    joint = (repo / joint).resolve() if not joint.is_absolute() else joint.resolve()
    source = night.source_arguments(joint, repo)
    for name in ("repo", "failed_run", "joint_run", "cache", "dataset"):
        immutable_root = Path(repo if name == "repo" else evidence["root"] if name == "failed_run" else source[name]).resolve()
        if parent == immutable_root or immutable_root in parent.parents:
            raise ValueError("Output parent enters immutable source: " + name)
    if args.wait_gpu_idle:
        night.gpu_selector(source["device"])
    root = parent / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    if args.plan_only:
        import shlex
        print("PLAN ONLY: no output, GPU, Train or Val access. Existing passed E0 is never regenerated.")
        print("check_01_full: " + shlex.join(check_command(args, evidence, root / "check_01", "full")))
        print("check_02_if_first_passed: " + shlex.join(check_command(args, evidence, root / "check_02", "full")))
        print("check_02_if_first_initial_gate_failed: " + shlex.join(check_command(args, evidence, root / "check_02", "capture")))
        if args.continue_to_training:
            print("IF BOTH FULL CHECKS PASS (not root-cause proof), one training sequence: "
                  + shlex.join(training_command(args, evidence, root, joint)))
        else:
            print("DEFAULT CHECKS ONLY: no training. At most two fresh diagnostic processes, no retries.")
        return 0
    parent.mkdir(parents=True, exist_ok=True)
    source_files = [path for path in (repo / "bda_benchmark").rglob("*")
                    if path.is_file() and path.suffix in (".py", ".sh")]
    package_files = [path for path in Path(__file__).parent.iterdir()
                     if path.is_file() and path.suffix in (".py", ".sh", ".md", ".json")]
    immutable = {**file_ledger(source_files + package_files), **night_ledger, **evidence["evidence_sha256"]}
    env = dict(os.environ, PYTHON=args.python, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with exclusive_lock(parent):
        root.mkdir(exist_ok=False)
        (root / "logs").mkdir()
        state = dict(format=FORMAT, complete=False, root=str(root), current_stage="initializing",
                     diagnostic_only=not args.continue_to_training, full_validation_evidence=False,
                     test_accessed=False, diagnostic_budget=2, diagnostic_processes_run=0,
                     checks=[], stages=[], training_entry_started=False, training_blocked=True,
                     head_updates_claimed=False,
                     training_entry_rule="Entry startup is not optimizer-update evidence; use original fit summaries for update counts",
                     eligible_for_training=False, root_cause_resolved=False,
                     protocol=evidence["protocol"], protocol_sha256=evidence["protocol_sha256"],
                     original_source_sha256=evidence["original_source_sha256"],
                     evidence_sha256=evidence["evidence_sha256"], immutable_files=immutable,
                     arguments=vars(args), rule="No retries; no altered profile, thresholds or protocol; any original full failure blocks training")
        save_state(root, state)
        print("OUTPUT_ROOT=" + str(root), flush=True)
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("Stopped this launcher's own child only")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            command = [args.python, "-c", night.SOURCE_CHECK, str(joint), evidence["protocol"]]
            rc, _row = run_stage("source_check", command, root, repo, env, state, night.run_logged, immutable)
            if rc != 0:
                raise RuntimeError("Original source/protocol check failed; no GPU diagnostic or training")
            wait = (lambda: night.wait_gpu_idle(source["device"], args.gpu_wait_timeout)) if args.wait_gpu_idle else None
            return execute_checks(args, root, repo, env, state, evidence, joint, night,
                                  immutable, gpu_wait=wait)
        except BaseException as error:
            state.update(complete=False, training_blocked=True,
                         failure=type(error).__name__ + ": " + str(error))
            if state["stages"] and state["stages"][-1]["status"] == "running":
                state["stages"][-1]["status"] = "interrupted"
            save_state(root, state)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt as error:
        print(str(error), file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
