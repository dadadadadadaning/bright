#!/usr/bin/env python3
"""Evaluate checkpoint2000 on all Val450 with explicit CUDA synchronization; zero updates."""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from grid_probe import read_json, sha256
from evaluate_checkpoint import write_json, ARMS, REQUIRED_CHECKS, concise_metrics
from index_eval_probe import prepared
from resume_context import inactive_parent
from resume_train import DEFAULT_OUTPUT, verify_files, require_scope
from probe import json_value

FAILURE_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_112222_495713"
FAILURE_TRAINER_SHA = "029137749200f9b4f4d0d5d26e4015dcb5b43a4c3a667fb616c97b8d4fed3570"
BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1")
EVAL_BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1")
FORMAT = "BRIGHT_SYNC_CHECKPOINT2000_VAL450_V1"


def validate_protocol(actual, expected):
    actual, expected = json_value(actual), json_value(expected)
    if actual["execution_settings"].get("cuda_launch_blocking") != "1":
        raise ValueError("Synchronous evaluation requires CUDA_LAUNCH_BLOCKING=1")
    compare = json_value(actual)
    compare["execution_settings"]["cuda_launch_blocking"] = expected["execution_settings"].get("cuda_launch_blocking")
    if compare != expected:
        raise ValueError("Evaluation protocol differs beyond declared launch-blocking override")
    return dict(original=expected["execution_settings"].get("cuda_launch_blocking"),
                evaluation="1", reason="standalone Val450 with declared CUDA synchronization")


def finalize_fresh_report(path, report, *, output, names, evidence, checks, provenance):
    """Finalize only this newly computed standalone evaluation, never the training run."""
    path, output = Path(path).resolve(), Path(output).resolve()
    if path != output / "val_step_002000/result.json":
        raise ValueError("Can finalize only the fresh standalone result")
    if provenance["source_root"] == str(output) or output.is_relative_to(Path(provenance["source_root"]).resolve()):
        raise ValueError("Standalone output must be outside its source run")
    if set(checks) != set(REQUIRED_CHECKS) or any(checks[k] is not True for k in REQUIRED_CHECKS):
        raise ValueError("Standalone runtime integrity checks are incomplete")
    if provenance.get("checkpoint_step") != 2000 or len(names) != 450:
        raise ValueError("Require checkpoint2000 and all450 validation images")
    override = validate_protocol(provenance["protocol_info"], provenance["original_protocol_info"])
    manifest = read_json(output / "manifest.json")
    if (manifest.get("protocol_info") != provenance["protocol_info"]
            or manifest.get("original_protocol_info") != provenance["original_protocol_info"]
            or manifest.get("launch_blocking_override") != override
            or provenance.get("launch_blocking_override") != override):
        raise ValueError("Saved report execution protocol differs from declared override")
    # The inherited evaluator omits protocol_info in result.json. Bind the new
    # result to the already written runtime manifest, leaving pending bytes intact.
    if "protocol_info" in report and report["protocol_info"] != provenance["protocol_info"]:
        raise ValueError("Saved report execution protocol differs from declared override")
    if read_json(path) != report:
        raise ValueError("In-memory report differs from fresh saved report")
    evidence.validate_report(report, step=2000, split="val", names=names, arms=list(ARMS), primary=False)
    evidence.validate_saved_report(path, report, names, sha256, output)
    original_hash = sha256(path)
    backup = path.with_name("result.pending.json")
    with backup.open("xb") as stream:
        stream.write(path.read_bytes())
    finished = copy.deepcopy(report)
    finished.update(complete=True, final_integrity_pending=False,
                    full_validation_evidence=True, phase="standalone_evaluation_complete",
                    primary_endpoint=False, evaluation_scope="saved_checkpoint_val450_synchronous",
                    historical_execution_replayed=False, cuda_launch_blocking="1",
                    protocol_info=provenance["protocol_info"],
                    source_training_run_complete=False, optimizer_updates=0,
                    standalone_provenance=provenance, standalone_integrity_checks=checks,
                    pending_report_sha256=original_hash,
                    nll_reduction_diagnostics=evidence.nll_reduction_diagnostics(report))
    write_json(path, finished)
    path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n", encoding="ascii")
    evidence.validate_saved_report(path, finished, names, sha256, output)
    return finished



@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited = os.environ.pop("BRIGHT_SYNC_VAL_FDS", "")
        if inherited:
            streams = [stack.enter_context(os.fdopen(int(fd), "a+")) for fd in inherited.split(",")]
        else:
            paths = (Path(DEFAULT_OUTPUT) / "active.lock", EVAL_BASE / "evaluation.lock",
                     BASE / "sync_val2000.lock")
            streams = []
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open("a+")))
        for stream in streams:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield streams


def run_evaluation(context, output):
    status = dict(format=FORMAT, phase="loading", complete=False,
                  full_validation_evidence=False, final_integrity_pending=True,
                  training_performed=False, optimizer_updates=0, test_accessed=False,
                  requested_images=450, checkpoint_step=2000,
                  evaluation_scope="saved_checkpoint_val450_synchronous", pid=os.getpid(),
                  original_experiment_complete=False, exact_historical_replay=False,
                  observation_changes_timing=True, per_operator_tracing=False,
                  cpu_index_snapshots=0, route_values_modified=False,
                  index_values_modified=False, cuda_launch_blocking="1",
                  source_root=FAILURE_ROOT, failure_root=FAILURE_ROOT, output=str(output))
    position = dict(index=-1, image=None, active_arm=None, stage="loading")
    observer = None
    stage = "verify_inputs"
    write_json(output / "summary.json", status)
    try:
        manifest, plan, bridge = (context[k] for k in ("manifest", "plan", "bridge"))
        source, addon = plan["source"], context["addon"]
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        verify_files(context["evidence_files"])
        import torch
        from evaluation_observer import EvaluationObserver, track_evaluation
        import evidence
        import checkpoint as checkpoint_io
        from restore_state import restore_training_state, validate_restored_rng
        from trajectory import reconstruct_order
        policy = read_json(source["config"]["protocol"])["numeric_policy"]
        torch.set_num_threads(int(policy["num_threads"]))
        if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
            torch.set_num_interop_threads(int(policy["num_interop_threads"]))
        if not torch.cuda.is_available():
            raise ValueError("Run on the original CUDA server; no CPU fallback")
        _, order, _ = reconstruct_order(torch, manifest, plan, context["train"] / "train.jsonl")
        stage = "load_checkpoint_and_runtime"
        bindings = bridge.load_runtime(source)
        state = bindings.legacy.state
        sys.modules["runtime_bridge"] = bridge
        sys.path.insert(0, str(addon))
        runner = bridge.named(addon / "ablation_runner.py", "sync_eval_runner")
        support_module = bridge.named(addon / "supports.py", "sync_eval_supports")
        evaluator = bridge.named(addon / "evaluation.py", "sync_eval_evaluation")
        saved, cp = checkpoint_io.load(torch, context["parent"]["path"], state, plan=plan,
                    original_manifest=manifest, lineage=context["lineage"],
                    journal_path=context["parent_journal"])
        inherited = saved["input_file_sha256"]
        print("SYNC VAL450: verifying inherited inputs: %s" % len(inherited), flush=True)
        verify_files(inherited)
        runtime_output = output / "runtime"
        runtime_output.mkdir()
        args = bridge.runtime_arguments(source, seed=context["seed"], output=runtime_output)
        print("SYNC VAL450: loading checkpoint2000; zero optimizer updates", flush=True)
        with bindings.Runtime(args, runtime_output) as runtime:
            bindings.original_train._formal_runtime(runtime, False)
            train_names, val_names = bindings.original_train._names(runtime, False)
            if train_names != manifest["train_names"] or val_names != manifest["val_names"]:
                raise ValueError("Train/Val identities differ")
            status["launch_blocking_override"] = validate_protocol(runtime.protocol_info, manifest["protocol_info"])
            bindings.original_train._checked_verify(runtime)
            factory = support_module.SupportFactory(runtime, seed=context["seed"])
            heads, optimizers, _ = runner.model_copies(torch, bindings.original_train, state,
                                                      ARMS, vars(args), runtime.device)
            status["restoration"] = restore_training_state(torch, heads, optimizers, saved, state)
            # Initialize a continuation-style remaining-Train iterator before
            # RNG restoration. This does not reproduce historical live workers.
            # Never consume a train batch or call an optimizer.
            train_iterator = iter(runtime.evaluation_batches(order[2000:2500], seed=context["seed"]))
            require_scope(runtime, inherited)
            status["rng_restoration"] = validate_restored_rng(state, saved["rng"], require_cuda=True)
            del saved
            before_heads = {a: state.state_fingerprint(h.state_dict()) for a, h in heads.items()}
            before_opt = {a: state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()}
            before_rng = state.state_fingerprint(state.capture_rng())
            names = val_names
            if len(names) != 450:
                raise ValueError("Require the complete original Val450")
            write_json(output / "manifest.json", dict(format=FORMAT, source_checkpoint=json_value(cp),
                       original_parent=json_value(context["parent"]), lineage=context["lineage"],
                       val_names=names, train_names=train_names, arm_order=list(ARMS),
                       training_performed=False, optimizer_updates=0, train_iterator_initialized=True,
                       train_batches_consumed=0, full_validation_evidence=False,
                       runtime_config=vars(args), protocol_info=json_value(runtime.protocol_info),
                       original_protocol_info=manifest["protocol_info"],
                       launch_blocking_override=status["launch_blocking_override"]))
            provenance = dict(source_root=FAILURE_ROOT, checkpoint_step=2000,
                checkpoint_sha256=cp["sha256"], trainer_sha256=FAILURE_TRAINER_SHA,
                evaluation_source_sha256=sha256(addon / "evaluation.py"),
                lineage=json_value(context["lineage"]), evaluation_only=True, optimizer_updates=0,
                historical_execution_replayed=False, source_training_complete=False,
                protocol_info=json_value(runtime.protocol_info),
                original_protocol_info=manifest["protocol_info"],
                launch_blocking_override=status["launch_blocking_override"])
            stage = "evaluate_val_2000"
            status["phase"] = stage
            write_json(output / "summary.json", status)
            folder = output / "val_step_002000"
            observer = EvaluationObserver(output, position)
            print("SYNC VAL450: all six arms + Grid; no per-operator tracing", flush=True)
            with track_evaluation(runtime, factory, evaluator, names, state, observer):
                report = evaluator.evaluate(runtime, heads, names, folder, step=2000,
                    split="val", seed=context["seed"], original_train=bindings.original_train,
                    legacy=bindings.legacy, factory=factory, primary_endpoint=False, smoke=False)
            stage = "verify_evaluation_and_runtime"
            evidence.validate_report(report, step=2000, split="val", names=names,
                                     arms=list(ARMS), primary=False)
            evidence.validate_saved_report(folder / "result.json", report, names, sha256, output)
            checks = dict(
                runtime_verified=False, source_verified=False, guard_restored=False,
                checkpoint_and_parent_unchanged=False, evaluation_inputs_match_checkpoint=False,
                head_state_unchanged=before_heads == {a: state.state_fingerprint(h.state_dict()) for a, h in heads.items()},
                optimizer_state_unchanged=before_opt == {a: state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()},
                rng_restored=before_rng == state.state_fingerprint(state.capture_rng()))
            if not all(checks[k] for k in ("head_state_unchanged", "optimizer_state_unchanged", "rng_restored")):
                raise ValueError("Evaluation changed model/optimizer/RNG state")
            status["runtime_integrity"] = json_value(bindings.original_train._checked_verify(runtime))
            checks["runtime_verified"] = True
            require_scope(runtime, inherited)
            verify_files({str(p): h for p, h in runtime.input_ledger.items()})
            checks["evaluation_inputs_match_checkpoint"] = True
            del train_iterator
        stage = "finalize_standalone_evaluation"
        guard = bindings.original_train._restored_guard(runtime)
        if guard is None or guard.get("failures"):
            raise ValueError("Pool guard integrity failure")
        checks["guard_restored"] = True
        bridge.verify_ledgers(source)
        bindings.legacy_bridge.verify_dependencies(bindings.legacy)
        if bridge.check_package(addon) != plan["addon_release_sha256"]:
            raise ValueError("Original addon changed")
        checks["source_verified"] = True
        verify_files(context["evidence_files"])
        checks["checkpoint_and_parent_unchanged"] = True
        finalized = finalize_fresh_report(folder / "result.json", report, output=output,
            names=names, evidence=evidence, checks=checks, provenance=provenance)
        status.update(phase="standalone_evaluation_complete", complete=True,
            full_validation_evidence=True, final_integrity_pending=False, image_count=450,
            result=str(folder / "result.json"), metrics=concise_metrics(finalized),
            integrity_checks=checks, source_files_unchanged=True)
    except BaseException as exc:
        status.update(phase="failed", complete=False, full_validation_evidence=False,
                      final_integrity_pending=True, failure_phase=stage,
                      error=type(exc).__name__ + ": " + str(exc))
        (output / "traceback.txt").write_text(traceback.format_exc())
    # Only CPU metadata is used after failure; no tensor copies or RNG calls.
    if observer is not None:
        status.update(first_failure=observer.first_failure,
            rng_restore_skipped_after_failure=observer.rng_restore_skipped,
            evidence_save_error=observer.evidence_error)
    journal = output / "val_step_002000/images.jsonl"
    if journal.is_file():
        with journal.open() as stream:
            status["completed_images"] = sum(1 for line in stream if line.strip())
    status["last_position"] = position
    status["conclusion"] = "Standalone Val450 under CUDA_LAUNCH_BLOCKING=1 only. This does not resolve the historical CUDA failure or complete training."
    write_json(output / "summary.json", status)
    print("BEGIN SYNCHRONOUS CHECKPOINT2000 VAL450 SUMMARY", flush=True)
    print(json.dumps({k: v for k, v in status.items() if k not in ("restoration", "rng_restoration")},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    print("END SYNCHRONOUS CHECKPOINT2000 VAL450 SUMMARY", flush=True)
    return 0 if status["complete"] else 2


def launch():
    artifact = Path(sys.argv[0]).resolve()
    if not artifact.is_file() or artifact.suffix != ".pyz":
        raise ValueError("Start using the standalone pyz")
    with leases() as streams:
        print("PREFLIGHT: checking checkpoint and recorded failure; this may take a moment", flush=True)
        context = prepared()
        output = Path(tempfile.mkdtemp(prefix="sync_val2000_", dir=BASE))
        pinned = output / "evaluator.pyz"
        shutil.copyfile(artifact, pinned)
        write_json(output / "request.json", dict(format=FORMAT, artifact_sha256=sha256(pinned),
                   evidence_files=context["evidence_files"]))
        write_json(output / "summary.json", dict(format=FORMAT, phase="queued", complete=False,
                   optimizer_updates=0, full_validation_evidence=False))
        command = [sys.executable, "-B", "-u", str(pinned), "worker", "--output", str(output)]
        with (output / "evaluate.log").open("x") as log:
            fds = tuple(s.fileno() for s in streams)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=fds, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                BRIGHT_SYNC_VAL_FDS=",".join(map(str, fds)), CUDA_LAUNCH_BLOCKING="1"))
        write_json(output / "launcher.json", dict(pid=process.pid, command=command))
        (BASE / "latest_sync_val2000.txt").write_text(str(output) + "\n")
        print("已启动第2000步检查点的完整Val450同步验证；零训练更新。", flush=True)
        print("PID=%s\nOUTPUT=%s\n日志=%s" % (process.pid, output, output / "evaluate.log"), flush=True)
        print("查看日志：%s %s logs" % (sys.executable, pinned), flush=True)
        print("查看结果：%s %s status" % (sys.executable, pinned), flush=True)
    return 0


def worker(output):
    output = Path(output).resolve()
    if output.parent != BASE or not output.name.startswith("sync_val2000_"):
        raise ValueError("Unexpected output directory")
    try:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        with leases():
            request = read_json(output / "request.json")
            if request["artifact_sha256"] != sha256(output / "evaluator.pyz"):
                raise ValueError("Evaluation artifact changed")
            context = prepared()
            if context["evidence_files"] != request["evidence_files"]:
                raise ValueError("Source evidence changed after launch")
            with inactive_parent(FAILURE_ROOT):
                return run_evaluation(context, output)
    except Exception as exc:
        write_json(output / "summary.json", dict(format=FORMAT, phase="preflight_failed", complete=False,
                   optimizer_updates=0, full_validation_evidence=False,
                   error=type(exc).__name__ + ": " + str(exc)))
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


def main():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "worker", "logs", "status"), nargs="?", default="start")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "start":
            return launch()
        if args.command == "worker":
            return worker(args.output)
        own = Path(sys.argv[0]).resolve().parent
        output = (Path(args.output).resolve() if args.output else
                  own if own.parent == BASE and own.name.startswith("sync_val2000_") else
                  Path((BASE / "latest_sync_val2000.txt").read_text().strip()).resolve())
        if output.parent != BASE:
            raise ValueError("Unexpected latest output")
        if args.command == "logs":
            print("当前完整验证日志：" + str(output / "evaluate.log"), flush=True)
            os.execvp("tail", ["tail", "-n", "80", "-F", str(output / "evaluate.log")])
        status = read_json(output / "summary.json")
        print(json.dumps({k: v for k, v in status.items()
                          if k not in ("restoration", "rng_restoration")}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print("STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
