#!/usr/bin/env python3
"""Diagnose checkpoint2000 Val indexing synchronously, with zero updates."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

from grid_probe import read_json, sha256
from evaluate_checkpoint import write_json, ARMS
from resume_context import prepare, inactive_parent
from resume_train import DEFAULT_RUN, DEFAULT_OUTPUT, verify_files, require_scope
from probe import json_value

FAILURE_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_112222_495713"
FAILURE_TRAINER_SHA = "029137749200f9b4f4d0d5d26e4015dcb5b43a4c3a667fb616c97b8d4fed3570"
BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS")
EVAL_BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1")
TARGET_INDEX = 30
TARGET_IMAGE = "beirut-explosion_00000016"
FORMAT = "BRIGHT_VAL_INDEX_PREFIX_DIAGNOSTIC_V1"


def check_failure(context, root):
    root = Path(root)
    failure, status, manifest = [read_json(root / n) for n in
                                ("evaluation_failure.json", "summary.json", "manifest.json")]
    names = context["manifest"]["val_names"]
    if (failure.get("index") != TARGET_INDEX or failure.get("image") != TARGET_IMAGE
            or names[TARGET_INDEX] != TARGET_IMAGE or failure.get("path") != "b"
            or "device-side assert" not in failure.get("error", "")
            or status.get("phase") != "failed" or status.get("last_complete_step") != 2000
            or status.get("failure_phase") != "evaluate_val_2000"
            or status.get("new_updates_per_arm") != {a: 500 for a in ARMS}
            or manifest.get("val_names") != names
            or context["parent"]["step"] != 2000
            or sha256(root / "trainer.pyz") != FAILURE_TRAINER_SHA):
        raise ValueError("Require recorded Val2000 B failure and complete checkpoint2000")
    journal = root / "val_step_002000/images.jsonl"
    with journal.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if [r["image"] for r in rows] != names[:TARGET_INDEX]:
        raise ValueError("Failed evaluation journal must contain exactly first30 Val images")
    paths = [root / n for n in ("evaluation_failure.json", "summary.json", "manifest.json",
                               "traceback.txt", "trainer.pyz", "route_observer_summary.json")] + [journal]
    context["evidence_files"].update({str(p): sha256(p) for p in paths})
    return context


def prepared():
    context = prepare(SimpleNamespace(original_run=DEFAULT_RUN, seed=3407,
                      checkpoint_every=50, addon_dir=None, parent_root=FAILURE_ROOT))
    return check_failure(context, FAILURE_ROOT)


def validate_protocol(actual, expected):
    actual, expected = json_value(actual), json_value(expected)
    if actual["execution_settings"].get("cuda_launch_blocking") != "1":
        raise ValueError("Synchronous diagnostic requires CUDA_LAUNCH_BLOCKING=1")
    compare = json_value(actual)
    compare["execution_settings"]["cuda_launch_blocking"] = expected["execution_settings"].get("cuda_launch_blocking")
    if compare != expected:
        raise ValueError("Diagnostic protocol differs beyond declared launch-blocking override")
    return dict(original=expected["execution_settings"].get("cuda_launch_blocking"),
                diagnostic="1", reason="localize CUDA index failure; not formal evaluation")


@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited = os.environ.pop("BRIGHT_INDEX_EVAL_FDS", "")
        if inherited:
            streams = [stack.enter_context(os.fdopen(int(fd), "a+")) for fd in inherited.split(",")]
        else:
            paths = (Path(DEFAULT_OUTPUT) / "active.lock", EVAL_BASE / "evaluation.lock",
                     BASE / "index_eval.lock")
            streams = []
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open("a+")))
        for stream in streams:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield streams


def run_probe(context, output):
    status = dict(format=FORMAT, phase="loading", complete=False,
                  diagnostic_complete=False, full_validation_evidence=False,
                  training_performed=False, optimizer_updates=0, test_accessed=False,
                  target_image=TARGET_IMAGE, target_index=TARGET_INDEX,
                  requested_images=TARGET_INDEX + 1, checkpoint_step=2000,
                  original_experiment_complete=False, exact_historical_replay=False,
                  observation_changes_timing=True, route_values_modified=False,
                  index_values_modified=False, cuda_launch_blocking="1",
                  source_root=FAILURE_ROOT, failure_root=FAILURE_ROOT, output=str(output))
    position = dict(index=-1, image=None, active_arm=None, stage="loading")
    trace = None
    stage = "verify_inputs"
    write_json(output / "summary.json", status)
    try:
        manifest, plan, bridge = (context[k] for k in ("manifest", "plan", "bridge"))
        source, addon = plan["source"], context["addon"]
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        verify_files(context["evidence_files"])
        import torch
        from index_observer import IndexTrace, track_evaluation
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
        runner = bridge.named(addon / "ablation_runner.py", "route_eval_runner")
        support_module = bridge.named(addon / "supports.py", "route_eval_supports")
        evaluator = bridge.named(addon / "evaluation.py", "route_eval_evaluation")
        saved, cp = checkpoint_io.load(torch, context["parent"]["path"], state, plan=plan,
                    original_manifest=manifest, lineage=context["lineage"],
                    journal_path=context["parent_journal"])
        inherited = saved["input_file_sha256"]
        print("INDEX DIAGNOSTIC: verifying inherited inputs: %s" % len(inherited), flush=True)
        verify_files(inherited)
        runtime_output = output / "runtime"
        runtime_output.mkdir()
        args = bridge.runtime_arguments(source, seed=context["seed"], output=runtime_output)
        print("INDEX DIAGNOSTIC: loading checkpoint2000; zero optimizer updates", flush=True)
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
            names = val_names[:TARGET_INDEX + 1]
            write_json(output / "manifest.json", dict(format=FORMAT, source_checkpoint=json_value(cp),
                       original_parent=json_value(context["parent"]), lineage=context["lineage"],
                       val_prefix_names=names, arm_order=list(ARMS), target_index=TARGET_INDEX,
                       training_performed=False, optimizer_updates=0, train_iterator_initialized=True,
                       train_batches_consumed=0, full_validation_evidence=False,
                       runtime_config=vars(args), protocol_info=json_value(runtime.protocol_info),
                       original_protocol_info=manifest["protocol_info"],
                       launch_blocking_override=status["launch_blocking_override"]))
            stage = "evaluate_prefix"
            status["phase"] = stage
            write_json(output / "summary.json", status)
            trace = IndexTrace(output, position, TARGET_INDEX)
            with track_evaluation(runtime, factory, evaluator, names, state, trace):
                evaluator.evaluate(runtime, heads, names, output / "diagnostic_val_prefix", step=2000,
                    split="val", seed=context["seed"], original_train=bindings.original_train,
                    legacy=bindings.legacy, factory=factory, primary_endpoint=False, smoke=True)
            stage = "verify_unchanged_state"
            checks = dict(
                head_state_unchanged=before_heads == {a: state.state_fingerprint(h.state_dict()) for a, h in heads.items()},
                optimizer_state_unchanged=before_opt == {a: state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()},
                rng_restored=before_rng == state.state_fingerprint(state.capture_rng()))
            if not all(checks.values()):
                raise ValueError("Diagnostic changed model/optimizer/RNG state")
            bindings.original_train._checked_verify(runtime)
            require_scope(runtime, inherited)
            verify_files({str(p): h for p, h in runtime.input_ledger.items()})
            del train_iterator
        guard = bindings.original_train._restored_guard(runtime)
        if guard is None or guard.get("failures"):
            raise ValueError("Pool guard integrity failure")
        bridge.verify_ledgers(source)
        bindings.legacy_bridge.verify_dependencies(bindings.legacy)
        if bridge.check_package(addon) != plan["addon_release_sha256"]:
            raise ValueError("Original addon changed")
        verify_files(context["evidence_files"])
        status.update(phase="diagnostic_prefix_complete", diagnostic_complete=True,
                      integrity_checks=checks, source_files_unchanged=True)
    except BaseException as exc:
        status.update(phase="diagnostic_failed", failure_phase=stage,
                      error=type(exc).__name__ + ": " + str(exc))
        (output / "traceback.txt").write_text(traceback.format_exc())
    # Retained evidence is CPU-only. No new GPU reads after an exception here.
    if trace is not None:
        try:
            report = trace.save()
            status["index_counts"] = {k: report[k] for k in ("operation_count", "index_operation_count", "cpu_index_snapshots")}
            status["first_failure"] = report["first_failure"]
            status["rng_restore_skipped_after_failure"] = report["rng_restore_skipped_after_failure"]
        except Exception as exc:
            status.update(evidence_save_error=str(exc), diagnostic_complete=False)
            if status["phase"] == "diagnostic_prefix_complete":
                status["phase"] = "diagnostic_evidence_save_failed"
    journal = output / "diagnostic_val_prefix/images.jsonl"
    if journal.is_file():
        with journal.open() as stream:
            status["completed_images"] = sum(1 for line in stream if line.strip())
    status["last_position"] = position
    status["conclusion"] = "Diagnostic only. A prefix pass does not resolve the historical failure or complete training."
    write_json(output / "summary.json", status)
    print("BEGIN INDEX EVAL DIAGNOSTIC SUMMARY", flush=True)
    print(json.dumps({k: v for k, v in status.items() if k not in ("restoration", "rng_restoration")},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    print("END INDEX EVAL DIAGNOSTIC SUMMARY", flush=True)
    return 0 if status["diagnostic_complete"] else 2


def launch():
    artifact = Path(sys.argv[0]).resolve()
    if not artifact.is_file() or artifact.suffix != ".pyz":
        raise ValueError("Start using the standalone pyz")
    with leases() as streams:
        print("PREFLIGHT: checking checkpoint and recorded failure; this may take a moment", flush=True)
        context = prepared()
        output = Path(tempfile.mkdtemp(prefix="index_eval2000_", dir=BASE))
        pinned = output / "probe.pyz"
        shutil.copyfile(artifact, pinned)
        write_json(output / "request.json", dict(format=FORMAT, artifact_sha256=sha256(pinned),
                   evidence_files=context["evidence_files"]))
        write_json(output / "summary.json", dict(format=FORMAT, phase="queued", complete=False,
                   diagnostic_complete=False, optimizer_updates=0, full_validation_evidence=False))
        command = [sys.executable, "-B", "-u", str(pinned), "worker", "--output", str(output)]
        with (output / "diagnostic.log").open("x") as log:
            fds = tuple(s.fileno() for s in streams)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=fds, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                BRIGHT_INDEX_EVAL_FDS=",".join(map(str, fds)), CUDA_LAUNCH_BLOCKING="1"))
        write_json(output / "launcher.json", dict(pid=process.pid, command=command))
        (BASE / "latest_index_eval.txt").write_text(str(output) + "\n")
        print("已启动后台索引同步诊断；从第2000步检查点计算前31张Val图，零训练更新。", flush=True)
        print("PID=%s\nOUTPUT=%s\n日志=%s" % (process.pid, output, output / "diagnostic.log"), flush=True)
        print("查看日志：%s %s logs" % (sys.executable, pinned), flush=True)
        print("查看结果：%s %s status" % (sys.executable, pinned), flush=True)
    return 0


def worker(output):
    output = Path(output).resolve()
    if output.parent != BASE or not output.name.startswith("index_eval2000_"):
        raise ValueError("Unexpected output directory")
    try:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        with leases():
            request = read_json(output / "request.json")
            if request["artifact_sha256"] != sha256(output / "probe.pyz"):
                raise ValueError("Diagnostic artifact changed")
            context = prepared()
            if context["evidence_files"] != request["evidence_files"]:
                raise ValueError("Source evidence changed after launch")
            with inactive_parent(FAILURE_ROOT):
                return run_probe(context, output)
    except Exception as exc:
        write_json(output / "summary.json", dict(format=FORMAT, phase="preflight_failed", complete=False,
                   diagnostic_complete=False, optimizer_updates=0, full_validation_evidence=False,
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
                  own if own.parent == BASE and own.name.startswith("index_eval2000_") else
                  Path((BASE / "latest_index_eval.txt").read_text().strip()).resolve())
        if output.parent != BASE:
            raise ValueError("Unexpected latest output")
        if args.command == "logs":
            print("当前诊断日志：" + str(output / "diagnostic.log"), flush=True)
            os.execvp("tail", ["tail", "-n", "80", "-F", str(output / "diagnostic.log")])
        print(json.dumps(read_json(output / "summary.json"), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print("STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
