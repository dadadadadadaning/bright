#!/usr/bin/env python3
"""Replay Val prefix at checkpoint1500 with route observations and zero updates."""
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

from grid_probe import read_json, sha256, DEFAULT_ROOT
from evaluate_checkpoint import write_json, ARMS
from training_common import certificate
from resume_context import prepare, inactive_parent
from resume_train import DEFAULT_RUN, DEFAULT_OUTPUT, verify_files, require_scope
from probe import json_value
from eval_route_observer import SelectiveRouteTrace, track_evaluation

FAILURE_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_090401_436171"
FAILURE_TRAINER_SHA = "994ee877c6edf332e4ebae70c48c81442877dcb6ab053e8cab4323ebb0075e3d"
BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS")
EVAL_BASE = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1")
TARGET_INDEX = 167
TARGET_IMAGE = "la_palma-volcano_00000822"
FORMAT = "BRIGHT_VAL_ROUTE_PREFIX_DIAGNOSTIC_V1"


def check_failure(context, root):
    root = Path(root)
    failure, status, manifest = [read_json(root / n) for n in
                                ("evaluation_failure.json", "summary.json", "manifest.json")]
    names = context["manifest"]["val_names"]
    if (failure.get("index") != TARGET_INDEX or failure.get("image") != TARGET_IMAGE
            or names[TARGET_INDEX] != TARGET_IMAGE
            or "route weights must be nonnegative and sum to one" not in failure.get("error", "")
            or status.get("phase") != "failed" or status.get("last_complete_step") != 1500
            or status.get("new_updates_per_arm") != {a: 0 for a in ARMS}
            or manifest.get("val_names") != names
            or manifest.get("parent", {}).get("sha256") != context["parent"]["sha256"]
            or sha256(root / "trainer.pyz") != FAILURE_TRAINER_SHA):
        raise ValueError("Latest failure is not the recorded zero-update boundary-Val1500 failure")
    journal = root / "val_step_001500/images.jsonl"
    with journal.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if [r["image"] for r in rows] != names[:TARGET_INDEX]:
        raise ValueError("Failed evaluation journal does not contain exactly the first167 Val images")
    paths = [root / n for n in ("evaluation_failure.json", "summary.json", "manifest.json",
                               "traceback.txt", "trainer.pyz")] + [journal]
    context["evidence_files"].update({str(p): sha256(p) for p in paths})
    return context


def prepared():
    context = certificate(prepare(SimpleNamespace(original_run=DEFAULT_RUN, seed=3407,
                          checkpoint_every=50, addon_dir=None, parent_root=DEFAULT_ROOT)))
    return check_failure(context, FAILURE_ROOT)


@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited = os.environ.pop("BRIGHT_ROUTE_EVAL_FDS", "")
        if inherited:
            streams = [stack.enter_context(os.fdopen(int(fd), "a+")) for fd in inherited.split(",")]
        else:
            paths = (Path(DEFAULT_OUTPUT) / "active.lock", EVAL_BASE / "evaluation.lock",
                     BASE / "route_eval.lock")
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
                  requested_images=TARGET_INDEX + 1, checkpoint_step=1500,
                  original_experiment_complete=False, exact_historical_replay=False,
                  observation_changes_timing=True, route_values_modified=False,
                  source_root=DEFAULT_ROOT, failure_root=FAILURE_ROOT, output=str(output))
    position = dict(index=-1, image=None, active_arm=None, stage="loading")
    trace = None
    stage = "verify_inputs"
    write_json(output / "summary.json", status)
    try:
        manifest, plan, bridge = (context[k] for k in ("manifest", "plan", "bridge"))
        source, addon = plan["source"], context["addon"]
        blocking = manifest["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
        if blocking is None:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        else:
            os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
        verify_files(context["evidence_files"])
        import torch
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
        verify_files(inherited)
        runtime_output = output / "runtime"
        runtime_output.mkdir()
        args = bridge.runtime_arguments(source, seed=context["seed"], output=runtime_output)
        print("ROUTE DIAGNOSTIC: loading checkpoint1500; zero optimizer updates", flush=True)
        with bindings.Runtime(args, runtime_output) as runtime:
            bindings.original_train._formal_runtime(runtime, False)
            train_names, val_names = bindings.original_train._names(runtime, False)
            if train_names != manifest["train_names"] or val_names != manifest["val_names"]:
                raise ValueError("Train/Val identities differ")
            if json_value(runtime.protocol_info) != manifest["protocol_info"]:
                raise ValueError("Runtime numeric protocol differs")
            bindings.original_train._checked_verify(runtime)
            factory = support_module.SupportFactory(runtime, seed=context["seed"])
            heads, optimizers, _ = runner.model_copies(torch, bindings.original_train, state,
                                                      ARMS, vars(args), runtime.device)
            status["restoration"] = restore_training_state(torch, heads, optimizers, saved, state)
            # Match the continuation's loader initialization before RNG restore.
            # Hold its prefetched iterator alive, but never consume a train batch.
            train_iterator = iter(runtime.evaluation_batches(order[1500:2500], seed=context["seed"]))
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
                       runtime_config=vars(args), protocol_info=json_value(runtime.protocol_info)))
            stage = "evaluate_prefix"
            status["phase"] = stage
            write_json(output / "summary.json", status)
            with (output / "route_trace.jsonl").open("x") as stream:
                def emit(event, **fields):
                    stream.write(json.dumps(dict(event=event, **fields), allow_nan=False) + "\n")
                    stream.flush()
                trace = SelectiveRouteTrace(torch, type(heads["d"]), emit, position)
                with track_evaluation(runtime, factory, evaluator, names, TARGET_INDEX, trace, emit):
                    evaluator.evaluate(runtime, heads, names, output / "diagnostic_val_prefix", step=1500,
                        split="val", seed=context["seed"], original_train=bindings.original_train,
                        legacy=bindings.legacy, factory=factory, primary_endpoint=False, smoke=True)
            stage = "verify_unchanged_state"
            if trace.diagnostic_error_count:
                raise ValueError("Prefix returned but route observations contain diagnostic errors")
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
        route = trace.summary()
        write_json(output / "route_summary.json", route)
        status["route_counts"] = {k: route[k] for k in ("validation_call_count", "validator_failure_count", "diagnostic_error_count")}
        status["first_route_failure"] = route["first_validation_failure"]
        try:
            trace.save_cpu_case(output / "route_cpu_case.pt")
        except Exception as exc:
            status["cpu_case_save_error"] = str(exc)
            status["diagnostic_complete"] = False
            if status["phase"] == "diagnostic_prefix_complete":
                status["phase"] = "diagnostic_evidence_save_failed"
    journal = output / "diagnostic_val_prefix/images.jsonl"
    if journal.is_file():
        with journal.open() as stream:
            status["completed_images"] = sum(1 for line in stream if line.strip())
    status["last_position"] = position
    status["conclusion"] = "Diagnostic only. A prefix pass does not resolve the historical failure or complete training."
    write_json(output / "summary.json", status)
    print("BEGIN ROUTE EVAL DIAGNOSTIC SUMMARY", flush=True)
    print(json.dumps({k: v for k, v in status.items() if k not in ("restoration", "rng_restoration")},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    print("END ROUTE EVAL DIAGNOSTIC SUMMARY", flush=True)
    return 0 if status["diagnostic_complete"] else 2


def launch():
    artifact = Path(sys.argv[0]).resolve()
    if not artifact.is_file() or artifact.suffix != ".pyz":
        raise ValueError("Start using the standalone pyz")
    with leases() as streams:
        print("PREFLIGHT: checking checkpoint and recorded failure; this may take a moment", flush=True)
        context = prepared()
        output = Path(tempfile.mkdtemp(prefix="route_eval1500_", dir=BASE))
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
                BRIGHT_ROUTE_EVAL_FDS=",".join(map(str, fds))))
        write_json(output / "launcher.json", dict(pid=process.pid, command=command))
        (BASE / "latest_route_eval.txt").write_text(str(output) + "\n")
        print("已启动后台路由诊断；只计算验证前缀，不更新参数。", flush=True)
        print("PID=%s\nOUTPUT=%s\n日志=%s" % (process.pid, output, output / "diagnostic.log"), flush=True)
        print("查看日志：%s %s logs" % (sys.executable, pinned), flush=True)
        print("查看结果：%s %s status" % (sys.executable, pinned), flush=True)
    return 0


def worker(output):
    output = Path(output).resolve()
    if output.parent != BASE or not output.name.startswith("route_eval1500_"):
        raise ValueError("Unexpected output directory")
    try:
        with leases():
            request = read_json(output / "request.json")
            if request["artifact_sha256"] != sha256(output / "probe.pyz"):
                raise ValueError("Diagnostic artifact changed")
            context = prepared()
            if context["evidence_files"] != request["evidence_files"]:
                raise ValueError("Source evidence changed after launch")
            with inactive_parent(DEFAULT_ROOT), inactive_parent(FAILURE_ROOT):
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
        output = Path((BASE / "latest_route_eval.txt").read_text().strip()).resolve()
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
