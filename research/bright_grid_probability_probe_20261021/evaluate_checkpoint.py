#!/usr/bin/env python3
"""Evaluate the saved six-arm step1500 checkpoint on Val450; never train."""
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
from types import SimpleNamespace
import zipfile

from grid_probe import DEFAULT_ROOT, DEFAULT_IMAGE, locate, probability_details, read_json, sha256

FORMAT = "BRIGHT_SIX_ARM_CHECKPOINT_EVAL_ONLY_V1"
TRAINER_SHA256 = "0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07"
ARMS = ("e", "f", "b", "d", "r", "l")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def observed_evaluation(torch, runtime, evaluator, names, output, markers):
    """Scoped observation only: call the original validator, and re-raise every error."""
    had_instance_prepare = "prepare" in vars(runtime)
    previous_instance_prepare = vars(runtime).get("prepare")
    original_prepare = runtime.prepare
    original_validator = evaluator._validate_probability
    state = dict(index=-1, image=None, path=None, stage="before_evaluation")

    def emit(event, **fields):
        markers.write(json.dumps(dict(event=event, **state, **fields),
                                 ensure_ascii=False, allow_nan=False) + "\n")
        markers.flush()

    def prepare(batch):
        index = state["index"] + 1
        if index >= len(names):
            raise ValueError("Observed evaluation exceeded requested image count")
        state.update(index=index, image=names[index], path=None, stage="prepare_begin")
        emit("prepare_begin")
        example = original_prepare(batch)
        if example.image != names[index]:
            raise ValueError("Observed evaluation image order differs")
        state["stage"] = "prepare_end"
        emit("prepare_end")
        return example

    def validate(probability, target, path):
        state.update(path=path, stage="probability_validation")
        emit("probability_validation_begin")
        try:
            result = original_validator(probability, target, path)
        except BaseException as exc:
            failure = dict(**state, error=type(exc).__name__ + ": " + str(exc),
                           probability_values_modified=False)
            # ValueError is the original pure validation failure. CUDA/runtime
            # exceptions must not trigger any further tensor reads or saves.
            if isinstance(exc, ValueError):
                try:
                    failure["probability"] = probability_details(torch, probability, target)
                    if isinstance(probability, torch.Tensor):
                        cpu_probability = probability.detach().cpu()
                        torch.save(dict(probability=cpu_probability, target=target.detach().cpu(),
                                        image=state["image"], path=path),
                                   output / "probability_failure.pt")
                        failure["saved_cpu_case"] = "probability_failure.pt"
                except BaseException as diagnostic_exc:
                    failure["diagnostic_error"] = type(diagnostic_exc).__name__ + ": " + str(diagnostic_exc)
            write_json(output / "probability_failure.json", failure)
            emit("probability_validation_failed", error=failure["error"])
            raise
        state["stage"] = "probability_validated"
        emit("probability_validation_passed")
        return result

    runtime.prepare = prepare
    evaluator._validate_probability = validate
    try:
        yield state
    finally:
        evaluator._validate_probability = original_validator
        if had_instance_prepare:
            runtime.prepare = previous_instance_prepare
        else:
            del runtime.prepare


REQUIRED_CHECKS = (
    "runtime_verified", "source_verified", "head_state_unchanged",
    "optimizer_state_unchanged", "rng_restored", "guard_restored",
    "checkpoint_and_parent_unchanged", "evaluation_inputs_match_checkpoint",
)


def finalize_fresh_report(path, report, *, output, names, evidence, checks, provenance):
    """Finalize only this newly computed standalone evaluation, never the training run."""
    path, output = Path(path).resolve(), Path(output).resolve()
    if path != output / "val_step_001500/result.json":
        raise ValueError("Can finalize only the fresh standalone result")
    if provenance["source_root"] == str(output) or output.is_relative_to(Path(provenance["source_root"]).resolve()):
        raise ValueError("Standalone output must be outside its source run")
    if set(checks) != set(REQUIRED_CHECKS) or any(checks[k] is not True for k in REQUIRED_CHECKS):
        raise ValueError("Standalone runtime integrity checks are incomplete")
    evidence.validate_report(report, step=1500, split="val", names=names, arms=list(ARMS), primary=False)
    evidence.validate_saved_report(path, report, names, sha256, output)
    original_hash = sha256(path)
    backup = path.with_name("result.pending.json")
    with backup.open("xb") as stream:
        stream.write(path.read_bytes())
    finished = copy.deepcopy(report)
    finished.update(complete=True, final_integrity_pending=False,
                    full_validation_evidence=True, phase="standalone_evaluation_complete",
                    primary_endpoint=False, evaluation_scope="saved_checkpoint_val450_only",
                    source_training_run_complete=False, optimizer_updates=0,
                    standalone_provenance=provenance, standalone_integrity_checks=checks,
                    pending_report_sha256=original_hash,
                    nll_reduction_diagnostics=evidence.nll_reduction_diagnostics(report))
    write_json(path, finished)
    path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n", encoding="ascii")
    evidence.validate_saved_report(path, finished, names, sha256, output)
    return finished


def concise_metrics(report):
    return {arm: dict(miou4_pct=(100 * sum(m["iou"]) / 4
                               if all(v is not None for v in m["iou"]) else None),
                      foreground_miou3_pct=100 * m["foreground_miou"],
                      iou_pct=[None if v is None else 100 * v for v in m["iou"]])
            for arm, m in report["metrics"].items()}


def output_base(root):
    return root.parent.parent / "SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1"


def worker(root, output):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.parent != output_base(root) or not output.name.startswith("eval1500_"):
        raise ValueError("Require a fresh dedicated evaluation output")
    launch = read_json(output / "launch.json")
    if launch.get("source_root") != str(root) or launch.get("format") != FORMAT:
        raise ValueError("Launch record differs")
    if launch.get("evaluator_sha256") != sha256(output / "evaluator.pyz"):
        raise ValueError("Pinned standalone evaluator differs")
    status = dict(format=FORMAT, phase="preflight", complete=False,
                  full_validation_evidence=False, final_integrity_pending=True,
                  training_performed=False, optimizer_updates=0, test_accessed=False,
                  source_training_run_complete=False,
                  evaluation_scope="saved_checkpoint_val450_only",
                  source_root=str(root), output=str(output), checkpoint_step=1500,
                  pid=os.getpid(), historical_training_state_replayed=False)
    write_json(output / "summary.json", status)
    stage = "preflight"
    observed_state = None
    try:
        with (output.parent / "evaluation.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("EVAL ONLY: verifying saved checkpoint and original code; optimizer updates=0", flush=True)
            manifest, checkpoint_record, _ = locate(root, DEFAULT_IMAGE)
            archive = root / "trainer.pyz"
            if sha256(archive) != TRAINER_SHA256:
                raise ValueError("Pinned trainer differs from the tested helper archive")
            sys.path.insert(0, str(archive))
            from resume_context import prepare, inactive_parent
            from resume_train import verify_files, require_scope
            from restore_state import restore_training_state, validate_restored_rng
            import checkpoint as checkpoint_io
            import evidence
            from probe import json_value
            context = prepare(SimpleNamespace(original_run=manifest["lineage"]["original_run"],
                                             seed=manifest["seed"], checkpoint_every=50,
                                             addon_dir=None, parent_root=str(root)))
            parent, plan, original = context["parent"], context["plan"], context["manifest"]
            if parent["step"] != 1500 or parent["sha256"] != checkpoint_record["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint differs from step1500 evidence")
            with inactive_parent(root):
                verify_files(context["evidence_files"])
                blocking = original["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
                if blocking is None:
                    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
                else:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
                import torch
                source, bridge, addon = plan["source"], context["bridge"], context["addon"]
                policy = read_json(source["config"]["protocol"])["numeric_policy"]
                torch.set_num_threads(int(policy["num_threads"]))
                if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
                    torch.set_num_interop_threads(int(policy["num_interop_threads"]))
                if not torch.cuda.is_available():
                    raise ValueError("Require the original CUDA environment")
                stage = "load_checkpoint_and_bindings"
                bindings = bridge.load_runtime(source)
                state = bindings.legacy.state
                sys.modules["runtime_bridge"] = bridge
                sys.path.insert(0, str(addon))
                runner = bridge.named(addon / "ablation_runner.py", "eval_only_runner")
                support_module = bridge.named(addon / "supports.py", "eval_only_supports")
                evaluator = bridge.named(addon / "evaluation.py", "eval_only_evaluation")
                saved, cp = checkpoint_io.load(torch, parent["path"], state, plan=plan,
                                              original_manifest=original, lineage=context["lineage"],
                                              journal_path=context["parent_journal"])
                inherited = saved["input_file_sha256"]
                runtime_output = output / "runtime"
                runtime_output.mkdir()
                runtime_args = bridge.runtime_arguments(source, seed=context["seed"], output=runtime_output)
                stage = "load_frozen_runtime"
                print("EVAL ONLY: loading original frozen frontend, banks and six saved heads", flush=True)
                with bindings.Runtime(runtime_args, runtime_output) as runtime:
                    bindings.original_train._formal_runtime(runtime, False)
                    train_names, val_names = bindings.original_train._names(runtime, False)
                    if train_names != original["train_names"] or val_names != original["val_names"]:
                        raise ValueError("Live split identities differ")
                    if json_value(runtime.protocol_info) != original["protocol_info"]:
                        raise ValueError("Live execution protocol differs")
                    bindings.original_train._checked_verify(runtime)
                    factory = support_module.SupportFactory(runtime, seed=context["seed"])
                    heads, optimizers, _ = runner.model_copies(torch, bindings.original_train, state,
                                                              ARMS, vars(runtime_args), runtime.device)
                    status["restoration"] = restore_training_state(torch, heads, optimizers, saved, state)
                    status["rng_restoration"] = validate_restored_rng(state, saved["rng"], require_cuda=True)
                    del saved
                    before_optimizers = {a: state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()}
                    before_heads = {a: state.state_fingerprint(h.state_dict()) for a, h in heads.items()}
                    before_rng = state.state_fingerprint(state.capture_rng())
                    require_scope(runtime, inherited)
                    provenance = dict(source_root=str(root), source_training_complete=False,
                                      checkpoint_sha256=cp["sha256"], checkpoint_step=1500,
                                      trainer_sha256=TRAINER_SHA256,
                                      evaluation_source_sha256=sha256(addon / "evaluation.py"),
                                      lineage=json_value(context["lineage"]),
                                      evaluation_only=True, optimizer_updates=0,
                                      historical_execution_replayed=False)
                    write_json(output / "manifest.json", dict(format=FORMAT, **provenance,
                               arms=list(ARMS), train_names=train_names, val_names=val_names,
                               runtime_config=vars(runtime_args), protocol_info=json_value(runtime.protocol_info)))
                    stage = "evaluate_val_1500"
                    status.update(phase=stage, parent_checkpoint=json_value(parent))
                    write_json(output / "summary.json", status)
                    print("EVAL ONLY: beginning all six arms + Grid on Val450", flush=True)
                    folder = output / "val_step_001500"
                    with (output / "markers.jsonl").open("x", encoding="utf-8") as markers:
                        with observed_evaluation(torch, runtime, evaluator, val_names, output, markers) as observed_state:
                            report = evaluator.evaluate(runtime, heads, val_names, folder, step=1500,
                                                        split="val", seed=context["seed"],
                                                        original_train=bindings.original_train,
                                                        legacy=bindings.legacy, factory=factory,
                                                        primary_endpoint=False, smoke=False)
                    stage = "verify_evaluation_and_runtime"
                    evidence.validate_report(report, step=1500, split="val", names=val_names,
                                             arms=list(ARMS), primary=False)
                    evidence.validate_saved_report(folder / "result.json", report, val_names, sha256, output)
                    status["runtime_integrity"] = json_value(bindings.original_train._checked_verify(runtime))
                    require_scope(runtime, inherited)
                    # Re-hash files actually used by this standalone evaluation.
                    # Unused historical Train image files are not claimed as reverified.
                    verify_files({str(p): h for p, h in runtime.input_ledger.items()})
                    checks = dict(runtime_verified=True, source_verified=False,
                                  head_state_unchanged=before_heads == {a: state.state_fingerprint(h.state_dict()) for a, h in heads.items()},
                                  optimizer_state_unchanged=before_optimizers == {a: state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()},
                                  rng_restored=before_rng == state.state_fingerprint(state.capture_rng()),
                                  guard_restored=False, checkpoint_and_parent_unchanged=False,
                                  evaluation_inputs_match_checkpoint=True)
                stage = "finalize_standalone_evaluation"
                guard = bindings.original_train._restored_guard(runtime)
                if guard is None or guard.get("failures"):
                    raise ValueError("PoolGuard closure evidence missing or contains failures")
                checks["guard_restored"] = True
                bridge.verify_ledgers(source)
                bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed")
                checks["source_verified"] = True
                verify_files(context["evidence_files"])
                checks["checkpoint_and_parent_unchanged"] = True
                finalized = finalize_fresh_report(folder / "result.json", report, output=output,
                                                 names=val_names, evidence=evidence, checks=checks,
                                                 provenance=provenance)
                status.update(phase="standalone_evaluation_complete", complete=True,
                              full_validation_evidence=True, final_integrity_pending=False,
                              image_count=450, result=str(folder / "result.json"),
                              metrics=concise_metrics(finalized), integrity_checks=checks)
    except BaseException as exc:
        # Do not inspect GPU tensors after an exception; validation-specific
        # evidence, when safe, has already been captured by the scoped observer.
        status.update(phase="failed", complete=False, full_validation_evidence=False,
                      final_integrity_pending=True, failure_phase=stage,
                      error=type(exc).__name__ + ": " + str(exc), last_marker=observed_state)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        failure = output / "probability_failure.json"
        if failure.is_file():
            status["probability_failure"] = read_json(failure)
    write_json(output / "summary.json", status)
    print("BEGIN CHECKPOINT VAL450 SUMMARY", flush=True)
    # Detailed restoration fingerprints remain in summary.json.
    print(json.dumps({k: v for k, v in status.items() if k not in ("restoration", "rng_restoration")},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print("END CHECKPOINT VAL450 SUMMARY", flush=True)
    return 0 if status["complete"] else 2


def launch(args):
    root = Path(args.root).resolve()
    locate(root, DEFAULT_IMAGE)
    if sha256(root / "trainer.pyz") != TRAINER_SHA256:
        raise ValueError("Pinned trainer differs from the tested helper archive")
    archive = Path(sys.argv[0]).resolve()
    if not zipfile.is_zipfile(archive):
        raise ValueError("Launch the built pyz archive")
    base = output_base(root)
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="eval1500_", dir=base))
    pinned = output / "evaluator.pyz"
    shutil.copyfile(archive, pinned)
    write_json(output / "launch.json", dict(format=FORMAT, source_root=str(root),
                                           evaluator_sha256=sha256(pinned)))
    write_json(output / "summary.json", dict(format=FORMAT, phase="starting", complete=False,
                                           optimizer_updates=0, training_performed=False, output=str(output)))
    log = output / "evaluate.log"
    with log.open("xb") as stream:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(pinned), "--worker",
                                    "--root", str(root), "--output", str(output)],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=output, start_new_session=True)
    print("已创建后台 Val450 评估进程；本次不训练。进程启动不代表评估完成。", flush=True)
    print("PID=" + str(process.pid), flush=True)
    print("OUTPUT=" + str(output), flush=True)
    print('查看日志：tail -n 80 -f "' + str(log) + '"', flush=True)
    print('读取结果：cat "' + str(output / "summary.json") + '"', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        return worker(args.root, args.output) if args.worker else launch(args)
    except BaseException as exc:
        print("EVAL ONLY STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
