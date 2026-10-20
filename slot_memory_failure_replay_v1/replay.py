#!/usr/bin/env python3
"""Bounded diagnostic updates from a verified checkpoint; preserves the failed run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from inspect_failure import (
    DEFAULT_RUN, checkpoint_info, read_journal, read_object, sha256, validate_identity,
)
from probe import (
    DEFAULT_OUTPUT, imported_file, json_value, read_bridge, validate_checkpoint,
)

FORMAT = "BRIGHT_SLOT_MEMORY_FAILURE_REPLAY_V1"
LIMIT = 500


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def preflight(args):
    root = Path(args.run_root).expanduser().resolve()
    plan, manifest = validate_identity(root, args.seed)
    train = root / ("seed_%s/train" % args.seed)
    status = read_object(train / "summary.json")
    if status.get("phase") != "failed" or status.get("complete") is not False:
        raise ValueError("Expected an already failed run")
    journal = read_journal(train / "train.jsonl", tuple(plan["arms"]))
    last = journal["contiguous_prefix_step"]
    if (journal["issue_count"] or journal["trailing_incomplete"]
            or journal["stable_during_read"] is not True or not last):
        raise ValueError("Original journal is not a stable complete contiguous prefix")
    if status.get("updates_per_arm") != {a: last for a in plan["arms"]}:
        raise ValueError("Original summary and journal update counts differ")
    options = [c for c in checkpoint_info(train / "checkpoints", tuple(plan["arms"]))
               if c["verified"] and c["step"] <= last]
    if not options:
        raise ValueError("No verified parent checkpoint")
    chosen = max(options, key=lambda c: c["step"])
    if not 0 < last + 1 - chosen["step"] <= LIMIT:
        raise ValueError("Diagnostic is limited to at most 500 new updates per arm")
    source = plan["source"]
    addon = (Path(args.addon_dir) if args.addon_dir else
             Path(source["config"]["repo"]) / "New/slot_memory_ablation_v1").resolve()
    bridge = read_bridge(addon)
    if bridge.check_package(addon) != plan["addon_release_sha256"]:
        raise ValueError("Original addon differs from the failed run")
    bridge.verify_ledgers(source)
    lineage = dict(
        run_root=str(root), seed=args.seed, parent_step=chosen["step"],
        parent_checkpoint_sha256=chosen["sha256"],
        original_journal_sha256=sha256(train / "train.jsonl"),
        original_manifest_sha256=sha256(train / "manifest.json"),
        last_original_complete_step=last, candidate_step=last + 1,
    )
    return root, train, plan, manifest, addon, bridge, chosen, lineage


def new_output(base, protected):
    base = Path(base).expanduser().resolve()
    for path in protected:
        path = Path(path).resolve()
        if base == path or path in base.parents:
            raise ValueError("Output must be outside the original run, addon, source and banks")
    base.mkdir(parents=True, exist_ok=True)
    folder = base / ("replay_%s_%s" % (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S"), os.getpid()))
    folder.mkdir()
    return folder


def protocol_check(current, recorded, dense):
    expected = json_value(recorded)
    before = expected["execution_settings"].get("cuda_launch_blocking")
    if dense:
        expected["execution_settings"]["cuda_launch_blocking"] = "1"
    if json_value(current) != expected:
        raise ValueError("Live numeric/source protocol differs beyond declared blocking setting")
    return dict(recorded=before, current="1" if dense else before)


def worker(args):
    # New processes are required after a device assertion. Set policy before Torch.
    root, train, plan, manifest, addon, bridge, chosen, lineage = preflight(args)
    output = Path(args.worker_output).resolve()
    if read_object(output / "request.json")["lineage"] != lineage:
        raise ValueError("Original inputs changed since supervisor preflight")
    dense = args.worker == "dense"
    blocking = ("1" if dense else
                manifest["protocol_info"]["execution_settings"].get("cuda_launch_blocking"))
    if blocking is None:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    from checkpoints import input_scope, save as save_checkpoint, load as load_diagnostic
    from observer import UpdateObserver
    from restore_state import restore_training_state
    from trajectory import TrajectoryComparison, reconstruct_order
    report = dict(format=FORMAT, phase="starting", diagnostic_only=True,
                  mode=args.worker, exact_failure_replay=False, test_accessed=False,
                  formal_experiment_complete=False, lineage=lineage,
                  CUDA_LAUNCH_BLOCKING=blocking, last_complete_step=None,
                  new_updates_per_arm={a: 0 for a in plan["arms"]},
                  checkpoints=[], output=str(output))
    write_json(output / "worker_summary.json", report)
    observer = None
    comparison = TrajectoryComparison(plan["arms"])
    phase = "import_torch"
    with (output / "markers.jsonl").open("x") as markers:
        def emit(event, **fields):
            record = dict(event=event, worker_phase=phase, **fields)
            markers.write(json.dumps(record, allow_nan=False) + "\n")
            markers.flush()
            if event == "nll_saved_target" and (
                    fields.get("stage") == "before_backward"
                    or fields.get("invalid_count") or fields.get("target_changed")):
                short = {k: fields.get(k) for k in
                         ("arm", "step", "stage", "classes", "invalid_count", "target_changed")}
                short["target_sha256"] = fields.get("target", {}).get("sha256")
                print("REPLAY NLL " + json.dumps(short), flush=True)

        try:
            import torch
            policy = read_object(plan["source"]["config"]["protocol"])["numeric_policy"]
            torch.set_num_threads(int(policy["num_threads"]))
            if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
                torch.set_num_interop_threads(int(policy["num_interop_threads"]))
            candidate, order, old_rows = reconstruct_order(
                torch, manifest, plan, train / "train.jsonl")
            report["candidate"] = candidate
            print("REPLAY CANDIDATE " + json.dumps(candidate), flush=True)
            if not torch.cuda.is_available():
                raise RuntimeError("This diagnostic requires the original CUDA environment")
            phase = "load_bindings"
            bindings = bridge.load_runtime(plan["source"])
            state = bindings.legacy.state
            # The original runner imports runtime_bridge by bare name.
            sys.modules["runtime_bridge"] = bridge
            runner = bridge.named(addon / "ablation_runner.py", "failure_replay_runner")
            support_module = bridge.named(addon / "supports.py", "failure_replay_supports")
            checkpoint = train / "checkpoints" / chosen["name"]
            if sha256(checkpoint) != chosen["sha256"]:
                raise ValueError("Parent checkpoint changed")
            if dense:
                parent = output.parent / "primary"
                saved, used_checkpoint = load_diagnostic(
                    torch, parent / "checkpoints" / (
                        "diagnostic_%06d.pt" % candidate["last_complete_step"]),
                    state, plan=plan, manifest=manifest, lineage=lineage,
                    journal_path=parent / "replay.jsonl",
                    expected_step=candidate["last_complete_step"])
            else:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                validate_checkpoint(saved, read_object(checkpoint.with_suffix(".pt.json")),
                                    plan, manifest, state.state_fingerprint)
                used_checkpoint = chosen
            start, end = saved["step"] + 1, candidate["candidate_step"]
            report.update(restored_step=saved["step"], last_complete_step=saved["step"],
                          used_checkpoint=used_checkpoint, endpoint_step=end)
            inherited = saved["input_file_sha256"]
            runtime_output = output / "runtime"
            runtime_output.mkdir()
            runtime_args = bridge.runtime_arguments(plan["source"], seed=args.seed, output=runtime_output)
            phase = "enter_frozen_runtime"
            with bindings.Runtime(runtime_args, runtime_output) as runtime:
                bindings.original_train._formal_runtime(runtime, False)
                names, val_names = bindings.original_train._names(runtime, False)
                if names != manifest["train_names"] or val_names != manifest["val_names"]:
                    raise ValueError("Runtime splits/order differ")
                report["blocking_policy"] = protocol_check(
                    runtime.protocol_info, manifest["protocol_info"], dense)
                bindings.original_train._checked_verify(runtime)
                report["actual_bindings"] = dict(
                    update=imported_file(runner.update_arms),
                    nll=imported_file(runtime.deps.region_nll),
                    lovasz=imported_file(runtime.deps.lovasz_softmax),
                    loader=imported_file(runtime.deps.full_image_loader),
                    parse_batch=imported_file(runtime.deps.parse_batch))
                factory = support_module.SupportFactory(runtime, seed=args.seed)
                heads, optimizers, _ = runner.model_copies(
                    torch, bindings.original_train, state, tuple(plan["arms"]),
                    vars(runtime_args), runtime.device)
                phase = "restore_training_state"
                report["restoration"] = restore_training_state(
                    torch, heads, optimizers, saved, state)
                # Same original inference transforms, fixed verified training order.
                iterator = iter(runtime.evaluation_batches(
                    order[start - 1:end], seed=args.seed))
                scope = input_scope(runtime, inherited)
                report["verified_input_count"] = len(scope)
                state.restore_rng(saved["rng"], require_cuda=True)
                expected_rng = state.state_fingerprint(saved["rng"])
                if state.state_fingerprint(state.capture_rng()) != expected_rng:
                    raise ValueError("Restored RNG differs")
                report["restored_rng_fingerprint"] = expected_rng
                write_json(output / "worker_summary.json", report)
                del saved
                observer = UpdateObserver(
                    torch, runtime, factory, heads, optimizers, emit, dense=dense)
                with (output / "replay.jsonl").open("x") as journal, observer:
                    for step in range(start, end + 1):
                        phase = "load_and_prepare_%s" % step
                        observer.set_step(step)
                        emit("step_begin", step=step, expected_image=order[step - 1])
                        batch = next(iterator)
                        example = runtime.prepare(batch)
                        bindings.original_train._assert_example(example)
                        if example.image != order[step - 1]:
                            raise ValueError("Loader order differs at step %s" % step)
                        phase = "original_update_%s" % step
                        rows = runner.update_arms(
                            heads, optimizers, example, runtime, factory, state,
                            step=step, multiplier=plan["post500_lr_multiplier"])
                        if step in old_rows:
                            comparison.observe(step, example.image, rows, old_rows[step])
                        record = dict(step=step, epoch=0, epoch_seed=args.seed,
                                      image=example.image, event=example.event, arms=rows)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        report["last_complete_step"] = step
                        report["new_updates_per_arm"] = {a: step - start + 1 for a in heads}
                        if step % 20 == 0 or step >= end - 1:
                            report["comparison"] = comparison.summary()
                            write_json(output / "worker_summary.json", report)
                            print("REPLAY PROGRESS " + json.dumps(dict(
                                step=step, image=example.image, endpoint=end,
                                mode=args.worker)), flush=True)
                        if step in (end - 1, end):
                            phase = "save_complete_diagnostic_checkpoint_%s" % step
                            emit("checkpoint_timing_intervention", step=step)
                            report["checkpoints"].append(save_checkpoint(
                                torch, output / "checkpoints", heads, optimizers, state,
                                runtime, factory, step=step, plan=plan, manifest=manifest,
                                inherited=inherited, lineage=lineage,
                                journal_path=output / "replay.jsonl", write_json=write_json))
                            write_json(output / "worker_summary.json", report)
                        del rows, record, example, batch
                phase = "final_integrity"
                bindings.original_train._checked_verify(runtime)
                input_scope(runtime, inherited)
                bridge.verify_ledgers(plan["source"])
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed during replay")
                if sha256(train / "train.jsonl") != lineage["original_journal_sha256"]:
                    raise ValueError("Original journal changed during replay")
            guard = getattr(runtime, "guard_report", {})
            if guard.get("failures") or guard.get("restore_conflicts") or guard.get("restored") is not True:
                raise RuntimeError("Pool guard closure failed")
            report.update(phase="bounded_replay_passed", pool_guard=guard,
                          conclusion="Failure not reproduced in this diagnostic branch; historical cause unresolved.")
        except BaseException as exc:
            # Device assertions poison CUDA: only cached CPU data and plain files here.
            report.update(phase="worker_failed", failure_phase=phase,
                          error=type(exc).__name__ + ": " + str(exc))
            if observer is not None:
                report["last_marker"] = observer.last_marker
                report["optimizer_steps_returned_in_failed_step"] = list(observer.completed_arms)
                trace = observer.failed_trace
                if trace is not None:
                    try:
                        case = trace.save_failure(output / "nll_failure_inputs.pt")
                        if case:
                            report["cpu_nll_case"] = dict(path=str(case), sha256=sha256(case))
                    except Exception as save_error:
                        report["cpu_nll_case_error"] = repr(save_error)
            (output / "traceback.txt").write_text(traceback.format_exc())
        report["comparison"] = comparison.summary()
        write_json(output / "worker_summary.json", report)
    print("REPLAY WORKER " + json.dumps(dict(
        mode=args.worker, phase=report["phase"], last_complete_step=report["last_complete_step"],
        error=report.get("error"), output=str(output))), flush=True)
    return 0 if report["phase"] == "bounded_replay_passed" else 1


def supervise(args):
    root, train, plan, manifest, addon, bridge, chosen, lineage = preflight(args)
    source = plan["source"]
    output = new_output(args.output_base, [
        root, addon, source["package"], source["source_run"],
        source["config"]["bank_root"], source["config"]["values"]])
    print("DIAGNOSTIC OUTPUT " + str(output), flush=True)
    request = dict(format=FORMAT, lineage=lineage, bounded_updates=True,
                   checkpoint_save_changes_timing=True,
                   evaluation_cache_and_allocator_history_not_reconstructed=True,
                   full_validation_not_repeated=True, exact_failure_replay=False)
    write_json(output / "request.json", request)

    def launch(mode):
        folder = output / mode
        folder.mkdir()
        write_json(folder / "request.json", request)
        command = [sys.executable, str(Path(sys.argv[0]).resolve()),
                   "--run-root", str(root), "--seed", str(args.seed),
                   "--addon-dir", str(addon), "--worker", mode,
                   "--worker-output", str(folder)]
        result = subprocess.run(command, check=False)
        path = folder / "worker_summary.json"
        summary = read_object(path) if path.exists() else dict(
            phase="worker_failed_before_summary", output=str(folder))
        summary["process_returncode"] = result.returncode
        return summary

    primary = launch("primary")
    primary_passed = (primary.get("phase") == "bounded_replay_passed"
                      and primary.get("process_returncode") == 0)
    dense = None
    last = lineage["last_original_complete_step"]
    cp = output / "primary/checkpoints" / ("diagnostic_%06d.pt" % last)
    if (not primary_passed
            and primary.get("last_complete_step") == last and cp.is_file()
            and cp.with_suffix(".pt.json").is_file()):
        print("Starting a NEW blocking=1 process from the complete diagnostic checkpoint.", flush=True)
        dense = launch("dense")
    summary = dict(request, output=str(output), primary=primary, dense=dense,
                   original_run_unmodified_by_this_tool=True,
                   conclusion=(
                       "No reproduction in bounded replay; the original failure remains unresolved."
                       if primary_passed else
                       "Primary replay failed; inspect saved stage markers and fresh-process result."))
    write_json(output / "summary.json", summary)
    print("BEGIN ABLATION REPLAY SUMMARY", flush=True)
    compact = {k: summary[k] for k in ("format", "output", "lineage", "conclusion")}
    for mode, result in (("primary", primary), ("dense", dense)):
        compact[mode] = None if result is None else {
            k: result.get(k) for k in (
                "phase", "process_returncode", "last_complete_step", "failure_phase",
                "error", "last_marker", "new_updates_per_arm", "comparison", "cpu_nll_case")}
        if result is not None and result.get("comparison") is not None:
            compact[mode]["comparison"] = {
                key: result["comparison"].get(key) for key in (
                    "steps_compared", "numeric_comparisons", "exact_matches",
                    "outside_tolerance", "first_outside_tolerance", "max_absolute_difference")}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print("END ABLATION REPLAY SUMMARY", flush=True)
    return 0 if primary_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--addon-dir")
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=("primary", "dense"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return worker(args) if args.worker else supervise(args)
    except Exception as exc:
        print("REPLAY PREFLIGHT STOPPED: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
