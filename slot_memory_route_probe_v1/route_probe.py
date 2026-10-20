#!/usr/bin/env python3
"""Observe one shared-route update in an isolated diagnostic directory."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import sys
import traceback

from inspect_failure import checkpoint_info, read_journal, read_object, sha256, validate_identity
from probe import DEFAULT_OUTPUT, imported_file, json_value, read_bridge, validate_checkpoint

FORMAT = "BRIGHT_SLOT_MEMORY_ROUTE_PROBE_V1"
DEFAULT_REPLAY = (
    "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/"
    "replay_20261020_061300_3928788"
)
ROUTE_ERROR = "route weights must be nonnegative and sum to one"
ARMS = ("e", "f", "b", "d", "r", "l")


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def preflight(args):
    replay = Path(args.replay_output).expanduser().resolve()
    summary_file = replay / "primary/worker_summary.json"
    failure = read_object(summary_file)
    if (failure.get("format") != "BRIGHT_SLOT_MEMORY_FAILURE_REPLAY_V1"
            or failure.get("phase") != "worker_failed"
            or failure.get("error") != "ValueError: " + ROUTE_ERROR):
        raise ValueError("Expected the failed route-validation replay worker")
    lineage = failure["lineage"]
    root = Path(lineage["run_root"]).resolve()
    seed = lineage["seed"]
    plan, manifest = validate_identity(root, seed)
    if tuple(plan["arms"]) != ARMS:
        raise ValueError("This probe requires the original six-arm order")
    train = root / ("seed_%s/train" % seed)
    status = read_object(train / "summary.json")
    journal = read_journal(train / "train.jsonl", ARMS)
    last = journal["contiguous_prefix_step"]
    if (status.get("phase") != "failed" or status.get("complete") is not False
            or journal["issue_count"] or journal["trailing_incomplete"]
            or journal["stable_during_read"] is not True
            or status.get("updates_per_arm") != {a: last for a in ARMS}
            or last != lineage["last_original_complete_step"]):
        raise ValueError("Original failed run/journal identity differs")
    if (sha256(train / "train.jsonl") != lineage["original_journal_sha256"]
            or sha256(train / "manifest.json") != lineage["original_manifest_sha256"]):
        raise ValueError("Original journal or manifest changed since replay")
    marker = failure["last_marker"]
    historical_step = marker["step"]
    if (type(historical_step) is not int
            or not lineage["parent_step"] < historical_step <= last
            or failure["last_complete_step"] != historical_step - 1
            or marker.get("arm") != "b" or marker.get("phase") != "forward_begin"
            or marker.get("completed_arms") != ["e", "f"]
            or not isinstance(marker.get("image"), str)):
        raise ValueError("Expected shared parent-route failure on B after E/F")
    trace_file = replay / "primary/traceback.txt"
    if "parent_ids = validate_routes(q, ids" not in trace_file.read_text():
        raise ValueError("Replay traceback does not identify the parent q validator")
    options = [
        c for c in checkpoint_info(train / "checkpoints", ARMS)
        if c["verified"] and c["step"] == lineage["parent_step"]
        and c["sha256"] == lineage["parent_checkpoint_sha256"]
    ]
    if len(options) != 1:
        raise ValueError("Replay's exact original parent checkpoint is unavailable")
    source = plan["source"]
    addon = (Path(args.addon_dir) if args.addon_dir else
             Path(source["config"]["repo"]) / "New/slot_memory_ablation_v1").resolve()
    bridge = read_bridge(addon)
    if bridge.check_package(addon) != plan["addon_release_sha256"]:
        raise ValueError("Original addon differs from the failed run")
    bridge.verify_ledgers(source)
    return dict(
        replay=replay, root=root, train=train, seed=seed, plan=plan, manifest=manifest,
        addon=addon, bridge=bridge, chosen=options[0], lineage=lineage,
        historical_step=historical_step, image=marker["image"],
        evidence_files={str(p): sha256(p) for p in (
            summary_file, trace_file, train / "train.jsonl", train / "manifest.json",
            train / "checkpoints" / options[0]["name"],
            (train / "checkpoints" / options[0]["name"]).with_suffix(".pt.json"),
        )},
    )


def new_output(base, protected):
    base = Path(base).expanduser().resolve()
    for path in protected:
        path = Path(path).resolve()
        if base == path or path in base.parents:
            raise ValueError("Output must be outside original inputs and source packages")
    base.mkdir(parents=True, exist_ok=True)
    output = base / ("route_probe_%s_%s" % (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S"), os.getpid()))
    output.mkdir()
    return output


def input_scope(runtime, inherited):
    current = {str(p): digest for p, digest in runtime.input_ledger.items()}
    if not current or not inherited:
        raise ValueError("Missing inherited or current input ledger")
    for path, digest in current.items():
        if inherited.get(path) != digest:
            raise ValueError("Input differs from parent checkpoint: " + path)
    return len(current)


@contextlib.contextmanager
def watch_factory(factory, trace, watched, snapshot):
    """Watch the exact supports returned by the unchanged original factory."""
    missing = object()
    saved_attribute = factory.__dict__.get("prepare", missing)
    original = factory.prepare

    def prepare(example, arms):
        trace.set_context("factory_prepare", None)
        result = original(example, arms)
        for arm, support in result.items():
            for key in ("route_q", "route_ids"):
                watched["prepared_%s.%s" % (arm, key)] = support[key]
        trace.watch("single_route_q", result["e"]["route_q"])
        snapshot("after_factory_prepare", None)
        return result

    factory.prepare = prepare
    try:
        yield
    finally:
        if saved_attribute is missing:
            delattr(factory, "prepare")
        else:
            factory.prepare = saved_attribute


def run_probe(args):
    # The archive runs in a fresh process, with no CUDA imports in preflight.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    setup = preflight(args)
    root, train, replay = (setup[k] for k in ("root", "train", "replay"))
    plan, manifest, bridge = (setup[k] for k in ("plan", "manifest", "bridge"))
    addon, chosen, seed = (setup[k] for k in ("addon", "chosen", "seed"))
    source = plan["source"]
    output = new_output(args.output_base, [
        root, replay, addon, source["package"], source["source_run"],
        source["config"]["bank_root"], source["config"]["values"],
    ])
    print("ROUTE PROBE OUTPUT " + str(output), flush=True)
    blocking = manifest["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
    if blocking is None:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    report = dict(
        format=FORMAT, phase="starting", diagnostic_only=True, test_accessed=False,
        output=str(output), lineage=setup["lineage"], source_replay=str(replay),
        historical_image_step=setup["historical_step"], image=setup["image"],
        restored_checkpoint_step=chosen["step"], diagnostic_optimizer_step=chosen["step"] + 1,
        exact_historical_state_claim=False, formal_experiment_complete=False,
        trained_checkpoint_saved=False, new_complete_updates_per_arm={a: 0 for a in plan["arms"]},
        observation_changes_timing=True, cuda_launch_blocking=blocking,
        intervention=(
            "One isolated original six-arm update on the historical failure image, "
            "from the original parent checkpoint. CPU snapshots synchronize CUDA. "
            "This does not reconstruct the historical failure state or execution timing."
        ),
    )
    write_json(output / "summary.json", report)
    trace = observer = None
    phase = "import_torch"
    storage_summary = dict(audits=0, errors=0, unsafe_alias_count=0, first_unsafe_alias=None)
    with (output / "observations.jsonl").open("x") as log:
        def emit(event, **fields):
            log.write(json.dumps(dict(event=event, driver_phase=phase, **fields),
                                 ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

        try:
            import torch
            from observer import UpdateObserver
            from restore_state import restore_training_state
            from route_trace import RouteTrace
            from storage_audit import audit_storage
            from trajectory import reconstruct_order

            policy = read_object(source["config"]["protocol"])["numeric_policy"]
            torch.set_num_threads(int(policy["num_threads"]))
            if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
                torch.set_num_interop_threads(int(policy["num_interop_threads"]))
            _, order, old_rows = reconstruct_order(torch, manifest, plan, train / "train.jsonl")
            historical_step, image = setup["historical_step"], setup["image"]
            if (order[historical_step - 1] != image
                    or old_rows[historical_step]["image"] != image):
                raise ValueError("Historical image differs from original journal and loader order")
            if not torch.cuda.is_available():
                raise RuntimeError("This diagnostic requires the original CUDA environment")
            phase = "load_bindings"
            bindings = bridge.load_runtime(source)
            state = bindings.legacy.state
            sys.modules["runtime_bridge"] = bridge
            runner = bridge.named(addon / "ablation_runner.py", "route_probe_runner")
            supports = bridge.named(addon / "supports.py", "route_probe_supports")
            checkpoint = train / "checkpoints" / chosen["name"]
            if sha256(checkpoint) != chosen["sha256"]:
                raise ValueError("Parent checkpoint changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            validate_checkpoint(saved, read_object(checkpoint.with_suffix(".pt.json")),
                                plan, manifest, state.state_fingerprint)
            if saved["step"] != chosen["step"]:
                raise ValueError("Parent checkpoint payload step differs")
            inherited = saved["input_file_sha256"]
            runtime_output = output / "runtime"
            runtime_output.mkdir()
            runtime_args = bridge.runtime_arguments(source, seed=seed, output=runtime_output)
            phase = "enter_frozen_runtime"
            with bindings.Runtime(runtime_args, runtime_output) as runtime:
                bindings.original_train._formal_runtime(runtime, False)
                names, val_names = bindings.original_train._names(runtime, False)
                if names != manifest["train_names"] or val_names != manifest["val_names"]:
                    raise ValueError("Runtime split order differs")
                if json_value(runtime.protocol_info) != manifest["protocol_info"]:
                    raise ValueError("Live protocol differs from the recorded original protocol")
                bindings.original_train._checked_verify(runtime)
                report["actual_bindings"] = dict(
                    update=imported_file(runner.update_arms),
                    model=imported_file(bindings.original_train.JointRegionMemoryClassifier.forward),
                    nll=imported_file(runtime.deps.region_nll),
                    lovasz=imported_file(runtime.deps.lovasz_softmax),
                    loader=imported_file(runtime.deps.full_image_loader),
                    parse_batch=imported_file(runtime.deps.parse_batch),
                )
                factory = supports.SupportFactory(runtime, seed=seed)
                heads, optimizers, _ = runner.model_copies(
                    torch, bindings.original_train, state, tuple(plan["arms"]),
                    vars(runtime_args), runtime.device)
                report["restoration"] = restore_training_state(
                    torch, heads, optimizers, saved, state)
                iterator = iter(runtime.evaluation_batches([image], seed=seed))
                report["verified_input_count"] = input_scope(runtime, inherited)
                state.restore_rng(saved["rng"], require_cuda=True)
                restored_rng = state.state_fingerprint(saved["rng"])
                if state.state_fingerprint(state.capture_rng()) != restored_rng:
                    raise ValueError("Restored RNG differs")
                report["restored_rng_fingerprint"] = restored_rng
                del saved
                watched = {}
                trace = RouteTrace(torch, type(next(iter(heads.values()))), emit)

                def snapshot(stage, arm=None):
                    trace.set_context(stage, arm)
                    trace.snapshot(stage, arm)
                    # Metadata-only audit errors must not mask an original validator error.
                    try:
                        audit = audit_storage(torch, watched, heads, optimizers)
                    except Exception as exc:
                        storage_summary["errors"] += 1
                        emit("storage_audit_error", stage=stage, arm=arm, error=repr(exc))
                    else:
                        storage_summary["audits"] += 1
                        aliases = audit["unsafe_aliases"]
                        storage_summary["unsafe_alias_count"] += len(aliases)
                        if aliases and storage_summary["first_unsafe_alias"] is None:
                            storage_summary["first_unsafe_alias"] = dict(
                                stage=stage, arm=arm, alias=aliases[0])
                        emit("storage_audit", stage=stage, arm=arm, **audit)

                def observe(event, **fields):
                    emit(event, **fields)
                    if event == "update_phase":
                        if fields["phase"] == "forward_begin":
                            print("ROUTE PROBE ARM " + str(fields.get("arm")), flush=True)
                        snapshot(fields["phase"], fields.get("arm"))

                with trace:
                    phase = "load_and_prepare_image"
                    trace.set_context(phase, None)
                    print("ROUTE PROBE IMAGE " + image, flush=True)
                    batch = next(iterator)
                    example = runtime.prepare(batch)
                    bindings.original_train._assert_example(example)
                    if example.image != image:
                        raise ValueError("Single-image loader returned a different image")
                    trace.watch("parent_q", example.common["q"])
                    soft = example.supports["b"]
                    trace.watch("soft_route_q", soft["route_q"])
                    watched.update(parent_q=example.common["q"], parent_ids=example.common["ids"],
                                   soft_route_q=soft["route_q"], soft_route_ids=soft["route_ids"])
                    snapshot("after_runtime_prepare")
                    observer = UpdateObserver(
                        torch, runtime, factory, heads, optimizers, observe, dense=False)
                    with observer, watch_factory(factory, trace, watched, snapshot):
                        observer.set_step(report["diagnostic_optimizer_step"])
                        phase = "isolated_original_update"
                        rows = runner.update_arms(
                            heads, optimizers, example, runtime, factory, state,
                            step=report["diagnostic_optimizer_step"],
                            multiplier=plan["post500_lr_multiplier"])
                        if (list(rows) != list(plan["arms"]) or
                                any(row["optimizer_step"] != report["diagnostic_optimizer_step"]
                                    for row in rows.values())):
                            raise ValueError("Incomplete isolated paired update")
                        report["new_complete_updates_per_arm"] = {a: 1 for a in heads}
                        write_json(output / "isolated_update.json", dict(
                            historical_image_step=historical_step, image=image,
                            diagnostic_optimizer_step=report["diagnostic_optimizer_step"],
                            arms=rows, diagnostic_only=True))
                        snapshot("after_complete_update")
                phase = "final_integrity"
                bindings.original_train._checked_verify(runtime)
                report["verified_input_count"] = input_scope(runtime, inherited)
                bridge.verify_ledgers(source)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed during probe")
                for path, digest in setup["evidence_files"].items():
                    if sha256(path) != digest:
                        raise ValueError("Original evidence changed during probe: " + path)
            guard = getattr(runtime, "guard_report", {})
            if (guard.get("failures") or guard.get("restore_conflicts")
                    or guard.get("restored") is not True):
                raise RuntimeError("Pool guard closure failed")
            report.update(phase="isolated_update_completed", final_input_integrity_verified=True)
        except BaseException as exc:
            # After any CUDA error, use only retained host evidence and ordinary files.
            report.update(phase="probe_failed", failure_phase=phase,
                          error=type(exc).__name__ + ": " + str(exc))
            (output / "traceback.txt").write_text(traceback.format_exc())
        if observer is not None:
            report["last_marker"] = observer.last_marker
            report["optimizer_steps_returned"] = list(observer.completed_arms)
        report["storage_audit"] = storage_summary
        if trace is not None:
            report["route_trace"] = trace.summary()
            # Save bounded retained CPU snapshots only when additional inspection is needed.
            stats = report["route_trace"]
            needs_case = (report["phase"] == "probe_failed"
                          or stats.get("watched_change_count", 0)
                          or stats.get("diagnostic_error_count", 0)
                          or storage_summary["unsafe_alias_count"])
            if needs_case:
                try:
                    case = output / "route_cpu_case.pt"
                    trace.save_cpu_case(case)
                    report["cpu_case"] = dict(path=str(case), sha256=sha256(case))
                except Exception as exc:
                    report["cpu_case_error"] = repr(exc)
        stats = report.get("route_trace", {})
        issues = (stats.get("diagnostic_error_count", 0)
                  or stats.get("watched_change_count", 0)
                  or stats.get("validator_failure_count", 0)
                  or storage_summary["errors"] or storage_summary["unsafe_alias_count"])
        if report["phase"] == "isolated_update_completed" and issues:
            report["phase"] = "isolated_update_completed_with_observations"
        report["conclusion"] = (
            "Inspect route_trace and stage evidence. This isolated update does not reproduce "
            "historical step state; a pass does not resolve the earlier route or NLL failure."
        )
        write_json(output / "summary.json", report)
    compact = {k: report.get(k) for k in (
        "format", "output", "phase", "image", "historical_image_step",
        "restored_checkpoint_step", "diagnostic_optimizer_step", "error", "failure_phase",
        "last_marker", "new_complete_updates_per_arm", "optimizer_steps_returned",
        "cpu_case", "cpu_case_error", "conclusion",
    )}
    compact["route_trace"] = {k: stats.get(k) for k in (
        "diagnostic_error_count", "validation_call_count", "validator_failure_count",
        "watched_change_count", "first_watch_change", "first_validation_failure",
    )}
    compact["storage_audit"] = storage_summary
    print("BEGIN ABLATION ROUTE PROBE SUMMARY", flush=True)
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print("END ABLATION ROUTE PROBE SUMMARY", flush=True)
    return 0 if report["phase"] == "isolated_update_completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-output", default=DEFAULT_REPLAY)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT)
    parser.add_argument("--addon-dir")
    args = parser.parse_args()
    try:
        return run_probe(args)
    except Exception as exc:
        print("ROUTE PROBE PREFLIGHT STOPPED: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
