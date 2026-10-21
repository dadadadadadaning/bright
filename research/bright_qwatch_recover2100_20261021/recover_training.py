#!/usr/bin/env python3
"""Continue original updates with declared parent-Q observations; separate evaluation processes."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from inspect_failure import DEFAULT_RUN, read_object, sha256
from probe import imported_file, json_value, validate_checkpoint
from resume_context import prepare, inactive_parent, protected_paths

FORMAT = "BRIGHT_SLOT_MEMORY_RESUME_V1"
DEFAULT_OUTPUT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1"


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_files(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError("Recorded input changed: " + str(path))


def prefix_bytes(path, step, arms, order, seed):
    """Copy only complete paired rows belonging to the restored state."""
    rows = []
    with Path(path).open("rb") as stream:
        for expected in range(1, step + 1):
            raw = stream.readline()
            if not raw.endswith(b"\n"):
                raise ValueError("Incomplete inherited journal prefix")
            row = json.loads(raw)
            if (type(row.get("step")) is not int or row["step"] != expected
                    or row.get("image") != order[expected - 1]
                    or row.get("epoch") != 0 or row.get("epoch_seed") != seed
                    or set(row.get("arms", {})) != set(arms)
                    or any(row["arms"][a].get("optimizer_step") != expected for a in arms)):
                raise ValueError("Inherited paired sample/update order differs at %s" % expected)
            rows.append(raw)
    return b"".join(rows)


def require_scope(runtime, inherited):
    current = {str(path): digest for path, digest in runtime.input_ledger.items()}
    if not current or not inherited:
        raise ValueError("Missing input ledger")
    for path, digest in current.items():
        if inherited.get(path) != digest:
            raise ValueError("Runtime input differs from restored checkpoint: " + path)
    return len(current)


def evaluations(plan, boundary, val_names, fixed_names):
    endpoint = plan["endpoint_step"]
    schedule = [
        (boundary, "val", val_names, boundary == endpoint),
        (boundary, "fixed_train", fixed_names, False),
    ]
    for step in plan["validation_steps"]:
        if boundary < step <= endpoint:
            schedule.append((step, "val", val_names, step == endpoint))
    if endpoint > boundary:
        schedule.append((endpoint, "fixed_train", fixed_names, False))
    if not any(step == endpoint and split == "val" for step, split, _, _ in schedule):
        raise ValueError("Original schedule lacks the primary endpoint")
    if len({(step, split) for step, split, _, _ in schedule}) != len(schedule):
        raise ValueError("Repeated evaluation schedule entry")
    return schedule


def run_training(context, output):
    """One worker, no automatic retries and no access to Test."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output = Path(output).resolve()
    plan, manifest, bridge = (context[k] for k in ("plan", "manifest", "bridge"))
    source, seed, addon = plan["source"], context["seed"], context["addon"]
    parent = context["parent"]
    start, endpoint = parent["step"], plan["endpoint_step"]
    arms = tuple(plan["arms"])
    blocking = manifest["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
    if blocking is None:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    started = time.monotonic()
    status = dict(
        format=FORMAT, complete=False, execution_complete=False, training_complete=False,
        full_validation_evidence=False, final_integrity_pending=True, phase="loading",
        smoke=False, test_accessed=False, pid=os.getpid(), seed=seed, arms=list(arms),
        restored_step=start, last_complete_step=start, endpoint_step=endpoint,
        schedule_horizon=plan["schedule_horizon"], updates_per_arm={a: start for a in arms},
        new_updates_per_arm={a: 0 for a in arms}, evaluations={}, output=str(output),
        lineage=context["lineage"], parent=json_value(parent),
        checkpoint_every=context["checkpoint_every"], historical_trajectory_claim=False,
        training_mode="checkpoint_continuation_in_new_directory",
        original_NLL_and_Lovasz_unchanged=True, cuda_launch_blocking=blocking,
        checkpoint_saves_change_timing=True,
        parent_q_observation=context.get("parent_q_observation"),
        inherited_initial_and_pre_boundary_evaluations_not_promoted=True,
    )
    write_json(output / "summary.json", status)
    observer = None
    phase = "verify_inputs"
    with (output / "markers.jsonl").open("x") as markers:
        def emit(event, **fields):
            markers.write(json.dumps(dict(event=event, worker_phase=phase, **fields),
                                     allow_nan=False) + "\n")
            markers.flush()

        try:
            verify_files(context["evidence_files"])
            import torch
            import checkpoint as checkpoint_io
            from observer import UpdateObserver
            from restore_state import restore_training_state, validate_restored_rng
            from trajectory import reconstruct_order

            policy = read_object(source["config"]["protocol"])["numeric_policy"]
            torch.set_num_threads(int(policy["num_threads"]))
            if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
                torch.set_num_interop_threads(int(policy["num_interop_threads"]))
            _, order, _ = reconstruct_order(
                torch, manifest, plan, context["train"] / "train.jsonl")
            if not 0 < start <= endpoint <= len(order):
                raise ValueError("This continuation must stay within the original first epoch")
            inherited_journal = prefix_bytes(
                context["parent_journal"], start, arms, order, seed)
            if not torch.cuda.is_available():
                raise RuntimeError("Formal BRIGHT continuation requires the original CUDA environment")
            phase = "load_bindings"
            bindings = bridge.load_runtime(source)
            state = bindings.legacy.state
            sys.modules["runtime_bridge"] = bridge
            # evaluation.py imports the original addon diagnostics by bare name.
            sys.path.insert(0, str(addon))
            runner = bridge.named(addon / "ablation_runner.py", "resume_runner")
            support_module = bridge.named(addon / "supports.py", "resume_supports")
            evaluation = bridge.named(addon / "evaluation.py", "resume_evaluation")
            path = Path(parent["path"])
            if sha256(path) != parent["sha256"]:
                raise ValueError("Selected checkpoint changed after preflight")
            if parent["kind"] == "original":
                saved = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(saved, read_object(path.with_suffix(".pt.json")),
                                    plan, manifest, state.state_fingerprint)
            elif parent["kind"] == "resume":
                saved, _ = checkpoint_io.load(
                    torch, path, state, plan=plan, original_manifest=manifest,
                    lineage=context["lineage"], journal_path=context["parent_journal"])
            else:
                raise ValueError("Diagnostic checkpoints cannot initialize formal continuation")
            if saved["step"] != start:
                raise ValueError("Restored checkpoint step differs from selected parent")
            inherited = saved["input_file_sha256"]
            print("RESUME verifying inherited input fingerprints:", len(inherited), flush=True)
            # The original shuffled loader hashed all Train inputs before training.
            # Reverify the entire inherited ledger before accepting its prefix.
            verify_files(inherited)
            status["inherited_inputs_all_reverified"] = True
            runtime_args = bridge.runtime_arguments(source, seed=seed, output=output)
            phase = "enter_frozen_runtime"
            with bindings.Runtime(runtime_args, output) as runtime:
                bindings.original_train._formal_runtime(runtime, False)
                train_names, val_names = bindings.original_train._names(runtime, False)
                if train_names != manifest["train_names"] or val_names != manifest["val_names"]:
                    raise ValueError("Runtime split identities/order changed")
                if json_value(runtime.protocol_info) != manifest["protocol_info"]:
                    raise ValueError("Live source/numeric protocol differs from the original run")
                bindings.original_train._checked_verify(runtime)
                factory = support_module.SupportFactory(runtime, seed=seed)
                heads, optimizers, _ = runner.model_copies(
                    torch, bindings.original_train, state, arms, vars(runtime_args), runtime.device)
                status["restoration"] = restore_training_state(
                    torch, heads, optimizers, saved, state)
                fixed_names = manifest["fixed_train_names"]
                expected = evaluations(plan, context.get("evaluation_boundary_step", start), val_names, fixed_names)
                # Exact verified first-epoch order, identical full-image inference transforms.
                iterator = (iter(runtime.evaluation_batches(order[start:endpoint], seed=seed))
                            if start < endpoint else iter(()))
                require_scope(runtime, inherited)
                status["rng_restoration"] = validate_restored_rng(
                    state, saved["rng"], require_cuda=True)
                del saved
                with (output / "train.jsonl").open("xb") as journal:
                    journal.write(inherited_journal)
                    journal.flush()
                    os.fsync(journal.fileno())
                del inherited_journal
                continuation_manifest = dict(
                    format=FORMAT, lineage=context["lineage"], original_manifest=manifest,
                    original_plan=plan, parent=json_value(parent), seed=seed, arms=list(arms),
                    resumed_from_step=start, endpoint_step=endpoint,
                    inherited_journal_prefix_steps=start,
                    inherited_journal_prefix_sha256=sha256(output / "train.jsonl"),
                    runtime_config=vars(runtime_args), protocol_info=runtime.protocol_info,
                    fixed_train_names=fixed_names, train_names=train_names, val_names=val_names,
                    checkpoint_every=context["checkpoint_every"],
                    fresh_initialization=False, historical_trajectory_claim=False,
                    test_accessed=False, evaluations_in_separate_processes=True,
                    boundary_evaluation_certificate=context.get("boundary_certificate"),
                    parent_q_observation=context.get("parent_q_observation"),
                    expected_evaluations=[
                        dict(step=s, split=split, image_count=len(names), primary=primary)
                        for s, split, names, primary in expected],
                    actual_bindings=dict(
                        update=imported_file(inspect.unwrap(runner.update_arms)),
                        update_observer_module="parent_q_observer (pinned trainer.pyz)",
                        evaluation=imported_file(evaluation.evaluate),
                        nll=imported_file(runtime.deps.region_nll),
                        lovasz=imported_file(runtime.deps.lovasz_softmax),
                        loader=imported_file(runtime.deps.full_image_loader)),
                )
                write_json(output / "manifest.json", continuation_manifest)
                write_json(output / "plan.json", plan)
                write_json(output / "support_provenance.json", factory.provenance())

                # Completed boundary Val is certified by the workflow preflight.
                # All remaining evaluations execute after this CUDA process exits.
                observer = UpdateObserver(
                    torch, runtime, factory, heads, optimizers, emit, dense=False)
                with (output / "train.jsonl").open("ab") as journal:
                    for step in range(start + 1, endpoint + 1):
                        phase = "prepare_%s" % step
                        status["phase"] = "training"
                        emit("step_begin", step=step, expected_image=order[step - 1])
                        example = runtime.prepare(next(iterator))
                        bindings.original_train._assert_example(example)
                        if example.image != order[step - 1]:
                            raise ValueError("Training sample order differs at step %s" % step)
                        # Restrict observation to updates; full validation uses its own
                        # no-grad lifecycle, so no stale update markers can leak into it.
                        with observer:
                            observer.set_step(step)
                            phase = "update_%s" % step
                            rows = runner.update_arms(
                                heads, optimizers, example, runtime, factory, state,
                                step=step, multiplier=plan["post500_lr_multiplier"])
                        record = dict(step=step, epoch=0, epoch_seed=seed,
                                      image=example.image, event=example.event, arms=rows)
                        journal.write((json.dumps(record, allow_nan=False) + "\n").encode())
                        journal.flush()
                        status.update(last_complete_step=step,
                                      updates_per_arm={a: step for a in arms},
                                      new_updates_per_arm={a: step - start for a in arms})
                        if step % 20 == 0 or step in (start + 1, endpoint):
                            write_json(output / "summary.json", status)
                            print("RESUME TRAIN " + json.dumps(dict(
                                step=step, endpoint=endpoint, image=example.image,
                                losses={a: row["original_loss"] for a, row in rows.items()})),
                                flush=True)
                        scheduled = [event for event in expected
                                     if event[0] == step and event[1] == "val"]
                        if step % context["checkpoint_every"] == 0 or scheduled or step == endpoint:
                            phase = "checkpoint_%s" % step
                            os.fsync(journal.fileno())
                            # Keep full source/input audits at the original evaluation
                            # boundaries. Frequent recovery checkpoints still verify
                            # all six model/optimizer states and the complete journal.
                            if scheduled or step == endpoint:
                                bindings.original_train._checked_verify(runtime)
                                bridge.verify_ledgers(source)
                            record_cp = checkpoint_io.save(
                                torch, output / "checkpoints", heads, optimizers, state,
                                runtime, factory, step=step, plan=plan,
                                original_manifest=manifest, inherited=inherited,
                                lineage=context["lineage"], journal_path=output / "train.jsonl",
                                write_json=write_json, inherited_all_reverified=True)
                            status["last_checkpoint"] = record_cp
                            write_json(output / "summary.json", status)
                            print("RESUME CHECKPOINT step=%s" % step, flush=True)
                        del example, rows, record
                        # The observer instance is one-shot; its hooks are removed
                        # at every paired-update boundary, before validation.
                        observer = UpdateObserver(
                            torch, runtime, factory, heads, optimizers, emit, dense=False)
                status["training_complete"] = True
                phase = "final_integrity"
                status["phase"] = phase
                write_json(output / "summary.json", status)
                status["integrity_checks"] = bindings.original_train._checked_verify(runtime)
                bridge.verify_ledgers(source)
                bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed during continuation")
                verify_files(context["evidence_files"])
                require_scope(runtime, inherited)
                write_json(output / "input_fingerprints.json", {
                    str(p): digest for p, digest in runtime.input_ledger.items()})
                write_json(output / "support_provenance.json", factory.provenance())
            guard = getattr(runtime, "guard_report", {})
            if (guard.get("failures") or guard.get("restore_conflicts")
                    or guard.get("restored") is not True):
                raise RuntimeError("Pool guard closure failed")
            status["pool_guard"] = guard
            status.update(
                complete=False, execution_complete=True, training_complete=True,
                full_validation_evidence=False, final_integrity_pending=False,
                phase="training_complete_evaluation_pending", error=None,
                remaining_evaluation_location="workflow_summary.json")
        except BaseException as exc:
            # A CUDA assertion can poison the device. No CUDA operations, tensor
            # serialization or attempted retry are allowed in this handler.
            status.update(
                complete=False, execution_complete=False, full_validation_evidence=False,
                final_integrity_pending=True, phase="failed", failure_phase=phase,
                error=type(exc).__name__ + ": " + str(exc))
            if observer is not None:
                status["last_marker"] = observer.last_marker
                status["optimizer_steps_returned_in_observed_step"] = list(observer.completed_arms)
            (output / "traceback.txt").write_text(traceback.format_exc())
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", status)
    print("RESUME RESULT " + json.dumps({k: status.get(k) for k in (
        "phase", "last_complete_step", "endpoint_step", "error", "failure_phase", "output"
    )}), flush=True)
    return 0 if status["training_complete"] and status["phase"] == "training_complete_evaluation_pending" else 1
