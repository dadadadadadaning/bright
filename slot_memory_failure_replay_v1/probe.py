#!/usr/bin/env python3
"""One candidate image, checkpoint weights, original loss; no optimizer updates."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

from inspect_failure import (
    DEFAULT_RUN, FORMAT, checkpoint_info, read_journal, read_object, sha256,
    validate_identity,
)

PROBE_FORMAT = "BRIGHT_SLOT_MEMORY_FAILURE_PROBE_V1"
DEFAULT_OUTPUT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS"


def candidate_from_journal(torch, manifest, plan, journal_path):
    """Match the complete first-epoch prefix before inferring the next name."""
    names = manifest["train_names"]
    seed = manifest["seed"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Train names must be nonempty and unique")
    checked = read_journal(journal_path, tuple(plan["arms"]))
    if (checked["issue_count"] or checked["trailing_incomplete"]
            or checked["stable_during_read"] is not True):
        raise ValueError("Journal is not a stable complete contiguous prefix")
    last = checked["contiguous_prefix_step"]
    if not 0 < last < len(names):
        raise ValueError("This diagnostic supports a nonempty unfinished first epoch only")
    # Match full_image_loader, including DataLoader's base-seed consumption.
    # A direct randperm(seed) would produce a different order.
    loader = torch.utils.data.DataLoader(
        list(names), batch_size=1, shuffle=True, num_workers=0, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    order = [str(batch[0]) for batch in loader]
    before = sha256(journal_path)
    count = 0
    with Path(journal_path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if (row["step"] != index + 1 or row.get("image") != order[index]
                    or row.get("epoch") != 0 or row.get("epoch_seed") != seed):
                raise ValueError("Reconstructed training order differs at step %s" % (index + 1))
            count += 1
    if count != last or sha256(journal_path) != before:
        raise ValueError("Journal changed during order verification")
    return dict(last_complete_step=last, candidate_step=last + 1, image=order[last],
                epoch=0, epoch_seed=seed, verified_prefix_count=count,
                journal_sha256=before, exact_failure_replay=False)


def json_value(value):
    return json.loads(json.dumps(value, default=str))


def validate_checkpoint(saved, sidecar, plan, manifest, state_fingerprint):
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError("Not an ablation checkpoint")
    if (type(saved.get("step")) is not int or saved["step"] != sidecar.get("step")
            or sidecar.get("roundtrip_verified") is not True):
        raise ValueError("Checkpoint step/roundtrip metadata differs")
    for key, expected in (
        ("study_plan", plan), ("runtime_config", manifest["runtime_config"]),
        ("protocol_info", manifest["protocol_info"]),
    ):
        if json_value(saved.get(key)) != expected:
            raise ValueError("Checkpoint differs from training manifest: " + key)
    heads = saved.get("heads")
    hashes = sidecar.get("head_hashes")
    if (not isinstance(heads, dict) or set(heads) != set(plan["arms"])
            or not isinstance(hashes, dict) or set(hashes) != set(plan["arms"])):
        raise ValueError("Checkpoint arm identities differ")
    for arm, head in heads.items():
        if state_fingerprint(head) != hashes[arm]:
            raise ValueError("Checkpoint head fingerprint differs: " + arm)


def imported_file(function):
    path = inspect.getsourcefile(function)
    return dict(module=function.__module__, name=function.__qualname__,
                path=path, sha256=sha256(path) if path else None)


def validate_live_protocol(current, recorded):
    current, expected = json_value(current), json_value(recorded)
    old_blocking = expected["execution_settings"].get("cuda_launch_blocking")
    expected["execution_settings"]["cuda_launch_blocking"] = "1"
    if current != expected:
        raise ValueError("Current source protocol differs beyond diagnostic CUDA_LAUNCH_BLOCKING")
    return dict(setting="CUDA_LAUNCH_BLOCKING", recorded=old_blocking, diagnostic="1")


def fresh_output(base, forbidden):
    base = Path(base).expanduser().resolve()
    for location in forbidden:
        location = Path(location).resolve()
        if base == location or location in base.parents:
            raise ValueError("Diagnostic output must be outside original run/source/addon: " + str(base))
    base.mkdir(parents=True, exist_ok=True)
    folder = base / ("probe_%s_%s" % (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S"), os.getpid()))
    folder.mkdir(exist_ok=False)
    return folder


def read_bridge(addon):
    path = addon / "runtime_bridge.py"
    spec = importlib.util.spec_from_file_location("_bright_failure_runtime_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(args):
    # Set before Torch/repository imports. Bytecode must not change hashed packages.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    root = Path(args.run_root).expanduser().resolve()
    plan, manifest = validate_identity(root, args.seed)
    train = root / ("seed_%s/train" % args.seed)
    original_status = read_object(train / "summary.json")
    if original_status.get("phase") != "failed" or original_status.get("complete") is not False:
        raise ValueError("Default diagnostic requires a recorded failed training run")
    source = plan["source"]
    addon = (
        Path(args.addon_dir).expanduser().resolve() if args.addon_dir
        else Path(source["config"]["repo"]) / "New/slot_memory_ablation_v1"
    )
    bridge = read_bridge(addon)
    if bridge.check_package(addon) != plan["addon_release_sha256"]:
        raise ValueError("Original addon differs from failed run; no GPU probe started")
    bridge.verify_ledgers(source)
    import torch
    # Set recorded thread policy before the sampler performs any Torch work.
    # NewRuntime applies this same policy when it later loads the frozen source.
    policy = read_object(source["config"]["protocol"])["numeric_policy"]
    torch.set_num_threads(int(policy["num_threads"]))
    if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
        torch.set_num_interop_threads(int(policy["num_interop_threads"]))
    candidate = candidate_from_journal(torch, manifest, plan, train / "train.jsonl")
    print("CANDIDATE " + json.dumps(candidate, ensure_ascii=False), flush=True)
    if args.locate_only:
        print("LOCATE ONLY: matched the entire recorded prefix; no CUDA runtime loaded.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; a CPU test cannot reproduce this GPU failure")
    available = [
        item for item in checkpoint_info(train / "checkpoints", tuple(plan["arms"]))
        if item["verified"] and item["step"] <= candidate["last_complete_step"]
    ]
    if not available:
        raise ValueError("No checkpoint with matching sidecar and completed journal prefix")
    chosen = max(available, key=lambda item: item["step"])
    checkpoint = train / "checkpoints" / chosen["name"]
    sidecar = read_object(checkpoint.with_suffix(".pt.json"))
    protected = [root, addon, source["package"], source["source_run"],
                 source["config"]["bank_root"], source["config"]["values"]]
    output = fresh_output(args.output_base, protected)
    print("DIAGNOSTIC OUTPUT " + str(output), flush=True)
    # These are evidence files only: never a training/completion manifest.
    report = dict(format=PROBE_FORMAT, phase="starting", root=str(root),
                  candidate=candidate, checkpoint=chosen, output=str(output),
                  optimizer_updates=0, exact_failure_replay=False,
                  CUDA_LAUNCH_BLOCKING="1", arms={}, active_arm=None)
    bridge.write_json(output / "probe_summary.json", report)
    phase = "load_runtime_bindings"
    active_arm = None
    trace = None
    with (output / "trace.jsonl").open("x", encoding="utf-8") as journal:
        def emit(event, **fields):
            line = json.dumps(dict(event=event, phase=phase, arm=active_arm, **fields),
                              ensure_ascii=False, allow_nan=False)
            journal.write(line + "\n")
            journal.flush()
            # Detailed snapshots stay in the artifact; console stays short.
            if event == "nll_saved_target":
                target = fields["target"]
                report["last_nll_target"] = dict(
                    arm=active_arm, stage=fields["stage"], classes=fields["classes"],
                    invalid_count=fields["invalid_count"], target_changed=fields["target_changed"],
                    version_changed=fields["version_changed"], shape=target["shape"],
                    dtype=target["dtype"], minimum=target["minimum"], maximum=target["maximum"],
                    sha256=target["sha256"],
                )
                if fields["stage"] == "before_backward" or fields["target_changed"] or fields["invalid_count"]:
                    print("PROBE NLL " + json.dumps(report["last_nll_target"]), flush=True)
            else:
                print("PROBE " + line, flush=True)

        try:
            emit("start", python=sys.version.split()[0], torch=torch.__version__,
                 cuda_build=torch.version.cuda, candidate=candidate,
                 checkpoint_step=chosen["step"], no_optimizer_updates=True)
            bindings = bridge.load_runtime(source)
            state = bindings.legacy.state
            phase = "load_checkpoint_cpu"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            validate_checkpoint(saved, sidecar, plan, manifest, state.state_fingerprint)
            supports_module = bridge.named(addon / "supports.py", "failure_probe_supports")
            (output / "runtime").mkdir()
            runtime_args = bridge.runtime_arguments(source, seed=args.seed, output=output / "runtime")
            phase = "enter_frozen_runtime"
            emit("loading_frozen_runtime")
            from nll_trace import NLLTrace, label_snapshot
            with bindings.Runtime(runtime_args, output / "runtime") as runtime:
                bindings.original_train._formal_runtime(runtime, False)
                names, val_names = bindings.original_train._names(runtime, False)
                if names != manifest["train_names"] or val_names != manifest["val_names"]:
                    raise ValueError("Current runtime split/order differs from failed training")
                intervention = validate_live_protocol(runtime.protocol_info, manifest["protocol_info"])
                report["declared_timing_intervention"] = intervention
                emit("declared_timing_intervention", **intervention)
                bindings.original_train._checked_verify(runtime)
                emit("actual_bindings", nll=imported_file(runtime.deps.region_nll),
                     lovasz=imported_file(runtime.deps.lovasz_softmax),
                     loader=imported_file(runtime.deps.full_image_loader),
                     parse_batch=imported_file(runtime.deps.parse_batch))
                phase = "load_candidate"
                batch = next(iter(runtime.batches(
                    [candidate["image"]], train=True, seed=candidate["epoch_seed"]
                )))
                emit("raw_damage_labels", labels=label_snapshot(torch, torch.as_tensor(batch[3])))
                phase = "frozen_prepare"
                example = runtime.prepare(batch)
                torch.cuda.synchronize(runtime.device)
                bindings.original_train._assert_example(example)
                if example.image != candidate["image"]:
                    raise ValueError("Prepared a different image")
                emit("prepared_damage_labels", labels=label_snapshot(torch, example.target))
                # Check every input actually used by this probe against the saved ledger.
                old_inputs = saved.get("input_file_sha256", {})
                for path, digest in runtime.input_ledger.items():
                    if old_inputs.get(str(path)) != digest:
                        raise ValueError("Probe input differs from checkpoint: " + str(path))
                phase = "prepare_supports"
                factory = supports_module.SupportFactory(runtime, seed=args.seed)
                supports = factory.prepare(example, tuple(plan["arms"]))
                torch.cuda.synchronize(runtime.device)
                for active_arm in plan["arms"]:
                    trace = None
                    phase = "load_head"
                    report["active_arm"] = active_arm
                    bridge.write_json(output / "probe_summary.json", report)
                    head = bindings.original_train.JointRegionMemoryClassifier(
                        chunk_size=int(runtime_args.chunk_size)
                    ).to(runtime.device)
                    head.load_state_dict(saved["heads"][active_arm], strict=True)
                    expected_head = sidecar["head_hashes"][active_arm]
                    if state.state_fingerprint(head.state_dict()) != expected_head:
                        raise ValueError("Loaded head differs: " + active_arm)
                    head.train()
                    head.zero_grad(set_to_none=True)
                    # Recorded RNG belongs to checkpoint1000, not the failed step.
                    state.restore_rng(saved["rng"], require_cuda=True)
                    phase = "head_forward"
                    emit("arm_begin", image=example.image, checkpoint_step=chosen["step"])
                    result = factory.forward(head, example, active_arm, supports[active_arm])
                    torch.cuda.synchronize(runtime.device)
                    probability = result["probabilities"]
                    if not bool(torch.isfinite(probability).all()):
                        raise ValueError("Nonfinite probabilities")
                    emit("forward_complete", shape=list(probability.shape),
                         minimum=float(probability.min()), maximum=float(probability.max()),
                         probability_sum_min=float(probability.sum(1).min()),
                         probability_sum_max=float(probability.sum(1).max()))
                    trace = NLLTrace(torch, runtime.deps, emit)
                    with trace:
                        phase = "original_loss"
                        loss = runtime.loss(probability, example.target)
                        torch.cuda.synchronize(runtime.device)
                        if loss.ndim or not bool(torch.isfinite(loss)):
                            raise ValueError("Nonfinite/nonscalar original loss")
                        trace.snapshot("after_original_loss")
                        phase = "logging_nll"
                        with torch.no_grad():
                            nll = (runtime.deps.region_nll(probability, example.target)
                                   if bool((example.target != 255).any()) else loss.detach() * 0)
                        torch.cuda.synchronize(runtime.device)
                        trace.snapshot("after_logging_nll")
                        phase = "before_backward"
                        trace.snapshot("before_backward")
                        emit("loss_ready", original_loss=float(loss.detach()), nll=float(nll))
                        phase = "backward"
                        loss.backward()
                        torch.cuda.synchronize(runtime.device)
                    phase = "verify_gradients_and_unchanged_head"
                    gradients = [p.grad for p in head.parameters() if p.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise ValueError("Missing/nonfinite gradients")
                    if state.state_fingerprint(head.state_dict()) != expected_head:
                        raise ValueError("Diagnostic changed head state without an optimizer update")
                    report["arms"][active_arm] = dict(
                        backward_completed=True, unchanged_head=True,
                        original_loss=float(loss.detach()), nll=float(nll),
                    )
                    emit("arm_passed", **report["arms"][active_arm])
                    bridge.write_json(output / "probe_summary.json", report)
                    del result, probability, loss, nll, gradients, head
                    trace = None
                active_arm = None
                report["active_arm"] = None
                phase = "final_input_integrity"
                bindings.original_train._checked_verify(runtime)
                bridge.verify_ledgers(source)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed during diagnostic")
            phase = "complete"
            report.update(phase="single_image_probe_passed", active_arm=None,
                          conclusion="Not reproduced at saved checkpoint; original failure is unresolved.")
            emit("finished", conclusion=report["conclusion"])
        except BaseException as exc:
            # Do not access CUDA here: device assertions poison that process.
            report.update(phase="diagnostic_failed", failure_stage=phase, active_arm=active_arm,
                          error=type(exc).__name__ + ": " + str(exc))
            (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if trace is not None:
                try:
                    case_path = trace.save_failure(output / "nll_failure_inputs.pt")
                    if case_path:
                        report["cpu_nll_case"] = dict(path=str(case_path), sha256=sha256(case_path))
                except Exception as save_error:
                    report["cpu_nll_case_save_error"] = repr(save_error)
            emit("failed", error=report["error"], failure_stage=phase,
                 cpu_nll_case=report.get("cpu_nll_case"))
        bridge.write_json(output / "probe_summary.json", report)
    print("BEGIN ABLATION PROBE SUMMARY", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("END ABLATION PROBE SUMMARY", flush=True)
    return 0 if report["phase"] == "single_image_probe_passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--addon-dir")
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT)
    parser.add_argument("--locate-only", action="store_true")
    args = parser.parse_args()
    try:
        return probe(args)
    except Exception as exc:
        print("PROBE PREFLIGHT STOPPED: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
