"""A separately trained D/H pair: H has no support-read contribution."""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

from training_common import wrap_evaluator
from evaluate_checkpoint import write_json
from grid_probe import sha256, read_json

FORMAT = "BRIGHT_D_H_NO_SUPPORT_READ_V1"
ARMS = ("d", "h")
READER_PREFIXES = ("query_projection", "key_projection", "value_projection",
                   "label_embedding", "context_projection")


def zero_read(self, query, *args, **kwargs):
    """No candidate features/labels enter this output; current-image paths remain."""
    import torch
    return torch.zeros_like(query)


def disable_support_read(model):
    if not callable(getattr(model, "_read", None)):
        raise ValueError("Expected original JointRegionMemoryClassifier._read")
    model._read = types.MethodType(zero_read, model)
    for name, parameter in model.named_parameters():
        if name.startswith(READER_PREFIXES):
            parameter.requires_grad_(False)
    model._support_read_disabled = True


class PairFactory:
    """Keep D's original legal support preparation and local soft-slot inputs."""
    def __init__(self, original):
        self.original = original

    def prepare(self, example, arms):
        if set(arms) != set(ARMS):
            raise ValueError("Require exactly the fresh D/H pair")
        supports = self.original.prepare(example, ("d",))
        return {arm: supports["d"] for arm in arms}

    def forward(self, model, example, arm, support):
        if arm == "d":
            return self.original.forward(model, example, "d", support)
        if arm != "h" or not getattr(model, "_support_read_disabled", False):
            raise ValueError("H support-read bypass has not been installed")
        # Original forward still validates inputs and builds the same local paths.
        # Only _read's output is bypassed. This is not an equal-FLOPs experiment.
        return model(**example.common, **support)

    def provenance(self):
        return dict(original_d_support=self.original.provenance(),
                    h_support_read="zero context, independent of support features and labels",
                    h_current_image_paths="pixel, parent region, soft slots and joint classifier",
                    h_is_last_layer_only=False, equal_compute_claim=False,
                    candidate_scaffolding_still_computed=True, query_GT_used_for_supports=False)


@contextlib.contextmanager
def pair_schema(evaluator, evidence):
    old = (evaluator.ARMS, evaluator.CONTRASTS,
           evidence.VALID_ARMS, evidence.CONTRASTS)
    evaluator.ARMS = (*evaluator.ARMS, "h")
    evaluator.CONTRASTS = (*evaluator.CONTRASTS, ("d", "h"))
    evidence.VALID_ARMS = set(evidence.VALID_ARMS) | {"h"}
    evidence.CONTRASTS = (*evidence.CONTRASTS, ("d", "h"))
    try:
        yield
    finally:
        (evaluator.ARMS, evaluator.CONTRASTS,
         evidence.VALID_ARMS, evidence.CONTRASTS) = old


def finalize_pair(results, *, schedule, output, evidence, gates):
    """Validate ALL fresh reports before promoting any; step0 is allowed here."""
    required = ("runtime", "source", "guard", "input_files", "extension")
    if any(gates.get(k) is not True for k in required):
        raise ValueError("Pair final integrity checks incomplete")
    expected = {(s, split): (names, primary) for s, split, names, primary in schedule}
    if len(expected) != len(schedule) or len(results) != len(expected):
        raise ValueError("Fresh pair evaluation schedule is incomplete")
    checked, seen = [], set()
    for path, report in results:
        key = (report["step"], report["split"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected/duplicate pair report")
        seen.add(key)
        names, primary = expected[key]
        evidence.validate_report(report, step=key[0], split=key[1], names=names,
                                 arms=list(ARMS), primary=primary)
        evidence.validate_saved_report(path, report, names, sha256, output)
        finished = copy.deepcopy(report)
        finished.update(complete=True, final_integrity_pending=False, phase="complete",
                        full_validation_evidence=key[1] == "val", primary_endpoint=primary,
                        fresh_pair_format=FORMAT)
        checked.append((path, report, finished))
    for path, original, finished in checked:
        write_json(path, finished)
        path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n")
        original.update(finished)


def save_pair(runner, output, heads, optimizers, state, runtime, step, config, plan, factory):
    """Original tensor roundtrip plus explicit RNG and journal association."""
    import torch
    record = runner.save_checkpoint(output / "checkpoints", heads, optimizers, state,
                                    runtime, step, config, plan, factory)
    path = output / "checkpoints" / ("checkpoint_%06d.pt" % step)
    reopened = torch.load(path, map_location="cpu", weights_only=False)
    if state.state_fingerprint(reopened["rng"]) != state.state_fingerprint(state.capture_rng()):
        raise ValueError("Checkpoint RNG roundtrip differs")
    if reopened["format"] != FORMAT or set(reopened["heads"]) != set(ARMS):
        raise ValueError("Pair checkpoint schema differs")
    record.update(format=FORMAT, rng_roundtrip_verified=True,
                  journal_sha256=sha256(output / "train.jsonl"),
                  updates_per_arm={a: step for a in ARMS},
                  requires_h_support_read_bypass_on_load=True)
    write_json(path.with_suffix(".pt.json"), record)
    return record


def run_pair(context, output):
    from resume_train import verify_files
    from probe import json_value
    import evidence
    output = Path(output)
    plan = copy.deepcopy(context["plan"])
    plan.update(format=FORMAT, arms=list(ARMS), seeds=[context["seed"]],
                old_head_weights_reused=False, fresh_initialization=True,
                no_common_trained_D_prefix=True, primary_metric="four_class_mIoU",
                h_support_read="zero context", h_is_last_layer_only=False)
    endpoint, seed = plan["endpoint_step"], context["seed"]
    source, bridge, addon = plan["source"], context["bridge"], context["addon"]
    blocking = context["manifest"]["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
    if blocking is None:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    status = dict(format=FORMAT, phase="loading", complete=False, training_complete=False,
                  full_validation_evidence=False, final_integrity_pending=True,
                  seed=seed, arms=list(ARMS), last_complete_step=0, endpoint_step=endpoint,
                  updates_per_arm={a: 0 for a in ARMS}, evaluations={}, pid=os.getpid(),
                  output=str(output), test_accessed=False, fresh_initialization=True,
                  h_is_last_layer_only=False, historical_trajectory_claim=False)
    write_json(output / "summary.json", status)
    started, stage, results = time.monotonic(), "verify_inputs", []
    try:
        verify_files(context["evidence_files"])
        import torch
        policy = read_json(source["config"]["protocol"])["numeric_policy"]
        torch.set_num_threads(int(policy["num_threads"]))
        if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
            torch.set_num_interop_threads(int(policy["num_interop_threads"]))
        if not torch.cuda.is_available():
            raise ValueError("Real D/H training requires the original CUDA environment")
        stage = "load_bindings"
        bindings = bridge.load_runtime(source)
        state, original = bindings.legacy.state, bindings.original_train
        sys.modules["runtime_bridge"] = bridge
        sys.path.insert(0, str(addon))
        runner = bridge.named(addon / "ablation_runner.py", "pair_runner")
        runner.FORMAT = FORMAT
        supports = bridge.named(addon / "supports.py", "pair_supports")
        evaluator = wrap_evaluator(bridge.named(addon / "evaluation.py", "pair_evaluator"), output)
        args = bridge.runtime_arguments(source, seed=seed, output=output)
        with pair_schema(evaluator, evidence):
            stage = "load_runtime"
            with bindings.Runtime(args, output) as runtime:
                original._formal_runtime(runtime, False)
                train, val = original._names(runtime, False)
                if train != context["manifest"]["train_names"] or val != context["manifest"]["val_names"]:
                    raise ValueError("Live Train/Val split changed")
                if json_value(runtime.protocol_info) != context["manifest"]["protocol_info"]:
                    raise ValueError("Execution protocol changed")
                original._checked_verify(runtime)
                factory = PairFactory(supports.SupportFactory(runtime, seed=seed))
                original._seed(seed)
                heads, optimizers, initial_hash = runner.model_copies(
                    torch, original, state, ARMS, vars(args), runtime.device)
                disable_support_read(heads["h"])
                if any(state.state_fingerprint(h.state_dict()) != initial_hash for h in heads.values()):
                    raise ValueError("D/H initial parameter values differ")
                fixed = context["manifest"]["fixed_train_names"]
                val_steps = runner.schedule_steps(endpoint)
                schedule = [(0, "val", val, False), (0, "fixed_train", fixed, False)] + [
                    (s, "val", val, s == endpoint) for s in val_steps] + [
                    (endpoint, "fixed_train", fixed, False)]
                write_json(output / "plan.json", plan)
                write_json(output / "manifest.json", dict(
                    format=FORMAT, plan=plan, arms=list(ARMS), seed=seed,
                    train_names=train, val_names=val, fixed_train_names=fixed,
                    initial_head_sha256=initial_hash, runtime_config=vars(args),
                    protocol_info=runtime.protocol_info, old_weights_reused=False,
                    trainable_parameters={a: sum(p.numel() for p in h.parameters() if p.requires_grad)
                                          for a, h in heads.items()},
                    total_parameters={a: sum(p.numel() for p in h.parameters()) for a, h in heads.items()},
                    support_provenance=factory.provenance(), test_accessed=False,
                    evaluation_schedule=[dict(step=s, split=k, count=len(n), primary=p)
                                         for s, k, n, p in schedule]))

                def assess(step, split, names, primary):
                    nonlocal stage
                    stage = status["phase"] = "evaluate_%s_%s" % (split, step)
                    write_json(output / "summary.json", status)
                    folder = output / ("%s_step_%06d" % (split, step))
                    report = evaluator.evaluate(runtime, heads, names, folder, step=step, split=split,
                        seed=seed, original_train=original, legacy=bindings.legacy,
                        factory=factory, primary_endpoint=primary, smoke=False)
                    evidence.validate_report(report, step=step, split=split, names=names,
                                             arms=list(ARMS), primary=primary)
                    evidence.validate_saved_report(folder / "result.json", report, names, sha256, output)
                    results.append((folder / "result.json", report))
                    status["evaluations"][folder.name] = dict(step=step, split=split,
                        image_count=len(names), complete=False, metrics=report["metrics"])
                    write_json(output / "summary.json", status)
                    evaluator.print_table(report)

                for event in schedule[:2]:
                    assess(*event)
                iterator = runner.training_batches(runtime, train, seed)
                with (output / "train.jsonl").open("x") as journal:
                    for step in range(1, endpoint + 1):
                        stage = "prepare_%s" % step
                        epoch, batch = next(iterator)
                        example = runtime.prepare(batch)
                        original._assert_example(example)
                        if example.image not in train:
                            raise ValueError("Training image is outside Train2918")
                        stage = "update_%s" % step
                        rows = runner.update_arms(heads, optimizers, example, runtime, factory, state,
                            step=step, multiplier=plan["post500_lr_multiplier"])
                        if rows["h"]["reader_grad_abs_sum"] != 0:
                            raise ValueError("H unexpectedly has Reader gradients")
                        record = dict(step=step, epoch=epoch, epoch_seed=seed+epoch,
                                      image=example.image, event=example.event, arms=rows)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        status.update(phase="training", last_complete_step=step,
                                      updates_per_arm={a: step for a in ARMS})
                        if step == 1 or step % 20 == 0 or step == endpoint:
                            write_json(output / "summary.json", status)
                            print("D/H TRAIN " + json.dumps(record), flush=True)
                        if step % 50 == 0 or step in val_steps:
                            stage = "checkpoint_%s" % step
                            os.fsync(journal.fileno())
                            if step in val_steps:
                                original._checked_verify(runtime)
                                bridge.verify_ledgers(source)
                            status["last_checkpoint"] = save_pair(runner, output, heads, optimizers,
                                state, runtime, step, vars(args), plan, factory)
                            write_json(output / "summary.json", status)
                            print("D/H CHECKPOINT step=%s" % step, flush=True)
                        if step in val_steps:
                            assess(step, "val", val, step == endpoint)
                        del example, rows, record
                status["training_complete"] = True
                assess(endpoint, "fixed_train", fixed, False)
                stage = "final_integrity"
                status["integrity_checks"] = original._checked_verify(runtime)
                bridge.verify_ledgers(source)
                bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed")
                verify_files(context["evidence_files"])
                verify_files({str(p): h for p, h in runtime.input_ledger.items()})
                write_json(output / "input_fingerprints.json", {str(p): h for p,h in runtime.input_ledger.items()})
                write_json(output / "support_provenance.json", factory.provenance())
            guard = getattr(runtime, "guard_report", {})
            if guard.get("restored") is not True or guard.get("failures") or guard.get("restore_conflicts"):
                raise ValueError("Pool guard closure failed")
            status["pool_guard"] = guard
            finalize_pair(results, schedule=schedule, output=output, evidence=evidence,
                          gates={k: True for k in ("runtime", "source", "guard", "input_files", "extension")})
        for value in status["evaluations"].values():
            value["complete"] = True
        status.update(phase="complete", complete=True, execution_complete=True,
                      training_complete=True, full_validation_evidence=True,
                      final_integrity_pending=False, error=None)
    except BaseException as exc:
        # No tensor inspection/save after CUDA or optimizer errors.
        status.update(phase="failed", error=type(exc).__name__ + ": " + str(exc),
                      failure_phase=stage, complete=False, execution_complete=False)
        (output / "traceback.txt").write_text(traceback.format_exc())
    status["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", status)
    print("D/H RESULT " + json.dumps({k: status.get(k) for k in (
        "phase", "last_complete_step", "endpoint_step", "error", "output")}), flush=True)
    return 0 if status["complete"] else 1
