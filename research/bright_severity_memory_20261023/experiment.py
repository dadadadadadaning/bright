#!/usr/bin/env python3
"""Fresh BRIGHT severity-memory experiments; original banks are read only.

`plan` is stdlib-only. `smoke`/`train` run in the foreground in a NEW output
directory. They never resume old heads, change original files, or access Test.
This evaluator reports confusion matrices and the exact original loss; it
deliberately does not reproduce the old per-class NLL accounting validator.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORMAT = "BRIGHT_SEVERITY_MEMORY_V1"
VARIANTS = ("original", "calibration", "severity_memory", "shuffled_memory")
DEFAULT_SOURCE = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_EXTENDED_V2/run_20261019_053508"
DEFAULT_ADDON = "/root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_file(path, label):
    path = Path(path).resolve()
    tag = "_bright_readout_" + label + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(tag, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(tag, None)
        raise
    return module


def code_identity():
    names = ("experiment.py", "old_model_adapter.py", "severity_adapter.py")
    return {name: sha256(HERE / name) for name in names}


def validate_options(args):
    if args.variant not in VARIANTS:
        raise ValueError("Unknown experiment variant")
    if args.seed < 0 or not 1 <= args.steps <= 5000:
        raise ValueError("Nonnegative seed and endpoint 1..5000 required")
    if not 1 <= args.smoke_steps <= 20 or not 1 <= args.fixed_train <= 2918:
        raise ValueError("smoke-steps must be 1..20; fixed-train must be 1..2918")
    if args.read_weight != 0:
        raise ValueError("This study preserves the original NLL + Lovasz loss; read-weight must be zero")
    if not math.isfinite(args.post500_lr) or not 0 < args.post500_lr <= 1:
        raise ValueError("post500-lr must be finite in (0,1]")


def prepare_plan(args):
    """No torch import, no tensors loaded, no output directory created."""
    validate_options(args)
    addon = Path(args.addon_dir).expanduser().resolve()
    print("SEVERITY PREFLIGHT: verifying original ablation package", file=sys.stderr, flush=True)
    bridge = load_file(addon / "runtime_bridge.py", "bridge")
    addon_sha = bridge.check_package(addon)
    print("SEVERITY PREFLIGHT: verifying source manifest, release and bank records; no training", file=sys.stderr, flush=True)
    source = bridge.inspect_source(args.source_run)
    effective = args.variant
    plan = dict(
        format=FORMAT, source=source, addon_dir=str(addon), addon_sha256=addon_sha,
        extension_files=code_identity(), variant=args.variant, effective_mode=effective,
        seed=args.seed, endpoint_step=args.steps, read_weight=args.read_weight,
        post500_lr_multiplier=args.post500_lr, scheduler_horizon=5000,
        warmup_steps=500, maximum_lr=1e-4, weight_decay=0.005, grad_clip=1.0,
        optimizer="AdamW", support_arm="d", fresh_initialization=True,
        old_head_weights_reused=False, historical_trajectory_claim=False,
        fixed_train_count=args.fixed_train, smoke_steps=args.smoke_steps,
        test_accessed=False, training_started=False, gpu_verified=False,
        original_loss="exact runtime.loss (NLL plus Lovasz), unchanged",
        added_loss=None,
        method=dict(name="bounded foreground severity evidence", bias_bound=2.0,
                    memory_gain_bound=2.0, memory_gain_signed=True, zero_initialization=True,
                    query_gt_in_forward=False, support_bank="unchanged Train-only arm d",
                    shuffle_seed=args.seed, shuffle_scope="new label readout only",
                    probability_mass_policy="background and total foreground mass preserved up to floating point; argmax building mask can change"),
        evaluation_format="aggregate/per-image/per-event confusion and exact original loss",
        evaluation_omissions=["old per-class NLL decomposition validator", "Test"],
        checkpoint_semantics="load with this version and variant-specific severity adapter; old six-arm Test loaders are incompatible",
    )
    return plan, bridge


def verify_plan(plan, bridge):
    if code_identity() != plan["extension_files"]:
        raise ValueError("Readout extension changed after plan")
    if bridge.check_package(plan["addon_dir"]) != plan["addon_sha256"]:
        raise ValueError("Original ablation package changed after plan")
    bridge.verify_ledgers(plan["source"])


def schedule(endpoint):
    return sorted({endpoint} | {s for s in (500, 1000, 1500, 2000, 2500, 3000,
                                           3500, 4000, 4500, 5000) if s <= endpoint})


def batches(runtime, names, seed):
    epoch = 0
    while True:
        found = False
        for batch in runtime.batches(names, train=True, seed=seed + epoch):
            found = True
            yield epoch, batch
        if not found:
            raise ValueError("Empty training loader")
        epoch += 1


def combined_loss(original_loss, output, target, support, common, slots, *, read_weight):
    """No query-label readout supervision in this isolated forward-path study."""
    if read_weight != 0:
        raise ValueError("Severity study requires zero auxiliary readout loss")
    return original_loss, {"read_loss": 0.0, "read_loss_evaluated": False}


def model_diagnostics(output):
    """Detach only tiny summary tensors; do not retain attention/pixel graphs."""
    result = {}
    for name in ("memory_gain", "severity_bias", "severity_evidence_l1", "severity_memory_mass"):
        if name in output:
            value = output[name].detach().cpu()
            if value.numel() > 3:
                raise ValueError("Unexpected non-summary diagnostic: " + name)
            result[name] = value.tolist()
    return result


def core_state_dict(model):
    """Match fresh original backbone/head initialization across all variants."""
    omitted = {"severity_bias_raw", "memory_gain_raw"}
    return {name: value for name, value in model.state_dict().items() if name not in omitted}


@contextlib.contextmanager
def capture_training_failure(output, status):
    """Persist the first error using only CPU/stdlib before CUDA teardown."""
    try:
        yield
    except BaseException as exc:
        path = Path(output) / "first_failure.json"
        record = dict(phase=status.get("phase"), image=status.get("current_image"),
                      last_complete_step=status.get("last_complete_step"),
                      error=type(exc).__name__ + ": " + str(exc),
                      traceback=traceback.format_exc(), tensors_modified=False,
                      automatic_retry=False)
        try:
            if not path.exists():
                write_json(path, record)
        except OSError as logging_error:
            print("Could not persist first failure: " + str(logging_error), file=sys.stderr, flush=True)
        raise


def update(model, optimizer, example, runtime, factory, state, slots, *, step,
           read_weight, multiplier):
    import torch
    support = factory.prepare(example, ("d",))["d"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = state.lr_at_step(step, maximum=1e-4, horizon=5000, multiplier=multiplier)
    for group in optimizer.param_groups:
        group["lr"] = lr
    output = model(**example.common, **support)
    original_loss = runtime.loss(output["probabilities"], example.target)
    loss, read_stats = combined_loss(original_loss, output, example.target, support,
                                     example.common, slots, read_weight=read_weight)
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite/non-scalar training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    label_gradient = sum(float(p.grad.detach().abs().sum())
                         for name, p in model.named_parameters()
                         if name.startswith("label_embedding") and p.grad is not None)
    reader_gradient = sum(float(p.grad.detach().abs().sum())
                          for name, p in model.named_parameters()
                          if name.startswith(("query_projection", "key_projection", "value_projection",
                                              "label_embedding", "context_projection")) and p.grad is not None)
    optimizer.step()
    counts = [int(value["step"]) for value in optimizer.state.values() if "step" in value]
    if not counts or any(count != step for count in counts):
        raise ValueError("Active optimizer parameter update count differs")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite updated parameters")
    result = dict(original_loss=float(original_loss.detach()), training_loss=float(loss.detach()),
                  lr=lr, optimizer_step=step, grad_norm_preclip=float(norm),
                  label_grad_abs_sum=label_gradient, reader_grad_abs_sum=reader_gradient,
                  **read_stats, severity_diagnostics=model_diagnostics(output))
    del output, original_loss, loss, support
    return result


def validate_probability(probability, target):
    import torch
    if (probability.dtype != torch.float32 or probability.ndim != 4
            or tuple(probability.shape) != (target.shape[0], 4, *target.shape[-2:])
            or probability.device != target.device
            or not bool(torch.isfinite(probability).all())
            or bool((probability < 0).any())
            or not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]),
                                  atol=2e-5, rtol=2e-5)):
        raise ValueError("Invalid full-image probabilities; no values were repaired")
    valid = target != 255
    if bool((valid & ((target < 0) | (target > 3))).any()):
        raise ValueError("Target must contain only classes 0..3 or ignore 255")


def metrics(confusion):
    import numpy as np
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.shape != (4, 4) or (cm < 0).any():
        raise ValueError("Expected nonnegative 4x4 confusion matrix")
    tp = cm.diagonal().astype(np.float64)
    gt, pred = cm.sum(1), cm.sum(0)
    union = gt + pred - tp
    iou = [float(tp[c] / union[c]) if union[c] else None for c in range(4)]
    fg_tp = int(cm[1:, 1:].sum())
    fg_fn, fg_fp = int(cm[1:, 0].sum()), int(cm[0, 1:].sum())
    return dict(confusion=cm.tolist(), iou_pct=[None if x is None else 100*x for x in iou],
                miou4_pct=100*sum(x or 0 for x in iou)/4,
                foreground_miou3_pct=100*sum(x or 0 for x in iou[1:])/3,
                precision_pct=[100*float(tp[c]/pred[c]) if pred[c] else None for c in range(4)],
                recall_pct=[100*float(tp[c]/gt[c]) if gt[c] else None for c in range(4)],
                building_iou_pct=100*fg_tp/(fg_tp+fg_fn+fg_fp) if fg_tp+fg_fn+fg_fp else None,
                oa_pct=100*float(tp.sum()/cm.sum()) if cm.sum() else None,
                averaging="fixed 4/3 classes; absent-class IoU contributes zero")


def verify_evaluation_journal(path, names, totals, transitions, events):
    """Bind aggregated tables to the persisted, ordered per-image evidence."""
    import numpy as np
    rebuilt = {key: np.zeros((4, 4), dtype=np.int64) for key in totals}
    rebuilt_pairs = np.zeros((4, 4, 4), dtype=np.int64)
    rebuilt_events = {}
    seen = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= len(names) or row["index"] != index or row["image"] != names[index]:
                raise ValueError("Saved evaluation journal order differs")
            if set(row["confusion"]) != set(totals):
                raise ValueError("Saved evaluation path identities differ")
            event = rebuilt_events.setdefault(row["event"], {
                key: np.zeros((4, 4), dtype=np.int64) for key in totals})
            for key in totals:
                cm = np.asarray(row["confusion"][key])
                if cm.shape != (4, 4) or cm.dtype.kind not in "iu" or (cm < 0).any():
                    raise ValueError("Invalid saved confusion matrix")
                rebuilt[key] += cm
                event[key] += cm
            pair = np.asarray(row["paired_counts_grid_to_variant"])
            if pair.shape != (4, 4, 4) or pair.dtype.kind not in "iu" or (pair < 0).any():
                raise ValueError("Invalid saved paired transition counts")
            rebuilt_pairs += pair
            seen.append(row["image"])
    if seen != list(names) or not np.array_equal(rebuilt_pairs, transitions):
        raise ValueError("Saved evaluation coverage/transitions differ")
    if any(not np.array_equal(rebuilt[key], totals[key]) for key in totals):
        raise ValueError("Saved evaluation totals differ")
    if set(events) != set(rebuilt_events) or any(
            not np.array_equal(events[event][key], rebuilt_events[event][key])
            for event in events for key in totals):
        raise ValueError("Saved per-event confusion totals differ")


def evaluate(runtime, model, factory, state, original, names, folder, *, step, split,
             seed, smoke, variant):
    """Bounded, no-autograd evaluation; full identity checks precede completion."""
    import numpy as np
    import torch
    names = list(names)
    allowed = list(runtime.val_names if split == "val" else runtime.train_names)
    if split not in ("val", "fixed_train") or not names or len(names) != len(set(names)) or not set(names) <= set(allowed):
        raise ValueError("Invalid evaluation split/names; Test is unavailable")
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    snapshot = state.capture_rng()
    modes = [(m, m.training) for m in model.modules()]
    state_before = state.state_fingerprint(model.state_dict())
    totals = {key: np.zeros((4, 4), dtype=np.int64) for key in ("grid_probq", variant)}
    losses = {key: 0.0 for key in totals}
    transitions = np.zeros((4, 4, 4), dtype=np.int64)
    events, identities = {}, []
    model.eval()
    success = False
    position = {"index": None, "image": None, "stage": "start"}
    try:
        with torch.no_grad(), (folder / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(runtime.evaluation_batches(names, seed=seed)):
                position = dict(index=index, image=names[index] if index < len(names) else None,
                                stage="prepare")
                example = runtime.prepare(batch)
                if index >= len(names) or example.image != names[index]:
                    raise ValueError("Evaluation image order differs from requested names")
                original._assert_example(example)
                support = factory.prepare(example, ("d",))["d"]
                position["stage"] = "forward"
                output = model(**example.common, **support)
                target = example.target.detach().cpu().numpy()
                row = dict(index=index, image=example.image, event=example.event,
                           confusion={}, original_loss={}, test_accessed=False,
                           severity_diagnostics=model_diagnostics(output))
                predictions = {}
                event_tables = events.setdefault(example.event, {
                    key: np.zeros((4, 4), dtype=np.int64) for key in totals})
                probabilities = {"grid_probq": example.common["base_probability_full"],
                                 variant: output["probabilities"]}
                for key, probability in probabilities.items():
                    position.update(stage="probability_and_loss", path=key)
                    validate_probability(probability, example.target)
                    value = runtime.loss(probability, example.target)
                    if value.ndim or not bool(torch.isfinite(value)):
                        raise ValueError("Nonfinite evaluation loss")
                    prediction = probability.argmax(1).cpu().numpy()
                    cm = original._confusion(target, prediction)
                    if int(cm.sum()) != int((target != 255).sum()):
                        raise ValueError("Confusion count differs from valid target count")
                    predictions[key] = prediction
                    totals[key] += cm
                    event_tables[key] += cm
                    losses[key] += float(value)
                    row["confusion"][key] = cm.tolist()
                    row["original_loss"][key] = float(value)
                paired = original._transition(target, predictions["grid_probq"], predictions[variant])
                transitions += paired
                row["paired_counts_grid_to_variant"] = paired.tolist()
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                identities.append(example.image)
                if (index + 1) % 10 == 0 or index + 1 == len(names):
                    print(f"SEVERITY {split} step={step} {index+1}/{len(names)}", flush=True)
                del output, support, probabilities, probability, value, example, batch
        if identities != names:
            raise ValueError("Evaluation did not complete exactly the requested image list")
        verify_evaluation_journal(folder / "images.jsonl", names, totals, transitions, events)
        if state.state_fingerprint(model.state_dict()) != state_before:
            raise ValueError("Evaluation changed model state")
        success = True
    except BaseException as exc:
        # Persist the first exception before Runtime.close or a poisoned CUDA
        # context can raise a secondary exception during outer cleanup.
        write_json(folder / "failure.json", dict(
            complete=False, position=position, completed_images=len(identities),
            requested_images=len(names), error=type(exc).__name__ + ": " + str(exc),
            traceback=traceback.format_exc(), tensor_values_modified=False,
        ))
        raise
    finally:
        for module, training in modes:
            module.training = training
        # After a CUDA error the context may be poisoned; preserve the first error.
        if success:
            state.restore_rng(snapshot)
            if state.state_fingerprint(state.capture_rng()) != state.state_fingerprint(snapshot):
                raise ValueError("Evaluation RNG restore mismatch")
    full_val = split == "val" and names == list(runtime.val_names) and len(names) == 450 and not smoke
    report = dict(format=FORMAT + "_EVALUATION", step=step, split=split, seed=seed,
                  variant=variant, complete=False, execution_complete=True,
                  full_validation_evidence=False, whole_val450_processed=full_val,
                  final_integrity_pending=True, phase="evaluation_complete_integrity_pending",
                  image_count=len(names), requested_names=names, test_accessed=False,
                  class_names=["background", "intact", "damaged", "destroyed"],
                  metrics={key: dict(metrics(cm), mean_original_loss=losses[key]/len(names)) for key, cm in totals.items()},
                  per_event={event: {key: metrics(cm) for key, cm in tables.items()} for event, tables in events.items()},
                  paired_counts_grid_to_variant=transitions.tolist(),
                  paired_axes=["GT", "grid_probq", variant],
                  head_state_unchanged=True, rng_restored=True, smoke=smoke,
                  journal_verified=True, images_sha256=sha256(folder / "images.jsonl"),
                  evaluation_objective="exact runtime.loss; no class-NLL decomposition check")
    write_json(folder / "result.json", report)
    return report


def save_checkpoint(path, model, optimizer, state, runtime, plan, step):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    payload = dict(format=FORMAT + "_CHECKPOINT", step=step, variant=plan["variant"],
                   effective_mode=plan["effective_mode"], read_weight=plan["read_weight"],
                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng=state.capture_rng(), plan=plan, protocol_info=runtime.protocol_info,
                   test_accessed=False)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    reopened = torch.load(temporary, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "rng"):
        if state.state_fingerprint(reopened[key]) != state.state_fingerprint(payload[key]):
            raise ValueError("Checkpoint roundtrip differs: " + key)
    temporary.replace(path)
    result = dict(path=str(path), step=step, sha256=sha256(path),
                  variant=plan["variant"], roundtrip_verified=True)
    write_json(path.with_suffix(".pt.json"), result)
    return result


@contextlib.contextmanager
def addon_imports(addon, bridge):
    """Bindings live only in this foreground process and are restored on exit."""
    previous = sys.modules.get("runtime_bridge")
    if previous is not None and Path(previous.__file__).resolve() != addon / "runtime_bridge.py":
        raise ImportError("An unrelated runtime_bridge is already imported")
    sys.modules["runtime_bridge"] = bridge
    sys.path.insert(0, str(addon))
    try:
        yield
    finally:
        sys.path.remove(str(addon))
        if previous is None:
            sys.modules.pop("runtime_bridge", None)
        else:
            sys.modules["runtime_bridge"] = previous


def run(args, plan, bridge):
    if not args.output:
        raise ValueError("smoke/train requires --output pointing to a NEW directory")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Preserve existing results; select a NEW output: " + str(output))
    # This lease covers this extension's sibling outputs only; it never kills jobs.
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".severity_active.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, plan, bridge, output)


def run_locked(args, plan, bridge, output):
    print("SEVERITY PREFLIGHT: rechecking recorded source and extension fingerprints", flush=True)
    verify_plan(plan, bridge)
    output.mkdir(exist_ok=False)
    smoke = args.command == "smoke"
    endpoint = args.smoke_steps if smoke else args.steps
    status = dict(format=FORMAT, variant=args.variant, effective_mode=plan["effective_mode"], seed=args.seed,
                  phase="loading", complete=False, training_complete=False,
                  full_validation_evidence=False, smoke=smoke, test_accessed=False,
                  last_complete_step=0, endpoint_step=endpoint, pid=os.getpid(),
                  output=str(output), evaluations={}, original_failed_runs_unchanged=True)
    write_json(output / "plan.json", plan)
    write_json(output / "summary.json", status)
    start = time.monotonic()
    results = []
    try:
        print("SEVERITY LOAD: importing verified runtime; no optimizer updates yet", flush=True)
        bindings = bridge.load_runtime(plan["source"])
        torch, original, state = bindings.torch, bindings.original_train, bindings.legacy.state
        if not torch.cuda.is_available():
            raise ValueError("Real smoke/train requires CUDA; local CPU contracts are not GPU evidence")
        adapter = load_file(HERE / "severity_adapter.py", "severity_adapter")
        old_adapter = load_file(HERE / "old_model_adapter.py", "old_adapter")
        base = original.JointRegionMemoryClassifier
        addon = Path(plan["addon_dir"])
        runtime_args = bridge.runtime_arguments(plan["source"], seed=args.seed, output=output)
        with addon_imports(addon, bridge):
            support_module = bridge.named(addon / "supports.py", "readout_supports")
            # The original importer binds slots before this point; reject ambiguity.
            import slots
            expected_slots = Path(plan["source"]["package"]) / "deps/original/slots.py"
            if Path(slots.__file__).resolve() != expected_slots.resolve():
                raise ImportError("Unexpected slots module")
            classifier = adapter.make_classifier(base, old_adapter, slots, mode=plan["effective_mode"])
            print("SEVERITY LOAD: loading frozen model and Memory; input verification may take several minutes", flush=True)
            with bindings.Runtime(runtime_args, output) as runtime:
                original._formal_runtime(runtime, False)
                full_train, full_val = original._names(runtime, False)
                print("SEVERITY VERIFY: checking frozen runtime and input fingerprints", flush=True)
                original._checked_verify(runtime)
                print("SEVERITY VERIFY: runtime checks passed; creating fresh head and optimizer", flush=True)
                train_names, val_names = list(full_train), list(full_val)
                if smoke:
                    train_names, val_names = map(list, runtime.smoke_names())
                fixed = list(train_names) if smoke else sorted(random.Random(args.seed+191).sample(
                    list(full_train), min(args.fixed_train, len(full_train))))
                factory = support_module.SupportFactory(runtime, seed=args.seed)
                original._seed(args.seed)
                model_kwargs = dict(chunk_size=int(runtime_args.chunk_size))
                if args.variant != "original":
                    model_kwargs["shuffle_seed"] = args.seed
                model = classifier(**model_kwargs).to(runtime.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime_args.lr),
                                             weight_decay=float(runtime_args.weight_decay))
                initial = state.state_fingerprint(model.state_dict())
                manifest = dict(format=FORMAT, plan=plan, seed=args.seed,
                                train_names=list(full_train), val_names=list(full_val),
                                fixed_train_names=fixed, actual_training_names=train_names,
                                initial_head_sha256=initial,
                                initial_core_head_sha256=state.state_fingerprint(core_state_dict(model)),
                                runtime_config=vars(runtime_args),
                                protocol_info=runtime.protocol_info, support_arm="d",
                                parameter_count=sum(p.numel() for p in model.parameters()),
                                fresh_initialization=True, test_accessed=False)
                write_json(output / "manifest.json", manifest)
                write_json(output / "support_provenance.json", factory.provenance())

                def assess(step, split, names, folder):
                    status["phase"] = f"evaluate_{split}_{step}"
                    write_json(output / "summary.json", status)
                    print(f"SEVERITY ASSESS: {split}, step={step}, requested_images={len(names)}", flush=True)
                    report = evaluate(runtime, model, factory, state, original, names,
                                      output / folder, step=step, split=split, seed=args.seed,
                                      smoke=smoke, variant=args.variant)
                    results.append((output / folder / "result.json", report))
                    status["evaluations"][folder] = dict(step=step, image_count=report["image_count"],
                                                         metrics=report["metrics"])
                    write_json(output / "summary.json", status)

                assess(0, "fixed_train" if smoke else "val",
                       train_names if smoke else val_names, "initial_train" if smoke else "initial_val")
                iterator = batches(runtime, train_names, args.seed)
                stops = [endpoint] if smoke else schedule(endpoint)
                with (output / "train.jsonl").open("x", encoding="utf-8") as journal:
                    for step in range(1, endpoint+1):
                        status["phase"] = f"prepare_{step}"
                        status["current_image"] = None
                        with capture_training_failure(output, status):
                            epoch, batch = next(iterator)
                            example = runtime.prepare(batch)
                            status["current_image"] = example.image
                            original._assert_example(example)
                            if example.image not in train_names:
                                raise ValueError("Training loader returned a non-Train image")
                            status["phase"] = f"update_{step}"
                            row = update(model, optimizer, example, runtime, factory, state, slots,
                                         step=step, read_weight=args.read_weight,
                                         multiplier=args.post500_lr)
                        row.update(step=step, epoch=epoch, epoch_seed=args.seed+epoch,
                                   image=example.image, event=example.event, variant=args.variant)
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        journal.flush()
                        status["last_complete_step"] = step
                        if step == endpoint:
                            # A failed endpoint evaluation must not erase the
                            # fact that the requested optimizer updates finished.
                            status["training_complete"] = True
                        if step == 1 or step % 20 == 0 or step == endpoint:
                            print("SEVERITY TRAIN " + json.dumps(row, allow_nan=False), flush=True)
                            write_json(output / "summary.json", status)
                        if step % 50 == 0 or step in stops:
                            os.fsync(journal.fileno())
                            status["last_checkpoint"] = save_checkpoint(
                                output / "checkpoints" / f"checkpoint_{step:06d}.pt", model,
                                optimizer, state, runtime, plan, step)
                            write_json(output / "summary.json", status)
                        del row, example, batch
                        if step in stops:
                            assess(step, "fixed_train" if smoke else "val",
                                   train_names if smoke else val_names,
                                   f"{'smoke_train' if smoke else 'val'}_step_{step:06d}")
                status["training_complete"] = True
                assess(endpoint, "fixed_train", fixed, "fixed_train_endpoint")
                status["phase"] = "final_integrity"
                status["integrity_checks"] = original._checked_verify(runtime)
                verify_plan(plan, bridge)
                bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                write_json(output / "input_fingerprints.json", {str(p): h for p, h in runtime.input_ledger.items()})
            guard = runtime.guard_report
            if guard.get("restored") is not True or guard.get("failures") or guard.get("restore_conflicts"):
                raise ValueError("PoolGuard close failed")
            status["pool_guard"] = guard
        verify_plan(plan, bridge)
        for path, report in results:
            report.update(complete=True, final_integrity_pending=False,
                          phase="complete_smoke" if smoke else "complete",
                          full_validation_evidence=bool(report["whole_val450_processed"]))
            write_json(path, report)
            path.with_suffix(".json.sha256").write_text(sha256(path) + "  result.json\n")
        status.update(method=plan["method"], phase="complete_smoke" if smoke else "complete", complete=True,
                      full_validation_evidence=not smoke and len(full_val) == 450,
                      elapsed_seconds=time.monotonic()-start, error=None)
        write_json(output / "summary.json", status)
        print("SEVERITY COMPLETE " + str(output), flush=True)
    except BaseException as exc:
        failed_phase = status.get("phase")
        status.update(phase="failed", complete=False, full_validation_evidence=False,
                      failure_phase=failed_phase, error=type(exc).__name__ + ": " + str(exc),
                      elapsed_seconds=time.monotonic()-start)
        write_json(output / "summary.json", status)
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "validate-source", "smoke", "train", "status"))
    p.add_argument("--source-run", default=DEFAULT_SOURCE)
    p.add_argument("--addon-dir", default=DEFAULT_ADDON)
    p.add_argument("--variant", choices=VARIANTS, default="original")
    p.set_defaults(read_weight=0.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--post500-lr", type=float, default=0.2)
    p.add_argument("--fixed-train", type=int, default=450)
    p.add_argument("--smoke-steps", type=int, default=3)
    p.add_argument("--output", help="NEW output directory for smoke/train; existing directory for status")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "status":
        if not args.output:
            raise ValueError("status requires --output")
        print(json.dumps(read_json(Path(args.output)/"summary.json"), ensure_ascii=False, indent=2))
        return
    plan, bridge = prepare_plan(args)
    if args.command in ("plan", "validate-source"):
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    return run(args, plan, bridge)


if __name__ == "__main__":
    main()
