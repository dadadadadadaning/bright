"""Matched A/B/C/D soft-routing and Memory training and independently re-opened prediction evidence.

This module never changes the frozen BRIGHT frontend or a Memory bank. Runtime
owns real-data loading, the V3 bindings, retrieval and the inherited loss. CPU
contract tests inject a small Runtime; those tests are not BRIGHT validation.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from common import ARMS, MEMORY_ARMS, ARM_RULES, FORMAT, digest, module_hash, source_hash, write_json
from model import JointRegionMemoryClassifier

PATHS = ("grid_probq",) + ARMS


def _config(args):
    return json.loads(json.dumps(vars(args), default=str))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError("A new output directory is required; preserving: " + str(path))
    path.mkdir(parents=True)
    return path


def _factory(factory):
    if factory is not None:
        return factory
    from server_runtime import Runtime
    return Runtime


def _checked_verify(runtime):
    result = runtime.verify()
    if not isinstance(result, dict):
        raise TypeError("Runtime.verify must return a dict")
    if "passed" in result:
        passed = result["passed"] is True
    else:
        def leaves(value):
            if isinstance(value, dict):
                return [x for v in value.values() for x in leaves(v)]
            return [value] if isinstance(value, bool) else []
        checks = leaves(result)
        passed = bool(checks) and all(checks)
    if not passed:
        raise RuntimeError("Frozen input / code / model verification failed: " + str(result))
    return result


def _names(runtime, smoke):
    train = list(runtime.train_names)
    val = list(runtime.val_names)
    if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError("Train/Val image identities are repeated or overlap")
    if smoke:
        picked_train, picked_val = runtime.smoke_names()
        picked_train, picked_val = list(picked_train), list(picked_val)
        if (not picked_train or not picked_val or not set(picked_train) <= set(train)
                or not set(picked_val) <= set(val)):
            raise ValueError("Smoke selection must use nonempty Train and Val subsets")
        if len(set(picked_train)) != len(picked_train) or len(set(picked_val)) != len(picked_val):
            raise ValueError("Repeated smoke image identity")
        return picked_train, picked_val
    if len(train) != 2918 or len(val) != 450:
        raise ValueError("Formal protocol requires exactly Train2918 and Val450")
    return train, val


def _new_heads(args, device):
    first = JointRegionMemoryClassifier(chunk_size=int(args.chunk_size)).to(device)
    heads = {arm: first if index == 0 else copy.deepcopy(first)
             for index, arm in enumerate(ARMS)}
    hashes = {arm: module_hash(head) for arm, head in heads.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError("Matched heads do not have identical initialization")
    return heads, hashes


def _lr(step, steps, maximum):
    # step is one-based: every arm sees the same schedule, ending at exactly 0.
    warmup = min(500, max(1, steps // 10))
    if steps == 1:
        return float(maximum)
    if step <= warmup:
        return float(maximum) * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return float(maximum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _confusion(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction shape differ")
    valid = target != 255
    if np.any((target[valid] < 0) | (target[valid] >= 4)):
        raise ValueError("Unexpected target class")
    if np.any((prediction < 0) | (prediction >= 4)):
        raise ValueError("Unexpected prediction class")
    return np.bincount(4 * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
                       minlength=16).reshape(4, 4).astype(np.int64)


def _metrics(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    target = cm.sum(1).astype(np.float64)
    prediction = cm.sum(0).astype(np.float64)
    union = target + prediction - tp
    iou = [float(tp[c] / union[c]) if union[c] else None for c in range(4)]
    return {
        "class_names": ["background", "intact", "damaged", "destroyed"],
        "iou": iou,
        "foreground_miou": sum(x if x is not None else 0.0 for x in iou[1:]) / 3.0,
        "valid_miou": sum(x if x is not None else 0.0 for x in iou[1:]) / 3.0,
        "foreground_average": "fixed classes 1/2/3; absent IoU contributes zero",
        "precision": [float(tp[c] / prediction[c]) if prediction[c] else None for c in range(4)],
        "recall": [float(tp[c] / target[c]) if target[c] else None for c in range(4)],
        "target_pixels": target.astype(np.int64).tolist(),
        "confusion": cm.tolist(),
    }


def _transition(target, base, prediction):
    valid = target != 255
    code = (target[valid].astype(np.int64) * 16 + base[valid].astype(np.int64) * 4
            + prediction[valid].astype(np.int64))
    return np.bincount(code, minlength=64).reshape(4, 4, 4).astype(np.int64)


def _transition_report(counts):
    a = np.asarray(counts, dtype=np.int64)
    rescue = [sum(int(a[c, b, c]) for b in range(4) if b != c) for c in range(4)]
    harm = [sum(int(a[c, c, p]) for p in range(4) if p != c) for c in range(4)]
    both_wrong = [sum(int(a[c, b, p]) for b in range(4) for p in range(4)
                      if b != c and p != c) for c in range(4)]
    return {"axes": ["GT", "grid_probq", "joint_prediction"], "counts": a.tolist(),
            "rescued_by_gt_class": rescue, "harmed_by_gt_class": harm,
            "both_wrong_by_gt_class": both_wrong,
            "correct_intact_changed_to_damaged": int(a[1, 1, 2]),
            "damaged_previously_intact_now_damaged": int(a[2, 1, 2])}


def _assert_example(example):
    if set(example.supports) != set(ARMS):
        raise ValueError("Example must provide all four matched arms")
    for fields in (example.common, *example.supports.values()):
        for value in fields.values():
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    raise ValueError("Frozen frontend/support inputs must not require gradients")
                if torch.is_inference(value):
                    raise ValueError("Use no_grad or clone inference tensors before trainable heads")
    if example.target.ndim != 3 or example.target.shape[0] != 1:
        raise ValueError("This matched experiment expects one image [1,H,W] at a time")
    target = example.target
    if target.dtype.is_floating_point or target.is_complex():
        raise ValueError("Training target must already have an integer dtype")
    legal = ((target >= 0) & (target < 4)) | (target == 255)
    if not bool(legal.all()):
        raise ValueError("Target labels must be 0/1/2/3 or 255 before any uint8 conversion")


def _coverage(example):
    """Count parent selections and routed query/candidate slots separately.

    Each selected parent owns at most two slots.  Neither those two slots nor
    the number of attention edges is a count of independent historical cases.
    """
    result = {}
    for arm in ARMS:
        fields = example.supports[arm]
        selected = fields["support_ids"]
        candidate_valid = fields["support_valid"]
        query_valid = fields["slot_valid"]
        if selected.ndim != 2 or selected.dtype != torch.long:
            raise ValueError("support_ids must be int64 [parent_queries, topk]")
        if candidate_valid.ndim != 2 or candidate_valid.shape[1] != 2 or candidate_valid.dtype != torch.bool:
            raise ValueError("support_valid must be bool [bank_parents,2]")
        if query_valid.shape != (1, selected.shape[0], 2) or query_valid.dtype != torch.bool:
            raise ValueError("slot_valid must be bool [1,parent_queries,2]")
        if selected.device != candidate_valid.device or selected.device != query_valid.device:
            raise ValueError("Coverage inputs must share one device")
        if bool(((selected < -1) | (selected >= candidate_valid.shape[0])).any()):
            raise ValueError("Out-of-range selected parent ID")
        ordered = selected.sort(dim=-1).values
        if ordered.shape[1] > 1 and bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
            raise ValueError("Repeated parent case within a query candidate list")
        present = selected >= 0
        if candidate_valid.shape[0]:
            slots = candidate_valid[selected.clamp_min(0)] & present[..., None]
        else:
            slots = torch.zeros((*selected.shape, 2), device=selected.device, dtype=torch.bool)
        qvalid = query_valid[0]
        active_parent = qvalid.any(-1)
        slots = slots & active_parent[:, None, None]
        parent_has_candidates = slots.any(-1).any(-1)
        valid_candidate_counts = slots.sum((1, 2))
        legal_parent = slots.any(-1)
        unique_members = selected[legal_parent].unique().numel()
        result[arm] = {
            "parent_queries": int(selected.shape[0]),
            "active_parent_queries": int(active_parent.sum().item()),
            "active_slot_queries": int(qvalid.sum().item()),
            "supported_parent_queries": int(parent_has_candidates.sum().item()),
            "supported_slot_queries": int((qvalid & parent_has_candidates[:, None]).sum().item()),
            "legal_parent_selections": int(legal_parent.sum().item()),
            "valid_candidate_slots": int(slots.sum().item()),
            "query_slot_candidate_edges": int((valid_candidate_counts * qvalid.sum(-1)).sum().item()),
            "sum_per_image_unique_parent_members": int(unique_members),
        }
    return result


def _merge_coverage(total, current):
    for arm in ARMS:
        for key, value in current[arm].items():
            total[arm][key] += value


def _blank_coverage():
    keys = ("parent_queries", "active_parent_queries", "active_slot_queries",
            "supported_parent_queries", "supported_slot_queries", "legal_parent_selections",
            "valid_candidate_slots", "query_slot_candidate_edges", "sum_per_image_unique_parent_members")
    return {arm: {key: 0 for key in keys} for arm in ARMS}


def _comparisons(metrics):
    value = {arm: float(metrics[arm]["foreground_miou"]) for arm in ARMS}
    return {
        "b_minus_a": 100 * (value["b"] - value["a"]),
        "d_minus_c": 100 * (value["d"] - value["c"]),
        "c_minus_a": 100 * (value["c"] - value["a"]),
        "d_minus_b": 100 * (value["d"] - value["b"]),
        "interaction_d_minus_c_minus_b_plus_a": 100 * (value["d"] - value["c"] - value["b"] + value["a"]),
    }


def _bind_baseline(args, recorded=None):
    """Recheck the earlier completed endpoint; never trust a copied evidence dict."""
    if args.smoke:
        args.baseline_evidence = None
        return
    if recorded is not None:
        if not isinstance(recorded, dict) or recorded.get("passed") is not True or not recorded.get("root"):
            raise ValueError("Formal checkpoint has no valid bound baseline evidence")
        args.baseline_root = recorded["root"]
    from gates import validate_baseline
    current = validate_baseline(args)
    previous = recorded if recorded is not None else getattr(args, "baseline_evidence", None)
    if previous is not None and current != previous:
        raise ValueError("Baseline evidence differs from the bound training evidence")
    args.baseline_evidence = current


def _formal_runtime(runtime, smoke):
    if not smoke:
        if str(runtime.device).split(":", 1)[0] != "cuda" or runtime.protocol_info.get("synthetic") is True:
            raise ValueError("Formal evidence requires the real BRIGHT CUDA runtime; CPU contracts are smoke only")


def _restored_guard(runtime):
    report = getattr(runtime, "guard_report", None)
    if report is not None:
        if report.get("restored") is not True or report.get("restore_conflicts"):
            raise RuntimeError("Runtime PoolGuard was not cleanly restored: " + str(report))
    return report


def _reopen_predictions(folder, journal, expected, transitions, names):
    """Re-read every saved array and independently reconstruct final evidence."""
    rebuilt = {path: np.zeros((4, 4), dtype=np.int64) for path in PATHS}
    rebuilt_transitions = {arm: np.zeros((4, 4, 4), dtype=np.int64) for arm in ARMS}
    rebuilt_events = {}
    seen = []
    for row in journal:
        path = folder / row["file"]
        if digest(path) != row["sha256"]:
            raise ValueError("Saved prediction hash changed: " + str(path))
        with np.load(path, allow_pickle=False) as archive:
            required = {"target"} | {"prediction_" + name for name in PATHS}
            if set(archive.files) != required:
                raise ValueError("Saved prediction schema mismatch")
            target = archive["target"]
            if target.dtype != np.uint8:
                raise ValueError("Saved target is not uint8")
            events = rebuilt_events.setdefault(row["event"],
                                               {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS})
            for name in PATHS:
                prediction = archive["prediction_" + name]
                if prediction.dtype != np.uint8:
                    raise ValueError("Saved prediction is not uint8")
                cm = _confusion(target, prediction)
                if cm.tolist() != row["confusion"][name]:
                    raise ValueError("Saved image confusion differs from journal")
                rebuilt[name] += cm
                events[name] += cm
            for arm in ARMS:
                rebuilt_transitions[arm] += _transition(
                    target, archive["prediction_grid_probq"], archive["prediction_" + arm])
        seen.append(row["image"])
    if seen != list(names):
        raise ValueError("Saved Val image identities/order do not match requested split")
    for name in PATHS:
        if not np.array_equal(rebuilt[name], expected[name]):
            raise ValueError("Reopened global confusion differs: " + name)
    for arm in ARMS:
        if not np.array_equal(rebuilt_transitions[arm], transitions[arm]):
            raise ValueError("Reopened transitions differ: " + arm)
    return rebuilt_events


def _evaluate(runtime, heads, names, folder, *, seed, smoke, initial=False, save_arrays=True):
    folder.mkdir(parents=True, exist_ok=False)
    for head in heads.values():
        head.eval()
    confusion = {path: np.zeros((4, 4), dtype=np.int64) for path in PATHS}
    transitions = {arm: np.zeros((4, 4, 4), dtype=np.int64) for arm in ARMS}
    events, journal, seen = {}, [], []
    initial_checks = {arm: {"passed": True, "max_abs": 0.0, "outside_tolerance": 0,
                            "prediction_differences": 0, "delta_grid_exact_zero": True,
                            "matched_initial_head_outputs": True} for arm in ARMS}
    coverage = _blank_coverage()
    start = time.monotonic()
    # no inference_mode: these are the same no_grad tensors the training path uses.
    with torch.no_grad(), (folder / "images.jsonl").open("w", encoding="utf-8") as log:
        for index, batch in enumerate(runtime.batches(names, train=False, seed=seed)):
            example = runtime.prepare(batch)
            _assert_example(example)
            if index >= len(names) or example.image != names[index]:
                raise ValueError("Evaluation loader did not preserve split order")
            _merge_coverage(coverage, _coverage(example))
            target = example.target.detach().cpu().numpy().astype(np.uint8)
            base = example.common["base_probability_full"]
            predictions = {"grid_probq": base.argmax(1).cpu().numpy().astype(np.uint8)}
            first_probability = None
            for arm in ARMS:
                decoded = heads[arm](**example.common, **example.supports[arm])
                probability = decoded["probabilities"]
                if not bool(torch.isfinite(probability).all()):
                    raise ValueError("Non-finite evaluated probabilities: " + arm)
                predictions[arm] = probability.argmax(1).cpu().numpy().astype(np.uint8)
                if initial:
                    difference = (probability - base).abs()
                    outside = int((difference > 2e-5 + 2e-5 * base.abs()).sum().item())
                    changed = int(np.count_nonzero(predictions[arm] != predictions["grid_probq"]))
                    check = initial_checks[arm]
                    check["max_abs"] = max(check["max_abs"], float(difference.max().item()))
                    check["outside_tolerance"] += outside
                    check["prediction_differences"] += changed
                    zero_delta = bool((decoded["delta_grid"] == 0).all())
                    if first_probability is None:
                        first_probability = probability.clone()
                    same_head_output = torch.equal(probability, first_probability)
                    check["delta_grid_exact_zero"] &= zero_delta
                    check["matched_initial_head_outputs"] &= same_head_output
                    # log(clamped P0) -> softmax can change an exact/near argmax tie.
                    # Record it; the preregistered probability test is the gate.
                    check["passed"] = check["passed"] and outside == 0 and zero_delta and same_head_output
                del decoded, probability
            current = {name: _confusion(target, prediction) for name, prediction in predictions.items()}
            event = events.setdefault(example.event, {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS})
            for name in PATHS:
                confusion[name] += current[name]
                event[name] += current[name]
            for arm in ARMS:
                transitions[arm] += _transition(target, predictions["grid_probq"], predictions[arm])
            row = {"index": index, "image": example.image, "event": example.event,
                   "confusion": {p: cm.tolist() for p, cm in current.items()},
                   "diagnostics": example.diagnostics}
            if save_arrays:
                relative = "predictions/%04d.npz" % index
                (folder / "predictions").mkdir(exist_ok=True)
                np.savez_compressed(folder / relative, target=target,
                                    **{"prediction_" + p: value for p, value in predictions.items()})
                row.update(file=relative, sha256=digest(folder / relative))
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            log.flush()
            journal.append(row)
            seen.append(example.image)
            if len(seen) % 10 == 0 or len(seen) == len(names):
                print("Joint Val images processed %d/%d (%s); final checks pending" %
                      (len(seen), len(names), folder.name), flush=True)
    if seen != list(names):
        raise ValueError("Evaluation ended before all requested images were processed")
    if save_arrays:
        events = _reopen_predictions(folder, journal, confusion, transitions, names)
    result = {"format": FORMAT, "phase": "initial_val" if initial else "evaluation",
              "image_count": len(seen), "execution_complete": True,
              "full_validation_evidence": False,
              "prediction_evidence_complete": save_arrays,
              "whole_val450_processed": not smoke and len(seen) == 450,
              "final_integrity_pending": True,
              "smoke": smoke, "test_accessed": False, "historical_reproduction_claim": False,
              "metrics": {p: _metrics(cm) for p, cm in confusion.items()},
              "per_event": {event: {p: _metrics(cm) for p, cm in arms.items()}
                            for event, arms in events.items()},
              "transitions": {p: _transition_report(cm) for p, cm in transitions.items()},
              "coverage": coverage, "initial_probability_checks": initial_checks if initial else None,
              "same_forward_baseline_deltas_pp": {arm: 100.0 * (_metrics(confusion[arm])["foreground_miou"]
                                                     - _metrics(confusion["grid_probq"])["foreground_miou"])
                                                   for arm in ARMS},
              "comparisons_pp": _comparisons({p: _metrics(cm) for p, cm in confusion.items()}),
              "saved_predictions_reopened": save_arrays, "elapsed_seconds": time.monotonic() - start}
    if initial and not all(row["passed"] for row in initial_checks.values()):
        result["execution_complete"] = False
        result["full_validation_evidence"] = False
        write_json(folder / "result.json", result)
        raise RuntimeError("Joint zero-initialization does not reproduce same-forward grid_probq")
    write_json(folder / "result.json", result)
    return result


def _checkpoint(folder, filename, heads, optimizers, step, args, protocol, initial_hashes, hashes,
                input_files):
    path = folder / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    record = {"format": FORMAT, "step": step, "mode": args.mode, "config": _config(args),
              "smoke": args.mode == "smoke", "training_complete": step == args.steps,
              "source_hash": hashes, "protocol_info": protocol,
              "initial_head_hashes": initial_hashes,
              "input_file_sha256": input_files,
              "heads": {arm: head.state_dict() for arm, head in heads.items()},
              "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
              "updates_per_arm": {arm: step for arm in ARMS},
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch_cpu": torch.get_rng_state(),
                      "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
              "test_accessed": False, "training_resume_supported": False}
    torch.save(record, temporary)
    os.replace(temporary, path)
    manifest = {"format": FORMAT, "file": filename, "sha256": digest(path), "step": step,
                "smoke": args.mode == "smoke", "training_complete": step == args.steps,
                "mode": args.mode, "source_hash": hashes, "protocol_info": protocol,
                "head_hashes": {arm: module_hash(head) for arm, head in heads.items()}}
    write_json(path.with_suffix(path.suffix + ".json"), manifest)
    return manifest


def _failure(output, summary, exc):
    summary.update(complete=False, execution_complete=False, full_validation_evidence=False,
                   error=type(exc).__name__ + ": " + str(exc))
    (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "summary.json", summary)


def run_training(args, runtime_factory=None):
    """Create a new matched run. Partial runs are preserved and never resumed."""
    output = _new_output(args.output)
    smoke = args.mode == "smoke"
    args.smoke = smoke
    summary = {"format": FORMAT, "phase": "initializing", "complete": False,
               "execution_complete": False, "full_validation_evidence": False,
               "smoke": smoke, "training": True, "test_accessed": False,
               "updates_per_arm": {arm: 0 for arm in ARMS}, "arguments": _config(args)}
    write_json(output / "summary.json", summary)
    try:
        if args.mode not in ("smoke", "train") or int(args.steps) < 1:
            raise ValueError("mode smoke/train and positive steps are required")
        if smoke and int(args.steps) != 20:
            raise ValueError("Smoke requires exactly 20 matched updates per arm")
        if not smoke and int(args.steps) != 5000:
            raise ValueError("Formal preregistered experiment requires 5000 updates per arm")
        if int(args.eval_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("eval_every and learning rate must be positive")
        _bind_baseline(args)
        summary["arguments"] = _config(args)
        _seed(int(args.seed))
        code_hash = source_hash()
        with _factory(runtime_factory)(args, output) as runtime:
            # Historical E0 loading restores its own seed; head initialization
            # belongs to this new experiment's independently declared seed.
            _seed(int(args.seed))
            _formal_runtime(runtime, smoke)
            train_names, val_names = _names(runtime, smoke)
            heads, initial_hashes = _new_heads(args, runtime.device)
            optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=args.lr,
                                                weight_decay=args.weight_decay)
                          for arm, head in heads.items()}
            protocol = copy.deepcopy(runtime.protocol_info)
            write_json(output / "manifest.json", {"format": FORMAT, "source_hash": code_hash,
                       "protocol_info": protocol, "arguments": _config(args),
                       "train_names": train_names, "val_names": val_names,
                       "initial_head_hashes": initial_hashes, "arm_rules": ARM_RULES,
                       "memory_arms": list(MEMORY_ARMS), "topk_unit": "parent cases; at most two child slots per parent",
                       "parameters_per_arm": {a: sum(p.numel() for p in h.parameters()) for a,h in heads.items()},
                       "claims": {"same_initialization": True, "shared_frozen_forward": True,
                                  "query_GT_not_model_input": True, "training_resume_supported": False}})
            summary["phase"] = "initial_val"
            write_json(output / "summary.json", summary)
            initial = _evaluate(runtime, heads, val_names, output / "initial_val", seed=args.seed,
                                smoke=smoke, initial=True)
            summary["initial_probability_checks"] = initial["initial_probability_checks"]
            summary["initial_integrity"] = _checked_verify(runtime)
            if any(initial["coverage"][arm]["supported_slot_queries"] == 0 for arm in MEMORY_ARMS):
                raise ValueError("Initial Val selection contains no legal historical supports")
            summary["phase"] = "training"
            write_json(output / "summary.json", summary)
            epoch, iterator = 0, iter(runtime.batches(train_names, train=True, seed=args.seed))
            coverage = _blank_coverage()
            train_seen = set()
            gradient_evidence = {arm: {"finite_updates": 0, "positive_gradient_updates": 0,
                                       "reader_positive_gradient_updates": 0} for arm in ARMS}
            start = time.monotonic()
            if str(runtime.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(runtime.device)
            checkpoint_folder = output / "checkpoints"
            checkpoint_folder.mkdir()
            with (output / "train.jsonl").open("w", encoding="utf-8") as journal:
                for step in range(1, int(args.steps) + 1):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        epoch += 1
                        iterator = iter(runtime.batches(train_names, train=True, seed=int(args.seed) + epoch))
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            raise ValueError("Training loader returned no batches")
                    example = runtime.prepare(batch)
                    _assert_example(example)
                    if example.image not in train_names:
                        raise ValueError("Training loader produced an image outside Train selection")
                    train_seen.add(example.image)
                    image_coverage = _coverage(example)
                    _merge_coverage(coverage, image_coverage)
                    lr = _lr(step, int(args.steps), float(args.lr))
                    row = {"step": step, "epoch": epoch, "image": example.image, "event": example.event,
                           "lr": lr, "heads": {}, "coverage": image_coverage}
                    for arm in ARMS:
                        head, optimizer = heads[arm], optimizers[arm]
                        head.train()
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        optimizer.zero_grad(set_to_none=True)
                        result = head(**example.common, **example.supports[arm])
                        loss = runtime.loss(result["probabilities"], example.target)
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise ValueError("Non-finite or non-scalar training loss: " + arm)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                        evidence = gradient_evidence[arm]
                        evidence["finite_updates"] += 1
                        evidence["positive_gradient_updates"] += int(float(grad_norm) > 0)
                        reader_grad = sum(float(p.grad.detach().abs().sum())
                                          for name,p in head.named_parameters()
                                          if p.grad is not None and (name.startswith("query_projection")
                                          or name.startswith("key_projection") or name.startswith("value_projection")
                                          or name.startswith("label_embedding") or name.startswith("context_projection")))
                        evidence["reader_positive_gradient_updates"] += int(reader_grad > 0)
                        optimizer.step()
                        if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
                            raise ValueError("Non-finite updated parameters: " + arm)
                        summary["updates_per_arm"][arm] = step
                        row["heads"][arm] = {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
                                             "reader_grad_abs_sum": reader_grad}
                        del result, loss
                    row["elapsed_seconds"] = time.monotonic() - start
                    row["peak_cuda_allocated_bytes"] = (int(torch.cuda.max_memory_allocated(runtime.device))
                                                         if str(runtime.device).startswith("cuda") else None)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    if step % 20 == 0 or step == int(args.steps):
                        print("Joint fit " + json.dumps(row, ensure_ascii=False), flush=True)
                    if step % int(args.eval_every) == 0 or step == int(args.steps):
                        name = "checkpoint_last.pt" if step == int(args.steps) else "checkpoint_%06d.pt" % step
                        manifest = _checkpoint(checkpoint_folder, name, heads, optimizers, step, args,
                                               protocol, initial_hashes, code_hash,
                                               {str(p): h for p,h in getattr(runtime, "input_ledger", {}).items()})
                        summary.update(last_checkpoint=str(checkpoint_folder / name), checkpoint=manifest)
                        if step == int(args.steps):
                            summary["training_complete"] = True
                        if step != int(args.steps):
                            summary["phase"] = "intermediate_val"
                            write_json(output / "summary.json", summary)
                            _evaluate(runtime, heads, val_names, output / ("val_step_%06d" % step),
                                      seed=args.seed, smoke=smoke, save_arrays=False)
                            summary["phase"] = "training"
                        write_json(output / "summary.json", summary)
                    del example
            if any(coverage[arm]["supported_slot_queries"] == 0 for arm in MEMORY_ARMS):
                raise ValueError("No legal historical support was consumed during training")
            if not smoke and train_seen != set(train_names):
                raise ValueError("Formal training did not visit all Train2918 query images")
            for arm in ARMS:
                if gradient_evidence[arm]["positive_gradient_updates"] == 0:
                    raise ValueError("No nonzero training gradient: " + arm)
                # Zero final layer blocks Reader gradients on the first update by design.
                if args.steps > 1 and gradient_evidence[arm]["reader_positive_gradient_updates"] == 0:
                    raise ValueError("Reader never received a nonzero gradient after initialization: " + arm)
            summary["phase"] = "final_val"
            write_json(output / "summary.json", summary)
            # Evaluate serialized last weights, not just the still-live modules.
            saved_last = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)
            if digest(summary["last_checkpoint"]) != summary["checkpoint"]["sha256"]:
                raise RuntimeError("Last checkpoint hash changed before final evaluation")
            for arm in ARMS:
                heads[arm].load_state_dict(saved_last["heads"][arm], strict=True)
            del saved_last
            final = _evaluate(runtime, heads, val_names, output / "final_val", seed=args.seed,
                              smoke=smoke, save_arrays=True)
            summary["phase"] = "final_integrity"
            write_json(output / "summary.json", summary)
            integrity = _checked_verify(runtime)
            if source_hash() != code_hash:
                raise RuntimeError("Training package code changed while experiment was running")
            final_manifest = summary["checkpoint"]
            if digest(summary["last_checkpoint"]) != final_manifest["sha256"]:
                raise RuntimeError("Saved final checkpoint changed")
            if any(module_hash(heads[a]) != final_manifest["head_hashes"][a] for a in ARMS):
                raise RuntimeError("Final heads differ from the saved last checkpoint")
            summary.update(phase="complete_smoke" if smoke else "complete", complete=True,
                           execution_complete=True, full_validation_evidence=not smoke,
                           image_count=len(val_names), train_image_count=len(train_seen),
                           training_coverage=coverage, gradient_evidence=gradient_evidence,
                           integrity=integrity, metrics=final["metrics"],
                           comparisons_pp=_comparisons(final["metrics"]),
                           elapsed_training_and_final_seconds=time.monotonic()-start)
        summary["guard_report"] = _restored_guard(runtime)
        fingerprints = {str(p): h for p,h in getattr(runtime, "input_ledger", {}).items()}
        write_json(output / "input_fingerprints.json", fingerprints)
        summary["input_fingerprints_sha256"] = digest(output / "input_fingerprints.json")
        final.update(final_integrity_pending=False, full_validation_evidence=not smoke,
                     primary_endpoint=not smoke, final_integrity=integrity,
                     guard_report=summary["guard_report"])
        write_json(output / "final_val" / "result.json", final)
        write_json(output / "summary.json", summary)
        print("JOINT %s COMPLETE. No Test. See %s" % ("SMOKE" if smoke else "TRAIN/VAL450", output), flush=True)
        return summary
    except BaseException as exc:
        _failure(output, summary, exc)
        raise


def run_evaluation(args, runtime_factory=None):
    """Re-evaluate one package checkpoint; no resume, optimization, or Test access."""
    output = _new_output(args.output)
    summary = {"format": FORMAT, "phase": "checkpoint_validation", "complete": False,
               "execution_complete": False, "full_validation_evidence": False,
               "training": False, "test_accessed": False, "arguments": _config(args)}
    write_json(output / "summary.json", summary)
    try:
        path = Path(args.checkpoint).expanduser().resolve()
        manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if manifest.get("format") != FORMAT or digest(path) != manifest["sha256"]:
            raise ValueError("Checkpoint manifest / file hash mismatch")
        if manifest["source_hash"] != source_hash():
            raise ValueError("Checkpoint belongs to a different training-code version")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT or saved.get("source_hash") != manifest["source_hash"]:
            raise ValueError("Checkpoint internal source binding mismatch")
        if saved.get("protocol_info") != manifest["protocol_info"]:
            raise ValueError("Checkpoint internal protocol binding mismatch")
        if (saved.get("step") != manifest.get("step")
                or saved.get("mode") != manifest.get("mode")
                or saved.get("training_complete") != manifest.get("training_complete")
                or saved.get("updates_per_arm") != {arm: saved["step"] for arm in ARMS}):
            raise ValueError("Checkpoint training-state binding mismatch")
        for key in ("topk", "chunk_size", "seed"):
            if getattr(args, key) != saved["config"][key]:
                raise ValueError("Evaluation %s must match checkpoint config" % key)
        if set(saved["heads"]) != set(ARMS):
            raise ValueError("Checkpoint must include all matched heads")
        if saved["mode"] not in ("smoke", "train"):
            raise ValueError("Unknown checkpoint training mode")
        smoke = saved["mode"] == "smoke"
        args.mode = saved["mode"]
        args.smoke = smoke
        if not smoke:
            recorded = saved.get("config", {}).get("baseline_evidence")
            if recorded is None:
                raise ValueError("Formal checkpoint has no bound baseline evidence")
            _bind_baseline(args, recorded=recorded)
        else:
            _bind_baseline(args)
        summary["smoke"] = smoke
        summary["arguments"] = _config(args)
        _seed(int(args.seed))
        with _factory(runtime_factory)(args, output) as runtime:
            _seed(int(args.seed))
            if runtime.protocol_info != saved["protocol_info"]:
                raise ValueError("Current V3 bank / frontend / dataset binding differs from checkpoint")
            _formal_runtime(runtime, smoke)
            _, val_names = _names(runtime, smoke)
            heads, _ = _new_heads(args, runtime.device)
            for arm in ARMS:
                heads[arm].load_state_dict(saved["heads"][arm], strict=True)
                if module_hash(heads[arm]) != manifest["head_hashes"][arm]:
                    raise ValueError("Loaded head hash differs from checkpoint manifest")
            summary["phase"] = "evaluation"
            write_json(output / "summary.json", summary)
            result = _evaluate(runtime, heads, val_names, output / "val", seed=args.seed,
                               smoke=smoke, save_arrays=True)
            primary_endpoint = not smoke and saved["step"] == 5000 and saved.get("training_complete") is True
            for p,h in getattr(runtime, "input_ledger", {}).items():
                if saved.get("input_file_sha256", {}).get(str(p)) != h:
                    raise RuntimeError("Evaluation input differs from saved training provenance: " + str(p))
            integrity = _checked_verify(runtime)
            if source_hash() != saved["source_hash"]:
                raise RuntimeError("Evaluation code changed while running")
            summary.update(phase="complete_smoke" if smoke else "complete", complete=True,
                           execution_complete=True, full_validation_evidence=primary_endpoint,
                           primary_endpoint=primary_endpoint, training_complete=saved.get("training_complete"),
                           image_count=len(val_names), checkpoint_sha256=manifest["sha256"],
                           checkpoint_step=saved["step"], metrics=result["metrics"],
                           comparisons_pp=_comparisons(result["metrics"]), integrity=integrity)
        summary["guard_report"] = _restored_guard(runtime)
        result.update(primary_endpoint=primary_endpoint,
                      full_validation_evidence=primary_endpoint, final_integrity_pending=False,
                      checkpoint_step=saved["step"], final_integrity=integrity,
                      guard_report=summary["guard_report"])
        write_json(output / "val" / "result.json", result)
        write_json(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _failure(output, summary, exc)
        raise
