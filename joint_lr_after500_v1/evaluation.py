"""Matched, RNG-neutral Train/Val evaluation for the post-500 LR experiment.

The four paths here are B/D x control/low LR, not the previous A/B/C/D.
The inherited Runtime owns the original loss and frozen BRIGHT forward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

BINDINGS = {
    "b_control": ("b", 1.0), "b_lr02": ("b", 0.2),
    "d_control": ("d", 1.0), "d_lr02": ("d", 0.2),
}
PATHS = ("grid_probq",) + tuple(BINDINGS)
PAIRS = {"b_lr02_vs_control": ("b_control", "b_lr02"),
         "d_lr02_vs_control": ("d_control", "d_lr02")}
PAIRS.update({name + "_vs_grid": ("grid_probq", name) for name in BINDINGS})


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def names_hash(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def confusion(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction shapes differ")
    if target.dtype.kind not in "iu" or prediction.dtype.kind not in "iu":
        raise ValueError("Integer target/prediction required")
    valid = target != 255
    if np.any((target[valid] < 0) | (target[valid] >= 4)):
        raise ValueError("Invalid nonignored target class")
    if np.any((prediction < 0) | (prediction >= 4)):
        raise ValueError("Invalid prediction class")
    return np.bincount(target[valid].astype(np.int64) * 4 +
                       prediction[valid].astype(np.int64), minlength=16).reshape(4, 4)


def transition(target, before, after):
    # Reuse validation, including predictions at ignored target locations.
    confusion(target, before)
    confusion(target, after)
    target, before, after = map(np.asarray, (target, before, after))
    valid = target != 255
    code = target[valid].astype(np.int64) * 16 + before[valid].astype(np.int64) * 4 + after[valid]
    return np.bincount(code, minlength=64).reshape(4, 4, 4)


def binary_foreground(confusion_matrix):
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    if cm.shape != (4, 4) or np.any(cm < 0):
        raise ValueError("Expected nonnegative 4x4 confusion")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1:].sum()), int(cm[1:, 0].sum()), int(cm[1:, 1:].sum())
    return {"confusion": [[tn, fp], [fn, tp]],
            "axes": ["GT background/foreground", "prediction background/foreground"],
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "predicted_foreground_pixels": tp + fp,
            "target_foreground_pixels": tp + fn}


def transition_report(counts, before, after):
    a = np.asarray(counts, dtype=np.int64)
    if a.shape != (4, 4, 4) or np.any(a < 0):
        raise ValueError("Expected nonnegative GT x before x after transition")
    rescue = [int(sum(a[c, old, c] for old in range(4) if old != c)) for c in range(4)]
    harm = [int(sum(a[c, c, new] for new in range(4) if new != c)) for c in range(4)]
    return {
        "axes": ["GT", before, after], "counts": a.tolist(),
        "rescued_by_gt_class": rescue, "harmed_by_gt_class": harm,
        "net_correct_change_by_gt_class": [r - h for r, h in zip(rescue, harm)],
        "changed_valid_pixels": int(sum(a[:, p, q].sum() for p in range(4) for q in range(4) if p != q)),
        "damaged_correct_to_background": int(a[2, 2, 0]),
        "damaged_background_to_correct": int(a[2, 0, 2]),
        "damaged_intact_to_correct": int(a[2, 1, 2]),
        "damaged_correct_to_intact": int(a[2, 2, 1]),
        "damaged_predicted_background_before": int(a[2, 0, :].sum()),
        "damaged_predicted_background_after": int(a[2, :, 0].sum()),
        "damaged_to_background_net_change": int(a[2, :, 0].sum() - a[2, 0, :].sum()),
        "correct_intact_changed_to_damaged": int(a[1, 1, 2]),
        "interpretation": "Joint pixel transitions; rescue/harm are paired counts, not differences of marginal confusion matrices.",
    }


def _check_exclusion(runtime, example):
    d = example.diagnostics
    if d.get("query_GT_used_for_supports") is not False:
        raise ValueError("Query GT separation assertion missing or false")
    counts = {}
    for arm in ("c", "d"):
        ids = example.supports[arm]["support_ids"]
        if ids.dtype != torch.long or ids.ndim != 2:
            raise ValueError("Historical support IDs must be int64 [query,K]")
        bank_count = len(runtime.bank["events"])
        if bool(((ids < -1) | (ids >= bank_count)).any()):
            raise ValueError("Historical support ID out of range")
        chosen = ids[ids >= 0]
        for field, key in (("events", "event_id"), ("images", "image_id")):
            query_id = d[key]
            if query_id >= 0 and bool((runtime.bank[field][chosen] == query_id).any()):
                raise ValueError("Historical support includes query " + field)
        if "eligible" in runtime.bank and chosen.numel() and not bool(runtime.bank["eligible"][chosen].all()):
            raise ValueError("Selected ineligible support")
        counts[arm] = int(chosen.numel())
    return counts


def _reopen_predictions(folder, journal, expected, expected_events, expected_transitions, seen):
    rebuilt = {name: np.zeros((4, 4), dtype=np.int64) for name in PATHS}
    rebuilt_events = {}
    rebuilt_transitions = {name: np.zeros((4, 4, 4), dtype=np.int64) for name in PAIRS}
    image_names = []
    for row in journal:
        path = Path(folder) / row["file"]
        if _digest(path) != row["sha256"]:
            raise ValueError("Saved prediction hash mismatch: " + str(path))
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"target"} | {"prediction_" + p for p in PATHS}:
                raise ValueError("Saved prediction array schema mismatch")
            if any(arrays[k].dtype != np.uint8 for k in arrays.files):
                raise ValueError("Saved labels/predictions must be uint8")
            target = arrays["target"]
            events = rebuilt_events.setdefault(row["event"], {n: np.zeros((4, 4), dtype=np.int64) for n in PATHS})
            for name in PATHS:
                cm = confusion(target, arrays["prediction_" + name])
                if cm.tolist() != row["confusion"][name]:
                    raise ValueError("Reopened image confusion differs: " + name)
                rebuilt[name] += cm
                events[name] += cm
            for pair, (before, after) in PAIRS.items():
                rebuilt_transitions[pair] += transition(target, arrays["prediction_" + before], arrays["prediction_" + after])
        image_names.append(row["image"])
    if image_names != list(seen):
        raise ValueError("Reopened image identity/order differs")
    for name in PATHS:
        if not np.array_equal(rebuilt[name], expected[name]):
            raise ValueError("Reopened global confusion differs: " + name)
    if set(rebuilt_events) != set(expected_events):
        raise ValueError("Reopened event identities differ")
    for event in expected_events:
        for name in PATHS:
            if not np.array_equal(rebuilt_events[event][name], expected_events[event][name]):
                raise ValueError("Reopened event confusion differs")
    for name in PAIRS:
        if not np.array_equal(rebuilt_transitions[name], expected_transitions[name]):
            raise ValueError("Reopened paired transitions differ: " + name)


def evaluate(runtime, heads, names, folder, *, step, split, seed, original_train,
             save_predictions=False):
    """Evaluate identical frozen examples for all paths and restore RNG/modes.

    ``train=True`` selects the fixed Train split, not head training mode. The
    original loader uses full images; Train ordering may be shuffled. The
    supplied names and their observed order are both fingerprinted.
    """
    from continuation import capture_rng, restore_rng

    if split not in ("val", "fixed_train"):
        raise ValueError("Only val/fixed_train are supported; Test is never accessed")
    if set(heads) != set(BINDINGS):
        raise ValueError("Expected B/D x control/low-LR heads")
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Evaluation identities must be nonempty and unique")
    allowed = runtime.val_names if split == "val" else runtime.train_names
    if not set(names) <= set(allowed):
        raise ValueError("Requested evaluation crosses declared split")
    if save_predictions and split != "val":
        raise ValueError("Prediction archives are reserved for Val")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if save_predictions:
        (folder / "predictions").mkdir()
    rng = capture_rng()
    modes = {name: head.training for name, head in heads.items()}
    cm = {name: np.zeros((4, 4), dtype=np.int64) for name in PATHS}
    totals = {name: {"loss": 0.0, "nll": 0.0, "lovasz": 0.0} for name in PATHS}
    transitions = {name: np.zeros((4, 4, 4), dtype=np.int64) for name in PAIRS}
    coverage = {"c": 0, "d": 0}
    events, event_totals, event_counts, journal, seen = {}, {}, {}, [], []
    seen_set = set()
    valid_pixels = 0
    start = time.monotonic()
    try:
        for head in heads.values():
            head.eval()
        with torch.no_grad(), (folder / "images.jsonl").open("x", encoding="utf-8") as log:
            for index, batch in enumerate(runtime.batches(names, train=split == "fixed_train", seed=seed)):
                example = runtime.prepare(batch)
                original_train._assert_example(example)
                if example.image not in names or example.image in seen_set:
                    raise ValueError("Unknown or duplicate evaluation identity")
                if split == "val" and (index >= len(names) or example.image != names[index]):
                    raise ValueError("Val loader did not preserve declared order")
                for arm, count in _check_exclusion(runtime, example).items():
                    coverage[arm] += count
                target = example.target.detach().cpu().numpy()
                image_valid_pixels = int(np.count_nonzero(target != 255))
                valid_pixels += image_valid_pixels
                event = events.setdefault(example.event, {name: np.zeros((4, 4), dtype=np.int64) for name in PATHS})
                event_losses = event_totals.setdefault(example.event, {name: {key: 0.0 for key in ("loss", "nll", "lovasz")} for name in PATHS})
                event_counts[example.event] = event_counts.get(example.event, 0) + 1
                row = {"index": index, "image": example.image, "event": example.event,
                       "valid_pixels": image_valid_pixels, "loss": {}, "nll": {}, "lovasz": {}, "confusion": {}}
                predictions = {}
                for name in PATHS:
                    if name == "grid_probq":
                        value = example.common["base_probability_full"]
                    else:
                        original_arm = BINDINGS[name][0]
                        value = heads[name](**example.common, **example.supports[original_arm])["probabilities"]
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite probability: " + name)
                    if value.shape != (1, 4, *example.target.shape[1:]):
                        raise ValueError("Probability shape differs from full-image target")
                    loss = runtime.loss(value, example.target)
                    nll = runtime.deps.region_nll(value, example.target) if image_valid_pixels else value.sum() * 0
                    pieces = {"loss": loss, "nll": nll, "lovasz": loss - nll}
                    for key, scalar in pieces.items():
                        if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
                            raise ValueError("Nonfinite/nonscalar objective: " + key)
                        number = float(scalar)
                        row[key][name] = number
                        totals[name][key] += number
                        event_losses[name][key] += number
                    pred = value.argmax(1).detach().cpu().numpy().astype(np.uint8)
                    predictions[name] = pred
                    image_cm = confusion(target, pred)
                    cm[name] += image_cm
                    event[name] += image_cm
                    row["confusion"][name] = image_cm.tolist()
                    del value, loss, nll, pieces
                for pair, (before, after) in PAIRS.items():
                    transitions[pair] += transition(target, predictions[before], predictions[after])
                if save_predictions:
                    relative = "predictions/%04d.npz" % index
                    np.savez_compressed(folder / relative, target=target.astype(np.uint8),
                                        **{"prediction_" + name: pred for name, pred in predictions.items()})
                    row.update(file=relative, sha256=_digest(folder / relative))
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                journal.append(row)
                seen.append(example.image)
                seen_set.add(example.image)
                if len(seen) % 10 == 0 or len(seen) == len(names):
                    print("LR DIAGNOSTIC step=%d split=%s %d/%d" % (step, split, len(seen), len(names)), flush=True)
        if len(seen) != len(names) or seen_set != set(names):
            raise ValueError("Evaluation did not finish exact requested identities")
        if save_predictions:
            _reopen_predictions(folder, journal, cm, events, transitions, seen)
        def metrics(matrix, sums, count):
            row = original_train._metrics(matrix)
            row.update({"mean_image_" + key: value / count for key, value in sums.items()})
            row["binary_foreground"] = binary_foreground(matrix)
            return row
        full_val = (split == "val" and len(seen) == 450 and names == list(runtime.val_names)
                    and str(runtime.device).startswith("cuda")
                    and runtime.protocol_info.get("synthetic") is not True)
        result = {
            "format": "SRAC_JOINT_LR_AFTER500_EVALUATION_V1", "step": int(step), "split": split,
            "phase": "evaluation_complete_integrity_pending", "complete": False,
            "execution_complete": True, "full_validation_evidence": False,
            "whole_val450_processed": full_val, "final_integrity_pending": True,
            "smoke": not full_val if split == "val" else bool(runtime.protocol_info.get("synthetic")),
            "test_accessed": False, "training": False, "image_count": len(seen),
            "names_sha256": names_hash(sorted(seen)), "order_sha256": names_hash(seen),
            "requested_order_sha256": names_hash(names), "valid_pixel_count": valid_pixels,
            "fixed_train_scope": "Declared fixed subset only; not full Train2918 evidence" if split == "fixed_train" else None,
            "loss_definition": "Original region_nll + Lovasz(classes=present,ignore=255); arithmetic image mean, including zero for all-ignore images",
            "metrics": {name: metrics(cm[name], totals[name], len(seen)) for name in PATHS},
            "per_event": {event: {name: metrics(values[name], event_totals[event][name], event_counts[event]) for name in PATHS} for event, values in events.items()},
            "per_event_image_count": event_counts,
            "transitions": {name: transition_report(a, *PAIRS[name]) for name, a in transitions.items()},
            "query_GT_not_support_input": True, "same_image_event_exclusion_passed": True,
            "historical_parent_selections": coverage,
            "prediction_evidence_complete": bool(save_predictions),
            "saved_predictions_reopened": bool(save_predictions),
            "journal_sha256": _digest(folder / "images.jsonl"),
            "elapsed_seconds": time.monotonic() - start,
        }
        result["comparisons_pp"] = metriccontrast(result)
        _write_json(folder / "result.json", result)
        return result
    finally:
        for name, head in heads.items():
            head.train(modes[name])
        restore_rng(rng)


def compare_initial_to_old(result, old_result, tolerance_pp=0.02):
    """Compare all class IoUs and fg mIoU to saved original B/D/grid Val."""
    if not np.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("Finite nonnegative pp tolerance required")
    checks = []
    for path in PATHS:
        source = "grid_probq" if path == "grid_probq" else BINDINGS[path][0]
        current, old = result["metrics"][path], old_result["metrics"][source]
        for key, a, b in [("foreground_miou", current["foreground_miou"], old["foreground_miou"])] + [
                ("iou_%d" % i, current["iou"][i], old["iou"][i]) for i in range(4)]:
            if a is None or b is None:
                delta, passed = None, a is None and b is None
            else:
                delta = 100 * (float(a) - float(b))
                passed = bool(np.isfinite(delta) and abs(delta) <= tolerance_pp)
            checks.append({"path": path, "old_path": source, "metric": key,
                           "delta_pp": delta, "passed": passed})
    same_count = result.get("image_count") == old_result.get("image_count")
    valid_evaluation = result.get("execution_complete") is True and old_result.get("execution_complete") is True
    return {"passed": same_count and valid_evaluation and all(row["passed"] for row in checks),
            "same_image_count": same_count, "evaluation_execution_complete": valid_evaluation,
            "tolerance_pp": tolerance_pp, "checks": checks,
            "max_abs_delta_pp": max((abs(row["delta_pp"]) for row in checks if row["delta_pp"] is not None), default=0.0),
            "claim": "Metric replay tolerance only; exact per-pixel equality is not claimed"}


def metriccontrast(result):
    metrics = result["metrics"]
    comparisons = {"b_lr02_minus_b_control": ("b_lr02", "b_control"),
                   "d_lr02_minus_d_control": ("d_lr02", "d_control"),
                   "d_minus_b_control": ("d_control", "b_control"),
                   "d_minus_b_lr02": ("d_lr02", "b_lr02")}
    comparisons.update({name + "_minus_grid_probq": (name, "grid_probq") for name in BINDINGS})
    return {key: 100 * (metrics[a]["foreground_miou"] - metrics[b]["foreground_miou"])
            for key, (a, b) in comparisons.items()}


def print_table(result):
    print("step=%s split=%s images=%s; final_integrity_pending=%s" % (
        result["step"], result["split"], result["image_count"], result.get("final_integrity_pending")), flush=True)
    print("path | loss | NLL | Lovasz | fg mIoU % | Damaged IoU % | precision % | recall %", flush=True)
    def percent(value):
        return "NA" if value is None else "%.4f" % (100 * value)
    for name in PATHS:
        r = result["metrics"][name]
        print("%s | %.6f | %.6f | %.6f | %s | %s | %s | %s" % (
            name, r["mean_image_loss"], r["mean_image_nll"], r["mean_image_lovasz"],
            percent(r["foreground_miou"]), percent(r["iou"][2]),
            percent(r["precision"][2]), percent(r["recall"][2])), flush=True)
    print("contrasts in percentage points: " + json.dumps(metriccontrast(result)), flush=True)
