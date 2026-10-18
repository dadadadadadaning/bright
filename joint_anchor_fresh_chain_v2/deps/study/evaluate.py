"""RNG-neutral paired evaluation; the original BRIGHT metrics remain authoritative.

Only Train/Val forwards are supported.  Per-image confusion and paired transition
counts are retained; full-resolution predictions are intentionally not archived.
The runner must finish its provenance/model checks before promoting a report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from anchor import anchor_terms, serialized_anchor_terms

PATHS = ("grid_probq", "control", "anchor")
PAIRS = {"control_vs_grid": ("grid_probq", "control"),
         "anchor_vs_grid": ("grid_probq", "anchor"),
         "anchor_vs_control": ("control", "anchor")}


def names_hash(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _add_numeric(totals, values):
    """Accumulate scalar/vector sufficient statistics without guessing averages."""
    for key, value in values.items():
        if key in ("anchor_probability_floor", "anchor_used_student_logits"):
            continue
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError("Nonfinite report term: " + key)
            totals[key] = totals.get(key, 0.0) + value
        elif isinstance(value, list):
            array = np.asarray([0.0 if x is None else x for x in value])
            if array.dtype.kind not in "biuf" or array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError("Unexpected report vector: " + key)
            totals[key] = totals.get(key, np.zeros_like(array, dtype=np.float64)) + array
        else:
            raise TypeError("Unexpected numeric report term: " + key)


def _mean_terms(totals, count):
    result = {"mean_image_" + key: ((value / count).tolist() if isinstance(value, np.ndarray)
                                 else float(value) / count)
            for key, value in totals.items() if key != "anchor_per_class_conditional_kl"}
    if "anchor_per_class_conditional_kl" in totals:
        present = totals["anchor_class_present_image_counts"]
        result["mean_present_image_anchor_per_class_conditional_kl"] = [
            float(value / n) if n else None
            for value, n in zip(totals["anchor_per_class_conditional_kl"], present)]
    return result


def _paired_report(counts, before, after, original_train):
    report = original_train._transition_report(counts)
    report["axes"] = ["GT", before, after]
    report["both_correct_by_gt_class"] = [int(counts[c, c, c]) for c in range(4)]
    report["before_wrong_by_gt_class"] = [int(counts[c].sum() - counts[c, c].sum())
                                          for c in range(4)]
    report["before_correct_by_gt_class"] = [int(counts[c, c].sum()) for c in range(4)]
    report["correct_intact_changed_to_destroyed"] = int(counts[1, 1, 3])
    report["correct_damaged_changed_to_destroyed"] = int(counts[2, 2, 3])
    report["correct_damaged_changed_to_background"] = int(counts[2, 2, 0])
    report["interpretation"] = "Paired same-pixel transitions, not differences of marginal confusion matrices."
    return report


def _check_journal(path, names, expected_cm, expected_pairs, expected_events):
    """Independently reopen the compact evidence and verify identities/counts."""
    cm = {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS}
    pairs = {p: np.zeros((4, 4, 4), dtype=np.int64) for p in PAIRS}
    events, seen = {}, []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            seen.append(row["image"])
            event = events.setdefault(row["event"], {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS})
            for name in PATHS:
                a = np.asarray(row["confusion"][name], dtype=np.int64)
                if a.shape != (4, 4) or (a < 0).any() or int(a.sum()) != row["valid_pixels"]:
                    raise ValueError("Invalid saved image confusion")
                cm[name] += a
                event[name] += a
            for pair, (before, after) in PAIRS.items():
                a = np.asarray(row["paired_counts"][pair], dtype=np.int64)
                if a.shape != (4, 4, 4) or (a < 0).any():
                    raise ValueError("Invalid saved paired counts")
                if not np.array_equal(a.sum(2), row["confusion"][before]) or not np.array_equal(a.sum(1), row["confusion"][after]):
                    raise ValueError("Saved transitions disagree with marginal confusion")
                pairs[pair] += a
    if seen != names:
        raise ValueError("Reopened journal identity/order changed")
    for name in PATHS:
        if not np.array_equal(cm[name], expected_cm[name]):
            raise ValueError("Reopened journal total confusion changed")
    for pair in PAIRS:
        if not np.array_equal(pairs[pair], expected_pairs[pair]):
            raise ValueError("Reopened journal paired counts changed")
    if set(events) != set(expected_events):
        raise ValueError("Reopened journal event identities changed")
    for event, tables in events.items():
        for name in PATHS:
            if not np.array_equal(tables[name], expected_events[event][name]):
                raise ValueError("Reopened journal event confusion changed")


def evaluate(runtime, heads, names, folder, *, step, split, seed, arm="d",
             weight=.1, original_train, legacy):
    """Score both objectives and all three paths using identical frozen inputs.

    ``legacy`` exposes ``state``, ``objectives`` and ``study`` from the pinned
    loss-study dependency.  No inherited model, loss or metric code is copied.
    """
    if split not in ("val", "fixed_train"):
        raise ValueError("Only val/fixed_train supported; Test is never accessed")
    if arm not in ("b", "d") or set(heads) != {"control", "anchor"}:
        raise ValueError("Expected matched control/anchor heads for arm b or d")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Anchor weight must be finite and nonnegative")
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Evaluation identities must be nonempty and unique")
    if set(runtime.train_names) & set(runtime.val_names):
        raise ValueError("Runtime Train and Val identities overlap")
    allowed = runtime.val_names if split == "val" else runtime.train_names
    if not set(names) <= set(allowed):
        raise ValueError("Evaluation identities cross declared split")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rng = legacy.state.capture_rng()
    rng_hash = legacy.state.state_fingerprint(rng)
    modes = {n: [(module, module.training) for module in head.modules()]
             for n, head in heads.items()}
    head_hashes = {n: legacy.state.state_fingerprint(head.state_dict()) for n, head in heads.items()}
    cm = {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS}
    pairs = {p: np.zeros((4, 4, 4), dtype=np.int64) for p in PAIRS}
    totals = {p: {} for p in PATHS}
    events, event_totals, event_counts, seen = {}, {}, {}, []
    seen_set, valid_pixels, start = set(), 0, time.monotonic()
    try:
        for head in heads.values():
            head.eval()
        with torch.no_grad(), (folder / "images.jsonl").open("x", encoding="utf-8") as log:
            for index, batch in enumerate(runtime.batches(names, train=split == "fixed_train", seed=seed)):
                example = runtime.prepare(batch)
                original_train._assert_example(example)
                coverage = legacy.study.validate_example(example, runtime, original_train)
                if example.image not in names or example.image in seen_set:
                    raise ValueError("Unknown or duplicate evaluation image")
                if split == "val" and (index >= len(names) or example.image != names[index]):
                    raise ValueError("Val loader did not preserve declared order")
                target = example.target.detach().cpu().numpy()
                nvalid = int(np.count_nonzero(target != 255))
                valid_pixels += nvalid
                teacher = example.common["base_probability_full"]
                event = events.setdefault(example.event, {p: np.zeros((4, 4), dtype=np.int64) for p in PATHS})
                event_sums = event_totals.setdefault(example.event, {p: {} for p in PATHS})
                event_counts[example.event] = event_counts.get(example.event, 0) + 1
                row = dict(index=index, image=example.image, event=example.event,
                           valid_pixels=nvalid, confusion={}, objectives={}, paired_counts={},
                           coverage=coverage)
                predictions = {}
                for path in PATHS:
                    output = None if path == "grid_probq" else heads[path](**example.common, **example.supports[arm])
                    probability = teacher if output is None else output["probabilities"]
                    logits = None if output is None else output["logits"]
                    if probability.shape != (1, 4, *target.shape[1:]) or not bool(torch.isfinite(probability).all()):
                        raise ValueError("Invalid full-image probabilities: " + path)
                    exact_loss = runtime.loss(probability, example.target)
                    pieces = legacy.objectives.serialized_terms(legacy.objectives.objective_terms(
                        probability, example.target, reference_loss=exact_loss, verify_reference=index == 0))
                    # Do not carry the unrelated no-background-Lovasz candidate
                    # into this study's objective names.
                    values = {key: pieces[key] for key in ("nll", "lovasz", "class_nll_contribution",
                              "class_lovasz_contribution", "class_valid_counts", "valid_count")}
                    values["original_loss"] = float(exact_loss.detach())
                    diagnostics = serialized_anchor_terms(anchor_terms(
                        probability, teacher, example.target, weight,
                        student_logits=logits))
                    values.update({"anchor_" + key: value for key, value in diagnostics.items()})
                    values["candidate_loss"] = values["original_loss"] + float(diagnostics["weighted_kl"])
                    if values["valid_count"] != nvalid:
                        raise ValueError("Loss and confusion valid-pixel counts differ")
                    row["objectives"][path] = values
                    _add_numeric(totals[path], values)
                    _add_numeric(event_sums[path], values)
                    pred = probability.argmax(1).detach().cpu().numpy()
                    image_cm = original_train._confusion(target, pred)
                    cm[path] += image_cm
                    event[path] += image_cm
                    row["confusion"][path] = image_cm.tolist()
                    predictions[path] = pred
                for pair, (before, after) in PAIRS.items():
                    counts = original_train._transition(target, predictions[before], predictions[after])
                    pairs[pair] += counts
                    row["paired_counts"][pair] = counts.tolist()
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                seen.append(example.image)
                seen_set.add(example.image)
                if len(seen) % 10 == 0 or len(seen) == len(names):
                    print(f"ANCHOR DIAGNOSTIC step={step} split={split} {len(seen)}/{len(names)}", flush=True)
        if len(seen) != len(names) or seen_set != set(names):
            raise ValueError("Evaluation did not finish exact requested identities")
        _check_journal(folder / "images.jsonl", seen, cm, pairs, events)
    finally:
        for states in modes.values():
            for module, training in states:
                module.training = training
        legacy.state.restore_rng(rng)
    if rng_hash != legacy.state.state_fingerprint(legacy.state.capture_rng()):
        raise RuntimeError("Evaluation did not restore RNG state")
    if any(head_hashes[n] != legacy.state.state_fingerprint(h.state_dict()) for n, h in heads.items()):
        raise RuntimeError("Evaluation changed trainable head parameters/buffers")

    def metrics(matrix, sums, count):
        if not np.array_equal(np.asarray(sums["class_valid_counts"], dtype=np.int64), matrix.sum(1)):
            raise ValueError("Class objective counts disagree with confusion")
        report = original_train._metrics(matrix)
        report.update(_mean_terms(sums, count))
        report["class_valid_counts"] = np.asarray(sums["class_valid_counts"], dtype=np.int64).tolist()
        report["loss_aggregation"] = "Arithmetic image mean over all images, including all-ignore zero contributions; class NLL/Lovasz fields are contributions, not conditional means."
        report["anchor_aggregation"] = "mean_image_anchor_* fields average per-image diagnostics; optional conditional statistics retain their valid-count denominators in images.jsonl."
        return report

    full_val = (split == "val" and len(names) == 450 and names == list(runtime.val_names)
                and str(runtime.device).startswith("cuda")
                and runtime.protocol_info.get("synthetic") is not True)
    result = dict(format="SRAC_JOINT_ANCHOR_EVALUATION_V1", step=int(step), split=split,
                  phase="evaluation_complete_integrity_pending", complete=False,
                  execution_complete=True, full_validation_evidence=False,
                  whole_val450_processed=full_val, final_integrity_pending=True,
                  primary_endpoint=False, smoke=not full_val if split == "val" else bool(runtime.protocol_info.get("synthetic")),
                  training=False, test_accessed=False, image_count=len(seen), valid_pixel_count=valid_pixels,
                  arm=arm, anchor_weight=float(weight), class_names=["background", "intact", "damaged", "destroyed"],
                  names_sha256=names_hash(sorted(seen)), order_sha256=names_hash(seen),
                  requested_order_sha256=names_hash(names),
                  fixed_train_scope="Declared fixed subset only; not full Train2918 evidence" if split == "fixed_train" else None,
                  trained_objective={"grid_probq": "frozen", "control": "original_nll_plus_lovasz",
                                     "anchor": "original_nll_plus_lovasz_plus_teacher_KL"},
                  candidate_loss_definition="Exact original runtime.loss + weight * KL(stopgrad(grid_probq) || joint); all valid pixels; no Memory rejection gate",
                  reference_formula_verified_on_first_image=True,
                  metrics={p: metrics(cm[p], totals[p], len(seen)) for p in PATHS},
                  per_event={e: {p: metrics(tables[p], event_totals[e][p], event_counts[e]) for p in PATHS}
                             for e, tables in events.items()},
                  per_event_image_count=event_counts,
                  transitions={p: _paired_report(counts, *PAIRS[p], original_train) for p, counts in pairs.items()},
                  query_GT_not_support_input=True, same_image_event_exclusion_passed=True,
                  prediction_arrays_saved=False, paired_counts_reopened=True,
                  head_state_unchanged=True, rng_restored=True,
                  journal_sha256=digest(folder / "images.jsonl"), elapsed_seconds=time.monotonic() - start)
    result["comparisons_pp"] = {name: 100 * (result["metrics"][after]["foreground_miou"] - result["metrics"][before]["foreground_miou"])
                                 for name, (before, after) in PAIRS.items()}
    _write_json(folder / "result.json", result)
    (folder / "result.json.sha256").write_text(digest(folder / "result.json") + "  result.json\n", encoding="ascii")
    return result


def print_table(result):
    print(f"step={result['step']} split={result['split']} images={result['image_count']}; final_integrity_pending={result['final_integrity_pending']}")
    print("All paths are scored with BOTH objectives; lower KL or candidate loss alone is not a benefit.")
    print("path | original loss | original+anchor loss | NLL | Lovasz | KL | fg mIoU % | Damaged IoU % | precision % | recall %")
    def f(value, scale=1.):
        return "null" if value is None else f"{value * scale:.6f}"
    for path in PATHS:
        m = result["metrics"][path]
        values = [m["mean_image_original_loss"], m["mean_image_candidate_loss"],
                  m["mean_image_nll"], m["mean_image_lovasz"], m.get("mean_image_anchor_kl")]
        print(path + " | " + " | ".join([f(v) for v in values] +
              [f(m["foreground_miou"], 100), f(m["iou"][2], 100),
               f(m["precision"][2], 100), f(m["recall"][2], 100)]))
    print("Paired foreground mIoU differences (percentage points):", result["comparisons_pp"])
