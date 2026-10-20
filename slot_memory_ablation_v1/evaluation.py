"""Paired, RNG-neutral evaluation for arbitrary slot/Memory ablation arms.

The pinned BRIGHT metrics and original NLL + present-class Lovasz remain
authoritative. Every arm receives the same single frozen image extraction.
Reports remain incomplete until the runner finishes its integrity checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

BASELINE = "grid_probq"
ARMS = ("e", "f", "b", "d", "a", "c", "r", "l", "n")
# Each tuple is (after, before), so a positive contrast favors its first arm.
CONTRASTS = (("b", "e"), ("d", "f"), ("f", "e"), ("d", "b"),
             ("c", "a"), ("b", "a"), ("d", "c"), ("d", "r"),
             ("d", "l"), ("d", "n"))
SCALAR_TERMS = ("original_loss", "nll", "lovasz")
VECTOR_TERMS = ("class_nll_contribution", "class_lovasz_contribution")
COUNT_TERMS = ("class_valid_counts", "class_present_image_counts", "valid_count")


def names_hash(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def comparison_pairs(arms):
    """Return comparison key -> (before, after), restricted to present arms."""
    present = set(arms)
    pairs = {f"{after}_minus_{before}": (before, after)
             for after, before in CONTRASTS if {before, after} <= present}
    pairs.update({f"{arm}_minus_{BASELINE}": (BASELINE, arm) for arm in arms})
    return pairs


def metric_contrasts(metrics, pairs):
    return {key: 100.0 * (metrics[after]["foreground_miou"] -
                         metrics[before]["foreground_miou"])
            for key, (before, after) in pairs.items()}


def interaction_contrasts(metrics):
    """Difference-in-differences on aggregate foreground mIoU, in pp."""
    result = {}
    for label, arms in (("(d-b)-(f-e)", ("d", "b", "f", "e")),
                        ("(d-c)-(b-a)", ("d", "c", "b", "a"))):
        if set(arms) <= set(metrics):
            d, b, f, e = (metrics[arm]["foreground_miou"] for arm in arms)
            result[label] = 100.0 * ((d - b) - (f - e))
    return result


def _add_terms(totals, values):
    for key in SCALAR_TERMS + VECTOR_TERMS + COUNT_TERMS:
        value = values[key]
        if isinstance(value, list):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (4,) or not np.isfinite(array).all():
                raise ValueError("Invalid objective vector: " + key)
            totals[key] = totals.get(key, np.zeros(4, dtype=np.float64)) + array
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid objective scalar: " + key)
            totals[key] = totals.get(key, 0.0) + value


def _metrics(matrix, sums, count, original_train):
    if count <= 0:
        raise ValueError("Metrics require at least one evaluated image")
    if not np.array_equal(np.asarray(sums["class_valid_counts"], dtype=np.int64), matrix.sum(1)):
        raise ValueError("Class objective counts disagree with confusion")
    if int(sums["valid_count"]) != int(matrix.sum()):
        raise ValueError("Objective valid count disagrees with confusion")
    result = original_train._metrics(matrix)
    for key in SCALAR_TERMS + VECTOR_TERMS:
        value = sums[key] / count
        result["mean_image_" + key] = value.tolist() if isinstance(value, np.ndarray) else float(value)
    for key in ("class_valid_counts", "class_present_image_counts"):
        result[key] = np.asarray(sums[key], dtype=np.int64).tolist()
    result["loss_aggregation"] = (
        "Arithmetic image mean, including all-ignore zero contributions; class "
        "NLL/Lovasz values are additive contributions, not conditional means."
    )
    return result


def _paired_report(counts, before, after, original_train):
    result = original_train._transition_report(counts)
    result["axes"] = ["GT", before, after]
    result["both_correct_by_gt_class"] = [int(counts[c, c, c]) for c in range(4)]
    result["before_correct_by_gt_class"] = [int(counts[c, c].sum()) for c in range(4)]
    result["before_wrong_by_gt_class"] = [
        int(counts[c].sum() - counts[c, c].sum()) for c in range(4)]
    result["net_correct_change_by_gt_class"] = [
        int(r - h) for r, h in zip(result["rescued_by_gt_class"], result["harmed_by_gt_class"])]
    result["correct_intact_changed_to_destroyed"] = int(counts[1, 1, 3])
    result["correct_damaged_changed_to_destroyed"] = int(counts[2, 2, 3])
    result["correct_damaged_changed_to_background"] = int(counts[2, 2, 0])
    result["interpretation"] = "Paired same-pixel transitions; axes are GT, before, after."
    return result


def _integer_table(value, shape, description):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu" or (array < 0).any():
        raise ValueError("Invalid saved " + description)
    return array.astype(np.int64)


def _journal_audits(path, names):
    """Stream saved audits, checking their exact image identities and order."""
    index = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "mechanism_audit" not in row:
                continue
            if index >= len(names) or row["image"] != names[index]:
                raise ValueError("Reopened mechanism audit identity/order changed")
            audit = row["mechanism_audit"]
            if not isinstance(audit, dict) or audit.get("image") != row["image"]:
                raise ValueError("Reopened mechanism audit/image identity differs")
            index += 1
            yield audit
    if index != len(names):
        raise ValueError("Reopened mechanism audit image count changed")


def _check_journal(path, identities, paths, comparisons, expected_cm, expected_pairs,
                   expected_events, expected_event_pairs, expected_totals,
                   expected_event_totals):
    """Reopen independent sufficient statistics, including event assignments."""
    cm = {p: np.zeros((4, 4), dtype=np.int64) for p in paths}
    pairs = {p: np.zeros((4, 4, 4), dtype=np.int64) for p in comparisons}
    totals = {p: {} for p in paths}
    events, event_pairs, event_totals, seen = {}, {}, {}, []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= len(identities) or row["index"] != index:
                raise ValueError("Reopened journal index/count changed")
            identity = (row["image"], row["event"], row["valid_pixels"])
            if identity != identities[index]:
                raise ValueError("Reopened journal identity/order/event/count changed")
            seen.append(row["image"])
            if (set(row["confusion"]) != set(paths) or set(row["objectives"]) != set(paths)
                    or set(row["paired_counts"]) != set(comparisons)):
                raise ValueError("Reopened journal arm/comparison identities changed")
            event = events.setdefault(row["event"], {p: np.zeros((4, 4), dtype=np.int64) for p in paths})
            event_pair = event_pairs.setdefault(row["event"], {
                p: np.zeros((4, 4, 4), dtype=np.int64) for p in comparisons})
            event_total = event_totals.setdefault(row["event"], {p: {} for p in paths})
            image_cm = {}
            for name in paths:
                table = _integer_table(row["confusion"][name], (4, 4), "image confusion")
                values = row["objectives"][name]
                if int(table.sum()) != row["valid_pixels"] or values["valid_count"] != row["valid_pixels"]:
                    raise ValueError("Saved valid-pixel totals disagree")
                if not np.array_equal(table.sum(1), values["class_valid_counts"]):
                    raise ValueError("Saved objective class counts disagree")
                if not np.array_equal(table.sum(1) > 0, values["class_present_image_counts"]):
                    raise ValueError("Saved objective present-class counts disagree")
                cm[name] += table
                event[name] += table
                image_cm[name] = table
                _add_terms(totals[name], values)
                _add_terms(event_total[name], values)
            for key, (before, after) in comparisons.items():
                table = _integer_table(row["paired_counts"][key], (4, 4, 4), "paired counts")
                if (not np.array_equal(table.sum(2), image_cm[before])
                        or not np.array_equal(table.sum(1), image_cm[after])):
                    raise ValueError("Saved transitions disagree with marginal confusion")
                pairs[key] += table
                event_pair[key] += table
    if seen != [identity[0] for identity in identities]:
        raise ValueError("Reopened journal image identities/count changed")

    def equal_nested(actual, expected, description):
        if isinstance(expected, dict):
            if set(actual) != set(expected):
                raise ValueError("Reopened journal " + description + " identities changed")
            for key in expected:
                equal_nested(actual[key], expected[key], description)
        elif not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise ValueError("Reopened journal " + description + " changed")

    for actual, expected, description in (
            (cm, expected_cm, "total confusion"), (pairs, expected_pairs, "paired totals"),
            (events, expected_events, "event confusion"),
            (event_pairs, expected_event_pairs, "event paired totals"),
            (totals, expected_totals, "objective totals"),
            (event_totals, expected_event_totals, "event objective totals")):
        equal_nested(actual, expected, description)


def _capture_modes(runtime, heads):
    """Preserve every child mode, including mixed train/eval module trees."""
    roots = list(heads.values())
    if isinstance(runtime, torch.nn.Module):
        roots.append(runtime)
    for value in vars(runtime).values():
        if isinstance(value, torch.nn.Module):
            roots.append(value)
        elif isinstance(value, dict):
            roots.extend(item for item in value.values() if isinstance(item, torch.nn.Module))
    modes = {}
    for root in roots:
        for module in root.modules():
            modes.setdefault(id(module), (module, module.training))
    return list(modes.values())


def _validate_probability(probability, target, path):
    if (not isinstance(probability, torch.Tensor) or not probability.is_floating_point()
            or probability.shape != (1, 4, *target.shape[1:])
            or probability.device != target.device or not bool(torch.isfinite(probability).all())
            or bool(((probability < 0) | (probability > 1)).any())):
        raise ValueError("Invalid full-image probabilities: " + path)
    tolerance = max(1e-5, 4 * torch.finfo(probability.dtype).eps)
    total = probability.to(torch.float64).sum(1)
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0):
        raise ValueError("Unnormalized full-image probabilities: " + path)


def evaluate(runtime, heads, names, folder, *, step, split, seed, original_train,
             legacy, factory, primary_endpoint=False, smoke=False):
    """Evaluate supplied heads and grid_probq with identical frozen inputs.

    ``factory.prepare(example, arms)`` returns the complete prepared support
    mapping; ``factory.forward(head, example, arm, mapping[arm])`` returns the
    inherited model output. The runtime's optional ``evaluation_batches``
    permits full-image fixed-Train diagnostics with evaluation transforms.
    """
    if split not in ("val", "fixed_train"):
        raise ValueError("Only val/fixed_train supported; Test is never accessed")
    arms = tuple(heads)
    if not arms or not set(arms) <= set(ARMS):
        raise ValueError("Expected a nonempty subset of the declared ablation arms")
    if any(not isinstance(head, torch.nn.Module) for head in heads.values()):
        raise TypeError("Every evaluated head must be a torch module")
    names = list(names)
    train_names, val_names = list(runtime.train_names), list(runtime.val_names)
    if (len(train_names) != len(set(train_names)) or len(val_names) != len(set(val_names))
            or set(train_names) & set(val_names)):
        raise ValueError("Runtime Train/Val identities repeat or overlap")
    if not names or len(names) != len(set(names)):
        raise ValueError("Evaluation identities must be nonempty and unique")
    allowed = val_names if split == "val" else train_names
    if not set(names) <= set(allowed):
        raise ValueError("Evaluation identities cross the declared split")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    paths = (BASELINE,) + arms
    comparisons = comparison_pairs(arms)
    rng = legacy.state.capture_rng()
    rng_hash = legacy.state.state_fingerprint(rng)
    modes = _capture_modes(runtime, heads)
    head_hashes = {arm: legacy.state.state_fingerprint(head.state_dict()) for arm, head in heads.items()}
    cm = {p: np.zeros((4, 4), dtype=np.int64) for p in paths}
    pairs = {p: np.zeros((4, 4, 4), dtype=np.int64) for p in comparisons}
    totals = {p: {} for p in paths}
    events, event_pairs, event_totals, event_counts = {}, {}, {}, {}
    identities, audit_names, audit_summary = [], [], None
    valid_pixels, start = 0, time.monotonic()
    try:
        for module, _ in modes:
            module.training = False
        if callable(getattr(runtime, "evaluation_batches", None)):
            batches = runtime.evaluation_batches(names, seed=seed)
        else:
            batches = runtime.batches(names, train=False, seed=seed)
        with torch.no_grad(), (folder / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(batches):
                example = runtime.prepare(batch)
                if index >= len(names) or example.image != names[index]:
                    raise ValueError("Evaluation loader did not preserve exact requested identities/order")
                original_train._assert_example(example)
                coverage = legacy.study.validate_example(example, runtime, original_train)
                support = factory.prepare(example, arms)
                if set(support) != set(arms):
                    raise ValueError("Support factory returned different arm identities")
                target = example.target.detach().cpu().numpy()
                nvalid = int(np.count_nonzero(target != 255))
                valid_pixels += nvalid
                event = events.setdefault(example.event, {p: np.zeros((4, 4), dtype=np.int64) for p in paths})
                event_pair = event_pairs.setdefault(example.event, {
                    p: np.zeros((4, 4, 4), dtype=np.int64) for p in comparisons})
                event_sums = event_totals.setdefault(example.event, {p: {} for p in paths})
                event_counts[example.event] = event_counts.get(example.event, 0) + 1
                row = dict(index=index, image=example.image, event=example.event,
                           valid_pixels=nvalid, confusion={}, objectives={}, paired_counts={},
                           inherited_coverage=coverage)
                predictions = {}
                for path in paths:
                    probability = (example.common["base_probability_full"] if path == BASELINE else
                                   factory.forward(heads[path], example, path, support[path])["probabilities"])
                    _validate_probability(probability, example.target, path)
                    exact_loss = runtime.loss(probability, example.target)
                    pieces = legacy.objectives.serialized_terms(legacy.objectives.objective_terms(
                        probability, example.target, reference_loss=exact_loss,
                        verify_reference=index == 0))
                    values = {key: pieces[key] for key in
                              ("nll", "lovasz", *VECTOR_TERMS, "class_valid_counts", "valid_count")}
                    values["original_loss"] = float(exact_loss.detach())
                    values["class_present_image_counts"] = [int(n > 0) for n in values["class_valid_counts"]]
                    if values["valid_count"] != nvalid:
                        raise ValueError("Loss and confusion valid-pixel counts differ")
                    _add_terms(totals[path], values)
                    _add_terms(event_sums[path], values)
                    prediction = probability.argmax(1).detach().cpu().numpy()
                    image_cm = original_train._confusion(target, prediction)
                    if int(image_cm.sum()) != nvalid:
                        raise ValueError("Confusion valid-pixel total differs from target")
                    cm[path] += image_cm
                    event[path] += image_cm
                    row["confusion"][path] = image_cm.tolist()
                    row["objectives"][path] = values
                    predictions[path] = prediction
                for key, (before, after) in comparisons.items():
                    counts = original_train._transition(target, predictions[before], predictions[after])
                    pairs[key] += counts
                    event_pair[key] += counts
                    row["paired_counts"][key] = counts.tolist()
                if split == "val" and getattr(runtime, "last_extract", None) is not None:
                    from diagnostics import audit_example
                    audit = audit_example(example, runtime, support)
                    row["mechanism_audit"] = audit
                    audit_names.append(example.image)
                factory_diagnostics = getattr(factory, "last_diagnostics", {}).get(example.image)
                if factory_diagnostics is not None:
                    row["support_diagnostics"] = factory_diagnostics
                row["support_selection_sha256"] = {
                    arm: legacy.state.state_fingerprint(fields["support_ids"])
                    for arm, fields in support.items() if "support_ids" in fields}
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                identities.append((example.image, example.event, nvalid))
                if len(identities) % 10 == 0 or len(identities) == len(names):
                    print(f"ABLATION VAL step={step} split={split} {len(identities)}/{len(names)}", flush=True)
        if [identity[0] for identity in identities] != names:
            raise ValueError("Evaluation did not finish exact requested identities")
        _check_journal(folder / "images.jsonl", identities, paths, comparisons, cm, pairs,
                       events, event_pairs, totals, event_totals)
        if audit_names:
            from diagnostics import summarize_audits
            audit_summary = summarize_audits(_journal_audits(folder / "images.jsonl", audit_names))
            if audit_summary.get("image_count") != len(audit_names):
                raise ValueError("Mechanism summary does not cover exact audited images")
    finally:
        for module, training in modes:
            module.training = training
        legacy.state.restore_rng(rng)
        if rng_hash != legacy.state.state_fingerprint(legacy.state.capture_rng()):
            raise RuntimeError("Evaluation did not restore RNG state")
        if any(head_hashes[arm] != legacy.state.state_fingerprint(head.state_dict())
               for arm, head in heads.items()):
            raise RuntimeError("Evaluation changed trainable head parameters/buffers")

    metrics = {p: _metrics(cm[p], totals[p], len(names), original_train) for p in paths}
    per_event = {event: {p: _metrics(tables[p], event_totals[event][p],
                                   event_counts[event], original_train) for p in paths}
                 for event, tables in events.items()}
    protocol = getattr(runtime, "protocol_info", {})
    full_val = (split == "val" and names == val_names and len(names) == 450
                and str(runtime.device).startswith("cuda") and protocol.get("synthetic") is not True
                and not smoke)
    result = dict(
        format="BRIGHT_SLOT_MEMORY_ABLATION_EVALUATION_V1", step=int(step), split=split,
        phase="evaluation_complete_integrity_pending", complete=False, execution_complete=True,
        full_validation_evidence=False, whole_val450_processed=full_val,
        final_integrity_pending=True, primary_endpoint=False,
        primary_endpoint_requested=bool(primary_endpoint),
        smoke=bool(smoke or protocol.get("synthetic") or not str(runtime.device).startswith("cuda")
                   or (split == "val" and not full_val)),
        training=False, test_accessed=False, arms=list(arms), image_count=len(names),
        valid_pixel_count=valid_pixels, class_names=["background", "intact", "damaged", "destroyed"],
        names_sha256=names_hash(sorted(names)), order_sha256=names_hash(names),
        requested_order_sha256=names_hash(names), evaluation_loader_train=False,
        fixed_train_scope=("Declared fixed subset only; not full Train2918 evidence"
                           if split == "fixed_train" else None),
        trained_objective={p: "frozen" if p == BASELINE else "original_nll_plus_lovasz" for p in paths},
        objective_definition="Exact original runtime.loss: NLL + Lovasz(classes=present); no KL.",
        reference_formula_verified_on_first_image=True,
        metrics=metrics, per_event=per_event, per_event_image_count=event_counts,
        transitions={key: _paired_report(counts, *comparisons[key], original_train)
                     for key, counts in pairs.items()},
        per_event_transitions={
            event: {key: _paired_report(counts, *comparisons[key], original_train)
                    for key, counts in tables.items()} for event, tables in event_pairs.items()},
        comparisons_pp=metric_contrasts(metrics, comparisons),
        per_event_comparisons_pp={event: metric_contrasts(tables, comparisons)
                                  for event, tables in per_event.items()},
        interactions_pp=interaction_contrasts(metrics),
        per_event_interactions_pp={event: interaction_contrasts(tables)
                                  for event, tables in per_event.items()},
        contrast_aggregation="Differences of metrics from summed confusion, not averages of image/event IoUs.",
        query_GT_not_support_input=True, inherited_support_exclusion_passed=True,
        prediction_arrays_saved=False, paired_counts_reopened=True,
        objective_totals_reopened=True, per_event_totals_reopened=True,
        head_state_unchanged=True, module_modes_restored=True, rng_restored=True,
        head_hashes=head_hashes, mechanism_audit_image_count=len(audit_names),
        mechanism_audit_evidence="images.jsonl:mechanism_audit" if audit_names else None,
        mechanism_audits_reopened=bool(audit_names),
        journal_sha256=digest(folder / "images.jsonl"), elapsed_seconds=time.monotonic() - start)
    if audit_summary is not None:
        result["mechanism_audit"] = audit_summary
    _write_json(folder / "result.json", result)
    (folder / "result.json.sha256").write_text(digest(folder / "result.json") + "  result.json\n",
                                               encoding="ascii")
    return result


def print_table(result):
    print(f"ABLATION VAL step={result['step']} split={result['split']} images={result['image_count']} "
          f"final_integrity_pending={result['final_integrity_pending']}")
    print("arm | original loss | NLL | Lovasz | fg mIoU % | Damaged IoU % | precision % | recall %")
    def number(value, scale=1.0):
        return "null" if value is None else f"{value * scale:.6f}"
    for path, metrics in result["metrics"].items():
        values = [number(metrics["mean_image_" + name]) for name in SCALAR_TERMS]
        values += [number(value, 100) for value in (
            metrics["foreground_miou"], metrics["iou"][2],
            metrics["precision"][2], metrics["recall"][2])]
        print(path + " | " + " | ".join(values))
    print("Paired foreground mIoU differences (percentage points):", result["comparisons_pp"])
    print("Interactions (percentage points):", result["interactions_pp"])
