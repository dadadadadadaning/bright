"""Validate newly evaluated resume evidence before its final promotion.

The caller must finish Runtime exit, PoolGuard restoration, frozen-state and
input-ledger verification before calling ``promote_reports``. This module
checks the complete scheduled report set and its persisted evidence first;
it does not promote historical results or certify the caller's runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_EVALUATION_V1"
BASELINE = "grid_probq"
CLASSES = ["background", "intact", "damaged", "destroyed"]
VALID_ARMS = set("efbdacrln")
CONTRASTS = (("b", "e"), ("d", "f"), ("f", "e"), ("d", "b"),
             ("c", "a"), ("b", "a"), ("d", "c"), ("d", "r"),
             ("d", "l"), ("d", "n"))


def names_hash(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _identities(values, label):
    values = list(values)
    if (not values or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(label + " must contain nonempty unique identities")
    return values


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer " + label)
    return value


def _number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid finite number " + label)
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError("Out-of-range " + label)
    return value


def _near(actual, expected, label):
    _number(actual, label)
    if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-7):
        raise ValueError("Inconsistent " + label)


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA256 " + label)


def _table(value, dimensions, label):
    if not dimensions:
        _integer(value, label)
        return
    if not isinstance(value, list) or len(value) != dimensions[0]:
        raise ValueError("Invalid table dimensions " + label)
    for item in value:
        _table(item, dimensions[1:], label)


def _sum_tables(tables):
    return [[sum(table[row][column] for table in tables)
             for column in range(4)] for row in range(4)]


def _pairs(arms):
    pairs = {f"{after}_minus_{before}": (before, after)
             for after, before in CONTRASTS if {before, after} <= set(arms)}
    pairs.update({f"{arm}_minus_{BASELINE}": (BASELINE, arm) for arm in arms})
    return pairs


def _metrics(metrics, paths, count, valid_pixels=None):
    if not isinstance(metrics, dict) or set(metrics) != set(paths):
        raise ValueError("Metric arm/baseline identities differ")
    common_target = None
    for path, metric in metrics.items():
        if not isinstance(metric, dict):
            raise ValueError("Expected metric object")
        cm = metric.get("confusion")
        _table(cm, (4, 4), "confusion/" + path)
        target = [sum(row) for row in cm]
        prediction = [sum(cm[row][column] for row in range(4)) for column in range(4)]
        if valid_pixels is not None and sum(target) != valid_pixels:
            raise ValueError("Metric valid-pixel total differs")
        if common_target is None:
            common_target = target
        elif target != common_target:
            raise ValueError("Arms have different GT class totals")
        if metric.get("target_pixels") != target or metric.get("class_valid_counts") != target:
            raise ValueError("Metric target/objective class counts differ")
        if metric.get("class_names") != CLASSES:
            raise ValueError("Metric class names differ")
        present = metric.get("class_present_image_counts")
        _table(present, (4,), "class present counts")
        if any(n > min(pixels, count) or bool(n) != bool(pixels)
               for n, pixels in zip(present, target)):
            raise ValueError("Metric class-presence denominator differs")
        expected_iou = []
        for key in ("iou", "precision", "recall"):
            values = metric.get(key)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("Metric class vector differs: " + key)
            for c, value in enumerate(values):
                denominator = (target[c] + prediction[c] - cm[c][c] if key == "iou"
                               else prediction[c] if key == "precision" else target[c])
                expected = cm[c][c] / denominator if denominator else None
                if expected is None:
                    if value is not None:
                        raise ValueError("Absent-class metric must be null: " + key)
                else:
                    _number(value, key, 0, 1)
                    _near(value, expected, key)
                if key == "iou":
                    expected_iou.append(expected)
        foreground = sum(value or 0.0 for value in expected_iou[1:]) / 3.0
        _near(metric.get("foreground_miou"), foreground, "foreground mIoU")
        _near(metric.get("valid_miou"), foreground, "valid mIoU")
        for key in ("original_loss", "nll", "lovasz"):
            _number(metric.get("mean_image_" + key), key, minimum=0)
        _near(metric["mean_image_original_loss"],
              metric["mean_image_nll"] + metric["mean_image_lovasz"], "original loss")
        for term in ("nll", "lovasz"):
            contributions = metric.get("mean_image_class_" + term + "_contribution")
            if not isinstance(contributions, list) or len(contributions) != 4:
                raise ValueError("Loss contribution dimensions differ")
            for value in contributions:
                _number(value, "class " + term, minimum=0)
            _near(sum(contributions), metric["mean_image_" + term], "class " + term + " sum")
    return sum(common_target)


def _transitions(reports, pairs, metrics):
    if not isinstance(reports, dict) or set(reports) != set(pairs):
        raise ValueError("Paired transition identities differ")
    for key, (before, after) in pairs.items():
        report = reports[key]
        if not isinstance(report, dict) or report.get("axes") != ["GT", before, after]:
            raise ValueError("Paired transition axes differ")
        counts = report.get("counts")
        _table(counts, (4, 4, 4), "paired counts")
        left = [[sum(counts[gt][b]) for b in range(4)] for gt in range(4)]
        right = [[sum(counts[gt][b][a] for b in range(4)) for a in range(4)] for gt in range(4)]
        if left != metrics[before]["confusion"] or right != metrics[after]["confusion"]:
            raise ValueError("Paired transition marginals differ")


def _contrasts(report, metrics, pairs):
    values = report.get("comparisons_pp")
    if not isinstance(values, dict) or set(values) != set(pairs):
        raise ValueError("Comparison identities differ")
    for key, (before, after) in pairs.items():
        _near(values[key], 100 * (metrics[after]["foreground_miou"] -
                                  metrics[before]["foreground_miou"]), key)
    expected = {}
    for label, arms in (("(d-b)-(f-e)", ("d", "b", "f", "e")),
                        ("(d-c)-(b-a)", ("d", "c", "b", "a"))):
        if set(arms) <= set(metrics):
            a, b, c, d = (metrics[arm]["foreground_miou"] for arm in arms)
            expected[label] = 100 * ((a - b) - (c - d))
    actual = report.get("interactions_pp")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("Interaction identities differ")
    for key, value in expected.items():
        _near(actual[key], value, key)


def validate_report(report, *, step, split, names, arms, primary=False):
    """Require an unpromoted, complete evaluation of exactly these identities."""
    names = _identities(names, "Evaluation names")
    arms = _identities(arms, "Arms")
    if not set(arms) <= VALID_ARMS:
        raise ValueError("Unknown evaluation arm")
    _integer(step, "step")
    if split not in ("val", "fixed_train") or type(primary) is not bool:
        raise ValueError("Invalid evaluation split/primary designation")
    if primary and split != "val":
        raise ValueError("Only Val can be the primary endpoint")
    if split == "val" and len(names) != 450:
        raise ValueError("Formal validation requires exact Val450 identities")
    if not isinstance(report, dict):
        raise ValueError("Expected evaluation report object")
    # This also rejects NaN/Inf hidden in nested optional diagnostics.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    if (report.get("format") != FORMAT or type(report.get("step")) is not int
            or report["step"] != step or report.get("split") != split
            or report.get("arms") != arms or report.get("class_names") != CLASSES):
        raise ValueError("Evaluation format/step/split/arm schema differs")
    required_true = (
        "execution_complete", "final_integrity_pending",
        "reference_formula_verified_on_first_image", "query_GT_not_support_input",
        "inherited_support_exclusion_passed", "paired_counts_reopened",
        "objective_totals_reopened", "per_event_totals_reopened",
        "head_state_unchanged", "module_modes_restored", "rng_restored")
    required_false = (
        "complete", "full_validation_evidence", "primary_endpoint", "smoke",
        "training", "test_accessed", "evaluation_loader_train", "prediction_arrays_saved")
    if any(report.get(key) is not True for key in required_true):
        raise ValueError("Evaluation completeness/integrity check is missing")
    if any(report.get(key) is not False for key in required_false):
        raise ValueError("Expected a fresh formal report awaiting integrity promotion")
    if (report.get("phase") != "evaluation_complete_integrity_pending"
            or report.get("primary_endpoint_requested") is not primary
            or report.get("whole_val450_processed") is not (split == "val")):
        raise ValueError("Evaluation pending phase/full-Val/primary status differs")
    if type(report.get("image_count")) is not int or report["image_count"] != len(names):
        raise ValueError("Evaluation image count differs")
    expected_hashes = {
        "names_sha256": names_hash(sorted(names)),
        "order_sha256": names_hash(names), "requested_order_sha256": names_hash(names)}
    if any(report.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("Evaluation identity/order hash differs")
    valid_pixels = _integer(report.get("valid_pixel_count"), "valid pixel count")
    paths = [BASELINE] + arms
    if report.get("trained_objective") != {
            path: "frozen" if path == BASELINE else "original_nll_plus_lovasz" for path in paths}:
        raise ValueError("Evaluation objective differs from the original protocol")
    _metrics(report.get("metrics"), paths, len(names), valid_pixels)
    pairs = _pairs(arms)
    _transitions(report.get("transitions"), pairs, report["metrics"])
    _contrasts(report, report["metrics"], pairs)
    event_counts = report.get("per_event_image_count")
    events = report.get("per_event")
    if not isinstance(event_counts, dict) or not event_counts or not isinstance(events, dict):
        raise ValueError("Per-event evidence is missing")
    if set(event_counts) != set(events):
        raise ValueError("Per-event identities differ")
    if sum(_integer(count, "event image count", minimum=1)
           for count in event_counts.values()) != len(names):
        raise ValueError("Per-event image counts differ")
    for field in ("per_event_transitions", "per_event_comparisons_pp", "per_event_interactions_pp"):
        if not isinstance(report.get(field), dict) or set(report[field]) != set(events):
            raise ValueError("Per-event report identities differ: " + field)
    event_pixels = 0
    for event, metrics in events.items():
        event_pixels += _metrics(metrics, paths, event_counts[event])
        _transitions(report["per_event_transitions"][event], pairs, metrics)
        _contrasts({
            "comparisons_pp": report["per_event_comparisons_pp"][event],
            "interactions_pp": report["per_event_interactions_pp"][event]}, metrics, pairs)
    if event_pixels != valid_pixels:
        raise ValueError("Per-event valid-pixel totals differ")
    for path in paths:
        tables = [metrics[path]["confusion"] for metrics in events.values()]
        if _sum_tables(tables) != report["metrics"][path]["confusion"]:
            raise ValueError("Per-event confusion totals differ")
        for key in ("original_loss", "nll", "lovasz"):
            mean = sum(events[event][path]["mean_image_" + key] * count
                       for event, count in event_counts.items()) / len(names)
            _near(report["metrics"][path]["mean_image_" + key], mean, "event mean " + key)
        for c in range(4):
            if sum(metric[path]["class_present_image_counts"][c] for metric in events.values()) != (
                    report["metrics"][path]["class_present_image_counts"][c]):
                raise ValueError("Per-event class-presence totals differ")
    for key in pairs:
        counts = report["transitions"][key]["counts"]
        event_tables = [event[key]["counts"] for event in report["per_event_transitions"].values()]
        if any(counts[g][b][a] != sum(table[g][b][a] for table in event_tables)
               for g in range(4) for b in range(4) for a in range(4)):
            raise ValueError("Per-event paired totals differ")
    hashes = report.get("head_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(arms):
        raise ValueError("Head hash identities differ")
    for arm, digest in hashes.items():
        _sha(digest, arm)
    _sha(report.get("journal_sha256"), "evaluation journal")
    if "mechanism_audit_image_count" in report:
        count = _integer(report["mechanism_audit_image_count"], "mechanism audit count")
        if count > len(names) or (split == "fixed_train" and count):
            raise ValueError("Mechanism audit image count differs")
        if count and (report.get("mechanism_audits_reopened") is not True
                      or report.get("mechanism_audit", {}).get("image_count") != count
                      or report.get("mechanism_audit_evidence") != "images.jsonl:mechanism_audit"):
            raise ValueError("Mechanism audit evidence is incomplete")


def expected_evaluations(boundary, endpoint, val_names, fixed_names):
    """Original 500-step schedule plus newly measured boundary diagnostics."""
    _integer(boundary, "boundary", minimum=1)
    _integer(endpoint, "endpoint", minimum=1)
    if endpoint < boundary:
        raise ValueError("Resume endpoint must not precede its checkpoint boundary")
    if endpoint == boundary:
        return [(boundary, "val", list(val_names), True),
                (boundary, "fixed_train", list(fixed_names), False)]
    steps = sorted({*range((boundary // 500 + 1) * 500, endpoint + 1, 500), endpoint})
    return [(boundary, "val", list(val_names), False),
            (boundary, "fixed_train", list(fixed_names), False)] + [
        (step, "val", list(val_names), step == endpoint) for step in steps] + [
        (endpoint, "fixed_train", list(fixed_names), False)]


def _read_json(path):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)


def _safe_file(path, output_root):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or symlinked fresh evaluation artifact: " + str(path))
    if output_root is not None:
        root = Path(output_root).absolute()
        if root.is_symlink() or root.resolve() != root:
            raise ValueError("Resume output root must not traverse symlinks")
        if not path.is_relative_to(root) or path.resolve() != path:
            raise ValueError("Evaluation artifact is outside the fresh resume output root")
    elif path.resolve() != path:
        raise ValueError("Evaluation artifact must not traverse symlinks")
    return path


def _verify_files(path, report, names, sha256, output_root):
    path = _safe_file(path, output_root)
    if path.name != "result.json":
        raise ValueError("Expected a freshly evaluated result.json")
    checksum = _safe_file(path.with_name("result.json.sha256"), output_root)
    journal = _safe_file(path.with_name("images.jsonl"), output_root)
    if _read_json(path) != report:
        raise ValueError("Persisted evaluation report differs from the pending result")
    if checksum.read_text(encoding="ascii").split() != [sha256(path), "result.json"]:
        raise ValueError("Pending evaluation report checksum differs")
    if sha256(journal) != report["journal_sha256"]:
        raise ValueError("Pending evaluation journal checksum differs")
    seen, pixels, events = [], 0, {}
    with journal.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if not line.endswith("\n") or row.get("index") != index:
                raise ValueError("Evaluation journal is incomplete or reordered")
            if index >= len(names) or row.get("image") != names[index]:
                raise ValueError("Evaluation journal image identity/order differs")
            event = row.get("event")
            if not isinstance(event, str) or not event:
                raise ValueError("Evaluation journal event identity is missing")
            seen.append(row["image"])
            pixels += _integer(row.get("valid_pixels"), "journal valid pixels")
            events[event] = events.get(event, 0) + 1
    if (seen != names or pixels != report["valid_pixel_count"]
            or events != report["per_event_image_count"]):
        raise ValueError("Evaluation journal identities/counts differ from the report")
    return path


def promote_reports(results, *, arms, val_names, fixed_names, endpoint,
                    boundary_step, write_json, sha256, expected=None, output_root=None):
    """Promote only the complete set of new reports after caller integrity gates.

    Production callers must supply their newly created ``output_root``.
    ``expected`` optionally supplies explicit ``(step, split, names, primary)``
    records; otherwise the original 500-step evaluation schedule is required.
    Every report/file is checked before the first report is rewritten.
    """
    val_names = _identities(val_names, "Val names")
    fixed_names = _identities(fixed_names, "Fixed Train names")
    arms = _identities(arms, "Arms")
    if len(val_names) != 450 or set(val_names) & set(fixed_names):
        raise ValueError("Require disjoint full Val450 and fixed Train identities")
    default = expected_evaluations(boundary_step, endpoint, val_names, fixed_names)
    expected = list(default if expected is None else expected)
    specifications = {}
    for item in expected:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            raise ValueError("Invalid expected evaluation specification")
        step, split, names, primary = item
        _integer(step, "expected step", minimum=boundary_step)
        if step > endpoint or split not in ("val", "fixed_train") or type(primary) is not bool:
            raise ValueError("Expected evaluation is outside the resume interval")
        names = list(names)
        if names != (val_names if split == "val" else fixed_names):
            raise ValueError("Expected evaluation names differ from the declared split")
        if primary != (step == endpoint and split == "val"):
            raise ValueError("Only the endpoint Val report can be primary")
        key = (step, split)
        if key in specifications:
            raise ValueError("Duplicate expected evaluation")
        specifications[key] = (names, primary)
    required = {(boundary_step, "val"), (boundary_step, "fixed_train"),
                (endpoint, "val"), (endpoint, "fixed_train")}
    if not required <= set(specifications):
        raise ValueError("Missing boundary or endpoint evaluation specification")
    results = list(results)
    staged, keys, paths = [], set(), set()
    for path, report in results:
        key = (report.get("step"), report.get("split")) if isinstance(report, dict) else None
        if key not in specifications or key in keys:
            raise ValueError("Unexpected, historical or repeated evaluation report")
        names, primary = specifications[key]
        validate_report(report, step=key[0], split=key[1], names=names, arms=arms, primary=primary)
        path = _verify_files(path, report, names, sha256, output_root)
        if path in paths:
            raise ValueError("Repeated evaluation output path")
        paths.add(path)
        keys.add(key)
        promoted = copy.deepcopy(report)
        promoted.update(
            complete=True, final_integrity_pending=False, phase="complete",
            full_validation_evidence=key[1] == "val", primary_endpoint=primary,
            resume_boundary_step=boundary_step)
        staged.append((path, report, promoted))
    if keys != set(specifications):
        raise ValueError("The required resume evaluation schedule is incomplete")
    for path, original, promoted in staged:
        write_json(path, promoted)
        path.with_name("result.json.sha256").write_text(
            sha256(path) + "  result.json\n", encoding="ascii")
        original.clear()
        original.update(promoted)
