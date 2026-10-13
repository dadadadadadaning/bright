#!/usr/bin/env python3
"""CPU-only, saved-prediction comparison of matched baseline-Q/SAR-Q routes.

All inputs are read-only. The output is a new comparison directory, never an
edited training or evaluation result. No torch, model loading or CUDA is used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback

import numpy as np


FORMAT = "SRAC_SAR_SOFT_ROUTE_V1"
REPORT_FORMAT = "SRAC_SAR_SOFT_CROSS_COMPARISON_V1"
ARMS = ("unsplit", "capacity_control", "hard", "soft")
HISTORICAL = ("a1_pixel", "direct_region", "j1", "j2", "j3")
ALL_ARMS = HISTORICAL + ("q_a1_pixel",) + ARMS
CLASS_NAMES = ("Background", "Intact", "Damaged", "Destroyed")
MAIN_CELLS = ("baseline_unsplit", "baseline_soft", "sar_unsplit", "sar_soft")
Q_HYPERPARAMETERS = ("steps", "eval_every", "batch", "accum", "workers", "lr",
                     "aux_weight", "cue_gain", "amp")
ROUTE_HYPERPARAMETERS = ("steps", "eval_every", "lr", "threshold", "soft_floor")
METRIC_ATOL = 1e-10
_UNSET = object()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Missing/invalid SHA256: " + label)
    return value


def _true_map(value, label):
    if not isinstance(value, dict) or not value or not all(v is True for v in value.values()):
        raise ValueError("Failed/missing immutable checks: " + label)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=_UNSET):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not _UNSET and value != _sha(expected, str(path)):
            raise ValueError("Input SHA256 mismatch: " + str(path))
        previous = self.hashes.setdefault(str(path), value)
        if previous != value:
            raise ValueError("Input changed during comparison: " + str(path))
        return value

    def json(self, path):
        self.bind(path)
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object: " + str(path))
        return value

    def journal(self, path):
        self.bind(path)
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Invalid journal rows: " + str(path))
        return rows

    def verify(self):
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError("Input changed during comparison: " + path)


def metrics_from_confusion(confusion):
    cm = np.asarray(confusion)
    if (cm.shape != (4, 4) or not np.issubdtype(cm.dtype, np.integer)
            or np.any(cm < 0)):
        raise ValueError("Expected nonnegative integer 4x4 confusion")
    cm = cm.astype(np.int64)
    h = cm.astype(np.float64)
    tp, rows, columns = np.diag(h), h.sum(1), h.sum(0)
    union = rows + columns - tp
    def ratios(numerator, denominator):
        return [float(n / d) if d > 0 else None for n, d in zip(numerator, denominator)]
    iou = ratios(tp, union)
    present = [v for v in iou[1:] if v is not None]
    return dict(count=int(cm.sum()), confusion=cm.tolist(),
                valid_miou=float((tp / np.maximum(union, 1))[1:].mean()) if cm.sum() else None,
                present_foreground_miou=sum(present)/len(present) if present else None,
                iou=iou, precision=ratios(tp, columns), recall=ratios(tp, rows),
                f1=ratios(2*tp, rows+columns), support=rows.astype(np.int64).tolist(),
                background_to_building=int(cm[0, 1:].sum()),
                building_to_background=int(cm[1:, 0].sum()))


def _close(actual, expected, label):
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Metric shape mismatch: " + label)
        for i, (a, e) in enumerate(zip(actual, expected)):
            _close(a, e, f"{label}[{i}]")
    elif expected is None:
        if actual is not None:
            raise ValueError("Missing-class metric differs: " + label)
    elif isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise ValueError("Metric count differs: " + label)
    elif (type(actual) not in (int, float) or not math.isfinite(actual)
          or not math.isclose(actual, expected, rel_tol=0, abs_tol=METRIC_ATOL)):
        raise ValueError("Metric value differs: " + label)


def check_metrics(stored, confusion, label):
    if not isinstance(stored, dict):
        raise ValueError("Missing metrics: " + label)
    actual = metrics_from_confusion(confusion)
    for key, value in actual.items():
        if key not in stored:
            raise ValueError("Incomplete metrics: " + label + "/" + key)
        _close(stored[key], value, label + "/" + key)
    return actual


def _gate_report(root, evidence, *, initial):
    root = Path(root)
    summary = evidence.json(root / "summary.json")
    required = dict(format=FORMAT + "_EVAL", complete=True, full_validation_evidence=True,
                    image_count=450, smoke=False, test_accessed=False, training=False,
                    initial=initial, result_file="result.json", reproduction_passed=True,
                    e0_protocol_comparison_passed=True, q_a1_reproduction_passed=True,
                    native_a1_replay_ok=True)
    for key, expected in required.items():
        if summary.get(key) != expected or isinstance(expected, bool) and summary.get(key) is not expected:
            raise ValueError("Evaluation summary gate failed: " + str(root) + "/" + key)
    _true_map(summary.get("immutable"), str(root))
    evidence.bind(root / "result.json", summary.get("result_sha256"))
    result = evidence.json(root / "result.json")
    for key in ("format", "complete", "full_validation_evidence", "image_count", "smoke",
                "test_accessed", "training", "initial", "native_a1_replay_ok"):
        if result.get(key) != required[key] or isinstance(required[key], bool) and result.get(key) is not required[key]:
            raise ValueError("Evaluation result gate failed: " + str(root) + "/" + key)
    _true_map(result.get("immutable"), str(root))
    for key in ("reproduction", "e0_metric_comparison", "q_a1_reproduction"):
        if result.get(key, {}).get("passed") is not True:
            raise ValueError("Evaluation reproduction failed: " + str(root) + "/" + key)
    guard = result.get("pool_guard", {})
    if (guard.get("backend") != "native" or guard.get("restored") is not True
            or guard.get("failures") != [] or guard.get("restore_conflicts") != []):
        raise ValueError("Evaluation native pool guard failed: " + str(root))
    if initial and (type(result.get("initial_unsplit_max_abs")) not in (float, int)
                    or not 0 <= result["initial_unsplit_max_abs"] <= 2e-5):
        raise ValueError("Initial unsplit probability replay failed")
    if set(result.get("metrics", {})) != set(ALL_ARMS):
        raise ValueError("Evaluation must contain every historical, Q-A1 and route arm")
    if not isinstance(result.get("per_event"), dict) or not result["per_event"]:
        raise ValueError("Missing per-event results")
    for scope, rows in [("all", result["metrics"]), *result["per_event"].items()]:
        if set(rows) != set(ALL_ARMS):
            raise ValueError("Incomplete metric arms: " + scope)
        for arm in ALL_ARMS:
            check_metrics(rows[arm], rows[arm].get("confusion"), str(root) + "/" + scope + "/" + arm)
    return summary, result


def _q_metadata(meta, arm, protocol_sha, evidence):
    if (not isinstance(meta, dict) or meta.get("kind") != "completed_" + arm + "_q"
            or meta.get("arm") != arm or meta.get("steps") != 2000
            or meta.get("protocol_sha256") != protocol_sha):
        raise ValueError("Unexpected Q-source identity: " + arm)
    path = Path(meta.get("checkpoint", ""))
    if path.name != "last.pth":
        raise ValueError("Q source must be final last.pth")
    evidence.bind(path, meta.get("sha256"))
    for key in ("train_source_hash", "inherited_source_hash"):
        _sha(meta.get(key), key)
    hp = meta.get("training_hyperparameters", {})
    if set(hp) != set(Q_HYPERPARAMETERS):
        raise ValueError("Missing Q-training hyperparameters")
    expected = dict(steps=2000, eval_every=2000, batch=2, accum=8, lr=1e-4, aux_weight=0.1)
    if any(hp.get(k) != v for k, v in expected.items()) or hp.get("amp") not in ("bf16", "none"):
        raise ValueError("Unmatched/nonformal Q-training recipe")
    if (type(hp.get("workers")) is not int or hp["workers"] < 0
            or hp.get("cue_gain") != meta.get("cue_gain") or meta.get("cue_gain") not in (0.1, 0.25)
            or type(meta.get("route_seed")) is not int):
        raise ValueError("Invalid Q/route settings")
    fingerprints = meta.get("initial_readout_fingerprints", {})
    if set(fingerprints) != {"embedding", "region_classifier"}:
        raise ValueError("Initial downstream readout hashes missing")
    for key, value in fingerprints.items():
        _sha(value, key)


def _load_fit(fit, evaluation, expected_arm, evidence):
    fit, evaluation = Path(fit).resolve(), Path(evaluation).resolve()
    summary = evidence.json(fit / "summary.json")
    manifest = evidence.json(fit / "manifest.json")
    for row in (summary, manifest):
        if (row.get("format") != FORMAT or row.get("smoke") is not False
                or row.get("test_accessed") is not False):
            raise ValueError("Unexpected fit identity: " + str(fit))
    if (summary.get("complete") is not True or summary.get("full_validation_evidence") is not True
            or summary.get("steps") != 2000 or summary.get("checkpoint") != "heads_last.pth"
            or summary.get("updates_per_arm") != dict.fromkeys(ARMS, 2000)):
        raise ValueError("Fit has not completed all four 2000-step arms")
    _true_map(summary.get("immutable"), str(fit))
    checkpoint_sha = evidence.bind(fit / "heads_last.pth", summary.get("checkpoint_sha256"))
    journal_sha = evidence.bind(fit / "training.jsonl", summary.get("training_journal_sha256"))
    for key in ("protocol_sha256", "source_sha256", "train_stream_sha256"):
        _sha(summary.get(key), key)
    for key in ("protocol_sha256", "source_sha256", "q_source", "trainable_parameters"):
        if manifest.get(key) != summary.get(key):
            raise ValueError("Fit manifest/summary disagree: " + key)
    parameters = summary.get("trainable_parameters", {})
    if (set(parameters) != set(ARMS) or any(type(n) is not int or n <= 0 for n in parameters.values())
            or len(set(parameters.values())) != 1):
        raise ValueError("Four route heads must have equal positive parameter counts")
    args = manifest.get("arguments", {})
    if (args.get("steps") != 2000 or args.get("smoke") is not False
            or type(args.get("eval_every")) is not int or args["eval_every"] <= 0
            or 2000 % args["eval_every"] or type(args.get("lr")) not in (int, float)
            or not math.isfinite(args["lr"]) or args["lr"] <= 0):
        raise ValueError("Invalid formal route-training recipe")
    for key in ("threshold", "soft_floor"):
        if summary.get(key) != args.get(key):
            raise ValueError("Fit threshold/floor metadata mismatch")
    if (type(args.get("threshold")) not in (int, float) or not 0 < args["threshold"] < 1
            or type(args.get("soft_floor")) not in (int, float) or not 0 <= args["soft_floor"] < 0.5):
        raise ValueError("Invalid route threshold/floor")
    meta = summary.get("q_source")
    _q_metadata(meta, expected_arm, summary["protocol_sha256"], evidence)
    if Path(args.get("q_checkpoint", "")).resolve() != Path(meta["checkpoint"]).resolve():
        raise ValueError("Fit CLI Q checkpoint differs from q_source")
    protocol_path = Path(args.get("protocol", ""))
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    evidence.bind(protocol_path, summary["protocol_sha256"])
    protocol = evidence.json(protocol_path)
    names = protocol.get("selected_images")
    if (not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("Source protocol lacks exact unique Val450 identities")
    if meta["route_seed"] != int(protocol["arguments"]["seed"]) + 31:
        raise ValueError("Route initialization seed is not the declared common E0 seed + 31")
    train_rows = evidence.journal(fit / "training.jsonl")
    if len(train_rows) != 2000:
        raise ValueError("Training journal must contain exactly 2000 updates")
    stream = hashlib.sha256()
    for index, row in enumerate(train_rows, 1):
        if row.get("step") != index or not isinstance(row.get("image"), str) or not row["image"]:
            raise ValueError("Training stream has missing/reordered steps")
        if set(row.get("heads", {})) != set(ARMS):
            raise ValueError("Training journal lacks a route arm")
        for arm in ARMS:
            for key in ("loss", "grad_norm"):
                value = row["heads"][arm].get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Invalid training loss/gradient record")
        stream.update((str(index) + "\t" + row["image"] + "\n").encode("utf-8"))
    if stream.hexdigest() != summary["train_stream_sha256"]:
        raise ValueError("Training stream fingerprint mismatch")
    for scope, initial in (("initial_val", True), ("final_val", False)):
        _, audit = _gate_report(fit / scope, evidence, initial=initial)
        for key in ("q_source", "protocol_sha256", "source_sha256"):
            if audit.get(key) != summary.get(key):
                raise ValueError("Fit audit source binding differs: " + scope + "/" + key)
        fragment = summary.get("initial" if initial else "final", {})
        for key in ("complete", "full_validation_evidence", "reproduction", "e0_metric_comparison",
                    "q_a1_reproduction", "metrics"):
            if fragment.get(key) != audit.get(key):
                raise ValueError("Fit saved audit disagrees with summary: " + scope + "/" + key)
    eval_summary, result = _gate_report(evaluation, evidence, initial=False)
    for key in ("q_source", "protocol_sha256", "source_sha256"):
        if result.get(key) != summary.get(key):
            raise ValueError("Independent evaluation source differs: " + key)
    worker = evidence.json(evaluation / "worker_record.json")
    if (worker.get("format") != "SAR_SOFT_WORKER_V1" or worker.get("action") != "evaluate"
            or worker.get("complete") is not True
            or worker.get("q_arm") != expected_arm or worker.get("smoke") is not False
            or worker.get("test_accessed") is not False
            or Path(worker.get("fit_root", "")).resolve() != fit
            or worker.get("protocol_sha256") != summary["protocol_sha256"]
            or worker.get("adapter_source_sha256") != summary["source_sha256"]
            or worker.get("environment", {}).get("CUDA_LAUNCH_BLOCKING") != "1"):
        raise ValueError("Independent evaluation worker binding failed")
    bindings = {"fit_summary_sha256": fit / "summary.json", "fit_manifest_sha256": fit / "manifest.json",
                "fit_checkpoint_sha256": fit / "heads_last.pth", "training_journal_sha256": fit / "training.jsonl",
                "result_sha256": evaluation / "result.json", "summary_sha256": evaluation / "summary.json",
                "source_certificate_sha256": worker.get("source_certificate", "")}
    for key, path in bindings.items():
        evidence.bind(path, worker.get(key))
    certificate = evidence.json(worker["source_certificate"])
    if (certificate.get("format") != "SAR_SOFT_SOURCE_CERTIFICATE_V1"
            or certificate.get("complete") is not True
            or certificate.get("full_validation_evidence") is not True
            or certificate.get("smoke") is not False or certificate.get("test_accessed") is not False
            or certificate.get("protocol_sha256") != summary["protocol_sha256"]):
        raise ValueError("Incomplete/unrelated four-arm source certificate")
    prefix = "baseline" if expected_arm == "baseline" else "sar"
    if (certificate.get(prefix + "_checkpoint_sha256") != meta["sha256"]
            or Path(certificate.get(prefix + "_checkpoint", "")).resolve() != Path(meta["checkpoint"]).resolve()):
        raise ValueError("Q source does not match certified checkpoint")
    return dict(fit=fit, evaluation=evaluation, summary=summary, manifest=manifest,
                result=result, worker=worker, val_names=names,
                checkpoint_sha256=checkpoint_sha, training_journal_sha256=journal_sha)


def _matched(left, right):
    for key in ("protocol_sha256", "source_sha256", "steps", "threshold", "soft_floor",
                "train_stream_sha256", "trainable_parameters"):
        if left["summary"].get(key) != right["summary"].get(key):
            raise ValueError("Cross-Q fits differ: " + key)
    a, b = left["summary"]["q_source"], right["summary"]["q_source"]
    for key in ("protocol_sha256", "train_source_hash", "inherited_source_hash", "cue_gain",
                "training_hyperparameters", "route_seed", "initial_readout_fingerprints"):
        if a.get(key) != b.get(key):
            raise ValueError("Cross-Q source recipe differs: " + key)
    for key in ROUTE_HYPERPARAMETERS:
        if left["manifest"]["arguments"].get(key) != right["manifest"]["arguments"].get(key):
            raise ValueError("Cross-Q route-training hyperparameter differs: " + key)
    for key in ("original_front_sha256", "original_a1_sha256", "train_list_sha256", "val_list_sha256"):
        av, bv = left["manifest"].get(key), right["manifest"].get(key)
        if _sha(av, key) != _sha(bv, key):
            raise ValueError("Cross-Q input provenance differs: " + key)
    if left["val_names"] != right["val_names"]:
        raise ValueError("Cross-Q validation image identities/order differ")
    if left["worker"]["source_certificate_sha256"] != right["worker"]["source_certificate_sha256"]:
        raise ValueError("The two Q sources were not selected from the same four-arm certificate")
    for key in ("CUDA_LAUNCH_BLOCKING", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"):
        if left["worker"]["environment"].get(key) != right["worker"]["environment"].get(key):
            raise ValueError("Independent evaluation CUDA environments differ: " + key)


def _integer_array(value, name, *, shape=None, labels=False):
    if not np.issubdtype(value.dtype, np.integer) or value.ndim not in (2, 3) or value.size == 0:
        raise ValueError("Invalid integer image array: " + name)
    if shape is not None and value.shape != shape:
        raise ValueError("Prediction/target shape mismatch: " + name)
    valid = ((value >= 0) & (value < 4)) | ((value == 255) if labels else False)
    if not bool(valid.all()):
        raise ValueError("Out-of-range classes: " + name)
    return value


def _journal(record, evidence):
    rows = evidence.journal(record["evaluation"] / "images.jsonl")
    if len(rows) != 450:
        raise ValueError("Independent evaluation journal must contain 450 images")
    files = set()
    for index, row in enumerate(rows):
        if (row.get("index") != index or row.get("image") != record["val_names"][index]
                or not isinstance(row.get("event"), str) or not row["event"]
                or set(row.get("confusion", {})) != set(ALL_ARMS)):
            raise ValueError("Evaluation journal identity/arm mismatch")
        if any(row.get("native_a1_replay", {}).get(key, {}).get("passed") is not True
               for key in ("original", "q_source")):
            raise ValueError("Per-image native A1 replay did not pass")
        relative = row.get("prediction_file")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Prediction path must be a relative artifact")
        path = (record["evaluation"] / relative).resolve()
        if not path.is_relative_to((record["evaluation"] / "predictions").resolve()) or path in files:
            raise ValueError("Repeated or escaping prediction artifact")
        files.add(path)
        row["_path"] = path
    return rows


def _load_image(row, evidence):
    path = row["_path"]
    evidence.bind(path, row.get("prediction_sha256"))
    with np.load(path, allow_pickle=False) as archive:
        required = {"target", "building_grid", "support_j2", "support_j3"}
        required |= {"prediction_" + arm for arm in ALL_ARMS}
        if not required <= set(archive.files):
            raise ValueError("Incomplete route prediction archive: " + str(path))
        target = _integer_array(archive["target"], "target", labels=True).copy()
        predictions = {arm: _integer_array(archive["prediction_" + arm], arm, shape=target.shape).copy()
                       for arm in ALL_ARMS}
        building = archive["building_grid"]
        if (building.ndim != 4 or building.shape[:2] != (1, 1) or not np.isfinite(building).all()
                or np.any((building < 0) | (building > 1))):
            raise ValueError("Invalid saved building probability")
        for name in ("support_j2", "support_j3"):
            ids = archive[name]
            if not np.issubdtype(ids.dtype, np.integer) or ids.ndim != 2 or ids.shape[-1] != 8 or np.any(ids < -1):
                raise ValueError("Invalid saved historical support IDs")
    keep = (target >= 0) & (target < 4)
    y = target[keep].astype(np.int64)
    predictions = {arm: value[keep].astype(np.int64) for arm, value in predictions.items()}
    cms = {arm: np.bincount(y*4 + value, minlength=16).reshape(4, 4)
           for arm, value in predictions.items()}
    for arm in ALL_ARMS:
        _close(row["confusion"][arm], cms[arm].tolist(), "per-image confusion/" + arm)
    return target, y, predictions, cms


def transition_summary(joint):
    joint = np.asarray(joint, dtype=np.int64)
    if joint.shape != (4, 4, 4) or np.any(joint < 0):
        raise ValueError("Invalid GT/old/new transition cube")
    classes = {}
    for c, name in enumerate(CLASS_NAMES):
        total = int(joint[c].sum())
        old_tp, new_tp, kept = int(joint[c, c].sum()), int(joint[c, :, c].sum()), int(joint[c, c, c])
        harm, rescue = old_tp-kept, new_tp-kept
        old_fp, new_fp = int(joint[:, c].sum())-old_tp, int(joint[:, :, c].sum())-new_tp
        classes[name] = dict(gt_count=total, reference_tp=old_tp, candidate_tp=new_tp,
                             rescue=rescue, harm=harm, net_tp=rescue-harm,
                             rescue_rate=rescue/(total-old_tp) if total > old_tp else None,
                             harm_rate=harm/old_tp if old_tp else None,
                             reference_fp=old_fp, candidate_fp=new_fp, delta_fp=new_fp-old_fp)
    return dict(joint_gt_reference_candidate=joint.tolist(), per_class=classes)


def _cell(metric):
    pct = lambda value: None if value is None else 100*value
    return dict(foreground_miou_pct=pct(metric["valid_miou"]),
                class_iou_pct=[pct(x) for x in metric["iou"]],
                damaged_precision_pct=pct(metric["precision"][2]),
                damaged_recall_pct=pct(metric["recall"][2]), confusion=metric["confusion"])


def _delta(old, new):
    sub = lambda a, b: None if a is None or b is None else 100*(b-a)
    return dict(foreground_miou_pp=sub(old["valid_miou"], new["valid_miou"]),
                class_iou_pp=[sub(a, b) for a, b in zip(old["iou"], new["iou"])],
                damaged_precision_pp=sub(old["precision"][2], new["precision"][2]),
                damaged_recall_pp=sub(old["recall"][2], new["recall"][2]))


def effects(metrics):
    baseline, sar = metrics["baseline"], metrics["sar"]
    pairs = {"soft_minus_unsplit_at_baseline_q": (baseline["unsplit"], baseline["soft"]),
             "soft_minus_unsplit_at_sar_q": (sar["unsplit"], sar["soft"]),
             "sar_minus_baseline_at_unsplit": (baseline["unsplit"], sar["unsplit"]),
             "sar_minus_baseline_at_soft": (baseline["soft"], sar["soft"]),
             "combined_minus_baseline_unsplit": (baseline["unsplit"], sar["soft"])}
    result = {name: _delta(*values) for name, values in pairs.items()}
    aa = result["soft_minus_unsplit_at_baseline_q"]
    bb = result["soft_minus_unsplit_at_sar_q"]
    result["interaction_pp"] = {key: ([None if a is None or b is None else b-a for a, b in zip(aa[key], bb[key])]
        if isinstance(aa[key], list) else None if aa[key] is None or bb[key] is None else bb[key]-aa[key]) for key in aa}
    result["within_q_controls"] = {name: {"soft_minus_capacity_control": _delta(rows["capacity_control"], rows["soft"]),
                                          "soft_minus_hard": _delta(rows["hard"], rows["soft"])}
                                   for name, rows in metrics.items()}
    return result


def run(baseline_fit, baseline_eval, sar_fit, sar_eval, output):
    output = Path(output).resolve()
    roots = [Path(p).resolve() for p in (baseline_fit, baseline_eval, sar_fit, sar_eval)]
    if output.exists() or any(output.is_relative_to(p) for p in roots) or len(set(roots)) != 4:
        raise ValueError("Use four distinct inputs and a NEW output outside all inputs")
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    report = dict(format=REPORT_FORMAT, complete=False, full_validation_evidence=False,
                  training=False, test_accessed=False, image_count=0, statistical_significance_claim=False)
    try:
        evidence.bind(__file__)
        records = {"baseline": _load_fit(roots[0], roots[1], "baseline", evidence),
                   "sar": _load_fit(roots[2], roots[3], "sar_cue", evidence)}
        _matched(records["baseline"], records["sar"])
        journals = {key: _journal(value, evidence) for key, value in records.items()}
        totals = {key: {arm: np.zeros((4, 4), dtype=np.int64) for arm in ALL_ARMS} for key in records}
        events = {key: {} for key in records}
        pair_names = list(itertools.combinations(MAIN_CELLS, 2))
        cubes = {pair: np.zeros((4, 4, 4), dtype=np.int64) for pair in pair_names}
        event_cubes = {}
        target_digest = hashlib.sha256()
        for index, (left, right) in enumerate(zip(journals["baseline"], journals["sar"])):
            if any(left[key] != right[key] for key in ("index", "image", "event")):
                raise ValueError("Cross-Q validation journal identity/order differs")
            a, ya, pa, ca = _load_image(left, evidence)
            b, yb, pb, cb = _load_image(right, evidence)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError("Cross-Q validation targets differ: " + left["image"])
            event = left["event"]
            target_digest.update(json.dumps([index, left["image"], event, list(a.shape)], ensure_ascii=False).encode())
            target_digest.update(a.astype("<i2").tobytes())
            for key, image_cms in (("baseline", ca), ("sar", cb)):
                event_cm = events[key].setdefault(event, {arm: np.zeros((4, 4), dtype=np.int64) for arm in ALL_ARMS})
                for arm in ALL_ARMS:
                    totals[key][arm] += image_cms[arm]
                    event_cm[arm] += image_cms[arm]
            predictions = dict(baseline_unsplit=pa["unsplit"], baseline_soft=pa["soft"],
                               sar_unsplit=pb["unsplit"], sar_soft=pb["soft"])
            by_event = event_cubes.setdefault(event, {pair: np.zeros((4, 4, 4), dtype=np.int64) for pair in pair_names})
            for pair in pair_names:
                counts = np.bincount(ya*16 + predictions[pair[0]]*4 + predictions[pair[1]], minlength=64).reshape(4, 4, 4)
                cubes[pair] += counts
                by_event[pair] += counts
            report["image_count"] = index + 1
            if (index + 1) % 25 == 0:
                print(f"Verified paired route predictions {index + 1}/450", flush=True)
        aggregate = {}
        per_event = {}
        for key, record in records.items():
            stored = record["result"]
            if set(stored["per_event"]) != set(events[key]):
                raise ValueError("Archived prediction events differ from evaluation metrics")
            aggregate[key] = {arm: check_metrics(stored["metrics"][arm], totals[key][arm], key + "/" + arm) for arm in ALL_ARMS}
            for event, rows in events[key].items():
                per_event.setdefault(event, {})[key] = {arm: check_metrics(stored["per_event"][event][arm], rows[arm], key + "/" + event + "/" + arm) for arm in ALL_ARMS}
        evidence.verify()
        _write(output / "input_sha256.json", evidence.hashes)
        report.update(complete=True, full_validation_evidence=True, image_count=450,
                      source_certificate_sha256=records["baseline"]["worker"]["source_certificate_sha256"],
                      protocol_sha256=records["baseline"]["summary"]["protocol_sha256"],
                      source_sha256=records["baseline"]["summary"]["source_sha256"],
                      route_steps=2000, q_steps=2000, ordered_targets_sha256=target_digest.hexdigest(),
                      input_sha256_file="input_sha256.json", input_sha256_file_sha256=digest(output / "input_sha256.json"),
                      cells={key: {arm: _cell(rows[arm]) for arm in ARMS} for key, rows in aggregate.items()},
                      effects=effects(aggregate), all_verified_metrics=aggregate,
                      per_event={event: dict(cells={key: {arm: _cell(rows[arm]) for arm in ARMS} for key, rows in groups.items()},
                                                  effects=effects(groups)) for event, groups in per_event.items()},
                      paired_transitions={"__to__".join(pair): dict(reference=pair[0], candidate=pair[1], **transition_summary(cube)) for pair, cube in cubes.items()},
                      paired_transitions_per_event={event: {"__to__".join(pair): transition_summary(cube) for pair, cube in rows.items()} for event, rows in event_cubes.items()},
                      interpretation="Matched final-step Val450 foreground classes 1/2/3 mIoU. All four route heads have equal parameter counts; split arms have 2K slots and 18 routes. Counts are pixels, not buildings. Positive interaction is descriptive on the mIoU scale, not an independent-pixel significance test or a stable multi-seed mechanism claim. No Memory training or Test access.")
        _write(output / "report.json", report)
        return report
    except BaseException as exc:
        report.update(complete=False, full_validation_evidence=False,
                      error=type(exc).__name__ + ": " + str(exc))
        _write(output / "report.json", report)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-fit", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--sar-fit", required=True)
    parser.add_argument("--sar-evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.baseline_fit, args.baseline_evaluation, args.sar_fit, args.sar_evaluation, args.output)
    print(json.dumps({key: result[key] for key in ("complete", "image_count", "cells", "effects")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
