"""Read-only, paired Val450 prediction transitions for two matched Q arms.

Axes of every transition cube are (ground truth, baseline prediction,
candidate prediction).  The source artifacts were written only after each
forward; this program cannot train a model or access Test images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_saved_q import verified_images
from .compare import checked_eval, compare


FORMAT = "SRAC_Q_PAIRED_PREDICTION_V1"
CLASSES = ("Background", "Intact", "Damaged", "Destroyed")
ARMS = ("q_grid_probq", "q_region")


def _map(value: np.ndarray, name: str, *, target: bool = False) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer map")
    allowed = ((array >= 0) & (array < 4)) | (array == 255 if target else False)
    if not bool(np.all(allowed)):
        raise ValueError(f"{name} contains an invalid class ID")
    return array


def transition_cube(gt: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Count one image without including ignore=255 or silently clipping IDs."""
    gt = _map(gt, "target", target=True)
    old = _map(old, "baseline prediction")
    new = _map(new, "candidate prediction")
    if gt.shape != old.shape or gt.shape != new.shape:
        raise ValueError("Target and paired predictions have different shapes")
    valid = gt < 4
    indices = gt[valid].astype(np.int64) * 16 + old[valid].astype(np.int64) * 4 + new[valid]
    return np.bincount(indices, minlength=64).reshape(4, 4, 4).astype(np.int64)


def _recorded_confusion(row: dict, arm: str) -> np.ndarray:
    value = row.get("confusion", {}).get(arm)
    array = np.asarray(value)
    if array.shape != (4, 4) or not np.issubdtype(array.dtype, np.integer) or bool((array < 0).any()):
        raise ValueError(f"Missing or invalid recorded image confusion: {arm}")
    return array.astype(np.int64)


def _assert_confusion(cube: np.ndarray, old_cm: np.ndarray, new_cm: np.ndarray,
                      description: str) -> None:
    if not np.array_equal(cube.sum(axis=2), old_cm):
        raise ValueError(f"Baseline confusion disagrees with paired predictions: {description}")
    if not np.array_equal(cube.sum(axis=1), new_cm):
        raise ValueError(f"Candidate confusion disagrees with paired predictions: {description}")


def _formal_confusion(record: dict, arm: str, event: str | None = None) -> np.ndarray:
    metrics = record.get("metrics", {}) if event is None else record.get("per_event", {}).get(event, {})
    value = metrics.get(arm, {}).get("confusion")
    array = np.asarray(value)
    if array.shape != (4, 4) or not np.issubdtype(array.dtype, np.integer) or bool((array < 0).any()):
        raise ValueError(f"Missing formal confusion for {arm} / {event or 'all'}")
    return array.astype(np.int64)


def summarize_cube(cube: np.ndarray) -> dict:
    if cube.shape != (4, 4, 4) or not np.issubdtype(cube.dtype, np.integer) or bool((cube < 0).any()):
        raise ValueError("Expected a nonnegative 4x4x4 transition cube")
    classes = {}
    for cls, name in enumerate(CLASSES):
        old_correct = int(cube[cls, cls, :].sum())
        new_correct = int(cube[cls, :, cls].sum())
        rescue_from = {CLASSES[other]: int(cube[cls, other, cls])
                       for other in range(4) if other != cls}
        harm_to = {CLASSES[other]: int(cube[cls, cls, other])
                   for other in range(4) if other != cls}
        rescue, harm = sum(rescue_from.values()), sum(harm_to.values())
        if new_correct - old_correct != rescue - harm:
            raise ValueError("Class transition identity failed")
        classes[name] = dict(gt_pixels=int(cube[cls].sum()), old_correct=old_correct,
                             new_correct=new_correct, rescue=rescue, harm=harm,
                             correct_delta=rescue - harm, rescue_from=rescue_from,
                             harm_to=harm_to)
    false_positive_damaged = {}
    for cls, name in enumerate(CLASSES):
        if cls == 2:
            continue
        added = int(cube[cls, :, 2].sum() - cube[cls, 2, 2])
        removed = int(cube[cls, 2, :].sum() - cube[cls, 2, 2])
        false_positive_damaged[name] = dict(added=added, removed=removed,
                                            net_change=added - removed)
    false_positive_damaged["total"] = {
        key: sum(row[key] for row in false_positive_damaged.values())
        for key in ("added", "removed", "net_change")
    }
    damaged_prediction_destinations = {
        name: dict(baseline=int(cube[2, cls, :].sum()),
                   candidate=int(cube[2, :, cls].sum()),
                   delta=int(cube[2, :, cls].sum() - cube[2, cls, :].sum()))
        for cls, name in enumerate(CLASSES)
    }
    return dict(gt_baseline_candidate=cube.tolist(), classes=classes,
                damaged_false_positive_by_gt=false_positive_damaged,
                gt_damaged_prediction_destinations=damaged_prediction_destinations,
                baseline_confusion=cube.sum(axis=2).tolist(),
                candidate_confusion=cube.sum(axis=1).tolist())


def paired_report(baseline_path: str | Path, candidate_path: str | Path) -> dict:
    base_summary, base = checked_eval(baseline_path)
    candidate_summary, candidate = checked_eval(candidate_path)
    comparison = compare(base, candidate)
    base_root, candidate_root = base_summary.parent, candidate_summary.parent
    base_verified_summary, base_audit, base_images = verified_images(base_root)
    candidate_verified_summary, candidate_audit, candidate_images = verified_images(candidate_root)
    for checked, verified_summary, audit in (
        (base, base_verified_summary, base_audit),
        (candidate, candidate_verified_summary, candidate_audit),
    ):
        if checked != verified_summary:
            raise ValueError("Formal summary changed during artifact verification")
        for field in ("arm", "checkpoint_sha256", "protocol_sha256", "source_sha256"):
            if audit.get(field) != checked.get(field):
                raise ValueError("Verified prediction report differs from formal summary: " + field)
        for arm in ARMS:
            if not np.array_equal(_formal_confusion(checked, arm), _formal_confusion(audit, arm)):
                raise ValueError("Verified prediction report confusion differs from formal summary: " + arm)
    if len(base_images) != 450 or len(candidate_images) != 450:
        raise ValueError("Paired analysis requires two complete Val450 journals")
    totals = {arm: np.zeros((4, 4, 4), dtype=np.int64) for arm in ARMS}
    event_totals: dict[str, dict[str, np.ndarray]] = {}
    for (base_row, base_file), (candidate_row, candidate_file) in zip(base_images, candidate_images):
        image = base_row["image"]
        event = base_row.get("event")
        if (image != candidate_row.get("image") or base_row["index"] != candidate_row.get("index")
                or not isinstance(event, str) or event != candidate_row.get("event")):
            raise ValueError("Val image, order, or event differs between Q arms")
        with np.load(base_file, allow_pickle=False) as first, np.load(candidate_file, allow_pickle=False) as second:
            gt = _map(first["target"], "baseline target", target=True)
            if not np.array_equal(gt, _map(second["target"], "candidate target", target=True)):
                raise ValueError("Val target differs between Q arms: " + image)
            for arm in ARMS:
                key = "prediction_" + arm
                if key not in first or key not in second:
                    raise ValueError(f"Missing paired prediction {arm}: {image}")
                cube = transition_cube(gt, first[key], second[key])
                _assert_confusion(cube, _recorded_confusion(base_row, arm),
                                  _recorded_confusion(candidate_row, arm), f"{arm} / {image}")
                totals[arm] += cube
                event_totals.setdefault(event, {name: np.zeros((4, 4, 4), dtype=np.int64)
                                                for name in ARMS})[arm] += cube
    if set(event_totals) != set(base.get("per_event", {})) or set(event_totals) != set(candidate.get("per_event", {})):
        raise ValueError("Recorded per-event set differs from paired predictions")
    for arm in ARMS:
        _assert_confusion(totals[arm], _formal_confusion(base, arm),
                          _formal_confusion(candidate, arm), arm + " / all images")
        for event, cubes in event_totals.items():
            _assert_confusion(cubes[arm], _formal_confusion(base, arm, event),
                              _formal_confusion(candidate, arm, event), arm + " / " + event)
    return dict(format=FORMAT, complete=True, full_validation_evidence=True, image_count=450,
                test_accessed=False, training=False, baseline_arm=base["arm"],
                candidate_arm=candidate["arm"], baseline_summary=str(base_summary.resolve()),
                candidate_summary=str(candidate_summary.resolve()),
                protocol_sha256=base["protocol_sha256"], source_sha256=base["source_sha256"],
                comparison=comparison,
                arms={arm: summarize_cube(cube) for arm, cube in totals.items()},
                per_event={event: {arm: summarize_cube(cube) for arm, cube in cubes.items()}
                           for event, cubes in sorted(event_totals.items())},
                interpretation=(
                    "Pixel transitions compare two matched trained Q arms on the same Val truth. "
                    "They are descriptive, not building counts, a Memory effect, or a significance test."
                ))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="Completed baseline Q Val450 evaluation directory")
    p.add_argument("--candidate", required=True, help="Completed candidate Q Val450 evaluation directory")
    p.add_argument("--output", required=True, help="Fresh read-only analysis output directory")
    return p


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    report = paired_report(args.baseline, args.candidate)
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for arm, result in report["arms"].items():
        damaged = result["classes"]["Damaged"]
        intact_fp = result["damaged_false_positive_by_gt"]["Intact"]
        print(f"{arm}: Damaged rescue={damaged['rescue']}, harm={damaged['harm']}, "
              f"Intact->Damaged FP net={intact_fp['net_change']}")
    print(output / "report.json")


if __name__ == "__main__":
    main()
