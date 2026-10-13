#!/usr/bin/env python3
"""Audit saved Q Val predictions against their immutable recorded confusion.

This helper lives outside the checkout and imports the checkout's original
formal and SHA256 gates. It reads archived targets/predictions; it never runs a
model, relabels maps, repairs evidence, or accesses Test. --image is a limited
diagnostic and cannot establish a complete Val450 confusion audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ("Background", "Intact", "Damaged", "Destroyed")
FORMAT = "SRAC_Q_ARCHIVED_CONFUSION_AUDIT_V1"


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _official(repo_root: Path):
    """Use original gates without writing bytecode inside the repository."""
    repo_root = repo_root.resolve()
    if not (repo_root / "bda_benchmark").is_dir():
        raise ValueError("--repo-root must contain bda_benchmark: " + str(repo_root))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo_root))
    prefix = "bda_benchmark.experiments.damageformer_srac_q_factorial_v1."
    compare = importlib.import_module(prefix + "compare")
    artifacts = importlib.import_module(prefix + "audit_saved_q")
    for module in (compare, artifacts):
        if not Path(module.__file__).resolve().is_relative_to(repo_root):
            raise ValueError("A different repository was already imported; use a fresh process")
    return compare.ARMS, compare.checked_eval, artifacts.verified_images


def _map(value: np.ndarray, name: str, *, target: bool = False) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(name + " must be a two-dimensional integer map")
    legal = (array >= 0) & (array < 4)
    if target:
        legal |= array == 255
    if not bool(legal.all()):
        raise ValueError(name + " contains invalid class IDs; no remapping was performed")
    return array


def confusion_from_maps(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Rows=GT, columns=prediction, ignoring target 255 only."""
    gt = _map(target, "target", target=True)
    predicted = _map(prediction, "prediction")
    if gt.shape != predicted.shape:
        raise ValueError("Target and prediction shapes differ")
    valid = gt != 255
    encoded = gt[valid].astype(np.int64) * 4 + predicted[valid].astype(np.int64)
    return np.bincount(encoded, minlength=16).reshape(4, 4).astype(np.int64)


def _matrix(value, description: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != (4, 4) or not np.issubdtype(array.dtype, np.integer)
            or bool((array < 0).any()) or bool((array > np.iinfo(np.int64).max).any())):
        raise ValueError("Missing or invalid nonnegative 4x4 confusion: " + description)
    return array.astype(np.int64)


def delta_cells(recomputed: np.ndarray, stored: np.ndarray) -> list[dict]:
    """Exact integer differences; no metric tolerance or correction."""
    cells = []
    for gt, pred in np.argwhere(recomputed != stored):
        actual, expected = int(recomputed[gt, pred]), int(stored[gt, pred])
        cells.append(dict(gt_id=int(gt), gt_class=CLASSES[gt],
                          prediction_id=int(pred), prediction_class=CLASSES[pred],
                          recomputed=actual, stored=expected, delta=actual - expected))
    return cells


def _check_matrix(actual: np.ndarray, value, context: dict, report: dict) -> None:
    report["matrix_checks"] += 1
    try:
        stored = _matrix(value, str(context))
    except (ValueError, TypeError) as error:
        report["errors"].append(dict(**context, error=str(error)))
        return
    cells = delta_cells(actual, stored)
    if cells:
        report["mismatches"].append(dict(**context, cells=cells,
                                          recomputed_total=int(actual.sum()),
                                          stored_total=int(stored.sum()),
                                          delta_total=int(actual.sum()) - int(stored.sum())))


def audit_evaluation(repo_root: Path, evaluation: Path, image: str | None = None) -> dict:
    """Return consistency evidence; fatal original gate/hash failures propagate.

    --image selects one exact journal name. Original formal/SHA256 checks still
    run for the input evaluation, but global/event confusion is deliberately
    not compared against a one-image subset. complete_audit remains false.
    """
    repo_root, evaluation = Path(repo_root).resolve(), Path(evaluation).resolve()
    arms, checked_eval, verified_images = _official(repo_root)
    summary_path, summary = checked_eval(evaluation)
    root = summary_path.parent
    verified_summary, source_report, images = verified_images(root)
    if summary != verified_summary:
        raise ValueError("Formal summary changed during official verification")
    if len(images) != 450:
        raise ValueError("Original verification did not establish a 450-image journal")
    snapshots = {p: _digest(p) for p in (
        summary_path, root / summary["result_file"], root / "images.jsonl")}
    selected = images if image is None else [(row, p) for row, p in images if row["image"] == image]
    if not selected:
        raise ValueError("Requested image is absent from the verified journal: " + str(image))
    report = dict(format=FORMAT, passed=False, complete_audit=False,
                  full_validation_evidence=False, training=False, test_accessed=False,
                  evaluation=str(root), arm=summary["arm"], requested_image=image,
                  official_formal_gate_passed=True, artifact_hashes_verified=len(images),
                  image_count=0, expected_image_count=450, arms=list(arms),
                  confusion_axes=["ground_truth", "prediction"], ignore_target=255,
                  matrix_checks=0, errors=[], mismatches=[], images=[],
                  interpretation=("Archived predictions versus recorded confusion only. "
                                  "A limited image diagnostic cannot establish a complete Val450 audit."))
    for field in ("format", "arm", "image_count", "complete", "full_validation_evidence",
                  "test_accessed", "protocol_sha256", "source_sha256", "checkpoint_sha256",
                  "checkpoint_steps", "numerical_profile", "pool_mode", "training_hyperparameters"):
        if summary.get(field) != source_report.get(field):
            report["errors"].append(dict(scope="summary_vs_result", field=field,
                                         error="Formal summary and SHA-verified result differ"))
    totals = {arm: np.zeros((4, 4), dtype=np.int64) for arm in arms}
    events: dict[str, dict[str, np.ndarray]] = {}
    complete_maps = True
    ordered_targets = hashlib.sha256()
    targets_hashed = 0
    for row, archive in selected:
        name, event = row["image"], row.get("event")
        current = dict(image=name, index=row["index"], event=event, passed=False)
        before = len(report["errors"]) + len(report["mismatches"])
        if not isinstance(event, str) or not event:
            report["errors"].append(dict(scope="image", image=name, error="Missing event name"))
            complete_maps = False
        try:
            with np.load(archive, allow_pickle=False) as artifact:
                gt = _map(artifact["target"], "target", target=True)
                current.update(target_shape=list(gt.shape), valid_pixels=int((gt != 255).sum()),
                               ignored_pixels=int((gt == 255).sum()))
                metadata = json.dumps(dict(index=row["index"], image=name, event=event,
                                           shape=list(gt.shape), dtype=gt.dtype.str),
                                      ensure_ascii=False, sort_keys=True).encode("utf-8")
                ordered_targets.update(len(metadata).to_bytes(8, "big"))
                ordered_targets.update(metadata)
                target_bytes = np.ascontiguousarray(gt).tobytes()
                ordered_targets.update(len(target_bytes).to_bytes(8, "big"))
                ordered_targets.update(target_bytes)
                targets_hashed += 1
                for arm in arms:
                    try:
                        actual = confusion_from_maps(gt, artifact["prediction_" + arm])
                        _check_matrix(actual, row.get("confusion", {}).get(arm),
                                      dict(scope="image", record="journal", image=name, arm=arm), report)
                        totals[arm] += actual
                        if isinstance(event, str) and event:
                            event_totals = events.setdefault(
                                event, {key: np.zeros((4, 4), dtype=np.int64) for key in arms})
                            event_totals[arm] += actual
                    except (KeyError, ValueError, TypeError) as error:
                        complete_maps = False
                        report["errors"].append(dict(scope="image", image=name, arm=arm,
                                                     error=f"{type(error).__name__}: {error}"))
            if _digest(archive) != row["prediction_sha256"]:
                complete_maps = False
                report["errors"].append(dict(scope="image", image=name,
                                             error="Artifact SHA256 changed while auditing"))
        except (OSError, KeyError, ValueError, TypeError) as error:
            complete_maps = False
            report["errors"].append(dict(scope="image", image=name,
                                         error=f"{type(error).__name__}: {error}"))
        current["passed"] = before == len(report["errors"]) + len(report["mismatches"])
        report["images"].append(current)
        report["image_count"] += 1
    if image is None and complete_maps:
        for record_name, recorded in (("summary", summary), ("result", source_report)):
            stored_events = recorded.get("per_event", {})
            if set(events) != set(stored_events):
                report["errors"].append(dict(scope="events", record=record_name,
                                             error="Recorded event set differs from prediction journal",
                                             recomputed_events=sorted(events), stored_events=sorted(stored_events)))
            for arm in arms:
                _check_matrix(totals[arm], recorded.get("metrics", {}).get(arm, {}).get("confusion"),
                              dict(scope="global", record=record_name, arm=arm), report)
                for event, cms in events.items():
                    value = stored_events.get(event, {}).get(arm, {}).get("confusion")
                    _check_matrix(cms[arm], value,
                                  dict(scope="event", record=record_name, event=event, arm=arm), report)
    for path, initial_hash in snapshots.items():
        if _digest(path) != initial_hash:
            report["errors"].append(dict(scope="immutable", path=str(path),
                                         error="Evidence file changed while auditing"))
    report["recomputed_confusion"] = {arm: cm.tolist() for arm, cm in totals.items()}
    report["recomputed_per_event"] = {
        event: {arm: cm.tolist() for arm, cm in cms.items()} for event, cms in sorted(events.items())}
    report["complete_audit"] = bool(image is None and complete_maps and report["image_count"] == 450)
    report["ordered_targets_sha256"] = (ordered_targets.hexdigest()
                                           if image is None and targets_hashed == 450 else None)
    report["targets_hashed"] = targets_hashed
    report["passed"] = not report["errors"] and not report["mismatches"]
    report["full_validation_evidence"] = report["passed"] and report["complete_audit"]
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="BRIGHT checkout; defaults to cwd")
    p.add_argument("--evaluation", type=Path, required=True, help="Formal Q Val450 directory or summary.json")
    p.add_argument("--image", help="Exact journal image name; limited diagnosis, never a full Val450 audit")
    p.add_argument("--output", type=Path, help="Optional fresh JSON file outside repo and source evaluation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output:
        output = args.output.resolve()
        repo, source = args.repo_root.resolve(), args.evaluation.resolve()
        source = source if source.is_dir() else source.parent
        if output.is_relative_to(repo) or output.is_relative_to(source):
            print("Output must be outside both the checkout and source evaluation", file=sys.stderr)
            return 2
        if output.exists() or not output.parent.is_dir():
            print("Use a fresh output file in an existing external directory", file=sys.stderr)
            return 2
    try:
        report = audit_evaluation(args.repo_root, args.evaluation, args.image)
    except Exception as error:
        report = dict(format=FORMAT, passed=False, complete_audit=False,
                      full_validation_evidence=False, training=False, test_accessed=False,
                      evaluation=str(args.evaluation.resolve()), requested_image=args.image,
                      errors=[dict(scope="official_gate_or_read", error=f"{type(error).__name__}: {error}")],
                      mismatches=[])
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
        except OSError as error:
            print(f"Could not create fresh output: {error}", file=sys.stderr)
            return 2
    print(content, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
