"""Read-only, paired spatial attribution of GT-derived Q label roundtrips.

The two oracles deliberately use Val truth *after* the frozen forward pass.
They are information-transfer probes, not model predictions or achievable
segmentation scores. This entry point never reads Test, trains, or changes Q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_saved_q import _q_arrays, _two_dim, verified_images
from .diagnostics import nearest_grid_pixels, probe_prediction


FORMAT = "SRAC_Q_PAIRED_SPATIAL_DIAGNOSTIC_V1"
BOUNDARY_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, None))
EXPOSURE_BINS = ((0.0, 1e-8), (1e-8, 0.2), (0.2, 0.5), (0.5, 1.000001))
COUNT_KEYS = (
    "gt_damaged_pixels", "grid_correct", "q_correct", "lost_by_q",
    "gained_by_q", "correct_in_both", "wrong_in_both", "lost_to_background",
    "lost_to_intact", "lost_to_destroyed",
)
FP_KEYS = (
    "gt_non_damaged_pixels", "grid_false_positive", "q_false_positive",
    "introduced", "removed", "introduced_from_background",
    "introduced_from_intact", "introduced_from_destroyed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def damaged_boundary_distance(target: np.ndarray) -> np.ndarray:
    """4-neighbor pixel distance *inside* GT Damaged to its nearest boundary.

    A Damaged pixel touching another class, ignore label, or image exterior has
    distance zero. Distance is measured in raster pixels, not building units.
    """
    from scipy import ndimage

    gt = np.asarray(target)
    if gt.ndim != 2:
        raise ValueError("Expected a two-dimensional target")
    mask = gt == 2
    # Padding makes image-edge behavior explicit, even for an all-Damaged map.
    steps = ndimage.distance_transform_cdt(
        np.pad(mask, 1, mode="constant", constant_values=False), metric="taxicab"
    )[1:-1, 1:-1]
    return np.where(mask, steps - 1, -1).astype(np.int32, copy=False)


def q_mixing_exposure(q: np.ndarray, ids: np.ndarray,
                      grid_shape: tuple[int, int], labels: np.ndarray,
                      labelled_mass: np.ndarray, output_shape: tuple[int, int]):
    """Soft Q-weighted exposure to mixed historical *regions* per pixel.

    The region labels are computed from GT solely for diagnosis. The two flags
    use the existing frozen diagnostic's distinct background/foreground and
    foreground-class definitions; they do not claim building instance IDs.
    """
    weights, rid = np.asarray(q, dtype=np.float64), np.asarray(ids)
    label, mass = np.asarray(labels, dtype=np.float64), np.asarray(labelled_mass)
    if (weights.ndim != 2 or weights.shape != rid.shape or
            weights.shape[1] != int(np.prod(grid_shape)) or
            label.ndim != 2 or label.shape[1] != 4 or mass.shape != label.shape[:1] or
            not np.issubdtype(rid.dtype, np.integer) or (rid < 0).any() or
            (rid >= len(label)).any() or not np.isfinite(weights).all() or
            (weights < -1e-6).any() or
            not np.allclose(weights.sum(axis=0), 1, atol=2e-5, rtol=0)):
        raise ValueError("Invalid Q/region labels for soft mixing exposure")
    valid = mass > 1e-8
    foreground = label[:, 1:].sum(axis=1)
    purity = label[:, 1:].max(axis=1) / np.maximum(foreground, 1e-12)
    flags = {
        "background_mixing": valid & (foreground >= 0.2) & (foreground < 0.8),
        "foreground_class_mixing": valid & (foreground >= 0.8) & (purity < 0.8),
    }
    out = {}
    for name, flag in flags.items():
        grid = (weights * flag[rid]).sum(axis=0).reshape(grid_shape)
        exposed = nearest_grid_pixels(grid, output_shape)
        if (not np.isfinite(exposed).all() or (exposed < -2e-5).any() or
                (exposed > 1 + 2e-5).any()):
            raise ValueError("Q mixing exposure is outside [0,1]")
        out[name] = np.clip(exposed, 0, 1)
    return out


def _empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def _add_counts(destination: dict, source: dict):
    for key in COUNT_KEYS:
        destination[key] += int(source[key])


def false_positive_counts(target: np.ndarray, grid_prediction: np.ndarray,
                          q_prediction: np.ndarray) -> dict:
    """Paired GT-non-Damaged false positives for the two offline probes."""
    gt, grid, q = (np.asarray(value) for value in
                   (target, grid_prediction, q_prediction))
    if gt.shape != grid.shape or gt.shape != q.shape:
        raise ValueError("False-positive arrays must have matching shapes")
    valid = (gt >= 0) & (gt < 4) & (gt != 2)
    old_fp = valid & (grid == 2)
    new_fp = valid & (q == 2)
    introduced = new_fp & ~old_fp
    result = dict(
        gt_non_damaged_pixels=int(valid.sum()),
        grid_false_positive=int(old_fp.sum()),
        q_false_positive=int(new_fp.sum()),
        introduced=int(introduced.sum()),
        removed=int((old_fp & ~new_fp).sum()),
        introduced_from_background=int((introduced & (gt == 0)).sum()),
        introduced_from_intact=int((introduced & (gt == 1)).sum()),
        introduced_from_destroyed=int((introduced & (gt == 3)).sum()),
    )
    if result["introduced"] != sum(result[key] for key in (
            "introduced_from_background", "introduced_from_intact",
            "introduced_from_destroyed")):
        raise ValueError("Introduced false positives do not balance")
    return result


def _add_false_positive(destination: dict, source: dict):
    for key in FP_KEYS:
        destination[key] += int(source[key])


def paired_counts(grid_correct: np.ndarray, q_correct: np.ndarray,
                  q_prediction: np.ndarray, select: np.ndarray) -> dict:
    """Counts for the selected GT-Damaged pixels, including gains and losses."""
    g, q, pred, selected = (np.asarray(grid_correct, dtype=bool),
                            np.asarray(q_correct, dtype=bool),
                            np.asarray(q_prediction), np.asarray(select, dtype=bool))
    if g.shape != q.shape or g.shape != pred.shape or g.shape != selected.shape:
        raise ValueError("Paired damaged arrays must have matching shapes")
    lost = selected & g & ~q
    result = dict(
        gt_damaged_pixels=int(selected.sum()),
        grid_correct=int((selected & g).sum()),
        q_correct=int((selected & q).sum()),
        lost_by_q=int(lost.sum()),
        gained_by_q=int((selected & ~g & q).sum()),
        correct_in_both=int((selected & g & q).sum()),
        wrong_in_both=int((selected & ~g & ~q).sum()),
        lost_to_background=int((lost & (pred == 0)).sum()),
        lost_to_intact=int((lost & (pred == 1)).sum()),
        lost_to_destroyed=int((lost & (pred == 3)).sum()),
    )
    if (result["gt_damaged_pixels"] != sum(result[k] for k in
            ("lost_by_q", "gained_by_q", "correct_in_both", "wrong_in_both")) or
            result["lost_by_q"] != sum(result[k] for k in
            ("lost_to_background", "lost_to_intact", "lost_to_destroyed"))):
        raise ValueError("Paired damaged accounting does not balance")
    return result


def _finish_counts(counts: dict, total_loss: int):
    result = dict(counts)
    gt, grid = result["gt_damaged_pixels"], result["grid_correct"]
    result["net_correct_change"] = result["gained_by_q"] - result["lost_by_q"]
    result["loss_fraction_of_gt_damaged"] = result["lost_by_q"] / gt if gt else None
    result["loss_rate_given_grid_correct"] = result["lost_by_q"] / grid if grid else None
    result["share_of_all_q_losses"] = (result["lost_by_q"] / total_loss
                                      if total_loss else None)
    return result


def _stratified_counts(values, bins, grid_ok, q_ok, q_prediction):
    output = {}
    for label, lower, upper in bins:
        selected = values >= lower
        if upper is not None:
            selected &= values < upper
        output[label] = paired_counts(grid_ok, q_ok, q_prediction, selected)
    return output


def _add_strata(destination: dict, incoming: dict):
    if not destination:
        destination.update({key: _empty_counts() for key in incoming})
    if set(destination) != set(incoming):
        raise ValueError("Paired diagnostic strata changed between images")
    for name, counts in incoming.items():
        _add_counts(destination[name], counts)


def _check_partition(strata: dict, overall: dict):
    for key in COUNT_KEYS:
        if sum(item[key] for item in strata.values()) != overall[key]:
            raise ValueError("Spatial stratum lost paired GT-Damaged pixels: " + key)


def _analyse_image(gt: np.ndarray, q: np.ndarray, ids: np.ndarray,
                   shape: tuple[int, int], building_grid: np.ndarray):
    grid, q_back, labels, mass = probe_prediction(gt, q, ids, shape)
    damaged = gt == 2
    grid_ok = (grid[damaged] == 2)
    q_ok = (q_back[damaged] == 2)
    q_classes = q_back[damaged]
    all_pixels = np.ones(grid_ok.shape, dtype=bool)
    overall = paired_counts(grid_ok, q_ok, q_classes, all_pixels)

    distance = damaged_boundary_distance(gt)[damaged]
    distance_bins = [(f"{lower}-{upper if upper is not None else 'inf'}", lower,
                      None if upper is None else upper + 1)
                     for lower, upper in BOUNDARY_BINS]
    by_boundary = _stratified_counts(distance, distance_bins, grid_ok, q_ok, q_classes)

    b = np.asarray(building_grid, dtype=np.float64)
    if (b.ndim != 2 or b.shape != shape or not np.isfinite(b).all() or
            (b < -1e-6).any() or (b > 1 + 1e-6).any()):
        raise ValueError("Saved predicted B must match the Q feature grid")
    pixel_b = nearest_grid_pixels(b, gt.shape)[damaged]
    b_bins = (("below_0.5", -1e-6, 0.5), ("at_or_above_0.5", 0.5, 1.000001))
    by_b = _stratified_counts(pixel_b, b_bins, grid_ok, q_ok, q_classes)
    by_joint = {}
    for distance_name, lower, upper in distance_bins:
        distance_mask = distance >= lower
        if upper is not None:
            distance_mask &= distance < upper
        for b_name, b_lower, b_upper in b_bins:
            selected = distance_mask & (pixel_b >= b_lower) & (pixel_b < b_upper)
            by_joint[distance_name + "|" + b_name] = paired_counts(
                grid_ok, q_ok, q_classes, selected)

    exposed = q_mixing_exposure(q, ids, shape, labels, mass, gt.shape)
    exposure_bins = [(f"[{lower:g},{upper:g})", lower, upper)
                     for lower, upper in EXPOSURE_BINS]
    by_exposure = {
        name: _stratified_counts(values[damaged], exposure_bins,
                                 grid_ok, q_ok, q_classes)
        for name, values in exposed.items()
    }
    for strata in (by_boundary, by_b, by_joint, *by_exposure.values()):
        _check_partition(strata, overall)
    mixing_sums = {
        name + "_sum_on_gt_damaged": float(values[damaged].sum())
        for name, values in exposed.items()
    }
    mixing_sums.update({
        name + "_sum_on_q_losses": float(values[damaged][grid_ok & ~q_ok].sum())
        for name, values in exposed.items()
    })
    return dict(paired=overall,
                false_positive=false_positive_counts(gt, grid, q_back),
                by_boundary_distance=by_boundary,
                by_building_probability=by_b, by_boundary_and_building=by_joint,
                by_background_mixing_exposure=by_exposure["background_mixing"],
                by_foreground_class_mixing_exposure=by_exposure["foreground_class_mixing"],
                mixing_exposure_sums=mixing_sums)


def _empty_arm():
    return dict(paired=_empty_counts(),
                false_positive={key: 0 for key in FP_KEYS}, by_boundary_distance={},
                by_building_probability={}, by_boundary_and_building={},
                by_background_mixing_exposure={},
                by_foreground_class_mixing_exposure={}, mixing_exposure_sums={})


def _add_arm(destination: dict, incoming: dict):
    _add_counts(destination["paired"], incoming["paired"])
    _add_false_positive(destination["false_positive"], incoming["false_positive"])
    for name in ("by_boundary_distance", "by_building_probability",
                 "by_boundary_and_building", "by_background_mixing_exposure",
                 "by_foreground_class_mixing_exposure"):
        _add_strata(destination[name], incoming[name])
    for name, value in incoming["mixing_exposure_sums"].items():
        destination["mixing_exposure_sums"][name] = (
            destination["mixing_exposure_sums"].get(name, 0.0) + value)


def _finish_arm(arm: dict):
    loss = arm["paired"]["lost_by_q"]
    for name in ("by_boundary_distance", "by_building_probability",
                 "by_boundary_and_building", "by_background_mixing_exposure",
                 "by_foreground_class_mixing_exposure"):
        _check_partition(arm[name], arm["paired"])
        arm[name] = {key: _finish_counts(value, loss)
                     for key, value in arm[name].items()}
    arm["paired"] = _finish_counts(arm["paired"], loss)
    fp = arm["false_positive"]
    if fp["q_false_positive"] - fp["grid_false_positive"] != fp["introduced"] - fp["removed"]:
        raise ValueError("False-positive oracle accounting does not balance")
    fp["net_change"] = fp["introduced"] - fp["removed"]
    for name in ("background_mixing", "foreground_class_mixing"):
        total = arm["paired"]["gt_damaged_pixels"]
        lost = arm["paired"]["lost_by_q"]
        arm["mixing_exposure_sums"][name + "_mean_on_gt_damaged"] = (
            arm["mixing_exposure_sums"].get(name + "_sum_on_gt_damaged", 0.0) / total
            if total else None)
        arm["mixing_exposure_sums"][name + "_mean_on_q_losses"] = (
            arm["mixing_exposure_sums"].get(name + "_sum_on_q_losses", 0.0) / lost
            if lost else None)


def diagnose(root: Path, output: Path, smoke: bool = False):
    """Verify frozen Val evidence, then attribute paired oracle changes."""
    source, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    summary, source_report, images = verified_images(source, smoke=smoke)
    output.mkdir(parents=True)
    aggregate = {"old": _empty_arm(), "new": _empty_arm()}
    per_event = {}
    image_rows = []
    seen = {"old": 0, "new": 0}
    for row, file in images:
        with np.load(file, allow_pickle=False) as saved:
            gt = _two_dim(saved["target"], "target")
            event = row.get("event", row["image"].rsplit("_", 1)[0])
            record = dict(image=row["image"], event=event, arms={})
            for arm, suffix in (("old", ""), ("new", "_new")):
                got = _q_arrays(saved, suffix)
                if got is None:
                    if arm == "old":
                        raise ValueError("Saved image lacks historical Q")
                    continue
                b_name = "building_grid" + suffix
                if b_name not in saved:
                    raise ValueError("Saved image lacks matching predicted B: " + b_name)
                q, ids, shape = got
                b = _two_dim(saved[b_name], b_name)
                result = _analyse_image(gt, q, ids, shape, b)
                _add_arm(aggregate[arm], result)
                event_arm = per_event.setdefault(event, {}).setdefault(arm, _empty_arm())
                _add_arm(event_arm, result)
                record["arms"][arm] = dict(paired=result["paired"],
                                            false_positive=result["false_positive"])
                seen[arm] += 1
            image_rows.append(record)
    if not seen["old"] or (seen["new"] and seen["new"] != seen["old"]):
        raise ValueError("New Q is present in only some images, or old Q is absent")
    for arm in list(aggregate):
        if seen[arm]:
            _finish_arm(aggregate[arm])
        else:
            del aggregate[arm]
    for arms in per_event.values():
        for value in arms.values():
            _finish_arm(value)
    source_name = summary.get("result_file", summary.get("audit_file"))
    source_file = (source / source_name).resolve()
    report = dict(
        format=FORMAT, complete=True,
        full_validation_evidence=bool(not smoke and
                                      summary.get("full_validation_evidence")),
        image_count=len(images), smoke=smoke, test_accessed=False, training=False,
        source_root=str(source), source_format=summary["format"],
        source_report_sha256=_sha256(source_file),
        source_reproduction=source_report.get("reproduction"),
        interpretation=(
            "These are paired Val-GT label-transfer probes, not deployable "
            "predictions or causal Q effects on real model IoU. Boundary distance "
            "uses 4-neighbor raster pixels, not building instances; B is frozen "
            "predicted feature-grid probability nearest-expanded for attribution. "
            "Mixing exposure is the Q-weighted share of regions meeting fixed "
            "GT-derived background/foreground or foreground-class mixing rules."
        ),
        definitions=dict(
            paired=("GT Damaged: grid correct and Q wrong = lost_by_q; grid wrong "
                    "and Q correct = gained_by_q. All strata retain the same "
                    "pixel denominator within their own subset."),
            false_positive=(
                "GT non-Damaged: introduced means grid did not predict Damaged but "
                "Q did; removed means grid predicted Damaged but Q did not."
            ),
            boundary_distance="0 is adjacent to non-Damaged or image exterior; pixel units",
            building_probability="predicted B <0.5 versus >=0.5, nearest-expanded",
            background_mixing="Q region foreground fraction in [0.2,0.8)",
            foreground_class_mixing=(
                "Q region foreground fraction >=0.8 and largest foreground "
                "class share <0.8"),
        ),
        arms=aggregate, per_event=per_event,
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (output / "images.jsonl").open("x", encoding="utf-8") as stream:
        for row in image_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--samepass", required=True,
                         help="Completed same-pass Val with SHA256 image journal")
    command.add_argument("--output", required=True, help="Fresh output directory")
    command.add_argument("--smoke", action="store_true",
                         help="Allow a completed partial Val; results are never formal")
    return command


if __name__ == "__main__":
    cli = parser().parse_args()
    result = diagnose(Path(cli.samepass), Path(cli.output), smoke=cli.smoke)
    overview = dict(complete=result["complete"],
                    full_validation_evidence=result["full_validation_evidence"],
                    image_count=result["image_count"], arms={})
    for name, arm in result["arms"].items():
        paired, fp = arm["paired"], arm["false_positive"]
        overview["arms"][name] = dict(
            damaged_lost=paired["lost_by_q"],
            damaged_gained=paired["gained_by_q"],
            damaged_net_correct=paired["net_correct_change"],
            damaged_false_positive_introduced=fp["introduced"],
            damaged_false_positive_removed=fp["removed"],
            damaged_false_positive_net=fp["net_change"],
        )
    print(json.dumps(overview, ensure_ascii=False))
    old = result["arms"]["old"]
    for dimension in ("by_boundary_distance", "by_building_probability",
                      "by_background_mixing_exposure",
                      "by_foreground_class_mixing_exposure"):
        strata = {
            label: dict(damaged=row["gt_damaged_pixels"], lost=row["lost_by_q"],
                        gained=row["gained_by_q"],
                        loss_rate_given_grid_correct=row["loss_rate_given_grid_correct"])
            for label, row in old[dimension].items()
        }
        print(json.dumps(dict(arm="old", dimension=dimension, strata=strata),
                         ensure_ascii=False))
