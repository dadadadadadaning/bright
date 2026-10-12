"""Audit saved, immutable same-pass Val Q artifacts without another GPU pass.

This is deliberately GT-after-forward analysis.  Its grid and Q label
roundtrips use validation truth and must never be presented as deployable
predictions, used for support selection, or used for Test tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnostics import (
    BUILDING_PROBABILITY_BINS, CLASSES, SIZE_BINS, area_class_fractions,
    building_probability_stats, component_building_stats, component_proxy_stats,
    confusion, prediction_confusion_by_building_probability, probe_prediction,
    region_mixing_mass,
)


FORMAT = "SRAC_Q_FROZEN_DIAGNOSTIC_V1"
SOURCE_FORMATS = {"SRAC_Q_MEMORY_SAMEPASS_V1", "SRAC_Q_FACTORIAL_EVAL_V1"}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _artifact(root, relative):
    p = (root / str(relative)).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError("Artifact path escapes root or is missing")
    return p


def verified_images(root: Path, smoke: bool = False):
    """Require complete formal evidence unless `smoke=True` is explicit."""
    root = root.resolve()
    summary = _json(root / "summary.json")
    if summary.get("format") not in SOURCE_FORMATS:
        raise ValueError("Unsupported same-pass source format")
    if summary.get("complete") is not True or summary.get("test_accessed") is not False:
        raise ValueError("Source audit is incomplete or accessed Test")
    if not smoke:
        if (summary.get("image_count") != 450 or
                summary.get("full_validation_evidence") is not True or
                summary.get("reproduction_passed") is not True or
                not summary.get("immutable") or
                not all(v is True for v in summary["immutable"].values())):
            raise ValueError("Only reproduced, immutable, full Val450 evidence is formal")
    report_name = summary.get("result_file", summary.get("audit_file"))
    if not isinstance(report_name, str):
        raise ValueError("Missing source report file")
    report_path = _artifact(root, report_name)
    if (summary["format"] == "SRAC_Q_FACTORIAL_EVAL_V1" and
            not isinstance(summary.get("result_sha256"), str)):
        raise ValueError("New Q evaluation requires result SHA256")
    if "result_sha256" in summary and _digest(report_path) != summary["result_sha256"]:
        raise ValueError("Source report hash mismatch")
    audit = _json(report_path)
    if (audit.get("format") != summary["format"] or
            audit.get("complete") is not True or
            audit.get("test_accessed") is not False):
        raise ValueError("Source audit report is incomplete or accessed Test")
    if not smoke:
        if (audit.get("full_validation_evidence") is not True or
                audit.get("image_count") != 450 or
                audit.get("reproduction", {}).get("passed") is not True or
                not audit.get("immutable") or
                not all(v is True for v in audit["immutable"].values())):
            raise ValueError("Source audit report failed its formal gate")
    journal = [json.loads(line) for line in
               (root / "images.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(journal) != summary.get("image_count") or len(journal) != audit.get("image_count"):
        raise ValueError("Journal count differs from evidence")
    if [row.get("index") for row in journal] != list(range(len(journal))):
        raise ValueError("Image journal is out of order")
    names = [row.get("image") for row in journal]
    if not all(isinstance(v, str) and v for v in names) or len(names) != len(set(names)):
        raise ValueError("Image names are missing or repeated")
    checked = []
    for row in journal:
        if not isinstance(row.get("prediction_sha256"), str):
            raise ValueError("Missing artifact SHA256")
        file = _artifact(root, row.get("prediction_file"))
        if _digest(file) != row["prediction_sha256"]:
            raise ValueError("Prediction artifact hash mismatch: " + str(file))
        checked.append((row, file))
    return summary, audit, checked


def _two_dim(value, name):
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(name + " must reduce to a 2-D map")
    return array


def _q_arrays(record, suffix):
    q_name = "q" + suffix
    id_name = "region_ids" + suffix
    shape_name = "grid_shape" + suffix
    if q_name not in record:
        return None
    for name in (id_name, shape_name):
        if name not in record:
            raise ValueError("Missing paired " + name)
    q = np.asarray(record[q_name])
    if q.ndim == 3 and q.shape[0] == 1:
        q = q[0]
    shape = tuple(int(v) for v in np.asarray(record[shape_name]).ravel())
    return q, np.asarray(record[id_name]), shape


def _empty_component_totals():
    return {
        name: [dict(pixel_area_range=f"{lo}-{hi or 'inf'}",
                    connected_raster_components=0, gt_pixels=0,
                    grid_oracle_correct=0, q_oracle_correct=0)
               for lo, hi in SIZE_BINS]
        for name in CLASSES[1:]
    }


def _accumulate_components(total, incoming):
    for name in CLASSES[1:]:
        for old, new in zip(total[name], incoming[name]):
            if old["pixel_area_range"] != new["pixel_area_range"]:
                raise ValueError("Incompatible component bins")
            for key in ("connected_raster_components", "gt_pixels",
                        "grid_oracle_correct", "q_oracle_correct"):
                old[key] += new[key]


def _empty_component_building():
    return {
        name: [dict(pixel_area_range=f"{lo}-{hi or 'inf'}",
                    connected_raster_components=0, gt_pixels=0,
                    predicted_building_probability_sum=0.0,
                    gt_pixels_at_or_above_threshold=0,
                    components_majority_below_threshold=0)
               for lo, hi in SIZE_BINS]
        for name in CLASSES[1:]
    }


def _accumulate_component_building(total, incoming):
    for name in CLASSES[1:]:
        for old, new in zip(total[name], incoming[name]):
            if old["pixel_area_range"] != new["pixel_area_range"]:
                raise ValueError("Incompatible component bins")
            for key in ("connected_raster_components", "gt_pixels",
                        "predicted_building_probability_sum",
                        "gt_pixels_at_or_above_threshold",
                        "components_majority_below_threshold"):
                old[key] += new[key]


def _add_building(total, stats):
    for name, value in stats.items():
        dest = total.setdefault(name, dict(gt_grid_mass=0.0,
                                           predicted_building_mass=0.0,
                                           low_probability_grid_mass={k: 0.0 for k in
                                                                      value["low_probability_mass_fraction"]}))
        mass = value["gt_grid_mass"]
        dest["gt_grid_mass"] += mass
        if mass:
            dest["predicted_building_mass"] += mass * value["mean_predicted_building_probability"]
            for threshold, fraction in value["low_probability_mass_fraction"].items():
                dest["low_probability_grid_mass"][threshold] += mass * fraction


def _finish_building(total):
    for value in total.values():
        mass = value["gt_grid_mass"]
        value["mean_predicted_building_probability"] = (
            value.pop("predicted_building_mass") / mass if mass else None)
        value["low_probability_mass_fraction"] = {
            key: (amount / mass if mass else None)
            for key, amount in value.pop("low_probability_grid_mass").items()
        }


def audit(root: Path, output: Path, smoke: bool = False):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    summary, source_audit, images = verified_images(root, smoke=smoke)
    output.mkdir(parents=True)
    arms = {"old": "", "new": "_new"}
    all_cm = {arm: {key: np.zeros((4, 4), dtype=np.int64)
                    for key in ("area_grid_oracle", "q_roundtrip_oracle")}
              for arm in arms}
    events = {}
    components = {arm: _empty_component_totals() for arm in arms}
    component_b = {arm: _empty_component_building() for arm in arms}
    building = {arm: {} for arm in arms}
    building_by_event = {}
    available_b = {arm: 0 for arm in arms}
    evaluated = {arm: 0 for arm in arms}
    prediction_by_b = {arm: {} for arm in arms}
    mixing = {arm: {} for arm in arms}
    detail = []
    for row, file in images:
        with np.load(file, allow_pickle=False) as item:
            gt = _two_dim(item["target"], "target")
            image = row["image"]
            event = row.get("event", image.rsplit("_", 1)[0])
            record = {"image": image, "event": event, "arms": {}}
            for arm, suffix in arms.items():
                got = _q_arrays(item, suffix)
                if got is None:
                    if arm == "old":
                        raise ValueError("Source artifact lacks historical Q")
                    continue
                q, ids, shape = got
                grid, q_back, labels, mass = probe_prediction(gt, q, ids, shape)
                region_mix = region_mixing_mass(labels, mass)
                summary_mix = mixing[arm]
                for name, value in region_mix.items():
                    if isinstance(value, (int, float)):
                        summary_mix[name] = summary_mix.get(name, 0) + value
                summary_mix["threshold_rule"] = region_mix["threshold_rule"]
                cms = {
                    "area_grid_oracle": confusion(gt, grid),
                    "q_roundtrip_oracle": confusion(gt, q_back),
                }
                for key, cm in cms.items():
                    all_cm[arm][key] += cm
                    events.setdefault(event, {}).setdefault(arm, {}).setdefault(
                        key, np.zeros((4, 4), dtype=np.int64))[:] += cm
                comp = component_proxy_stats(gt, grid, q_back)
                _accumulate_components(components[arm], comp)
                fractions, _ = area_class_fractions(gt, shape)
                b_name = "building_grid" + suffix
                if b_name in item:
                    b = _two_dim(item[b_name], b_name)
                    b_stats = building_probability_stats(b, fractions)
                    _add_building(building[arm], b_stats)
                    _add_building(building_by_event.setdefault(event, {}).setdefault(arm, {}),
                                  b_stats)
                    _accumulate_component_building(component_b[arm],
                                                   component_building_stats(gt, b))
                    prediction_names = (("a1_pixel", "grid_probq", "j3") if arm == "old"
                                        else ("q_a1_pixel", "q_grid_probq", "q_region"))
                    for name in prediction_names:
                        key = "prediction_" + name
                        if key not in item:
                            continue
                        pred = _two_dim(item[key], key)
                        by_bin = prediction_confusion_by_building_probability(gt, pred, b)
                        current = prediction_by_b[arm].setdefault(
                            name, np.zeros_like(by_bin))
                        current += by_bin
                    available_b[arm] += 1
                evaluated[arm] += 1
                record["arms"][arm] = dict(
                    grid_correct=int(np.trace(cms["area_grid_oracle"])),
                    q_correct=int(np.trace(cms["q_roundtrip_oracle"])),
                    gt_pixels=int(cms["area_grid_oracle"].sum()),
                    empty_labelled_regions=int((mass <= 1e-8).sum()),
                    q_region_count=int(len(labels)),
                    building_probability_available=b_name in item,
                )
            detail.append(record)
    for arm in arms:
        _finish_building(building[arm])
    for arms_data in building_by_event.values():
        for values in arms_data.values():
            _finish_building(values)
    if 0 < evaluated["new"] < evaluated["old"]:
        raise ValueError("Only some images contain the new Q variant")
    for arm in arms:
        if 0 < available_b[arm] < evaluated[arm]:
            raise ValueError("Building probability is missing from some images")
    report = dict(
        format=FORMAT, complete=True,
        full_validation_evidence=bool(not smoke and summary["full_validation_evidence"]),
        image_count=len(images), smoke=smoke, test_accessed=False, training=False,
        source_root=str(root), source_format=summary.get("format"),
        source_audit_sha256=_digest(_artifact(
            root, summary.get("result_file", summary.get("audit_file")))),
        source_reproduction=source_audit.get("reproduction"),
        oracle_warning=("Both area-grid and Q-roundtrip labels use Val GT after "
                        "frozen predictions and nearest grid expansion for the probe. "
                        "They are neither deployable predictions nor the model's "
                        "bilinear probability output."),
        component_warning=("Four-connected class raster components are size proxies, "
                           "not building instance IDs; adjacent buildings may merge."),
        arms={},
    )
    for arm in arms:
        if not evaluated[arm]:
            continue
        report["arms"][arm] = dict(
            image_count=evaluated[arm],
            confusion={key: cm.tolist() for key, cm in all_cm[arm].items()},
            raster_component_proxy=components[arm],
            region_mixing={**mixing[arm], "mean_foreground_class_purity": (
                mixing[arm].get("foreground_purity_weighted_sum", 0.0) /
                mixing[arm].get("high_foreground_mass", 1.0)
                if mixing[arm].get("high_foreground_mass", 0.0) > 0 else None)},
            raster_component_building_probability=(component_b[arm] if available_b[arm]
                                                   else {"status": "unavailable_in_saved_artifact"}),
            building_probability=(building[arm] if available_b[arm]
                                  else {"status": "unavailable_in_saved_artifact"}),
            prediction_confusion_by_building_probability={
                name: [dict(bin=f"[{lower},{upper})", confusion=cm.tolist(),
                            gt_intact_pixels=int(cm[1].sum()),
                            intact_to_damaged=int(cm[1, 2]),
                            gt_damaged_pixels=int(cm[2].sum()),
                            damaged_correct=int(cm[2, 2]))
                       for (lower, upper), cm in zip(BUILDING_PROBABILITY_BINS, values)]
                for name, values in prediction_by_b[arm].items()
            },
        )
    report["per_event_confusion"] = {
        event: {arm: {key: cm.tolist() for key, cm in data.items()}
                for arm, data in arms_data.items()}
        for event, arms_data in events.items()
    }
    report["per_event_building_probability"] = building_by_event
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                  indent=2, allow_nan=False), encoding="utf-8")
    with (output / "images.jsonl").open("x", encoding="utf-8") as stream:
        for row in detail:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samepass", required=True,
                   help="Completed Val same-pass output with per-image SHA256 journal")
    p.add_argument("--output", required=True, help="Fresh diagnostic output directory")
    p.add_argument("--smoke", action="store_true",
                   help="Explicitly allow completed partial validation; never formal")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    result = audit(Path(args.samepass), Path(args.output), smoke=args.smoke)
    print(json.dumps(dict(complete=result["complete"],
                          full_validation_evidence=result["full_validation_evidence"],
                          image_count=result["image_count"],
                          arms=list(result["arms"])), ensure_ascii=False))
