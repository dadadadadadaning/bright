"""Summarize a completed Q x Memory audit without rerunning or changing a model.

The hard-owner Q bins are observational cohorts, not building instances. In
particular, their association with Memory errors does not identify the cause.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FORMAT = "SRAC_Q_MEMORY_SAMEPASS_V1"
ARMS = ("j2", "j3")
FOCUS_KEYS = (
    "valid_pixels", "damaged_j1_wrong", "damaged_rescue",
    "damaged_j1_correct", "damaged_harm", "intact_j1_correct",
    "intact_correct_to_damaged", "intact_gt_old_not_d_to_new_d",
    "damaged_false_positive_old", "damaged_false_positive_new",
    "damaged_false_positive_delta",
)
IMMUTABLE_KEYS = (
    "files", "inherited_code", "new_code", "frozen_models", "runtime",
    "protocol", "own_source",
)
STRATA = (
    ("all", "all"),
    ("foreground_fraction", "fg_lt020"),
    ("foreground_fraction", "fg_020_080"),
    ("foreground_fraction", "fg_ge080"),
    ("joint_foreground_and_purity", "fg_ge080/class_lt080"),
    ("joint_foreground_and_purity", "fg_ge080/class_080_095"),
    ("joint_foreground_and_purity", "fg_ge080/class_ge095"),
)


def _required_bool(value, name):
    if value is not True:
        raise ValueError(f"Formal evidence gate failed: {name} must be true")


def check_formal(summary, audit):
    """Fail closed on any partial, failed, or mismatched run."""
    for name, data in (("summary", summary), ("audit", audit)):
        if data.get("format") != FORMAT:
            raise ValueError(f"{name} has unexpected format")
        for key in ("complete", "full_validation_evidence"):
            _required_bool(data.get(key), f"{name}.{key}")
        if type(data.get("image_count")) is not int or data["image_count"] != 450:
            raise ValueError(f"{name}.image_count must be 450")
        if data.get("test_accessed") is not False or data.get("training") is not False:
            raise ValueError(f"{name} is not a frozen Val-only audit")
    if audit.get("expected_image_count") != 450:
        raise ValueError("audit.expected_image_count must be 450")
    if audit.get("numerical_profile") != "legacy" or audit.get("pool_mode") != "native":
        raise ValueError("This report requires the verified legacy/native numerical protocol")
    _required_bool(summary.get("reproduction_passed"), "summary.reproduction_passed")
    _required_bool(audit.get("reproduction", {}).get("passed"), "audit.reproduction.passed")
    for name, data in (("summary", summary), ("audit", audit)):
        imm = data.get("immutable")
        if not isinstance(imm, dict) or not all(imm.get(k) is True for k in IMMUTABLE_KEYS):
            raise ValueError(f"{name}.immutable has a missing or failed check")
        if not all(v is True for v in imm.values()):
            raise ValueError(f"{name}.immutable contains a failed check")
    if summary["immutable"] != audit["immutable"]:
        raise ValueError("summary and audit integrity checks disagree")
    if audit.get("strata", {}).get("images") != 450:
        raise ValueError("strata does not cover 450 images")
    if summary.get("audit_file") != "audit.json":
        raise ValueError("summary does not identify the expected audit file")


def _checked_count(value, name, allow_negative=False):
    if type(value) is not int or (value < 0 and not allow_negative):
        raise ValueError(f"Invalid integer count: {name}")
    return value


def _focus(row, label):
    if not isinstance(row, dict):
        raise ValueError(f"Missing focus row: {label}")
    counts = {
        key: _checked_count(row.get(key), f"{label}.{key}",
                            allow_negative=key == "damaged_false_positive_delta")
        for key in FOCUS_KEYS
    }
    if counts["damaged_false_positive_new"] - counts["damaged_false_positive_old"] != counts["damaged_false_positive_delta"]:
        raise ValueError(f"Inconsistent Damaged false-positive delta: {label}")
    if counts["damaged_rescue"] > counts["damaged_j1_wrong"] or counts["damaged_harm"] > counts["damaged_j1_correct"]:
        raise ValueError(f"Inconsistent Damaged denominator: {label}")
    if counts["intact_correct_to_damaged"] > counts["intact_j1_correct"]:
        raise ValueError(f"Inconsistent Intact denominator: {label}")
    return {
        **counts,
        "damaged_rescue_rate": counts["damaged_rescue"] / counts["damaged_j1_wrong"] if counts["damaged_j1_wrong"] else None,
        "damaged_harm_rate": counts["damaged_harm"] / counts["damaged_j1_correct"] if counts["damaged_j1_correct"] else None,
        "intact_correct_to_damaged_rate": counts["intact_correct_to_damaged"] / counts["intact_j1_correct"] if counts["intact_j1_correct"] else None,
    }


def _strata(table, context):
    out = {}
    for group, bin_name in STRATA:
        rows = table.get(group, {}).get(bin_name, {})
        out[bin_name] = {
            arm: _focus(rows.get(arm), f"{context}.{group}.{bin_name}.{arm}")
            for arm in ARMS
        }
    return out


def _exposures(source, context):
    """Copy count and mean of each soft-Q exposure with basic consistency checks."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for cohort in (
            "intact_j1_correct", "damaged_j1_wrong", "intact_correct_to_damaged",
            "damaged_rescued", "damaged_harmed",
        ):
            out[arm][cohort] = {}
            for field in ("background_foreground_mixed", "foreground_class_mixed"):
                row = source.get(arm, {}).get(cohort, {}).get(field)
                if not isinstance(row, dict):
                    raise ValueError(f"Missing soft exposure: {context}.{arm}.{cohort}.{field}")
                count = _checked_count(row.get("count"), f"{context}.{arm}.{cohort}.{field}.count")
                amount, mean = row.get("sum"), row.get("mean")
                if (not isinstance(amount, (float, int)) or not math.isfinite(amount)
                        or amount < -1e-4 or amount > count + 1e-4):
                    raise ValueError(f"Invalid soft exposure sum: {context}.{arm}.{cohort}.{field}")
                if count:
                    if (not isinstance(mean, (float, int)) or not math.isfinite(mean)
                            or abs(float(mean) - amount / count) > 1e-5):
                        raise ValueError(f"Invalid soft exposure mean: {context}.{arm}.{cohort}.{field}")
                elif mean is not None or abs(amount) > 1e-5:
                    raise ValueError(f"Zero-count soft exposure must have zero sum and no mean: {context}.{arm}.{cohort}.{field}")
                out[arm][cohort][field] = {"count": count, "mean": mean}
    return out


def build_report(summary, audit):
    check_formal(summary, audit)
    source_strata = audit["strata"]
    overall = _strata(source_strata["overall"], "overall")
    by_event = {
        event: _strata(table, f"event.{event}")
        for event, table in source_strata["by_event"].items()
    }
    if not by_event:
        raise ValueError("No per-event strata were saved")
    for arm in ARMS:
        full_fg = source_strata["overall"]["foreground_fraction"]
        extra = _focus(full_fg["unlabelled"][arm], f"overall.unlabelled.{arm}")
        for key in FOCUS_KEYS:
            if overall["all"][arm][key] != sum(overall[n][arm][key] for n in
                                                 ("fg_lt020", "fg_020_080", "fg_ge080")) + extra[key]:
                raise ValueError(f"Foreground partition including unlabelled disagrees: {arm}.{key}")
            if overall["all"][arm][key] != sum(item["all"][arm][key] for item in by_event.values()):
                raise ValueError(f"Per-event totals disagree: {arm}.{key}")
    soft = audit["soft_q_mixed_exposure"]
    overall_exposure = _exposures(soft["overall"], "overall")
    event_exposure = {event: _exposures(table, f"event.{event}")
                      for event, table in soft["by_event"].items()}
    if set(by_event) != set(event_exposure):
        raise ValueError("Strata and exposure event lists disagree")
    for arm in ARMS:
        cohort_denominators = {
            "intact_j1_correct": overall["all"][arm]["intact_j1_correct"],
            "damaged_j1_wrong": overall["all"][arm]["damaged_j1_wrong"],
            "intact_correct_to_damaged": overall["all"][arm]["intact_correct_to_damaged"],
            "damaged_rescued": overall["all"][arm]["damaged_rescue"],
            "damaged_harmed": overall["all"][arm]["damaged_harm"],
        }
        for cohort, expected in cohort_denominators.items():
            for field in overall_exposure[arm][cohort]:
                if overall_exposure[arm][cohort][field]["count"] != expected:
                    raise ValueError(f"Soft exposure cohort count disagrees: {arm}.{cohort}.{field}")
    return {
        "format": "SRAC_Q_MEMORY_REPORT_V1",
        "source_format": FORMAT,
        "image_count": 450,
        "formal_evidence": True,
        "source_reproduction_passed": True,
        "region_inventory": audit["region_inventory"],
        "metrics": {arm: audit["metrics"][arm] for arm in ("a1_pixel", "j1", "j2", "j3")},
        "hard_owner_strata": overall,
        "hard_owner_by_event": by_event,
        "soft_q_exposure": overall_exposure,
        "soft_q_exposure_by_event": event_exposure,
        "interpretation": (
            "Q hard-owner bins and soft exposures are observational pixel/region cohorts, "
            "not building instances or causal proof. Compare the rescue and harm rates "
            "using their fixed J1 denominators; final IoU remains decisive."
        ),
    }


def human_report(report):
    lines = ["Val450 同次前向诊断：完整性和原 1e-5 复现检查通过。",
             "以下按 Q 的最大权重区域给像素分组；占比来自验证标注，纯度是前景三类的比例，均不代表建筑实例。",
             "软 Q 的“前景类别混合”专指前景占比≥0.8、前景三类纯度<0.8 的区域。"]
    lines.append("同次 Val 前景 mIoU：" + "，".join(
        f"{arm} {report['metrics'][arm]['valid_miou'] * 100:.4f}%"
        for arm in ("a1_pixel", "j1", "j2", "j3")
    ))
    names = {
        "all": "所有有效像素",
        "fg_ge080/class_lt080": "标注前景占比≥0.8、前景类别混合(<0.8)",
        "fg_ge080/class_ge095": "标注前景占比≥0.8、前景类别较纯(≥0.95)",
    }
    for arm in ARMS:
        lines.append(f"{arm.upper()} 相对固定 J1：")
        for name, label in names.items():
            row = report["hard_owner_strata"][name][arm]
            rate_d = row["damaged_rescue_rate"]
            rate_h = row["damaged_harm_rate"]
            rate_i = row["intact_correct_to_damaged_rate"]
            lines.append(
                f"  {label}: Damaged 救回 {row['damaged_rescue']}/{row['damaged_j1_wrong']} "
                + (f"({rate_d * 100:.3f}%)" if rate_d is not None else "(无分母)")
            )
            lines.append(
                f"    原本正确的 Intact→Damaged: {row['intact_correct_to_damaged']}/{row['intact_j1_correct']} "
                + (f"({rate_i * 100:.3f}%)" if rate_i is not None else "(无分母)")
                + f"；原本正确的 Damaged 被误伤 {row['damaged_harm']}/{row['damaged_j1_correct']} "
                + (f"({rate_h * 100:.3f}%)" if rate_h is not None else "(无分母)")
                + f"；Damaged 假阳性净变化 {row['damaged_false_positive_delta']:+d}"
            )
        e = report["soft_q_exposure"][arm]
        for field, name in (("foreground_class_mixed", "前景类别混合"),
                            ("background_foreground_mixed", "建筑/背景混合")):
            a = e["intact_correct_to_damaged"][field]["mean"]
            b = e["intact_j1_correct"][field]["mean"]
            if a is not None and b is not None:
                lines.append(f"  软 Q 暴露于{name}区域: Intact 误改平均 {a:.4f}，原本正确 Intact 平均 {b:.4f}")
        events = sorted(report["hard_owner_by_event"].items(),
                        key=lambda item: (
                            item[1]["all"][arm]["intact_correct_to_damaged_rate"]
                            if item[1]["all"][arm]["intact_correct_to_damaged_rate"] is not None
                            else -1
                        ),
                        reverse=True)
        lines.append("  逐事件 Intact 误改率最高的五项（仅描述，不据此挑模型）：")
        for event, rows in events[:5]:
            row = rows["all"][arm]
            rate = row["intact_correct_to_damaged_rate"]
            lines.append(f"    {event}: {row['intact_correct_to_damaged']}/{row['intact_j1_correct']} "
                         + (f"({rate * 100:.3f}%)" if rate is not None else "(无分母)"))
    lines.append("同一 Q 分层中的相关性不能证明 Q 混合造成了误改；需固定模型、库、检索及门控做干预对照。")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Completed q_memory_samepass output directory")
    p.add_argument("--json-output", type=Path, help="New path for machine-readable report")
    args = p.parse_args(argv)
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((args.run / "audit.json").read_text(encoding="utf-8"))
    report = build_report(summary, audit)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(human_report(report))
    if not args.json_output:
        print("\n机器可读 JSON：", json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
