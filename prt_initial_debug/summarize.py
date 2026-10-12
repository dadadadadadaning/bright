"""Print the small, decision-relevant part of a diagnostic report; stdlib only."""
import argparse
import json
from pathlib import Path


def stages(values):
    return {name: {key: report[key] for key in (
        "passed", "max_abs", "outside_tolerance", "prediction_differences") if key in report}
        for name, report in (values or {}).items()}


def summary(report):
    result = {key: report[key] for key in (
        "complete", "diagnostic_only", "full_validation_evidence", "training",
        "test_accessed", "error", "selected_images", "image_count",
        "immutable", "immutable_not_checked_due_to_error", "old_evidence_unchanged",
        "elapsed_seconds") if key in report}
    images = []
    for item in report.get("images", []):
        selected = {key: item[key] for key in (
            "image", "current_failure_reproduced", "initial_head_checks",
            "native_vs_feature_branch_logits", "actual_original_vs_candidate_logits",
            "original_mean_vs_recomputed", "original_mass_vs_recomputed",
            "snapshot_unchanged") if key in item}
        for arm in ("p", "r"):
            evidence = item.get(arm + "_fixed_input", {})
            compact = {key: evidence[key] for key in (
                "backend_restored", "parameters_unchanged", "input_unchanged") if key in evidence}
            compact["captured_vs_fixed"] = stages(evidence.get("captured_vs_fixed"))
            modes = {}
            for name, mode in evidence.get("modes", {}).items():
                comparisons = mode.get("comparisons", {})
                modes[name] = {
                    "backend": mode.get("backend"),
                    "copy_vs_parent": stages(comparisons.get("clone_1_vs_original_1")),
                    "parent_repeat": stages(comparisons.get("original_2_vs_original_1")),
                    "parent_vs_legacy": stages(mode.get("original_1_vs_legacy_original_1")),
                }
            compact["modes"] = modes
            selected[arm] = compact
        images.append(selected)
    result["images"] = images
    result["interpretation"] = (
        "诊断完成不代表故障修复；单图未复现也不能替代完整初始化 Val450。"
        "精度探针只作用于固定分类输入，未改变正式 A1/E0。")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    print(json.dumps(summary(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
