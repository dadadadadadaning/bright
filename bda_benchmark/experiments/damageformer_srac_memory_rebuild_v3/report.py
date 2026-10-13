"""Read V3 retrieval evidence on CPU; never run models, train or access Test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .diagnose import FORMAT


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def retrieval_means(row):
    """Ratios from whole-split additive counts, never a mean of image means."""
    total, supported = row.get("query_count", 0), row.get("supported_query_count", 0)
    return dict(query_count=total, supported_query_count=supported,
        support_coverage=_ratio(supported, total),
        soft_compatibility_all_queries=_ratio(row.get("soft_compatibility_sum", 0), total),
        soft_compatibility_supported_queries=_ratio(row.get("soft_compatibility_sum", 0), supported),
        best_neighbor_soft_compatibility_all_queries=_ratio(row.get("best_neighbor_soft_compatibility_sum", 0), total),
        dominant_agreement_all_queries=_ratio(row.get("dominant_label_agreement_count", 0), total),
        dominant_class_coverage=_ratio(row.get("dominant_class_covered_count", 0), total),
        weighted_support_purity_supported_queries=_ratio((row.get("quality_sum") or [0] * 6)[4], supported),
        class_soft_compatibility=row.get("class_soft_compatibility_sum", []),
        class_covered_query_mass=row.get("covered_query_class_mass", []))


def overlap_means(row):
    total, supported = row.get("query_count", 0), row.get("both_supported_query_count", 0)
    return dict(query_count=total, both_supported_query_count=supported,
        mean_jaccard_all_queries=_ratio(row.get("neighbor_jaccard_sum", 0), total),
        mean_jaccard_both_supported_queries=_ratio(row.get("neighbor_jaccard_sum", 0), supported),
        top1_agreement_all_queries=_ratio(row.get("top1_agreement_count", 0), total))


def weighted_means(row):
    return dict(query_count=row.get("queries", 0),
        supported_query_count=row.get("supported_queries", 0),
        supported_query_fraction=_ratio(row.get("supported_queries", 0), row.get("queries", 0)),
        mean_soft_compatibility_supported_queries=_ratio(row.get("soft_compatibility_sum", 0), row.get("supported_queries", 0)),
        mass_weighted_soft_compatibility=_ratio(row.get("mass_weighted_compatibility_sum", 0), row.get("supported_valid_mass", 0)),
        supported_mass_fraction=_ratio(row.get("supported_valid_mass", 0), row.get("valid_mass", 0)))


def read_report(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("format") != FORMAT:
        raise ValueError("Expected a V3 paired Memory diagnostic summary")
    result_path = root / "result.json"
    if not result_path.is_file():
        return summary, None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != FORMAT:
        raise ValueError("Expected a V3 paired Memory diagnostic result")
    for key in ("image_count", "complete", "full_validation_evidence", "execution_complete"):
        if result.get(key) != summary.get(key):
            raise ValueError("Diagnostic result and summary disagree: " + key)
    if (result.get("test_accessed") is not False or result.get("training") is not False
            or result.get("historical_reproduction_claim") is not False
            or result.get("labelvote_is_untrained_diagnostic") is not True):
        raise ValueError("Unexpected experimental claim/scope in diagnostic report")
    if result.get("full_validation_evidence") is True:
        from .diagnose import formal_gate
        if not formal_gate(result.get("image_count"), 450, result.get("validation"),
                           smoke=result.get("smoke"), bank_smoke=result.get("paired_bank_smoke")):
            raise ValueError("Full-validation claim lacks required V3 evidence gates")
    return summary, result


def render(root):
    summary, result = read_report(root)
    lines = ["V3 新建 FP32 配对库：检索与标签证据诊断", json.dumps(summary, ensure_ascii=False, indent=2)]
    if result is None:
        lines.append("尚无完成的 result.json；请检查保存的 traceback。当前不能给出完整结果结论。")
        return "\n".join(lines)
    lines.append("\n当前为未训练的 label vote；分数不能解释为 Reader 收益，也不代表历史旧库复现。")
    lines.append("Foreground mIoU = classes 1/2/3 的 IoU 均值，按整个 split 的 confusion 汇总计算。")
    lines.append("\narm\tforeground_mIoU\tclass_IoUs")
    for arm, row in result.get("metrics", {}).items():
        ious = row.get("iou", row.get("IoU", row.get("per_class_iou", [])))
        lines.append(f"{arm}\t{row.get('valid_miou')}\t{json.dumps(ious, ensure_ascii=False)}")
    lines.append("\nRetrieval means (all sums/counts aggregated over the split):")
    lines.append(json.dumps({arm: retrieval_means(row) for arm, row in result.get("posthoc_retrieval", {}).items()}, ensure_ascii=False, indent=2))
    lines.append("\nFixed Train support inventory (not full-Train candidate coverage):")
    inventory = result.get("bank_label_inventory", {})
    lines.append(json.dumps(dict(class_names=inventory.get("class_names"),
        eligible=inventory.get("eligible")), ensure_ascii=False, indent=2))
    lines.append("\nT128/F256 neighbor overlap:")
    lines.append(json.dumps({k: overlap_means(row) for k, row in result.get("paired_neighbor_overlap", {}).items()}, ensure_ascii=False, indent=2))
    lines.append("\nPost-hoc foreground purity/background dilution partitions:")
    lines.append(json.dumps({arm: {part: weighted_means(row) for part, row in parts.items()}
        for arm, parts in result.get("posthoc_weighted_label_evidence", {}).items()}, ensure_ascii=False, indent=2))
    for key in ("query_eligibility", "validation", "rebuild_protocol_sha256", "paired_bank_sha256"):
        lines.extend(("\n" + key, json.dumps(result.get(key), ensure_ascii=False, indent=2)))
    if result.get("full_validation_evidence") is not True:
        lines.append("\n这是 smoke 或未通过 formal gate 的结果，不能当作完整 Val450 证据。")
    else:
        lines.append("\n完整 Val450 新库配对诊断通过。下一步依据检索与标签证据决定 Reader 实验；本程序没有训练 Reader。")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    cli = p.parse_args(argv)
    print(render(cli.input))


if __name__ == "__main__":
    main()
