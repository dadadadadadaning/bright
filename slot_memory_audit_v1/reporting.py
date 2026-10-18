"""Human-readable and tabular views of saved read-only audit measurements.

This module interprets neither statistical significance nor root cause. Undefined
statistics remain None/empty cells and always retain their own denominators.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Mapping


CLASSES = ("背景", "完好", "受损", "毁坏")
STRATA = {
    "all_labeled": "全部有效标签",
    "gt_building": "真值建筑",
    "gt_building_boundary": "真值建筑/背景边缘",
    "gt_building_interior": "真值建筑内部",
    "low_B_true_building": "低B真值建筑",
    "gt_intact": "真值完好",
    "gt_damaged": "真值受损",
    "gt_destroyed": "真值毁坏",
    "role1_gt_intact": "偏建筑槽中的完好真值质量",
    "role1_gt_damaged": "偏建筑槽中的受损真值质量",
}
SLOT_FIELDS = (
    ("feature_cosine", "两槽特征余弦"),
    ("feature_relative_l2", "两槽相对L2"),
    ("label_total_variation", "两槽标签TV距离"),
    ("label_js_bits", "两槽标签JS(bits)"),
    ("building_probability_variance", "父区域内B方差"),
    ("both_slots_active", "两槽均有效比例"),
    ("slot0_fraction", "偏背景槽权重比例"),
    ("slot1_fraction", "偏建筑槽权重比例"),
    ("slot0_ess", "偏背景槽ESS"),
    ("slot1_ess", "偏建筑槽ESS"),
    ("slot0_ess_lt_2", "偏背景槽ESS<2比例"),
    ("slot1_ess_lt_2", "偏建筑槽ESS<2比例"),
)
CANDIDATE_FIELDS = (
    ("has_candidate_labels", "有标签证据比例"),
    ("has_cosine_evidence", "有固定余弦证据比例"),
    ("valid_parent_count", "有效父案例数"),
    ("labelled_child_slot_count", "有标签子槽数"),
    ("candidate_mean_damaged", "候选平均受损软标签分量"),
    ("candidate_max_damaged", "候选最大受损软标签分量"),
    ("damaged50_parent_count", "包含受损质量≥0.5子槽的父案例数"),
    ("intact50_parent_count", "包含完好质量≥0.5子槽的父案例数"),
    ("unique_source_images", "不同来源图像数"),
    ("unique_source_events", "不同来源事件数"),
    ("damaged50_unique_images", "受损质量≥0.5案例来源图像数"),
    ("damaged50_unique_events", "受损质量≥0.5案例来源事件数"),
    ("duplicate_image_fraction", "同来源图像重复比例"),
    ("similar_intact50_fraction", "高相似完好候选比例"),
    ("similar_damaged50_fraction", "高相似受损候选比例"),
    ("cosine_vote_damaged", "固定余弦投票受损质量"),
    ("cosine_vote_foreground_mass", "固定余弦投票前景质量"),
    ("posthoc_vote_tv_compatibility", "事后标签TV相容度"),
    ("posthoc_foreground_conditional_vote_tv", "事后前景条件TV相容度"),
)


def _number(value, digits=6, *, percentage=False):
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite report value; accumulator must emit None")
    return f"{number * (100 if percentage else 1):.{digits}f}"


def _index(values, index):
    return values[index] if isinstance(values, (list, tuple)) and len(values) > index else None


def _metric_line(name, field):
    if not isinstance(field, Mapping):
        return f"  {name}: NA（尚无统计）"
    return (f"  {name}: mean={_number(field.get('mean'))}; "
            f"n={field.get('n', 'NA')}; weight_sum={_number(field.get('weight_sum'))}")


def _overall_rows(pixels):
    return pixels.get("all_labeled", {}).get("metrics", {})


def _paired_line(name, row):
    rescue, harm = row.get("rescue_by_gt_class"), row.get("harm_by_gt_class")
    if not isinstance(rescue, (list, tuple)) or not isinstance(harm, (list, tuple)):
        return f"{name}: 尚无配对计数"
    return (f"{name}: 救回={list(rescue)}; 误伤={list(harm)}; "
            f"受损救回={_index(rescue, 2)}; 受损误伤={_index(harm, 2)}; "
            f"原正确完好→受损={row.get('intact_correct_to_damaged', 'NA')}; "
            f"受损由背景救回={row.get('damaged_background_to_correct', 'NA')}; "
            f"原正确受损→背景={row.get('damaged_correct_to_background', 'NA')}")


def render_report(result: Mapping) -> str:
    """Render known fields without inventing completion or positive conclusions."""
    # NaN/Infinity in any nested saved statistic is an error, never printable data.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    lines = ["soft双槽与Memory只读诊断", ""]
    state_keys = ("complete", "execution_complete", "full_validation_evidence", "smoke",
                  "checkpoint_step", "image_count", "training", "test_accessed",
                  "final_integrity_pending", "error")
    for key in state_keys:
        if key in result:
            lines.append(f"{key}: {json.dumps(result[key], ensure_ascii=False)}")
    lines.extend([
        "",
        "这是已有权重的诊断，没有新训练；单个检查点的差异不等于独立种子稳定收益。",
        "百分数仅用于预测IoU/精确率/召回率；区域统计保持原单位。NA表示未定义或尚无统计，不等于0。",
        "像素类别顺序：背景/完好/受损/毁坏。前景mIoU固定平均类1/2/3；缺失并集按原实现计0。",
        "",
        "完整已处理图像的预测汇总（请结合image_count及最终校验状态）：",
        "path | fg mIoU % | IoU背景/完好/受损/毁坏 % | Damaged precision % | Damaged recall % | valid pixels",
    ])
    pixels = result.get("pixels") or {}
    overall = _overall_rows(pixels)
    for path in ("grid_probq", "b", "d"):
        metric = overall.get(path) or {}
        iou = "/".join(_number(_index(metric.get("iou"), i), 4, percentage=True) for i in range(4))
        lines.append(" | ".join((path, _number(metric.get("foreground_miou"), 4, percentage=True),
                                  iou, _number(_index(metric.get("precision"), 2), 4, percentage=True),
                                  _number(_index(metric.get("recall"), 2), 4, percentage=True),
                                  str(metric.get("valid_pixels", "NA")))))
    for first, second in (("b", "grid_probq"), ("d", "grid_probq"), ("d", "b")):
        a, b = overall.get(first, {}).get("foreground_miou"), overall.get(second, {}).get("foreground_miou")
        if a is not None and b is not None:
            lines.append(f"{first} − {second}: {_number(a-b, 4, percentage=True)} 个百分点")
    lines.extend(["", "逐像素配对：救回=原错现对，误伤=原对现错；单位是像素，不是建筑。",
                  "真值类别/边缘分层只展示配对计数；其限定区域混淆会截断部分假阳性，不能当作全图mIoU比较。"])
    paired = pixels.get("all_labeled", {}).get("paired", {})
    for pair in ("b_vs_grid", "d_vs_grid", "d_vs_b"):
        lines.append(_paired_line(pair, paired.get(pair) or {}))
    for stratum in STRATA:
        if stratum == "all_labeled" or stratum not in pixels:
            continue
        row = pixels[stratum]
        n = row.get("metrics", {}).get("grid_probq", {}).get("valid_pixels", "NA")
        lines.append(f"  {STRATA[stratum]}（{n}像素） " + _paired_line("d_vs_b", row.get("paired", {}).get("d_vs_b") or {}))

    regions = result.get("regions") or {}
    strata = regions.get("strata") or {}
    lines.extend([
        "", "双槽分层统计：",
        "槽0=Q×(1−B)偏背景；槽1=Q×B偏建筑。各槽均可包含任意损毁类别。",
        "每项分别报告有效区域数n及加权分母weight_sum；分母只包括该项有定义的观测。",
        "加权单位为真值在特征网格的质量；ESS单位为特征网格位置，不是原图像素/建筑数。",
        "分层相互重叠；GT边缘为建筑/背景边界，不是逐栋实例边界。真值仅用于事后统计。",
    ])
    for stratum, fields in strata.items():
        lines.append(f"[{stratum}] {STRATA.get(stratum, stratum)}")
        for key, name in SLOT_FIELDS:
            if key in fields:
                lines.append(_metric_line(name, fields[key]))

    lines.extend(["", "Train案例库存量（已保存的库，不代表完整Train候选覆盖）：",
                  json.dumps(result.get("inventory") or {}, ensure_ascii=False, indent=2, allow_nan=False)])
    lines.extend([
        "", "候选证据重点：全部、真值建筑、偏建筑槽内的受损/完好真值质量；后两项按槽内GT质量加权，不是另行筛选纯类别查询。",
        "Memory使用Train软标签，图内参考使用预测概率；来源及标签质量同时变化，不能全部归因于检索。",
        "固定余弦/温度0.1投票不是训练过的Reader注意力，也不是像素预测。标签质量(mass)指概率分量，不是准确率。",
        "图内GT oracle只评价参考标签理想质量，不进入模型。去掉自身的图内探针不回填，因此不是等预算训练对照。",
    ])
    for stratum in ("all_labeled", "gt_building", "role1_gt_damaged", "role1_gt_intact"):
        fields = strata.get(stratum)
        if not fields:
            continue
        lines.append(f"[{stratum}]")
        for prefix, title in (("memory", "历史案例"), ("local_pred", "图内预测标签"),
                              ("local_other_pred", "去自身图内预测标签")):
            lines.append(" " + title)
            for suffix, name in CANDIDATE_FIELDS:
                key = prefix + "_" + suffix
                if key in fields:
                    lines.append(_metric_line(name, fields[key]))
        for prefix in ("local_gt_oracle", "local_other_gt_oracle"):
            key = prefix + "_posthoc_vote_tv_compatibility"
            if key in fields:
                lines.append(_metric_line(prefix + " 事后TV相容度", fields[key]))

    lines.extend(["", "实际已训练Reader（偏建筑查询槽，跨注意力头平均）：",
                  "注意力加权标签质量描述读到什么，不等于最终类别概率，也不证明某案例造成预测变化。"])
    for stratum in ("all_labeled", "gt_building", "gt_damaged", "low_B_true_building"):
        fields = strata.get(stratum) or {}
        found = [(name, field) for name, field in fields.items() if name.startswith("reader_")]
        if found:
            lines.append(f"[{stratum}]")
            for name, field in found:
                if name.endswith(("damaged_label_mass", "intact_label_mass", "has_support", "background_role_attention",
                                  "tiny_fraction_attention", "effective_slots", "posthoc_label_TV")):
                    lines.append(_metric_line(name, field))

    lines.extend(["", "逐事件（按事件累计混淆；不能平均此表代替总体混淆结果）：",
                  "event | grid fg mIoU % | B fg mIoU % | D fg mIoU % | D−B pp"])
    for event, record in sorted((result.get("per_event") or {}).items()):
        event_metrics = _overall_rows(record.get("pixels") or {})
        values = [event_metrics.get(p, {}).get("foreground_miou") for p in ("grid_probq", "b", "d")]
        delta = values[2] - values[1] if values[1] is not None and values[2] is not None else None
        lines.append(" | ".join([str(event)] + [_number(v, 4, percentage=True) for v in values] + [_number(delta, 4, percentage=True)]))
    lines.extend(["", "原报告重放校验：完整Val450预设容差为0.001个百分点；小样本不宣称完成该重放。",
                  json.dumps(result.get("replay") or {}, ensure_ascii=False, indent=2, allow_nan=False),
                  "", "本报告列出测量值，不自动判定唯一根因或模块是否有效。最终以完整Val450、来源终检和匹配重复实验判断。"])
    return "\n".join(lines) + "\n"


def write_tables(output, result: Mapping) -> None:
    """Write derived summaries only; source files and model artifacts stay intact."""
    report = render_report(result)
    events = json.dumps(result.get("per_event") or {}, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("stratum", "metric", "n", "weight_sum", "weighted_sum", "mean"))
    strata = (result.get("regions") or {}).get("strata") or {}
    for stratum, fields in sorted(strata.items()):
        for name, row in sorted(fields.items()):
            writer.writerow((stratum, name, row.get("n"), row.get("weight_sum"),
                             row.get("weighted_sum"), row.get("mean")))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "REPORT.txt").write_text(report, encoding="utf-8")
    with (root / "regions_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(stream.getvalue())
    (root / "per_event.json").write_text(events, encoding="utf-8")
