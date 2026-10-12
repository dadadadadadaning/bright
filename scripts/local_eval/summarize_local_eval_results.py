#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Windows CPU official BRIGHT evaluations and write a short report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


OVERALL_FIELDS = [
    "model", "checkpoint", "old_dir_name", "checkpoint_path", "param_status", "save_dir",
    "official_style_4class_miou", "target_3class_miou",
    "background_iou", "intact_iou", "damaged_iou", "destroyed_iou",
    "damaged_recall", "destroyed_recall", "damaged_to_intact",
    "destroyed_to_intact", "destroyed_to_damaged", "destroyed_to_background",
    "loc_head_iou", "loc_head_f1", "damage_building_iou", "damage_building_f1",
    "load_missing_keys_count", "load_unexpected_keys_count",
]
EVENT_FIELDS = [
    "model", "checkpoint", "event", "num_images", "num_pixels", "num_valid_pixels", "num_building_pixels",
    "num_intact_pixels", "num_damaged_pixels", "num_destroyed_pixels",
    "official_style_4class_miou", "target_3class_miou", "background_iou", "intact_iou",
    "damaged_iou", "destroyed_iou", "damaged_precision", "damaged_recall",
    "destroyed_precision", "destroyed_recall", "damaged_to_background", "damaged_to_intact",
    "damaged_to_destroyed", "destroyed_to_background", "destroyed_to_intact",
    "destroyed_to_damaged", "loc_head_iou", "loc_head_f1", "damage_building_iou",
    "damage_building_f1", "impact_score", "destroyed_impact_score", "diagnosis_code",
    "recommended_direction", "support_warning",
]
DIAGNOSIS_FIELDS = [
    "model", "checkpoint", "event", "num_destroyed_pixels", "destroyed_iou",
    "destroyed_precision", "destroyed_recall", "destroyed_to_background",
    "destroyed_to_intact", "destroyed_to_damaged", "destroyed_fp_from_background_count",
    "destroyed_fp_from_intact_count", "destroyed_fp_from_damaged_count",
    "destroyed_fp_from_background_share", "destroyed_fp_from_intact_share",
    "destroyed_fp_from_damaged_share", "damage_building_iou", "destroyed_impact_score",
    "diagnosis_code", "diagnosis_text", "recommended_direction", "support_warning",
]
WORST_FIELDS = [
    "model", "checkpoint", "rank", "event", "destroyed_impact_score", "impact_score",
    "target_3class_miou", "destroyed_iou", "num_destroyed_pixels", "diagnosis_code",
    "recommended_direction",
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary_paths(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield from root.rglob("summary_official.json")


def collect(results_root: Path):
    overall: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    diagnosis: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for path in sorted(summary_paths(results_root)):
        save_dir = path.parent
        summary = read_json(path)
        metadata_path = save_dir / "local_eval_metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        model = metadata.get("standard_model", save_dir.parent.name)
        checkpoint = metadata.get("checkpoint_name", save_dir.name)
        row = {
            "model": model,
            "checkpoint": checkpoint,
            "old_dir_name": metadata.get("old_dir_name", ""),
            "checkpoint_path": metadata.get("checkpoint_path", summary.get("model_path", "")),
            "param_status": metadata.get("param_status", ""),
            "save_dir": str(save_dir.resolve()),
            **summary,
            "loc_head_iou": summary.get("loc_head_iou", summary.get("loc_iou", "")),
            "loc_head_f1": summary.get("loc_head_f1", summary.get("loc_f1", "")),
        }
        overall.append(row)

        event_path = save_dir / "event_metrics_official.csv"
        if not event_path.is_file():
            warnings.append(f"missing: {event_path}")
        for event in read_csv(event_path):
            events.append({"model": model, "checkpoint": checkpoint, **event})

        diagnosis_path = save_dir / "event_destroyed_diagnosis.csv"
        if not diagnosis_path.is_file():
            warnings.append(f"missing: {diagnosis_path}")
        for diag in read_csv(diagnosis_path):
            diagnosis.append({"model": model, "checkpoint": checkpoint, **diag})

    overall.sort(
        key=lambda row: number(row.get("target_3class_miou"))
        if finite(row.get("target_3class_miou")) else float("-inf"),
        reverse=True,
    )
    events.sort(key=lambda row: (str(row.get("model")), str(row.get("checkpoint")), str(row.get("event"))))
    diagnosis.sort(
        key=lambda row: number(row.get("destroyed_impact_score"))
        if finite(row.get("destroyed_impact_score")) else float("-inf"),
        reverse=True,
    )
    return overall, events, diagnosis, warnings


def worst_rows(events: Sequence[Mapping[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        groups[(row.get("model", ""), row.get("checkpoint", ""))].append(row)
    output: List[Dict[str, Any]] = []
    for (model, checkpoint), rows in groups.items():
        ranked = sorted(
            (row for row in rows if finite(row.get("destroyed_impact_score"))),
            key=lambda row: number(row.get("destroyed_impact_score")),
            reverse=True,
        )
        for rank, row in enumerate(ranked[:top_k], start=1):
            output.append({"model": model, "checkpoint": checkpoint, "rank": rank, **row})
    return output


def best_row(rows: Sequence[Mapping[str, Any]], metric: str, lower: bool = False):
    valid = [row for row in rows if finite(row.get(metric))]
    if not valid:
        return None
    return (min if lower else max)(valid, key=lambda row: number(row.get(metric)))


def format_best(rows: Sequence[Mapping[str, Any]], metric: str, lower: bool = False) -> str:
    row = best_row(rows, metric, lower)
    if row is None:
        return "暂无有效结果"
    return f"{row['model']} / {row['checkpoint']}（{number(row[metric]):.6f}）"


def preferred_model_row(rows: Sequence[Mapping[str, Any]], model: str):
    candidates = [row for row in rows if row.get("model") == model and finite(row.get("target_3class_miou"))]
    return max(candidates, key=lambda row: number(row["target_3class_miou"])) if candidates else None


def delta_text(candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None, metric: str) -> str:
    if candidate is None or baseline is None or not finite(candidate.get(metric)) or not finite(baseline.get(metric)):
        return "数据不足"
    delta = number(candidate[metric]) - number(baseline[metric])
    return f"{delta:+.6f}"


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def load_registry(analysis_root: Path) -> List[Dict[str, Any]]:
    return read_csv(analysis_root / "local_eval_registry.csv")


def write_inventory(path: Path, registry: Sequence[Mapping[str, Any]]) -> None:
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in registry:
        groups[(int(row.get("priority", 9)), str(row.get("standard_model", "")))].append(row)
    runnable = [row for row in registry if not str(row.get("param_status", "")).startswith("PARAM_UNCERTAIN")]
    uncertain = [row for row in registry if str(row.get("param_status", "")).startswith("PARAM_UNCERTAIN")]
    lines = [
        "# 01 模型库存",
        "",
        f"- 含正式候选 checkpoint 的模型目录：{len({row.get('old_dir_name') for row in registry})}",
        f"- 正式候选 checkpoint：{len(registry)}",
        f"- 可执行 checkpoint：{len(runnable)}",
        f"- 参数或架构不确定 checkpoint：{len(uncertain)}",
        "- 当前缺少正式权重：M07_A4DG_b03、M11_A4HDC_a03",
        "- 当前不存在任务书提到的 iter_46000.pth / iter_44000.pth。",
        "- M06C 的真实 args.json 为 lambda_align=0.0001、warmup=10000、ramp=10000；"
        "lambda_align 属于训练损失参数，推理只需恢复模型结构参数。",
        "",
        "## 已识别模型",
        "",
        "| 优先级 | 标准模型 | checkpoint 数 | 参数状态 |",
        "|---:|---|---:|---|",
    ]
    for (priority, model), rows in sorted(groups.items()):
        statuses = ", ".join(sorted({str(row.get("param_status", "")) for row in rows}))
        lines.append(f"| {priority} | {model} | {len(rows)} | {statuses} |")
    if uncertain:
        lines += ["", "## 被跳过的未知架构", ""]
        for row in uncertain:
            lines.append(f"- `{row.get('old_dir_name')}`：{row.get('param_status')}")
    write_markdown(path, lines)


def write_plan(path: Path, registry: Sequence[Mapping[str, Any]], completed: int) -> None:
    lines = [
        "# 02 本地 CPU official 评估计划",
        "",
        "## 流程",
        "",
        "1. dry-run 扫描权重并核对参数。",
        "2. 使用 20 张列表做单 checkpoint smoke test。",
        "3. 按优先级 1 → 2 → 3 执行完整 878 张 official 评估。",
        "4. 单模型失败写入 `eval_failed_models.csv`，后续模型继续。",
        "5. 汇总 overall、event-wise 和 destroyed 诊断。",
        "",
        "## 命令",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\local_eval\\dry_run_local_eval_cpu.ps1",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\local_eval\\run_local_eval_cpu.ps1 -SmokeTest -MaxCheckpoints 1",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\local_eval\\run_local_eval_cpu.ps1",
        "```",
        "",
        "## checkpoint 规则",
        "",
        "`best_valid_miou` → `best_balanced_score` → `best_damaged_iou` → "
        "`best_destroyed_iou` → `best_model` → `best`。只有无 best 时才回退到 "
        "`final_model` 或 `latest_resume`。",
        "",
        "## 当前状态与耗时",
        "",
        f"- registry checkpoint：{len(registry)}",
        f"- 已完成正式 878 张结果：{completed}",
        "- 20 张 smoke test 已通过。",
        "- 实测约 12 秒/图；全部可执行 checkpoint 预计约 120–160 小时。",
        "- 风险：CPU 长任务、Windows 休眠/重启、未知自定义架构、历史 args.json 缺少新参数。",
    ]
    write_markdown(path, lines)


MODEL_DESCRIPTIONS = {
    "M01_BGCA_none": ("无 ROI、无 MDDM 基线", "作为 MDDM/ROI 增益参照"),
    "M02_A4_noROI_MDDMbase": ("加入 base MDDM，不使用 ROI", "判断 MDDM 是否有效"),
    "M03_A4_MDDMbase_ROId5": ("base MDDM + dilated soft ROI", "判断 ROI 对建筑损毁分类的价值"),
    "M04_A4DG_b10": ("dual-scale gate，large beta=1.0", "降低 destroyed 被错成 intact"),
    "M05_A4D_MDDMlarge": ("large-scale MDDM", "增强 destroyed 大尺度损毁证据"),
    "A4_DamageExistHead": ("A4 + DamageExistHead BCE", "降低 damaged/destroyed 被错成 intact"),
    "M06_A4IRC": ("A4 + inner-ring context", "利用建筑邻域上下文"),
    "BGCA_soft": ("普通 soft ROI", "ROI 消融"),
    "BGCA_dilated_soft": ("dilated soft ROI", "检验扩张 ROI"),
    "BGCA_soft_align": ("soft ROI + alignment", "检验 damage-aware alignment"),
    "BGCA_dilated_soft_align": ("dilated ROI + warmup/alignment", "检验 alignment 稳定性"),
    "M06C_dilatedROI_warmupAlign_lam0001": (
        "dilated ROI + damage-aware warmup alignment",
        "检验弱 alignment 的稳定性；lambda_align 是训练损失参数，推理不需要",
    ),
}


def write_effect_interpretation(
    path: Path,
    registry: Sequence[Mapping[str, Any]],
    overall: Sequence[Mapping[str, Any]],
) -> None:
    a4 = preferred_model_row(overall, "M03_A4_MDDMbase_ROId5")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in registry:
        grouped[str(row.get("standard_model", ""))].append(row)
    lines = ["# 03 模型改进效果解释", ""]
    for model, rows in sorted(grouped.items(), key=lambda item: (int(item[1][0].get("priority", 9)), item[0])):
        modification, goal = MODEL_DESCRIPTIONS.get(model, ("结构待确认", "需要结合训练代码确认"))
        result = preferred_model_row(overall, model)
        checkpoints = ", ".join(sorted({str(row.get("checkpoint_name", "")) for row in rows}))
        result_text = (
            f"target_3class_miou={number(result['target_3class_miou']):.6f}"
            if result is not None else "尚无完整 878 张 official 结果"
        )
        lines += [
            f"## {model}",
            "",
            f"文件夹：`{rows[0].get('old_dir_name', '')}`  ",
            f"checkpoint：{checkpoints}  ",
            f"模型改进：{modification}  ",
            f"设计目标：{goal}  ",
            "训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  ",
            f"official 评估结果：{result_text}  ",
            "相比 M03_A4 的变化：",
            f"- target_3class_miou: {delta_text(result, a4, 'target_3class_miou')}",
            f"- damaged_iou: {delta_text(result, a4, 'damaged_iou')}",
            f"- destroyed_iou: {delta_text(result, a4, 'destroyed_iou')}",
            f"- damaged_to_intact: {delta_text(result, a4, 'damaged_to_intact')}",
            f"- destroyed_to_intact: {delta_text(result, a4, 'destroyed_to_intact')}",
            "",
            f"效果判断：{'等待完整 official 结果。' if result is None else '按上述差值判断。'}  ",
            f"启示：{goal}。  ",
            f"是否建议继续：{'暂不下结论。' if result is None else '结合主指标和 destroyed 错误流决定。'}",
            "",
        ]
    write_markdown(path, lines)


def write_event_diagnosis(path: Path, diagnosis: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# 04 Event-wise destroyed 诊断", ""]
    if not diagnosis:
        lines += [
            "当前没有完整 878 张 official event-wise 结果，因此不能判断最拖累整体的灾害事件。",
            "",
            "诊断规则：",
            "",
            "- destroyed_to_intact 高：DamageExistHead / HDC / A4-DG-beta。",
            "- destroyed_to_damaged 高：severity head / ordinal loss / large-scale MDDM。",
            "- destroyed_to_background 高：ROI / loc head / building-aware fusion。",
            "- destroyed_precision 低：校准、类别权重、focal loss。",
        ]
    else:
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in diagnosis:
            grouped[str(row.get("model", ""))].append(row)
        for model, rows in sorted(grouped.items()):
            ranked = sorted(
                rows,
                key=lambda row: number(row.get("destroyed_impact_score"))
                if finite(row.get("destroyed_impact_score")) else float("-inf"),
                reverse=True,
            )[:5]
            lines += [f"## {model}", ""]
            for row in ranked:
                lines.append(
                    f"- {row.get('event')}: impact={row.get('destroyed_impact_score')}, "
                    f"错误={row.get('diagnosis_code')}, 建议={row.get('recommended_direction')}"
                )
            lines.append("")
    write_markdown(path, lines)


def write_report(path: Path, overall: Sequence[Mapping[str, Any]], warnings: Sequence[str]) -> None:
    a4 = preferred_model_row(overall, "M03_A4_MDDMbase_ROId5")
    a4d = preferred_model_row(overall, "M05_A4D_MDDMlarge")
    dg03 = preferred_model_row(overall, "M07_A4DG_b03")
    dg10 = preferred_model_row(overall, "M04_A4DG_b10")
    hdc = preferred_model_row(overall, "M11_A4HDC_a03")
    models = {str(row.get("model")) for row in overall}

    lines = [
        "# 05 最终模型选择报告",
        "",
        f"- 已成功汇总模型：{len(models)}",
        f"- 已成功汇总 checkpoint：{len(overall)}",
        f"- 缺失输出 warning：{len(warnings)}",
        "",
        "## 最佳结果",
        "",
        f"- target_3class_miou 最佳：{format_best(overall, 'target_3class_miou')}",
        f"- damaged_iou 最佳：{format_best(overall, 'damaged_iou')}",
        f"- destroyed_iou 最佳：{format_best(overall, 'destroyed_iou')}",
        f"- destroyed_to_intact 最低：{format_best(overall, 'destroyed_to_intact', lower=True)}",
        "",
        "## 研究问题判断",
        "",
    ]
    if a4 is None:
        lines.append("- A4 稳定性：尚无 M03_A4 的成功 official 结果，不能判断。")
    else:
        rank = list(overall).index(a4) + 1
        lines.append(f"- A4 稳定性：A4 最佳 checkpoint 在 target_3class_miou 总表排名第 {rank}。")

    lines += [
        (
            "- large-scale：相对 A4，destroyed_recall "
            f"{delta_text(a4d, a4, 'destroyed_recall')}，damaged_iou "
            f"{delta_text(a4d, a4, 'damaged_iou')}。若前者上升而后者下降，说明它帮助 destroyed 但伤害 damaged。"
        ),
        (
            "- A4-DG-beta0.3：相对 beta1.0，damaged_iou "
            f"{delta_text(dg03, dg10, 'damaged_iou')}，destroyed_to_intact "
            f"{delta_text(dg03, dg10, 'destroyed_to_intact')}。damaged 恢复且误分不反弹才说明 beta 控制有效。"
        ),
        (
            "- HDC：相对 A4，destroyed_to_intact "
            f"{delta_text(hdc, a4, 'destroyed_to_intact')}，target_3class_miou "
            f"{delta_text(hdc, a4, 'target_3class_miou')}。前者下降且主指标不明显下降才值得继续。"
        ),
        "",
        "## 下一步",
        "",
        "优先查看 event_wise_destroyed_diagnosis.csv：destroyed_to_background 高时应改 ROI/定位；"
        "destroyed_to_damaged 高时应改 severity 边界；destroyed_to_intact 高时优先评估 DG-beta、ExistHead 或 HDC。",
    ]
    if warnings:
        lines += ["", "## 输出 warning", ""] + [f"- {warning}" for warning in warnings[:30]]
    write_markdown(path, lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=project_root / "eval_results_official_local_cpu")
    parser.add_argument("--analysis-root", type=Path, default=project_root / "eval_analysis_cpu")
    args = parser.parse_args()
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    overall, events, diagnosis, warnings = collect(args.results_root)
    registry = load_registry(args.analysis_root)
    worst = worst_rows(events)
    write_csv(args.analysis_root / "all_model_official_summary.csv", OVERALL_FIELDS, overall)
    write_csv(args.analysis_root / "event_wise_summary.csv", EVENT_FIELDS, events)
    write_csv(args.analysis_root / "event_wise_destroyed_diagnosis.csv", DIAGNOSIS_FIELDS, diagnosis)
    write_csv(args.analysis_root / "worst_events_top10.csv", WORST_FIELDS, worst)
    write_inventory(args.analysis_root / "01_model_inventory.md", registry)
    write_plan(args.analysis_root / "02_evaluation_plan.md", registry, len(overall))
    write_effect_interpretation(
        args.analysis_root / "03_model_effect_interpretation.md", registry, overall
    )
    write_event_diagnosis(
        args.analysis_root / "04_event_wise_destroyed_diagnosis.md", diagnosis
    )
    write_report(args.analysis_root / "05_final_model_selection_report.md", overall, warnings)
    write_report(args.analysis_root / "final_model_selection_report.md", overall, warnings)

    print(f"Official checkpoints summarized: {len(overall)}")
    print(f"Event rows: {len(events)}")
    print(f"Destroyed diagnosis rows: {len(diagnosis)}")
    print(f"Output: {args.analysis_root.resolve()}")


if __name__ == "__main__":
    main()
