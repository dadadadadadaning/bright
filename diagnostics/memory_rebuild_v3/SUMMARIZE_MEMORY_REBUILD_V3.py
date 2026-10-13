#!/usr/bin/env python3
"""Read saved V3 JSON records only; stdout only, no models, data, or bank loading."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665"
PIPE_FORMAT = "SRAC_MEMORY_REBUILD_PIPELINE_V3"
BANK_FORMAT = "SRAC_MEMORY_REBUILD_BANK_V3"
PROTOCOL_FORMAT = "SRAC_MEMORY_REBUILD_PROTOCOL_V3"
DIAG_FORMAT = "SRAC_MEMORY_REBUILD_DIAGNOSE_V3"
CHECKS = ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")
ARMS = ("t128_top8", "t128_top64", "f256_top8", "f256_top64")
FOCUS_ARMS = ("t128_top64", "f256_top64")
CLASS_NAMES = ("Background", "Intact", "Damaged", "Destroyed")
MAX_JSON_BYTES = 16 * 1024 * 1024


class RecordReadError(ValueError):
    """A saved JSON may still be in the middle of a non-atomic stage write."""


def _json(path, optional=False):
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
    except FileNotFoundError:
        if optional:
            return None, None
        raise ValueError("缺少记录：" + str(path)) from None
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON 记录超出大小限制：" + str(path))
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise RecordReadError("JSON 记录不完整或格式错误，请稍后重读：" + str(path)) from error
    if not isinstance(value, dict):
        raise ValueError("JSON 顶层必须为对象：" + str(path))
    return value, hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scope(record, label):
    for field in ("test_accessed", "training"):
        _require(record.get(field) is False, label + " 缺少明确的 no-Test/no-training 范围：" + field)


def _true_map(record, field, label):
    value = record.get(field)
    _require(isinstance(value, dict) and bool(value) and all(v is True for v in value.values()),
             label + " 检查未全部通过：" + field)


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "无效的 SHA256：" + label)
    return value


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "无效的非负有限统计量：" + label)
    return float(value)


def _value(row, field, missing, prefix):
    if field not in row or row[field] is None:
        missing.append(prefix + "." + field)
        return None
    return _number(row[field], prefix + "." + field)


def _array_value(row, field, index, length, missing, prefix):
    array = row.get(field)
    if array is None:
        missing.append(prefix + "." + field)
        return None
    _require(isinstance(array, list) and len(array) == length, "统计向量维度错误：" + prefix + "." + field)
    if array[index] is None:
        missing.append(prefix + "." + field + "[" + str(index) + "]")
        return None
    return _number(array[index], prefix + "." + field + "[" + str(index) + "]")


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _metric(metrics, arm, missing, prefix="metrics"):
    row = metrics.get(arm)
    if not isinstance(row, dict):
        missing.append(prefix + "." + arm)
        return None
    value = _value(row, "valid_miou", missing, prefix + "." + arm)
    _require(value is None or value <= 1, "mIoU 超出 [0,1]：" + arm)
    return value


def _transition_derived(joint, prefix):
    """Recompute saved pixel counts only; no prediction maps or image files."""
    _require(isinstance(joint, list) and len(joint) == 4
             and all(isinstance(plane, list) and len(plane) == 4 for plane in joint)
             and all(isinstance(row, list) and len(row) == 4 for plane in joint for row in plane)
             and all(type(value) is int and value >= 0 for plane in joint for row in plane for value in row),
             "像素转移 joint 必须为非负整数 [4,4,4]：" + prefix)
    derived = {}
    for c, name in enumerate(CLASS_NAMES):
        gt = sum(joint[c][old][new] for old in range(4) for new in range(4))
        tp0 = sum(joint[c][c][new] for new in range(4))
        tp1 = sum(joint[c][old][c] for old in range(4))
        kept = joint[c][c][c]
        harm, rescue = tp0 - kept, tp1 - kept
        fp0 = sum(joint[gt_cls][c][new] for gt_cls in range(4) for new in range(4)) - tp0
        fp1 = sum(joint[gt_cls][old][c] for gt_cls in range(4) for old in range(4)) - tp1
        derived[name] = dict(gt_count=gt, reference_tp=tp0, candidate_tp=tp1,
                             harm=harm, rescue=rescue, net_tp=rescue - harm,
                             harm_rate=_ratio(harm, tp0), rescue_rate=_ratio(rescue, gt - tp0),
                             reference_fp=fp0, candidate_fp=fp1, delta_fp=fp1 - fp0)
    return derived


def _transition_tables(result, missing):
    transitions = result.get("transitions")
    if transitions is None:
        missing.append("transitions")
        transitions = {}
    _require(isinstance(transitions, dict), "transitions 应为对象")
    rows, checks = [], {}
    for arm in FOCUS_ARMS:
        pair = "grid_probq__labelvote_" + arm
        prefix = "transitions." + pair
        record = transitions.get(pair)
        if record is None:
            missing.append(prefix)
            record = {}
        _require(isinstance(record, dict), "转移统计应为对象：" + pair)
        joint = record.get("joint_gt_reference_candidate")
        derived = None
        if joint is None:
            missing.append(prefix + ".joint_gt_reference_candidate")
        else:
            derived = _transition_derived(joint, prefix)
        classes = record.get("per_class")
        if classes is None:
            missing.append(prefix + ".per_class")
            classes = {}
        _require(isinstance(classes, dict), "转移 per_class 应为对象：" + pair)
        checks[arm] = derived is not None
        for name in CLASS_NAMES[1:]:
            row_prefix = prefix + ".per_class." + name
            saved = classes.get(name)
            if saved is None:
                missing.append(row_prefix)
                saved = {}
            _require(isinstance(saved, dict), "转移类别统计应为对象：" + row_prefix)
            if derived is not None:
                for field, expected in derived[name].items():
                    if field not in saved:
                        continue
                    value = saved[field]
                    if expected is None:
                        _require(value is None, "转移统计与 joint 不一致：" + row_prefix + "." + field)
                    elif field.endswith("_rate"):
                        _require(type(value) in (int, float) and math.isfinite(value)
                                 and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
                                 "转移统计与 joint 不一致：" + row_prefix + "." + field)
                    else:
                        _require(type(value) is int and value == expected,
                                 "转移统计与 joint 不一致：" + row_prefix + "." + field)
            values = {}
            for field in ("rescue", "harm", "rescue_rate", "harm_rate", "net_tp", "delta_fp"):
                if field not in saved:
                    missing.append(row_prefix + "." + field)
                    values[field] = None
                    continue
                value = saved[field]
                if field.endswith("_rate"):
                    _require(value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1),
                             "转移率应为 [0,1] 或零分母的 null：" + row_prefix + "." + field)
                else:
                    _require(type(value) is int and (field in ("net_tp", "delta_fp") or value >= 0),
                             "转移计数应为整数：" + row_prefix + "." + field)
                values[field] = value
            rows.append(dict(arm=arm, class_name=name, **values))
    return rows, checks


def _event_tables(result, missing):
    events = result.get("per_event")
    if events is None:
        missing.append("per_event")
        events = {}
    _require(isinstance(events, dict), "per_event 应为对象")
    rows = []
    for name in sorted(events):
        metrics = events[name]
        _require(isinstance(name, str) and bool(name) and isinstance(metrics, dict), "事件统计格式错误")
        grid = _metric(metrics, "grid_probq", missing, "per_event." + name)
        deltas = {}
        for arm in FOCUS_ARMS:
            vote = _metric(metrics, "labelvote_" + arm, missing, "per_event." + name)
            deltas[arm + "_delta_pp"] = None if vote is None or grid is None else 100 * (vote - grid)
        rows.append(dict(event=name, grid_miou=grid, **deltas))
    return rows


def _tables(result):
    missing = []
    metrics = result.get("metrics")
    if metrics is None:
        missing.append("metrics")
        metrics = {}
    _require(isinstance(metrics, dict), "metrics 应为对象")
    grid = _metric(metrics, "grid_probq", missing)
    baseline = {"a1_pixel": _metric(metrics, "a1_pixel", missing), "grid_probq": grid}
    retrieval = result.get("posthoc_retrieval")
    if retrieval is None:
        missing.append("posthoc_retrieval")
        retrieval = {}
    _require(isinstance(retrieval, dict), "posthoc_retrieval 应为对象")
    rows = []
    for arm in ARMS:
        row = retrieval.get(arm)
        if row is None:
            missing.append("posthoc_retrieval." + arm)
            row = {}
        _require(isinstance(row, dict), "检索统计应为对象：" + arm)
        n = _value(row, "query_count", missing, arm)
        s = _value(row, "supported_query_count", missing, arm)
        _require(n is None or s is None or s <= n, "支持查询数大于查询总数：" + arm)
        dot = _value(row, "soft_compatibility_sum", missing, arm)
        coverage = []
        for cls in (2, 3):
            covered = _array_value(row, "covered_query_class_mass", cls, 4, missing, arm)
            total = _array_value(row, "query_class_mass", cls, 4, missing, arm)
            coverage.append(_ratio(covered, total))
        class_compatibility = []
        for cls in range(4):
            dot_class = _array_value(row, "class_soft_compatibility_sum", cls, 4, missing, arm)
            total = _array_value(row, "query_class_mass", cls, 4, missing, arm)
            class_compatibility.append(_ratio(dot_class, total))
        quality = _array_value(row, "quality_sum", 4, 6, missing, arm)
        vote = _metric(metrics, "labelvote_" + arm, missing)
        rows.append(dict(arm=arm, query_count=n, supported_query_count=s,
                         support_coverage=_ratio(s, n), soft_compatibility=_ratio(dot, s),
                         damaged_soft_coverage=coverage[0], destroyed_soft_coverage=coverage[1],
                         class_soft_compatibility=class_compatibility,
                         weighted_support_purity=_ratio(quality, s), labelvote_miou=vote,
                         delta_grid_pp=None if vote is None or grid is None else 100 * (vote - grid)))
    evidence = result.get("posthoc_weighted_label_evidence")
    if evidence is None:
        missing.append("posthoc_weighted_label_evidence")
        evidence = {}
    _require(isinstance(evidence, dict), "posthoc_weighted_label_evidence 应为对象")
    foreground = []
    for arm in ARMS:
        parts = evidence.get(arm)
        if parts is None:
            missing.append("posthoc_weighted_label_evidence." + arm)
            parts = {}
        _require(isinstance(parts, dict), "前景分组应为对象：" + arm)
        for partition in ("foreground_pure", "foreground_mixed"):
            row = parts.get(partition)
            prefix = arm + "." + partition
            if row is None:
                missing.append(prefix)
                row = {}
            _require(isinstance(row, dict), "前景分组统计应为对象：" + prefix)
            mass = _value(row, "supported_valid_mass", missing, prefix)
            dot = _value(row, "mass_weighted_compatibility_sum", missing, prefix)
            foreground.append(dict(arm=arm, partition=partition,
                                   mass_weighted_compatibility=_ratio(dot, mass),
                                   supported_valid_mass=mass))
    inventory = result["bank_label_inventory"]
    for scope in ("all", "eligible"):
        row = inventory.get(scope)
        if row is None:
            missing.append("bank_label_inventory." + scope)
            continue
        _require(isinstance(row, dict), "库存统计应为对象：" + scope)
        count = _value(row, "support_count", missing, "bank_label_inventory." + scope)
        counts = row.get("dominant_counts")
        if counts is None:
            missing.append("bank_label_inventory." + scope + ".dominant_counts")
            continue
        _require(isinstance(counts, list) and len(counts) == 4
                 and all(type(c) is int and c >= 0 for c in counts), "库存主类计数应为四个非负整数：" + scope)
        _require(count is None or count == sum(counts), "库存主类计数和不等于区域数：" + scope)
    transition_rows, transition_checks = _transition_tables(result, missing)
    event_rows = _event_tables(result, missing)
    return baseline, rows, foreground, transition_rows, transition_checks, event_rows, sorted(set(missing))


def _verify_complete(root, state, bank, manifest, protocol, protocol_sha, manifest_sha, summary, result):
    _require(bank is not None and manifest is not None and protocol is not None,
             "完整结果缺少建库 summary/manifest/protocol")
    _require(bank.get("format") == manifest.get("format") == BANK_FORMAT, "建库格式错误")
    _require(protocol.get("format") == PROTOCOL_FORMAT, "新协议格式错误")
    for label, record in (("建库摘要", bank), ("建库清单", manifest), ("新协议", protocol)):
        _scope(record, label)
        _require(record.get("smoke") is False and record.get("train_only") is True, label + " 必须为全量 Train 支持库")
        _require(record.get("historical_replay_required") is False
                 and record.get("historical_fp32_reproduction_claim") is False, label + " 历史重放声明错误")
    for label, record in (("建库摘要", bank), ("建库清单", manifest)):
        _require(record.get("complete") is True and record.get("full_validation_evidence") is False,
                 label + " 完成/范围标记错误")
        _true_map(record, "immutable", label)
    count = bank.get("image_count")
    _require(type(count) is int and count > 0 and count == manifest.get("image_count")
             == bank.get("extraction_forward_count") == manifest.get("extraction_forward_count"),
             "建库同次前向/来源图像计数不一致")
    _require(bank.get("support_count") == manifest.get("support_count") == protocol.get("support_count"),
             "支持成员计数不一致")
    for field in ("fresh_label_checks_passed", "same_pass_q_checks_passed", "saved_fp32_roundtrip_exact", "pool_ok"):
        _require(manifest.get(field) is True, "建库验证未通过：" + field)
    for field in ("t128_storage", "f256_storage", "label_storage"):
        _require(manifest.get(field) == "torch.float32", "建库必须保存 FP32：" + field)
    _require(protocol.get("storage_dtype") == "torch.float32" and protocol.get("query_dtype") == "torch.float32",
             "新协议必须为 FP32")
    for field in ("image_count", "complete", "full_validation_evidence", "execution_complete", "smoke", "validation"):
        _require(summary.get(field) == result.get(field), "Val 摘要与结果不一致：" + field)
    for label, record in (("Val 摘要", summary), ("Val 结果", result)):
        _scope(record, label)
        _require(record.get("format") == DIAG_FORMAT, label + " 格式错误")
        _require(type(record.get("image_count")) is int and record.get("image_count") == 450 and record.get("smoke") is False
                 and all(record.get(f) is True for f in ("complete", "full_validation_evidence", "execution_complete")),
                 label + " 不是完整 Val450")
        _require(record.get("historical_reproduction_claim") is False
                 and record.get("labelvote_is_untrained_diagnostic") is True, label + " 实验声明错误")
        _true_map(record, "immutable", label)
        checks = record.get("validation")
        _require(isinstance(checks, dict) and all(checks.get(c) is True for c in CHECKS), label + " 九项验证未全部通过")
    _require(result.get("paired_bank_smoke") is False, "不能用 smoke 库声明完整验证")
    source = _sha(state.get("source_sha256"), "pipeline.source_sha256")
    e0 = _sha(state.get("protocol_sha256"), "pipeline.protocol_sha256")
    bank_sha = _sha(manifest.get("paired_bank_sha256"), "manifest.paired_bank_sha256")
    membership_sha = _sha(manifest.get("membership_sha256"), "manifest.membership_sha256")
    for label, record in (("manifest", manifest), ("protocol", protocol)):
        _require(record.get("source_sha256") == source and record.get("protocol_sha256") == e0
                 and record.get("source_e0_sha256") == e0, label + " source/E0 绑定不一致")
    for label, record in (("summary", bank), ("protocol", protocol)):
        _require(record.get("paired_bank_sha256") == bank_sha and record.get("membership_sha256") == membership_sha,
                 label + " bank/member 绑定不一致")
    _require(bank.get("rebuild_protocol_sha256") == manifest.get("rebuild_protocol_sha256") == protocol_sha,
             "新协议 JSON SHA256 不一致")
    _require(summary.get("rebuild_protocol_sha256") == protocol_sha, "Val 摘要新协议绑定不一致")
    for field, expected in (("source_sha256", source), ("source_anchor_protocol_sha256", e0),
                            ("paired_bank_sha256", bank_sha), ("rebuild_protocol_sha256", protocol_sha),
                            ("paired_manifest_sha256", manifest_sha)):
        _require(result.get(field) == expected, "Val 结果来源绑定不一致：" + field)
    _require(result.get("bank_label_inventory") == protocol.get("bank_label_inventory")
             and isinstance(result.get("bank_label_inventory"), dict), "标签库存记录不一致或缺失")


def read_summary(root):
    """Return pending/ready saved-record summary. Never import experiment code or write."""
    root = Path(root)
    state, _ = _json(root / "pipeline_state.json")
    _require(state.get("format") == PIPE_FORMAT, "不是 V3 流水线记录")
    _scope(state, "流水线")
    _require(state.get("historical_replay_required") is False
             and state.get("historical_reproduction_claim") is False, "流水线历史重放声明错误")
    _require(isinstance(state.get("stages"), dict), "缺少流水线阶段记录")
    current = state.get("current")
    _require(current is None or isinstance(current, str), "无效的当前阶段")
    claims_full = state.get("complete") is True or state.get("full_validation_evidence") is True
    record_read_notes = []
    def stage_summary(name):
        try:
            return _json(root / name / "summary.json", optional=True)[0]
        except RecordReadError:
            if claims_full:
                raise
            record_read_notes.append(name + "/summary.json 正在写入或尚不完整，请稍后重读；这不是实验失败结论。")
            return None
    bank = stage_summary("keys")
    diag = stage_summary("diagnose")
    for record, expected, label in ((bank, BANK_FORMAT, "建库"), (diag, DIAG_FORMAT, "诊断")):
        if record is not None:
            _require(record.get("format") == expected, label + " 格式错误")
            _scope(record, label)
    chosen = diag if isinstance(current, str) and current.startswith("diagnose") else bank
    progress = dict(status="pending", root=str(root), current=current,
                    smoke_complete=state.get("smoke_complete"), image_count=None if chosen is None else chosen.get("image_count"),
                    phase=None if chosen is None else chosen.get("phase"), error=state.get("error") or (chosen or {}).get("error"),
                    record_read_notes=record_read_notes,
                    log=str(root / "logs" / (current + ".log")) if isinstance(current, str) and current not in ("complete", "complete_smoke") else None,
                    stages={name: row.get("status") for name, row in state["stages"].items() if isinstance(row, dict)})
    diag_claim = diag is not None and (diag.get("complete") is True or diag.get("full_validation_evidence") is True)
    if not claims_full and not diag_claim:
        return progress
    manifest, manifest_sha = _json(root / "keys" / "manifest.json")
    protocol, protocol_sha = _json(root / "keys" / "protocol.json")
    result, _ = _json(root / "diagnose" / "result.json")
    _require(diag is not None, "完整声明缺少诊断摘要")
    _verify_complete(root, state, bank, manifest, protocol, protocol_sha, manifest_sha, diag, result)
    if not claims_full:
        progress.update(phase="Val450 已完成，等待流水线报告阶段结束", image_count=450)
        return progress
    _require(state.get("complete") is True and state.get("full_validation_evidence") is True and not state.get("error"),
             "流水线完整声明不一致或有错误")
    for stage in ("keys", "diagnose", "report"):
        _require(progress["stages"].get(stage) == "complete", "流水线完整声明缺少已完成阶段：" + stage)
    baseline, rows, foreground, transitions, transition_checks, events, missing = _tables(result)
    progress.update(status="ready", image_count=450, baseline=baseline, retrieval=rows,
                    foreground=foreground, transitions=transitions, transition_checks=transition_checks,
                    per_event=events, inventory=result["bank_label_inventory"], missing_fields=missing)
    return progress


def _fmt(value, percent=False, signed=False):
    if value is None:
        return "N/A"
    value = value * 100 if percent else value
    return ("{:+.3f}" if signed else "{:.3f}").format(value)


def _count_fmt(value, signed=False):
    if value is None:
        return "N/A"
    return ("{:+d}" if signed else "{:d}").format(value)


def render(summary):
    lines = ["V3 Memory 主线：已保存记录的只读汇总", "", "结果目录：`" + summary["root"] + "`"]
    if summary["status"] != "ready":
        lines += ["", "当前阶段：" + str(summary.get("current")), "阶段状态：" + json.dumps(summary["stages"], ensure_ascii=False),
                  "已处理图像：" + str(summary.get("image_count")), "阶段记录：" + str(summary.get("phase")),
                  "错误：" + str(summary.get("error")), ""]
        if summary.get("log"):
            lines.append("阶段日志：`" + summary["log"] + "`")
        lines.extend(summary.get("record_read_notes", []))
        lines.append("当前未达到完整流水线 + Val450 汇总条件；等待当前任务完成。若有错误，先检查阶段日志。不要重复启动，不用 smoke 分数判断检索效果。")
        return "\n".join(lines)
    lines += ["", "完整 Val450 的完成、范围、九项验证及小 JSON 来源绑定记录一致。这里只汇总保存的记录，没有重新校验大 bank、权重、源代码或数据文件。",
              "", "基线前景三类 mIoU：A1=" + _fmt(summary["baseline"]["a1_pixel"], True) + "%；grid_probq=" + _fmt(summary["baseline"]["grid_probq"], True) + "%。",
              "", "| key / TopK | 支持覆盖 % | 支持查询软相容度 | Damaged 软覆盖 % | Destroyed 软覆盖 % | 加权支持纯度 | 裸标签投票 mIoU % | 相对 grid 百分点 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["retrieval"]:
        lines.append("| " + row["arm"] + " | " + " | ".join((
            _fmt(row["support_coverage"], True), _fmt(row["soft_compatibility"]),
            _fmt(row["damaged_soft_coverage"], True), _fmt(row["destroyed_soft_coverage"], True),
            _fmt(row["weighted_support_purity"]), _fmt(row["labelvote_miou"], True),
            _fmt(row["delta_grid_pp"], signed=True))) + " |")
    lines += ["", "支持覆盖=支持查询数/查询数；软相容度=标签分布点积和/支持查询数；类别软覆盖=covered_query_class_mass[c]/query_class_mass[c]；支持纯度=quality_sum[4]/支持查询数。类软覆盖不是 Recall，也不是建筑数量。",
              "", "| key / TopK | 背景类软相容度 | Intact 类软相容度 | Damaged 类软相容度 | Destroyed 类软相容度 |",
              "|---|---:|---:|---:|---:|"]
    for row in summary["retrieval"]:
        lines.append("| " + row["arm"] + " | " + " | ".join(_fmt(v) for v in row["class_soft_compatibility"]) + " |")
    lines += ["", "类软相容度=class_soft_compatibility_sum[c]/query_class_mass[c]，即按查询的该类软标签质量加权的检索后验该类概率；不是类别 IoU，也不是 Recall。",
              "", "| key / TopK | 前景分组 | 有支持标签质量 | 质量加权软相容度 |", "|---|---|---:|---:|"]
    for row in summary["foreground"]:
        lines.append("| " + row["arm"] + " | " + row["partition"] + " | " +
                     _fmt(row["supported_valid_mass"]) + " | " + _fmt(row["mass_weighted_compatibility"]) + " |")
    lines += ["", "前景分组沿用 V3 预设：foreground_pure 为前景比例≥0.8且前景内纯度≥0.8；foreground_mixed 为前景比例≥0.8且纯度<0.8。分组仅用于事后诊断。",
              "", "Top64 裸标签投票相对 grid_probq 的像素帮助/伤害（不是训练后 Reader 收益）：",
              "", "| key / TopK | GT 类别 | 救回像素 | 伤害像素 | 救回率 % | 伤害率 % | 净 TP | FP 变化 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["transitions"]:
        lines.append("| " + row["arm"] + " | " + row["class_name"] + " | " + " | ".join((
            _count_fmt(row["rescue"]), _count_fmt(row["harm"]), _fmt(row["rescue_rate"], True),
            _fmt(row["harm_rate"], True), _count_fmt(row["net_tp"], True), _count_fmt(row["delta_fp"], True))) + " |")
    checked = sum(summary["transition_checks"].values())
    lines += ["", "救回率=该类原错后对/该类基线错误像素；伤害率=该类原对后错/该类基线正确像素；净 TP=救回−伤害；FP 变化=候选误报−基线误报。"
              + "读取了 " + str(checked) + "/" + str(len(FOCUS_ARMS)) + " 组 joint[GT,reference,candidate]，用整数循环核对已有 per_class 统计；缺项不由 joint 补造。",
              "", "| 事件 | grid_probq mIoU % | T128 Top64 相对 grid 百分点 | F256 Top64 相对 grid 百分点 |",
              "|---|---:|---:|---:|"]
    if not summary["per_event"]:
        lines.append("| N/A | N/A | N/A | N/A |")
    for row in summary["per_event"]:
        name = row["event"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append("| " + name + " | " + _fmt(row["grid_miou"], True) + " | "
                     + _fmt(row["t128_top64_delta_pp"], signed=True) + " | "
                     + _fmt(row["f256_top64_delta_pp"], signed=True) + " |")
    lines += ["", "逐事件数值仅作描述，不包含显著性或全新事件泛化结论；未重新运行模型或读取预测 NPZ。",
              "", "| 库存 | 支持区域数 | 背景主类 | Intact 主类 | Damaged 主类 | Destroyed 主类 |", "|---|---:|---:|---:|---:|---:|"]
    for scope in ("all", "eligible"):
        row = summary["inventory"].get(scope)
        if not isinstance(row, dict) or not isinstance(row.get("dominant_counts"), list) or len(row["dominant_counts"]) != 4:
            lines.append("| " + scope + " | N/A | N/A | N/A | N/A | N/A |")
        else:
            count = row.get("support_count")
            lines.append("| " + scope + " | " + ("N/A" if count is None else str(count)) + " | " + " | ".join(str(v) for v in row["dominant_counts"]) + " |")
    if summary["missing_fields"]:
        lines += ["", "缺少统计字段（对应单元格 N/A，不补成 0）：" + ", ".join(summary["missing_fields"])]
    lines += ["", "N/A 表示字段缺失或分母为零。裸标签投票是未训练诊断，不能解释为 Reader 收益，不能单凭其 mIoU 选择 key；应同时检查前景证据、覆盖和基线错误处的互补性。当前脚本不自动推荐 key、不训练 Reader、不宣称历史旧库复现。"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT), help="V3 结果目录；只读取保存的 JSON")
    args = parser.parse_args(argv)
    try:
        print(render(read_summary(args.root)))
    except (ValueError, OSError) as error:
        print("无法汇总：" + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
