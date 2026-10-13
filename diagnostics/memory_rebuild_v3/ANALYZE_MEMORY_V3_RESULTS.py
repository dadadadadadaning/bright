#!/usr/bin/env python3
"""只读 V3 的六份 JSON，输出详细分析；不导入模型、不读 PT/NPZ、不写文件。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665"
CHECKS = ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")
ARMS = ("t128_top8", "t128_top64", "f256_top8", "f256_top64")
CLASSES = ("Background", "Intact", "Damaged", "Destroyed")
PARTS = ("all", "foreground_pure", "foreground_mixed", "background_diluted", "mostly_background")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    need(len(raw) <= 16 * 1024 * 1024, "JSON 超出大小限制：" + str(path))
    obj = json.loads(raw)
    need(isinstance(obj, dict), "JSON 顶层不是对象：" + str(path))
    return obj, hashlib.sha256(raw).hexdigest()


def all_true(obj):
    return isinstance(obj, dict) and bool(obj) and all(v is True for v in obj.values())


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def mapping(obj):
    return obj if isinstance(obj, dict) else {}


def value(row, key, index=None):
    if not isinstance(row, dict):
        return None
    obj = row.get(key)
    if index is not None:
        obj = obj[index] if isinstance(obj, list) and len(obj) > index else None
    obj = number(obj)
    return obj if obj is not None and obj >= 0 else None


def ratio(a, b):
    a, b = number(a), number(b)
    return a / b if a is not None and a >= 0 and b is not None and b > 0 else None


def difference(a, b):
    a, b = number(a), number(b)
    return 100 * (a - b) if a is not None and b is not None else None


def vector_total(row, key):
    vector = row.get(key) if isinstance(row, dict) else None
    return sum(vector) if isinstance(vector, list) and len(vector) == 4 and all(number(x) is not None and x >= 0 for x in vector) else None


def fmt(x, percent=False, signed=False):
    x = number(x)
    if x is None:
        return "N/A"
    return ("{:+.4f}" if signed else "{:.4f}").format(x * 100 if percent else x)


def count(x, signed=False):
    return ("{:+d}" if signed else "{:d}").format(x) if type(x) is int and (signed or x >= 0) else "N/A"


def table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    if not rows:
        rows = [["N/A"] * len(headers)]
    for row in rows:
        print("| " + " | ".join(cell(x) for x in row) + " |")


def load_results(root):
    root = Path(root)
    s, _ = read_json(root / "pipeline_state.json")
    b, _ = read_json(root / "keys/summary.json")
    m, mh = read_json(root / "keys/manifest.json")
    p, ph = read_json(root / "keys/protocol.json")
    d, _ = read_json(root / "diagnose/summary.json")
    r, _ = read_json(root / "diagnose/result.json")
    need(s.get("format") == "SRAC_MEMORY_REBUILD_PIPELINE_V3", "流水线格式错误")
    need(s.get("complete") is True and s.get("full_validation_evidence") is True and not s.get("error"),
         "流水线尚未完整结束或有错误：" + str(s.get("current")) + " / " + str(s.get("error")))
    stages = s.get("stages")
    need(isinstance(stages, dict) and all(isinstance(stages.get(k), dict) and stages[k].get("status") == "complete"
         for k in ("keys", "diagnose", "report")), "全量建库、诊断、报告阶段未全部完成")
    for label, obj in (("流水线", s), ("建库摘要", b), ("清单", m), ("协议", p), ("诊断摘要", d), ("结果", r)):
        need(obj.get("training") is False and obj.get("test_accessed") is False, label + " 缺少明确 no-training/no-Test")
    need(s.get("historical_replay_required") is False and s.get("historical_reproduction_claim") is False,
         "流水线历史复现声明错误")
    for label, obj in (("建库摘要", b), ("清单", m)):
        need(obj.get("format") == "SRAC_MEMORY_REBUILD_BANK_V3" and obj.get("complete") is True
             and obj.get("full_validation_evidence") is False and all_true(obj.get("immutable")), label + " 未完成或不可变检查失败")
    for label, obj in (("建库摘要", b), ("清单", m), ("协议", p)):
        need(obj.get("smoke") is False and obj.get("train_only") is True
             and obj.get("historical_replay_required") is False
             and obj.get("historical_fp32_reproduction_claim") is False, label + " 不是全量新 Train 库")
    need(p.get("format") == "SRAC_MEMORY_REBUILD_PROTOCOL_V3", "新协议格式错误")
    need(p.get("storage_dtype") == p.get("query_dtype") == "torch.float32", "新协议不是 FP32")
    need(all(m.get(k) == "torch.float32" for k in ("t128_storage", "f256_storage", "label_storage")), "库不是 FP32")
    need(all(m.get(k) is True for k in ("fresh_label_checks_passed", "same_pass_q_checks_passed", "saved_fp32_roundtrip_exact", "pool_ok")),
         "建库关键检查未通过")
    n = b.get("image_count")
    need(type(n) is int and n > 0 and all(type(obj.get(k)) is int for obj, k in ((m, "image_count"), (b, "extraction_forward_count"), (m, "extraction_forward_count")))
         and n == m.get("image_count") == b.get("extraction_forward_count")
         == m.get("extraction_forward_count"), "建库同次前向计数不一致")
    need(all(type(obj.get("support_count")) is int for obj in (b, m, p)) and b["support_count"] > 0
         and b["support_count"] == m.get("support_count") == p.get("support_count"), "支持成员数不一致")
    for label, obj in (("诊断摘要", d), ("结果", r)):
        need(obj.get("format") == "SRAC_MEMORY_REBUILD_DIAGNOSE_V3", label + " 格式错误")
        need(type(obj.get("image_count")) is int and obj["image_count"] == 450 and obj.get("smoke") is False
             and all(obj.get(k) is True for k in ("complete", "execution_complete", "full_validation_evidence")), label + " 不是完整 Val450")
        need(isinstance(obj.get("validation"), dict) and all(obj["validation"].get(k) is True for k in CHECKS)
             and all_true(obj.get("immutable")), label + " 九项检查或不可变检查失败")
        need(obj.get("historical_reproduction_claim") is False and obj.get("labelvote_is_untrained_diagnostic") is True,
             label + " 标签投票范围声明错误")
    need(r.get("paired_bank_smoke") is False, "不能用 smoke 库作完整诊断")
    need(all(d.get(k) == r.get(k) for k in ("image_count", "complete", "execution_complete", "full_validation_evidence", "smoke", "validation")),
         "诊断 summary/result 不一致")
    source, e0, bank, members = s.get("source_sha256"), s.get("protocol_sha256"), m.get("paired_bank_sha256"), m.get("membership_sha256")
    need(all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in (source, e0, bank, members)), "来源 SHA256 格式错误")
    need(all(obj.get("source_sha256") == source and obj.get("protocol_sha256") == obj.get("source_e0_sha256") == e0
         for obj in (m, p)), "source/E0 来源绑定不一致")
    need(all(obj.get("paired_bank_sha256") == bank and obj.get("membership_sha256") == members for obj in (b, p)), "bank/member 绑定不一致")
    need(b.get("rebuild_protocol_sha256") == m.get("rebuild_protocol_sha256") == d.get("rebuild_protocol_sha256") == ph,
         "新协议 JSON 字节 SHA256 不一致")
    for k, expected in (("source_sha256", source), ("source_anchor_protocol_sha256", e0), ("paired_bank_sha256", bank),
                        ("paired_manifest_sha256", mh), ("rebuild_protocol_sha256", ph)):
        need(r.get(k) == expected, "结果来源绑定不一致：" + k)
    need(isinstance(r.get("bank_label_inventory"), dict) and r["bank_label_inventory"] == p.get("bank_label_inventory"), "库存记录不一致")
    for arm in ARMS:
        checked_transition(mapping(r.get("transitions")).get("grid_probq__labelvote_" + arm))
    return dict(root=str(root), state=s, bank=b, manifest=m, protocol=p, result=r)


def checked_transition(record):
    record = record if isinstance(record, dict) else {}
    classes, joint = mapping(record.get("per_class")), record.get("joint_gt_reference_candidate")
    if joint is None:
        return classes, False
    need(isinstance(joint, list) and len(joint) == 4 and all(isinstance(x, list) and len(x) == 4 for x in joint)
         and all(isinstance(y, list) and len(y) == 4 for x in joint for y in x)
         and all(type(v) is int and v >= 0 for x in joint for y in x for v in y), "joint 不是非负整数 [4,4,4]")
    for c, name in enumerate(CLASSES):
        gt = sum(joint[c][a][z] for a in range(4) for z in range(4))
        tp0, tp1, kept = sum(joint[c][c]), sum(joint[c][a][c] for a in range(4)), joint[c][c][c]
        harm, rescue = tp0 - kept, tp1 - kept
        fp0 = sum(joint[y][c][z] for y in range(4) for z in range(4)) - tp0
        fp1 = sum(joint[y][a][c] for y in range(4) for a in range(4)) - tp1
        expected = dict(gt_count=gt, reference_tp=tp0, candidate_tp=tp1, harm=harm, rescue=rescue,
                        net_tp=rescue-harm, reference_fp=fp0, candidate_fp=fp1, delta_fp=fp1-fp0,
                        rescue_rate=ratio(rescue, gt-tp0), harm_rate=ratio(harm, tp0))
        saved = mapping(classes.get(name))
        need(isinstance(saved, dict), "转移类别格式错误：" + name)
        for key, correct in expected.items():
            if key not in saved:
                continue
            actual = saved[key]
            match = actual is None if correct is None else (number(actual) is not None and math.isclose(actual, correct, rel_tol=1e-12, abs_tol=1e-12))
            if not key.endswith("_rate"):
                match = type(actual) is int and actual == correct
            need(match, "转移统计与 joint 不一致：" + name + "." + key)
    return classes, True


def report(records):
    r, b = records["result"], records["bank"]
    metrics, retrieval = mapping(r.get("metrics")), mapping(r.get("posthoc_retrieval"))
    grid = value(metrics.get("grid_probq"), "valid_miou")
    print("V3 完整结果的详细只读分析\n结果目录：" + records["root"])
    print("完成/范围/九项检查及小 JSON 来源绑定一致；未重新校验大 bank、权重、代码或数据。")
    print("Train 支持来源图像：", b.get("image_count"), "；区域成员：", b.get("support_count"), "；Val：450。")
    rows = []
    for arm in ("a1_pixel", "direct_region", "grid_probq", *("labelvote_" + a for a in ARMS)):
        row = metrics.get(arm, {})
        rows.append([arm, fmt(value(row, "valid_miou"), True), *(fmt(value(row, "iou", c), True) for c in range(4)),
                     fmt(difference(value(row, "valid_miou"), grid), signed=True)])
    table(["路径", "前景 mIoU %", "背景 IoU %", "Intact IoU %", "Damaged IoU %", "Destroyed IoU %", "相对 grid 百分点"], rows)
    print("valid_miou 使用原记录的固定前景三类均值；不自行用非 null 类均值替代。")
    rows, class_rows = [], []
    for arm in ARMS:
        row = mapping(retrieval.get(arm))
        n, s = value(row, "query_count"), value(row, "supported_query_count")
        rows.append([arm, count(row.get("query_count")), count(row.get("supported_query_count")), fmt(ratio(s, n), True),
                     fmt(ratio(value(row, "soft_compatibility_sum"), s)), fmt(ratio(value(row, "quality_sum", 4), s))])
        for c, name in enumerate(CLASSES):
            mass = value(row, "query_class_mass", c)
            class_rows.append([arm, name, fmt(mass), fmt(ratio(value(row, "class_soft_compatibility_sum", c), mass)),
                               fmt(ratio(value(row, "covered_query_class_mass", c), mass), True)])
    table(["key/TopK", "查询数", "支持查询数", "支持覆盖 %", "支持查询软相容度", "加权支持纯度"], rows)
    table(["key/TopK", "类别", "查询类软质量", "类软相容度", "TopK 主类覆盖 %"], class_rows)
    print("类软相容度=Σ(query_p[c]×posterior[c])/Σquery_p[c]；主类覆盖=Σ(query_p[c]×TopK存在argmax为c的支持)/Σquery_p[c]，都不是像素 Recall。")
    rows, evidence = [], mapping(r.get("posthoc_weighted_label_evidence"))
    for arm in ARMS:
        parts = mapping(evidence.get(arm))
        total = value(parts.get("all"), "queries")
        for part in PARTS:
            row = mapping(parts.get(part))
            n, s = value(row, "queries"), value(row, "supported_queries")
            rows.append([arm, part, count(row.get("queries")), fmt(ratio(n, total), True), count(row.get("supported_queries")),
                         fmt(ratio(s, n), True), fmt(ratio(value(row, "soft_compatibility_sum"), s)),
                         fmt(ratio(value(row, "mass_weighted_compatibility_sum"), value(row, "supported_valid_mass"))),
                         fmt(ratio(value(row, "intact_vote_damaged_or_destroyed"), value(row, "intact_queries")), True),
                         fmt(ratio(value(row, "damaged_or_destroyed_vote_intact"), value(row, "damaged_or_destroyed_queries")), True)])
    table(["key/TopK", "分组", "查询数", "查询占比 %", "支持查询数", "支持比例 %", "等权软相容度", "标签质量加权相容度", "Intact→损毁 %", "损毁→Intact %"], rows)
    print("沿用原分组：pure 前景比例≥0.8且前景内纯度≥0.8；mixed 前景≥0.8且纯度<0.8；diluted 前景[0.2,0.8)；mostly_background 前景<0.2。")
    print("后两列按有支持、GT 区域软标签主类分组，分母分别为 intact_queries 和 damaged_or_destroyed_queries；是区域后验误投比例，不是像素 rescue，也不区分基线原对/原错。")
    rows, verified = [], 0
    for arm in ARMS:
        classes, checked = checked_transition(mapping(r.get("transitions")).get("grid_probq__labelvote_" + arm))
        verified += int(checked)
        for name in CLASSES[1:]:
            row = mapping(classes.get(name))
            rows.append([arm, name, count(row.get("rescue")), count(row.get("harm")), fmt(value(row, "rescue_rate"), True),
                         fmt(value(row, "harm_rate"), True), count(row.get("net_tp"), True), count(row.get("delta_fp"), True)])
    table(["key/TopK", "GT 类", "救回", "伤害", "救回率 %", "伤害率 %", "净 TP", "FP 变化"], rows)
    print("joint 校核：", verified, "/4。救回率分母=基线该类错；伤害率分母=基线该类对；净TP=救回−伤害；FP变化=新−基线。缺字段不由 joint 补造。")
    rows, events = [], mapping(r.get("per_event"))
    for event in sorted(events):
        arms = mapping(events[event])
        for label, index in (("mIoU", None), ("Intact", 1), ("Damaged", 2), ("Destroyed", 3)):
            key = "valid_miou" if index is None else "iou"
            base = value(arms.get("grid_probq"), key, index)
            gt = None if index is None else value(arms.get("grid_probq"), "support", index)
            rows.append([event, label, "N/A" if index is None else count(gt), fmt(base, True),
                         *(fmt(difference(value(arms.get("labelvote_" + a), key, index), base), signed=True) for a in ("t128_top64", "f256_top64"))])
    table(["事件", "指标", "GT 像素数", "grid %", "T128 Top64 Δ百分点", "F256 Top64 Δ百分点"], rows)
    print("逐事件仅描述，无显著性结论，也没有逐事件 rescue。GT 缺类而有 FP 时 IoU 可为0，是否有该类 GT 应看 support，而不是 IoU 非 null。")
    inv = r["bank_label_inventory"]
    table(["库存", "区域数", "来源图像数", *CLASSES], [[scope, count(mapping(inv.get(scope)).get("support_count")),
           count(mapping(inv.get(scope)).get("source_image_count")), *(count(value(inv.get(scope), "dominant_counts", c)) for c in range(4))] for scope in ("all", "eligible")])
    eligible = mapping(inv.get("eligible"))
    table(["eligible 类别", "主类条数占比 %", "软标签质量占比 %", "有效标签质量占比 %"], [[name,
          *(fmt(ratio(value(eligible, key, c), vector_total(eligible, key)), True)
            for key in ("dominant_counts", "soft_mass", "valid_label_mass_by_class"))] for c, name in enumerate(CLASSES)])
    print("库存三个比例各自以对应四类向量之和为分母，保留背景；区域条数均衡不等于损毁标签质量均衡，也不是建筑数量。")
    rows, by_event = [], mapping(inv.get("by_event"))
    total = value(inv.get("eligible"), "support_count")
    for event in sorted(set(by_event) | set(events)):
        row = mapping(mapping(by_event.get(event)).get("eligible"))
        same = value(row, "support_count")
        other = total - same if total is not None and same is not None else None
        rows.append([event, count(same), count(other), *(count(value(row, "dominant_counts", c)) for c in range(4))])
    table(["事件", "同事件合格支持", "排除同事件后支持", *CLASSES], rows)
    rows = []
    for topk in ("top8", "top64"):
        row = mapping(mapping(r.get("paired_neighbor_overlap")).get(topk))
        both, total = value(row, "both_supported_query_count"), value(row, "query_count")
        rows.append([topk, fmt(ratio(both, total), True), fmt(ratio(value(row, "neighbor_jaccard_sum"), both)),
                     fmt(ratio(value(row, "neighbor_overlap_fraction_sum"), both)), fmt(ratio(value(row, "top1_agreement_count"), both), True)])
    table(["T128/F256", "双方支持比例 %", "双方支持 Jaccard", "双方支持重合比例", "双方支持 Top1 相同 %"], rows)
    print("邻居重合仅比较选出的案例，不代表标签质量。query_eligibility：", json.dumps(r.get("query_eligibility"), ensure_ascii=False))
    print("N/A=字段缺失、不是有效数字或分母为零；不补0。上述是未训练裸标签投票，不是 Reader 收益。净TP还需结合FP变化；不能单凭票分数选key，脚本不自动推荐训练或模型。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        report(load_results(args.root))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("拒绝汇总：" + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
