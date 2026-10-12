#!/usr/bin/env python3
"""Print a compact, decision-oriented report for change-branch diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value):
    return "NA" if value is None else f"{100.0 * float(value):.4f}%"


def _num(value):
    return "NA" if value is None else f"{float(value):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    alignment = data["alignment"]
    gate = data["gate"]
    residual = data["residual_path"]
    gradients = data["gain_gradient"].get("mean", {})

    confidence = alignment["confidence"]["mean"]
    entropy = alignment["alignment_entropy"]["mean"]
    score_margin = alignment["score_margin"]["mean"]
    weight_margin = alignment["weight_margin"]["mean"]
    top1 = alignment["top1_weight"]["mean"]
    center = alignment["center_is_winner"]["mean"]
    gate_mean = gate["distribution"]["mean"]
    corr = gate["pearson_with_confidence"]
    bins = gate.get("by_confidence_quintile", [])
    gate_gap = None
    if len(bins) >= 2 and bins[0]["gate_mean"] is not None and bins[-1]["gate_mean"] is not None:
        gate_gap = bins[-1]["gate_mean"] - bins[0]["gate_mean"]

    structural = residual["structural_logit_abs"]["mean"] or 0.0
    applied = residual["applied_residual_abs"]["mean"] or 0.0
    residual_ratio = applied / max(structural, 1.0e-12)
    counterfactual = residual["counterfactual_alpha_scale"]

    print("=" * 78)
    print("1. 3x3 matching / confidence")
    print(f"confidence={_num(confidence)} entropy={_num(entropy)}")
    print(f"score_margin={_num(score_margin)} weight_margin={_num(weight_margin)}")
    print(f"top1_weight={_pct(top1)} (uniform 3x3 = 11.1111%)")
    print(f"center_is_winner={_pct(center)}")
    for name, row in alignment.get("synthetic_shift_equivariance", {}).items():
        print(f"{name} known-shift equivariance={_pct(row.get('accuracy'))} n={row.get('total')}")
    if confidence is not None and confidence < 0.02 and top1 is not None and top1 < 0.16:
        print("判断：匹配分布接近均匀，当前Key缺少位置区分力。")
    else:
        print("判断：不能仅凭平均置信度判失败，继续结合margin和事件分布。")

    print("\n" + "=" * 78)
    print("2. Gate是否服从可靠性")
    print(f"gate_mean={_pct(gate_mean)} corr(gate,confidence)={_num(corr)}")
    print(f"highest-minus-lowest confidence quintile gate={_num(gate_gap)}")
    if (corr is None or corr < 0.20) and (gate_gap is None or gate_gap < 0.05):
        print("判断：Gate基本没有随置信度变化；它不是有效的可靠性门控。")
    else:
        print("判断：Gate与置信度存在响应，仍需检查低置信度分位的绝对Gate值。")

    print("\n" + "=" * 78)
    print("3. SAR变化残差是否进入区域logits")
    print(f"equation_max_abs_error={_num(residual['equation_max_abs_error'])}")
    print(f"structural_abs={_num(structural)} applied_residual_abs={_num(applied)}")
    print(f"applied/structural={_pct(residual_ratio)}")
    print(f"raw_gain={_num(residual['raw_change_gain'])} tanh_gain={_num(residual['tanh_change_gain'])}")
    if residual["equation_max_abs_error"] < 1.0e-5 and residual_ratio < 0.01:
        print("判断：计算图连接正确，但数值贡献不足结构logits的1%，属于‘进入但几乎不起作用’。")
    elif residual["equation_max_abs_error"] < 1.0e-5:
        print("判断：残差已按预期进入logits，且具有可测量幅度。")
    else:
        print("判断：残差恒等式未通过，需要先修复forward或诊断口径。")

    print("\n" + "=" * 78)
    print("4. change_alpha为何小：梯度与反事实")
    for key in ("total", "bag", "change_reg", "damage_reconstruction", "occurrence_boundary"):
        print(f"grad[{key}]={_num(gradients.get(key))}")
    print("正梯度表示梯度下降会把gain压低；负梯度表示会把gain推高。")
    for scale in ("0.0", "1.0", "5.0", "10.0"):
        row = counterfactual.get(scale, {})
        print(
            f"scale={scale:>4s} hard_change={_pct(row.get('hard_assignment_change_fraction'))} "
            f"recon={_num(row.get('damage_reconstruction_loss'))} "
            f"boundary={_num(row.get('occurrence_boundary_loss'))}"
        )
    base = counterfactual.get("0.0", {})
    strong = counterfactual.get("5.0", {})
    improves = (
        strong.get("damage_reconstruction_loss") is not None
        and base.get("damage_reconstruction_loss") is not None
        and strong["damage_reconstruction_loss"] < base["damage_reconstruction_loss"]
    )
    reg_grad = gradients.get("change_reg", 0.0) or 0.0
    total_grad = gradients.get("total", 0.0) or 0.0
    raw_gain = float(residual["raw_change_gain"])
    # SGD: g <- g - lr*grad.  grad and gain with the same sign means the
    # update moves gain toward zero; opposite signs increase |gain|.
    reg_toward_zero = reg_grad * raw_gain > 0
    total_toward_zero = total_grad * raw_gain > 0
    print(f"change_reg pushes |gain| toward zero: {reg_toward_zero}")
    print(f"total loss pushes |gain| toward zero: {total_toward_zero}")
    if improves and reg_toward_zero and total_toward_zero:
        print("判断：放大残差改善语义目标，但总梯度仍压低gain；存在过强正则/目标冲突嫌疑。")
    elif improves:
        print("判断：更强残差有潜力；优先检查学习率、零初始化和监督强度。")
    else:
        print("判断：放大残差未改善语义目标，alpha小更可能是残差信号无效，而非单纯正则过强。")

    print("\n" + "=" * 78)
    print("5. 下一步决策")
    print("A. Key近均匀：先做匹配Key正负样本训练/多尺度上下文，不先放大alpha。")
    print("B. Key可分但Gate失效：改成 gate = confidence * learned_utility，并加单调性对照。")
    print("C. Key与Gate正常但残差太弱：再做CHANGE_ALPHA_MAX与正则权重消融。")
    print("D. scale=5/10语义损失变差：停止放大残差，重做delta_logits的监督目标。")


if __name__ == "__main__":
    main()
