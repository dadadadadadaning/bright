#!/usr/bin/env python3
"""CPU 只读预览 Train 支持筛选；不修改库、不重新检索 Val、不运行模型。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import torch

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665"
NAMES = ("Background", "Intact", "Damaged", "Destroyed")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def read_json(path):
    raw = path.read_bytes()
    need(len(raw) <= 16 * 1024 * 1024, "JSON 记录过大")
    obj = json.loads(raw)
    need(isinstance(obj, dict), "JSON 顶层必须为对象")
    return obj, hashlib.sha256(raw).hexdigest()


def load_bank(root):
    root = Path(root)
    s, _ = read_json(root / "keys/summary.json")
    m, mh = read_json(root / "keys/manifest.json")
    p, ph = read_json(root / "keys/protocol.json")
    r, _ = read_json(root / "diagnose/result.json")
    for label, obj in (("summary", s), ("manifest", m), ("protocol", p)):
        need(obj.get("smoke") is False and obj.get("train_only") is True
             and obj.get("training") is False and obj.get("test_accessed") is False, label + " 不是完整 Train 库")
        need(obj.get("historical_replay_required") is False and obj.get("historical_fp32_reproduction_claim") is False,
             label + " 历史复现范围错误")
    for obj in (s, m):
        need(obj.get("format") == "SRAC_MEMORY_REBUILD_BANK_V3" and obj.get("complete") is True
             and obj.get("full_validation_evidence") is False
             and isinstance(obj.get("immutable"), dict) and bool(obj["immutable"])
             and all(v is True for v in obj["immutable"].values()), "库未完成或不可变检查失败")
    need(p.get("format") == "SRAC_MEMORY_REBUILD_PROTOCOL_V3" and p.get("storage_dtype") == p.get("query_dtype") == "torch.float32", "新协议不是 FP32")
    need(all(m.get(k) == "torch.float32" for k in ("t128_storage", "f256_storage", "label_storage")), "库保存类型不是 FP32")
    need(all(m.get(k) is True for k in ("fresh_label_checks_passed", "same_pass_q_checks_passed", "saved_fp32_roundtrip_exact", "pool_ok")), "提取检查失败")
    need(r.get("format") == "SRAC_MEMORY_REBUILD_DIAGNOSE_V3" and r.get("complete") is True
         and r.get("full_validation_evidence") is True and r.get("execution_complete") is True
         and r.get("paired_bank_smoke") is False and r.get("smoke") is False and type(r.get("image_count")) is int and r.get("image_count") == 450
         and r.get("training") is False and r.get("test_accessed") is False, "原 Val450 结果范围错误")
    need(r.get("historical_reproduction_claim") is False and r.get("labelvote_is_untrained_diagnostic") is True
         and isinstance(r.get("validation"), dict) and all(r["validation"].get(k) is True for k in
             ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")), "原诊断九项检查未通过")
    need(isinstance(r.get("immutable"), dict) and bool(r["immutable"])
         and all(v is True for v in r["immutable"].values()), "原诊断不可变检查失败")
    need(r.get("paired_manifest_sha256") == mh and s.get("rebuild_protocol_sha256") == m.get("rebuild_protocol_sha256")
         == r.get("rebuild_protocol_sha256") == ph, "协议/manifest JSON 哈希绑定不一致")
    need(m.get("source_sha256") == p.get("source_sha256") == r.get("source_sha256")
         and m.get("protocol_sha256") == m.get("source_e0_sha256") == p.get("protocol_sha256")
         == p.get("source_e0_sha256") == r.get("source_anchor_protocol_sha256"), "source/E0 绑定不一致")
    expected = m.get("paired_bank_sha256")
    need(all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in
         (expected, m.get("source_sha256"), m.get("protocol_sha256"), m.get("membership_sha256"))), "SHA256 格式无效")
    need(m.get("membership_sha256") == s.get("membership_sha256") == p.get("membership_sha256"), "成员绑定不一致")
    need(isinstance(expected, str) and len(expected) == 64
         and expected == s.get("paired_bank_sha256") == p.get("paired_bank_sha256") == r.get("paired_bank_sha256"), "bank SHA256 绑定不一致")
    with (root / "keys/paired_bank.pt").open("rb") as stream:
        need(digest(stream) == expected, "paired_bank.pt SHA256 错误")
        stream.seek(0)
        bank = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        need(digest(stream) == expected, "paired_bank.pt 在读取期间发生变化")
    need(isinstance(bank, dict), "bank 必须为字典")
    n = s.get("support_count")
    need(type(n) is int and n > 0 and n == m.get("support_count") == p.get("support_count"), "成员计数不一致")
    validate(bank, n)
    return bank


def validate(bank, n):
    for field, shape in (("t128", (n, 128)), ("f256", (n, 256)), ("distribution", (n, 4)), ("label_mass", (n,))):
        t = bank.get(field)
        need(torch.is_tensor(t) and t.device.type == "cpu" and t.shape == shape and t.dtype == torch.float32
             and bool(torch.isfinite(t).all()), "无效 FP32 CPU 张量：" + field)
    y, mass = bank["distribution"], bank["label_mass"]
    need(bool((y >= 0).all()) and bool((mass > 0).all())
         and torch.allclose(y.sum(1), torch.ones(n), atol=1e-4, rtol=1e-4), "标签/有效质量无效")
    names = bank.get("event_names")
    need(isinstance(names, (list, tuple)) and bool(names) and all(isinstance(v, str) and v for v in names)
         and len(set(names)) == len(names), "event_names 无效")
    for field in ("original_rows", "member_ids", "images", "events", "regions"):
        t = bank.get(field)
        need(torch.is_tensor(t) and t.device.type == "cpu" and t.shape == (n,) and t.dtype == torch.long
             and bool((t >= 0).all()), "成员 ID 无效：" + field)
        if field in ("original_rows", "member_ids"):
            need(t.unique().numel() == n, "成员 ID 重复：" + field)
    need(int(bank["events"].max()) < len(names), "事件 ID 越界")
    need(torch.is_tensor(bank.get("eligible")) and bank["eligible"].dtype == torch.bool
         and bank["eligible"].shape == (n,) and bank["eligible"].device.type == "cpu", "原 eligible 标记无效")
    need("split" not in bank or bank["split"] in ("train", "Train"), "存储 split 不是 Train")


def select(bank, min_fg=.5, min_purity=0):
    need(all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in (min_fg, min_purity)), "阈值必须在 [0,1]")
    validate(bank, len(bank["distribution"]))
    base = bank["eligible"].clone()
    for name in ("t128", "f256"):
        key = bank[name]
        peak = key.abs().amax(1)
        norm = (key / peak.clamp_min(torch.finfo(torch.float32).tiny)[:, None]).norm(dim=1)
        base &= peak * norm > 1e-8
    fg = bank["distribution"][:, 1:].sum(1)
    purity = torch.where(fg > 0, bank["distribution"][:, 1:].amax(1) / fg, torch.zeros_like(fg))
    return base, base & (fg >= min_fg) & (purity >= min_purity)


def stats(bank, mask):
    labels = bank["distribution"][mask].double()
    soft = labels.sum(0)
    mass = (labels * bank["label_mass"][mask, None].double()).sum(0)
    return dict(regions=int(mask.sum()), dominant_counts=torch.bincount(labels.argmax(1), minlength=4).tolist(),
                soft_share=[float(v / soft.sum()) if soft.sum() > 0 else None for v in soft],
                label_quality_share=[float(v / mass.sum()) if mass.sum() > 0 else None for v in mass])


def report(bank, base, selected, min_fg, min_purity):
    print("CPU 只读筛选预览：min_fg=", min_fg, "; min_purity=", min_purity)
    print("主类顺序：", NAMES, "；质量占比包含背景，空分母为 N/A。")
    for label, mask in (("原全部区域", torch.ones_like(base)), ("原可检索区域", base), ("筛选后区域", selected)):
        row = stats(bank, mask)
        shares = ["N/A" if v is None else f"{100*v:.3f}%" for v in row["label_quality_share"]]
        soft = ["N/A" if v is None else f"{100*v:.3f}%" for v in row["soft_share"]]
        print(label, "：区域数", row["regions"], "；主类数", row["dominant_counts"], "；软质量占比", soft, "；有效标签质量占比", shares)
    print("事件 | 原跨事件支持总量/主类数 | 筛选后跨事件支持总量/主类数 | K64容量")
    for i, event in enumerate(bank["event_names"]):
        other = bank["events"] != i
        before, after = stats(bank, base & other), stats(bank, selected & other)
        print(event, "|", before["regions"], before["dominant_counts"], "|", after["regions"], after["dominant_counts"],
              "|", "可用 K64" if after["regions"] >= 64 else "不足64")
    print("fg=p1+p2+p3；purity=max(p1,p2,p3)/fg，仅在前景内部计算，fg=0 时为0。边界使用 >=。")
    print("这里只生成内存中的原行位置 mask，保留 original_rows/member_ids；未保存新库、未改检索设置，未重新评价 Val 或 Reader。跨事件候选数不是最终 TopK 支持数；未重新校验源模型/Train 输入文件。")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--min-fg", type=float, default=.5)
    p.add_argument("--min-purity", type=float, default=0)
    args = p.parse_args(argv)
    try:
        torch.set_num_threads(2)
        bank = load_bank(args.root)
        base, selected = select(bank, args.min_fg, args.min_purity)
        report(bank, base, selected, args.min_fg, args.min_purity)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("拒绝预览：" + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
