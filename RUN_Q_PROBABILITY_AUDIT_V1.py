#!/usr/bin/env python3
"""CPU-only rescue/harm audit of archived A1/pixel-grid/Q Val450 predictions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

SOURCE = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/restored_20261014_134825_xno2hi6o/sar_q/formal/val_baseline"
REPO = "/root/shared-nvme/BRIGHT-master"
PARENT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_Q_PROBABILITY_AUDIT_V1"
FORMAT = "SRAC_Q_PROBABILITY_ARCHIVE_AUDIT_V1"
CLASSES = ("Background", "Intact", "Damaged", "Destroyed")
ARMS = ("q_a1_pixel", "q_pixel_grid", "q_grid_probq")
PAIRS = {"a1_to_pixel": ARMS[:2], "pixel_to_q": ARMS[1:],
         "a1_to_q": (ARMS[0], ARMS[2])}
Q_GROUPS = ("background_dominant", "background_foreground_mixed",
            "foreground_class_mixed", "foreground_pure", "unlabelled_primary_region")
EDGE_GROUPS = ("building_boundary_band", "away_from_building_boundary", "ignore_neighborhood")
np = None


def init_numpy():
    global np
    import numpy
    np = numpy


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inside(root, relative):
    require(isinstance(relative, str) and bool(relative), "Missing relative artifact path")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe artifact path")
    path = (root / rel).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Artifact escapes source: " + str(path))
    return path


def signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def load_source(source, repo):
    root, repo = Path(source).resolve(strict=True), Path(repo).resolve(strict=True)
    require(root.is_dir() and (repo / "bda_benchmark").is_dir(), "Missing source evaluation or repository")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    module = importlib.import_module("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.compare")
    require(Path(module.__file__).resolve().is_relative_to(repo), "Formal gate imported from another checkout")
    # Use the existing formal gate; never alter or bypass its conditions.
    summary_path, summary = module.checked_eval(root)
    report_path = inside(root, summary.get("result_file"))
    require(re.fullmatch(r"[0-9a-f]{64}", str(summary.get("result_sha256", "")))
            and digest(report_path) == summary["result_sha256"], "Source result SHA256 mismatch")
    _, report = module.checked_eval(report_path)
    for key in ("format", "arm", "image_count", "checkpoint_sha256", "protocol_sha256", "source_sha256", "metrics", "per_event"):
        require(summary.get(key) == report.get(key), "Source summary/result disagree: " + key)
    protocol_path = Path(report["protocol_path"]).resolve(strict=True)
    require(digest(protocol_path) == report["protocol_sha256"], "Bound source protocol changed")
    protocol = read_json(protocol_path)
    names = protocol.get("selected_images")
    require(isinstance(names, list) and len(names) == 450 and len(set(names)) == 450,
            "Protocol must bind ordered Val450 image identities")
    journal_path = root / "images.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 450 and [r.get("index") for r in rows] == list(range(450))
            and [r.get("image") for r in rows] == names, "Prediction journal differs from ordered Val450")
    for row in rows:
        require(isinstance(row.get("event"), str) and bool(row["event"])
                and row["image"].startswith(row["event"] + "_"), "Invalid journal event identity")
        require(re.fullmatch(r"[0-9a-f]{64}", str(row.get("prediction_sha256", ""))), "Missing prediction SHA256")
        inside(root, row.get("prediction_file"))
    snapshots = {str(p): digest(p) for p in (summary_path, report_path, protocol_path, journal_path,
                                            Path(module.__file__).resolve(), Path(__file__).resolve())}
    return dict(root=root, repo=repo, report=report, rows=rows, protocol=protocol,
                source_files=snapshots, dataset=Path(protocol["arguments"]["dataset"]).expanduser().resolve())


def label_map(value, name, target=False):
    a = np.asarray(value)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), name + " must be an integer 2-D map")
    require(bool((((a >= 0) & (a < 4)) | ((a == 255) if target else False)).all()), "Invalid class ID: " + name)
    return a


def matrix(value, name):
    a = np.asarray(value)
    require(a.shape == (4, 4) and np.issubdtype(a.dtype, np.integer) and bool((a >= 0).all()), "Missing 4x4 counts: " + name)
    return a.astype(np.int64)


def confusion(gt, pred):
    valid = gt != 255
    return np.bincount(4 * gt[valid].astype(np.int64) + pred[valid], minlength=16).reshape(4, 4)


def metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    rows, cols = cm.sum(1), cm.sum(0)
    union = rows + cols - tp
    iou = [float(tp[c] / union[c]) if union[c] else None for c in range(4)]
    return dict(confusion=cm.tolist(), iou=iou,
                foreground_miou=sum(x or 0.0 for x in iou[1:]) / 3,
                four_class_miou=sum(x or 0.0 for x in iou) / 4,
                oa=float(tp.sum() / cm.sum()) if cm.sum() else None,
                precision=[float(tp[c] / cols[c]) if cols[c] else None for c in range(4)],
                recall=[float(tp[c] / rows[c]) if rows[c] else None for c in range(4)],
                absent_class_rule="fixed class average; zero-union IoU is null and contributes zero")


def summarize_cube(cube):
    old, new = cube.sum(2), cube.sum(1)
    result = dict(axes=["GT", "old_prediction", "new_prediction"], counts=cube.tolist(),
                  old=metrics(old), new=metrics(new), classes={})
    for c, name in enumerate(CLASSES):
        rescue_from = {CLASSES[j]: int(cube[c, j, c]) for j in range(4) if j != c}
        harm_to = {CLASSES[j]: int(cube[c, c, j]) for j in range(4) if j != c}
        rescue, harm = sum(rescue_from.values()), sum(harm_to.values())
        require(int(new[c, c] - old[c, c]) == rescue - harm, "Rescue/harm identity failed")
        old_wrong = int(old[c].sum() - old[c, c])
        fp_added = sum(int(cube[g, :, c].sum() - cube[g, c, c]) for g in range(4) if g != c)
        fp_removed = sum(int(cube[g, c, :].sum() - cube[g, c, c]) for g in range(4) if g != c)
        result["classes"][name] = dict(gt_pixels=int(old[c].sum()), old_correct=int(old[c, c]),
            new_correct=int(new[c, c]), rescue=rescue, harm=harm, net_tp=rescue-harm,
            rescue_fraction_of_old_errors=rescue/old_wrong if old_wrong else None,
            harm_fraction_of_old_correct=harm/int(old[c, c]) if old[c, c] else None,
            rescue_from=rescue_from, harm_to=harm_to, fp_added=fp_added, fp_removed=fp_removed,
            net_fp=fp_added-fp_removed)
    result["foreground_delta_pp"] = 100 * (result["new"]["foreground_miou"] - result["old"]["foreground_miou"])
    return result


def dilate(mask, radius):
    require(type(radius) is int and 0 <= radius <= 8, "Boundary radius must be 0..8")
    a = np.asarray(mask, dtype=bool)
    h, w = a.shape
    padded = np.pad(a, radius)
    out = np.zeros_like(a)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            out |= padded[y:y+h, x:x+w]
    return out


def edge_groups(gt, radius):
    valid, building = gt != 255, (gt > 0) & (gt < 4)
    boundary = np.zeros(gt.shape, dtype=bool)
    vertical = valid[1:] & valid[:-1] & (building[1:] != building[:-1])
    horizontal = valid[:, 1:] & valid[:, :-1] & (building[:, 1:] != building[:, :-1])
    boundary[1:] |= vertical; boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal; boundary[:, :-1] |= horizontal
    band = dilate(boundary, radius)
    ignore_near = dilate(~valid, radius + 1)
    # Ignore-adjacent valid pixels get a separate group instead of a false edge.
    return np.where(ignore_near, 2, np.where(band, 0, 1)).astype(np.uint8)


def q_groups(gt, archive):
    for key in ("q_new", "region_ids_new", "grid_shape_new"):
        require(key in archive, "Missing current-Q archive field: " + key)
    q = np.asarray(archive["q_new"], dtype=np.float64)
    if q.ndim == 3 and q.shape[0] == 1:
        q = q[0]
    ids = np.asarray(archive["region_ids_new"])
    shape = np.asarray(archive["grid_shape_new"]).reshape(-1)
    require(len(shape) == 2 and np.issubdtype(shape.dtype, np.integer), "Invalid saved Q grid shape")
    gh, gw = map(int, shape)
    h, w = gt.shape
    require(min(gh, gw) > 0 and h % gh == 0 and w % gw == 0,
            "Q grouping requires exact integer GT-to-grid area ratios; no approximate resize")
    require(q.ndim == 2 and ids.shape == q.shape and q.shape[1] == gh*gw
            and np.issubdtype(ids.dtype, np.integer) and bool((ids >= 0).all())
            and int(ids.max()) < gh*gw*q.shape[0], "Invalid or out-of-range saved Q IDs")
    require(bool(np.isfinite(q).all()) and bool((q >= 0).all())
            and np.allclose(q.sum(0), 1, atol=2e-5, rtol=0), "Saved Q is not a finite normalized soft partition")
    bh, bw = h//gh, w//gw
    area = lambda a: a.reshape(gh, bh, gw, bw).mean(axis=(1, 3)).reshape(-1)
    valid = area(gt != 255)
    fractions = np.stack([area(gt == c) for c in range(4)], 1)
    n = int(ids.max()) + 1
    mass = np.bincount(ids.ravel(), weights=(q*valid[None]).ravel(), minlength=n)
    class_mass = np.stack([np.bincount(ids.ravel(), weights=(q*fractions[:, c][None]).ravel(), minlength=n) for c in range(4)], 1)
    labels = np.divide(class_mass, mass[:, None], out=np.zeros_like(class_mass), where=mass[:, None] > 0)
    fg = labels[:, 1:].sum(1)
    purity = np.divide(labels[:, 1:].max(1), fg, out=np.zeros(n), where=fg > 0)
    categories = np.where(fg < .2, 0, np.where(fg < .8, 1, np.where(purity < .8, 2, 3)))
    categories[mass == 0] = 4
    # This hard association is ONLY a post-forward descriptive grouping.
    # The original soft-Q model predictions are read unchanged from the archive.
    primary = ids[q.argmax(0), np.arange(q.shape[1])]
    grouped = categories[primary].reshape(gh, gw)
    pixels = np.repeat(np.repeat(grouped, bh, 0), bw, 1).astype(np.uint8)
    inventory = {name: dict(regions=int(((categories == j) & (mass > 0)).sum()),
                           labelled_grid_mass=float(mass[categories == j].sum())) for j, name in enumerate(Q_GROUPS)}
    return pixels, inventory


def image_counts(gt, predictions, qg, edges):
    valid = gt != 255
    result = {}
    for pair, (old_name, new_name) in PAIRS.items():
        code = gt[valid].astype(np.int64)*16 + predictions[old_name][valid].astype(np.int64)*4 + predictions[new_name][valid]
        total = np.bincount(code, minlength=64).reshape(4, 4, 4)
        qcounts = np.bincount(qg[valid].astype(np.int64)*64 + code, minlength=len(Q_GROUPS)*64).reshape(len(Q_GROUPS), 4, 4, 4)
        ecounts = np.bincount(edges[valid].astype(np.int64)*64 + code, minlength=len(EDGE_GROUPS)*64).reshape(len(EDGE_GROUPS), 4, 4, 4)
        require(np.array_equal(qcounts.sum(0), total) and np.array_equal(ecounts.sum(0), total), "Diagnostic strata lost pixels")
        result[pair] = dict(all=total, q_groups=qcounts, edge_groups=ecounts)
    return result


def empty_counts():
    return {p: dict(all=np.zeros((4, 4, 4), dtype=np.int64),
                    q_groups=np.zeros((len(Q_GROUPS), 4, 4, 4), dtype=np.int64),
                    edge_groups=np.zeros((len(EDGE_GROUPS), 4, 4, 4), dtype=np.int64)) for p in PAIRS}


def add_counts(total, incoming):
    for pair in PAIRS:
        for key in total[pair]:
            total[pair][key] += incoming[pair][key]


def summarise_counts(counts):
    return {pair: dict(old_arm=PAIRS[pair][0], new_arm=PAIRS[pair][1],
        overall=summarize_cube(value["all"]),
        q_groups={name: summarize_cube(value["q_groups"][i]) for i, name in enumerate(Q_GROUPS)},
        edge_groups={name: summarize_cube(value["edge_groups"][i]) for i, name in enumerate(EDGE_GROUPS)}) for pair, value in counts.items()}


def check_aggregate(counts, report, event=None):
    expected = report["metrics"] if event is None else report["per_event"][event]
    for pair, (old, new) in PAIRS.items():
        cube = counts[pair]["all"]
        require(np.array_equal(cube.sum(2), matrix(expected[old]["confusion"], old))
                and np.array_equal(cube.sum(1), matrix(expected[new]["confusion"], new)),
                "Aggregate confusion disagrees: " + str(event or "all") + "/" + pair)
        for arm, cm in ((old, cube.sum(2)), (new, cube.sum(1))):
            actual = metrics(cm)
            stored = expected[arm]
            require(len(stored.get("iou", [])) == 4, "Missing stored class IoUs")
            require(all((a is None and b is None) or
                        (a is not None and b is not None and abs(a-b) <= 1e-12)
                        for a,b in zip(actual["iou"], stored["iou"]))
                    and abs(actual["foreground_miou"] - stored["valid_miou"]) <= 1e-12,
                    "Recomputed metrics disagree with source: " + str(event or "all") + "/" + arm)


def protected_output(raw, source):
    path = Path(raw).expanduser().resolve()
    require(not path.exists(), "Use a new result directory; preserving: " + str(path))
    for root in (source["root"], source["repo"], source["dataset"]):
        require(not path.is_relative_to(root), "Output is inside a protected source/code/dataset directory")
    return path


def report_text(result):
    lines = ["Q概率汇总的Val配对诊断", "范围：%d/450张；完整诊断=%s" % (result["image_count"], result["diagnostic_complete"]),
             "只读CPU统计；无训练、无新推理、无Test；数量单位为像素，不是建筑。", ""]
    for pair, item in result["pairs"].items():
        m = item["overall"]
        lines.append("%s: %s → %s; 前景mIoU %.4f%% → %.4f%%; 差值 %+.4f个百分点" %
                     (pair, item["old_arm"], item["new_arm"], 100*m["old"]["foreground_miou"], 100*m["new"]["foreground_miou"], m["foreground_delta_pp"]))
        for name, row in m["classes"].items():
            lines.append("  %s: 救回=%d 误改=%d 净TP=%+d 净FP=%+d; 误改去向=%s" %
                         (name, row["rescue"], row["harm"], row["net_tp"], row["net_fp"], json.dumps(row["harm_to"], ensure_ascii=False)))
    lines += ["", "A1→Q：分组内完好误改与毁坏救回（计数不能脱离组大小比较）"]
    for family in ("q_groups", "edge_groups"):
        for group, item in result["pairs"]["a1_to_q"][family].items():
            intact, destroyed = item["classes"]["Intact"], item["classes"]["Destroyed"]
            rate = intact["harm_fraction_of_old_correct"]
            lines.append("  %s: 完好GT=%d 救回=%d 误改=%d 误改率=%s; 毁坏GT=%d 救回=%d 误改=%d" %
                         (group, intact["gt_pixels"], intact["rescue"], intact["harm"],
                          "N/A" if rate is None else "%.3f%%" % (100*rate),
                          destroyed["gt_pixels"], destroyed["rescue"], destroyed["harm"]))
    lines += ["", "A1→Q：逐事件前景mIoU差值（百分点）"]
    for event, pairs in result["per_event"].items():
        lines.append("  %s: %+.4f" % (event, pairs["a1_to_q"]["overall"]["foreground_delta_pp"]))
    lines += ["", "重点查看 a1_to_q 的 Intact误改，以及Destroyed救回；pixel_to_q用于单独分析Q汇总。", "分组：类别×Q混合类型见groups.csv；逐事件见events.csv；完整矩阵和分组计数见result.json。", "边缘/混合组使用事后GT，仅用于解释既有预测，不是部署规则或新模型效果。", "分组结果是相关性分析，不能单独证明因果或显著性。"]
    return "\n".join(lines) + "\n"


def write_tables(output, result):
    header = ["pair", "family", "group", "class", "gt_pixels", "rescue", "harm", "net_tp", "net_fp", "harm_fraction_of_old_correct", "rescue_fraction_of_old_errors"]
    with (output / "groups.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(header)
        for pair, record in result["pairs"].items():
            for family, groups in (("overall", {"all": record["overall"]}), ("q_groups", record["q_groups"]), ("edge_groups", record["edge_groups"])):
                for group, values in groups.items():
                    for name, c in values["classes"].items():
                        writer.writerow([pair, family, group, name, *[c[k] for k in header[4:]]])
    with (output / "events.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(["event", "pair", "old_foreground_miou_percent", "new_foreground_miou_percent", "delta_pp", "Intact_rescue", "Intact_harm", "Damaged_rescue", "Damaged_harm", "Destroyed_rescue", "Destroyed_harm"])
        for event, pairs in result["per_event"].items():
            for pair, record in pairs.items():
                m = record["overall"]
                writer.writerow([event, pair, 100*m["old"]["foreground_miou"], 100*m["new"]["foreground_miou"], m["foreground_delta_pp"],
                                 *[m["classes"][c][k] for c in CLASSES[1:] for k in ("rescue", "harm")]])


def run_audit(args):
    init_numpy()
    print("VERIFY formal Val450 summary, report and protocol; no model/CUDA import", flush=True)
    source = load_source(args.source, args.repo)
    require("torch" not in sys.modules, "CPU archive audit unexpectedly imported Torch")
    output = protected_output(args.output, source)
    output.mkdir(parents=True)
    n = args.limit or 450
    require(1 <= n <= 450, "--limit must be 0 or1..450")
    require(0 <= args.boundary_radius <= 7, "--boundary-radius must be 0..7")
    summary = dict(format=FORMAT, complete=False, diagnostic_complete=False, full_validation_evidence=False,
                   training=False, test_accessed=False, cuda_context_created=False, new_inference=False,
                   source=str(source["root"]), image_count=0, requested_images=n, error=None)
    write_json(output / "summary.json", summary)
    started = time.monotonic()
    try:
        totals, events, artifact_stats, artifacts = empty_counts(), {}, {}, {}
        inventory = {k: dict(regions=0, labelled_grid_mass=0.0) for k in Q_GROUPS}
        with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, row in enumerate(source["rows"][:n], 1):
                file = inside(source["root"], row["prediction_file"])
                before = signature(file)
                data = file.read_bytes()
                require(hashlib.sha256(data).hexdigest() == row["prediction_sha256"], "Prediction SHA256 mismatch: " + str(file))
                with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                    gt = label_map(archive["target"], "target", True)
                    predictions = {a: label_map(archive["prediction_" + a], a) for a in ARMS}
                    require(all(p.shape == gt.shape for p in predictions.values()), "Prediction shape mismatch")
                    for a in ARMS:
                        require(np.array_equal(confusion(gt, predictions[a]), matrix(row["confusion"][a], a)), "Saved per-image confusion mismatch: " + row["image"] + "/" + a)
                    qg, inv = q_groups(gt, archive)
                    edges = edge_groups(gt, args.boundary_radius)
                    counts = image_counts(gt, predictions, qg, edges)
                    add_counts(totals, counts)
                    add_counts(events.setdefault(row["event"], empty_counts()), counts)
                    for group in Q_GROUPS:
                        for key in inventory[group]: inventory[group][key] += inv[group][key]
                    journal.write(json.dumps(dict(index=index-1, image=row["image"], event=row["event"],
                        pairs={p: counts[p]["all"].tolist() for p in PAIRS},
                        q_group_pixels={g: int(((qg == i) & (gt != 255)).sum()) for i, g in enumerate(Q_GROUPS)},
                        edge_group_pixels={g: int(((edges == i) & (gt != 255)).sum()) for i, g in enumerate(EDGE_GROUPS)}), allow_nan=False) + "\n")
                require(signature(file) == before, "Prediction changed while being consumed")
                artifact_stats[str(file)] = before
                artifacts[str(file)] = row["prediction_sha256"]
                summary.update(phase="archive_analysis", image_count=index)
                write_json(output / "summary.json", summary)
                if index % 10 == 0 or index == n:
                    print("CPU paired Val audit %d/%d" % (index, n), flush=True)
        if n == 450:
            check_aggregate(totals, source["report"])
            require(set(events) == set(source["report"]["per_event"]), "Event set changed")
            for event, counts in events.items(): check_aggregate(counts, source["report"], event)
        for path, sha in source["source_files"].items():
            require(digest(path) == sha, "Source metadata/code changed during audit: " + path)
        require(all(signature(p) == tuple(s) for p, s in artifact_stats.items()), "Prediction archive changed during audit")
        result = dict(summary, phase="complete", complete=True, execution_complete=True,
            diagnostic_complete=n == 450, full_validation_evidence=n == 450, image_count=n,
            source_confusions_reproduced=n == 450, prediction_arrays_saved=False,
            pairs=summarise_counts(totals), per_event={e: summarise_counts(c) for e,c in sorted(events.items())},
            q_region_inventory=inventory, source_metadata_sha256=source["source_files"],
            consumed_prediction_sha256=artifacts, elapsed_seconds=time.monotonic()-started,
            definitions=dict(q_group_rule="GT-after-forward; foreground<0.2 background dominant;0.2<=fg<0.8 BG/FG mixed;fg>=0.8 split at foreground purity0.8",
                pixel_q_group="largest-Q route, first route wins ties; nearest integer grid expansion; descriptive association only, not actual bilinear decoder footprint",
                edge_rule="GT building/background 4-neighbor boundary, marked on both sides; Chebyshev dilation radius=%d pixels; ignore halo radius=%d is separate" % (args.boundary_radius, args.boundary_radius+1),
                units="pixels and Q regions; no building instance counts", gt_usage="post-forward diagnostic groups only; original predictions never changed",
                a1_to_pixel_caveat="decoding path contrast may include extra-classifier numerical differences; no native-logit replay claim",
                scope="archived Val diagnosis, not new training, Test evaluation, or proof of significance"))
        write_json(output / "result.json", result)
        write_tables(output, result)
        (output / "report.txt").write_text(report_text(result), encoding="utf-8")
        summary.update(phase="complete", complete=True, execution_complete=True, diagnostic_complete=n == 450,
                       full_validation_evidence=n == 450, source_confusions_reproduced=n == 450,
                       result_sha256=digest(output / "result.json"), elapsed_seconds=time.monotonic()-started)
        write_json(output / "summary.json", summary)
        print(report_text(result), flush=True)
        print("REPORT=" + str(output / "report.txt"), flush=True)
        return result
    except BaseException as error:
        summary.update(phase="failed", error=type(error).__name__ + ": " + str(error))
        write_json(output / "summary.json", summary)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--output-parent", default=PARENT)
    p.add_argument("--limit", type=int, default=0, help="0=full450; positive limits are incomplete diagnostics")
    p.add_argument("--boundary-radius", type=int, default=2)
    p.add_argument("--report", metavar="RESULT_DIRECTORY")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--output", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    require(0 <= a.limit <= 450 and 0 <= a.boundary_radius <= 7, "Invalid limit or boundary radius")
    if a.report:
        root = Path(a.report)
        if not (root / "summary.json").is_file():
            print("尚无summary.json；任务可能正在前置校验或已在校验时停止。查看启动日志：" + str(root) + ".log")
            return 1
        summary = read_json(root / "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary.get("complete") is True:
            require(digest(root / "result.json") == summary["result_sha256"], "Result changed")
            print(report_text(read_json(root / "result.json")))
        return 0
    source = Path(a.source).resolve(strict=True)
    require((source / "summary.json").is_file() and (source / "predictions").is_dir(), "Source summary/predictions unavailable")
    parent = Path(a.output_parent).expanduser().resolve()
    # Protect before even creating a lock or log in a user-supplied parent.
    source_summary = read_json(source / "summary.json")
    protocol = read_json(source_summary["protocol_path"])
    protected = (source, Path(a.repo).expanduser().resolve(),
                 Path(protocol["arguments"]["dataset"]).expanduser().resolve())
    require(all(not parent.is_relative_to(p) for p in protected), "Output parent is inside a protected source/code/dataset directory")
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / ("active_" + hashlib.sha256(str(source).encode()).hexdigest()[:24] + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("该来源已有CPU诊断在运行，本次不重复启动。")
        if a.worker:
            require(a.output, "Worker output required")
            run_audit(a)
            return 0
    fd, log = tempfile.mkstemp(prefix="run_%s_" % datetime.now().strftime("%Y%m%d_%H%M%S"), suffix=".log", dir=parent)
    output = log[:-4]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--source", str(source),
        "--repo", a.repo, "--output-parent", str(parent), "--output", output,
        "--limit", str(a.limit), "--boundary-radius", str(a.boundary_radius)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    with os.fdopen(fd, "w") as handle:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    info = dict(pid=child.pid, source=str(source), output=output, log=log,
                mode="CPU_existing_Val_predictions_only", started_is_not_passed=True)
    Path(output + ".launcher.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("已后台启动CPU诊断：读取已有Val预测；无训练、无GPU前向、无Test、无大数组输出。")
    print("日志：tail -n 60 -f '" + log + "'")
    print("报告：" + sys.executable + " '" + str(Path(__file__).resolve()) + "' --report '" + output + "'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
