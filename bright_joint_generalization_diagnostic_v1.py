#!/usr/bin/env python3
"""Read-only fixed-Train/Val diagnosis for the Q-rebased A/B/C/D run.

The only writes are under a new --output directory. This is a post-hoc
diagnostic, not another formal Val endpoint and never evaluates Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

STEPS = (500, 1500, 5000)
PATHS = ("grid_probq", "a", "b", "c", "d")
DEFAULT_ROOT = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D")
DEFAULT_TOOL = Path("/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2")


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return data


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def checkpoint_path(train_root: Path, step: int) -> Path:
    if step not in STEPS:
        raise ValueError("This diagnosis is fixed to 500, 1500 and 5000 steps")
    name = "checkpoint_last.pt" if step == 5000 else "checkpoint_%06d.pt" % step
    return train_root / "checkpoints" / name


def choose_seen_train(journal: Path, allowed: set[str], count: int = 450) -> tuple[list[str], dict]:
    if count < 1 or count > 500:
        raise ValueError("Fixed Train count must be within the first 500 updates")
    first = []
    all_seen = []
    with journal.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            expected_step = len(all_seen) + 1
            if row.get("step") != expected_step:
                raise ValueError("Train journal does not have contiguous 1..5000 steps")
            name = row.get("image")
            if not isinstance(name, str) or name not in allowed:
                raise ValueError("Train journal contains an image outside formal Train2918")
            all_seen.append(name)
            if len(first) < 500:
                first.append(name)
    if len(first) != 500 or len(set(first)) != 500:
        raise ValueError("Expected 500 distinct seen Train images by the 500-step checkpoint")
    if len(all_seen) != 5000:
        raise ValueError("Expected complete 5000-step Train journal")
    picked = first[:count]
    exposures = {}
    for step in STEPS:
        frequencies = {name: 0 for name in picked}
        for name in all_seen[:step]:
            if name in frequencies:
                frequencies[name] += 1
        values = list(frequencies.values())
        exposures[str(step)] = {"min": min(values), "mean": sum(values) / len(values),
                                "max": max(values),
                                "histogram": {str(v): values.count(v) for v in sorted(set(values))}}
    return picked, {"rule": "first %d unique images of contiguous steps 1..500" % count,
                    "first500_sha256": names_sha256(first),
                    "selected_sha256": names_sha256(picked), "selected_count": len(picked),
                    "selected_exposure_counts": exposures,
                    "scope_note": "Fixed subset of 450 previously seen Train images; not all Train2918"}


def reference_for_step(train_root: Path, step: int) -> tuple[dict, Path]:
    folder = train_root / ("final_val" if step == 5000 else "val_step_%06d" % step)
    report = read_json(folder / "result.json")
    if report.get("image_count") != 450 or report.get("execution_complete") is not True:
        raise ValueError("Original step %d did not process Val450" % step)
    if step == 5000 and (report.get("full_validation_evidence") is not True
                         or report.get("final_integrity_pending") is not False):
        raise ValueError("Original final Val450 lacks its formal integrity gate")
    if set(PATHS) - set(report.get("metrics", {})):
        raise ValueError("Original Val450 report lacks one or more paths")
    return report, folder


def validate_static(root: Path, tool_dir: Path, *, count: int) -> dict:
    root = root.expanduser().resolve()
    tool_dir = tool_dir.expanduser().resolve()
    train_root = root / "train"
    summary = read_json(train_root / "summary.json")
    workflow = read_json(root / "workflow_state.json")
    manifest = read_json(train_root / "manifest.json")
    if (summary.get("complete") is not True or summary.get("full_validation_evidence") is not True
            or summary.get("smoke") is not False or workflow.get("complete") is not True
            or workflow.get("full_validation_evidence") is not True):
        raise ValueError("The original A/B/C/D formal run is not complete")
    if len(manifest.get("train_names", [])) != 2918 or len(manifest.get("val_names", [])) != 450:
        raise ValueError("Original run does not bind Train2918/Val450")
    train_names, val_names = manifest["train_names"], manifest["val_names"]
    if len(set(train_names)) != 2918 or len(set(val_names)) != 450 or set(train_names) & set(val_names):
        raise ValueError("Original Train/Val identities repeat or overlap")
    selected, selection = choose_seen_train(train_root / "train.jsonl", set(train_names), count)
    checks = {}
    for step in STEPS:
        path = checkpoint_path(train_root, step)
        record = read_json(path.with_suffix(path.suffix + ".json"))
        if (record.get("step") != step or record.get("mode") != "train"
                or record.get("smoke") is not False or record.get("training_complete") is not (step == 5000)
                or file_sha256(path) != record.get("sha256")):
            raise ValueError("Checkpoint manifest/hash mismatch at step %d" % step)
        reference, folder = reference_for_step(train_root, step)
        checks[str(step)] = {"checkpoint": str(path), "sha256": record["sha256"],
                             "manifest": record, "reference_folder": str(folder),
                             "reference_sha256": file_sha256(folder / "result.json")}
    source_hashes = {row["manifest"].get("source_hash") for row in checks.values()}
    protocol_json = {json.dumps(row["manifest"].get("protocol_info"), sort_keys=True)
                     for row in checks.values()}
    if len(source_hashes) != 1 or len(protocol_json) != 1:
        raise ValueError("The three checkpoints do not share source/protocol bindings")
    if manifest.get("source_hash") != next(iter(source_hashes)):
        raise ValueError("Training manifest and checkpoints bind different source code")
    if not tool_dir.is_dir() or not (tool_dir / "run.py").is_file():
        raise FileNotFoundError("Original fix2 tool directory is missing: " + str(tool_dir))
    return {"root": str(root), "train_root": str(train_root), "tool_dir": str(tool_dir),
            "train_names": selected, "val_names": val_names, "selection": selection,
            "checkpoint_checks": checks, "source_hash": next(iter(source_hashes))}


def compare_iou(actual: dict, original: dict, *, tolerance_pp: float = 0.02) -> dict:
    """Compare all four class IoUs and the foreground-three mean in pp."""
    differences = {}
    for name in PATHS:
        new_row, old_row = actual[name], original[name]
        diffs = []
        for new, old in zip(new_row["iou"], old_row["iou"]):
            if (new is None) != (old is None):
                raise ValueError("Val class presence differs: " + name)
            diffs.append(None if new is None else 100.0 * (new - old))
        if len(diffs) != 4:
            raise ValueError("Val IoU vector must contain four classes")
        fg = 100.0 * (new_row["foreground_miou"] - old_row["foreground_miou"])
        maximum = max([abs(x) for x in diffs if x is not None] + [abs(fg)])
        new_cm, old_cm = new_row.get("confusion"), old_row.get("confusion")
        if new_cm is not None and old_cm is not None:
            if (len(new_cm) != 4 or len(old_cm) != 4
                    or any(len(row) != 4 for row in new_cm + old_cm)):
                raise ValueError("Val confusion must be 4x4")
            confusion_abs_delta = sum(abs(int(new_cm[i][j]) - int(old_cm[i][j]))
                                      for i in range(4) for j in range(4))
            confusion_exact = confusion_abs_delta == 0
        else:
            confusion_abs_delta = None
            confusion_exact = None
        differences[name] = {"class_iou_difference_pp": diffs,
                             "foreground_miou_difference_pp": fg,
                             "max_abs_difference_pp": maximum,
                             "confusion_exact": confusion_exact,
                             "sum_abs_confusion_cell_delta": confusion_abs_delta,
                             "passed": maximum <= tolerance_pp + 1e-12}
    return {"tolerance_pp": tolerance_pp, "paths": differences,
            "passed": all(row["passed"] for row in differences.values())}


def _original_modules(tool_dir: Path, source_hash: str):
    # Import only after the original package location and manifests are known.
    sys.path.insert(0, str(tool_dir))
    import torch  # noqa: PLC0415 - Torch/CUDA must not be initialized in --plan.
    import numpy as np
    import common as original_common
    import run as original_run
    import train as original_train
    from server_runtime import Runtime
    release = original_run.verify_release()
    if release["code_sha256"] != source_hash or original_common.source_hash() != source_hash:
        raise ValueError("Installed original fix2 package differs from training checkpoints")
    return torch, np, original_common, original_run, original_train, Runtime, release


def _loaded_checkpoints(static: dict, torch, common) -> dict:
    saved = {}
    for step in STEPS:
        check = static["checkpoint_checks"][str(step)]
        path = Path(check["checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = check["manifest"]
        if (payload.get("format") != common.FORMAT or payload.get("step") != step
                or payload.get("mode") != "train" or payload.get("source_hash") != static["source_hash"]
                or payload.get("source_hash") != record["source_hash"]
                or payload.get("protocol_info") != record["protocol_info"]
                or payload.get("updates_per_arm") != {arm: step for arm in common.ARMS}
                or set(payload.get("heads", {})) != set(common.ARMS)):
            raise ValueError("Checkpoint payload does not match its manifest at step %d" % step)
        saved[step] = payload
    first_config = saved[500]["config"]
    for step in STEPS[1:]:
        config = saved[step]["config"]
        for key in ("repo", "membership_root", "bank_root", "protocol", "device", "workers",
                    "seed", "key_name", "topk", "chunk_size", "values", "lr", "weight_decay"):
            if first_config.get(key) != config.get(key):
                raise ValueError("Training config changed between saved checkpoints: " + key)
    return saved


def _new_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _eval_split(runtime, heads, names, train_split, seed, step, output, original_train, torch, np):
    split = "fixed_train" if train_split else "val"
    cm = {name: np.zeros((4, 4), dtype=np.int64) for name in PATHS}
    loss_sum = {name: 0.0 for name in PATHS}
    nll_sum = {name: 0.0 for name in PATHS}
    lovasz_sum = {name: 0.0 for name in PATHS}
    valid_pixel_count = 0
    observed = []
    excluded_query_collisions = True
    query_support_counts = {"c": 0, "d": 0}
    started = time.monotonic()
    journal_path = output / ("%s_step_%06d.jsonl" % (split, step))
    with torch.no_grad(), journal_path.open("x", encoding="utf-8") as journal:
        for batch in runtime.batches(names, train=train_split, seed=seed):
            example = runtime.prepare(batch)
            original_train._assert_example(example)
            if example.image not in names or example.image in observed:
                raise ValueError("Evaluation produced an unknown or duplicate " + split + " image")
            if example.diagnostics.get("query_GT_used_for_supports") is not False:
                raise ValueError("Query GT support-separation assertion is absent")
            for arm in ("c", "d"):
                ids = example.supports[arm]["support_ids"]
                chosen = ids[ids >= 0]
                eid = example.diagnostics["event_id"]
                iid = example.diagnostics["image_id"]
                if ((eid >= 0 and bool((runtime.bank["events"][chosen] == eid).any()))
                        or (iid >= 0 and bool((runtime.bank["images"][chosen] == iid).any()))):
                    excluded_query_collisions = False
                    raise ValueError("Historical supports include the query image/event")
                query_support_counts[arm] += int(chosen.numel())
            target = example.target.detach().cpu().numpy()
            image_valid_pixels = int((example.target != 255).sum().item())
            valid_pixel_count += image_valid_pixels
            probability = {"grid_probq": example.common["base_probability_full"]}
            for arm in original_train.ARMS:
                probability[arm] = heads[arm](**example.common, **example.supports[arm])["probabilities"]
            row = {"image": example.image, "event": example.event,
                   "valid_pixels": image_valid_pixels, "loss": {}, "nll": {},
                   "lovasz": {}, "confusion": {}}
            for name in PATHS:
                value = probability[name]
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Non-finite predicted probability: " + name)
                loss = runtime.loss(value, example.target)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite original objective: " + name)
                nll = (runtime.deps.region_nll(value, example.target)
                       if image_valid_pixels else value.sum() * 0)
                lovasz = loss - nll
                if not bool(torch.isfinite(nll)) or not bool(torch.isfinite(lovasz)):
                    raise ValueError("Non-finite loss component: " + name)
                pred = value.argmax(1).detach().cpu().numpy()
                image_cm = original_train._confusion(target, pred)
                cm[name] += image_cm
                loss_sum[name] += float(loss)
                nll_sum[name] += float(nll)
                lovasz_sum[name] += float(lovasz)
                row["loss"][name] = float(loss)
                row["nll"][name] = float(nll)
                row["lovasz"][name] = float(lovasz)
                row["confusion"][name] = image_cm.tolist()
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            observed.append(example.image)
            if len(observed) % 10 == 0 or len(observed) == len(names):
                print("DIAGNOSTIC step=%d split=%s %d/%d" % (step, split, len(observed), len(names)), flush=True)
    if len(observed) != len(names) or set(observed) != set(names):
        raise ValueError("Evaluation did not finish the exact fixed " + split + " set")
    rows = {name: {"mean_image_loss": loss_sum[name] / len(names),
                   "mean_image_nll": nll_sum[name] / len(names),
                   "mean_image_lovasz": lovasz_sum[name] / len(names),
                   "metrics": original_train._metrics(cm[name])} for name in PATHS}
    return {"step": step, "split": split, "image_count": len(observed),
            "names_sha256": names_sha256(sorted(observed)), "order_sha256": names_sha256(observed),
            "loss_definition": "original region_nll + lovasz_softmax(classes=present, ignore=255); arithmetic mean over images",
            "valid_pixel_count": valid_pixel_count,
            "metrics": rows, "journal": str(journal_path),
            "same_image_event_exclusion_passed": excluded_query_collisions,
            "historical_parent_selections": query_support_counts,
            "journal_sha256": file_sha256(journal_path), "elapsed_seconds": time.monotonic() - started}


def print_report(results: dict):
    print("Step | Split | Path | mean loss (NLL + Lovasz) | foreground mIoU (%) | IoU BG/Intact/Damaged/Destroyed (%)", flush=True)
    for step in sorted(results, key=int):
        for split in ("fixed_train", "val"):
            report = results[step][split]
            for path in PATHS:
                row = report["metrics"][path]
                iou = row["metrics"]["iou"]
                iou_text = "/".join("NA" if x is None else "%.4f" % (100*x) for x in iou)
                print("%s | %s | %s | %.6f (%.6f + %.6f) | %.4f | %s" %
                      (step, split, path, row["mean_image_loss"],
                       row["mean_image_nll"], row["mean_image_lovasz"],
                       100*row["metrics"]["foreground_miou"], iou_text), flush=True)
                print("CONFUSION step=%s split=%s path=%s GT_rows=BG/Intact/Damaged/Destroyed "
                      "pred_columns=BG/Intact/Damaged/Destroyed matrix=%s" %
                      (step, split, path,
                       json.dumps(row["metrics"]["confusion"], separators=(",", ":"))),
                      flush=True)
        replay = results[step].get("val_reproduction", {})
        if replay.get("passed") is not None:
            maximum = max(row["max_abs_difference_pp"] for row in replay["paths"].values())
            exact = all(row["confusion_exact"] is True for row in replay["paths"].values())
            print("VAL REPLAY step=%s passed=%s max_abs_IoU_delta=%.6f pp exact_confusion=%s" %
                  (step, replay["passed"], maximum, exact), flush=True)


def report_saved(output: Path):
    path = output.expanduser().resolve() / "summary.json"
    state = read_json(path)
    print("诊断目录：", path.parent)
    print("完成：", state.get("complete"), "模式：", state.get("mode"),
          "错误：", state.get("error"))
    results = state.get("results", {})
    if results:
        print_report(results)
    else:
        print("尚无完整单步报告；查看后台日志或 traceback.txt。")
    return state


def execute(args):
    static = validate_static(args.train_root, args.tool_dir, count=450)
    if args.plan:
        info = {"mode": "plan_only", "read_only": True, "original_root": static["root"],
                "tool_dir": static["tool_dir"], "fixed_train_selection": static["selection"],
                "val_count": 450, "planned_smoke_steps": [500],
                "planned_full_steps": list(STEPS), "no_test": True, "no_training": True,
                "checkpoint_sha256": {s: r["sha256"] for s,r in static["checkpoint_checks"].items()},
                "note": "Train selection is post-hoc; only the 5000-step original Val is a formal endpoint."}
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    steps = STEPS if args.run else (500,)
    check = static["checkpoint_checks"]["500"]["manifest"]
    execution = check.get("protocol_info", {}).get("execution_settings", {})
    blocking = execution.get("cuda_launch_blocking")
    if blocking is not None:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    torch, np, common, original_run, original_train, Runtime, release = _original_modules(
        Path(static["tool_dir"]), static["source_hash"])
    saved = _loaded_checkpoints(static, torch, common)
    config = dict(saved[500]["config"])
    config.update(command="evaluate", mode="train", smoke=False, baseline_policy="independent",
                  baseline_root=None, baseline_evidence=None, allow_smoke_values=False)
    runtime_args = SimpleNamespace(**config)
    output = _new_output(args.output)
    runtime_args.output = str(output)
    state = {"format": "BRIGHT_JOINT_GENERALIZATION_DIAGNOSTIC_V1", "complete": False,
             "mode": "smoke" if args.smoke else "run", "formal_endpoint": False,
             "training": False, "test_accessed": False, "original_root": static["root"],
             "original_source_hash": static["source_hash"], "original_release": release,
             "fixed_train_selection": static["selection"],
             "fixed_train_names": static["train_names"],
             "val_names_sha256": names_sha256(static["val_names"]),
             "steps": list(steps), "results": {}}
    common.write_json(output / "summary.json", state)
    try:
        original_run.check_inputs(runtime_args)
        with Runtime(runtime_args, output) as runtime:
            if runtime.protocol_info != saved[500]["protocol_info"]:
                raise ValueError("Runtime input/numeric protocol differs from original checkpoints")
            if runtime.train_names != read_json(Path(static["train_root"]) / "manifest.json")["train_names"]:
                raise ValueError("Runtime Train split differs from the original run")
            if runtime.val_names != static["val_names"]:
                raise ValueError("Runtime Val split differs from the original run")
            original_train._checked_verify(runtime)
            original_train._seed(int(runtime_args.seed))
            heads, _ = original_train._new_heads(runtime_args, runtime.device)
            selected_train = static["train_names"][:2] if args.smoke else static["train_names"]
            selected_val = static["val_names"][:2] if args.smoke else static["val_names"]
            for step in steps:
                payload = saved[step]
                ckpt_manifest = static["checkpoint_checks"][str(step)]["manifest"]
                if runtime.protocol_info != payload["protocol_info"]:
                    raise ValueError("Runtime protocol changed between saved checkpoints")
                for arm in common.ARMS:
                    heads[arm].load_state_dict(payload["heads"][arm], strict=True)
                    if common.module_hash(heads[arm]) != ckpt_manifest["head_hashes"][arm]:
                        raise ValueError("Loaded head differs from checkpoint manifest: " + arm)
                    heads[arm].eval()
                train_result = _eval_split(runtime, heads, selected_train, True,
                                           int(runtime_args.seed), step, output, original_train, torch, np)
                val_result = _eval_split(runtime, heads, selected_val, False,
                                         int(runtime_args.seed), step, output, original_train, torch, np)
                for source_path, observed_hash in runtime.input_ledger.items():
                    original_hash = payload.get("input_file_sha256", {}).get(str(source_path))
                    if original_hash != observed_hash:
                        raise ValueError("Input is absent/changed against saved checkpoint %d: %s" %
                                         (step, source_path))
                integrity = original_train._checked_verify(runtime)
                reference, folder = reference_for_step(Path(static["train_root"]), step)
                if args.smoke:
                    # This is a two-image numerical preview, not a Val450 reproduction gate.
                    val_check = {"checked": False, "reason": "smoke has only two Val images"}
                else:
                    actual = {p: val_result["metrics"][p]["metrics"] for p in PATHS}
                    val_check = compare_iou(actual, reference["metrics"])
                    if not val_check["passed"]:
                        raise ValueError("Recomputed Val450 differs from original by over 0.02 pp at step %d" % step)
                state["results"][str(step)] = {"fixed_train": train_result, "val": val_result,
                                               "original_val_result_sha256": file_sha256(folder / "result.json"),
                                               "val_reproduction": val_check, "integrity": integrity,
                                               "checkpoint_sha256": static["checkpoint_checks"][str(step)]["sha256"]}
                common.write_json(output / "summary.json", state)
                print_report({str(step): state["results"][str(step)]})
        if runtime.guard_report and (runtime.guard_report.get("restored") is not True
                                     or runtime.guard_report.get("restore_conflicts")):
            raise RuntimeError("Frozen pooling guard did not restore cleanly")
        state.update(complete=True, execution_complete=True,
                     fixed_train_count=len(selected_train), val_count=len(selected_val),
                     validation_reproduced=not args.smoke,
                     guard_report=runtime.guard_report,
                     interpretation="Loss/IoU trends are diagnostic; a single seed and post-hoc sample do not prove overfitting or stable gains.")
        common.write_json(output / "summary.json", state)
        print("DIAGNOSTIC COMPLETE; no training, no Test. Report:", output / "summary.json", flush=True)
        return state
    except BaseException as error:
        state.update(complete=False, execution_complete=False,
                     error=type(error).__name__ + ": " + str(error))
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        common.write_json(output / "summary.json", state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="Only inspect saved JSON, journals and checkpoint hashes")
    group.add_argument("--smoke", action="store_true", help="Two seen Train + two Val images, step 500 only")
    group.add_argument("--run", action="store_true", help="Fixed Train450 + Val450 for steps 500, 1500 and 5000")
    group.add_argument("--report", action="store_true", help="Print a saved diagnosis without Torch or CUDA")
    p.add_argument("--train-root", type=Path, default=DEFAULT_ROOT,
                   help="Parent four-arm run root, which contains the train/ directory")
    p.add_argument("--tool-dir", type=Path, default=DEFAULT_TOOL,
                   help="Original unmodified fix2 tool directory")
    p.add_argument("--output", type=Path, help="New diagnosis directory; required for --smoke or --run")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.report:
        if args.output is None:
            parser().error("--report requires --output pointing to an existing diagnosis directory")
        return report_saved(args.output)
    if not args.plan and args.output is None:
        parser().error("--output is required for --smoke or --run")
    if args.plan and args.output is not None:
        parser().error("--plan does not write an output directory; omit --output")
    return execute(args)


if __name__ == "__main__":
    main()
