"""One-time, read-only Test878 evaluation of a completed Q-rebased A/B/C/D run.

The training package is imported, hash checked and left untouched.  Test is
opened only after all four 5000-step heads and saved Val450 evidence pass the
formal gates.  A single frozen forward supplies grid_probq and all four heads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

FORMAT = "SRAC_SOFT_MEMORY_Q_REBASED_LOCKED_TEST_V1"
ARMS = ("grid_probq", "a", "b", "c", "d")
JOINT_ARMS = ARMS[1:]
STAGES = ("preflight", "parent_smoke", "build_smoke", "smoke", "parent_full", "build", "train")
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")


def read_json(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def all_true(value):
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def verify_release():
    root = Path(__file__).resolve().parent
    manifest = read_json(root / "SHA256SUMS.json")
    require(bool(manifest), "Empty Test package hash manifest")
    for name, sha in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name
                and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
                "Unsafe Test package manifest entry")
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root)
                and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                "Test package file changed: " + name)
    require({p.name for p in root.glob("*.py")} <= set(manifest),
            "Unbound Test package Python file")
    return hashlib.sha256((root / "SHA256SUMS.json").read_bytes()).hexdigest()


def _verify_tool_before_import(root, train_root):
    """Hash the released training files before executing any of their code."""
    manifest_path = root / "SHA256SUMS.json"
    hashes = read_json(manifest_path)
    require(bool(hashes), "Empty fix2 release hash manifest")
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name
                and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Unsafe fix2 release manifest entry")
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root)
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "Fix2 release file changed before import: " + name)
    require({p.name for p in root.glob("*.py")} <= set(hashes),
            "Unbound fix2 Python file before import")
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        if not path.name.startswith("test_"):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    observed = h.hexdigest()
    if train_root is not None:
        state = read_json(Path(train_root) / "workflow_state.json")
        require(state.get("code_sha256") == observed,
                "Current fix2 code differs from the saved training workflow before import")
    return observed


def state_hash(state):
    """Match the released training package's module_hash without a live model."""
    h = hashlib.sha256()
    for name, value in state.items():
        require(isinstance(name, str) and torch.is_tensor(value), "Invalid head state tensor")
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _load_tool(path, train_root=None):
    """Load only the exact provided fix2 package, never a shadowed module."""
    root = Path(path).expanduser().resolve(strict=True)
    require(root.is_dir() and root.name == "soft_memory_q_rebased_v1_fix2",
            "Expected the released fix2 training package directory")
    observed = _verify_tool_before_import(root, train_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    names = ("run", "common", "train", "server_runtime", "model")
    modules = {name: importlib.import_module(name) for name in names}
    for name, module in modules.items():
        require(Path(module.__file__).resolve().parent == root,
                "Training module import escaped supplied fix2 directory: " + name)
    release = modules["run"].verify_release()
    require(release.get("code_sha256") == observed,
            "Imported fix2 code differs from pre-import raw code hash")
    return SimpleNamespace(root=root, release=release, **modules)


def _read_journal(path, count):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    require(len(rows) == count and all(isinstance(row, dict) for row in rows),
            "Saved Val450 journal is incomplete")
    return rows


def _verify_val_predictions(tool, folder, result, names):
    rows = _read_journal(folder / "images.jsonl", 450)
    confusion = {arm: np.asarray(result["metrics"][arm]["confusion"], dtype=np.int64)
                 for arm in ARMS}
    transitions = {arm: np.asarray(result["transitions"][arm]["counts"], dtype=np.int64)
                   for arm in JOINT_ARMS}
    tool.train._reopen_predictions(folder, rows, confusion, transitions, names)
    require(result.get("saved_predictions_reopened") is True
            and result.get("prediction_evidence_complete") is True,
            "Final Val450 was not saved and independently reopened")


def _val_selection(metrics):
    require(isinstance(metrics, dict) and set(ARMS) <= set(metrics),
            "Final Val450 lacks a fixed prediction path")
    values = {}
    for arm in ARMS:
        item = metrics[arm]
        require(isinstance(item, dict) and type(item.get("foreground_miou")) in (int, float)
                and np.isfinite(item["foreground_miou"])
                and 0 <= item["foreground_miou"] <= 1,
                "Invalid final Val450 foreground mIoU: " + arm)
        values[arm] = float(item["foreground_miou"])
    # Stable tie order is declared before reading Test. Test cannot change this.
    return dict(selected=max(ARMS, key=lambda arm: values[arm]),
                criterion="largest final-step Val450 foreground three-class mIoU; ties: grid_probq,a,b,c,d",
                tie_order=list(ARMS), val_foreground_miou=values)


def validate_formal(train_root, tool):
    """Gate all saved training evidence before Test list/GT is accessed."""
    root = Path(train_root).expanduser().resolve(strict=True)
    train = root / "train"
    state = read_json(root / "workflow_state.json")
    summary = read_json(train / "summary.json")
    manifest = read_json(train / "manifest.json")
    val = read_json(train / "final_val" / "result.json")
    release_hash = tool.common.source_hash()
    require(state.get("format") == tool.common.FORMAT
            and state.get("complete") is True
            and state.get("execution_complete") is True
            and state.get("training_complete") is True
            and state.get("full_validation_evidence") is True
            and state.get("current_stage") == "complete"
            and state.get("smoke") is False
            and state.get("test_accessed") is False
            and state.get("code_sha256") == release_hash
            and not state.get("error"),
            "Q-rebased full workflow has not passed formal final gates")
    stages = state.get("stages", {})
    require(all(stages.get(stage, {}).get("status") == "complete" for stage in STAGES),
            "A formal source stage is incomplete")
    require(stages["train"].get("summary_sha256") == tool.common.digest(train / "summary.json"),
            "Workflow's final train-summary hash differs")
    require(summary.get("format") == tool.common.FORMAT
            and summary.get("complete") is True
            and summary.get("execution_complete") is True
            and summary.get("training_complete") is True
            and summary.get("full_validation_evidence") is True
            and summary.get("phase") == "complete"
            and summary.get("image_count") == 450
            and summary.get("smoke") is False
            and summary.get("test_accessed") is False
            and summary.get("updates_per_arm") == {a: 5000 for a in JOINT_ARMS}
            and all_true(summary.get("integrity"))
            and summary.get("old_three_arm_baseline") == "not_comparable_new_Q_source"
            and not summary.get("error"),
            "A/B/C/D final Train/Val450 summary is incomplete or failed")
    require(manifest.get("format") == tool.common.FORMAT
            and manifest.get("source_hash") == release_hash
            and len(manifest.get("train_names", [])) == 2918
            and len(manifest.get("val_names", [])) == 450
            and len(set(manifest["train_names"])) == 2918
            and len(set(manifest["val_names"])) == 450
            and not (set(manifest["train_names"]) & set(manifest["val_names"]))
            and manifest.get("old_three_arm_baseline") == "not_comparable_new_Q_source",
            "Formal training manifest/splits/code do not match")
    require(val.get("format") == tool.common.FORMAT
            and val.get("phase") == "evaluation"
            and val.get("execution_complete") is True
            and val.get("whole_val450_processed") is True
            and val.get("image_count") == 450
            and val.get("primary_endpoint") is True
            and val.get("full_validation_evidence") is True
            and val.get("final_integrity_pending") is False
            and val.get("test_accessed") is False
            and val.get("smoke") is False
            and all_true(val.get("final_integrity"))
            and val.get("final_integrity") == summary.get("integrity")
            and val.get("metrics") == summary.get("metrics"),
            "Saved last-step Val450 endpoint is incomplete or differs from training")
    guard = summary.get("guard_report")
    require(isinstance(guard, dict) and guard.get("restored") is True
            and not guard.get("restore_conflicts")
            and val.get("guard_report") == guard,
            "Final training PoolGuard did not restore cleanly")
    checkpoint = (train / "checkpoints" / "checkpoint_last.pt").resolve(strict=True)
    checkpoint_manifest = read_json(checkpoint.with_suffix(checkpoint.suffix + ".json"))
    require(Path(summary.get("last_checkpoint", "")).resolve() == checkpoint
            and summary.get("checkpoint") == checkpoint_manifest
            and checkpoint_manifest.get("format") == tool.common.FORMAT
            and checkpoint_manifest.get("file") == checkpoint.name
            and checkpoint_manifest.get("sha256") == tool.common.digest(checkpoint)
            and checkpoint_manifest.get("step") == 5000
            and checkpoint_manifest.get("mode") == "train"
            and checkpoint_manifest.get("smoke") is False
            and checkpoint_manifest.get("training_complete") is True
            and checkpoint_manifest.get("source_hash") == release_hash
            and checkpoint_manifest.get("protocol_info") == manifest.get("protocol_info"),
            "Last-step checkpoint manifest/hash does not match final training")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(saved.get("format") == tool.common.FORMAT
            and saved.get("step") == 5000
            and saved.get("mode") == "train"
            and saved.get("smoke") is False
            and saved.get("training_complete") is True
            and saved.get("test_accessed") is False
            and saved.get("updates_per_arm") == {a: 5000 for a in JOINT_ARMS}
            and saved.get("source_hash") == release_hash
            and saved.get("protocol_info") == manifest.get("protocol_info")
            and saved.get("old_three_arm_baseline") == "not_comparable_new_Q_source"
            and set(saved.get("heads", {})) == set(JOINT_ARMS)
            and saved.get("config") == summary.get("arguments")
            and all(state_hash(saved["heads"][a]) ==
                    checkpoint_manifest.get("head_hashes", {}).get(a) for a in JOINT_ARMS),
            "Serialized final checkpoint has wrong steps, model or source binding")
    fingerprints = train / "input_fingerprints.json"
    require(summary.get("input_fingerprints_sha256") == tool.common.digest(fingerprints),
            "Final source-file ledger changed")
    ledger = read_json(fingerprints)
    require(bool(ledger) and saved.get("input_file_sha256") and
            all(ledger.get(path) == sha for path, sha in saved["input_file_sha256"].items()),
            "Saved checkpoint input ledger conflicts with final source ledger")
    for index, (raw_path, sha) in enumerate(ledger.items(), 1):
        path = Path(raw_path)
        require(path.is_file() and tool.common.digest(path) == sha,
                "Training source image/artifact changed: " + raw_path)
        if index % 500 == 0:
            print("Train/Val source file hashes verified %d/%d" % (index, len(ledger)), flush=True)
    _verify_val_predictions(tool, train / "final_val", val, manifest["val_names"])
    selection = _val_selection(val["metrics"])
    return SimpleNamespace(root=root, train=train, state=state, summary=summary,
        manifest=manifest, val=val, checkpoint=checkpoint, checkpoint_manifest=checkpoint_manifest,
        saved=saved, selected=selection,
        hashes={"workflow_state": tool.common.digest(root / "workflow_state.json"),
                "train_manifest": tool.common.digest(train / "manifest.json"),
                "train_summary": tool.common.digest(train / "summary.json"),
                "val_result": tool.common.digest(train / "final_val" / "result.json"),
                "checkpoint": tool.common.digest(checkpoint),
                "checkpoint_manifest": tool.common.digest(checkpoint.with_suffix(checkpoint.suffix + ".json")),
                "input_fingerprints": tool.common.digest(fingerprints)})


def _canonical_test_split(source, formal):
    protocol = read_json(source)
    require(protocol.get("format") == "SOFT_MEMORY_Q_SOURCE_V1", "Wrong Q source protocol")
    e0 = read_json(protocol["e0_metadata_protocol"])
    split_paths = e0["input_paths"]
    train_list, val_list = (Path(split_paths[name]).resolve(strict=True)
                            for name in ("train_list", "val_list"))
    require(train_list.name == "train_set.txt" and val_list.name == "val_set.txt"
            and train_list.parent == val_list.parent
            and train_list.parent.name == "standard_ML",
            "Require canonical standard_ML split directory")
    for path, expected in ((train_list, formal.manifest["train_names"]),
                           (val_list, formal.manifest["val_names"])):
        names = [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
        require(names == expected, "Current Train/Val list differs from final training: " + str(path))
    test_list = train_list.parent / "test_set.txt"
    names = [row.strip() for row in test_list.read_text(encoding="utf-8").splitlines() if row.strip()]
    require(len(names) == 878 and len(set(names)) == 878
            and all(SAFE_NAME.fullmatch(name) for name in names)
            and not (set(names) & (set(formal.manifest["train_names"]) |
                                   set(formal.manifest["val_names"]))),
            "Canonical Test878 is incomplete, unsafe, repeated or overlaps Train/Val")
    dataset = Path(e0["arguments"]["dataset"]).resolve(strict=True)
    for index, name in enumerate(names, 1):
        for subdir, suffix in (("pre-event", "_pre_disaster.tif"),
                               ("post-event", "_post_disaster.tif"),
                               ("target", "_building_damage.tif")):
            path = dataset / subdir / (name + suffix)
            require(path.is_file(), "Scored Test needs RGB, SAR and ground truth: " + str(path))
        if index % 200 == 0:
            print("Test file existence checked %d/878" % index, flush=True)
    return test_list, names, dataset


def _protect_output(output, formal, tool):
    """Never write Test artifacts into source, training or the released tool."""
    source = read_json(formal.saved["config"]["protocol"])
    e0 = read_json(source["e0_metadata_protocol"])
    dataset = Path(e0["arguments"]["dataset"]).expanduser().resolve(strict=True)
    protected = (formal.root, formal.train, tool.root, dataset)
    path = Path(output).expanduser().resolve()
    require(not path.exists(), "Use a fresh Test output directory; preserving: " + str(path))
    require(all(not path.is_relative_to(root) for root in protected),
            "Test output must be outside the training run, released tool and dataset")
    return path


def metrics(cm, inherited):
    base = inherited._metrics(cm)
    iou = base["iou"]
    base["four_class_miou"] = sum(x if x is not None else 0.0 for x in iou) / 4
    base["four_class_average"] = "fixed classes 0/1/2/3; absent IoU contributes zero"
    return base


def _reopen(folder, rows, expected, event_expected, names, tool):
    rebuilt = {arm: np.zeros((4, 4), dtype=np.int64) for arm in ARMS}
    events = {}
    require(len(rows) == len(names), "Incomplete Test878 prediction journal")
    for index, row in enumerate(rows):
        require(row.get("index") == index and row.get("image") == names[index]
                and row.get("file") == "predictions/%04d.npz" % index,
                "Saved Test prediction order/path differs from the locked list")
        path = folder / row["file"]
        require(path.is_file() and tool.common.digest(path) == row.get("sha256"),
                "Saved Test prediction file changed: " + str(path))
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"target"} | {"prediction_" + a for a in ARMS},
                    "Saved Test prediction schema differs")
            target = archive["target"]
            require(target.dtype == np.uint8, "Saved Test target dtype differs")
            event = events.setdefault(row["event"],
                {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS})
            for arm in ARMS:
                prediction = archive["prediction_" + arm]
                require(prediction.dtype == np.uint8, "Saved Test prediction dtype differs")
                cm = tool.train._confusion(target, prediction)
                require(cm.tolist() == row["confusion"][arm],
                        "Saved Test per-image confusion differs: " + arm)
                rebuilt[arm] += cm
                event[arm] += cm
    for arm in ARMS:
        require(np.array_equal(rebuilt[arm], expected[arm]),
                "Reopened Test878 confusion differs: " + arm)
    for event, arms in event_expected.items():
        require(event in events and all(np.array_equal(events[event][a], arms[a]) for a in ARMS),
                "Reopened Test878 event confusion differs: " + event)
    require(set(events) == set(event_expected), "Test event list changed on reopen")
    return rebuilt, events


def evaluate(formal, tool, output):
    output = _protect_output(output, formal, tool)
    output.mkdir(parents=True)
    write = tool.common.write_json
    summary = dict(format=FORMAT, complete=False, full_test_evidence=False,
                   training=False, test_accessed=False, selected_on_val=formal.selected,
                   source_run=str(formal.root), source_hashes=formal.hashes,
                   training_package_release=tool.release,
                   test_package_manifest_sha256=verify_release())
    write(output / "summary.json", summary)
    started = time.monotonic()
    try:
        # This Val-frozen selection is written before the first Test list read.
        write(output / "val_locked_selection.json", formal.selected)
        summary["val_locked_selection_sha256"] = tool.common.digest(output / "val_locked_selection.json")
        summary["test_accessed"] = True
        write(output / "summary.json", summary)
        source = formal.saved["config"]["protocol"]
        test_list, names, dataset = _canonical_test_split(source, formal)
        summary.update(test_list=str(test_list), test_list_sha256=tool.common.digest(test_list),
                       test_image_count_expected=878, dataset=str(dataset), phase="inference")
        write(output / "summary.json", summary)
        args = SimpleNamespace(**formal.saved["config"])
        args.smoke = False
        args.mode = "train"  # strict full-bank gate; no optimizer is constructed.
        args.output = str(output)
        require(args.topk == 64 and args.key_name in ("t128", "f256")
                and args.values == str(formal.root / "values")
                and args.bank_root == str(formal.root / "parent")
                and args.protocol == formal.state["arguments"]["protocol"],
                "Final checkpoint does not bind this complete parent/child bank")
        parent_runtime = tool.server_runtime.Runtime
        class TestRuntime(parent_runtime):
            def __enter__(self):
                super().__enter__()
                try:
                    require(self.protocol_info == formal.saved["protocol_info"],
                            "Test source/bank/runtime differs from saved final checkpoint")
                    for name in names:
                        established = {key: Path(path).resolve() for key, path in
                            self.deps.sample_paths(self.bundle["args"].dataset, name).items()}
                        expected = {"pre": dataset / "pre-event" / (name + "_pre_disaster.tif"),
                                    "post": dataset / "post-event" / (name + "_post_disaster.tif"),
                                    "label": dataset / "target" / (name + "_building_damage.tif")}
                        require(established == expected,
                                "Test path differs from established BRIGHT labelled loader: " + name)
                    # Reuse the frozen forward/Reader verbatim. Its Val-only
                    # loader gate is retargeted only in this Test subclass,
                    # after the canonical disjoint Test878 list was checked.
                    self._val_set = set(names)
                    return self
                except BaseException:
                    self.close(*sys.exc_info())
                    raise
        runtime = TestRuntime(args, output)
        confusion = {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS}
        event_confusion = {}
        rows = []
        heads = {}
        output_predictions = output / "predictions"
        output_predictions.mkdir()
        with runtime, torch.no_grad(), (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            require(runtime.device.type == "cuda", "Formal Test requires the original CUDA numerical protocol")
            for arm in JOINT_ARMS:
                head = tool.model.JointRegionMemoryClassifier(chunk_size=args.chunk_size).to(runtime.device)
                head.load_state_dict(formal.saved["heads"][arm], strict=True)
                head.requires_grad_(False).eval()
                require(tool.common.module_hash(head) == formal.checkpoint_manifest["head_hashes"][arm],
                        "Loaded Test head differs from saved final checkpoint: " + arm)
                heads[arm] = head
            for index, batch in enumerate(runtime.batches(names, train=False, seed=args.seed)):
                example = runtime.prepare(batch)
                tool.train._assert_example(example)
                require(index < 878 and example.image == names[index],
                        "Test loader order differs from fixed Test878 split")
                target = example.target.detach().cpu().numpy().astype(np.uint8)
                base = example.common["base_probability_full"]
                require(bool(torch.isfinite(base).all()), "Nonfinite grid_probq on Test")
                predictions = {"grid_probq": base.argmax(1).cpu().numpy().astype(np.uint8)}
                for arm in JOINT_ARMS:
                    probability = heads[arm](**example.common, **example.supports[arm])["probabilities"]
                    require(bool(torch.isfinite(probability).all()), "Nonfinite Test head: " + arm)
                    predictions[arm] = probability.argmax(1).cpu().numpy().astype(np.uint8)
                per_image = {arm: tool.train._confusion(target, predictions[arm]) for arm in ARMS}
                event = event_confusion.setdefault(example.event,
                    {arm: np.zeros((4, 4), dtype=np.int64) for arm in ARMS})
                for arm in ARMS:
                    confusion[arm] += per_image[arm]
                    event[arm] += per_image[arm]
                relative = "predictions/%04d.npz" % index
                np.savez_compressed(output / relative, target=target,
                                    **{"prediction_" + arm: predictions[arm] for arm in ARMS})
                row = dict(index=index, image=example.image, event=example.event,
                           file=relative, sha256=tool.common.digest(output / relative),
                           confusion={arm: per_image[arm].tolist() for arm in ARMS},
                           diagnostics=example.diagnostics)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
                journal.flush()
                rows.append(row)
                if (index + 1) % 10 == 0 or index == 877:
                    print("Locked Test %d/878; final integrity pending" % (index + 1), flush=True)
            require(len(rows) == 878, "Test loader did not cover all 878 images")
            reopened, events = _reopen(output, rows, confusion, event_confusion, names, tool)
            integrity = tool.train._checked_verify(runtime)
            require(tool.common.source_hash() == formal.saved["source_hash"],
                    "Training code changed during Test")
            require(all(tool.common.module_hash(heads[a]) ==
                        formal.checkpoint_manifest["head_hashes"][a] for a in JOINT_ARMS),
                    "A Test head changed despite inference-only execution")
        guard = runtime.guard_report
        require(isinstance(guard, dict) and guard.get("restored") is True
                and not guard.get("restore_conflicts"), "Test PoolGuard did not restore cleanly")
        require(tool.common.digest(test_list) == summary["test_list_sha256"],
                "Test list changed during inference")
        require(verify_release() == summary["test_package_manifest_sha256"],
                "Test package changed during long inference")
        require(_verify_tool_before_import(tool.root, formal.root) == formal.saved["source_hash"],
                "Fix2 training code changed during long inference")
        input_fingerprints = {str(path): sha for path, sha in runtime.input_ledger.items()}
        write(output / "input_fingerprints.json", input_fingerprints)
        input_fingerprints_sha256 = tool.common.digest(output / "input_fingerprints.json")
        for label, path in (("workflow_state", formal.root / "workflow_state.json"),
                            ("train_manifest", formal.train / "manifest.json"),
                            ("train_summary", formal.train / "summary.json"),
                            ("val_result", formal.train / "final_val" / "result.json"),
                            ("checkpoint", formal.checkpoint),
                            ("checkpoint_manifest", formal.checkpoint.with_suffix(formal.checkpoint.suffix + ".json")),
                            ("input_fingerprints", formal.train / "input_fingerprints.json")):
            require(tool.common.digest(path) == formal.hashes[label],
                    "Formal Train/Val evidence changed during Test: " + label)
        result = dict(format=FORMAT, split="standard_ML/test_set.txt", image_count=878,
                      execution_complete=True, full_test_evidence=True, training=False,
                      test_accessed=True, no_test_based_selection=True,
                      selected_on_val=formal.selected,
                      test_list_sha256=summary["test_list_sha256"],
                      input_fingerprints_sha256=input_fingerprints_sha256,
                      source_hashes=formal.hashes,
                      test_package_manifest_sha256=summary["test_package_manifest_sha256"],
                      metrics={arm: metrics(reopened[arm], tool.train) for arm in ARMS},
                      per_event={event: {arm: metrics(cm, tool.train) for arm, cm in arms.items()}
                                 for event, arms in events.items()},
                      saved_predictions_reopened=True, runtime_integrity=integrity,
                      pool_guard=guard, elapsed_seconds=time.monotonic()-started)
        write(output / "result.json", result)
        summary.update(phase="complete", complete=True, full_test_evidence=True,
                       image_count=878, result_sha256=tool.common.digest(output / "result.json"),
                       input_fingerprints_sha256=input_fingerprints_sha256,
                       runtime_integrity=integrity, pool_guard=guard,
                       elapsed_seconds=time.monotonic()-started)
        write(output / "summary.json", summary)
        return result
    except BaseException as exc:
        summary.update(complete=False, full_test_evidence=False,
                       error=type(exc).__name__ + ": " + str(exc))
        write(output / "summary.json", summary)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "report"))
    p.add_argument("--train-root", required=False)
    p.add_argument("--tool", default="/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2")
    p.add_argument("--output")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    own_release = verify_release()
    if args.command == "report":
        require(args.output, "--output is required for report")
        folder = Path(args.output)
        summary = read_json(folder / "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary.get("complete") is True:
            result = read_json(folder / "result.json")
            for arm in ARMS:
                score = result["metrics"][arm]
                print("%s: foreground3=%.4f%% fourclass=%.4f%% IoU=%s" %
                      (arm, 100*score["foreground_miou"], 100*score["four_class_miou"],
                       " / ".join("%.4f" % (100*x) if x is not None else "NA"
                                  for x in score["iou"])))
            print("Val-selected path:", result["selected_on_val"]["selected"])
        return
    require(args.train_root, "--train-root is required")
    tool = _load_tool(args.tool, args.train_root)
    formal = validate_formal(args.train_root, tool)
    if args.command == "plan":
        print(json.dumps(dict(format=FORMAT, stage="ready_no_Test_access",
                              full_train_val450_verified=True, train_root=str(formal.root),
                              selected_on_val=formal.selected,
                              checkpoint_sha256=formal.hashes["checkpoint"],
                              test_package_manifest_sha256=own_release,
                              test_accessed=False), ensure_ascii=False, indent=2))
        return
    require(args.output, "--output is required for run")
    result = evaluate(formal, tool, args.output)
    print("LOCKED TEST878 COMPLETE. No training. Selected on Val:",
          result["selected_on_val"]["selected"])


if __name__ == "__main__":
    main()
