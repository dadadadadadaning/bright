"""Source/package integrity and canonical split helpers, adapted from Test v1."""
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


