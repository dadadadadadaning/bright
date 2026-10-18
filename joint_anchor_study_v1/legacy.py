"""Read-only bridge to pinned, historical BRIGHT implementations.

No model, slot, retrieval, bank, or supervised-loss implementation lives here.
Preflight inspects files without loading checkpoints or creating CUDA tensors.
The original runtime retains its complete source/bank/data integrity checks.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace


DEPENDENCY_MANIFEST_SHA256 = "1876bbc147e8f98b09406a8c42a0dd8e564d18e05d705cf73d9d27fce86221a0"
EXPECTED_SOURCE = "b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142"
_FRESH_REQUIRED = {"repo", "protocol", "membership_root", "bank_root", "values"}
_FRESH_DEFAULTS = dict(device="cuda:0", workers=4, seed=3407, key_name="t128",
                       topk=64, chunk_size=32, lr=1e-4, weight_decay=0.005,
                       steps=5000, eval_every=500)
_BANK_FILES = ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl")


class MissingInputsError(FileNotFoundError):
    """All discoverable missing inputs, rather than one failure per launch."""
    def __init__(self, missing):
        self.missing = sorted(set(map(str, missing)))
        super().__init__("Missing required inputs; no training was started:\n" + "\n".join(self.missing))


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_named(path, name):
    """Register only a unique private name, never patch a historical bare name."""
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError("Private module name collision: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import verified dependency: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _package_files(path, *, expected_manifest=None, expected_source=None):
    path = Path(path).expanduser().resolve()
    manifest = path / "SHA256SUMS.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise MissingInputsError([manifest])
    manifest_sha = _digest(manifest)
    if expected_manifest and manifest_sha != expected_manifest:
        raise ValueError("Pinned dependency manifest mismatch: " + str(manifest))
    hashes = _read(manifest)
    if not hashes:
        raise ValueError("Empty package manifest: " + str(manifest))
    files = {str(manifest): manifest_sha}
    for name, sha in hashes.items():
        relative = Path(name)
        target = path / relative
        if (relative.is_absolute() or len(relative.parts) != 1 or name in (".", "..", "SHA256SUMS.json")
                or "\\" in name or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or target.is_symlink() or not target.is_file()):
            raise ValueError("Invalid or missing package artifact: " + str(target))
        if _digest(target) != sha:
            raise ValueError("Package artifact changed: " + str(target))
        files[str(target)] = sha
    unbound = {p.name for p in path.glob("*.py")} - set(hashes)
    if unbound:
        raise ValueError("Unbound Python files in pinned package: " + ", ".join(sorted(unbound)))
    if expected_source:
        source = hashlib.sha256()
        for item in sorted(path.glob("*.py")):
            if not item.name.startswith("test_"):
                source.update(item.name.encode()); source.update(item.read_bytes())
        if source.hexdigest() != expected_source:
            raise ValueError("Original runtime code hash differs from fixed fix2 release")
    return files


def load_legacy(path):
    """Load immutable loss-study helpers under unique names (Torch import only)."""
    root = Path(path).expanduser().resolve()
    files = _package_files(root, expected_manifest=DEPENDENCY_MANIFEST_SHA256)
    tag = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    modules = {key: _load_named(root / (filename + ".py"), "_bright_anchor_legacy_" + tag + "_" + key)
               for key, filename in (("study", "study"), ("state", "continuation"), ("objectives", "objectives"))}
    return SimpleNamespace(root=str(root), files=files, **modules)


def verify_dependencies(legacy):
    for path, expected in legacy.files.items():
        if not Path(path).is_file() or _digest(path) != expected:
            raise ValueError("Pinned helper changed: " + path)
    return True


def _fresh_config(path):
    if not path:
        raise ValueError("--start fresh requires an explicit --config JSON; no automatic source fallback")
    config = _read(path)
    unknown = set(config) - _FRESH_REQUIRED - set(_FRESH_DEFAULTS)
    missing = _FRESH_REQUIRED - set(config)
    if unknown or missing:
        raise ValueError("Fresh config keys: missing=" + str(sorted(missing)) + "; unsupported=" + str(sorted(unknown)))
    result = dict(_FRESH_DEFAULTS, **config)
    for key in _FRESH_REQUIRED:
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError("Fresh config needs explicit path: " + key)
        result[key] = str(Path(result[key]).expanduser().resolve())
    for key in ("workers", "seed", "topk", "chunk_size", "steps", "eval_every"):
        if type(result[key]) is not int:
            raise ValueError("Fresh integer config required: " + key)
    if (result["workers"] < 0 or result["seed"] < 0 or result["chunk_size"] < 1
            or result["topk"] != 64 or result["steps"] != 5000 or result["eval_every"] != 500
            or result["key_name"] != "t128" or not isinstance(result["device"], str)
            or not re.fullmatch(r"cuda:\d+", result["device"])):
        raise ValueError("Fresh configuration violates fixed Top64/T128/5000-horizon/500-evaluation settings")
    for key, expected in (("lr", 1e-4), ("weight_decay", 0.005)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != expected:
            raise ValueError("Fresh " + key + " must retain the declared original setting " + str(expected))
    result.update(command="train", mode="train", smoke=False, baseline_policy="independent",
                  baseline_root=None, baseline_evidence=None, allow_smoke_values=False)
    return result


def missing_inputs(config):
    """Inspect declared runtime paths and readable ledgers; never load tensors.

    Missing parent metadata prevents discovering its contents; the report lists
    that metadata as missing, without inventing paths or declaring completeness.
    Exact hashes and tensor semantics remain enforced by the original runtime.
    """
    missing = []
    def require(raw, *, directory=False):
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw).expanduser().resolve()
        if not (path.is_dir() if directory else path.is_file()):
            missing.append(str(path))
        return path
    def read_if(path):
        return _read(path) if path is not None and path.is_file() else {}
    def ledger_files(mapping, base=None):
        if not isinstance(mapping, dict):
            raise ValueError("Invalid file ledger")
        for name in mapping:
            if not isinstance(name, str):
                raise ValueError("File ledger key must be a path")
            relative = Path(name)
            if base is not None:
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe artifact ledger path: " + name)
                require(str(base / relative))
            else:
                require(name)
    repo = require(config.get("repo"), directory=True)
    if repo:
        source = repo / "bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3"
        for name in ("shared.py", "paired.py", "core.py"):
            require(str(source / name))
    source = read_if(require(config.get("protocol")))
    for field in ("q_checkpoint", "a1_checkpoint", "e0_metadata_protocol"):
        if source and not isinstance(source.get(field), str):
            raise ValueError("Source protocol lacks " + field)
        require(source.get(field))
    if source.get("q_checkpoint"):
        q_dir = Path(source["q_checkpoint"]).expanduser().resolve().parent
        for name in ("summary.json", "manifest.json"):
            require(str(q_dir / name))
        level = source.get("evidence_level")
        if level == "independent_val450":
            for name in ("summary.json", "result.json"):
                require(str(q_dir.parent / "val_baseline" / name))
        elif level == "training_final_val450":
            require(str(q_dir / "validation.jsonl"))
        else:
            raise ValueError("Unknown source evidence_level")
    e0_path = Path(source["e0_metadata_protocol"]) if source.get("e0_metadata_protocol") else None
    e0 = read_if(e0_path)
    if e0:
        require(e0.get("arguments", {}).get("dataset"), directory=True)
        for field in ("train_list", "val_list"):
            require(e0.get("input_paths", {}).get(field))
        # Deliberately do not require deleted historical E0/J123 weights:
        # this source uses E0 only for dataset/numerical metadata.
    for field in ("membership_root", "bank_root"):
        root = require(config.get(field), directory=True)
        if root:
            for name in _BANK_FILES:
                require(str(root / name))
            manifest = read_if(root / "manifest.json")
            ledger_files(manifest.get("source_image_fingerprints", {}))
            if manifest.get("archived_membership_root"):
                archive = Path(manifest["archived_membership_root"]).expanduser().resolve()
                if (archive / "keys").is_dir() and not (archive / "paired_bank.pt").is_file():
                    archive /= "keys"
                ledger_files(manifest.get("archived_files_sha256", {}), archive)
    values = require(config.get("values"), directory=True)
    if values:
        for name in ("manifest.json", "summary.json", "slot_bank.pt", "images.jsonl", "input_fingerprints.json"):
            require(str(values / name))
        manifest = read_if(values / "manifest.json")
        ledger_files(manifest.get("file_sha256", {}), values)
        ledger_files(read_if(values / "input_fingerprints.json"))
    return sorted(set(missing))


def prepare_source(args, legacy):
    """Prepare explicit checkpoint500 or fresh start; never load weights/CUDA."""
    verify_dependencies(legacy)
    start = getattr(args, "start", None)
    if start not in ("checkpoint500", "fresh"):
        raise ValueError("--start must explicitly be checkpoint500 or fresh")
    tool = Path(args.tool_dir).expanduser().resolve()
    if start == "fresh":
        path = getattr(args, "config", None)
        config = _fresh_config(path)
        missing = missing_inputs(config)
        if not (tool / "SHA256SUMS.json").is_file():
            missing.append(str(tool / "SHA256SUMS.json"))
        if missing:
            raise MissingInputsError(missing)
        tool_files = _package_files(tool, expected_source=EXPECTED_SOURCE)
        return dict(start=start, start_step=0, tool=str(tool), config=config,
                    source_sha256=EXPECTED_SOURCE, checkpoint=None, saved=None,
                    old_files={}, config_path=str(Path(path).expanduser().resolve()),
                    config_sha256=_digest(path), tool_files=tool_files,
                    original_trajectory_claim=False, strict_runtime_checks_pending=True)
    root = Path(args.train_root).expanduser().resolve()
    train = root / "train"
    required = [train / name for name in ("summary.json", "manifest.json", "checkpoints/checkpoint_000500.pt",
                "checkpoints/checkpoint_000500.pt.json", "val_step_000500/result.json", "train.jsonl")]
    required += [root / "workflow_state.json", tool / "SHA256SUMS.json"]
    missing = [str(path) for path in required if not path.is_file()]
    manifest = _read(train / "manifest.json") if (train / "manifest.json").is_file() else {}
    config = manifest.get("arguments")
    if isinstance(config, dict):
        missing += missing_inputs(config)
    if missing:
        raise MissingInputsError(missing)
    if not isinstance(config, dict):
        raise ValueError("Original train manifest lacks recorded arguments")
    static = legacy.study.validate_static(args)
    static.update(start=start, start_step=500, config=copy.deepcopy(config), saved=None,
                  source_sha256=EXPECTED_SOURCE, strict_runtime_checks_pending=True,
                  original_trajectory_claim=False)
    return static


def _guard_original_imports(tool):
    # The original package intentionally uses bare module imports. Reject a
    # different cached module rather than replacing or monkeypatching it.
    for path in Path(tool).glob("*.py"):
        name = path.stem
        if name in sys.modules:
            module_file = getattr(sys.modules[name], "__file__", None)
            if module_file is None or Path(module_file).resolve() != path.resolve():
                raise ImportError("Historical bare module already loaded from another directory: " + name)


def import_runtime(static, legacy):
    """Import original code after hashes; never enter Runtime or initialize CUDA."""
    verify_dependencies(legacy)
    tool = static["tool"]
    _package_files(tool, expected_source=EXPECTED_SOURCE)
    _guard_original_imports(tool)
    if static["start"] == "checkpoint500":
        result = legacy.study.import_original(static)
    else:
        if tool not in sys.path:
            sys.path.insert(0, tool)
        torch = importlib.import_module("torch")
        common = importlib.import_module("common")
        train = importlib.import_module("train")
        run = importlib.import_module("run")
        Runtime = importlib.import_module("server_runtime").Runtime
        if run.verify_release()["code_sha256"] != EXPECTED_SOURCE:
            raise ValueError("Imported wrong original runtime")
        result = (torch, common, train, run, Runtime)
    for module in result[1:4]:
        if Path(module.__file__).resolve().parent != Path(tool):
            raise ImportError("Imported historical code outside pinned tool")
    result[3].check_inputs(SimpleNamespace(**static["config"]))
    return result


def load_checkpoint(static, legacy, torch):
    """Load only in worker. Reuse state validator without bare-import tricks."""
    if static["start"] == "fresh":
        return None
    saved = torch.load(static["checkpoint"], map_location="cpu", weights_only=False)
    legacy.state.validate_checkpoint_payload(saved, require_cuda=True)
    # These binding checks are the original study.load_saved checks, written
    # explicitly because that function imports a bare `continuation` module.
    if (saved.get("source_hash") != EXPECTED_SOURCE
            or saved.get("protocol_info") != static["meta"]["protocol_info"]
            or saved.get("format") != "SRAC_SOFT_MEMORY_Q_REBASED_V1"
            or saved.get("updates_per_arm") != {arm: 500 for arm in ("a", "b", "c", "d")}):
        raise ValueError("Checkpoint payload differs from its verified source binding")
    if not math.isclose(float(saved["config"]["lr"]), 1e-4, rel_tol=1e-12):
        raise ValueError("Source maximum LR is not 1e-4")
    if saved["config"] != static["config"]:
        raise ValueError("Checkpoint config differs from original training manifest")
    return saved
