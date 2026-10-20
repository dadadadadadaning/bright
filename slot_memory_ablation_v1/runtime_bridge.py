"""Read-only reuse of the exact surviving-Q runtime recorded by a server run.

This addon never edits the original package, source protocol or Memory banks.
The original runtime still verifies weights, data, labels and frozen modules.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


DEFAULT_SOURCE = (
    "/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_EXTENDED_V2/"
    "run_20261019_053508"
)
RUNTIME_KEYS = (
    "repo", "protocol", "membership_root", "bank_root", "values", "device",
    "workers", "seed", "key_name", "topk", "chunk_size", "lr", "weight_decay",
    "steps", "eval_every",
)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def check_package(package):
    """Check one manifest's actual scope; never compare unlike release scopes."""
    package = Path(package).resolve()
    manifest = package / "SHA256SUMS.json"
    entries = read_json(manifest)
    if not entries:
        raise ValueError("Empty release manifest: " + str(manifest))
    for name, sha in entries.items():
        rel = Path(name)
        p = package / rel
        if rel.is_absolute() or ".." in rel.parts or p.is_symlink():
            raise ValueError("Unsafe release path: " + name)
        if not p.is_file() or digest(p) != sha:
            raise ValueError("Release file missing/changed: " + str(p))
    unbound = {str(p.relative_to(package)) for p in package.rglob("*.py")} - set(entries)
    if unbound:
        raise ValueError("Unbound Python files: " + repr(sorted(unbound)))
    return digest(manifest)


def inspect_source(source_run, package=None):
    """Stdlib-only plan: no weight deserialization, data read, Torch or CUDA."""
    root = Path(source_run).expanduser().resolve()
    manifest_path = root / "train/manifest.json"
    manifest = read_json(manifest_path)
    config = manifest.get("runtime_config")
    if not isinstance(config, dict):
        raise ValueError("Source manifest lacks runtime_config")
    missing_keys = set(RUNTIME_KEYS) - set(config)
    if missing_keys:
        raise ValueError("Missing runtime configuration keys: " + repr(sorted(missing_keys)))
    tool = Path(manifest["arguments"]["tool_dir"]).expanduser().resolve()
    recorded_package = tool.parents[1]
    package = Path(package).expanduser().resolve() if package else recorded_package
    if package != recorded_package:
        raise ValueError(
            "--package must be the source manifest's actual package, not a newly generated copy"
        )
    required = [
        package / "fresh_chain.py", package / "runtime_adapter.py",
        package / "io_utils.py", package / "deps/study/legacy.py",
        tool / "model.py", tool / "train.py", tool / "slots.py",
        Path(config["protocol"]),
    ]
    for name in ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl"):
        required.append(Path(config["bank_root"]) / name)
    for name in ("slot_bank.pt", "manifest.json", "summary.json", "images.jsonl", "input_fingerprints.json"):
        required.append(Path(config["values"]) / name)
    for name in ("core.py", "paired.py", "shared.py"):
        required.append(
            Path(config["repo"]) / "bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3" / name
        )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required files unavailable; no training started:\n" + "\n".join(missing))
    release = check_package(package)
    source = read_json(config["protocol"])
    if source.get("format") != "BRIGHT_SURVIVING_Q_SOURCE_V2":
        raise ValueError("This addon requires the fresh surviving-Q source protocol")
    if source.get("release_sha256") != release:
        raise ValueError("Source protocol and its actual outer package do not match")
    source_paths = source.get("input_paths", {})
    missing = [str(p) for p in source_paths.values() if not Path(p).is_file()]
    if not Path(source.get("dataset", "")).is_dir():
        missing.append(str(source.get("dataset", "<dataset>")))
    if missing:
        raise FileNotFoundError("Frozen source/data inputs unavailable:\n" + "\n".join(missing))
    for key in ("bank_root", "values"):
        summary = read_json(Path(config[key]) / "summary.json")
        if summary.get("complete") is not True or summary.get("smoke") is not False:
            raise ValueError("Reuse requires a completed FORMAL bank: " + config[key])
    if config["topk"] != 64 or config["key_name"] != "t128":
        raise ValueError("Protocol requires Top64 T128")
    if float(config["lr"]) != 1e-4 or float(config["weight_decay"]) != 0.005:
        raise ValueError("Unexpected original optimizer settings")
    if int(config["steps"]) != 5000:
        raise ValueError("Expected original 5000-step scheduler horizon")
    ledger_paths = [
        manifest_path, Path(config["protocol"]), package / "SHA256SUMS.json",
        package / "deps/study/SHA256SUMS.json", package / "deps/helpers/SHA256SUMS.json",
        Path(config["bank_root"]) / "manifest.json",
        Path(config["values"]) / "manifest.json",
    ]
    return dict(
        format="BRIGHT_SLOT_MEMORY_ABLATION_SOURCE_V1",
        source_run=str(root), package=str(package),
        config={k: config[k] for k in RUNTIME_KEYS},
        source_manifest=str(manifest_path), outer_release_sha256=release,
        study_release_sha256=digest(package / "deps/study/SHA256SUMS.json"),
        original_source_sha256=manifest.get("source_sha256"),
        ledgers={str(p): digest(p) for p in ledger_paths},
        files_present=True, release_files_verified=True,
        weights_loaded=False, cuda_forward_verified=False,
        bank_tensor_checks_pending=True, historical_checkpoint_reused=False,
    )


def verify_ledgers(plan):
    for path, sha in plan["ledgers"].items():
        if digest(path) != sha:
            raise ValueError("Source ledger changed after plan: " + path)
    if check_package(plan["package"]) != plan["outer_release_sha256"]:
        raise ValueError("Outer source package changed")


def named(path, label):
    path = Path(path).resolve()
    tag = "_bright_ablation_" + label + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if tag in sys.modules:
        return sys.modules[tag]
    spec = importlib.util.spec_from_file_location(tag, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(tag, None)
        raise
    return module


def load_runtime(plan):
    """Import the recorded package once in an isolated worker process."""
    verify_ledgers(plan)
    package = Path(plan["package"])
    for name in ("io_utils", "new_source", "new_members", "runtime_adapter"):
        cached = sys.modules.get(name)
        if cached is not None and Path(getattr(cached, "__file__", "/")).resolve() != package / (name + ".py"):
            raise ImportError("Unrelated cached source module: " + name)
    sys.path.insert(0, str(package))
    import io_utils
    _, _, base, _, original_train, common = io_utils.old_modules()
    if plan["original_source_sha256"] and common.source_hash() != plan["original_source_sha256"]:
        raise ValueError("Original runtime code differs from source training manifest")
    from runtime_adapter import NewRuntime
    legacy_bridge = named(package / "deps/study/legacy.py", "legacy")
    legacy = legacy_bridge.load_legacy(package / "deps/helpers")
    import torch

    class AuditedRuntime(NewRuntime):
        def extract(self, batch):
            result = super().extract(batch)
            self.last_extract = result
            return result

        def evaluation_batches(self, names, seed=3407):
            names = list(names)
            if not names or len(names) != len(set(names)):
                raise ValueError("Evaluation names must be nonempty and unique")
            requested = set(names)
            if not (requested <= set(self.train_names) or requested <= set(self.val_names)):
                raise ValueError("Evaluation crosses Train/Val or accesses another split")
            for name in names:
                for path in self.deps.sample_paths(self.bundle["args"].dataset, name).values():
                    path = Path(path).resolve()
                    if path not in self.input_ledger:
                        self.input_ledger[path] = digest(path)
            return self.deps.full_image_loader(
                self.bundle["args"].dataset, names, False, self.args.workers, seed
            )

    return SimpleNamespace(
        torch=torch, original_train=original_train, common=common,
        Runtime=AuditedRuntime, legacy=legacy, legacy_bridge=legacy_bridge,
    )


def runtime_arguments(plan, *, seed, output):
    config = dict(plan["config"])
    config.update(
        seed=int(seed), output=str(output), command="train", mode="train",
        smoke=False, baseline_policy="independent", baseline_root=None,
        baseline_evidence=None, allow_smoke_values=False,
    )
    # The short smoke uses the SAME verified formal bank. It is only its
    # training/evaluation budget that is small; do not masquerade a smoke bank.
    return SimpleNamespace(**config)
