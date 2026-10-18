"""Read-only fresh-source planning; rebuild derived banks, never missing models.

The existing anchor bridge owns config parsing, dependency discovery and package
verification. This module only decides whether the chain can reuse or regenerate
derived Train banks. Tensor validation remains in the original CUDA runtime.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

PARENT_FORMAT = "SRAC_Q_REBASED_PARENT_BANK_V1"
VALUES_FORMAT = "SRAC_SOFT_MEMORY_VALUES_V1"
MEMBERSHIP_FORMAT = "SRAC_MEMORY_REBUILD_BANK_V3"
PARENT_FILES = ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl")
VALUES_FILES = ("slot_bank.pt", "manifest.json", "summary.json", "images.jsonl", "input_fingerprints.json")


def _bridge(anchor_tool):
    root = Path(anchor_tool).expanduser().resolve()
    path = root / "legacy.py"
    if not path.is_file():
        raise FileNotFoundError("Anchor package is required: " + str(path))
    spec = importlib.util.spec_from_file_location("_anchor_chain_source_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._package_files(root)
    return module


from chain_io import read as _read, digest as _digest


def _check_hash(path, expected, errors, hashes):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        errors.append("Missing/invalid expected SHA256: " + str(path))
        return
    actual = _digest(path)
    hashes[str(path)] = actual
    if actual != expected:
        errors.append("Existing artifact SHA256 differs: " + str(path))


def _safe_entries(root, mapping, errors, hashes):
    if not isinstance(mapping, dict) or not mapping:
        errors.append("Missing/invalid file ledger: " + str(root))
        return
    for name, expected in mapping.items():
        rel = Path(name) if isinstance(name, str) else Path("/")
        if rel.is_absolute() or ".." in rel.parts or "\\" in str(name):
            errors.append("Unsafe bank ledger path: " + str(name))
            continue
        _check_hash(Path(root) / rel, expected, errors, hashes)


def _inspect_bank(root, kind, source_sha, errors, hashes):
    root = Path(root)
    files = PARENT_FILES if kind == "parent" else VALUES_FILES
    expected_format = PARENT_FORMAT if kind == "parent" else VALUES_FORMAT
    missing = [str(root / name) for name in files if not (root / name).is_file()]
    records = {}
    for name in ("manifest.json", "summary.json"):
        path = root / name
        if path.is_file():
            try:
                records[name] = _read(path)
            except (OSError, ValueError) as exc:
                errors.append(str(path) + ": " + str(exc))
                continue
            if records[name].get("format") != expected_format:
                errors.append("Unsupported bank format: " + str(path))
    state = dict(path=str(root), missing=missing, status="missing")
    if len(records) != 2:
        state["status"] = "invalid" if not missing else "missing"
        return state
    manifest, summary = records["manifest.json"], records["summary.json"]
    if manifest.get("complete") is not True or summary.get("complete") is not True:
        state["status"] = "incomplete"
        return state
    if manifest.get("smoke") is True or summary.get("smoke") is True:
        state["status"] = "smoke_only"
        return state
    before = len(errors)
    if manifest.get("smoke") is not False or summary.get("smoke") is not False:
        errors.append("Completed bank lacks explicit smoke=false: " + str(root))
    if (manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
            or manifest.get("execution_complete") is not True):
        errors.append("Completed bank lacks Train-only execution evidence: " + str(root))
    if manifest.get("source_protocol_sha256") != source_sha:
        errors.append("Completed bank binds a different Q source protocol: " + str(root))
    if kind == "parent":
        _safe_entries(root, {"paired_bank.pt": manifest.get("paired_bank_sha256"),
                            "protocol.json": manifest.get("rebuild_protocol_sha256"),
                            "images.jsonl": manifest.get("images_journal_sha256")}, errors, hashes)
    else:
        ledger = manifest.get("file_sha256", {})
        if not {"slot_bank.pt", "images.jsonl", "input_fingerprints.json"} <= set(ledger):
            errors.append("Completed slot bank lacks required file hashes: " + str(root))
        _safe_entries(root, ledger, errors, hashes)
        _check_hash(root / "manifest.json", summary.get("manifest_sha256"), errors, hashes)
        _check_hash(root / "slot_bank.pt", summary.get("slot_bank_sha256"), errors, hashes)
    state["status"] = ("missing" if missing else "reusable") if len(errors) == before else "invalid"
    return state


def _foundation_checks(config, errors, hashes):
    """Check recorded model/evidence bytes without opening model payloads."""
    protocol = Path(config["protocol"])
    if protocol.is_file():
        source = _read(protocol)
        hashes[str(protocol)] = _digest(protocol)
        if (source.get("format") != "SOFT_MEMORY_Q_SOURCE_V1"
                or source.get("test_accessed") is not False
                or source.get("historical_e0_source_reverified") is not False):
            errors.append("Unsupported or contradictory frozen Q-source protocol: " + str(protocol))
        if source.get("device") != config["device"]:
            errors.append("Configured device differs from pinned source protocol")
        if (not isinstance(source.get("repo"), str)
                or Path(source["repo"]).expanduser().resolve() != Path(config["repo"])):
            errors.append("Configured repository differs from pinned source protocol")
        for field, sha in (("q_checkpoint", "q_checkpoint_sha256"),
                           ("a1_checkpoint", "a1_sha256"),
                           ("e0_metadata_protocol", "e0_metadata_protocol_sha256")):
            if isinstance(source.get(field), str):
                _check_hash(source[field], source.get(sha), errors, hashes)
        if isinstance(source.get("q_checkpoint"), str):
            q = Path(source["q_checkpoint"])
            if q.name != "last.pth" or q.parent.name != "train_baseline" or q.parent.parent.name != "formal":
                errors.append("Q source must be formal/train_baseline/last.pth")
            evidence = (q.parent.parent / "val_baseline/summary.json"
                        if source.get("evidence_level") == "independent_val450"
                        else q.parent / "validation.jsonl")
            _check_hash(evidence, source.get("evidence_sha256"), errors, hashes)
            summary = q.parent / "summary.json"
            if summary.is_file():
                status = _read(summary)
                if (status.get("complete") is not True or status.get("smoke") is not False
                        or status.get("arm") != "baseline" or status.get("steps") != 2000
                        or status.get("full_validation_evidence") is not True):
                    errors.append("Frozen Q baseline lacks complete formal training evidence: " + str(summary))
    root = Path(config["membership_root"])
    for name in ("manifest.json", "summary.json"):
        path = root / name
        if path.is_file():
            record = _read(path)
            if (record.get("format") != MEMBERSHIP_FORMAT or record.get("complete") is not True
                    or record.get("smoke") is not False):
                errors.append("Archived Train membership is unsupported/incomplete: " + str(path))
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        if (manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
                or manifest.get("training") is not False):
            errors.append("Archived membership lacks Train-only untrained evidence: " + str(manifest_path))
        for name, field in (("paired_bank.pt", "paired_bank_sha256"),
                            ("protocol.json", "rebuild_protocol_sha256"),
                            ("images.jsonl", "images_journal_sha256")):
            _check_hash(root / name, manifest.get(field), errors, hashes)


def prepare_plan(config_path, anchor_tool, original_tool, output_root):
    """Return a read-only chain plan; never create or modify source files.

    Missing source models block execution. Missing/incomplete *derived* banks
    are regenerated in permanent new output directories. Existing complete
    banks with contradictory metadata or changed hashes block execution.
    """
    bridge = _bridge(anchor_tool)
    config_path = Path(config_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    config = bridge._fresh_config(config_path)
    foundation = dict(config, bank_root=None, values=None)
    errors, hashes = [], {}
    try:
        missing = bridge.missing_inputs(foundation)
    except (ValueError, OSError) as exc:
        missing = []
        errors.append("Foundation metadata error: " + str(exc))
    original = Path(original_tool).expanduser().resolve()
    if not (original / "SHA256SUMS.json").is_file():
        missing.append(str(original / "SHA256SUMS.json"))
    else:
        try:
            hashes.update(bridge._package_files(original, expected_source=bridge.EXPECTED_SOURCE))
        except (ValueError, OSError) as exc:
            errors.append("Original runtime package error: " + str(exc))
    try:
        _foundation_checks(config, errors, hashes)
    except (ValueError, OSError, TypeError) as exc:
        errors.append("Foundation validation error: " + str(exc))
    original_config = copy.deepcopy(config)
    original_roots = [Path(config[k]) for k in ("bank_root", "values", "membership_root")]
    for path in [*original_roots, config_path, Path(config["protocol"]), *map(Path, hashes)]:
        if path == output or path.is_relative_to(output) or output.is_relative_to(path):
            errors.append("Output overlaps protected source: " + str(path))
    source_sha = hashes.get(str(Path(config["protocol"])))
    parent = _inspect_bank(config["bank_root"], "parent", source_sha, errors, hashes)
    actions = []
    if parent["status"] in ("missing", "incomplete", "smoke_only"):
        config["bank_root"] = str(output / "parent")
        config["values"] = str(output / "values")
        actions = ["parent", "build"]
        values = dict(path=original_config["values"], status="replace_with_new_parent")
    else:
        values = _inspect_bank(config["values"], "values", source_sha, errors, hashes)
        if values["status"] in ("missing", "incomplete", "smoke_only"):
            config["values"] = str(output / "values")
            actions = ["build"]
    # Existing reused ledger dependencies must survive too. The original bridge
    # discovers absolute source/image paths; never call an incomplete bank ready.
    if not actions and parent["status"] == "reusable" and values["status"] == "reusable":
        try:
            missing.extend(bridge.missing_inputs(original_config))
        except (ValueError, OSError) as exc:
            errors.append("Reused bank ledger error: " + str(exc))
        manifest = _read(Path(config["values"]) / "manifest.json")
        parent_manifest = _read(Path(config["bank_root"]) / "manifest.json")
        if manifest.get("paired_bank_sha256") != parent_manifest.get("paired_bank_sha256"):
            errors.append("Existing slot bank binds a different parent bank")
    elif actions == ["build"]:
        try:
            missing.extend(bridge.missing_inputs(dict(original_config, values=None)))
        except (ValueError, OSError) as exc:
            errors.append("Reused parent ledger error: " + str(exc))
    for action in actions:
        target = Path(config["bank_root"] if action == "parent" else config["values"])
        if target.exists():
            errors.append("New bank destination already exists; nothing will be overwritten: " + str(target))
    # The parser's internal execution defaults are useful to the worker but must
    # not leak into a generated user config that the same parser will reject.
    config = {k: v for k, v in config.items() if k in bridge._FRESH_REQUIRED or k in bridge._FRESH_DEFAULTS}
    return dict(format="SRAC_ANCHOR_CHAIN_SOURCE_PLAN_V1", training=False, cuda_used=False,
                ready_for_build_or_reuse=not missing and not errors,
                foundation_missing=sorted(set(missing)), metadata_errors=errors,
                bank_actions=actions, bank_inspection={"parent": parent, "values": values},
                config=config, original_config_path=str(config_path),
                config_sha256=_digest(config_path), source_protocol_sha256=source_sha,
                original_tool=str(original), anchor_tool=str(Path(anchor_tool).resolve()),
                output_root=str(output), verified_file_sha256=hashes,
                protected_paths=sorted(set([str(config_path), *map(str, original_roots),
                                            original_config["protocol"], *hashes])),
                strict_runtime_checks_pending=True, historical_trajectory_claim=False)
