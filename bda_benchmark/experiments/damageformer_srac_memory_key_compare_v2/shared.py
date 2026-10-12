"""Strict E0 provenance and same-member key artifacts; no source overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "SRAC_MEMORY_PAIRED_KEYS_V2"


def source_hash():
    root = Path(__file__).parent
    experiments = root.parent
    files = [p for p in root.iterdir() if p.suffix in (".py", ".sh")
             and not p.name.startswith("test_")]
    for package, name in (
        ("damageformer_srac_memory_metric_eval_v2", "native.py"),
        ("damageformer_srac_stage_diagnostics_v1", "feature_core.py"),
        ("damageformer_srac_stage_diagnostics_v1", "grouped_pool.py"),
        ("damageformer_srac_stage_diagnostics_v1", "shared.py"),
        ("damageformer_srac_evidence_followup_v1", "native_pool.py"),
        ("damageformer_srac_region_test_v1", "core.py"),
        ("damageformer_srac_hardquery_memory_v1", "io_utils.py"),
        ("damageformer_srac_region_head_adapt_v1", "train.py"),
        ("damageformer_srac_fourstage_v2", "losses.py"),
    ):
        files.append(experiments / package / name)
    files.append(experiments.parent / "script/standard_ML/train_S5_S8_GPU.py")
    checksum = hashlib.sha256()
    for path in sorted(files):
        checksum.update(path.relative_to(experiments.parent).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def load_bundle(protocol, output):
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import protocol_record
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import load_verified_bundle
    path, record = protocol_record(protocol)
    if record.get("test_accessed") is not False:
        raise ValueError("E0 must explicitly certify no Test access")
    args = argparse.Namespace(**record["arguments"])
    args.protocol, args.output, args.smoke = str(path.resolve()), str(Path(output).resolve()), False
    bundle = load_verified_bundle(args)
    if (len(bundle["train_names"]) != 2918 or len(bundle["val_names"]) != 450
            or bundle["val_names"] != record["selected_images"]
            or set(bundle["train_names"]) & set(bundle["val_names"])):
        raise ValueError("Expected the disjoint, unchanged Train2918/Val450 E0 lists")
    bundle["args"] = args
    return bundle


def provenance(bundle):
    return dict(protocol_sha256=bundle["protocol_sha256"],
                original_bank_sha256=bundle["hashes"]["bank_sha256"],
                source_hashes=bundle["hashes"], source_head_hashes=bundle["head_hashes"],
                reference_hashes=bundle["reference_hashes"],
                inherited_source_sha256=bundle["own_source_sha256"],
                source_sha256=source_hash())


def verify_all(bundle, own_before, extra_ledger=None):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, verify_unchanged
    checks = verify_unchanged(bundle)
    checks["paired_code"] = source_hash() == own_before
    combined = {**bundle.get("paired_image_ledger", {}), **(extra_ledger or {})}
    checks["paired_input_files"] = all(digest(path) == expected
                                      for path, expected in combined.items())
    if not all(checks.values()):
        raise RuntimeError("Paired-source immutability check failed: " + str(checks))
    return checks


def paired_ledger(path):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    root = Path(path)
    return {p: digest(p) for p in (root / "manifest.json", root / "paired_bank.pt")}


def validate_member_identity(bank, manifest, original):
    """Every new key row is bound to exactly one unchanged original support row."""
    import torch
    from .core import validate_paired_bank
    validate_paired_bank(bank)
    rows = bank.get("original_rows")
    if (not torch.is_tensor(rows) or rows.dtype != torch.long or rows.ndim != 1
            or len(rows) != len(bank["t128"]) or not len(rows)):
        raise ValueError("Missing original bank row identities")
    rows = rows.cpu()
    if (bool((rows < 0).any()) or bool((rows >= len(original["tokens"])).any())
            or not torch.equal(rows, rows.sort().values) or len(rows.unique()) != len(rows)):
        raise ValueError("Original row IDs must be unique, valid, and in original order")
    if manifest.get("smoke") is False and not torch.equal(rows, torch.arange(len(original["tokens"]))):
        raise ValueError("Formal keys must retain every original support in original order")
    fields = {"t128": "tokens", "distribution": "distribution", "events": "events",
              "images": "images", "regions": "regions"}
    for new, old in fields.items():
        actual = bank[new].detach().cpu()
        expected = original[old].detach().cpu()[rows]
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or not torch.equal(actual, expected)):
            raise ValueError("Original support changed or reordered: " + new)
    if bank["event_names"] != original["event_names"]:
        raise ValueError("Original event names changed")
    if manifest.get("support_count") != len(rows):
        raise ValueError("Paired support count differs from manifest")


def load_paired_bank(path, bundle, allow_smoke=False):
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("complete") is not True
            or manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
            or type(manifest.get("smoke")) is not bool):
        raise ValueError("Incomplete or invalid paired Train key artifact")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Smoke key subset cannot enter formal work")
    for key, expected in provenance(bundle).items():
        if manifest.get(key) != expected:
            raise ValueError("Paired key provenance differs: " + key)
    file = root / "paired_bank.pt"
    if digest(file) != manifest.get("paired_bank_sha256"):
        raise ValueError("Paired bank checksum differs from manifest")
    image_ledger = manifest.get("source_image_fingerprints")
    if not isinstance(image_ledger, dict) or not image_ledger:
        raise ValueError("Paired bank has no source-image content ledger")
    for name, expected in image_ledger.items():
        if digest(name) != expected:
            raise ValueError("Source Train image changed since paired extraction: " + name)
    bundle["paired_image_ledger"] = {Path(name): value for name, value in image_ledger.items()}
    bank = torch.load(file, map_location="cpu", weights_only=False)
    validate_member_identity(bank, manifest, bundle["bank_cpu"])
    bank = {key: value.to(bundle["device"]) if torch.is_tensor(value) else value
            for key, value in bank.items()}
    return bank, manifest
