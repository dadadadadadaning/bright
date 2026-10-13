"""New-bank provenance; historical E0 binds the frozen source, not new values.

Only historical support member identities are retained. New T128, F256 and
labels never have to reproduce an archived FP16 tensor or historical Reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    digest, new_output, write_json)

FORMAT = "SRAC_MEMORY_REBUILD_BANK_V3"
PROTOCOL_FORMAT = "SRAC_MEMORY_REBUILD_PROTOCOL_V3"
IDENTITY_FIELDS = ("original_rows", "member_ids", "images", "events", "regions")
EXTRACTION_GATES = ("label_reference_passed", "q_normalization_passed",
                    "native_a1_passed", "pool_guard_passed", "saved_fp32_roundtrip_exact")


def source_hash():
    root = Path(__file__).resolve().parent
    experiments = root.parent
    files = [p for p in root.iterdir() if p.suffix in (".py", ".sh")
             and not p.name.startswith("test_")]
    for package, name in (
        ("damageformer_srac_memory_metric_eval_v2", "native.py"),
        ("damageformer_srac_stage_diagnostics_v1", "feature_core.py"),
        ("damageformer_srac_stage_diagnostics_v1", "grouped_pool.py"),
        ("damageformer_srac_stage_diagnostics_v1", "memory_core.py"),
        ("damageformer_srac_stage_diagnostics_v1", "memory_audit.py"),
        ("damageformer_srac_stage_diagnostics_v1", "shared.py"),
        ("damageformer_srac_evidence_followup_v1", "native_pool.py"),
        ("damageformer_srac_region_test_v1", "core.py"),
        ("damageformer_srac_hardquery_memory_v1", "io_utils.py"),
        ("damageformer_srac_transformer_v1", "common.py"),
        ("damageformer_srac_joint_context_v1", "common.py"),
    ):
        files.append(experiments / package / name)
    files.append(experiments.parent / "script/standard_ML/train_S5_S8_GPU.py")
    checksum = hashlib.sha256()
    for path in sorted(files):
        checksum.update(path.relative_to(experiments.parent).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def validate_splits(train_names, val_names, expected_train=None, expected_val=None):
    for name, values, expected in (("Train", train_names, expected_train),
                                    ("Val", val_names, expected_val)):
        if (not isinstance(values, list) or not values
                or any(not isinstance(s, str) or not s or Path(s).name != s for s in values)
                or len(set(values)) != len(values)):
            raise ValueError(name + " list must contain unique image names")
        if expected is not None and len(values) != expected:
            raise ValueError("Expected " + name + str(expected))
    if set(train_names) & set(val_names):
        raise ValueError("Train and Val lists overlap")
    return True


def split_identity(bundle):
    train, val = bundle["train_names"], bundle["val_names"]
    validate_splits(train, val)
    def names_sha(values):
        return hashlib.sha256(json.dumps(values, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    return dict(train_count=len(train), val_count=len(val),
                train_names_sha256=names_sha(train), val_names_sha256=names_sha(val),
                train_val_disjoint=True, test_accessed=False)


def load_bundle(protocol, output):
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import protocol_record
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import load_verified_bundle
    path, record = protocol_record(protocol)
    if record.get("test_accessed") is not False:
        raise ValueError("Frozen source E0 must explicitly certify no Test access")
    args = argparse.Namespace(**record["arguments"])
    args.protocol, args.output, args.smoke = str(path.resolve()), str(Path(output).resolve()), False
    bundle = load_verified_bundle(args)
    validate_splits(bundle["train_names"], bundle["val_names"], 2918, 450)
    if bundle["val_names"] != record["selected_images"]:
        raise ValueError("Val450 names/order differ from the frozen source E0")
    models = [bundle["front"], bundle["metric"], *bundle["decoders"].values()]
    if (any(module.training for model in models for module in model.modules())
            or any(p.requires_grad for model in models for p in model.parameters())):
        raise ValueError("All source models must be frozen and in eval mode")
    bundle["args"] = args
    return bundle


def provenance(bundle):
    return dict(protocol_sha256=bundle["protocol_sha256"],
                source_e0_sha256=bundle["protocol_sha256"],
                original_member_bank_sha256=bundle["hashes"]["bank_sha256"],
                source_hashes=bundle["hashes"], source_head_hashes=bundle["head_hashes"],
                reference_hashes=bundle["reference_hashes"],
                inherited_source_sha256=bundle["own_source_sha256"],
                source_sha256=source_hash(), split_identity=split_identity(bundle))


def verify_all(bundle, own_before, extra_ledger=None):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import verify_unchanged
    checks = verify_unchanged(bundle)
    checks["rebuild_code"] = source_hash() == own_before
    ledger = {**bundle.get("rebuild_image_ledger", {}), **(extra_ledger or {})}
    checks["rebuild_input_files"] = all(digest(path) == sha for path, sha in ledger.items())
    if not all(value is True for value in checks.values()):
        raise RuntimeError("Rebuild immutability check failed: " + str(checks))
    return checks


def _check_original_metadata(original, train_names=None):
    n = None
    for key in ("images", "events", "regions"):
        value = original.get(key)
        if (not torch.is_tensor(value) or value.ndim != 1
                or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or bool((value < 0).any())):
            raise ValueError("Invalid original member metadata: " + key)
        n = len(value) if n is None else n
        if len(value) != n:
            raise ValueError("Original member metadata lengths differ")
    names = original.get("event_names")
    if (not n or not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(s, str) or not s for s in names)
            or int(original["events"].max()) >= len(names)):
        raise ValueError("Invalid original event identities")
    pairs = torch.stack([original["images"].long(), original["regions"].long()], 1)
    if len(torch.unique(pairs, dim=0)) != n:
        raise ValueError("Duplicate original image/region support identities")
    if train_names is not None:
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event
        if int(original["images"].max()) >= len(train_names):
            raise ValueError("Original member image outside Train list")
        for index in original["images"].unique().tolist():
            expected = event(train_names[index])
            if expected not in names:
                raise ValueError("Original event absent from Train identity")
            if not bool((original["events"][original["images"] == index] == names.index(expected)).all()):
                raise ValueError("Original member event/image binding differs")
    return n


def member_plan(original, train_names, val_names, limit_images=0):
    validate_splits(train_names, val_names)
    if type(limit_images) is not int or limit_images < 0:
        raise ValueError("--limit-images must be a nonnegative integer")
    _check_original_metadata(original, train_names)
    images = original["images"].long().unique(sorted=True).tolist()
    selected = images[:limit_images] if limit_images else images
    mask = torch.zeros(len(original["images"]), dtype=torch.bool)
    for index in selected:
        mask |= original["images"].cpu() == index
    rows = mask.nonzero().flatten().long()
    return dict(rows=rows, images=selected, names=[train_names[i] for i in selected],
                event_names=list(original["event_names"]),
                original_rows=rows.clone(), member_ids=torch.arange(len(rows)),
                **{key: original[key].cpu()[rows].long().clone()
                   for key in ("events", "regions")},
                image_ids=original["images"].cpu()[rows].long().clone())


def membership_digest(bank):
    checksum = hashlib.sha256()
    for key in IDENTITY_FIELDS:
        value = bank.get(key)
        if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != torch.long:
            raise ValueError("Membership digest requires long vector: " + key)
        checksum.update(key.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    checksum.update(json.dumps(bank["event_names"], ensure_ascii=False,
                               separators=(",", ":")).encode())
    return checksum.hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    checksum = hashlib.sha256(header)
    checksum.update(value.view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def label_inventory(bank):
    """Train-bank coverage observations; never a support-selection rule."""
    labels = bank["distribution"].detach().cpu().double()
    weights = bank["label_mass"].detach().cpu().double()
    events = bank["events"].detach().cpu()
    images = bank["images"].detach().cpu()
    eligible = bank["eligible"].detach().cpu()
    def aggregate(mask):
        selected = labels[mask]
        return dict(support_count=int(mask.sum()), source_image_count=len(images[mask].unique()),
                    dominant_counts=torch.bincount(selected.argmax(-1), minlength=4).tolist(),
                    soft_mass=selected.sum(0).tolist(),
                    valid_label_mass_by_class=(selected * weights[mask, None]).sum(0).tolist())
    result = dict(class_names=["background", "intact", "damaged", "destroyed"],
                  all=aggregate(torch.ones(len(labels), dtype=torch.bool)),
                  eligible=aggregate(eligible),
                  by_event={name: dict(all=aggregate(events == index),
                                      eligible=aggregate((events == index) & eligible))
                            for index, name in enumerate(bank["event_names"])},
                  interpretation="Fixed Train support inventory; not full-Train candidate coverage or building counts")
    return result


def verify_bank_journal(root, bank, manifest, bundle):
    path = Path(root) / "images.jsonl"
    if digest(path) != manifest.get("images_journal_sha256"):
        raise ValueError("Train extraction journal checksum mismatch")
    frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = bank["images"].unique(sorted=True).tolist()
    names = [bundle["train_names"][i] for i in images]
    if (len(frames) != len(images) or manifest.get("selected_source_images") != names
            or manifest.get("selected_train_image_ids") != images):
        raise ValueError("Train extraction journal source-image plan differs")
    for index, (image_id, name, frame) in enumerate(zip(images, names, frames)):
        members = (bank["images"] == image_id).nonzero().flatten()
        if (not isinstance(frame, dict) or frame.get("index") != index
                or frame.get("frame_id") != index or frame.get("image") != name
                or frame.get("train_image_id") != image_id
                or frame.get("extraction_forward_count") != 1
                or frame.get("forward_count") != index + 1 or frame.get("rows") != len(members)):
            raise ValueError("Train extraction journal order/one-forward identity differs")
        for field, name_in_frame in (("member_ids", "member_ids"), ("original_rows", "original_rows"),
                                     ("regions", "region_ids")):
            if frame.get(name_in_frame) != bank[field][members].tolist():
                raise ValueError("Journal member correspondence differs: " + field)
        for gate in ("native_a1", "q_check", "label_check"):
            if not isinstance(frame.get(gate), dict) or frame[gate].get("passed") is not True:
                raise ValueError("Train extraction journal contains a failed current check: " + gate)
        reference = frame["label_check"]
        if (reference.get("reference_dtype") != "torch.float64"
                or reference.get("reference_device") != "cpu"
                or reference.get("atol") != 2e-5 or reference.get("rtol") != 2e-5):
            raise ValueError("Train label reference settings differ")
        expected_hashes = {field: tensor_digest(bank[field][members])
                           for field in ("t128", "f256", "distribution", "mass", "label_mass")}
        if frame.get("paired_sha256") != expected_hashes:
            raise ValueError("Journal paired-value fingerprints differ from saved bank")
        if frame.get("joint_ineligible_rows") != int((~bank["eligible"][members]).sum()):
            raise ValueError("Journal paired eligibility count differs")
    return True


def validate_member_identity(bank, manifest, original):
    from .core import validate_paired_bank
    validate_paired_bank(bank)
    n = _check_original_metadata(original)
    rows = bank.get("original_rows")
    count = len(bank["t128"])
    if (not torch.is_tensor(rows) or rows.dtype != torch.long or rows.shape != (count,)
            or not count or bool((rows < 0).any()) or bool((rows >= n).any())
            or not torch.equal(rows, rows.sort().values) or len(rows.unique()) != count):
        raise ValueError("Original member rows must be unique, valid and ordered")
    rows = rows.cpu()
    member_ids = bank.get("member_ids")
    if (not torch.is_tensor(member_ids) or member_ids.dtype != torch.long
            or not torch.equal(member_ids.cpu(), torch.arange(count))):
        raise ValueError("New member IDs must be canonical row identities")
    if manifest.get("smoke") is False and not torch.equal(rows, torch.arange(n)):
        raise ValueError("Formal new bank must retain every original support identity")
    for key in ("images", "events", "regions"):
        if not torch.equal(bank[key].cpu(), original[key].cpu()[rows].long()):
            raise ValueError("Original support identity changed: " + key)
    if bank["event_names"] != original["event_names"]:
        raise ValueError("Original event names changed")
    if manifest.get("support_count") != count:
        raise ValueError("Support count differs from manifest")
    for key in ("mass", "label_mass"):
        value = bank.get(key)
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or value.shape != (count,)
                or not bool(torch.isfinite(value).all()) or bool((value <= 0).any())):
            raise ValueError("Selected members require positive actual FP32 " + key)
    if bool((bank["label_mass"] > bank["mass"] * (1 + 2e-5) + 2e-5).any()):
        raise ValueError("Valid-label mass exceeds total Q mass")
    eligible = ((bank["t128"].norm(dim=-1) > 1e-8)
                & (bank["f256"].norm(dim=-1) > 1e-8))
    for name in ("active", "eligible"):
        value = bank.get(name)
        if not torch.is_tensor(value) or value.dtype != torch.bool or value.shape != (count,):
            raise ValueError("Missing paired eligibility: " + name)
        if not torch.equal(value, eligible):
            raise ValueError("Paired eligibility differs from both key spaces: " + name)
    expected = manifest.get("membership_sha256")
    if expected is not None and membership_digest(bank) != expected:
        raise ValueError("Membership fingerprint differs")
    # Deliberately no comparison to historical tokens or historical labels.
    return True


def _protocol_record(bundle, bank, manifest):
    return dict(format=PROTOCOL_FORMAT, **provenance(bundle),
                paired_bank_sha256=manifest["paired_bank_sha256"],
                membership_sha256=membership_digest(bank), support_count=len(bank["t128"]),
                smoke=manifest["smoke"], train_only=True, test_accessed=False,
                training=False, historical_replay_required=False,
                historical_fp32_reproduction_claim=False,
                source_e0_role="immutable frozen weights/runtime/member identities only",
                member_rule="original Train image/region identities in original row order",
                key_rule=dict(t128="actual same-pass FP32 front tokens",
                              f256="same-pass native F256 exact positive-Q-mass mean"),
                storage_dtype="torch.float32", query_dtype="torch.float32",
                label_rule="current Train GT area fractions/current Q; exact positive valid-label mass normalization",
                label_reference="independent CPU FP64 area fractions and additive route reductions",
                retrieval=dict(metric="cosine", policies=["global"], topk=[8, 64],
                               dot_product="full_fp32_local_tf32_disabled_restored",
                               temperature=0.1, exclude_same_event=True, exclude_same_image=True,
                               eligibility="joint nonzero T128 and F256; no query GT",
                               tie_break="smaller canonical bank row"),
                runtime_settings=bundle["runtime_settings"],
                extraction_checks={key: manifest[key] for key in EXTRACTION_GATES},
                images_journal_sha256=manifest["images_journal_sha256"],
                bank_label_inventory=label_inventory(bank),
                source_image_fingerprints=manifest["source_image_fingerprints"])


def write_rebuild_protocol(output, bundle, bank, manifest):
    protocol = _protocol_record(bundle, bank, manifest)
    path = Path(output) / "protocol.json"
    if path.exists():
        raise ValueError("New experiment protocol already exists")
    write_json(path, protocol)
    return dict(rebuild_protocol_sha256=digest(path), rebuild_protocol_format=PROTOCOL_FORMAT)


def paired_ledger(path):
    root = Path(path).resolve()
    return {root / name: digest(root / name)
            for name in ("manifest.json", "paired_bank.pt", "protocol.json", "summary.json", "images.jsonl")}


def load_paired_bank(path, bundle, allow_smoke=False):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("complete") is not True
            or summary.get("format") != FORMAT or summary.get("complete") is not True
            or manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
            or manifest.get("training") is not False or type(manifest.get("smoke")) is not bool
            or manifest.get("historical_replay_required") is not False
            or manifest.get("historical_fp32_reproduction_claim") is not False):
        raise ValueError("Incomplete or invalid independent new bank")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Smoke bank cannot enter full Val450 diagnosis")
    if (any(manifest.get(key) is not True for key in EXTRACTION_GATES)
            or not isinstance(manifest.get("immutable"), dict) or not manifest["immutable"]
            or not all(value is True for value in manifest["immutable"].values())
            or manifest.get("t128_storage") != "torch.float32"
            or manifest.get("f256_storage") != "torch.float32"):
        raise ValueError("Independent bank extraction/FP32/integrity gate failed")
    for key, expected in provenance(bundle).items():
        if manifest.get(key) != expected:
            raise ValueError("New bank provenance differs: " + key)
    bank_file, protocol_file = root / "paired_bank.pt", root / "protocol.json"
    if digest(bank_file) != manifest.get("paired_bank_sha256"):
        raise ValueError("New bank checksum mismatch")
    if (manifest.get("rebuild_protocol_format") != PROTOCOL_FORMAT
            or digest(protocol_file) != manifest.get("rebuild_protocol_sha256")):
        raise ValueError("Independent experiment protocol checksum mismatch")
    bank = torch.load(bank_file, map_location="cpu", weights_only=True)
    validate_member_identity(bank, manifest, bundle["bank_cpu"])
    if manifest.get("membership_sha256") != membership_digest(bank):
        raise ValueError("New bank lacks its exact membership fingerprint")
    for key in ("smoke", "support_count", "membership_sha256", "paired_bank_sha256", "rebuild_protocol_sha256"):
        if summary.get(key) != manifest.get(key):
            raise ValueError("New bank summary/manifest identity differs: " + key)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if protocol != _protocol_record(bundle, bank, manifest):
        raise ValueError("Independent experiment protocol differs from bank/source/split")
    expected_image_count = len(bank["images"].unique())
    if (manifest.get("image_count") != expected_image_count
            or manifest.get("extraction_forward_count") != expected_image_count
            or summary.get("image_count") != expected_image_count
            or summary.get("extraction_forward_count") != expected_image_count):
        raise ValueError("Bank must contain exactly one extraction forward per source image")
    verify_bank_journal(root, bank, manifest, bundle)
    image_ledger = manifest.get("source_image_fingerprints")
    if not isinstance(image_ledger, dict) or not image_ledger:
        raise ValueError("New bank has no source Train content ledger")
    for name, expected in image_ledger.items():
        if digest(name) != expected:
            raise ValueError("Source Train image changed since extraction: " + name)
    bundle["rebuild_image_ledger"] = {Path(name): value for name, value in image_ledger.items()}
    return {key: value.to(bundle["device"]) if torch.is_tensor(value) else value
            for key, value in bank.items()}, manifest
