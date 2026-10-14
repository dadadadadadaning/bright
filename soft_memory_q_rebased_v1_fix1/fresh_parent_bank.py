"""Rebuild a Q-matched FP32 parent bank from fixed V3 Train identities.

The archived V3 tensors supply image/event/region membership only. All keys,
mass, eligibility and labels in the new bank come from the current Q front.
Neither this builder nor its reader invokes an historical E0 loader.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    from .common import digest, module_hash, write_json
except ImportError:
    from common import digest, module_hash, write_json


FORMAT = "SRAC_Q_REBASED_PARENT_BANK_V1"
PROTOCOL_FORMAT = "SRAC_Q_REBASED_PARENT_PROTOCOL_V1"
ARCHIVE_FORMAT = "SRAC_MEMORY_REBUILD_BANK_V3"
ARCHIVE_FILES = ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl")
IDENTITY_FIELDS = ("original_rows", "member_ids", "images", "events", "regions")
VALUE_FIELDS = ("t128", "f256", "distribution", "mass", "label_mass")
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_DROP_FRACTION = 0.05


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _names_sha(names):
    return hashlib.sha256(json.dumps(names, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _tensor_sha(value):
    cpu = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(header)
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def membership_sha(bank):
    """Match the archived V3 canonical identity digest without reading values."""
    h = hashlib.sha256()
    for name in IDENTITY_FIELDS:
        value = bank.get(name)
        if not torch.is_tensor(value) or value.dtype != torch.long or value.ndim != 1:
            raise ValueError("Identity must be a long vector: " + name)
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps(bank["event_names"], ensure_ascii=False,
                        separators=(",", ":")).encode())
    return h.hexdigest()


def _source_identity(bundle):
    required = ("source_protocol_sha256", "source_checkpoint_sha256",
                "original_front_sha256", "a1_sha256")
    identity = {}
    provenance = bundle.get("source_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Q source provenance must be a dictionary")
    for key in required:
        value = bundle.get(key, provenance.get(key))
        if not isinstance(value, str) or SHA.fullmatch(value) is None:
            raise ValueError("Missing exact Q source identity: " + key)
        identity[key] = value
    identity["source_provenance"] = provenance
    identity["front_state_sha256"] = module_hash(bundle["front"])
    return identity


def _split_identity(bundle):
    train, val = bundle["train_names"], bundle["val_names"]
    for label, names in (("Train", train), ("Val", val)):
        if (not isinstance(names, list) or not names or len(names) != len(set(names))
                or any(not isinstance(n, str) or not n or Path(n).name != n for n in names)):
            raise ValueError("Invalid " + label + " image list")
    if set(train) & set(val):
        raise ValueError("Train and Val lists overlap")
    return dict(train_count=len(train), val_count=len(val),
                train_names_sha256=_names_sha(train), val_names_sha256=_names_sha(val),
                train_val_disjoint=True, test_accessed=False)


def _repo_dependencies(repo):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "bda_benchmark").is_dir():
        raise FileNotFoundError("BRIGHT repository missing: " + str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    base = "bda_benchmark.experiments."
    audit = importlib.import_module(base + "damageformer_srac_stage_diagnostics_v1.memory_audit")
    native = importlib.import_module(base + "damageformer_srac_memory_metric_eval_v2.native")
    pool = importlib.import_module(base + "damageformer_srac_evidence_followup_v1.native_pool")
    region = importlib.import_module(base + "damageformer_srac_region_test_v1.core")
    transformer = importlib.import_module(base + "damageformer_srac_transformer_v1.common")
    try:
        paired = importlib.import_module(base + "damageformer_srac_memory_rebuild_v3.paired")
    except ModuleNotFoundError as exc:
        if exc.name != base + "damageformer_srac_memory_rebuild_v3":
            raise
        # The synced cloud checkout keeps the exact V3 source in references_v3.
        reference = Path(__file__).resolve().parent.parent / "references_v3/paired.py"
        if not reference.is_file():
            raise
        spec = importlib.util.spec_from_file_location("_q_rebased_v3_paired", reference)
        paired = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(paired)
    return audit, native, pool, region, transformer, paired


def _keys_root(path):
    root = Path(path).expanduser().resolve()
    if (root / "keys").is_dir() and not (root / "paired_bank.pt").is_file():
        root /= "keys"
    for name in ARCHIVE_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError("V3 identity artifact missing: " + str(root / name))
    return root


def _check_archive_members(bundle, root, event):
    """Validate V3 as a Train identity ledger; discard all archived numeric values."""
    split = _split_identity(bundle)
    source = _source_identity(bundle)
    manifest, summary, protocol = (_json(root / name) for name in
                                   ("manifest.json", "summary.json", "protocol.json"))
    if (manifest.get("format") != ARCHIVE_FORMAT or summary.get("format") != ARCHIVE_FORMAT
            or manifest.get("complete") is not True or summary.get("complete") is not True
            or manifest.get("smoke") is not False or summary.get("smoke") is not False
            or manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
            or manifest.get("training") is not False):
        raise ValueError("V3 membership source must be a complete formal Train-only bank")
    for name, expected in (("paired_bank.pt", manifest.get("paired_bank_sha256")),
                           ("protocol.json", manifest.get("rebuild_protocol_sha256")),
                           ("images.jsonl", manifest.get("images_journal_sha256"))):
        if not isinstance(expected, str) or digest(root / name) != expected:
            raise ValueError("V3 identity artifact checksum differs: " + name)
    if protocol.get("format") != "SRAC_MEMORY_REBUILD_PROTOCOL_V3":
        raise ValueError("V3 bank protocol format differs")
    for field in ("paired_bank_sha256", "membership_sha256", "support_count"):
        if manifest.get(field) != summary.get(field) or manifest.get(field) != protocol.get(field):
            raise ValueError("V3 manifest/summary/protocol identity differs: " + field)
    for record in (manifest, protocol):
        old_split = record.get("split_identity")
        if not isinstance(old_split, dict) or any(old_split.get(k) != v for k, v in split.items()):
            raise ValueError("V3 identity inventory binds a different Train/Val split")
        hashes = record.get("source_hashes")
        if (not isinstance(hashes, dict)
                or hashes.get("front_sha256") != source["original_front_sha256"]
                or hashes.get("a1_sha256") != source["a1_sha256"]):
            raise ValueError("V3 identity selection and Q have different initial front/A1")
    raw = torch.load(root / "paired_bank.pt", map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("V3 bank is not a dictionary")
    identity = {name: raw[name].detach().cpu().clone() for name in IDENTITY_FIELDS}
    identity["event_names"] = list(raw["event_names"])
    del raw  # Archived T128/F256/labels are never read or copied.
    n = len(identity["member_ids"])
    names = identity["event_names"]
    if (n < 1 or not isinstance(names, list) or not names or len(names) != len(set(names))
            or any(not isinstance(s, str) or not s for s in names)):
        raise ValueError("Invalid V3 identity inventory")
    for key in IDENTITY_FIELDS:
        value = identity[key]
        if value.shape != (n,) or bool((value < 0).any()):
            raise ValueError("Invalid V3 member identity: " + key)
    if (not torch.equal(identity["member_ids"], torch.arange(n))
            or len(identity["original_rows"].unique()) != n
            or not torch.equal(identity["original_rows"], identity["original_rows"].sort().values)
            or int(identity["images"].max()) >= len(bundle["train_names"])
            or int(identity["events"].max()) >= len(names)):
        raise ValueError("V3 member IDs, rows, image IDs or event IDs invalid")
    pairs = torch.stack((identity["images"], identity["regions"]), dim=1)
    if len(torch.unique(pairs, dim=0)) != n:
        raise ValueError("Duplicate V3 Train image/region identity")
    for image_id in identity["images"].unique().tolist():
        expected_event = event(bundle["train_names"][image_id])
        if expected_event not in names or not bool((identity["events"][identity["images"] == image_id]
                                                  == names.index(expected_event)).all()):
            raise ValueError("V3 Train image/event membership differs")
    if manifest.get("support_count") != n or manifest.get("membership_sha256") != membership_sha(identity):
        raise ValueError("V3 bank membership digest or row count differs")
    image_ids = identity["images"].unique(sorted=True).tolist()
    image_names = [bundle["train_names"][i] for i in image_ids]
    if (manifest.get("selected_train_image_ids") != image_ids
            or manifest.get("selected_source_images") != image_names
            or manifest.get("image_count") != len(image_ids)):
        raise ValueError("V3 Train image selection differs from saved inventory")
    frames = [json.loads(line) for line in (root / "images.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(frames) != len(image_ids):
        raise ValueError("V3 identity journal image count differs")
    for index, (image_id, name, frame) in enumerate(zip(image_ids, image_names, frames)):
        rows = (identity["images"] == image_id).nonzero().flatten()
        if (frame.get("index") != index or frame.get("frame_id") != index
                or frame.get("image") != name or frame.get("train_image_id") != image_id
                or frame.get("rows") != len(rows) or frame.get("forward_count") != index + 1
                or frame.get("extraction_forward_count") != 1):
            raise ValueError("V3 identity journal image/order differs")
        for key, field in (("member_ids", "member_ids"), ("original_rows", "original_rows"),
                           ("regions", "region_ids")):
            if frame.get(field) != identity[key][rows].tolist():
                raise ValueError("V3 identity journal member rows differ: " + field)
        if any(frame.get(gate, {}).get("passed") is not True
               for gate in ("native_a1", "q_check", "label_check")):
            raise ValueError("V3 identity journal extraction gate failed")
    ledger = manifest.get("source_image_fingerprints")
    if not isinstance(ledger, dict) or len(ledger) != 3 * len(image_ids):
        raise ValueError("V3 source image fingerprint ledger incomplete")
    files = {name: digest(root / name) for name in ARCHIVE_FILES}
    return identity, manifest, ledger, files


def _new_bank(identity, selected_ids):
    mask = torch.zeros(len(identity["images"]), dtype=torch.bool)
    for image_id in selected_ids:
        mask |= identity["images"] == image_id
    rows = mask.nonzero().flatten()
    count = len(rows)
    bank = {name: identity[name][rows].clone() for name in IDENTITY_FIELDS}
    bank["member_ids"] = torch.arange(count, dtype=torch.long)
    bank["event_names"] = list(identity["event_names"])
    for name, width in (("t128", 128), ("f256", 256), ("distribution", 4)):
        bank[name] = torch.zeros(count, width, dtype=torch.float32)
    for name in ("mass", "label_mass"):
        bank[name] = torch.zeros(count, dtype=torch.float32)
    for name in ("active", "eligible"):
        bank[name] = torch.zeros(count, dtype=torch.bool)
    return bank


def _smoke_image_ids(identity):
    """Deterministic four-image plan with distinct archived Train events first."""
    images, events = identity["images"], identity["events"]
    ids = images.unique(sorted=True).tolist()
    selected, seen_events = [], set()
    for image_id in ids:
        event_ids = events[images == image_id].unique().tolist()
        if len(event_ids) != 1:
            raise ValueError("One V3 Train image has inconsistent event IDs")
        if event_ids[0] not in seen_events:
            selected.append(image_id)
            seen_events.add(event_ids[0])
        if len(selected) == 4:
            break
    if len(seen_events) < 2:
        raise ValueError("Smoke requires at least two V3 Train source events")
    selected += [image_id for image_id in ids if image_id not in selected][:4 - len(selected)]
    if len(selected) != 4:
        raise ValueError("Smoke requires four V3 Train source images")
    return selected


def _validate_bank(bank, *, expected_count=None):
    n = len(bank["member_ids"])
    if n < 1 or (expected_count is not None and n != expected_count):
        raise ValueError("Fresh parent support count differs")
    for name in IDENTITY_FIELDS:
        item = bank.get(name)
        if not torch.is_tensor(item) or item.dtype != torch.long or item.shape != (n,):
            raise ValueError("Fresh parent identity differs: " + name)
    if not torch.equal(bank["member_ids"].cpu(), torch.arange(n)):
        raise ValueError("Fresh parent member IDs are not canonical")
    if len(torch.unique(torch.stack((bank["images"], bank["regions"]), 1), dim=0)) != n:
        raise ValueError("Fresh parent image/region IDs are duplicated")
    for name, shape in (("t128", (n, 128)), ("f256", (n, 256)),
                        ("distribution", (n, 4)), ("mass", (n,)), ("label_mass", (n,))):
        value = bank.get(name)
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or value.shape != shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Fresh parent FP32 field invalid: " + name)
    if (bool((bank["mass"] <= 0).any()) or bool((bank["label_mass"] <= 0).any())
            or bool((bank["label_mass"] > bank["mass"] * (1 + 2e-5) + 2e-5).any())
            or bool((bank["distribution"] < 0).any())
            or not torch.allclose(bank["distribution"].sum(-1), torch.ones(n), atol=2e-5, rtol=2e-5)):
        raise ValueError("Fresh parent mass or Train distribution invalid")
    eligible = (bank["t128"].norm(dim=1) > 1e-8) & (bank["f256"].norm(dim=1) > 1e-8)
    for name in ("active", "eligible"):
        value = bank.get(name)
        if not torch.is_tensor(value) or value.dtype != torch.bool or value.shape != (n,) or not torch.equal(value, eligible):
            raise ValueError("Fresh joint key eligibility invalid: " + name)
    return True


def _store(bank, rows, result, pair, image_name):
    count = result["count"]
    regions = bank["regions"][rows]
    if result["valid_regions"].dtype != torch.bool or result["valid_regions"].shape != (1, count):
        raise ValueError("Current Q validity mask has wrong shape or dtype")
    valid_rows = []
    dropped = []
    for row, region in zip(rows.tolist(), regions.tolist()):
        reason = None
        if region >= count:
            reason = "region_outside_current_grid"
        elif not bool(result["valid_regions"][0, region]):
            reason = "current_region_invalid"
        elif float(pair["mass"][region]) <= 0:
            reason = "zero_current_q_mass"
        elif float(pair["label_mass"][region]) <= 0:
            reason = "zero_current_train_label_mass"
        if reason is None:
            valid_rows.append(row)
        else:
            dropped.append(dict(selected_member_id=int(bank["member_ids"][row]),
                                original_row=int(bank["original_rows"][row]),
                                train_image_id=int(bank["images"][row]), image=image_name,
                                event_id=int(bank["events"][row]), region_id=region,
                                reason=reason))
    if not valid_rows:
        return dropped
    rows = torch.tensor(valid_rows, dtype=torch.long)
    selected = bank["regions"][rows].to(pair["mass"].device)
    for name, width in (("t128", 128), ("f256", 256), ("distribution", 4),
                        ("mass", None), ("label_mass", None)):
        value = pair[name]
        expected = (count, width) if width else (count,)
        if value.dtype != torch.float32 or value.shape != expected:
            raise ValueError("Current paired value must be actual FP32: " + name)
        bank[name][rows] = value[selected].detach().cpu()
    eligible = ((bank["t128"][rows].norm(dim=1) > 1e-8)
                & (bank["f256"][rows].norm(dim=1) > 1e-8))
    if not torch.equal(eligible, pair["active"][selected].detach().cpu()):
        raise ValueError("Current paired joint eligibility differs")
    bank["active"][rows] = eligible
    bank["eligible"][rows] = eligible
    return dropped


def _filter_invalid(bank, dropped):
    if not dropped:
        return bank
    drop_ids = {row["selected_member_id"] for row in dropped}
    keep = torch.tensor([int(mid) not in drop_ids for mid in bank["member_ids"]], dtype=torch.bool)
    if not bool(keep.any()) or int((~keep).sum()) != len(dropped):
        raise ValueError("Invalid Q member filter produced no or ambiguous supports")
    filtered = {name: value[keep].clone() if torch.is_tensor(value) else list(value)
                for name, value in bank.items()}
    filtered["member_ids"] = torch.arange(int(keep.sum()), dtype=torch.long)
    return filtered


def _coverage(bank, *, smoke=False):
    eligible = bank["eligible"]
    events = bank["events"][eligible].unique().tolist()
    labels = bank["distribution"][eligible].sum(0)
    report = dict(eligible_count=int(eligible.sum()), eligible_event_count=len(events),
                  eligible_event_ids=events, eligible_soft_label_mass=labels.tolist())
    if len(events) < 2:
        raise ValueError("Fresh bank lacks cross-event eligible support")
    if not smoke and not bool((labels[2:] > 0).all()):
        raise ValueError("Formal fresh bank lacks damaged/destroyed Train labels")
    return report


def _source_files_unchanged(bundle):
    paths = bundle.get("input_paths", {})
    fingerprints = bundle.get("input_fingerprints", bundle.get("inputs_before", {}))
    if not isinstance(paths, dict) or not isinstance(fingerprints, dict):
        raise ValueError("Q source input ledgers must be dictionaries")
    checks = {}
    for name, path in paths.items():
        expected = fingerprints.get(name)
        if not isinstance(expected, str) or digest(path) != expected:
            raise ValueError("Q source input changed: " + str(name))
        checks[str(name)] = expected
    return checks


def _archive_image_sha(ledger, path):
    matches = {sha for old, sha in ledger.items() if Path(old).name == path.name}
    if len(matches) != 1 or any(SHA.fullmatch(sha) is None for sha in matches):
        raise ValueError("V3 image content identity is absent or ambiguous: " + path.name)
    return next(iter(matches))


def build_bank(bundle, membership_root, output, repo, workers=4, seed=3407, smoke=False):
    """Create a new Q-matched bank under an initially absent ``output`` directory."""
    if type(workers) is not int or workers < 0 or type(seed) is not int or type(smoke) is not bool:
        raise ValueError("Invalid workers, seed or smoke setting")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("New parent-bank directory required: " + str(output))
    output.mkdir(parents=True)
    started = time.perf_counter()
    summary = dict(format=FORMAT, complete=False, execution_complete=False, smoke=smoke,
                   train_only=True, training=False, test_accessed=False,
                   full_validation_evidence=False, historical_reproduction_claim=False,
                   phase="load", image_count=0)
    write_json(output / "summary.json", summary)
    capture = native_capture = forward_handle = None
    guard = None
    try:
        audit, native, pool, region, transformer, paired = _repo_dependencies(repo)
        if getattr(bundle["args"], "pool_mode", None) != "native":
            raise ValueError("Q source must use its native pooling mode")
        front = bundle["front"]
        if any(m.training for m in front.modules()) or any(p.requires_grad for p in front.parameters()):
            raise ValueError("Q source front must be frozen and in eval mode")
        source = _source_identity(bundle)
        source_files = _source_files_unchanged(bundle)
        builder_sha = digest(__file__)
        root = _keys_root(membership_root)
        identity, archived, old_ledger, archive_files = _check_archive_members(bundle, root, transformer.event)
        image_ids = identity["images"].unique(sorted=True).tolist()
        if smoke:
            image_ids = _smoke_image_ids(identity)
        names = [bundle["train_names"][i] for i in image_ids]
        bank = _new_bank(identity, image_ids)
        selected_files = {}
        for name in names:
            for path in region.sample_paths(bundle["args"].dataset, name).values():
                path = path.resolve()
                sha = digest(path)
                if sha != _archive_image_sha(old_ledger, path):
                    raise ValueError("Train image differs from archived membership source: " + str(path))
                selected_files[str(path)] = sha
        split = _split_identity(bundle)
        manifest = dict(**summary, source_identity=source, source_protocol_sha256=source["source_protocol_sha256"],
                        source_checkpoint_sha256=source["source_checkpoint_sha256"],
                        source_input_fingerprints=source_files, builder_sha256=builder_sha,
                        split_identity=split, archived_membership_root=str(root),
                        archived_files_sha256=archive_files,
                        archived_membership_sha256=archived["membership_sha256"],
                        archived_support_count=archived["support_count"],
                        source_image_fingerprints=selected_files,
                        selected_train_image_ids=image_ids, selected_source_images=names,
                        attempted_train_image_ids=image_ids, attempted_source_images=names,
                        support_count=len(bank["t128"]), membership_sha256=membership_sha(bank),
                        original_selected_count=len(bank["t128"]), dropped_invalid_count=0,
                        dropped_members=[], max_drop_fraction=MAX_DROP_FRACTION,
                        t128_storage="torch.float32", f256_storage="torch.float32",
                        label_storage="torch.float32", key_rule="current Q same-pass native T128 and Q-mean F256",
                        label_rule="current Q-weighted four-class Train area fractions; no Val selection",
                        selection_rule="fixed archived V3 Train image/event/region identities only")
        write_json(output / "manifest.json", manifest)
        calls = [0]
        def count_forward(_module, _inputs, _value):
            calls[0] += 1
        capture = audit.FrozenFrontCapture(front)
        native_capture = native.NativeA1LogitsCapture(front)
        forward_handle = front.register_forward_hook(count_forward)
        guard = pool.PoolGuard(output / "pool_debug", mode="native", capture="failure", check_reference=False)
        data = audit.full_image_loader(bundle["args"].dataset, names, False, workers, seed + 1)
        observations = []
        dropped = []
        with torch.no_grad(), guard:
            for index, batch in enumerate(data):
                if index >= len(image_ids):
                    raise ValueError("Extra Train bank extraction batch")
                pre, post, _loc, target, image_names = audit.parse_batch(batch, bundle["device"])
                if len(pre) != 1 or list(image_names) != [names[index]]:
                    raise ValueError("Train source batch/order differs from fixed membership")
                guard.set_image(names[index])
                native_capture.clear()
                before = calls[0]
                result, feature = capture(pre, post)
                if calls[0] != before + 1:
                    raise ValueError("Expected exactly one current Q forward per Train image")
                logits = native_capture.take()
                _grid, native_check = native.native_grid_probq(result, feature, logits, target.shape[-2:])
                if native_check.get("passed") is not True:
                    raise ValueError("Same-pass native A1 gate failed: " + names[index])
                pair = paired.extract_pair(result, feature, target)
                if (pair["q_check"].get("passed") is not True
                        or pair["label_check"].get("passed") is not True):
                    raise ValueError("Current Q/Train-label conservation gate failed")
                rows = (bank["images"] == image_ids[index]).nonzero().flatten()
                if not len(rows):
                    raise ValueError("Fixed source image has no parent members")
                dropped.extend(_store(bank, rows, result, pair, names[index]))
                observations.append(dict(attempt_index=index, image=names[index],
                                         train_image_id=image_ids[index], forward_count=calls[0],
                                         native_a1=native_check, q_check=pair["q_check"],
                                         label_check=pair["label_check"],
                                         q_sha256=_tensor_sha(result["q"]),
                                         ids_sha256=_tensor_sha(result["ids"])))
                summary.update(phase="Train_paired_rebuild", image_count=index + 1,
                               current_image=names[index])
                write_json(output / "summary.json", summary)
        if summary["image_count"] != len(names) or calls[0] != len(names):
            raise ValueError("Incomplete one-forward-per-Train-image extraction")
        archive_ids_by_original_row = dict(zip(identity["original_rows"].tolist(),
                                               identity["member_ids"].tolist()))
        for item in dropped:
            item["archived_member_id"] = archive_ids_by_original_row[item["original_row"]]
        bank = _filter_invalid(bank, dropped)
        available_image_ids = set(bank["images"].unique(sorted=True).tolist())
        retained_image_ids = [image_id for image_id in image_ids if image_id in available_image_ids]
        retained_names = [bundle["train_names"][i] for i in retained_image_ids]
        retained_archive_ids = [archive_ids_by_original_row[row]
                                for row in bank["original_rows"].tolist()]
        manifest.update(original_selected_count=manifest["original_selected_count"],
                        dropped_invalid_count=len(dropped), dropped_members=dropped,
                        dropped_source_image_ids=sorted(set(image_ids) - set(retained_image_ids)),
                        archived_member_ids_in_bank_order=retained_archive_ids,
                        member_id_renumber_map=[dict(archived_member_id=old,
                                                      new_member_id=new)
                                                for new, old in enumerate(retained_archive_ids)],
                        support_count=len(bank["t128"]), membership_sha256=membership_sha(bank),
                        selected_train_image_ids=retained_image_ids,
                        selected_source_images=retained_names,
                        image_count=len(retained_names), attempted_image_count=len(names),
                        extraction_forward_count=calls[0],
                        dropped_fraction=len(dropped) / manifest["original_selected_count"])
        write_json(output / "manifest.json", manifest)
        if manifest["dropped_fraction"] > MAX_DROP_FRACTION:
            raise ValueError("Material current-Q member loss exceeds 5%: " + str(len(dropped)))
        if smoke and len(retained_image_ids) != 4:
            raise ValueError("Smoke lost one of its four fixed Train source images")
        _validate_bank(bank)
        manifest["support_coverage"] = _coverage(bank, smoke=smoke)
        with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            retained_observations = (item for item in observations
                                     if item["train_image_id"] in retained_image_ids)
            for index, observation in enumerate(retained_observations):
                image_id = observation["train_image_id"]
                rows = (bank["images"] == image_id).nonzero().flatten()
                frame = dict(index=index, frame_id=index, **observation,
                             rows=len(rows), member_ids=bank["member_ids"][rows].tolist(),
                             archived_member_ids=[retained_archive_ids[row] for row in rows.tolist()],
                             original_rows=bank["original_rows"][rows].tolist(),
                             region_ids=bank["regions"][rows].tolist(),
                             extraction_forward_count=1,
                             paired_sha256={name: _tensor_sha(bank[name][rows]) for name in VALUE_FIELDS},
                             joint_ineligible_rows=int((~bank["eligible"][rows]).sum()))
                journal.write(json.dumps(frame, allow_nan=False) + "\n")
        if not guard.restored or guard.failures or getattr(guard, "restore_conflicts", None):
            raise ValueError("Native pooling guard did not restore cleanly")
        if module_hash(front) != source["front_state_sha256"] or digest(__file__) != builder_sha:
            raise ValueError("Q source front or builder changed during extraction")
        if _source_files_unchanged(bundle) != source_files:
            raise ValueError("Q source input ledger changed during extraction")
        if {name: digest(root / name) for name in ARCHIVE_FILES} != archive_files:
            raise ValueError("Archived V3 membership artifacts changed during extraction")
        if any(digest(path) != sha for path, sha in selected_files.items()):
            raise ValueError("Train image changed during extraction")
        with (output / "paired_bank.pt").open("xb") as stream:
            torch.save(bank, stream)
        restored = torch.load(output / "paired_bank.pt", map_location="cpu", weights_only=True)
        _validate_bank(restored, expected_count=len(bank["t128"]))
        for name in bank:
            if (not torch.equal(restored[name], bank[name]) if torch.is_tensor(bank[name])
                    else restored[name] != bank[name]):
                raise ValueError("Fresh parent bank failed exact save/reopen: " + name)
        manifest.update(paired_bank_sha256=digest(output / "paired_bank.pt"),
                        images_journal_sha256=digest(output / "images.jsonl"),
                        image_count=len(retained_names), extraction_forward_count=calls[0],
                        joint_eligible_count=int(bank["eligible"].sum()),
                        joint_ineligible_count=int((~bank["eligible"]).sum()),
                        native_a1_passed=True, q_normalization_passed=True,
                        label_reference_passed=True, pool_guard_passed=True,
                        saved_fp32_roundtrip_exact=True, pool_guard=guard.report(),
                        elapsed_seconds=time.perf_counter() - started)
        protocol = dict(format=PROTOCOL_FORMAT, source_identity=source,
                        split_identity=split, archived_files_sha256=archive_files,
                        archived_membership_sha256=archived["membership_sha256"],
                        paired_bank_sha256=manifest["paired_bank_sha256"],
                        membership_sha256=manifest["membership_sha256"],
                        support_count=manifest["support_count"], image_count=len(retained_names),
                        attempted_image_count=len(names),
                        selected_train_image_ids=retained_image_ids,
                        attempted_train_image_ids=image_ids,
                        original_selected_count=manifest["original_selected_count"],
                        dropped_invalid_count=len(dropped),
                        max_drop_fraction=MAX_DROP_FRACTION,
                        dropped_members_sha256=_hash_json(dropped),
                        support_coverage=manifest["support_coverage"],
                        images_journal_sha256=manifest["images_journal_sha256"],
                        source_image_fingerprints=selected_files, smoke=smoke,
                        train_only=True, test_accessed=False,
                        historical_reproduction_claim=False)
        write_json(output / "protocol.json", protocol)
        manifest.update(rebuild_protocol_sha256=digest(output / "protocol.json"),
                        complete=True, execution_complete=True, phase="complete")
        write_json(output / "manifest.json", manifest)
        summary.update(complete=True, execution_complete=True, phase="complete",
                       image_count=len(retained_names), attempted_image_count=len(names),
                       original_selected_count=manifest["original_selected_count"],
                       dropped_invalid_count=len(dropped),
                       support_count=manifest["support_count"],
                       membership_sha256=manifest["membership_sha256"],
                       paired_bank_sha256=manifest["paired_bank_sha256"],
                       rebuild_protocol_sha256=manifest["rebuild_protocol_sha256"],
                       joint_eligible_count=manifest["joint_eligible_count"],
                       elapsed_seconds=manifest["elapsed_seconds"])
        write_json(output / "summary.json", summary)
        load_bank(output, bundle, allow_smoke=smoke)
        return summary
    except BaseException as exc:
        summary.update(complete=False, execution_complete=False,
                       error=type(exc).__name__ + ": " + str(exc),
                       elapsed_seconds=time.perf_counter() - started)
        write_json(output / "summary.json", summary)
        if (output / "manifest.json").is_file():
            failed = _json(output / "manifest.json")
            failed.update(complete=False, execution_complete=False, error=summary["error"])
            write_json(output / "manifest.json", failed)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if forward_handle is not None:
            forward_handle.remove()
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


def load_bank(keys_root, bundle, allow_smoke=False):
    """Strictly reopen a fresh bank and place its tensors on the Q source device."""
    root = Path(keys_root).expanduser().resolve()
    manifest, summary, protocol = (_json(root / name) for name in
                                   ("manifest.json", "summary.json", "protocol.json"))
    if (manifest.get("format") != FORMAT or summary.get("format") != FORMAT
            or protocol.get("format") != PROTOCOL_FORMAT
            or manifest.get("complete") is not True or summary.get("complete") is not True
            or manifest.get("execution_complete") is not True
            or manifest.get("train_only") is not True or manifest.get("test_accessed") is not False
            or summary.get("smoke") is not manifest.get("smoke")
            or protocol.get("smoke") is not manifest.get("smoke")
            or protocol.get("train_only") is not True or protocol.get("test_accessed") is not False
            or manifest.get("smoke") is not False and not allow_smoke):
        raise ValueError("Incomplete, non-Train or disallowed smoke Q parent bank")
    source = _source_identity(bundle)
    if (manifest.get("source_identity") != source or protocol.get("source_identity") != source
            or manifest.get("split_identity") != _split_identity(bundle)
            or protocol.get("split_identity") != manifest["split_identity"]):
        raise ValueError("Fresh bank binds a different Q source or Train/Val split")
    for name, expected in (("paired_bank.pt", manifest.get("paired_bank_sha256")),
                           ("protocol.json", manifest.get("rebuild_protocol_sha256")),
                           ("images.jsonl", manifest.get("images_journal_sha256"))):
        if not isinstance(expected, str) or digest(root / name) != expected:
            raise ValueError("Fresh bank artifact checksum differs: " + name)
    for name in ("paired_bank_sha256", "membership_sha256", "support_count"):
        if manifest.get(name) != summary.get(name) or manifest.get(name) != protocol.get(name):
            raise ValueError("Fresh bank manifest/summary/protocol differs: " + name)
    if manifest.get("image_count") != protocol.get("image_count"):
        raise ValueError("Fresh bank image count differs")
    archive_root = _keys_root(manifest["archived_membership_root"])
    if {name: digest(archive_root / name) for name in ARCHIVE_FILES} != manifest.get("archived_files_sha256"):
        raise ValueError("Archived V3 identity artifact changed since fresh extraction")
    for path, sha in manifest.get("source_image_fingerprints", {}).items():
        if digest(path) != sha:
            raise ValueError("Fresh bank Train image changed: " + path)
    if _source_files_unchanged(bundle) != manifest.get("source_input_fingerprints"):
        raise ValueError("Fresh bank Q source input ledger changed")
    bank = torch.load(root / "paired_bank.pt", map_location="cpu", weights_only=True)
    _validate_bank(bank, expected_count=manifest["support_count"])
    if membership_sha(bank) != manifest.get("membership_sha256"):
        raise ValueError("Fresh parent membership digest differs")
    archived = torch.load(archive_root / "paired_bank.pt", map_location="cpu", weights_only=True)
    selected_archive = archived["images"].unique(sorted=True).tolist()
    attempted = manifest.get("attempted_train_image_ids")
    archive_identity = {name: archived[name] for name in IDENTITY_FIELDS}
    archive_identity["event_names"] = archived["event_names"]
    if attempted != (_smoke_image_ids(archive_identity) if manifest["smoke"] else selected_archive):
        raise ValueError("Fresh parent attempted Train member plan differs")
    archive_ids = manifest.get("archived_member_ids_in_bank_order")
    if (not isinstance(archive_ids, list) or len(archive_ids) != len(bank["member_ids"])
            or len(set(archive_ids)) != len(archive_ids)
            or any(type(i) is not int or i < 0 or i >= len(archived["member_ids"]) for i in archive_ids)):
        raise ValueError("Fresh parent archived member-ID map invalid")
    old_rows = torch.tensor(archive_ids, dtype=torch.long)
    for name in ("original_rows", "images", "events", "regions"):
        if not torch.equal(bank[name], archived[name][old_rows]):
            raise ValueError("Fresh parent retained V3 identity differs: " + name)
    if (manifest.get("member_id_renumber_map") != [dict(archived_member_id=old, new_member_id=new)
                                                   for new, old in enumerate(archive_ids)]
            or manifest.get("original_selected_count") != sum(int(i in attempted) for i in archived["images"])
            or manifest.get("dropped_invalid_count") != (manifest["original_selected_count"] - len(bank["t128"]))):
        raise ValueError("Fresh parent dropped-member or renumber audit differs")
    dropped = manifest.get("dropped_members")
    if not isinstance(dropped, list) or len(dropped) != manifest["dropped_invalid_count"]:
        raise ValueError("Fresh parent dropped-member ledger incomplete")
    dropped_ids = [item.get("archived_member_id") for item in dropped]
    selected_ids = set(archived["member_ids"][torch.isin(archived["images"],
                                                        torch.tensor(attempted))].tolist())
    if (len(set(dropped_ids)) != len(dropped_ids)
            or set(dropped_ids) & set(archive_ids)
            or set(dropped_ids) | set(archive_ids) != selected_ids
            or protocol.get("dropped_members_sha256") != _hash_json(dropped)
            or manifest.get("dropped_fraction") != len(dropped) / manifest["original_selected_count"]
            or manifest.get("max_drop_fraction") != MAX_DROP_FRACTION
            or manifest["dropped_fraction"] > MAX_DROP_FRACTION):
        raise ValueError("Fresh parent dropped-member audit does not partition selected V3 identities")
    if (protocol.get("max_drop_fraction") != manifest.get("max_drop_fraction")
            or protocol.get("original_selected_count") != manifest.get("original_selected_count")
            or protocol.get("dropped_invalid_count") != manifest.get("dropped_invalid_count")
            or protocol.get("attempted_train_image_ids") != attempted
            or protocol.get("selected_train_image_ids") != manifest.get("selected_train_image_ids")
            or manifest.get("support_coverage") != _coverage(bank, smoke=manifest["smoke"])
            or protocol.get("support_coverage") != manifest["support_coverage"]):
        raise ValueError("Fresh parent material-loss or support-coverage gate differs")
    for item in dropped:
        old = item["archived_member_id"]
        if (item.get("original_row") != int(archived["original_rows"][old])
                or item.get("train_image_id") != int(archived["images"][old])
                or item.get("event_id") != int(archived["events"][old])
                or item.get("region_id") != int(archived["regions"][old])
                or item.get("reason") not in ("region_outside_current_grid", "current_region_invalid",
                                               "zero_current_q_mass", "zero_current_train_label_mass")):
            raise ValueError("Fresh parent dropped-member identity or reason differs")
    if (bank["images"].unique(sorted=True).tolist()
            != sorted(manifest.get("selected_train_image_ids", []))):
        raise ValueError("Fresh parent Train image IDs differ")
    frames = [json.loads(line) for line in (root / "images.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(frames) != manifest["image_count"]:
        raise ValueError("Fresh parent journal length differs")
    for index, frame in enumerate(frames):
        image_id = manifest["selected_train_image_ids"][index]
        rows = (bank["images"] == image_id).nonzero().flatten()
        if (frame.get("index") != index or frame.get("image") != bundle["train_names"][image_id]
                or frame.get("train_image_id") != image_id or frame.get("rows") != len(rows)
                or frame.get("member_ids") != bank["member_ids"][rows].tolist()
                or frame.get("original_rows") != bank["original_rows"][rows].tolist()
                or frame.get("region_ids") != bank["regions"][rows].tolist()
                or frame.get("extraction_forward_count") != 1
                or any(frame.get(gate, {}).get("passed") is not True
                       for gate in ("native_a1", "q_check", "label_check"))
                or frame.get("paired_sha256") != {name: _tensor_sha(bank[name][rows]) for name in VALUE_FIELDS}):
            raise ValueError("Fresh parent journal does not match saved Train values/identities")
    return {name: value.to(bundle["device"]) if torch.is_tensor(value) else value
            for name, value in bank.items()}, manifest
