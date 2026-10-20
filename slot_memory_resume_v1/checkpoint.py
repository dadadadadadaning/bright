"""Atomic full-state checkpoints for a separately recorded formal continuation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


FORMAT = "BRIGHT_SLOT_MEMORY_RESUME_CHECKPOINT_V1"
ORIGINAL_FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_V1"
ARMS = ("e", "f", "b", "d", "r", "l")
MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_value(value):
    return json.loads(json.dumps(value, default=str))


def _read_object(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return result


def _context(plan, original_manifest, step):
    if (type(step) is not int or step < 1
            or plan.get("format") != ORIGINAL_FORMAT
            or tuple(plan.get("arms", ())) != ARMS
            or type(plan.get("endpoint_step")) is not int
            or step > plan["endpoint_step"]
            or original_manifest.get("format") != ORIGINAL_FORMAT
            or original_manifest.get("plan") != json_value(plan)):
        raise ValueError("Original plan/manifest, six-arm order or endpoint differs")
    seed = original_manifest.get("seed")
    names = original_manifest.get("train_names")
    if (type(seed) is not int or seed not in plan.get("seeds", ())
            or not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n for n in names)
            or len(names) != len(set(names)) or step > len(names)):
        raise ValueError("Checkpoint must have a valid first-epoch data position")
    return seed, set(names)


def journal_prefix(journal_path, *, step, plan, original_manifest):
    """Bind exactly records 1..step; later appends are intentionally outside it."""
    seed, names = _context(plan, original_manifest, step)
    digest = hashlib.sha256()
    size = 0
    last = None
    with Path(journal_path).open("rb") as stream:
        for expected in range(1, step + 1):
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("Checkpoint journal prefix is missing or unterminated")
            row = json.loads(raw)
            if (not isinstance(row, dict) or type(row.get("step")) is not int
                    or row["step"] != expected
                    or type(row.get("epoch")) is not int or row["epoch"] != 0
                    or type(row.get("epoch_seed")) is not int or row["epoch_seed"] != seed
                    or not isinstance(row.get("image"), str) or row["image"] not in names):
                raise ValueError("Checkpoint journal prefix data position differs")
            arms = row.get("arms")
            if not isinstance(arms, dict) or set(arms) != set(ARMS):
                raise ValueError("Checkpoint journal prefix has incomplete arms")
            for arm in ARMS:
                values = arms[arm]
                if (not isinstance(values, dict)
                        or type(values.get("optimizer_step")) is not int
                        or values["optimizer_step"] != expected):
                    raise ValueError("Checkpoint journal prefix has a partial update")
            digest.update(raw)
            size += len(raw)
            last = row
    return dict(
        journal_prefix_sha256=digest.hexdigest(), journal_prefix_bytes=size,
        last_data_position=dict(epoch=0, epoch_seed=seed, step=step,
                                consumed_in_epoch=step, image=last["image"]),
    )


def input_scope(runtime, inherited):
    current = {str(path): digest for path, digest in runtime.input_ledger.items()}
    recorded = {str(path): digest for path, digest in inherited.items()}
    _ledger_subset(current, recorded)
    return current


def _ledger_subset(current, inherited):
    if not isinstance(current, dict) or not isinstance(inherited, dict) or not current or not inherited:
        raise ValueError("Missing input ledger")
    for mapping in (current, inherited):
        if not all(isinstance(p, str) and p and isinstance(h, str)
                   and re.fullmatch(r"[0-9a-f]{64}", h) for p, h in mapping.items()):
            raise ValueError("Invalid input ledger path/hash")
    for path, digest in current.items():
        if inherited.get(path) != digest:
            raise ValueError("Continuation input differs from inherited checkpoint: " + path)


def _counter(torch, value):
    if torch.is_tensor(value):
        if value.numel() != 1 or value.dtype == torch.bool:
            raise ValueError("Optimizer step must be a numeric scalar")
        number = float(value.item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        raise ValueError("Invalid optimizer step")
    if not math.isfinite(number) or number != int(number):
        raise ValueError("Optimizer step must be a finite integer")
    return int(number)


def _states(torch, heads, optimizers, step):
    if (not isinstance(heads, dict) or tuple(heads) != ARMS
            or not isinstance(optimizers, dict) or tuple(optimizers) != ARMS):
        raise ValueError("Checkpoint heads/optimizers must contain all six ordered arms")
    for arm in ARMS:
        model = heads[arm]
        if (not isinstance(model, dict) or not model
                or any(not torch.is_tensor(t) or not bool(torch.isfinite(t).all())
                       for t in model.values())):
            raise ValueError("Checkpoint has nonfinite or invalid head: " + arm)
        optimizer = optimizers[arm]
        if not isinstance(optimizer, dict):
            raise ValueError("Invalid optimizer payload: " + arm)
        groups, values = optimizer.get("param_groups"), optimizer.get("state")
        if not isinstance(groups, list) or not groups or not isinstance(values, dict):
            raise ValueError("Missing optimizer state/groups: " + arm)
        parameters = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("params"), list):
                raise ValueError("Invalid optimizer parameter group: " + arm)
            parameters.extend(group["params"])
        if (not parameters or any(type(p) is not int for p in parameters)
                or len(parameters) != len(set(parameters)) or set(values) != set(parameters)):
            raise ValueError("Incomplete optimizer parameter states: " + arm)
        for group in groups:
            for parameter in group["params"]:
                entry = values[parameter]
                required = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad"):
                    required.add("max_exp_avg_sq")
                if not isinstance(entry, dict) or not required <= set(entry):
                    raise ValueError("Incomplete AdamW moments: " + arm)
                if _counter(torch, entry["step"]) != step:
                    raise ValueError("Cannot checkpoint a partial update: " + arm)
                first = entry["exp_avg"]
                for key in MOMENTS:
                    if key not in entry:
                        continue
                    tensor = entry[key]
                    if (not torch.is_tensor(first) or not torch.is_tensor(tensor)
                            or tensor.shape != first.shape or tensor.dtype != first.dtype
                            or not bool(torch.isfinite(tensor).all())):
                        raise ValueError("Nonfinite or invalid AdamW moment: " + arm + "." + key)


def _payload(torch, saved, *, plan, original_manifest, lineage, prefix):
    step = saved.get("step")
    _context(plan, original_manifest, step)
    if (saved.get("format") != FORMAT or "diagnostic_only" in saved
            or saved.get("test_accessed") is not False
            or saved.get("optimizer_kind") != "AdamW"
            or json_value(saved.get("study_plan")) != json_value(plan)
            or json_value(saved.get("original_manifest")) != json_value(original_manifest)
            or json_value(saved.get("protocol_info")) != original_manifest.get("protocol_info")
            or json_value(saved.get("lineage")) != json_value(lineage)
            or saved.get("updates_per_arm") != {a: step for a in ARMS}
            or any(saved.get(key) != value for key, value in prefix.items())
            or not isinstance(saved.get("rng"), dict) or not saved["rng"]):
        raise ValueError("Resume checkpoint format, lineage, protocol or complete-step metadata differs")
    _ledger_subset(saved.get("inputs_verified_this_worker"), saved.get("input_file_sha256"))
    _states(torch, saved.get("heads"), saved.get("optimizers"), step)


def save(torch, folder, heads, optimizers, state, runtime, factory, *,
         step, plan, original_manifest, inherited, lineage, journal_path, write_json,
         inherited_all_reverified=False):
    """Publish a roundtrip-checked checkpoint and sidecar without replacing files."""
    _context(plan, original_manifest, step)
    if tuple(heads) != ARMS or tuple(optimizers) != ARMS:
        raise ValueError("Cannot checkpoint incomplete or reordered arms")
    for arm in ARMS:
        optimizer = optimizers[arm]
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Formal continuation requires AdamW")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if (len(parameters) != len(set(map(id, parameters)))
                or set(map(id, parameters)) != set(map(id, heads[arm].parameters()))):
            raise ValueError("Optimizer parameters differ from head: " + arm)
        for parameter, values in optimizer.state.items():
            for key in MOMENTS:
                if key in values and (
                        not torch.is_tensor(values[key]) or values[key].shape != parameter.shape):
                    raise ValueError("AdamW moment shape differs from parameter: " + arm)
    prefix = journal_prefix(journal_path, step=step, plan=plan, original_manifest=original_manifest)
    scope = input_scope(runtime, inherited)
    payload = dict(
        format=FORMAT, step=step, test_accessed=False, optimizer_kind="AdamW",
        heads={a: h.state_dict() for a, h in heads.items()},
        optimizers={a: o.state_dict() for a, o in optimizers.items()},
        rng=state.capture_rng(), updates_per_arm={a: step for a in ARMS},
        study_plan=plan, original_manifest=original_manifest,
        protocol_info=runtime.protocol_info, runtime_config=vars(runtime.args),
        support_provenance=factory.provenance(),
        input_file_sha256={str(p): h for p, h in inherited.items()},
        inputs_verified_this_worker=scope,
        inherited_inputs_all_reverified=bool(inherited_all_reverified),
        lineage=dict(lineage), **prefix,
    )
    _payload(torch, payload, plan=plan, original_manifest=original_manifest,
             lineage=lineage, prefix=prefix)
    expected = state.state_fingerprint(payload)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("checkpoint_%06d.pt" % step)
    sidecar = path.with_suffix(".pt.json")
    if any(p.exists() or p.is_symlink() for p in (path, sidecar)):
        raise FileExistsError(path)
    temporary = []
    try:
        handle, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=folder)
        temp = Path(name)
        temporary.append(temp)
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        reopened = torch.load(temp, map_location="cpu", weights_only=False)
        if (state.state_fingerprint(reopened) != expected
                or state.state_fingerprint(payload) != expected):
            raise ValueError("Resume checkpoint full-state roundtrip differs")
        record = dict(
            format=FORMAT, step=step, sha256=sha256(temp), payload_fingerprint=expected,
            roundtrip_verified=True, path=str(path), lineage=json_value(lineage),
            head_hashes={a: state.state_fingerprint(payload["heads"][a]) for a in ARMS},
            updates_per_arm={a: step for a in ARMS}, **prefix,
        )
        handle, name = tempfile.mkstemp(prefix="." + sidecar.name + ".", suffix=".staging", dir=folder)
        os.close(handle)
        sidecar_temp = Path(name)
        temporary.extend((sidecar_temp, sidecar_temp.with_name(sidecar_temp.name + ".tmp")))
        write_json(sidecar_temp, record)
        if _read_object(sidecar_temp) != record:
            raise ValueError("Resume checkpoint sidecar roundtrip differs")
        # Hard-link publication is atomic and fails if either final name exists.
        # A crash between links leaves an unverified checkpoint, never a valid
        # sidecar pointing at partially written bytes.
        os.link(temp, path)
        os.link(sidecar_temp, sidecar)
        return record
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def load(torch, path, state, *, plan, original_manifest, lineage, journal_path):
    path = Path(path)
    record = _read_object(path.with_suffix(".pt.json"))
    step = record.get("step")
    _context(plan, original_manifest, step)
    if (path.name != "checkpoint_%06d.pt" % step or record.get("format") != FORMAT
            or record.get("roundtrip_verified") is not True
            or record.get("updates_per_arm") != {a: step for a in ARMS}
            or record.get("lineage") != json_value(lineage)
            or sha256(path) != record.get("sha256")):
        raise ValueError("Resume checkpoint file/sidecar mismatch")
    prefix = journal_prefix(journal_path, step=step, plan=plan, original_manifest=original_manifest)
    if any(record.get(key) != value for key, value in prefix.items()):
        raise ValueError("Resume checkpoint journal prefix differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(saved, dict)
            or state.state_fingerprint(saved) != record.get("payload_fingerprint")
            or sha256(path) != record["sha256"]):
        raise ValueError("Resume checkpoint payload fingerprint or file differs")
    if saved.get("step") != step:
        raise ValueError("Resume checkpoint step differs from sidecar")
    _payload(torch, saved, plan=plan, original_manifest=original_manifest,
             lineage=lineage, prefix=prefix)
    if record.get("head_hashes") != {
            a: state.state_fingerprint(saved["heads"][a]) for a in ARMS}:
        raise ValueError("Resume checkpoint sidecar head fingerprints differ")
    return saved, record
