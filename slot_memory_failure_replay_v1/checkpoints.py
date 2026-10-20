"""Separate diagnostic lineage; never write an ordinary experiment checkpoint."""
from pathlib import Path

from inspect_failure import read_object, sha256
from probe import json_value

FORMAT = "BRIGHT_SLOT_MEMORY_DIAGNOSTIC_CHECKPOINT_V1"


def input_scope(runtime, inherited):
    current = {str(p): h for p, h in runtime.input_ledger.items()}
    if not inherited or not current:
        raise ValueError("Missing input ledger")
    for path, digest in current.items():
        if inherited.get(path) != digest:
            raise ValueError("Replay input differs from parent checkpoint: " + path)
    return current


def save(torch, folder, heads, optimizers, state, runtime, factory, *,
         step, plan, manifest, inherited, lineage, journal_path, write_json):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    path = folder / ("diagnostic_%06d.pt" % step)
    temp = path.with_suffix(".pt.tmp")
    if any(p.exists() for p in (path, temp, path.with_suffix(".pt.json"))):
        raise FileExistsError(path)
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be a positive integer")
    if list(heads) != list(plan["arms"]) or list(optimizers) != list(heads):
        raise ValueError("Cannot checkpoint incomplete or reordered arms")
    for arm, opt in optimizers.items():
        parameters = [p for group in opt.param_groups for p in group["params"]]
        if (not parameters or set(parameters) != set(heads[arm].parameters())
                or set(opt.state) != set(parameters)):
            raise ValueError("Cannot checkpoint incomplete parameter states: " + arm)
        for parameter in parameters:
            value = opt.state[parameter]
            if "step" not in value or float(value["step"]) != step:
                raise ValueError("Cannot checkpoint a partial update: " + arm)
            for key in ("exp_avg", "exp_avg_sq"):
                moment = value.get(key)
                if (not torch.is_tensor(moment) or moment.shape != parameter.shape
                        or not bool(torch.isfinite(moment).all())):
                    raise ValueError("Cannot checkpoint invalid Adam moments: " + arm)
    scope = input_scope(runtime, inherited)
    payload = dict(
        format=FORMAT, step=step, diagnostic_only=True, test_accessed=False,
        heads={a: h.state_dict() for a, h in heads.items()},
        optimizers={a: o.state_dict() for a, o in optimizers.items()},
        rng=state.capture_rng(), updates_per_arm={a: step for a in heads},
        study_plan=plan, original_manifest=manifest,
        protocol_info=runtime.protocol_info, runtime_config=vars(runtime.args),
        support_provenance=factory.provenance(),
        input_file_sha256=dict(inherited), inputs_verified_this_worker=scope,
        inherited_inputs_all_reverified=False,
        lineage=dict(lineage), replay_journal_sha256=sha256(journal_path),
        exact_historical_state_claim=False,
    )
    expected = state.state_fingerprint(payload)
    torch.save(payload, temp)
    reopened = torch.load(temp, map_location="cpu", weights_only=False)
    if state.state_fingerprint(reopened) != expected:
        raise ValueError("Diagnostic checkpoint roundtrip differs")
    temp.replace(path)
    record = dict(format=FORMAT, step=step, sha256=sha256(path),
                  payload_fingerprint=expected, roundtrip_verified=True,
                  path=str(path), diagnostic_only=True,
                  lineage=dict(lineage), replay_journal_sha256=payload["replay_journal_sha256"])
    write_json(path.with_suffix(".pt.json"), record)
    return record


def load(torch, path, state, *, plan, manifest, lineage, journal_path, expected_step):
    path = Path(path)
    record = read_object(path.with_suffix(".pt.json"))
    if (record.get("format") != FORMAT or record.get("roundtrip_verified") is not True
            or record.get("step") != expected_step or sha256(path) != record.get("sha256")
            or record.get("lineage") != lineage
            or record.get("replay_journal_sha256") != sha256(journal_path)):
        raise ValueError("Diagnostic checkpoint file/sidecar mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if state.state_fingerprint(saved) != record.get("payload_fingerprint"):
        raise ValueError("Diagnostic checkpoint payload fingerprint differs")
    if (saved.get("format") != FORMAT or saved.get("step") != expected_step
            or saved.get("diagnostic_only") is not True
            or json_value(saved.get("study_plan")) != plan
            or json_value(saved.get("original_manifest")) != manifest
            or saved.get("lineage") != lineage
            or saved.get("replay_journal_sha256") != sha256(journal_path)
            or saved.get("updates_per_arm") != {a: expected_step for a in plan["arms"]}):
        raise ValueError("Diagnostic checkpoint lineage/complete-step mismatch")
    return saved, record
