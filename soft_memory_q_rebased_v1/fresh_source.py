"""Load a completed Q baseline as the frozen source of a *new* memory run.

The Q checkpoint contains the entire trained non-A1 adapter.  Reconstructing
that model requires the immutable A1 checkpoint, but not the deleted E0 front,
J heads, or J bank.  The old E0 protocol is read only for dataset/split
metadata.  Its historical source-file assertions are deliberately not replayed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


FORMAT = "SOFT_MEMORY_Q_SOURCE_V1"
Q_FORMAT = "SRAC_Q_FACTORIAL_V1"
Q_EVAL_FORMAT = "SRAC_Q_FACTORIAL_EVAL_V1"
INDEPENDENT_EVIDENCE = "independent_val450"
TRAINING_EVIDENCE = "training_final_val450"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            sha.update(block)
    return sha.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _existing(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _all_true(value) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def create_source_protocol(q_checkpoint: str | Path, a1_checkpoint: str | Path,
                           e0_metadata_protocol: str | Path, output: str | Path,
                           repo: str | Path, device: str | torch.device) -> Path:
    """Create a new, immutable source identity before building the fresh bank.

    ``output`` can be a JSON filename or the directory to contain
    ``source_protocol.json``.  An identical protocol may be reused on retry;
    an existing different protocol is never overwritten.
    """
    q, a1, e0 = map(_existing, (q_checkpoint, a1_checkpoint, e0_metadata_protocol))
    repository = Path(repo).expanduser().resolve()
    _require(repository.is_dir() and (repository / "bda_benchmark").is_dir(),
             "Source repository is unavailable")
    _require(q.name == "last.pth" and q.parent.name == "train_baseline"
             and q.parent.parent.name == "formal",
             "Formal Q baseline must be formal/train_baseline/last.pth")
    independent = q.parent.parent / "val_baseline" / "summary.json"
    try:
        independent_record = _read(independent) if independent.is_file() else {}
    except (OSError, ValueError):
        # An independent evaluation may still be writing its summary.  Its
        # incomplete file does not disqualify completed training Val450.
        independent_record = {}
    if independent_record.get("complete") is True:
        evidence_level, evidence_path = INDEPENDENT_EVIDENCE, independent
    else:
        evidence_level = TRAINING_EVIDENCE
        evidence_path = _existing(q.parent / "validation.jsonl")
    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        destination /= "source_protocol.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(format=FORMAT, q_checkpoint=str(q), q_checkpoint_sha256=digest(q),
                  a1_checkpoint=str(a1), a1_sha256=digest(a1),
                  e0_metadata_protocol=str(e0), e0_metadata_protocol_sha256=digest(e0),
                  repo=str(repository), device=str(torch.device(device)),
                  evidence_level=evidence_level, evidence_sha256=digest(evidence_path),
                  historical_e0_source_reverified=False, test_accessed=False)
    if destination.exists():
        _require(_read(destination) == record,
                 "Existing Q source protocol differs; use a fresh output")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return destination


def _dependencies(repo: Path) -> SimpleNamespace:
    """Import model code from the requested checkout, rejecting cached others."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    prefix = "bda_benchmark.experiments."
    packages = dict(
        q=prefix + "damageformer_srac_q_factorial_v1.q_variants",
        common=prefix + "damageformer_srac_transformer_v1.common",
        numeric=prefix + "damageformer_srac_hardquery_memory_v1.numeric",
    )
    modules = {name: importlib.import_module(package) for name, package in packages.items()}
    for module in modules.values():
        _require(Path(module.__file__).resolve().is_relative_to(repo),
                 "Imported source code is outside requested repo: " + module.__name__)
    return SimpleNamespace(**modules)


def _source_paths(record: dict, protocol: Path) -> dict[str, Path]:
    _require(record.get("format") == FORMAT and record.get("test_accessed") is False
             and record.get("historical_e0_source_reverified") is False,
             "Invalid new Q source protocol")
    names = ("q_checkpoint", "a1_checkpoint", "e0_metadata_protocol")
    paths = {name: _existing(record[name]) for name in names}
    sha_keys = dict(q_checkpoint="q_checkpoint_sha256", a1_checkpoint="a1_sha256",
                    e0_metadata_protocol="e0_metadata_protocol_sha256")
    for name in names:
        _require(digest(paths[name]) == record.get(sha_keys[name]),
                 "New Q source input changed: " + name)
    q = paths["q_checkpoint"]
    _require(q.name == "last.pth" and q.parent.name == "train_baseline"
             and q.parent.parent.name == "formal",
             "Formal Q baseline must be formal/train_baseline/last.pth")
    level = record.get("evidence_level")
    _require(level in (INDEPENDENT_EVIDENCE, TRAINING_EVIDENCE),
             "New Q source protocol must pin an explicit Val450 evidence level")
    paths.update(source_protocol=protocol,
                 q_train_summary=_existing(q.parent / "summary.json"),
                 q_train_manifest=_existing(q.parent / "manifest.json"))
    if level == INDEPENDENT_EVIDENCE:
        paths.update(q_val_summary=_existing(q.parent.parent / "val_baseline" / "summary.json"),
                     q_val_result=_existing(q.parent.parent / "val_baseline" / "result.json"))
        evidence_path = paths["q_val_summary"]
    else:
        paths["q_train_validation"] = _existing(q.parent / "validation.jsonl")
        evidence_path = paths["q_train_validation"]
    _require(digest(evidence_path) == record.get("evidence_sha256"),
             "Pinned Q Val450 evidence changed")
    return paths


def _validate_formal(payload: dict, train: dict, manifest: dict,
                     val: dict | None, result: dict | None,
                     paths: dict[str, Path], deps: SimpleNamespace,
                     evidence_level: str) -> None:
    """Check formal Q evidence without dereferencing the vanished E0 front."""
    _require(isinstance(payload, dict) and payload.get("format") == Q_FORMAT,
             "Expected Q-factorial checkpoint")
    _require(payload.get("arm") == "baseline" and payload.get("smoke") is False
             and payload.get("test_accessed") is False,
             "Only a formal Train-only baseline Q is a source")
    _require(payload.get("run_complete") is True and payload.get("completed_steps") == 2000
             and payload.get("step") == 2000
             and payload.get("arguments", {}).get("steps") == 2000,
             "Q baseline did not complete its formal 2000 updates")
    _require(payload.get("full_validation_evidence") is True
             and payload.get("initial_a1_reproduction") is True
             and payload.get("final_a1_reproduction", {}).get("passed") is True,
             "Q baseline lacks full frozen-A1 validation evidence")
    _require(payload.get("a1_unchanged") is True
             and payload.get("downstream_unchanged") is True
             and _all_true(payload.get("immutable"))
             and payload["immutable"].get("frozen_downstream") is True,
             "Q baseline did not certify frozen model and training inputs")
    _require(payload.get("source_hash") == deps.q.q_source_hash()
             and payload.get("inherited_source_hash") == deps.q.inherited_source_hash(),
             "Q or inherited source code differs from checkpoint")
    _require(payload.get("a1_sha256") == digest(paths["a1_checkpoint"]),
             "Q baseline A1 fingerprint differs")
    _require(isinstance(payload.get("front_sha256"), str)
             and SHA256.fullmatch(payload["front_sha256"]) is not None,
             "Q baseline lacks its recorded original-front identity")
    _require(payload.get("e0_pool_mode") == "native"
             and isinstance(payload.get("e0_runtime"), dict)
             and bool(payload["e0_runtime"]),
             "Q baseline did not use the native E0 numeric protocol")
    _require(isinstance(payload.get("config"), dict)
             and isinstance(payload.get("adapter"), dict)
             and bool(payload["adapter"]),
             "Q baseline has no complete adapter or model configuration")
    for name, value in (("train summary", train), ("training manifest", manifest)):
        _require(value.get("format") == Q_FORMAT and value.get("arm") == "baseline"
                 and value.get("smoke") is False and value.get("test_accessed") is False,
                 "Incompatible Q " + name)
    _require(train.get("complete") is True and train.get("steps") == 2000
             and train.get("formal_checkpoint") == "last.pth"
             and train.get("full_validation_evidence") is True
             and train.get("initial_a1_reproduction") is True
             and train.get("a1_unchanged") is True
             and train.get("downstream_unchanged") is True
             and _all_true(train.get("immutable")),
             "Q training summary is not formal and complete")
    for field in ("arguments", "source_hash", "inherited_source_hash", "a1_sha256",
                  "train_sha256", "val_sha256", "protocol_sha256", "front_sha256",
                  "e0_pool_mode", "e0_profile", "e0_runtime", "config", "cue_gain"):
        _require(manifest.get(field) == payload.get(field),
                 "Q manifest/checkpoint mismatch: " + field)
    if evidence_level == TRAINING_EVIDENCE:
        _validate_training_val(payload, train, paths["q_train_validation"])
        return
    _require(evidence_level == INDEPENDENT_EVIDENCE
             and isinstance(val, dict) and isinstance(result, dict),
             "Independent Q baseline Val450 records are missing")
    _require(val.get("format") == Q_EVAL_FORMAT and val.get("arm") == "baseline"
             and val.get("complete") is True and val.get("smoke") is False
             and val.get("test_accessed") is False
             and val.get("training") is False
             and val.get("full_validation_evidence") is True
             and val.get("image_count") == 450
             and val.get("expected_image_count") == 450
             and val.get("checkpoint_steps") == 2000
             and val.get("native_a1_replay_ok") is True
             and val.get("reproduction", {}).get("passed") is True
             and val.get("q_a1_reproduction", {}).get("passed") is True
             and _all_true(val.get("immutable")),
             "Independent Q baseline Val450 is incomplete or failed")
    _require(val.get("checkpoint_sha256") == digest(paths["q_checkpoint"])
             and val.get("result_sha256") == digest(paths["q_val_result"]),
             "Q Val450 checkpoint/result fingerprint differs")
    _require(result.get("format") == Q_EVAL_FORMAT and result.get("complete") is True
             and result.get("image_count") == 450 and result.get("arm") == "baseline"
             and result.get("checkpoint_sha256") == val["checkpoint_sha256"],
             "Q Val450 result does not match its summary")


def _validate_training_val(payload: dict, train: dict, journal: Path) -> None:
    """Bind final-step checkpoint metrics to Q training's own full Val450."""
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    _require(len(rows) == 1 and isinstance(rows[0], dict),
             "Q training Val450 journal must contain its one formal final step")
    final = rows[0]
    _require(payload.get("arguments", {}).get("eval_every") == 2000
             and final.get("step") == 2000 and final.get("image_count") == 450
             and final.get("test_accessed") is False,
             "Q training final-step Val450 is incomplete")
    metrics = final.get("metrics")
    _require(isinstance(metrics, dict)
             and all(isinstance(metrics.get(name), dict)
                     for name in ("final", "front_region", "a1_pixel"))
             and metrics["final"] == metrics["front_region"],
             "Q final-step Val450 metrics are incomplete")
    score = metrics["front_region"].get("valid_miou")
    _require(type(score) in (int, float) and math.isfinite(score)
             and score == payload.get("valid_miou")
             and score == train.get("best_valid_miou"),
             "Q final-step Val450 region metric differs from saved checkpoint")
    report = payload.get("final_a1_reproduction")
    _require(isinstance(report, dict) and report.get("passed") is True
             and report.get("image_count") == 450 and report.get("tolerance") == 1e-5
             and report.get("scope") == "a1_only_not_a_joint_training_gate"
             and isinstance(report.get("checks"), list) and len(report["checks"]) >= 17,
             "Q final-step Val450 A1 reproduction report is incomplete")
    for check in report["checks"]:
        _require(isinstance(check, dict) and check.get("passed") is True
                 and isinstance(check.get("name"), str),
                 "Q final-step Val450 A1 metric reproduction failed")
        parts = check["name"].split("/")
        _require(len(parts) == 3 and parts[0] == "a1_pixel"
                 and parts[1] in ("valid_miou", "iou", "precision", "recall", "f1")
                 and parts[2].isdigit(),
                 "Unexpected Q A1 reproduction metric")
        field, index = parts[1], int(parts[2])
        actual = metrics["a1_pixel"].get(field)
        if field == "valid_miou":
            _require(index == 0, "Invalid Q A1 scalar metric index")
            observed = actual
        else:
            _require(isinstance(actual, list) and index < len(actual),
                     "Q A1 class metric is missing")
            observed = actual[index]
        _require(observed == check.get("actual"),
                 "Q final-step Val450 A1 metric differs from reproduction report")


def _split_names(path: Path, count: int) -> list[str]:
    _require(path.name == ("train_set.txt" if count == 2918 else "val_set.txt"),
             "Expected standard Train2918/Val450 list filename")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    _require(len(names) == count and len(set(names)) == count,
             "Expected full, unique Train2918/Val450 split")
    return names


def _check_adapter(front, adapter: dict) -> None:
    expected = front.adapter_state_dict()
    _require(set(expected) == set(adapter), "Q adapter parameter set is incomplete")
    for name, template in expected.items():
        value = adapter[name]
        _require(torch.is_tensor(value) and value.shape == template.shape
                 and value.dtype == template.dtype and bool(torch.isfinite(value).all()),
                 "Invalid Q adapter tensor: " + name)


def load_q_source(protocol_path: str | Path, output: str | Path) -> dict:
    """Return a frozen Q source bundle for fresh Train-only bank and A/B/C/D.

    Historical E0 source files, old front and J bank are *not* checked.  The
    bundle records the old front SHA as provenance only, never as revalidated
    file evidence.  ``output`` is the current run directory for caller context.
    """
    protocol = _existing(protocol_path)
    record = _read(protocol)
    paths = _source_paths(record, protocol)
    repo = Path(record["repo"]).expanduser().resolve()
    _require(repo.is_dir() and (repo / "bda_benchmark").is_dir(),
             "Requested source repository is unavailable")
    deps = _dependencies(repo)
    e0 = _read(paths["e0_metadata_protocol"])
    _require(e0.get("format") == "SRAC_HARDQUERY_PROTOCOL_V1"
             and e0.get("scope") == "all"
             and isinstance(e0.get("arguments"), dict)
             and isinstance(e0.get("input_paths"), dict),
             "E0 metadata cannot identify the dataset and split lists")
    dataset = Path(e0["arguments"]["dataset"]).expanduser().resolve()
    _require(dataset.is_dir(), "Dataset root recorded by E0 metadata is unavailable")
    for name in ("train_list", "val_list"):
        paths[name] = _existing(e0["input_paths"][name])
    train_names = _split_names(paths["train_list"], 2918)
    val_names = _split_names(paths["val_list"], 450)
    _require(not set(train_names).intersection(val_names), "Train/Val names overlap")
    payload = torch.load(paths["q_checkpoint"], map_location="cpu", weights_only=False)
    train = _read(paths["q_train_summary"])
    manifest = _read(paths["q_train_manifest"])
    evidence_level = record["evidence_level"]
    val = _read(paths["q_val_summary"]) if evidence_level == INDEPENDENT_EVIDENCE else None
    result = _read(paths["q_val_result"]) if evidence_level == INDEPENDENT_EVIDENCE else None
    _validate_formal(payload, train, manifest, val, result, paths, deps, evidence_level)
    _require(payload.get("protocol_sha256") == digest(paths["e0_metadata_protocol"]),
             "Q baseline was trained against different E0 metadata")
    for split, path in (("train", paths["train_list"]), ("val", paths["val_list"])):
        _require(payload.get(split + "_sha256") == digest(path),
                 "Q baseline " + split + " split differs from current list")
    device = torch.device(record["device"])
    _require(device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_available()),
             "Requested Q source device is unavailable")
    training_device = torch.device(e0["arguments"]["device"])
    def identity(value: torch.device):
        return value.type, (value.index or 0) if value.type == "cuda" else None
    _require(identity(device) == identity(training_device),
             "Q source device differs from its training numerical protocol")
    _require(e0.get("profile") == payload.get("e0_profile")
             and isinstance(e0.get("numeric_overrides", {}), dict)
             and isinstance(e0.get("runtime_settings"), dict)
             and e0["runtime_settings"] == payload["e0_runtime"],
             "Q checkpoint/E0 numerical-protocol metadata differs")
    deps.numeric.configure_runtime(e0["profile"], **e0.get("numeric_overrides", {}))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    current_runtime = deps.numeric.runtime_settings()
    _require(current_runtime == payload["e0_runtime"],
             "Current runtime differs from Q training's recorded numerical profile")
    front = deps.q.QVariantFront(arm="baseline", cue_gain=payload["cue_gain"],
                                 **payload["config"]).to(device)
    deps.common.load_a1_strict(front, paths["a1_checkpoint"])
    _check_adapter(front, payload["adapter"])
    front.load_adapter_state_dict(payload["adapter"])
    front.requires_grad_(False).eval()
    _require(not any(parameter.requires_grad for parameter in front.parameters()),
             "Q source was not fully frozen")
    fingerprints = {name: digest(path) for name, path in paths.items()}
    _require(fingerprints["q_checkpoint"] == record["q_checkpoint_sha256"],
             "Q checkpoint changed during source load")
    provenance = dict(
        kind="completed_q_baseline", arm="baseline", completed_steps=2000,
        formal_val_image_count=450, evidence_level=evidence_level,
        validation_evidence_path=str(paths["q_val_summary"] if evidence_level == INDEPENDENT_EVIDENCE
                                     else paths["q_train_validation"]),
        validation_evidence_sha256=record["evidence_sha256"],
        independent_validation_completed=evidence_level == INDEPENDENT_EVIDENCE,
        q_source_hash=payload["source_hash"],
        inherited_source_hash=payload["inherited_source_hash"],
        historical_e0_protocol_sha256=payload["protocol_sha256"],
        historical_initial_front_sha256_recorded=payload.get("front_sha256"),
        historical_e0_source_reverified=False,
        historical_initial_front_file_reverified=False,
        historical_j_bank_reused=False,
        current_split_files_reverified=True,
        current_q_checkpoint_reverified=True,
        q_numeric_profile_reproduced=True,
    )
    source_sha = digest(protocol)
    return dict(
        front=front, device=device,
        args=SimpleNamespace(dataset=str(dataset), pool_mode="native",
                             seed=int(e0["arguments"].get("seed", 3407)),
                             workers=int(e0["arguments"].get("workers", 4)),
                             device=str(device)),
        train_names=train_names, val_names=val_names,
        runtime_settings=current_runtime,
        hashes=dict(a1_sha256=fingerprints["a1_checkpoint"],
                    train_sha256=fingerprints["train_list"],
                    val_sha256=fingerprints["val_list"],
                    q_checkpoint_sha256=fingerprints["q_checkpoint"]),
        head_hashes={}, own_source_sha256=digest(__file__),
        input_paths={name: str(path) for name, path in paths.items()},
        input_fingerprints=fingerprints, inputs_before=fingerprints,
        source_protocol_sha256=source_sha,
        source_checkpoint_sha256=fingerprints["q_checkpoint"],
        original_front_sha256=payload.get("front_sha256"),
        a1_sha256=fingerprints["a1_checkpoint"],
        evidence_level=evidence_level,
        source_provenance=provenance, source_kind="completed_q_baseline",
    )
