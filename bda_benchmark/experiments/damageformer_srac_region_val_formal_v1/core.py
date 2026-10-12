"""Small, CPU-testable contracts for the frozen Val region-path comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REGION_ARMS = (
    "a1_pixel", "pixel_grid", "feature_mean", "region_mean", "grid_probq", "region",
)
ARMS = (*REGION_ARMS, "j1", "j2", "j3")
PAIRS = (
    "pixel_grid__grid_probq", "pixel_grid__feature_mean",
    "feature_mean__region_mean", "region_mean__region",
    "a1_pixel__pixel_grid", "a1_pixel__grid_probq",
    "a1_pixel__j1", "region__j1", "j1__j2", "j1__j3",
)
FORMAT = "SRAC_REGION_VAL_FORMAL_V1"


def formal_gate(image_count, expected_count, reproduction, immutable, pool_ok,
                native_a1_replay_ok):
    """A prefix, failed reproduction, or changed source is never formal Val evidence."""
    return bool(
        image_count == expected_count == 450
        and reproduction.get("passed") is True
        and immutable and all(value is True for value in immutable.values())
        and pool_ok is True and native_a1_replay_ok is True
    )


def reproduction_metrics(metrics):
    """The existing region output is the historical direct_region arm."""
    if not all(name in metrics for name in ("a1_pixel", "region", "j1", "j2", "j3")):
        raise ValueError("A historical reproduction arm is missing")
    return {
        "a1_pixel": metrics["a1_pixel"],
        "direct_region": metrics["region"],
        "j1": metrics["j1"], "j2": metrics["j2"], "j3": metrics["j3"],
    }


def metric_deltas(metrics, pairs=PAIRS):
    """Percentage-point paired mIoU changes; descriptive, not causal claims."""
    result = {}
    for pair in pairs:
        before, after = pair.split("__")
        a, b = metrics[before]["valid_miou"], metrics[after]["valid_miou"]
        result[pair] = None if a is None or b is None else 100.0 * (b - a)
    return result


def source_hash():
    """Bind this new package and imported Q-probe code absent from E0's hash."""
    root = Path(__file__).parent
    experiments = root.parent
    files = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    files += list(root.glob("*.sh"))
    files += [
        experiments / "damageformer_srac_region_test_v1" / "core.py",
        experiments / "damageformer_srac_stage_diagnostics_v1" / "feature_core.py",
        experiments / "damageformer_srac_stage_diagnostics_v1" / "grouped_pool.py",
        experiments / "damageformer_srac_stage_diagnostics_v1" / "shared.py",
        experiments / "damageformer_srac_predsplit_v1" / "audit.py",
    ]
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.relative_to(experiments).as_posix().encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def checked_protocol(path):
    """Accept only the passed, unmodified legacy/native Val450 E0 source."""
    path = Path(path)
    path = path / "protocol.json" if path.is_dir() else path
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Expected an E0 JSON object")
    required = (
        record.get("format") == "SRAC_HARDQUERY_PROTOCOL_V1",
        record.get("complete") is True,
        record.get("full_validation_evidence") is True,
        record.get("image_count") == 450,
        record.get("scope") == "all",
        record.get("profile") == "legacy",
        record.get("pool_mode") == "native",
        record.get("smoke") is False,
        record.get("reproduction", {}).get("passed") is True,
    )
    if not all(required):
        raise ValueError("Requires passed legacy/native all-path Val450 E0")
    for key in ("input_files_unchanged", "code_unchanged",
                "frozen_evidence_unchanged", "numerical_unchanged"):
        if record.get(key) is not True:
            raise ValueError("E0 immutable gate missing or failed: " + key)
    names = record.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 Val450 image order is incomplete or duplicated")
    return path, record


def checked_journal(output, names, digest, safe_child):
    """Re-open every saved per-image map to catch missing/out-of-order evidence."""
    output = Path(output)
    path = output / "images.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != len(names) or [row.get("image") for row in rows] != list(names):
        raise ValueError("Val image journal is missing or out of order")
    if [row.get("index") for row in rows] != list(range(len(names))):
        raise ValueError("Val image indices are missing or out of order")
    for row in rows:
        artifact = safe_child(output, row.get("prediction_file"))
        if digest(artifact) != row.get("prediction_sha256"):
            raise ValueError("Val per-image artifact checksum mismatch: " + str(artifact))
    return rows
