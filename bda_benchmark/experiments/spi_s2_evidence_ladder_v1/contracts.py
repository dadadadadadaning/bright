from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    parent: str | None
    required_outputs: tuple[str, ...]
    branches: tuple[str, ...] = ()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str | Path | None = None) -> dict:
    source = Path(path) if path else Path(__file__).with_name("protocol.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != "SPI_S2_EVIDENCE_LADDER_V1":
        raise RuntimeError("unexpected evidence-ladder protocol")
    return payload


def stage_specs(protocol: Mapping[str, object]) -> dict[str, StageSpec]:
    output = {}
    for row in protocol["stages"]:
        spec = StageSpec(
            stage_id=str(row["id"]),
            parent=row.get("parent"),
            required_outputs=tuple(row["required_outputs"]),
            branches=tuple(row.get("branches", ())),
        )
        if spec.stage_id in output:
            raise RuntimeError(f"duplicate stage id: {spec.stage_id}")
        output[spec.stage_id] = spec
    return output


def validate_result(
    result: Mapping[str, object],
    spec: StageSpec,
    *,
    parent_result_path: str | Path | None = None,
) -> None:
    if result.get("protocol") != "SPI_S2_EVIDENCE_RESULT_V1":
        raise RuntimeError("result has an incompatible protocol")
    if result.get("stage_id") != spec.stage_id:
        raise RuntimeError(f"result is not for {spec.stage_id}")
    if bool(result.get("test_accessed", True)):
        raise RuntimeError("result accessed Test")
    if bool(result.get("official_instances_accessed", True)):
        raise RuntimeError("result accessed official instances")
    summary = result.get("summary", {})
    missing = [name for name in spec.required_outputs if name not in summary]
    if missing:
        raise RuntimeError(f"stage result misses required outputs: {missing}")
    if spec.branches and summary.get("branch") not in spec.branches:
        raise RuntimeError(f"invalid branch for {spec.stage_id}: {summary.get('branch')}")
    if spec.parent is None:
        return
    if parent_result_path is None:
        raise RuntimeError(f"{spec.stage_id} requires a parent result")
    parent_path = Path(parent_result_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("stage_id") != spec.parent:
        raise RuntimeError(f"expected parent {spec.parent}, got {parent.get('stage_id')}")
    if result.get("parent_result_sha256") != sha256_file(parent_path):
        raise RuntimeError("parent result fingerprint mismatch")
    if parent.get("decision") != "ACCEPT":
        raise RuntimeError("cannot advance from a rejected parent")


def gate_decision(summary: Mapping[str, object], protocol: Mapping[str, object]) -> tuple[str, dict[str, bool]]:
    thresholds = protocol["default_gate"]
    checks = {
        "delta_miou": float(summary["delta_miou_pp"]) >= float(thresholds["minimum_delta_miou_pp"]),
        "harm_reduction": float(summary["harm_reduction_fraction"]) >= float(thresholds["minimum_harm_reduction_fraction"]),
        "selector_auroc": float(summary["selector_auroc"]) >= float(thresholds["minimum_selector_auroc"]),
        "event_stability": float(summary["event_nonnegative_fraction"]) >= float(thresholds["minimum_event_nonnegative_fraction"]),
        "foreground_safety": float(summary["minimum_foreground_delta_pp"]) >= -float(thresholds["maximum_foreground_iou_drop_pp"]),
    }
    return ("ACCEPT" if all(checks.values()) else "REJECT"), checks


def stage_specific_checks(stage_id: str, summary: Mapping[str, object]) -> dict[str, bool]:
    if stage_id == "s4_memory_verifier":
        return {"same_event_excluded": bool(summary["same_event_excluded"])}
    if stage_id == "s5_matching_relation":
        return {"shift_stability": bool(summary["shift_stability"])}
    if stage_id == "s6_selective_rgid":
        return {
            "support_conserved": bool(summary["support_conserved"]),
            "beats_matched_random_split": float(summary["matched_random_split_control_pp"]) > 0.0,
        }
    if stage_id == "s7_soft_superpixel":
        return {
            "shared_partition": bool(summary["shared_partition"]),
            "beats_equal_budget_grid": float(summary["equal_budget_grid_control_pp"]) > 0.0,
        }
    if stage_id == "s8_combination":
        leave_one_out = summary["leave_one_module_out"]
        return {
            "has_multiple_accepted_parents": len(summary["accepted_parents"]) >= 2,
            "every_module_contributes": bool(leave_one_out) and min(float(v) for v in leave_one_out.values()) > 0.0,
        }
    return {}


def decide_stage(stage_id: str, summary: Mapping[str, object], protocol: Mapping[str, object]) -> tuple[str, dict[str, bool]]:
    if stage_id == "s1_s2_audit":
        checks = {
            "has_changes": int(summary["changed"]) > 0,
            "positive_net": int(summary["corrected"]) > int(summary["harmed"]),
            "neutral_accounted": int(summary["neutral"]) == int(summary["changed"]) - int(summary["corrected"]) - int(summary["harmed"]),
            "direction_table_present": bool(summary["direction_table"]),
            "soft_nll_audited": bool(summary["soft_nll_audit"]),
        }
    elif stage_id == "s2_direction_policy":
        checks = {
            "accepted_direction_exists": bool(summary["accepted_directions"]),
            "event_stability_present": bool(summary["event_direction_stability"]),
        }
    elif stage_id in {"s6_selective_rgid", "s7_soft_superpixel"}:
        thresholds = protocol["default_gate"]
        checks = {
            "delta_miou": float(summary["delta_miou_pp"]) >= float(thresholds["minimum_delta_miou_pp"]),
            "event_stability": float(summary["event_nonnegative_fraction"]) >= float(thresholds["minimum_event_nonnegative_fraction"]),
            "foreground_safety": float(summary["minimum_foreground_delta_pp"]) >= -float(thresholds["maximum_foreground_iou_drop_pp"]),
        }
        checks.update(stage_specific_checks(stage_id, summary))
    else:
        _decision, checks = gate_decision(summary, protocol)
        checks.update(stage_specific_checks(stage_id, summary))
    return ("ACCEPT" if all(checks.values()) else "REJECT"), checks
