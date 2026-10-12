from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import predicted_building_label_map
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
    file_sha256,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import (
    change_audit,
    event_from_name,
    partition_from_selected_parents,
    random_selected_parent_ids,
    s2_prediction,
    selected_parent_ids,
    split_oracle_prediction,
)
from .runtime import decode_and_score, load_rgid


def _parse_grid(raw: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not values or any(x < 0 or x > 1 for x in values):
        raise ValueError("risk threshold grid must contain values in [0,1]")
    return values


def _key(threshold: float, prefix: str) -> str:
    return f"{prefix}_{threshold:.2f}".replace(".", "p")


def _sum_audits(rows):
    return {key: sum(int(row[key]) for row in rows) for key in ("changed", "corrected", "harmed", "neutral", "net")}


def _reports(totals):
    return {name: metrics_from_histogram(hist) for name, hist in totals.items()}


def _gain(reports, method, reference="r0_loccc_s2", metric="valid_miou"):
    return 100.0 * (reports[method][metric] - reports[reference][metric])


def _event_report(event_totals, method, reference="r0_loccc_s2"):
    gains = {}
    for event, table in sorted(event_totals.items()):
        gains[event] = 100.0 * (
            metrics_from_histogram(table[method])["valid_miou"]
            - metrics_from_histogram(table[reference])["valid_miou"]
        )
    return {
        "gains_miou_pp": gains,
        "macro_gain_pp": float(np.mean(list(gains.values()))) if gains else float("nan"),
        "improved": sum(value > 0 for value in gains.values()),
        "nonnegative": sum(value >= 0 for value in gains.values()),
        "total": len(gains),
    }


def _bootstrap(image_rows, candidate, reference, iterations, seed):
    grouped = defaultdict(list)
    for event, table in image_rows:
        grouped[event].append(table)
    rng = np.random.default_rng(seed)
    cand = np.zeros((iterations, 4, 4), dtype=np.float64)
    ref = np.zeros_like(cand)
    for rows in grouped.values():
        a = np.stack([row[candidate].numpy() for row in rows])
        b = np.stack([row[reference].numpy() for row in rows])
        index = rng.integers(0, len(rows), size=(iterations, len(rows)))
        cand += a[index].sum(1)
        ref += b[index].sum(1)

    def miou(hist):
        inter = np.diagonal(hist, axis1=1, axis2=2)
        union = hist.sum(1) + hist.sum(2) - inter
        return (inter / np.maximum(union, 1.0))[:, 1:4].mean(1)

    values = 100.0 * (miou(cand) - miou(ref))
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": int(iterations),
        "unit": "images resampled within each event",
    }


def _save_resume_state(path: Path, signature: dict, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"signature": signature, "state": state}, temporary)
    temporary.replace(path)


def _resume_state(
    *, totals, events, changes, image_rows, split_counts, oracle_counts,
    decode_totals, processed,
) -> dict:
    return {
        "totals": totals,
        "events": dict(events),
        "changes": changes,
        "image_rows": image_rows,
        "split_counts": split_counts,
        "oracle_counts": oracle_counts,
        "decode_totals": dict(decode_totals),
        "processed": int(processed),
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Selective RGID-E0 train-only policy fit / frozen calibration audit")
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgid-checkpoint", required=True)
    parser.add_argument("--rgid-provenance", required=True)
    parser.add_argument("--rgid-train-list", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-json", default="")
    parser.add_argument("--reference-json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--risk-threshold-grid", default="0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--minimum-oracle-gain-pp", type=float, default=0.20)
    parser.add_argument("--minimum-selective-gain-pp", type=float, default=0.10)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--state-path", default="")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args(argv)

    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    if args.phase == "evaluate" and not args.policy_json:
        raise ValueError("evaluate requires --policy-json")
    device = torch.device(args.device)
    p3, p3_provenance = load_frozen_p3(args.checkpoint, device)
    head, decoder, boundary_threshold, rgid_provenance = load_rgid(
        args.rgid_checkpoint, args.rgid_provenance, args.checkpoint,
        args.rgid_train_list, device,
    )
    grid = _parse_grid(args.risk_threshold_grid)
    policy = None
    if args.phase == "evaluate":
        policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
        if policy.get("protocol") != PROTOCOL or policy.get("phase") != "fit":
            raise RuntimeError("invalid Selective RGID train-only policy")
        if any(policy.get(key, True) for key in ("meta_calibration_accessed", "meta_confirm_accessed", "val450_accessed", "official_instances_accessed", "test_accessed")):
            raise RuntimeError("policy provenance says a forbidden split was accessed")
        if policy.get("p3_sha256") != file_sha256(args.checkpoint):
            raise RuntimeError("policy/P3 fingerprint mismatch")
        if policy.get("rgid_provenance", {}).get("train_list_sha256") != rgid_provenance.get("train_list_sha256"):
            raise RuntimeError("policy/current RGID provenance mismatch")
        selected_threshold = float(policy["selected_risk_threshold"])
        grid = (selected_threshold,)
        train_names = {x.strip() for x in Path(args.rgid_train_list).read_text(encoding="utf-8").splitlines() if x.strip()}
        eval_names = {x.strip() for x in Path(args.list_path).read_text(encoding="utf-8").splitlines() if x.strip()}
        if train_names & eval_names:
            raise RuntimeError("meta_train and evaluation list overlap")

    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    fixed_names = ("p3", "r0_loccc_s2", "r1_all_rgid_s2", "r4_split_oracle")
    dynamic_names = tuple(name for t in grid for name in (_key(t, "r2"), _key(t, "r3")))
    names_all = fixed_names + dynamic_names
    totals = {name: torch.zeros(4, 4, dtype=torch.float64) for name in names_all}
    events = defaultdict(lambda: {name: torch.zeros(4, 4, dtype=torch.float64) for name in names_all})
    changes = {name: [] for name in names_all if name != "p3"}
    image_rows = []
    split_counts = {str(t): {"selected_parents": 0, "splittable_parents": 0} for t in grid}
    oracle_counts = {"rgid_selected": 0, "ties_kept_loccc": 0}
    decode_totals = defaultdict(int)
    signature = {
        "protocol": PROTOCOL,
        "phase": args.phase,
        "list_path_sha256": file_sha256(args.list_path),
        "p3_sha256": file_sha256(args.checkpoint),
        "rgid_checkpoint_sha256": file_sha256(args.rgid_checkpoint),
        "grid": grid,
        "seed": args.seed,
        "minimum_area": args.minimum_area,
        "maximum_images": args.max_images,
        "names": names_all,
    }
    state_path = Path(args.state_path) if args.state_path else None
    processed = 0
    if state_path is not None and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError(f"resume-state signature mismatch: {state_path}")
        state = saved["state"]
        totals = state["totals"]
        events = defaultdict(
            lambda: {name: torch.zeros(4, 4, dtype=torch.float64) for name in names_all},
            state["events"],
        )
        changes = state["changes"]
        image_rows = state["image_rows"]
        split_counts = state["split_counts"]
        oracle_counts = state["oracle_counts"]
        decode_totals = defaultdict(int, state["decode_totals"])
        processed = int(state["processed"])
        print(f"Resuming Selective RGID {args.phase} at image {processed}", flush=True)

    stop_at = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    if processed < 0 or processed > stop_at:
        raise RuntimeError("invalid processed count in resume state")
    remaining_dataset = Subset(dataset, range(processed, stop_at))
    loader = DataLoader(
        remaining_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    started = time.perf_counter()
    resumed_at = processed

    for image_index, (rgb, sar, sample_names) in enumerate(loader, start=resumed_at):
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits, auxiliary = p3(rgb, sar, return_aux=True)
        probabilities = damage_logits[0].float().softmax(0).cpu()
        p3_prediction = probabilities.argmax(0)
        parents = predicted_building_label_map(loc_logits[0].argmax(0).cpu(), args.minimum_area)
        parent_np = parents.labels.numpy().astype(np.int32, copy=False)
        children, base_evidence, decode_stats = decode_and_score(
            head=head, decoder=decoder, boundary_threshold=boundary_threshold,
            fused_roi=auxiliary["fused_roi"], probabilities=probabilities,
            parent_labels=parents.labels, device=device,
        )
        for key, value in decode_stats.to_dict().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                decode_totals[key] += value
        r0 = s2_prediction(probabilities, parent_np)
        r1 = s2_prediction(probabilities, children)
        sample_name = str(sample_names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()
        oracle, oracle_row = split_oracle_prediction(r0, r1, target, parent_np)
        for key in oracle_counts:
            oracle_counts[key] += oracle_row[key]
        predictions = {"p3": p3_prediction, "r0_loccc_s2": r0, "r1_all_rgid_s2": r1, "r4_split_oracle": oracle}
        splittable = sum(row.child_count >= 2 for row in base_evidence.values())
        for threshold_index, threshold in enumerate(grid):
            selected_ids = selected_parent_ids(base_evidence, threshold)
            selected_count = len(selected_ids)
            selective_labels = partition_from_selected_parents(
                parent_np, children, selected_ids
            )
            random_ids = random_selected_parent_ids(
                base_evidence,
                selected_count,
                args.seed + image_index * 101 + threshold_index,
            )
            random_labels = partition_from_selected_parents(
                parent_np, children, random_ids
            )
            predictions[_key(threshold, "r2")] = s2_prediction(probabilities, selective_labels)
            predictions[_key(threshold, "r3")] = s2_prediction(probabilities, random_labels)
            split_counts[str(threshold)]["selected_parents"] += selected_count
            split_counts[str(threshold)]["splittable_parents"] += splittable
        event = event_from_name(sample_name)
        row = {}
        for name, prediction in predictions.items():
            hist = fast_histogram(prediction, target)
            totals[name] += hist
            events[event][name] += hist
            row[name] = hist
            if name != "p3":
                changes[name].append(change_audit(prediction, r0, target))
        image_rows.append((event, row))
        processed += 1
        if state_path is not None and processed % args.checkpoint_every == 0:
            _save_resume_state(
                state_path,
                signature,
                _resume_state(
                    totals=totals, events=events, changes=changes,
                    image_rows=image_rows, split_counts=split_counts,
                    oracle_counts=oracle_counts, decode_totals=decode_totals,
                    processed=processed,
                ),
            )
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            current_run = processed - resumed_at
            print(
                f"Selective RGID {args.phase}: {processed}/{stop_at} | "
                f"{current_run/max(elapsed,1e-9):.2f} img/s",
                flush=True,
            )

    if state_path is not None:
        _save_resume_state(
            state_path,
            signature,
            _resume_state(
                totals=totals, events=events, changes=changes,
                image_rows=image_rows, split_counts=split_counts,
                oracle_counts=oracle_counts, decode_totals=decode_totals,
                processed=processed,
            ),
        )

    reports = _reports(totals)
    reproduction = None
    if args.phase == "evaluate":
        if not args.reference_json:
            raise ValueError("evaluate requires --reference-json for fixed-S2 reproduction")
        reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
        expected = reference["reports"]["fixed_s2"]
        observed = reports["r0_loccc_s2"]
        errors = {
            key: abs(float(observed[key]) - float(expected[key]))
            for key in ("valid_miou", "damaged_f1", "intact_iou", "damaged_iou", "destroyed_iou")
        }
        reproduction = {"errors": errors, "passed": max(errors.values()) <= 1e-8}
        if not reproduction["passed"]:
            raise RuntimeError(f"R0 did not reproduce locked fixed S2: {errors}")
    event_reports = {name: _event_report(events, name) for name in names_all if name != "r0_loccc_s2"}
    result = {
        "protocol": PROTOCOL,
        "phase": args.phase,
        "selection_contract": "RGID and risk threshold use meta_train only; meta_calibration evaluates one frozen threshold once",
        "settings": vars(args),
        "processed_images": processed,
        "list_path_sha256": file_sha256(args.list_path),
        "p3_sha256": file_sha256(args.checkpoint),
        "p3_provenance": p3_provenance,
        "rgid_provenance": rgid_provenance,
        "locked_s2_configuration": asdict(SelectiveContextConfig()),
        "reports": reports,
        "event_reports": event_reports,
        "change_audits_vs_r0": {name: _sum_audits(rows) for name, rows in changes.items()},
        "split_counts": split_counts,
        "rgid_decode_totals": dict(decode_totals),
        "split_oracle_counts": oracle_counts,
        "r0_fixed_s2_reproduction": reproduction,
        "meta_calibration_accessed": args.phase == "evaluate",
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }

    if args.phase == "fit":
        selected_threshold = max(
            grid,
            key=lambda t: (
                _gain(reports, _key(t, "r2")),
                event_reports[_key(t, "r2")]["macro_gain_pp"],
                -abs(t - 0.55),
            ),
        )
        result["selected_risk_threshold"] = selected_threshold
        result["train_threshold_table"] = {
            str(t): {
                "selective_gain_vs_r0_miou_pp": _gain(reports, _key(t, "r2")),
                "random_gain_vs_r0_miou_pp": _gain(reports, _key(t, "r3")),
                "events": event_reports[_key(t, "r2")],
                **split_counts[str(t)],
            }
            for t in grid
        }
        result["decision"] = "FREEZE_TRAIN_ONLY_SELECTIVE_RGID_POLICY"
        print(f"Selected train-only risk threshold: {selected_threshold:.2f}", flush=True)
    else:
        threshold = grid[0]
        r2, r3 = _key(threshold, "r2"), _key(threshold, "r3")
        bootstraps = {
            "r2_minus_r0": _bootstrap(image_rows, r2, "r0_loccc_s2", args.bootstrap_iterations, args.seed),
            "r2_minus_r1": _bootstrap(image_rows, r2, "r1_all_rgid_s2", args.bootstrap_iterations, args.seed + 1),
            "r2_minus_r3": _bootstrap(image_rows, r2, r3, args.bootstrap_iterations, args.seed + 2),
        }
        checks = {
            "split_oracle_gain_vs_r0_ge_threshold": _gain(reports, "r4_split_oracle") >= args.minimum_oracle_gain_pp,
            "split_oracle_nonnegative_events_ge_threshold": event_reports["r4_split_oracle"]["nonnegative"] >= args.minimum_nonnegative_events,
            "r2_gain_vs_r0_ge_threshold": _gain(reports, r2) >= args.minimum_selective_gain_pp,
            "r2_beats_r1": _gain(reports, r2, "r1_all_rgid_s2") > 0,
            "r2_beats_r3": _gain(reports, r2, r3) > 0,
            "r2_nonnegative_events_ge_threshold": event_reports[r2]["nonnegative"] >= args.minimum_nonnegative_events,
            "r2_bootstrap_ci_low_gt_zero": bootstraps["r2_minus_r0"]["ci95_low_pp"] > 0,
            "r2_damaged_f1_not_below_r0": _gain(reports, r2, metric="damaged_f1") >= 0,
        }
        oracle_passed = checks["split_oracle_gain_vs_r0_ge_threshold"] and checks["split_oracle_nonnegative_events_ge_threshold"]
        if not oracle_passed:
            decision = "STOP_SELECTIVE_RGID_SPLIT_ORACLE_CEILING_LOW"
        elif all(checks.values()):
            decision = "ADMIT_SELECTIVE_RGID_META_CONFIRM"
        else:
            decision = "STOP_SELECTIVE_RGID_NO_DEPLOYABLE_GAIN"
        result.update({
            "selected_risk_threshold": threshold,
            "gains_miou_pp": {
                "r1_minus_r0": _gain(reports, "r1_all_rgid_s2"),
                "r2_minus_r0": _gain(reports, r2),
                "r2_minus_r1": _gain(reports, r2, "r1_all_rgid_s2"),
                "r2_minus_r3": _gain(reports, r2, r3),
                "r4_minus_r0": _gain(reports, "r4_split_oracle"),
            },
            "bootstrap": bootstraps,
            "checks": checks,
            "decision": decision,
        })
        print(json.dumps({"gains_miou_pp": result["gains_miou_pp"], "checks": checks, "decision": decision}, indent=2), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Decision: {result['decision']}", flush=True)
    print(f"Saved: {output}", flush=True)
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
