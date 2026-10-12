#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from rgid_diag.common import bootstrap_project, write_json


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True, help="BRIGHT-master or bda_benchmark")
    parser.add_argument("--cache-dir", required=True, help="RGID .pt cache directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--damage-template", default="target/{name}_building_damage.tif")
    parser.add_argument("--true-instance-root", default=None)
    parser.add_argument("--true-instance-template", default="{name}.tif")
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Debug on first N cache files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated RGID pseudo-GT, Oracle Center and parent diagnostics"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("pseudo-gt", help="Compare binary CC and damage-class CC")
    _add_shared(audit)
    audit.add_argument("--ignore-index", type=int, default=255)

    parents = sub.add_parser("parent-stats", help="GT children per parent and center spacing")
    _add_shared(parents)
    _add_parent_options(parents)

    oracle = sub.add_parser("oracle-center", help="Learned Center versus GT Oracle Center")
    _add_shared(oracle)
    _add_parent_options(oracle)
    oracle.add_argument("--checkpoint", required=True)
    oracle.add_argument("--device", default="cuda")
    oracle.add_argument("--boundary-gate-threshold", type=float, default=None)

    all_parser = sub.add_parser("all", help="Run all diagnostics and write a decision summary")
    _add_shared(all_parser)
    _add_parent_options(all_parser)
    all_parser.add_argument("--checkpoint", required=True)
    all_parser.add_argument("--device", default="cuda")
    all_parser.add_argument("--boundary-gate-threshold", type=float, default=None)
    all_parser.add_argument("--ignore-index", type=int, default=255)
    return parser


def _add_parent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gt-source", choices=("cached", "damage", "true"), default="cached")
    parser.add_argument("--min-overlap-pixels", type=int, default=1)
    parser.add_argument("--min-gt-fraction-in-parent", type=float, default=0.0)
    parser.add_argument("--max-markers", type=int, default=4)
    parser.add_argument("--head-stride", type=float, default=4.0)
    parser.add_argument("--min-peak-distance-head", type=float, default=3.0)


def _common(args: argparse.Namespace) -> dict:
    return dict(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        damage_template=args.damage_template,
        true_instance_root=args.true_instance_root,
        true_instance_template=args.true_instance_template,
        min_area=args.min_area,
        limit=args.limit,
    )


def _parent(args: argparse.Namespace) -> dict:
    return dict(
        gt_source=args.gt_source,
        min_overlap_pixels=args.min_overlap_pixels,
        min_gt_fraction_in_parent=args.min_gt_fraction_in_parent,
    )


def _decision(audit: dict, parents: dict, oracle: dict) -> dict:
    audit_summary = audit["summary"]
    parent_summary = parents["summary"]
    gains = oracle["oracle_minus_learned_f1"]
    g3_gain = float(gains["g3"])
    added = float(audit_summary.get("mean_instances_added_per_damage_sample", 0.0))
    cap_fraction = float(parent_summary.get("parent_over_marker_cap_fraction", 0.0))
    nms_fraction = float(parent_summary.get("nearest_centers_below_current_nms_fraction", 0.0))
    priorities = []
    if added >= 0.10:
        priorities.append({
            "priority": "pseudo_instance_gt",
            "evidence": f"Damage-class CC adds {added:.3f} instances per labelled sample on average.",
            "interpretation": "Binary CC demonstrably merges at least some differently damaged neighbours; same-class merges remain unmeasured.",
        })
    if g3_gain >= 0.02:
        priorities.append({
            "priority": "center_branch",
            "evidence": f"Oracle G3 F1 exceeds learned-center G3 by {g3_gain:.4f}.",
            "interpretation": "Center prediction/selection is a material current ceiling under fixed downstream geometry.",
        })
    else:
        priorities.append({
            "priority": "parent_watershed_or_supervision",
            "evidence": f"Oracle G3 F1 gain is only {g3_gain:.4f}.",
            "interpretation": "Perfect marker placement alone does not materially lift the fixed pipeline; inspect parent recall, watershed energy and GT quality first.",
        })
    if cap_fraction >= 0.01:
        priorities.append({
            "priority": "marker_cap",
            "evidence": f"{cap_fraction:.2%} of parents exceed max_markers.",
            "interpretation": "The current marker cap imposes a measurable hard ceiling.",
        })
    if nms_fraction >= 0.05:
        priorities.append({
            "priority": "center_nms_distance",
            "evidence": f"{nms_fraction:.2%} of nearest-center distances are below the configured equivalent NMS distance.",
            "interpretation": "Close valid centers may be suppressed even with a strong center map.",
        })
    return {
        "protocol": "rgid_diagnostic_decision_v1",
        "rule_note": "Thresholds are diagnostic heuristics, not promotion criteria.",
        "recommended_order": priorities,
        "defer_until_after_above": ["more_complex_gate", "more_complex_rollback"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_area < 1:
        raise ValueError("--min-area must be >= 1")
    bootstrap_project(args.project_root)
    Path(args.output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    if args.command == "pseudo-gt":
        from rgid_diag.pseudo_gt import audit_pseudo_gt

        report = audit_pseudo_gt(**_common(args), ignore_index=args.ignore_index)
    elif args.command == "parent-stats":
        from rgid_diag.parent_stats import analyze_parents

        report = analyze_parents(
            **_common(args), **_parent(args), max_markers=args.max_markers,
            head_stride=args.head_stride,
            min_peak_distance_head=args.min_peak_distance_head,
        )
    elif args.command == "oracle-center":
        from rgid_diag.oracle_center import evaluate_oracle_centers

        report = evaluate_oracle_centers(
            **_common(args), **_parent(args), checkpoint=args.checkpoint,
            device=args.device, boundary_gate_threshold=args.boundary_gate_threshold,
        )
    else:
        from rgid_diag.oracle_center import evaluate_oracle_centers
        from rgid_diag.parent_stats import analyze_parents
        from rgid_diag.pseudo_gt import audit_pseudo_gt

        audit = audit_pseudo_gt(**_common(args), ignore_index=args.ignore_index)
        parents = analyze_parents(
            **_common(args), **_parent(args), max_markers=args.max_markers,
            head_stride=args.head_stride,
            min_peak_distance_head=args.min_peak_distance_head,
        )
        oracle = evaluate_oracle_centers(
            **_common(args), **_parent(args), checkpoint=args.checkpoint,
            device=args.device, boundary_gate_threshold=args.boundary_gate_threshold,
        )
        report = _decision(audit, parents, oracle)
        write_json(Path(args.output_dir) / "diagnostic_decision.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
