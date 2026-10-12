# -*- coding: utf-8 -*-
"""CMR-M1BG-B: Damaged-anchor training from a locked global graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from model.cmr.metric import CMRResidualMetric
from script.standard_ML.cmr_global_graph_utils import (
    cosine_preservation_loss,
    damaged_anchor_nce_loss,
)
from script.standard_ML.cmr_metric_utils import (
    cross_event_retrieval_eval,
    frozen_memory_view,
    load_cmr_cache,
    relation_statistics,
)
from script.standard_ML.rm_region_retrieval_runtime import validate_disjoint_ids
from script.standard_ML.train_CMR_M1B_selective import (
    atomic_torch_save,
    project_features,
    selective_metric_gate,
)


def sample_cross_event_negatives(
    labels: torch.Tensor,
    events: torch.Tensor,
    anchor_events: torch.Tensor,
    *,
    negatives_per_class: int,
    negative_classes: tuple[int, ...] = (0, 2),
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample equal declared-class negatives outside the anchor event."""

    rows = []
    for anchor_event in anchor_events.tolist():
        selected = []
        for cls in negative_classes:
            pool = torch.nonzero(
                (labels == cls) & (events != int(anchor_event)), as_tuple=False
            ).flatten()
            if pool.numel() == 0:
                raise RuntimeError(f"no class-{cls} negatives outside event {anchor_event}")
            if pool.numel() >= int(negatives_per_class):
                local = torch.randperm(pool.numel(), generator=generator)[
                    : int(negatives_per_class)
                ]
            else:
                local = torch.randint(
                    pool.numel(), (int(negatives_per_class),), generator=generator
                )
            selected.append(pool[local])
        rows.append(torch.cat(selected))
    return torch.stack(rows)


def event_nonnegative_diagnostics(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    ap_k: int,
) -> dict:
    key = f"macro_mAP@{int(ap_k)}"
    candidate_events = candidate.get("per_event", {})
    baseline_events = baseline.get("per_event", {})
    common = sorted(set(candidate_events) & set(baseline_events))
    delta = {
        event: float(candidate_events[event][key]) - float(baseline_events[event][key])
        for event in common
    }
    return {
        "comparable_events": len(common),
        "nonnegative_events": sum(value >= -1e-12 for value in delta.values()),
        "event_macro_map_delta": delta,
    }


def severity_metric_gate(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    ap_k: int,
    minimum_damaged_gain: float,
    maximum_destroyed_drop: float,
    minimum_nonnegative_events: int,
) -> tuple[dict, dict]:
    """M1BG-S direct-task gate; DeltaCE remains diagnostic only."""

    candidate_pc = candidate["per_class"]
    baseline_pc = baseline["per_class"]
    event_diagnostics = event_nonnegative_diagnostics(
        candidate, baseline, ap_k=ap_k
    )
    checks = {
        "damaged_r1_plus_declared_gain": (
            float(candidate_pc[1]["R@1"])
            >= float(baseline_pc[1]["R@1"]) + float(minimum_damaged_gain)
        ),
        "damaged_r5_nondecreasing": (
            float(candidate_pc[1]["R@5"]) >= float(baseline_pc[1]["R@5"])
        ),
        "intact_r1_nondecreasing": (
            float(candidate_pc[0]["R@1"]) >= float(baseline_pc[0]["R@1"])
        ),
        "destroyed_r1_drop_within_limit": (
            float(candidate_pc[2]["R@1"])
            >= float(baseline_pc[2]["R@1"]) - float(maximum_destroyed_drop)
        ),
        "macro_map_improves": (
            float(candidate[f"macro_mAP@{int(ap_k)}"])
            > float(baseline[f"macro_mAP@{int(ap_k)}"])
        ),
        "minimum_nonnegative_events": (
            int(event_diagnostics["nonnegative_events"])
            >= int(minimum_nonnegative_events)
        ),
    }
    return checks, event_diagnostics


def checkpoint_payload(
    model: CMRResidualMetric,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    validation: Mapping[str, object],
    args: argparse.Namespace,
) -> dict:
    severity_mode = str(args.gate_protocol) in ("m1bgs", "m1bgsp")
    preserve_mode = str(args.gate_protocol) == "m1bgsp"
    negative_names = {0: "Intact", 2: "Destroyed"}
    return {
        "protocol": (
            "CMR_M1BGSP_geometry_preserved_graph_metric_v2"
            if preserve_mode
            else "CMR_M1BGS_severity_aware_graph_metric_v1"
            if severity_mode
            else "CMR_M1BG_damaged_anchor_graph_metric_v1"
        ),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "validation": dict(validation),
        "config": vars(args),
        "loss_anchors": "Damaged only",
        "positive_source": "locked train-only global graph",
        "negative_classes": [negative_names[cls] for cls in args.negative_class_ids],
        "delta_cross_event_role": "diagnostic_only" if severity_mode else "gate",
        "preserve_weight": float(args.preserve_weight),
        "preserve_classes": ["Intact", "Destroyed"] if preserve_mode else [],
        "fuses_p3": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M1BG global-graph Damaged-anchor training")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--anchors_per_batch", type=int, default=24)
    parser.add_argument("--negatives_per_class", type=int, default=8)
    parser.add_argument("--negative_classes", default="0,2")
    parser.add_argument(
        "--gate_protocol", choices=("m1bg", "m1bgs", "m1bgsp"), default="m1bg"
    )
    parser.add_argument("--preserve_weight", type=float, default=0.0)
    parser.add_argument("--preserve_samples_per_class", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--query_chunk", type=int, default=1024)
    parser.add_argument("--project_batch_size", type=int, default=4096)
    parser.add_argument("--ap_k", type=int, default=10)
    parser.add_argument("--relation_pairs", type=int, default=200000)
    parser.add_argument("--minimum_damaged_gain", type=float, default=0.01)
    parser.add_argument("--maximum_destroyed_drop", type=float, default=0.02)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if min(args.epochs, args.steps_per_epoch, args.eval_every) <= 0:
        raise ValueError("epochs, steps_per_epoch and eval_every must be positive")
    if args.anchors_per_batch <= 0 or args.negatives_per_class <= 0:
        raise ValueError("batch and negative counts must be positive")
    if args.preserve_weight < 0 or args.preserve_samples_per_class <= 0:
        raise ValueError("preserve weight must be nonnegative and sample count positive")
    try:
        args.negative_class_ids = tuple(
            int(value.strip())
            for value in str(args.negative_classes).split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError("negative_classes must be comma-separated class IDs") from error
    if not args.negative_class_ids or any(cls not in (0, 2) for cls in args.negative_class_ids):
        raise ValueError("negative_classes may contain only 0 (Intact) and 2 (Destroyed)")
    if len(set(args.negative_class_ids)) != len(args.negative_class_ids):
        raise ValueError("negative_classes contains duplicates")
    if args.gate_protocol in ("m1bgs", "m1bgsp") and args.negative_class_ids != (0,):
        raise ValueError("M1BG-S/SP requires negative_classes=0 exactly")
    if args.gate_protocol == "m1bgsp" and args.preserve_weight <= 0:
        raise ValueError("M1BG-SP requires preserve_weight > 0")
    if args.gate_protocol != "m1bgsp" and args.preserve_weight != 0:
        raise ValueError("preserve_weight is only allowed by the M1BG-SP protocol")
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))
    # Preserve sampling must never consume the Anchor/Positive/Negative RNG.
    # This keeps M1BG-S and M1BG-SP InfoNCE batches exactly comparable.
    preserve_generator = torch.Generator().manual_seed(int(args.seed) + 104729)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_e, train_event_map, train_cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train"
    )
    val_x, val_y, val_e, val_event_map, val_cache = load_cmr_cache(
        args.val_cache, "mean", expected_role="query"
    )
    validate_disjoint_ids(train_cache["image_ids"], val_cache["image_ids"])
    if train_event_map != val_event_map:
        raise RuntimeError("Train/Validation event maps differ")
    train_sha = train_cache["metadata"].get("p3_checkpoint_sha256")
    if train_sha != val_cache["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("Train/Validation P3 checkpoint hashes differ")

    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    if graph.get("protocol") != "CMR_M1BG_train_only_global_damaged_graph_v1":
        raise RuntimeError("CMR-M1BG refuses an unknown graph protocol")
    if bool(graph.get("test_accessed", True)) or not bool(graph.get("passed", False)):
        raise RuntimeError("global graph did not pass the train-only audit")
    if graph.get("p3_checkpoint_sha256") != train_sha:
        raise RuntimeError("graph and CMR cache use different P3 checkpoints")
    damaged_indices = graph["damaged_train_indices"].long().flatten()
    neighbour_indices = graph["neighbour_train_indices"].long()
    if neighbour_indices.shape[0] != damaged_indices.numel():
        raise RuntimeError("graph anchor and neighbour counts differ")
    if not bool((train_y[damaged_indices] == 1).all()):
        raise RuntimeError("graph contains a non-Damaged anchor")
    if not bool((train_y[neighbour_indices] == 1).all()):
        raise RuntimeError("graph contains a non-Damaged positive")
    if bool((train_e[damaged_indices, None] == train_e[neighbour_indices]).any()):
        raise RuntimeError("graph contains a same-event positive")

    memory_x, memory_y, memory_e, memory_indices = frozen_memory_view(
        train_x, train_y, train_e, train_cache
    )
    raw_retrieval = cross_event_retrieval_eval(
        memory_x,
        memory_y,
        memory_e,
        val_x,
        val_y,
        val_e,
        k_values=(1, 5, 10),
        ap_k=args.ap_k,
        query_chunk=args.query_chunk,
        device=device,
    )
    raw_relation = relation_statistics(
        train_x,
        train_y,
        train_e,
        num_pairs=args.relation_pairs,
        seed=args.seed,
    )
    model = CMRResidualMetric(train_x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    with torch.inference_mode():
        check = min(256, len(train_x))
        identity_delta = float(
            (model(train_x[:check].to(device)).cpu() - F.normalize(train_x[:check], dim=1))
            .abs()
            .max()
        )
    if identity_delta > 1e-7:
        raise AssertionError(f"CMR-M1BG identity guard failed: {identity_delta}")

    if args.gate_protocol in ("m1bgs", "m1bgsp"):
        initial_checks, initial_event_diagnostics = severity_metric_gate(
            raw_retrieval,
            raw_retrieval,
            ap_k=args.ap_k,
            minimum_damaged_gain=args.minimum_damaged_gain,
            maximum_destroyed_drop=args.maximum_destroyed_drop,
            minimum_nonnegative_events=args.minimum_nonnegative_events,
        )
    else:
        initial_checks = selective_metric_gate(
            raw_retrieval,
            raw_retrieval,
            raw_relation,
            raw_relation,
            ap_k=args.ap_k,
            minimum_damaged_gain=args.minimum_damaged_gain,
            maximum_destroyed_drop=args.maximum_destroyed_drop,
        )
        initial_event_diagnostics = event_nonnegative_diagnostics(
            raw_retrieval, raw_retrieval, ap_k=args.ap_k
        )
    initial = {
        "epoch": 0,
        "global_step": 0,
        "retrieval": raw_retrieval,
        "relation_statistics": raw_relation,
        "checks": initial_checks,
        "event_diagnostics": initial_event_diagnostics,
        "passed": False,
        "identity_max_abs_delta": identity_delta,
    }
    history = [initial]
    best = initial
    global_step = 0
    preserve_pools = {
        cls: torch.nonzero(train_y == cls, as_tuple=False).flatten()
        for cls in (0, 2)
    }

    def selection_key(result: Mapping[str, object]) -> tuple:
        r = result["retrieval"]
        c = result["checks"]
        safe = bool(c["damaged_r5_nondecreasing"]) and bool(c["intact_r1_nondecreasing"]) and bool(c["destroyed_r1_drop_within_limit"])
        return (
            bool(result["passed"]),
            safe,
            sum(bool(value) for value in c.values()),
            float(r["per_class"][1]["R@1"]),
            float(r[f"macro_mAP@{args.ap_k}"]),
        )

    atomic_torch_save(
        checkpoint_payload(model, optimizer, epoch=0, global_step=0, validation=initial, args=args),
        output_dir / "best_metric.pth",
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        nce_loss_sum = 0.0
        preserve_loss_sum = 0.0
        for local_step in range(1, int(args.steps_per_epoch) + 1):
            rows = torch.randint(
                damaged_indices.numel(),
                (int(args.anchors_per_batch),),
                generator=generator,
            )
            columns = torch.randint(
                neighbour_indices.shape[1],
                (int(args.anchors_per_batch),),
                generator=generator,
            )
            anchor_indices = damaged_indices[rows]
            positive_indices = neighbour_indices[rows, columns]
            negative_indices = sample_cross_event_negatives(
                train_y,
                train_e,
                train_e[anchor_indices],
                negatives_per_class=args.negatives_per_class,
                negative_classes=args.negative_class_ids,
                generator=generator,
            )
            anchor_z = model(train_x[anchor_indices].to(device))
            positive_z = model(train_x[positive_indices].to(device))
            flat_negative = negative_indices.reshape(-1)
            negative_z = model(train_x[flat_negative].to(device)).reshape(
                args.anchors_per_batch, -1, train_x.shape[1]
            )
            nce_loss = damaged_anchor_nce_loss(
                anchor_z, positive_z, negative_z, temperature=args.temperature
            )
            if args.gate_protocol == "m1bgsp":
                guard_indices = []
                for cls in (0, 2):
                    pool = preserve_pools[cls]
                    take = int(args.preserve_samples_per_class)
                    if pool.numel() >= take:
                        local = torch.randperm(
                            pool.numel(), generator=preserve_generator
                        )[:take]
                    else:
                        local = torch.randint(
                            pool.numel(), (take,), generator=preserve_generator
                        )
                    guard_indices.append(pool[local])
                guard_indices = torch.cat(guard_indices)
                raw_guard = train_x[guard_indices].to(device)
                adapted_guard = model(raw_guard)
                preserve_loss = cosine_preservation_loss(adapted_guard, raw_guard)
            else:
                preserve_loss = nce_loss.new_zeros(())
            loss = nce_loss + float(args.preserve_weight) * preserve_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite M1BG loss at epoch={epoch}, step={local_step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            loss_sum += float(loss.detach())
            nce_loss_sum += float(nce_loss.detach())
            preserve_loss_sum += float(preserve_loss.detach())
            if global_step % int(args.log_every) == 0:
                print(
                    f"CMR-M1BG train | epoch={epoch} | step={global_step} | "
                    f"loss={float(loss):.6f} | nce={float(nce_loss):.6f} | "
                    f"preserve={float(preserve_loss):.6f}",
                    flush=True,
                )
        print(
            f"CMR-M1BG epoch complete | epoch={epoch} | "
            f"mean_loss={loss_sum / args.steps_per_epoch:.6f} | "
            f"mean_nce={nce_loss_sum / args.steps_per_epoch:.6f} | "
            f"mean_preserve={preserve_loss_sum / args.steps_per_epoch:.6f}",
            flush=True,
        )
        if epoch % int(args.eval_every) != 0 and epoch != int(args.epochs):
            continue
        model.eval()
        memory_z = project_features(model, memory_x, device=device, batch_size=args.project_batch_size)
        val_z = project_features(model, val_x, device=device, batch_size=args.project_batch_size)
        train_z = project_features(model, train_x, device=device, batch_size=args.project_batch_size)
        retrieval = cross_event_retrieval_eval(
            memory_z,
            memory_y,
            memory_e,
            val_z,
            val_y,
            val_e,
            k_values=(1, 5, 10),
            ap_k=args.ap_k,
            query_chunk=args.query_chunk,
            device=device,
        )
        relation = relation_statistics(
            train_z, train_y, train_e, num_pairs=args.relation_pairs, seed=args.seed
        )
        if args.gate_protocol in ("m1bgs", "m1bgsp"):
            checks, event_diagnostics = severity_metric_gate(
                retrieval,
                raw_retrieval,
                ap_k=args.ap_k,
                minimum_damaged_gain=args.minimum_damaged_gain,
                maximum_destroyed_drop=args.maximum_destroyed_drop,
                minimum_nonnegative_events=args.minimum_nonnegative_events,
            )
        else:
            checks = selective_metric_gate(
                retrieval,
                raw_retrieval,
                relation,
                raw_relation,
                ap_k=args.ap_k,
                minimum_damaged_gain=args.minimum_damaged_gain,
                maximum_destroyed_drop=args.maximum_destroyed_drop,
            )
            event_diagnostics = event_nonnegative_diagnostics(
                retrieval, raw_retrieval, ap_k=args.ap_k
            )
        result = {
            "epoch": epoch,
            "global_step": global_step,
            "retrieval": retrieval,
            "relation_statistics": relation,
            "checks": checks,
            "event_diagnostics": event_diagnostics,
            "passed": all(checks.values()),
        }
        raw_val_normalized = F.normalize(val_x, dim=1)
        result["embedding_drift"] = {
            str(cls): float(
                1.0 - (val_z[val_y == cls] * raw_val_normalized[val_y == cls]).sum(dim=1).mean()
            )
            for cls in range(3)
        }
        history.append(result)
        print("=" * 104)
        print(
            f"CMR-M1BG validation | epoch={epoch} | macro={retrieval[f'macro_mAP@{args.ap_k}']:.4f} | "
            f"I-R1={retrieval['per_class'][0]['R@1']:.4f} | D-R1={retrieval['per_class'][1]['R@1']:.4f} | "
            f"D-R5={retrieval['per_class'][1]['R@5']:.4f} | X-R1={retrieval['per_class'][2]['R@1']:.4f} | "
            f"events={event_diagnostics['nonnegative_events']}/{event_diagnostics['comparable_events']} | "
            f"drift(I/X)={result['embedding_drift']['0']:.5f}/{result['embedding_drift']['2']:.5f} | "
            f"DeltaCE={relation['delta_cross_event']:+.4f} | passed={result['passed']}",
            flush=True,
        )
        print(f"Checks: {checks}", flush=True)
        if selection_key(result) > selection_key(best):
            best = result
            atomic_torch_save(
                checkpoint_payload(model, optimizer, epoch=epoch, global_step=global_step, validation=result, args=args),
                output_dir / "best_metric.pth",
            )
        atomic_torch_save(
            checkpoint_payload(model, optimizer, epoch=epoch, global_step=global_step, validation=result, args=args),
            output_dir / "latest_resume.pth",
        )

    if args.gate_protocol == "m1bgsp":
        decision = (
            "ADMIT_CMR_M1BGSP_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
            if bool(best["passed"])
            else "STOP_GLOBAL_ADAPTER_THEN_USE_FROZEN_PAIRWISE_RERANKER"
        )
    elif args.gate_protocol == "m1bgs":
        decision = (
            "ADMIT_CMR_M1BGS_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
            if bool(best["passed"])
            else "STOP_OR_REVISE_SEVERITY_NEGATIVES_DO_NOT_FUSE_OR_TEST"
        )
    else:
        decision = (
            "ADMIT_CMR_M1BG_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
            if bool(best["passed"])
            else "STOP_GLOBAL_METRIC_ROUTE_DO_NOT_FUSE_OR_TEST"
        )
    summary = {
        "protocol": (
            "CMR_M1BGSP_geometry_preserved_graph_metric_summary_v2"
            if args.gate_protocol == "m1bgsp"
            else "CMR_M1BGS_severity_aware_graph_metric_summary_v1"
            if args.gate_protocol == "m1bgs"
            else "CMR_M1BG_damaged_anchor_graph_metric_summary_v1"
        ),
        "graph": str(args.graph),
        "train_regions": len(train_x),
        "memory_items": len(memory_indices),
        "validation_queries": len(val_x),
        "raw_retrieval": raw_retrieval,
        "raw_relation_statistics": raw_relation,
        "selected_epoch": best["epoch"],
        "selected_validation": best,
        "history": history,
        "negative_classes": list(args.negative_class_ids),
        "preserve_weight": float(args.preserve_weight),
        "preserve_samples_per_class": int(args.preserve_samples_per_class),
        "preserve_rng_seed": int(args.seed) + 104729,
        "main_rng_isolated_from_preserve_sampling": True,
        "delta_cross_event_role": (
            "diagnostic_only" if args.gate_protocol == "m1bgs" else "gate"
        ),
        "decision": decision,
        "fuses_p3": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Selected CMR-M1BG epoch: {best['epoch']}")
    print(f"Decision: {decision}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
