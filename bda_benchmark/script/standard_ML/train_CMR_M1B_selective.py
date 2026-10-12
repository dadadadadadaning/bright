# -*- coding: utf-8 -*-
"""CMR-M1B: selective-Damaged cross-event metric learning.

This is an isolated successor to CMR-M1.  It keeps the frozen Mean key and P0
Memory, but changes only the definition of Damaged positives: for every
Damaged anchor, only the top-k cross-event Damaged neighbours in the frozen
raw-Mean space are pulled closer.  Non-selected Damaged pairs are ignored.

P3, CMR-P0, CMR-M1, CMR-M2 and Test data are never modified or accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from model.cmr.metric import CMRResidualMetric, selective_damaged_supcon_loss
from script.standard_ML.cmr_metric_utils import (
    class_event_index,
    cross_event_retrieval_eval,
    frozen_memory_view,
    load_cmr_cache,
    relation_statistics,
    sample_class_event_batch,
)
from script.standard_ML.rm_region_retrieval_runtime import validate_disjoint_ids


@torch.inference_mode()
def project_features(
    model: CMRResidualMetric,
    feature: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, feature.shape[0], int(batch_size)):
        outputs.append(model(feature[start : start + int(batch_size)].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def selective_metric_gate(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    candidate_relation: Mapping[str, object],
    baseline_relation: Mapping[str, object],
    *,
    ap_k: int,
    minimum_damaged_gain: float,
    maximum_destroyed_drop: float,
) -> Dict[str, bool]:
    """Predeclared M1B gate: Damaged gain may not spend Intact accuracy."""

    candidate_pc = candidate["per_class"]
    baseline_pc = baseline["per_class"]
    return {
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
        "delta_cross_event_nondecreasing": (
            float(candidate_relation["delta_cross_event"])
            >= float(baseline_relation["delta_cross_event"])
        ),
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: CMRResidualMetric,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    validation: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "protocol": "CMR_M1B_selective_damaged_metric_v1",
        "variant": "mean",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "validation": dict(validation),
        "config": vars(args),
        "positive_policy": {
            "intact": "all same-class cross-event pairs",
            "damaged": "raw-Mean top-k same-class cross-event pairs only",
            "destroyed": "all same-class cross-event pairs",
            "unselected_damaged_pairs": "ignored, never negatives",
        },
        "fuses_p3": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M1B selective-Damaged metric learning")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--events_per_class", type=int, default=4)
    parser.add_argument("--samples_per_event", type=int, default=2)
    parser.add_argument("--damaged_positive_topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--query_chunk", type=int, default=1024)
    parser.add_argument("--project_batch_size", type=int, default=4096)
    parser.add_argument("--ap_k", type=int, default=10)
    parser.add_argument("--relation_pairs", type=int, default=200000)
    parser.add_argument("--minimum_damaged_gain", type=float, default=0.01)
    parser.add_argument("--maximum_destroyed_drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--print_cache_schema", action="store_true")
    args = parser.parse_args()

    if args.epochs < 0 or args.steps_per_epoch <= 0 or args.eval_every <= 0:
        raise ValueError("epochs must be nonnegative; step/eval intervals must be positive")
    if args.damaged_positive_topk <= 0:
        raise ValueError("damaged_positive_topk must be positive")
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_e, train_event_map, train_cache = load_cmr_cache(
        args.train_cache,
        "mean",
        expected_role="train",
        print_schema=args.print_cache_schema,
    )
    val_x, val_y, val_e, val_event_map, val_cache = load_cmr_cache(
        args.val_cache,
        "mean",
        expected_role="query",
        print_schema=args.print_cache_schema,
    )
    validate_disjoint_ids(train_cache["image_ids"], val_cache["image_ids"])
    if train_event_map != val_event_map:
        raise RuntimeError("Train/Validation event maps differ")
    if train_cache["metadata"].get("p3_checkpoint_sha256") != val_cache[
        "metadata"
    ].get("p3_checkpoint_sha256"):
        raise RuntimeError("Train/Validation P3 checkpoint hashes differ")

    memory_x, memory_y, memory_e, memory_indices = frozen_memory_view(
        train_x, train_y, train_e, train_cache
    )
    print("\nCMR-M1B frozen experiment contract", flush=True)
    print(f"Train regions     : {len(train_x)}")
    print(f"Memory items      : {len(memory_indices)}")
    print(f"Validation queries: {len(val_x)}")
    print(f"Damaged positives : raw-Mean top-{args.damaged_positive_topk}")
    print("P3/Memory/Test    : frozen/frozen/not accessed", flush=True)

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
        check_count = min(256, train_x.shape[0])
        expected = F.normalize(train_x[:check_count], dim=1)
        observed = model(train_x[:check_count].to(device)).cpu()
        identity_delta = float((observed - expected).abs().max())
    if identity_delta > 1e-7:
        raise AssertionError(f"CMR-M1B identity guard failed: {identity_delta}")

    pools = class_event_index(train_y, train_e)
    generator = torch.Generator().manual_seed(int(args.seed))
    initial_checks = selective_metric_gate(
        raw_retrieval,
        raw_retrieval,
        raw_relation,
        raw_relation,
        ap_k=args.ap_k,
        minimum_damaged_gain=args.minimum_damaged_gain,
        maximum_destroyed_drop=args.maximum_destroyed_drop,
    )
    initial = {
        "epoch": 0,
        "global_step": 0,
        "retrieval": raw_retrieval,
        "relation_statistics": raw_relation,
        "checks": initial_checks,
        "passed": False,
        "identity_max_abs_delta": identity_delta,
    }
    history = [initial]
    best = initial
    global_step = 0

    def selection_key(result: Mapping[str, object]) -> tuple:
        retrieval = result["retrieval"]
        checks = result["checks"]
        safe = (
            bool(checks["damaged_r5_nondecreasing"])
            and bool(checks["intact_r1_nondecreasing"])
            and bool(checks["destroyed_r1_drop_within_limit"])
        )
        return (
            bool(result["passed"]),
            safe,
            sum(bool(value) for value in checks.values()),
            float(retrieval["per_class"][1]["R@1"]),
            float(retrieval[f"macro_mAP@{int(args.ap_k)}"]),
            float(retrieval["per_class"][0]["R@1"]),
        )

    atomic_torch_save(
        checkpoint_payload(
            model,
            optimizer,
            epoch=0,
            global_step=0,
            validation=initial,
            args=args,
        ),
        output_dir / "best_metric.pth",
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        for local_step in range(1, int(args.steps_per_epoch) + 1):
            batch = sample_class_event_batch(
                pools,
                events_per_class=args.events_per_class,
                samples_per_event=args.samples_per_event,
                generator=generator,
            )
            raw_batch = train_x[batch].to(device)
            embedding = model(raw_batch)
            loss = selective_damaged_supcon_loss(
                embedding,
                raw_batch,
                train_y[batch].to(device),
                train_e[batch].to(device),
                temperature=args.temperature,
                damaged_topk=args.damaged_positive_topk,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite CMR-M1B loss at epoch={epoch}, step={local_step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            epoch_loss += float(loss.detach())
            if global_step % int(args.log_every) == 0:
                print(
                    f"CMR-M1B train | epoch={epoch} | step={global_step} | "
                    f"loss={float(loss):.6f}",
                    flush=True,
                )
        print(
            f"CMR-M1B epoch complete | epoch={epoch} | "
            f"mean_loss={epoch_loss / int(args.steps_per_epoch):.6f}",
            flush=True,
        )

        if epoch % int(args.eval_every) != 0 and epoch != int(args.epochs):
            continue
        model.eval()
        memory_z = project_features(
            model, memory_x, device=device, batch_size=args.project_batch_size
        )
        val_z = project_features(
            model, val_x, device=device, batch_size=args.project_batch_size
        )
        train_z = project_features(
            model, train_x, device=device, batch_size=args.project_batch_size
        )
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
            train_z,
            train_y,
            train_e,
            num_pairs=args.relation_pairs,
            seed=args.seed,
        )
        checks = selective_metric_gate(
            retrieval,
            raw_retrieval,
            relation,
            raw_relation,
            ap_k=args.ap_k,
            minimum_damaged_gain=args.minimum_damaged_gain,
            maximum_destroyed_drop=args.maximum_destroyed_drop,
        )
        result = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "retrieval": retrieval,
            "relation_statistics": relation,
            "checks": checks,
            "passed": all(checks.values()),
        }
        history.append(result)
        print("=" * 104)
        print(
            f"CMR-M1B validation | epoch={epoch} | "
            f"macro-mAP={retrieval[f'macro_mAP@{args.ap_k}']:.4f} | "
            f"I-R1={retrieval['per_class'][0]['R@1']:.4f} | "
            f"D-R1={retrieval['per_class'][1]['R@1']:.4f} | "
            f"D-R5={retrieval['per_class'][1]['R@5']:.4f} | "
            f"X-R1={retrieval['per_class'][2]['R@1']:.4f} | "
            f"DeltaCE={relation['delta_cross_event']:+.4f} | "
            f"passed={result['passed']}",
            flush=True,
        )
        print(f"Checks: {checks}", flush=True)
        if selection_key(result) > selection_key(best):
            best = result
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    validation=result,
                    args=args,
                ),
                output_dir / "best_metric.pth",
            )
            print(f"New best CMR-M1B checkpoint: epoch={epoch}", flush=True)
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                validation=result,
                args=args,
            ),
            output_dir / "latest_resume.pth",
        )

    decision = (
        "ADMIT_CMR_M1B_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
        if bool(best["passed"])
        else "STOP_GLOBAL_METRIC_ROUTE_DO_NOT_FUSE_OR_TEST"
    )
    summary = {
        "protocol": "CMR_M1B_selective_damaged_metric_summary_v1",
        "variant": "mean",
        "train_regions": int(train_x.shape[0]),
        "memory_items": int(memory_indices.numel()),
        "validation_queries": int(val_x.shape[0]),
        "raw_retrieval": raw_retrieval,
        "raw_relation_statistics": raw_relation,
        "selected_epoch": int(best["epoch"]),
        "selected_validation": best,
        "history": history,
        "selection_order": (
            "strict pass -> I/D-R5/X safety -> passed-check count -> "
            "Damaged R@1 -> Macro mAP -> Intact R@1"
        ),
        "controlled_change": (
            "Damaged positives are raw-Mean top-k cross-event neighbours; "
            "all other M1 settings remain fixed"
        ),
        "gate": {
            "damaged_R@1": f"raw + {args.minimum_damaged_gain}",
            "damaged_R@5": "nondecreasing",
            "intact_R@1": "nondecreasing",
            "destroyed_R@1": f"drop <= {args.maximum_destroyed_drop}",
            "macro_mAP": "strictly improves",
            "delta_cross_event": "nondecreasing",
        },
        "decision": decision,
        "fuses_p3": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected CMR-M1B epoch: {best['epoch']}")
    print(f"Decision: {decision}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
