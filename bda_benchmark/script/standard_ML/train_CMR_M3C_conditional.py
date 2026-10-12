# -*- coding: utf-8 -*-
"""CMR-M3C: candidate-label conditional Top-20 pairwise reranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from model.cmr.pairwise import (
    CMRCandidateConditionalPairwiseScorer,
    class_balanced_hard_top1_loss,
    multi_positive_candidate_loss,
)
from script.standard_ML.cmr_metric_utils import class_event_index, sample_class_event_batch
from script.standard_ML.cmr_pairwise_utils import (
    candidate_metrics,
    per_event_r1_diagnostics,
)
from script.standard_ML.train_CMR_M3_pairwise import load_candidates


@torch.no_grad()
def evaluate_conditional(
    model: CMRCandidateConditionalPairwiseScorer,
    payload: Mapping[str, object],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict, torch.Tensor]:
    query = payload["query_features"].float()
    memory = payload["memory_features"].float()
    indices = payload["candidate_indices"].long()
    base = payload["candidate_scores"].float()
    labels = payload["candidate_labels"].long()
    ranked_chunks = []
    model.eval()
    for start in range(0, query.shape[0], int(batch_size)):
        end = min(start + int(batch_size), query.shape[0])
        local_labels = labels[start:end].to(device)
        q = query[start:end].to(device)[:, None, :].expand(-1, indices.shape[1], -1)
        c = memory[indices[start:end]].to(device)
        score = base[start:end].to(device) + model(q, c, local_labels)
        order = torch.argsort(score, dim=1, descending=True, stable=True).cpu()
        ranked_chunks.append(torch.gather(labels[start:end], 1, order))
    ranked = torch.cat(ranked_chunks)
    return candidate_metrics(ranked, payload["query_labels"]), ranked


def admission_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    event_diagnostics: Mapping[str, object],
    *,
    minimum_damaged_gain: float,
    maximum_destroyed_drop: float,
    minimum_damaged_events: int,
) -> dict:
    return {
        "damaged_r1_plus_declared_gain": (
            candidate["per_class"][1]["R@1"]
            >= baseline["per_class"][1]["R@1"] + float(minimum_damaged_gain)
        ),
        "damaged_r5_nondecreasing": (
            candidate["per_class"][1]["R@5"] >= baseline["per_class"][1]["R@5"]
        ),
        "damaged_nonnegative_events": (
            int(event_diagnostics["damaged_nonnegative_events"])
            >= int(minimum_damaged_events)
        ),
        "intact_r1_nondecreasing": (
            candidate["per_class"][0]["R@1"] >= baseline["per_class"][0]["R@1"]
        ),
        "destroyed_r1_drop_within_limit": (
            candidate["per_class"][2]["R@1"]
            >= baseline["per_class"][2]["R@1"] - float(maximum_destroyed_drop)
        ),
        "macro_map_nondecreasing": candidate["macro_mAP@10"] >= baseline["macro_mAP@10"],
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M3C candidate-conditional reranker")
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--oracle_result", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--events_per_class", type=int, default=4)
    parser.add_argument("--samples_per_event", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--maximum_residual", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument(
        "--loss_protocol",
        choices=("listwise", "class_balanced_top1"),
        default="listwise",
    )
    parser.add_argument("--top1_margin", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--minimum_damaged_gain", type=float, default=0.01)
    parser.add_argument("--maximum_destroyed_drop", type=float, default=0.01)
    parser.add_argument("--minimum_damaged_events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    oracle = json.loads(Path(args.oracle_result).read_text(encoding="utf-8"))
    if not bool(oracle.get("passed", False)):
        raise RuntimeError("Oracle gate failed; refusing CMR-M3C training")
    train = load_candidates(args.train_candidates, "train")
    val = load_candidates(args.val_candidates, "validation")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))

    model = CMRCandidateConditionalPairwiseScorer(
        train["query_features"].shape[1],
        hidden_dim=args.hidden_dim,
        maximum_residual=args.maximum_residual,
        num_candidate_classes=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pools = class_event_index(train["query_labels"], train["query_events"])
    query = train["query_features"].float()
    memory = train["memory_features"].float()
    indices = train["candidate_indices"].long()
    base = train["candidate_scores"].float()
    candidate_labels = train["candidate_labels"].long()
    query_labels = train["query_labels"].long()

    # Fixed budget and identical M3 sampler/loss. Validation is used once.
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        used = 0
        for step in range(1, int(args.steps_per_epoch) + 1):
            rows = sample_class_event_batch(
                pools,
                events_per_class=args.events_per_class,
                samples_per_event=args.samples_per_event,
                generator=generator,
            )
            labels = candidate_labels[rows]
            eligible = (labels == query_labels[rows, None]).any(dim=1) & (
                labels != query_labels[rows, None]
            ).any(dim=1)
            rows = rows[eligible]
            if rows.numel() == 0:
                continue
            local_indices = indices[rows]
            local_labels = candidate_labels[rows].to(device)
            q = query[rows].to(device)[:, None, :].expand(-1, local_indices.shape[1], -1)
            c = memory[local_indices].to(device)
            score = base[rows].to(device) + model(q, c, local_labels)
            if args.loss_protocol == "class_balanced_top1":
                loss = class_balanced_hard_top1_loss(
                    score,
                    local_labels,
                    query_labels[rows].to(device),
                    temperature=args.temperature,
                    margin=args.top1_margin,
                )
            else:
                loss = multi_positive_candidate_loss(
                    score,
                    local_labels,
                    query_labels[rows].to(device),
                    temperature=args.temperature,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            used += 1
            if step % 50 == 0:
                print(
                    f"CMR-M3C train | epoch={epoch} | step={step} | "
                    f"loss={running/max(used,1):.6f}",
                    flush=True,
                )
        print(
            f"CMR-M3C epoch complete | epoch={epoch} | "
            f"mean_loss={running/max(used,1):.6f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_top1 = args.loss_protocol == "class_balanced_top1"
    checkpoint = {
        "protocol": (
            "CMR_M3T_candidate_conditional_top1_v1"
            if is_top1 else "CMR_M3C_candidate_label_conditional_v1"
        ),
        "model": model.state_dict(),
        "config": vars(args),
        "candidate_label_source": "train-only Memory metadata",
        "query_label_is_model_input": False,
        "test_accessed": False,
    }
    torch.save(
        checkpoint,
        output_dir / (
            "final_top1_pairwise.pth" if is_top1 else "final_conditional_pairwise.pth"
        ),
    )

    split_results = {}
    for role, payload in (("train", train), ("validation", val)):
        baseline_ranked = payload["candidate_labels"].long()
        baseline = candidate_metrics(baseline_ranked, payload["query_labels"])
        candidate, ranked = evaluate_conditional(model, payload, device=device)
        event_diagnostics = per_event_r1_diagnostics(
            baseline_ranked,
            ranked,
            payload["query_labels"],
            payload["query_events"],
        )
        split_results[role] = {
            "baseline": baseline,
            "candidate": candidate,
            "event_diagnostics": event_diagnostics,
        }

    validation = split_results["validation"]
    checks = admission_checks(
        validation["candidate"],
        validation["baseline"],
        validation["event_diagnostics"],
        minimum_damaged_gain=args.minimum_damaged_gain,
        maximum_destroyed_drop=args.maximum_destroyed_drop,
        minimum_damaged_events=args.minimum_damaged_events,
    )
    passed = all(checks.values())
    summary = {
        "protocol": (
            "CMR_M3T_class_balanced_hard_top1_validation_once_v1"
            if is_top1 else "CMR_M3C_fixed_budget_validation_once_v1"
        ),
        "controlled_change": (
            "Listwise loss -> class-balanced best-positive vs hardest-negative Top-1 loss"
            if is_top1 else "shared final head -> candidate-label conditional I/D/X heads"
        ),
        "unchanged": [
            "Raw Mean Top-20", "candidate caches", "relation features",
            "sampler", "training budget", "residual bound",
        ],
        "loss_protocol": args.loss_protocol,
        "top1_margin": float(args.top1_margin),
        "results": split_results,
        "checks": checks,
        "passed": passed,
        "decision": (
            (
                "ADMIT_CMR_M3T_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
                if passed else "STOP_LEARNED_PAIRWISE_RELATION_FEATURES_INSUFFICIENT"
            )
            if is_top1
            else (
                "ADMIT_CMR_M3C_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
                if passed else "STOP_CONDITIONAL_HEAD_THEN_TEST_TOP1_LOSS_DO_NOT_FUSE"
            )
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 112)
    print(
        "CMR-M3T class-balanced hard Top-1 validation"
        if is_top1 else "CMR-M3C candidate-label conditional validation"
    )
    for role in ("train", "validation"):
        item = split_results[role]
        for name, result in (("Raw", item["baseline"]), ("Conditional", item["candidate"])):
            print(
                f"{role:<10} {name:<11} | macro={result['macro_mAP@10']:.4f} | "
                f"I-R1={result['per_class'][0]['R@1']:.4f} | "
                f"D-R1={result['per_class'][1]['R@1']:.4f} | "
                f"D-R5={result['per_class'][1]['R@5']:.4f} | "
                f"X-R1={result['per_class'][2]['R@1']:.4f}"
            )
        diagnostics = item["event_diagnostics"]
        print(
            f"{role:<10} events      | macro={diagnostics['macro_nonnegative_events']} | "
            f"Damaged={diagnostics['damaged_nonnegative_events']}/"
            f"{diagnostics['damaged_comparable_events']}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {summary['decision']}")
    print(f"Saved           : {output_dir}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()
