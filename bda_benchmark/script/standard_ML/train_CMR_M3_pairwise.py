# -*- coding: utf-8 -*-
"""CMR-M3P: train a bounded scorer over frozen Raw Mean Top-20 candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from model.cmr.pairwise import CMRBoundedPairwiseScorer, multi_positive_candidate_loss
from script.standard_ML.cmr_metric_utils import class_event_index, sample_class_event_batch
from script.standard_ML.cmr_pairwise_utils import (
    candidate_metrics,
    per_event_nonnegative_count,
)


def load_candidates(path: str, role: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("CMR-M3 requires its frozen candidate cache")
    if payload.get("role") != role or bool(payload.get("test_accessed", True)):
        raise RuntimeError("candidate role/Test contract violation")
    return payload


@torch.no_grad()
def evaluate(
    model: CMRBoundedPairwiseScorer,
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
        q = query[start:end].to(device)[:, None, :].expand(-1, indices.shape[1], -1)
        c = memory[indices[start:end]].to(device)
        score = base[start:end].to(device) + model(q, c)
        order = torch.argsort(score, dim=1, descending=True, stable=True).cpu()
        ranked_chunks.append(torch.gather(labels[start:end], 1, order))
    ranked = torch.cat(ranked_chunks)
    return candidate_metrics(ranked, payload["query_labels"]), ranked


def validation_checks(candidate: Mapping[str, object], baseline: Mapping[str, object], events: int) -> dict:
    return {
        "damaged_r1_plus_0_01": candidate["per_class"][1]["R@1"] >= baseline["per_class"][1]["R@1"] + 0.01,
        "damaged_r5_nondecreasing": candidate["per_class"][1]["R@5"] >= baseline["per_class"][1]["R@5"],
        "intact_r1_nondecreasing": candidate["per_class"][0]["R@1"] >= baseline["per_class"][0]["R@1"],
        "destroyed_r1_drop_at_most_0_01": candidate["per_class"][2]["R@1"] >= baseline["per_class"][2]["R@1"] - 0.01,
        "macro_map_improves": candidate["macro_mAP@10"] > baseline["macro_mAP@10"],
        "ten_events_nondecreasing": int(events) >= 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M3 bounded pairwise reranker")
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
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    oracle = json.loads(Path(args.oracle_result).read_text(encoding="utf-8"))
    if not bool(oracle.get("passed", False)):
        raise RuntimeError("Oracle gate failed; refusing pairwise training")
    train = load_candidates(args.train_candidates, "train")
    val = load_candidates(args.val_candidates, "validation")
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    model = CMRBoundedPairwiseScorer(
        train["query_features"].shape[1],
        hidden_dim=args.hidden_dim,
        maximum_residual=args.maximum_residual,
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

    # Fixed training budget; Validation is not used for epoch selection.
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
            candidate_index = indices[rows]
            q = query[rows].to(device)[:, None, :].expand(-1, candidate_index.shape[1], -1)
            c = memory[candidate_index].to(device)
            score = base[rows].to(device) + model(q, c)
            loss = multi_positive_candidate_loss(
                score, candidate_labels[rows].to(device), query_labels[rows].to(device),
                temperature=args.temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            used += 1
            if step % 50 == 0:
                print(f"CMR-M3 train | epoch={epoch} | step={step} | loss={running/max(used,1):.6f}", flush=True)
        print(f"CMR-M3 epoch complete | epoch={epoch} | mean_loss={running/max(used,1):.6f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": "CMR_M3_bounded_pairwise_reranker_v1",
        "model": model.state_dict(),
        "config": vars(args),
        "test_accessed": False,
    }, output_dir / "final_pairwise.pth")

    baseline_labels = val["candidate_labels"].long()
    baseline = candidate_metrics(baseline_labels, val["query_labels"])
    candidate, ranked = evaluate(model, val, device=device)
    event_count, event_delta = per_event_nonnegative_count(
        baseline_labels, ranked, val["query_labels"], val["query_events"]
    )
    checks = validation_checks(candidate, baseline, event_count)
    passed = all(checks.values())
    summary = {
        "protocol": "CMR_M3_fixed_budget_validation_once_v1",
        "baseline": baseline,
        "candidate": candidate,
        "event_nonnegative": int(event_count),
        "event_delta_r1_macro": event_delta,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_CMR_M3_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
            if passed else "STOP_PAIRWISE_RERANKER_DO_NOT_FUSE_OR_TEST"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 104)
    print("CMR-M3 fixed-budget pairwise validation")
    for name, result in (("Raw Mean", baseline), ("Pairwise", candidate)):
        print(
            f"{name:<10} | macro={result['macro_mAP@10']:.4f} | "
            f"I-R1={result['per_class'][0]['R@1']:.4f} | "
            f"D-R1={result['per_class'][1]['R@1']:.4f} | "
            f"D-R5={result['per_class'][1]['R@5']:.4f} | "
            f"X-R1={result['per_class'][2]['R@1']:.4f}"
        )
    print(f"Nonnegative events : {event_count}")
    print(f"Checks             : {checks}")
    print(f"Decision           : {summary['decision']}")
    print(f"Saved              : {output_dir}")
    print("Test accessed      : False")


if __name__ == "__main__":
    main()

