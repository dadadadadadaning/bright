from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import PROTOCOL
from .audit_e0 import load, load_metadata, train_key
from .core import balanced_event_class_rows, binary_auroc, binary_f1, cross_event_topk, event_auc


class LocalSupportHead(nn.Module):
    """Same-capacity head for every support control in LSM-E1."""
    def __init__(self, query_dim: int, support_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(query_dim + support_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((query, support), 1)).squeeze(1)


def weighted_support(retrieval, values: torch.Tensor) -> torch.Tensor:
    valid = retrieval.indices >= 0
    weights = torch.softmax(retrieval.scores.masked_fill(~valid, -1e9), 1) * valid
    return (values[retrieval.indices.clamp_min(0)].float() * weights[..., None]).sum(1)


def event_f1(target: torch.Tensor, prediction: torch.Tensor, events: list[str]) -> dict[str, float]:
    output = {}
    for event in sorted(set(events)):
        mask = torch.tensor([item == event for item in events])
        if bool(mask.any()): output[event] = binary_f1(target[mask], prediction[mask])
    return output


def train_variant(name: str, query_train: torch.Tensor, support_train: torch.Tensor, target_train: torch.Tensor,
                  query_eval: torch.Tensor, support_eval: torch.Tensor, target_eval: torch.Tensor, *,
                  steps: int, batch_size: int, hidden: int, lr: float, device: torch.device, seed: int) -> tuple[LocalSupportHead, torch.Tensor]:
    torch.manual_seed(seed)
    model = LocalSupportHead(query_train.shape[1], support_train.shape[1], hidden).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = target_train.sum().float(); neg = len(target_train) - pos
    pos_weight = (neg / pos.clamp_min(1)).to(device)
    generator = torch.Generator().manual_seed(seed + 17)
    for step in range(1, steps + 1):
        row = torch.randint(len(target_train), (min(batch_size, len(target_train)),), generator=generator)
        logits = model(query_train[row].to(device), support_train[row].to(device))
        loss = F.binary_cross_entropy_with_logits(logits, target_train[row].float().to(device), pos_weight=pos_weight)
        optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(json.dumps({"variant": name, "step": step, "loss": float(loss.detach()), "positive": int(target_train[row].sum())}), flush=True)
    with torch.no_grad():
        score = torch.sigmoid(model(query_eval.to(device), support_eval.to(device))).cpu()
    return model.cpu(), score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("LSM-E1: true-neighbour dependency test")
    parser.add_argument("--train-cache", required=True); parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--e0-result", required=True); parser.add_argument("--output", required=True); parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--minimum-purity", type=float, default=.60); parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--maximum-train-queries-per-event-class", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    e0 = json.loads(Path(args.e0_result).read_text(encoding="utf-8"))
    if e0.get("decision") != "ADMIT_LSM_E1_TRUE_NEIGHBOUR_DEPENDENCY":
        raise RuntimeError("LSM-E0 did not admit E1; do not train support heads after a failed retrieval audit")
    train_meta = load_metadata(args.train_cache)
    memory_rows = balanced_event_class_rows(train_meta["labels"], train_meta["row_events"],
                                            maximum_per_event_class=args.maximum_per_event_class, seed=args.seed,
                                            eligible=train_meta["label_purity"].float() >= args.minimum_purity)
    train_query_rows = balanced_event_class_rows(
        train_meta["labels"], train_meta["row_events"], maximum_per_event_class=args.maximum_train_queries_per_event_class,
        seed=args.seed + 19, eligible=(train_meta["label_purity"].float() >= args.minimum_purity) & (train_meta["p3_predictions"].long() <= 1),
    )
    memory, train, query = load(args.train_cache, memory_rows), load(args.train_cache, train_query_rows), load(args.calibration_cache)
    fields = ("rgb_key", "sar_key", "fused_key", "structure_key")
    memory_key, train_key_value = train_key(memory, train, fields)
    _, query_key = train_key(memory, query, fields)
    train_events = train["row_events"]
    train_labels = train["labels"]
    # Every training query is retrieved from another event, preventing same-event
    # copies and ensuring the test protocol matches training-time use.
    train_retrieval = cross_event_topk(train_key_value, memory_key, train_events, memory["row_events"],
                                       topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device)
    query_retrieval = cross_event_topk(query_key, memory_key, query["row_events"], memory["row_events"],
                                       topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device)
    support_value = memory["fused_key"].float()
    true_train, true_query = weighted_support(train_retrieval, support_value), weighted_support(query_retrieval, support_value)
    zero_train, zero_query = torch.zeros_like(true_train), torch.zeros_like(true_query)
    generator = torch.Generator().manual_seed(args.seed + 101)
    train_permutation = torch.randperm(len(train_key_value), generator=generator)
    query_permutation = torch.randperm(len(query_key), generator=generator)
    random_train_r = cross_event_topk(train_key_value[train_permutation], memory_key, train_events, memory["row_events"], topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device)
    random_query_r = cross_event_topk(query_key[query_permutation], memory_key, query["row_events"], memory["row_events"], topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device)
    random_train, random_query = weighted_support(random_train_r, support_value), weighted_support(random_query_r, support_value)
    shuffled_value = support_value[torch.randperm(len(support_value), generator=torch.Generator().manual_seed(args.seed + 102))]
    shuffled_train, shuffled_query = weighted_support(train_retrieval, shuffled_value), weighted_support(query_retrieval, shuffled_value)
    candidate_train = torch.ones(len(train_labels), dtype=torch.bool); candidate_query = query["p3_predictions"].long() <= 1
    target_train = (train_labels.long() >= 2).long(); target_query = (query["labels"].long() >= 2).long()
    variants = {
        "e1_a0_no_retrieval": (zero_train, zero_query), "e1_a1_true_neighbours": (true_train, true_query),
        "e1_a2_random_neighbours": (random_train, random_query), "e1_a3_shuffled_neighbours": (shuffled_train, shuffled_query),
        "e1_a4_zero_support": (zero_train, zero_query),
    }
    reports = {}; Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    for number, (name, (support_train, support_query)) in enumerate(variants.items()):
        model, score = train_variant(name, train_key_value[candidate_train], support_train[candidate_train], target_train[candidate_train],
                                     query_key[candidate_query], support_query[candidate_query], target_query[candidate_query],
                                     steps=args.steps, batch_size=args.batch_size, hidden=args.hidden, lr=args.learning_rate,
                                     device=torch.device(args.device if torch.cuda.is_available() else "cpu"), seed=args.seed)
        local_target = target_query[candidate_query]; prediction = score >= .5
        events = [event for event, keep in zip(query["row_events"], candidate_query.tolist()) if keep]
        per_event = event_f1(local_target, prediction, events)
        reports[name] = {"affected_miss_f1": binary_f1(local_target, prediction), "affected_miss_auroc": binary_auroc(local_target, score),
                         "event_macro_f1": sum(per_event.values()) / max(len(per_event), 1), "nonnegative_events": sum(value > 0 for value in per_event.values()),
                         "event_total": len(per_event), "events": per_event, "candidate_count": int(candidate_query.sum())}
        torch.save({"protocol": PROTOCOL, "variant": name, "model_state_dict": model.state_dict(), "configuration": vars(args)}, Path(args.checkpoint_dir) / f"{name}.pth")
        print(json.dumps({"variant": name, **reports[name]}, ensure_ascii=False), flush=True)
    true = reports["e1_a1_true_neighbours"]
    checks = {
        "true_beats_no_retrieval_by_0p50pp": true["affected_miss_f1"] >= reports["e1_a0_no_retrieval"]["affected_miss_f1"] + .005,
        "true_beats_random_by_0p50pp": true["affected_miss_f1"] >= reports["e1_a2_random_neighbours"]["affected_miss_f1"] + .005,
        "true_beats_shuffled_by_0p50pp": true["affected_miss_f1"] >= reports["e1_a3_shuffled_neighbours"]["affected_miss_f1"] + .005,
        "true_beats_zero_by_0p50pp": true["affected_miss_f1"] >= reports["e1_a4_zero_support"]["affected_miss_f1"] + .005,
        "affected_miss_auroc_ge_0p65": true["affected_miss_auroc"] >= .65,
        "nonnegative_events_ge_10": true["nonnegative_events"] >= 10,
    }
    output = {"protocol": PROTOCOL, "stage": "LSM_E1_TRUE_NEIGHBOUR_DEPENDENCY", "configuration": vars(args), "reports": reports,
              "training_query_rows": int(len(train_labels)), "checks": checks,
              "decision": "ADMIT_LSM_E2_PIXEL_LEVEL_INTEGRATION_DESIGN" if all(checks.values()) else "STOP_LSM_AFTER_E1_MEMORY_NOT_CAUSAL",
              "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": output["decision"], "checks": checks}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
