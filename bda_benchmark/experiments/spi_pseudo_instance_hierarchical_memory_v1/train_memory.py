from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.model.PseudoInstanceHierarchicalMemory import (
    PseudoInstanceHierarchicalMemory, semantic_labels_from_delta,
)
from . import PROTOCOL
from .core import (DEFAULT_GROUPS, balanced_memory_indices,
                   classification_metrics, feature_matrix, parent_areas,
                   parent_ids, provenance)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: str, split: str) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("split") != split:
        raise RuntimeError(f"expected {split}, got {cache.get('split')}")
    if cache.get("meta_confirm_accessed", False) or cache.get("val450_accessed", False) or cache.get("test_accessed", False):
        raise RuntimeError("forbidden cache supplied to memory development")
    return cache


def prepare(cache: dict, groups, mean=None, std=None):
    raw = feature_matrix(cache, groups)
    labels = semantic_labels_from_delta(cache["direction_code"], cache["delta_hist"])
    keep = labels >= 0
    raw, labels = raw[keep], labels[keep]
    if mean is None:
        mean = raw.mean(0); std = raw.std(0, unbiased=False).clamp_min(1e-4)
    x = ((raw - mean) / std).clamp(-8, 8)
    parent = parent_ids(cache)[keep]
    area = cache["cluster_area"].float()[keep]
    event, image = provenance(cache)
    return {
        "x": x, "label": labels, "parent": parent,
        "direction": cache["direction_code"].long()[keep], "area": area,
        "parent_area": parent_areas(parent, area),
        "event": event[keep], "image": image[keep], "keep": keep,
    }, mean, std


def probabilities(output: dict) -> torch.Tensor:
    occurrence = output["occurrence_logits"].softmax(1)
    severity = output["severity_logits"].softmax(1)
    return torch.stack((occurrence[:, 0], occurrence[:, 1] * severity[:, 0],
                        occurrence[:, 1] * severity[:, 1]), 1)


def forward(model, data, use_memory):
    return model(data["x"], data["parent"], data["direction"], data["area"],
                 data["parent_area"], data["event"], data["image"], use_memory)


def event_report(cache: dict, keep: torch.Tensor, target: torch.Tensor,
                 probability: torch.Tensor) -> dict:
    event_rows = [cache["image_events"][int(i)] for i in cache["image_index"].long()[keep]]
    rows = {}
    for event in sorted(set(event_rows)):
        mask = torch.tensor([value == event for value in event_rows])
        rows[event] = classification_metrics(target[mask], probability[mask])
    return {
        "event_macro_f1": sum(row["macro_f1"] for row in rows.values()) / max(len(rows), 1),
        "events": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train pseudo-instance hierarchical semantic memory")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--capacity-per-class", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--parametric-steps", type=int, default=1000)
    parser.add_argument("--retrieval-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    groups = tuple(value.strip() for value in args.groups.split(",") if value.strip())
    train_cache = load_cache(args.train_cache, "meta_train")
    calibration_cache = load_cache(args.calibration_cache, "meta_calibration")
    train, mean, std = prepare(train_cache, groups)
    calibration, _, _ = prepare(calibration_cache, groups, mean, std)
    if min(int((train["label"] == label).sum()) for label in range(3)) < 20:
        raise RuntimeError("too few labelled clusters for a three-class semantic memory")

    torch.manual_seed(args.seed)
    model = PseudoInstanceHierarchicalMemory(
        train["x"].shape[1], args.hidden, args.capacity_per_class, args.topk
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    count = torch.bincount(train["label"], minlength=3).float()
    class_weight = count.sum() / count.clamp_min(1); class_weight /= class_weight.mean()
    event_count = torch.unique(train["event"], return_counts=True)
    event_weights = {int(k): 1.0 / int(v) for k, v in zip(*event_count)}
    sample_weight = torch.tensor([event_weights[int(value)] for value in train["event"]])
    sample_weight = sample_weight / sample_weight.sum()
    generator = torch.Generator().manual_seed(args.seed + 1)
    batch_size = min(args.batch_size, len(train["x"]))

    for step in range(1, args.parametric_steps + 1):
        ids = torch.multinomial(sample_weight, batch_size, replacement=True, generator=generator)
        batch = {key: value[ids] for key, value in train.items() if key != "keep"}
        output = forward(model, batch, False)
        target = batch["label"]
        occurrence_target = (target > 0).long()
        loss_occ = F.cross_entropy(output["occurrence_logits"], occurrence_target)
        affected = target > 0
        loss_sev = F.cross_entropy(output["severity_logits"][affected], target[affected] - 1,
                                   weight=class_weight[1:]) if affected.any() else loss_occ * 0
        loss = loss_occ + loss_sev
        optimizer.zero_grad(); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"stage": "parametric", "step": step,
                              "loss": float(loss.detach()), "gradient_norm": float(gradient)}), flush=True)

    # Freeze the representation before constructing the labelled external bank.
    retrieval_trainable = (
        "attention", "metadata_encoder", "label_embedding", "retrieval_gate"
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name == "retrieval_gate"
            or any(name.startswith(prefix) for prefix in retrieval_trainable[:-1])
        )
    with torch.no_grad():
        embedding = model.encode(train["x"], train["parent"], train["direction"])
        chosen = balanced_memory_indices(train["label"], train["event"],
                                         args.capacity_per_class, args.seed + 2)
        metadata = model.metadata(train["area"], train["parent_area"])
        model.semantic_memory.enqueue(embedding[chosen], train["label"][chosen],
                                      train["event"][chosen], train["image"][chosen], metadata[chosen])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate * 0.25, weight_decay=1e-4)
    for step in range(1, args.retrieval_steps + 1):
        ids = torch.multinomial(sample_weight, batch_size, replacement=True, generator=generator)
        batch = {key: value[ids] for key, value in train.items() if key != "keep"}
        output = forward(model, batch, True); target = batch["label"]
        loss_occ = F.cross_entropy(output["occurrence_logits"], (target > 0).long())
        affected = target > 0
        loss_sev = F.cross_entropy(output["severity_logits"][affected], target[affected] - 1,
                                   weight=class_weight[1:]) if affected.any() else loss_occ * 0
        loss = loss_occ + loss_sev
        optimizer.zero_grad(); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"stage": "retrieval", "step": step,
                              "loss": float(loss.detach()), "gate": float(torch.tanh(model.retrieval_gate).detach()),
                              "gradient_norm": float(gradient)}), flush=True)

    model.eval()
    with torch.no_grad():
        base_output = forward(model, calibration, False)
        memory_output = forward(model, calibration, True)
        base_probability = probabilities(base_output)
        memory_probability = probabilities(memory_output)
    base = classification_metrics(calibration["label"], base_probability)
    memory = classification_metrics(calibration["label"], memory_probability)
    base_event = event_report(calibration_cache, calibration["keep"], calibration["label"], base_probability)
    memory_event = event_report(calibration_cache, calibration["keep"], calibration["label"], memory_probability)
    event_improved = sum(
        memory_event["events"][name]["macro_f1"] >= base_event["events"][name]["macro_f1"]
        for name in memory_event["events"]
    )
    checks = {
        "memory_macro_f1_gain_ge_0p01": memory["macro_f1"] - base["macro_f1"] >= 0.01,
        "memory_event_macro_not_below_parametric": memory_event["event_macro_f1"] >= base_event["event_macro_f1"],
        "damaged_recall_not_below_parametric": memory["damaged_recall"] >= base["damaged_recall"],
        "destroyed_recall_not_below_parametric": memory["destroyed_recall"] >= base["destroyed_recall"],
        "at_least_ten_events_nonnegative": event_improved >= min(10, len(memory_event["events"])),
        "memory_nonempty": model.semantic_memory.size > 0,
    }
    decision = "ADMIT_PI_HIERARCHICAL_MEMORY_FULL_MODEL_E1" if all(checks.values()) else "STOP_PI_HIERARCHICAL_MEMORY_NO_CROSS_EVENT_GAIN"
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": PROTOCOL, "model_state_dict": model.state_dict(),
        "input_dim": train["x"].shape[1], "hidden": args.hidden, "groups": groups,
        "feature_mean": mean, "feature_std": std, "topk": args.topk,
        "capacity_per_class": args.capacity_per_class, "seed": args.seed,
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "memory_scope": "meta_train_only_class_and_event_balanced",
        "same_event_retrieval": False, "same_image_retrieval": False,
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    torch.save(checkpoint, output / "pi_hierarchical_memory_frozen.pth")
    report = {
        "protocol": PROTOCOL, "configuration": vars(args), "groups": groups,
        "train_class_counts": count.long().tolist(), "memory_size": model.semantic_memory.size,
        "parametric": base, "memory": memory,
        "parametric_event": base_event, "memory_event": memory_event,
        "event_nonnegative": f"{event_improved}/{len(memory_event['events'])}",
        "retrieval_gate": float(torch.tanh(model.retrieval_gate).detach()),
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    (output / "calibration_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {output / 'pi_hierarchical_memory_frozen.pth'}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
