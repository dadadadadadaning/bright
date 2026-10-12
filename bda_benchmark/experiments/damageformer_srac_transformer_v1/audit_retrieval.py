"""Read-only Train retrieval audit; query labels are used only after IDs are fixed."""
from __future__ import annotations

import argparse
import json

import torch

from .build_neighbors import open_table
from .common import digest, new_output, read_cache, save_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "cache", "neighbors", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--policy", choices=("global", "per_class"), default="global")
    parser.add_argument("--exclusion", choices=("same_image", "cross_event"), default="cross_event")
    args = parser.parse_args()
    root, cache, bank = read_cache(args.cache, args.checkpoint)
    table, table_manifest = open_table(
        args.neighbors,
        root,
        cache["bank_sha256"],
        args.topk,
        args.policy,
        args.exclusion,
    )
    output = new_output(args.output)
    bank_label = bank["distribution"].argmax(-1)
    confusion = torch.zeros(4, 4, dtype=torch.long)
    query_count = torch.zeros(4, dtype=torch.long)
    any_same = torch.zeros(4, dtype=torch.long)
    same_slots = torch.zeros(4, dtype=torch.long)
    valid_slots = torch.zeros(4, dtype=torch.long)
    supported_queries = torch.zeros(4, dtype=torch.long)
    majority_correct = torch.zeros(4, dtype=torch.long)
    per_event = {}
    for shard_index, (record, offset) in enumerate(zip(cache["shards"], table_manifest["offsets"])):
        if digest(root / record["path"]) != record["sha256"]:
            raise ValueError("Shard hash mismatch: " + record["path"])
        shard = torch.load(root / record["path"], map_location="cpu", weights_only=False)
        eligible = (shard["weights"] > 0).nonzero().flatten()
        if not len(eligible):
            continue
        query_label = shard["distribution"][eligible].argmax(-1)
        indices = torch.from_numpy(table[offset["start"] + eligible.numpy()].copy()).long()
        valid = indices >= 0
        labels = bank_label[indices.clamp_min(0)]
        for class_index in range(4):
            select = query_label == class_index
            if not select.any():
                continue
            selected_labels = labels[select]
            selected_valid = valid[select]
            query_count[class_index] += int(select.sum())
            matches = (selected_labels == class_index) & selected_valid
            supported = selected_valid.any(1)
            any_same[class_index] += int(matches.any(1).sum())
            same_slots[class_index] += int(matches.sum())
            valid_slots[class_index] += int(selected_valid.sum())
            supported_queries[class_index] += int(supported.sum())
            vote = torch.zeros(len(selected_labels), 4, dtype=torch.long)
            for support_class in range(4):
                vote[:, support_class] = ((selected_labels == support_class) & selected_valid).sum(1)
                confusion[class_index, support_class] += int(vote[:, support_class].sum())
            majority_correct[class_index] += int(((vote.argmax(-1) == class_index) & supported).sum())
        event_key = cache["event_names"][int(shard["event"])]
        event_stats = per_event.setdefault(event_key, dict(queries=0, any_same=0))
        event_stats["queries"] += len(eligible)
        event_stats["any_same"] += int(((labels == query_label[:, None]) & valid).any(1).sum())
    per_class = {}
    names = ("Background", "Intact", "Damaged", "Destroyed")
    for index, name in enumerate(names):
        count = int(query_count[index])
        supported = int(supported_queries[index])
        per_class[name] = dict(
            queries=count,
            supported_queries=supported,
            supported_query_fraction=float(supported / max(1, count)),
            same_class_recall_at_k=float(any_same[index] / max(1, count)),
            same_class_slot_fraction=float(same_slots[index] / max(1, int(valid_slots[index]))),
            majority_knn_accuracy_supported=float(majority_correct[index] / max(1, supported)),
            valid_neighbor_slots=int(valid_slots[index]),
        )
    result = dict(
        complete=True,
        policy=args.policy,
        exclusion=args.exclusion,
        topk=args.topk,
        query_gt_role="audit only after immutable neighbor IDs; never used to select neighbors",
        per_class=per_class,
        query_to_support_class_slot_confusion=confusion.tolist(),
        per_event={
            event_id: dict(**values, same_class_recall_at_k=values["any_same"] / max(1, values["queries"]))
            for event_id, values in per_event.items()
        },
        test_accessed=False,
    )
    save_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

