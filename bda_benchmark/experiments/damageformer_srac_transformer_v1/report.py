"""Compact read-only report for a completed experiment root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def metric_line(name, summary):
    metrics = summary["metrics"]
    key = "final" if "final" in metrics else "front_region"
    value = metrics[key]
    damaged_f1 = value.get("f1", [None, None, None])[2]
    text = f"{name:44s} FG mIoU={100 * value['valid_miou']:.4f}%"
    if damaged_f1 is not None:
        text += f"  Damaged F1={100 * damaged_f1:.4f}%"
    return text


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--policy", default="global")
    parser.add_argument("--exclusion", default="cross_event")
    args = parser.parse_args()
    root = Path(args.root)
    paths = [("front", root / "evaluation/front/summary.json")]
    paths.append(
        (
            "nomemory",
            root / f"evaluation/nomemory_{args.policy}_{args.exclusion}/summary.json",
        )
    )
    for intervention in ("normal", "random", "shuffle_labels", "off", "no_metadata"):
        paths.append(
            (
                "memory_" + intervention,
                root / f"evaluation/memory_{args.policy}_{args.exclusion}_{intervention}/summary.json",
            )
        )
    print("All values are full-image Val metrics; no Test selection.")
    for name, path in paths:
        if path.is_file():
            print(metric_line(name, read(path)))
        else:
            print(f"{name:44s} missing: {path}")
    audit = root / f"retrieval_audit_{args.policy}_{args.exclusion}/summary.json"
    if audit.is_file():
        data = read(audit)
        print("\nRetrieval audit (Train labels used only after neighbor IDs were fixed):")
        for name, values in data["per_class"].items():
            print(
                f"{name:12s} Recall@K={100 * values['same_class_recall_at_k']:.2f}% "
                f"slot_same={100 * values['same_class_slot_fraction']:.2f}% "
                f"supported={100 * values['supported_query_fraction']:.2f}% "
                f"majority_acc_supported={100 * values['majority_knn_accuracy_supported']:.2f}%"
            )


if __name__ == "__main__":
    main()

