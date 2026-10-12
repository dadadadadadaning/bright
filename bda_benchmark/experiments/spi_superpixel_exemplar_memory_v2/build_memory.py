#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import PROTOCOL, MetricKeyEncoder


def balanced_indices(labels, events, capacity_per_class: int, event_cap: int) -> torch.Tensor:
    selected = []
    for label in (1, 2, 3):
        buckets = defaultdict(list)
        for index in torch.nonzero(labels == label, as_tuple=False)[:, 0].tolist():
            if len(buckets[events[index]]) < event_cap:
                buckets[events[index]].append(index)
        names = sorted(buckets)
        cursor = {name: 0 for name in names}
        class_selected = 0
        while class_selected < capacity_per_class:
            changed = False
            for name in names:
                pos = cursor[name]
                if pos < len(buckets[name]):
                    selected.append(buckets[name][pos]); cursor[name] += 1
                    class_selected += 1; changed = True
                    if class_selected >= capacity_per_class:
                        break
            if not changed:
                break
    return torch.tensor(selected, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser("Build a class/event-balanced bank of actual superpixel exemplars")
    parser.add_argument("--records", required=True); parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--capacity_per_class", type=int, default=4096)
    parser.add_argument("--event_cap", type=int, default=256); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    saved = torch.load(args.metric, map_location="cpu", weights_only=False)
    if Path(saved["records"]).resolve() != Path(args.records).resolve():
        raise RuntimeError(
            "metric checkpoint was trained from a different record cache; "
            "use a matching METRIC_CKPT instead of mixing experiments"
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MetricKeyEncoder(saved["input_dim"], saved["key_dim"]).to(device).eval()
    model.load_state_dict(saved["model"])
    indices = balanced_indices(records["labels"], records["events"], args.capacity_per_class, args.event_cap)
    keys = []
    with torch.inference_mode():
        for start in range(0, indices.numel(), 1024):
            batch = records["descriptors"][indices[start:start + 1024]].to(device)
            keys.append(model(batch)[0].cpu())
    payload = {
        "protocol": PROTOCOL + "_BANK", "keys": torch.cat(keys),
        "labels": records["labels"][indices], "soft_labels": records["soft_labels"][indices],
        "events": [records["events"][i] for i in indices.tolist()],
        "images": [records["images"][i] for i in indices.tolist()],
        "purities": records["purities"][indices], "areas": records["areas"][indices],
        "metadata": {"actual_superpixel_exemplars": True, "same_image_class_averaging": False,
                     "capacity_per_class": args.capacity_per_class, "event_cap": args.event_cap,
                     "entry_count": int(indices.numel()), "test_accessed": False},
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination); print(f"Saved: {destination} entries={indices.numel()}")


if __name__ == "__main__":
    main()
