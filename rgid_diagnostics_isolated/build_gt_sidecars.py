#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from variant_common import (
    atomic_json,
    atomic_torch_save,
    damage_class_cc,
    read_label,
    torch_load,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build lightweight damage-class-CC instance GT sidecars."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--damage-template", default="target/{name}_building_damage.tif")
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files: {cache_dir}")

    rows = []
    total_binary = total_damage = 0
    for index, source in enumerate(files, start=1):
        destination = output_dir / source.name
        if destination.exists() and not args.overwrite:
            saved = torch_load(destination)
            if saved.get("source_file") == source.name:
                print(f"[{index}/{len(files)}] reuse {source.name}", flush=True)
                rows.append(saved["summary"])
                total_binary += int(saved["summary"]["binary_instances"])
                total_damage += int(saved["summary"]["damage_instances"])
                continue
            raise RuntimeError(f"Existing sidecar does not match source: {destination}")

        payload = torch_load(source)
        name = str(payload["name"])
        damage_path = dataset_root / args.damage_template.format(name=name)
        if not damage_path.is_file():
            raise FileNotFoundError(f"Damage label missing for {name}: {damage_path}")
        damage = read_label(damage_path)
        binary_instances = np.asarray(payload["instance_ids"])
        if damage.shape != binary_instances.shape:
            raise ValueError(
                f"Shape mismatch for {name}: damage={damage.shape}, cache={binary_instances.shape}"
            )
        variant = damage_class_cc(
            damage,
            min_area=args.min_area,
            ignore_index=args.ignore_index,
        )
        binary_count = int(np.count_nonzero(np.unique(binary_instances) > 0))
        damage_count = int(variant.max())
        summary = {
            "source_file": source.name,
            "name": name,
            "damage_path": str(damage_path),
            "binary_instances": binary_count,
            "damage_instances": damage_count,
            "instances_added": damage_count - binary_count,
        }
        atomic_torch_save(
            {
                "source_file": source.name,
                "name": name,
                "gt_variant": "damage_class_cc",
                "instance_ids": variant,
                "summary": summary,
            },
            destination,
        )
        rows.append(summary)
        total_binary += binary_count
        total_damage += damage_count
        print(
            f"[{index}/{len(files)}] {name}: {binary_count} -> {damage_count}",
            flush=True,
        )

    manifest = {
        "protocol": "rgid_damage_class_cc_sidecar_v1",
        "source_cache": str(cache_dir),
        "dataset_root": str(dataset_root),
        "damage_template": args.damage_template,
        "min_area": args.min_area,
        "ignore_index": args.ignore_index,
        "sample_count": len(rows),
        "binary_instances": total_binary,
        "damage_instances": total_damage,
        "instances_added": total_damage - total_binary,
        "samples": rows,
    }
    atomic_json(manifest, output_dir / "manifest.json")
    print(f"Wrote {output_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
