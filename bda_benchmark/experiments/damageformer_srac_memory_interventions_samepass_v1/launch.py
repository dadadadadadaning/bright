"""Launch with the exact source and numeric arguments saved by a passed E0."""

from __future__ import annotations

import argparse
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import read_json
from .evaluate import run


def arguments_from_reference(reference_root, output, random_seeds=(3407,),
                             shuffle_seed=13407):
    root = Path(reference_root)
    record = read_json(root / "protocol" / "protocol.json")
    original = record.get("arguments")
    required = ("joint_run", "a1_checkpoint", "dataset", "train_list", "val_list",
                "device", "workers", "seed", "profile", "pool_mode", "smoke")
    if (record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
            or record.get("complete") is not True
            or record.get("full_validation_evidence") is not True
            or record.get("scope") != "all"
            or record.get("image_count") != 450
            or record.get("reproduction", {}).get("passed") is not True
            or not isinstance(original, dict)
            or any(key not in original for key in required)):
        raise ValueError("Requires a passed formal all-path legacy/native E0")
    values = dict(original)
    values.update(output=str(output), protocol=str(root / "protocol"),
                  evidence=str(root / "evidence"),
                  random_seeds=tuple(int(x) for x in random_seeds),
                  shuffle_seed=int(shuffle_seed))
    return argparse.Namespace(**values)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--random_seeds", type=int, nargs="+", default=[3407])
    p.add_argument("--shuffle_seed", type=int, default=13407)
    args = p.parse_args()
    run(arguments_from_reference(args.reference_root, args.output,
                                 args.random_seeds, args.shuffle_seed))


if __name__ == "__main__":
    main()
