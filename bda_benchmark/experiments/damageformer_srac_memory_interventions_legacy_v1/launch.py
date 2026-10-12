"""Launch from the exact source arguments stored in a completed E0 record."""
from __future__ import annotations

import argparse
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import read_json
from .evaluate import run


def arguments_from_reference(reference_root, output, intervention_seed=3407):
    root = Path(reference_root)
    record = read_json(root / "protocol" / "protocol.json")
    original = record.get("arguments")
    required = ("joint_run", "a1_checkpoint", "dataset", "train_list", "val_list",
                "device", "workers", "seed", "profile", "pool_mode", "smoke")
    if (record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
            or record.get("complete") is not True
            or record.get("scope") != "all"
            or not isinstance(original, dict)
            or any(key not in original for key in required)):
        raise ValueError("Reference root does not contain a complete all-path E0 argument ledger")
    values = dict(original)
    values.update(output=str(output), protocol=str(root / "protocol"),
                  evidence=str(root / "evidence"), intervention_seed=int(intervention_seed))
    return argparse.Namespace(**values)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--intervention_seed", type=int, default=3407)
    args = p.parse_args()
    run(arguments_from_reference(args.reference_root, args.output, args.intervention_seed))


if __name__ == "__main__":
    main()
