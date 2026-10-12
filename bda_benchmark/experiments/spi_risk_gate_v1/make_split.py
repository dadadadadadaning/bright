from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser("Deterministic event-stratified gate split")
    parser.add_argument("--source-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    names = [line.strip().rstrip("\r") for line in Path(args.source_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for name in names:
        groups[name.rsplit("_", 1)[0]].append(name)
    train, calibration, counts = [], [], {}
    for event, rows in sorted(groups.items()):
        local = list(rows)
        random.Random(f"{args.seed}:{event}").shuffle(local)
        count = max(1, round(len(local) * args.calibration_fraction)) if len(local) > 1 else 0
        calibration.extend(local[:count])
        train.extend(local[count:])
        counts[event] = {"all": len(local), "train": len(local) - count, "calibration": count}
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(calibration)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (output / "gate_calibration.txt").write_text("\n".join(calibration) + "\n", encoding="utf-8")
    manifest = {"seed": args.seed, "source": str(Path(args.source_list).resolve()), "train": len(train), "calibration": len(calibration), "events": counts}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

