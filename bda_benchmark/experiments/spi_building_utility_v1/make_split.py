from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Freeze event-stratified BU meta split")
    parser.add_argument("--source-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--confirm-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.calibration_fraction <= 0 or args.confirm_fraction <= 0:
        raise ValueError("calibration and confirm fractions must be positive")
    if args.calibration_fraction + args.confirm_fraction >= 0.5:
        raise ValueError("at least half of every event must remain in meta_train")

    source = Path(args.source_list)
    output = Path(args.output_dir)
    manifest_path = output / "split_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol") != "SPI_BUILDING_UTILITY_META_SPLIT_v1"
            or int(existing.get("seed", -1)) != args.seed
            or existing.get("source_sha256") != sha256(source)
            or float(existing.get("calibration_fraction", -1)) != args.calibration_fraction
            or float(existing.get("confirm_fraction", -1)) != args.confirm_fraction
        ):
            raise RuntimeError("existing BU meta split does not match source/seed")
        print(json.dumps(existing, indent=2))
        print("Reusing frozen BU meta split.")
        return 0
    names = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for name in names:
        grouped[name.rsplit("_", 1)[0]].append(name)
    train, calibration, confirm, events = [], [], [], {}
    for event, rows in sorted(grouped.items()):
        local = list(rows)
        random.Random(f"BU:{args.seed}:{event}").shuffle(local)
        calibration_count = max(1, round(len(local) * args.calibration_fraction))
        confirm_count = max(1, round(len(local) * args.confirm_fraction))
        if calibration_count + confirm_count >= len(local):
            raise RuntimeError(f"event too small for three-way split: {event}")
        calibration.extend(local[:calibration_count])
        confirm.extend(local[calibration_count:calibration_count + confirm_count])
        train.extend(local[calibration_count + confirm_count:])
        events[event] = {
            "all": len(local), "meta_train": len(local) - calibration_count - confirm_count,
            "meta_calibration": calibration_count, "meta_confirm": confirm_count,
        }
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(calibration)
    random.Random(args.seed + 2).shuffle(confirm)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "meta_train": output / "meta_train.txt",
        "meta_calibration": output / "meta_calibration.txt",
        "meta_confirm": output / "meta_confirm.txt",
    }
    for name, rows in (("meta_train", train), ("meta_calibration", calibration), ("meta_confirm", confirm)):
        files[name].write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "SPI_BUILDING_UTILITY_META_SPLIT_v1",
        "seed": args.seed,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "calibration_fraction": args.calibration_fraction,
        "confirm_fraction": args.confirm_fraction,
        "counts": {"all": len(names), "meta_train": len(train), "meta_calibration": len(calibration), "meta_confirm": len(confirm)},
        "events": events,
        "files": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in files.items()},
        "validation_accessed": False,
        "test_accessed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
