from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bda_benchmark.experiments.spi_rgid_semantic_v1.event_protocol import (
    assert_event_disjoint,
    balanced_event_folds,
    sample_event,
)


def _read_names(path: Path) -> list[str]:
    names = [row.strip().rstrip("\r") for row in path.read_text(encoding="utf-8").splitlines()]
    names = [row for row in names if row]
    if len(names) != len(set(names)):
        raise ValueError("input split contains duplicate sample names")
    return names


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create event-disjoint train/calibration folds without reading images or labels."
    )
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-count", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input_list).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    names = _read_names(source)
    folds = balanced_event_folds(names, args.fold_count)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "SPI_EVENT_DISJOINT_FOLDS_v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_count": len(names),
        "event_count": len({sample_event(name) for name in names}),
        "fold_count": len(folds),
        "folds": [],
    }
    universe = set(names)
    for index, validation in enumerate(folds):
        train = sorted(universe.difference(validation))
        validation = sorted(validation)
        train_events, validation_events = assert_event_disjoint(train, validation)
        prefix = f"fold_{index:02d}"
        _atomic_text(output / f"{prefix}_train.txt", "\n".join(train) + "\n")
        _atomic_text(output / f"{prefix}_validation.txt", "\n".join(validation) + "\n")
        manifest["folds"].append(
            {
                "fold": index,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "train_events": sorted(train_events),
                "validation_events": sorted(validation_events),
            }
        )
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
