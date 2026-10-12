#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser("Select a safe RGID G3 checkpoint")
    parser.add_argument("evaluation_json", nargs="+")
    args = parser.parse_args()
    candidates = []
    for path in args.evaluation_json:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload["g3"]["instance_metrics"]
        if (
            float(metrics["precision"]) >= 0.60
            and float(metrics["mean_matched_iou"]) >= 0.75
        ):
            candidates.append(
                (
                    float(metrics["f1"]),
                    -int(metrics["split"]),
                    float(metrics["recall"]),
                    -int(metrics["merge"]),
                    path,
                )
            )
    if not candidates:
        print("NO_SAFE_G3_CHECKPOINT")
        return
    selected = max(candidates)
    print("SAFE_G3_CHECKPOINT", selected[-1])


if __name__ == "__main__":
    main()
