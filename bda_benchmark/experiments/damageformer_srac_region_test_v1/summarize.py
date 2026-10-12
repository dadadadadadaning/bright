"""Print the predefined Test878 comparisons from a completed result.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value):
    return "NA" if value is None else f"{100 * value:.4f}"


def summarize(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("format") != "SRAC_EXISTING_REGION_TEST_V1" or record.get("complete") is not True:
        raise ValueError("The result is missing, incomplete, or has a different format")
    mode = "formal Test878" if record.get("full_test_evidence") else "diagnostic Test prefix"
    print(f"{mode}: {record['image_count']} images; percentage points below")
    print("Order: Background, Intact, Damaged, Destroyed")
    print("arm                 fg mIoU       class IoU")
    for arm in record["arms"]:
        result = record["metrics"][arm]
        print(f"{arm:20} {pct(result['valid_miou']):>7}       "
              + " / ".join(pct(value) for value in result["iou"]))
    print("\nPaired changes: candidate minus reference")
    for pair in record["comparison_pairs"]:
        before, after = pair.split("__")
        left = record["metrics"][before]["valid_miou"]
        right = record["metrics"][after]["valid_miou"]
        damaged = record["transitions"][pair]["per_class"]["Damaged"]
        print(f"{pair:29} mIoU {100*(right-left):+.4f} pp; "
              f"Damaged rescue {damaged['rescue']}, harm {damaged['harm']}, "
              f"FP delta {damaged['delta_fp']:+d}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path, help="Completed Test run's result.json")
    summarize(p.parse_args().result)


if __name__ == "__main__":
    main()
