"""Read diagnostic evidence without loading models or starting any experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    cli = p.parse_args()
    root = Path(cli.input)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    print("Status:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    result = root / "result.json"
    if not result.is_file():
        print("No completed result.json; inspect preserved traceback and summary. No score claim.")
        return
    record = json.loads(result.read_text(encoding="utf-8"))
    print("\nForeground mIoU / class IoUs (these label votes are untrained diagnostics):")
    for arm, row in record.get("metrics", {}).items():
        print(arm, json.dumps({k: row[k] for k in ("valid_miou", "iou", "IoU", "per_class_iou") if k in row}))
    for key in ("support_token_replay", "posthoc_retrieval_topk", "posthoc_retrieval_top64", "posthoc_weighted_label_evidence", "query_eligibility"):
        if key in record:
            print("\n" + key)
            print(json.dumps(record[key], ensure_ascii=False, indent=2))
    if record.get("full_validation_evidence") is not True:
        print("Not full formal Val450 evidence. Smoke/failed scores cannot be treated as final results.")


if __name__ == "__main__":
    main()
