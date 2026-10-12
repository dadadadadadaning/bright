"""Print an already completed formal new head evaluation; never run a model."""
import argparse
import hashlib
import json
from pathlib import Path


def read_formal(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("result_file") != "result.json":
        raise ValueError("Summary does not reference result.json")
    payload = (root / "result.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != summary.get("result_sha256"):
        raise ValueError("Completed result hash differs from summary")
    result = json.loads(payload)
    from .core import assert_formal_report
    assert_formal_report(result)
    expected = "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
    if (summary.get("format") != expected or result.get("format") != expected
            or result.get("test_accessed") is not False
            or summary.get("complete") is not True
            or summary.get("full_validation_evidence") is not True):
        raise ValueError("Summary does not identify a completed formal Val450 evaluation")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = read_formal(args.output)
    print("Formal Val450; IoU in percent. Order: Background / Intact / Damaged / Destroyed")
    print("P/R: matched 256-to-4 heads; T: separate 128-to-4 token-head diagnostic")
    for arm, m in result["metrics"].items():
        vals = " / ".join("NA" if x is None else f"{100*x:.4f}" for x in m["iou"])
        value = m["valid_miou"]
        score = "NA" if value is None else f"{100*value:.4f}"
        print(f"{arm:<22} fg_mIoU={score:>9}  {vals}")
    print("Paired Damaged pixel changes (not building counts):")
    for pair, value in result.get("transitions", {}).items():
        row = value["per_class"]["Damaged"]
        print(f"{pair}: rescue={row['rescue']} harm={row['harm']} "
              f"net_TP={row['net_tp']} FP_delta={row['delta_fp']}")


if __name__ == "__main__":
    main()
