"""Compare matched original-Q and completed building-Q route experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ARMS, FORMAT
from .report import read_formal


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-fit", required=True)
    p.add_argument("--original-evaluation", required=True)
    p.add_argument("--building-fit", required=True)
    p.add_argument("--building-evaluation", required=True)
    return p


def read_fit(path):
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("full_validation_evidence") is not True
            or summary.get("smoke") is not False or summary.get("test_accessed") is not False
            or not summary.get("immutable") or not all(summary["immutable"].values())):
        raise ValueError("Route fit is not a completed, immutable formal run: " + str(root))
    return summary


def crossed(original_fit, original_eval, building_fit, building_eval):
    """Return four matched arm differences in percentage points."""
    keys = ("protocol_sha256", "source_sha256", "steps", "threshold",
            "soft_floor", "train_stream_sha256", "trainable_parameters")
    mismatches = [key for key in keys if original_fit.get(key) != building_fit.get(key)]
    if mismatches:
        raise ValueError("Original/building route fits are not matched: " + ", ".join(mismatches))
    if original_fit.get("q_source", {}).get("kind") != "e0_original_q":
        raise ValueError("First fit is not original E0 Q")
    if building_fit.get("q_source", {}).get("kind") != "completed_building_reconstruction_q":
        raise ValueError("Second fit is not a completed building-reconstruction Q")
    for fit, evaluation in ((original_fit, original_eval), (building_fit, building_eval)):
        if evaluation.get("protocol_sha256") != fit["protocol_sha256"]:
            raise ValueError("Fit/evaluation E0 protocol differs")
        if evaluation.get("source_sha256") != fit["source_sha256"]:
            raise ValueError("Fit/evaluation code differs")
        if evaluation.get("q_source") != fit["q_source"]:
            raise ValueError("Fit/evaluation Q source differs")
    old = {arm: 100 * original_eval["metrics"][arm]["valid_miou"] for arm in ARMS}
    new = {arm: 100 * building_eval["metrics"][arm]["valid_miou"] for arm in ARMS}
    return dict(original_q_miou_pct=old, building_q_miou_pct=new,
                building_q_minus_original_q_pp={arm: new[arm] - old[arm] for arm in ARMS},
                original_q_hard_minus_unsplit_pp=old["hard"] - old["unsplit"],
                building_q_hard_minus_unsplit_pp=new["hard"] - new["unsplit"],
                q_route_interaction_pp=(new["hard"] - new["unsplit"])
                                       - (old["hard"] - old["unsplit"]))


def main():
    args = parser().parse_args()
    result = crossed(read_fit(args.original_fit), read_formal(args.original_evaluation),
                     read_fit(args.building_fit), read_formal(args.building_evaluation))
    print("Formal matched Val450, percentage points")
    for arm in ARMS:
        print(f"{arm}: oldQ={result['original_q_miou_pct'][arm]:.4f}, "
              f"buildingQ={result['building_q_miou_pct'][arm]:.4f}, "
              f"Q gain={result['building_q_minus_original_q_pp'][arm]:+.4f}")
    print("Hard-route gain under original Q:",
          f"{result['original_q_hard_minus_unsplit_pp']:+.4f}")
    print("Hard-route gain under building Q:",
          f"{result['building_q_hard_minus_unsplit_pp']:+.4f}")
    print("Q×route interaction:", f"{result['q_route_interaction_pp']:+.4f}")


if __name__ == "__main__":
    main()
