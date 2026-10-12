# -*- coding: utf-8 -*-
"""Summarize completed CMR-C0 qualitative annotations without scoring them."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model.cmr_c0.visual_diagnostics import summarize_manual_observations


ANNOTATION_FIELDS = (
    "hotspot_location",
    "hotspot_modality",
    "hotspot_shape",
    "top_ratio_change",
    "nms_effect",
    "intact_case_match_quality",
    "damaged_case_match_quality",
    "why_all_is_better",
    "manual_judgement",
)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize CMR-C0 visual observations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("manual observation CSV contains no cases")
    missing_columns = [field for field in ANNOTATION_FIELDS if field not in rows[0]]
    if missing_columns:
        raise RuntimeError(f"manual observation CSV lacks columns: {missing_columns}")

    summary = summarize_manual_observations(rows, ANNOTATION_FIELDS)
    summary.update({
        "protocol": "CMR_C0_ISOLATED_manual_visual_summary_v1",
        "source": str(source),
        "test_accessed": False,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    annotated = summary["overall"]["fields"]["manual_judgement"]["annotated"]
    total = summary["overall"]["cases"]
    print("=" * 88)
    print("CMR-C0 qualitative observation summary")
    print(f"Cases                    : {total}")
    print(f"Manual judgements filled : {annotated}/{total}")
    print(f"Saved                    : {output}")
    print("Automatic go/no-go       : disabled")
    print("Test accessed            : False")


if __name__ == "__main__":
    main()
