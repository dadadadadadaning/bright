from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from . import PROTOCOL


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_numbers(value, path="root") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, child in value.items(): errors.extend(finite_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value): errors.extend(finite_numbers(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    return errors


def verify_manifest(path: Path, expected_split: str, expected_images: int, expected_buildings: int) -> dict:
    item = torch.load(path, map_location="cpu", weights_only=False)
    errors = []
    if item.get("protocol") != PROTOCOL: errors.append("protocol")
    if item.get("split") != expected_split: errors.append("split")
    if int(item.get("processed_images", -1)) != expected_images: errors.append("processed_images")
    if int(item.get("buildings", -1)) != expected_buildings: errors.append("buildings")
    shards = [Path(x) for x in item.get("shards", [])]
    if not shards or any(not x.is_file() for x in shards): errors.append("shards")
    if any(item.get(key, True) for key in ("meta_confirm_accessed", "val450_accessed", "test_accessed")):
        errors.append("dataset_access")
    return {"path": str(path), "images": item.get("processed_images"), "buildings": item.get("buildings"),
            "shards": len(shards), "passed": not errors, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Read-only Independent Memory result verification")
    parser.add_argument("--cache-root", required=True); parser.add_argument("--result-root", required=True)
    args = parser.parse_args(argv)
    cache = Path(args.cache_root); root = Path(args.result_root)
    required = {"d0_d2": root / "mi_d0_d2_independent_information.json",
                "d1": root / "mi_d1_capacity_diversity.json",
                "e2": root / "mi_e2_role_dependency.json",
                "summary": root / "independent_memory_summary.json"}
    manifests = [verify_manifest(cache / "meta_train_cards.pt", "meta_train", 2101, 88897),
                 verify_manifest(cache / "meta_calibration_cards.pt", "meta_calibration", 263, 10932)]
    print("===== Cache integrity =====")
    for row in manifests:
        print(f"{Path(row['path']).name:28s} images={row['images']} buildings={row['buildings']} shards={row['shards']} passed={row['passed']}")
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        print("\n===== INCOMPLETE =====\nMissing result files:")
        for path in missing: print(" -", path)
        print("The queue stopped or is still running after MI-D1; do not interpret this as a completed experiment.")
        return 3
    d0 = load_json(required["d0_d2"]); d1 = load_json(required["d1"])
    e2 = load_json(required["e2"]); summary = load_json(required["summary"])
    invalid = []
    for name, payload in (("d0_d2", d0), ("d1", d1), ("e2", e2), ("summary", summary)):
        if payload.get("protocol") != PROTOCOL: invalid.append(name + ".protocol")
        invalid.extend(f"{name}.{x}" for x in finite_numbers(payload))
    if not all(x["passed"] for x in manifests): invalid.append("cache_manifest")

    p3 = d0["p3"]; independent = d0["independent_only"]
    conditional = d0["conditional_e1"]; shuffled = d0["shuffled_independent"]
    print("\n===== MI-D0: information beyond P3 =====")
    print(f"P3 AUC                         {p3['macro_ovr_auroc']:.4f}")
    print(f"Independent-only AUC           {independent['macro_ovr_auroc']:.4f}")
    print(f"P3 + independent AUC           {conditional['macro_ovr_auroc']:.4f}")
    print(f"P3 + shuffled-independent AUC  {shuffled['macro_ovr_auroc']:.4f}")
    print(f"Conditional gain vs P3         {conditional['gain_macro_auc']:+.4f}")
    print(f"Nonnegative events             {conditional['nonnegative_events']}/{len(conditional['events'])}")
    print("Checks:", d0["checks"])

    print("\n===== MI-D1: capacity/selection =====")
    ranked = sorted(d1["results"].items(), key=lambda item: item[1]["macro_ovr_auroc"], reverse=True)
    for name, row in ranked[:8]:
        print(f"{name:24s} rows={row['memory_rows']:6d} AUC={row['macro_ovr_auroc']:.4f} purity={row['mean_neighbour_purity']:.4f}")
    print("Best:", ranked[0][0])

    print("\n===== MI-D2: card ablation =====")
    for name, row in d0["ablations"].items():
        if name == "p3_global": continue
        print(f"{name:30s} AUC={row['macro_ovr_auroc']:.4f} gain={row['gain_macro_auc']:+.4f} F1gain={row['gain_macro_f1']:+.4f}")

    print("\n===== MI-E2: retrieval causality =====")
    for name, row in e2["reports"].items():
        print(f"{name:28s} F1={row['macro_f1']:.4f} gain-vs-P3={row['gain_macro_f1']:+.4f} gate={e2['gates'][name]:+.4f}")
    print("Checks:", e2["checks"]); print("Decision:", e2["decision"])
    print("\n===== Final =====")
    print("Stages:", summary["stages"])
    print("Memory model integration allowed:", summary["memory_model_integration_allowed"])
    print("Val450/Test accessed:", summary["val450_accessed"], "/", summary["test_accessed"])
    if invalid:
        print("Integrity failures:", invalid); return 4
    print("Result files and cache manifests are structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
