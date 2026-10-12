from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .core import (
    MEMORY_PROTOCOL,
    PROTOCOL,
    build_memory,
    building_events,
    building_ids,
    compatible_representations,
    concatenate_features,
    event_label_balanced_indices,
    evaluate_scores,
    fit_robust_normalizer,
    permutation_null,
    retrieve_topk_by_utility_label,
    scores_from_neighbors,
    sha256,
    utility_labels,
    validate_cache,
)


ADVANCED_REPRESENTATIONS = {
    "base_direct", "base_relation", "base_matching", "base_grid", "base_slic"
}


def parse_numbers(raw: str, cast):
    values = tuple(dict.fromkeys(cast(item.strip()) for item in raw.split(",") if item.strip()))
    if not values:
        raise ValueError("hyperparameter grid cannot be empty")
    return values


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def select_configuration(
    retrieval: dict[str, torch.Tensor],
    labels: torch.Tensor,
    events: tuple[str, ...],
    ks: tuple[int, ...],
    temperatures: tuple[float, ...],
) -> dict[str, object]:
    candidates = []
    for k in ks:
        for temperature in temperatures:
            signals = scores_from_neighbors(retrieval, k=k, temperature=temperature)
            coverage = float(signals["available"].float().mean())
            for signal_name in ("utility_vote", "affinity_margin", "label_vote"):
                report = evaluate_scores(labels, signals[signal_name], events)
                candidates.append(
                    {
                        "k": int(k),
                        "temperature": float(temperature),
                        "signal": signal_name,
                        "availability": coverage,
                        "report": report,
                    }
                )
    return max(
        candidates,
        key=lambda item: (
            float(item["report"]["auroc"]),
            float(item["report"]["event_macro_auroc"]),
            float(item["availability"]),
            -int(item["k"]),
        ),
    )


def evaluate_fixed_configuration(
    retrieval: dict[str, torch.Tensor],
    labels: torch.Tensor,
    events: tuple[str, ...],
    configuration: dict[str, object],
) -> tuple[dict[str, object], torch.Tensor]:
    signals = scores_from_neighbors(
        retrieval,
        k=int(configuration["k"]),
        temperature=float(configuration["temperature"]),
    )
    scores = signals[str(configuration["signal"])]
    report = evaluate_scores(labels, scores, events)
    report["availability"] = float(signals["available"].float().mean())
    report["neighbor_entropy_mean"] = float(signals["neighbor_entropy"].mean())
    return report, scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("PI-M0 cross-event pseudo-instance utility-memory audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--bu-e0-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-memory", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--capacity-per-event-label", type=int, default=256)
    parser.add_argument("--max-neighbors-per-event", type=int, default=2)
    parser.add_argument("--ks", default="1,3,5,8")
    parser.add_argument("--temperatures", default="0.03,0.07,0.15,0.30")
    parser.add_argument("--representations", default="")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    parser.add_argument("--minimum-availability", type=float, default=0.95)
    parser.add_argument("--minimum-geometry-gain", type=float, default=0.02)
    parser.add_argument("--minimum-base-gain", type=float, default=0.01)
    args = parser.parse_args(argv)

    if not 0.0 < args.utility_margin < 1.0:
        raise ValueError("utility margin must be in (0,1)")
    ks = parse_numbers(args.ks, int)
    temperatures = parse_numbers(args.temperatures, float)
    if min(ks) <= 0 or min(temperatures) <= 0:
        raise ValueError("k and temperature values must be positive")

    train_path, calibration_path = Path(args.train_cache), Path(args.calibration_cache)
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    validate_cache(train, "meta_train")
    validate_cache(calibration, "meta_calibration")
    if train.get("p3_sha256") != calibration.get("p3_sha256"):
        raise RuntimeError("P3 provenance mismatch")
    if train.get("s2_configuration") != calibration.get("s2_configuration"):
        raise RuntimeError("fixed-S2 configuration mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train and meta_calibration overlap")

    train_sha, calibration_sha = sha256(train_path), sha256(calibration_path)
    bu_result = json.loads(Path(args.bu_e0_result).read_text(encoding="utf-8"))
    if bu_result.get("protocol") != "SPI_BUILDING_UTILITY_E0_AUDIT_v1":
        raise RuntimeError("unexpected BU-E0 result protocol")
    if bu_result.get("train_cache_sha256") != train_sha:
        raise RuntimeError("BU-E0/train-cache fingerprint mismatch")
    if bu_result.get("calibration_cache_sha256") != calibration_sha:
        raise RuntimeError("BU-E0/calibration-cache fingerprint mismatch")
    if any(
        bool(bu_result.get(name, True))
        for name in ("validation_accessed", "official_instances_accessed", "test_accessed")
    ):
        raise RuntimeError("BU-E0 result accessed a forbidden split")
    if not bool(bu_result.get("oracle_passed", False)):
        raise RuntimeError(
            "BU-E0 Building Oracle did not pass; PI-M0 is scientifically inadmissible"
        )

    available = compatible_representations(train, calibration)
    requested = tuple(
        item.strip() for item in args.representations.split(",") if item.strip()
    )
    if requested:
        unknown = [item for item in requested if item not in available]
        if unknown:
            raise ValueError(f"unknown or unavailable representations: {unknown}")
        representations = {name: available[name] for name in requested}
    else:
        representations = available
    if "geometry" not in representations or "base" not in representations:
        raise RuntimeError("formal PI-M0 requires geometry and base controls")

    train_labels_all = utility_labels(train, args.utility_margin)
    calibration_labels_all = utility_labels(calibration, args.utility_margin)
    train_valid = train_labels_all >= 0
    calibration_valid = calibration_labels_all >= 0
    train_labels = train_labels_all[train_valid]
    calibration_labels = calibration_labels_all[calibration_valid]
    if min(int((train_labels == label).sum()) for label in (0, 1)) == 0:
        raise RuntimeError("meta_train lacks one utility class")
    if min(int((calibration_labels == label).sum()) for label in (0, 1)) == 0:
        raise RuntimeError("meta_calibration lacks one utility class")

    train_events_all, calibration_events_all = building_events(train), building_events(calibration)
    train_ids_all, calibration_ids_all = building_ids(train), building_ids(calibration)
    train_rows = torch.nonzero(train_valid, as_tuple=False).flatten()
    calibration_rows = torch.nonzero(calibration_valid, as_tuple=False).flatten()
    train_events = tuple(train_events_all[i] for i in train_rows.tolist())
    calibration_events = tuple(calibration_events_all[i] for i in calibration_rows.tolist())
    train_ids = tuple(train_ids_all[i] for i in train_rows.tolist())
    calibration_ids = tuple(calibration_ids_all[i] for i in calibration_rows.tolist())
    train_utilities = train["utility"]["utility_ratio"].float()[train_valid]

    selected_memory_rows = event_label_balanced_indices(
        train_labels, train_events, args.capacity_per_event_label, args.seed
    )
    max_k = max(ks)
    representation_results: dict[str, dict[str, object]] = {}
    representation_state: dict[str, tuple] = {}
    for name, groups in representations.items():
        train_raw = concatenate_features(train, groups)[train_valid]
        calibration_raw = concatenate_features(calibration, groups)[calibration_valid]
        normalizer = fit_robust_normalizer(train_raw[selected_memory_rows])
        train_keys = normalizer.transform(train_raw)
        calibration_keys = normalizer.transform(calibration_raw)
        memory = build_memory(
            train_keys, train_labels, train_utilities, train_events, train_ids,
            selected_memory_rows,
        )
        train_retrieval = retrieve_topk_by_utility_label(
            train_keys, train_events, train_ids, memory,
            max_k=max_k,
            max_neighbors_per_event=args.max_neighbors_per_event,
            exclude_same_event=True,
            chunk_size=args.chunk_size,
        )
        selected = select_configuration(
            train_retrieval, train_labels, train_events, ks, temperatures
        )
        representation_results[name] = {
            "groups": list(groups),
            "raw_dimensions": int(train_raw.shape[1]),
            "active_dimensions": int(normalizer.active.sum()),
            "train_loeo_selection": selected,
        }
        representation_state[name] = (normalizer, memory, calibration_keys)
        print(
            f"PI-M0 {name:14s} train={selected['report']['auroc']:.4f} "
            f"event={selected['report']['event_macro_auroc']:.4f} "
            f"correct={selected['report']['event_direction_correct']}/{selected['report']['event_total']}",
            flush=True,
        )

    non_geometry = [name for name in representations if name != "geometry"]
    selected_name = max(
        non_geometry,
        key=lambda name: (
            representation_results[name]["train_loeo_selection"]["report"]["auroc"],
            representation_results[name]["train_loeo_selection"]["report"]["event_macro_auroc"],
        ),
    )
    # Meta-calibration is touched only for the train-selected representation and
    # the two preregistered controls. Other representations remain train-only.
    calibration_reports: dict[str, dict[str, object]] = {}
    calibration_scores_by_name: dict[str, torch.Tensor] = {}
    for name in dict.fromkeys((selected_name, "geometry", "base")):
        normalizer_i, memory_i, calibration_keys_i = representation_state[name]
        retrieval_i = retrieve_topk_by_utility_label(
            calibration_keys_i, calibration_events, calibration_ids, memory_i,
            max_k=max_k,
            max_neighbors_per_event=args.max_neighbors_per_event,
            exclude_same_event=True,
            chunk_size=args.chunk_size,
        )
        report_i, scores_i = evaluate_fixed_configuration(
            retrieval_i,
            calibration_labels,
            calibration_events,
            representation_results[name]["train_loeo_selection"],
        )
        calibration_reports[name] = report_i
        calibration_scores_by_name[name] = scores_i
        representation_results[name]["calibration_cross_event"] = report_i

    selected_configuration = representation_results[selected_name]["train_loeo_selection"]
    selected_report = calibration_reports[selected_name]
    geometry_report = calibration_reports["geometry"]
    base_report = calibration_reports["base"]
    normalizer, memory, calibration_keys = representation_state[selected_name]
    selected_scores = calibration_scores_by_name[selected_name]

    same_event_retrieval = retrieve_topk_by_utility_label(
        calibration_keys, calibration_events, calibration_ids, memory,
        max_k=max_k,
        max_neighbors_per_event=args.max_neighbors_per_event,
        exclude_same_event=False,
        chunk_size=args.chunk_size,
    )
    same_event_report, _ = evaluate_fixed_configuration(
        same_event_retrieval, calibration_labels, calibration_events, selected_configuration
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    order = torch.randperm(memory.labels.numel(), generator=generator)
    shuffled_memory = type(memory)(
        keys=memory.keys,
        labels=memory.labels[order],
        utilities=memory.utilities[order],
        events=memory.events,
        instance_ids=memory.instance_ids,
        source_indices=memory.source_indices,
    )
    shuffled_retrieval = retrieve_topk_by_utility_label(
        calibration_keys, calibration_events, calibration_ids, shuffled_memory,
        max_k=max_k,
        max_neighbors_per_event=args.max_neighbors_per_event,
        exclude_same_event=True,
        chunk_size=args.chunk_size,
    )
    shuffled_report, _ = evaluate_fixed_configuration(
        shuffled_retrieval, calibration_labels, calibration_events, selected_configuration
    )
    random_scores = torch.rand(calibration_labels.numel(), generator=generator)
    random_report = evaluate_scores(calibration_labels, random_scores, calibration_events)
    permutation = permutation_null(
        calibration_labels, selected_scores, args.permutations, args.seed + 202
    )

    advanced_gain_check = True
    if selected_name in ADVANCED_REPRESENTATIONS:
        advanced_gain_check = (
            selected_report["auroc"] >= base_report["auroc"] + args.minimum_base_gain
        )
    checks = {
        "bu_e0_oracle_passed": True,
        "calibration_auroc_ge_threshold": selected_report["auroc"] >= args.minimum_auroc,
        "event_macro_ge_threshold": (
            selected_report["event_macro_auroc"] >= args.minimum_event_macro
        ),
        "correct_events_ge_threshold": (
            selected_report["event_direction_correct"] >= args.minimum_correct_events
        ),
        "at_least_ten_evaluable_events": (
            selected_report["event_total"] >= args.minimum_correct_events
        ),
        "positive_count_ge_threshold": (
            selected_report["positive"] >= args.minimum_class_count
        ),
        "negative_count_ge_threshold": (
            selected_report["negative"] >= args.minimum_class_count
        ),
        "availability_ge_threshold": (
            selected_report["availability"] >= args.minimum_availability
        ),
        "beats_geometry_by_margin": (
            selected_report["auroc"] >= geometry_report["auroc"] + args.minimum_geometry_gain
        ),
        "advanced_representation_beats_base_by_margin_if_selected": advanced_gain_check,
        "permutation_p_le_0_05": permutation["one_sided_p"] <= 0.05,
        "beats_shuffled_memory": selected_report["auroc"] > shuffled_report["auroc"],
    }
    decision = (
        "ADMIT_PI_M1_LOW_CAPACITY_VERIFIER"
        if all(checks.values())
        else "STOP_PI_MEMORY_NO_CROSS_EVENT_UTILITY_SIGNAL"
    )

    memory_path = Path(args.output_memory)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_payload = {
        "protocol": MEMORY_PROTOCOL,
        "representation": selected_name,
        "groups": list(representations[selected_name]),
        "keys": memory.keys.half(),
        "labels": memory.labels,
        "utilities": memory.utilities,
        "events": list(memory.events),
        "instance_ids": list(memory.instance_ids),
        "source_indices": memory.source_indices,
        "normalizer": normalizer.payload(),
        "p3_sha256": train.get("p3_sha256"),
        "train_cache_sha256": train_sha,
        "utility_margin": args.utility_margin,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    atomic_torch_save(memory_payload, memory_path)

    payload = {
        "protocol": PROTOCOL,
        "method": "cross-event event-balanced pseudo-instance S2-utility retrieval",
        "selection_contract": (
            "representation/k/temperature/signal selected on meta_train LOEO only; "
            "meta_calibration evaluated once"
        ),
        "train_cache": str(train_path.resolve()),
        "train_cache_sha256": train_sha,
        "calibration_cache": str(calibration_path.resolve()),
        "calibration_cache_sha256": calibration_sha,
        "bu_e0_result": str(Path(args.bu_e0_result).resolve()),
        "bu_e0_result_sha256": sha256(args.bu_e0_result),
        "memory_path": str(memory_path.resolve()),
        "memory_size": int(memory.labels.numel()),
        "memory_class_counts": {
            "negative": int((memory.labels == 0).sum()),
            "positive": int((memory.labels == 1).sum()),
        },
        "settings": vars(args),
        "representations": representation_results,
        "selected_representation_train_only": selected_name,
        "selected_configuration_train_only": selected_configuration,
        "formal_cross_event": selected_report,
        "controls": {
            "geometry_only": geometry_report,
            "base": base_report,
            "same_event_leakage_diagnostic": same_event_report,
            "shuffled_memory_labels": shuffled_report,
            "random_scores": random_report,
            "target_permutation_null": permutation,
        },
        "checks": checks,
        "decision": decision,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 112)
    print(f"Selected on train LOEO : {selected_name} / {selected_configuration['signal']} "
          f"k={selected_configuration['k']} tau={selected_configuration['temperature']}")
    print(f"Formal cross-event AUC : {selected_report['auroc']:.4f}")
    print(f"Event macro/correct    : {selected_report['event_macro_auroc']:.4f} / "
          f"{selected_report['event_direction_correct']}/{selected_report['event_total']}")
    print(f"Geometry/base AUC      : {geometry_report['auroc']:.4f} / {base_report['auroc']:.4f}")
    print(f"Same-event diagnostic : {same_event_report['auroc']:.4f}")
    print(f"Shuffled/random AUC    : {shuffled_report['auroc']:.4f} / {random_report['auroc']:.4f}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved memory: {memory_path}")
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
