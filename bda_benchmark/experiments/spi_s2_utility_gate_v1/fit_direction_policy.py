from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    compute_transition_audit,
    derive_direction_policy,
    empty_transition_audit,
    finalize_transition_audit,
    fixed_s2_candidate_from_labels,
    merge_transition_audit,
    predicted_building_label_map,
)
from .runtime import load_frozen_p3


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Fit a Train-only P3-to-S2 direction policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--minimum-pixels", type=int, default=100)
    parser.add_argument("--minimum-decisive-precision", type=float, default=0.50)
    parser.add_argument("--minimum-event-nonnegative-fraction", type=float, default=0.60)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if args.expected_images > 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    total = empty_transition_audit()
    events = defaultdict(empty_transition_audit)
    processed = 0
    started = time.perf_counter()
    timing = defaultdict(float)
    config = SelectiveContextConfig()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        stage_started = time.perf_counter()
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits = model(rgb, sar)
        base = damage_logits[0].float().softmax(0).cpu()
        localization = loc_logits[0].argmax(0).cpu()
        timing["p3_and_transfer"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        region_map = predicted_building_label_map(localization, args.minimum_area)
        timing["loccc"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        candidate = fixed_s2_candidate_from_labels(base, region_map, config)
        timing["s2_candidate"] += time.perf_counter() - stage_started
        name = str(names[0])
        stage_started = time.perf_counter()
        target = load_semantic_target(args.dataset_path, name)
        timing["target"] += time.perf_counter() - stage_started
        event = name.rsplit("_", 1)[0]
        stage_started = time.perf_counter()
        image_audit = compute_transition_audit(
            base, candidate.probabilities, target, candidate.support
        )
        merge_transition_audit(total, image_audit)
        merge_transition_audit(events[event], image_audit)
        timing["transition"] += time.perf_counter() - stage_started
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"S2 direction fit: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)
            print(
                "S2 timing: "
                + " ".join(
                    f"{key}={value / processed:.3f}s/img"
                    for key, value in timing.items()
                ),
                flush=True,
            )
    if processed and processed % args.progress_every != 0:
        elapsed = time.perf_counter() - started
        print(
            f"S2 direction fit: {processed}/{len(dataset)} | "
            f"{processed / max(elapsed, 1e-9):.2f} img/s",
            flush=True,
        )
        print(
            "S2 timing: "
            + " ".join(
                f"{key}={value / processed:.3f}s/img"
                for key, value in timing.items()
            ),
            flush=True,
        )
    policy = derive_direction_policy(
        total,
        events,
        minimum_pixels=args.minimum_pixels,
        minimum_decisive_precision=args.minimum_decisive_precision,
        minimum_event_nonnegative_fraction=args.minimum_event_nonnegative_fraction,
    )
    payload = {
        "protocol": "SPI_S2_DIRECTION_POLICY_TRAIN_ONLY_v1",
        "split": str(Path(args.list_path).resolve()),
        "processed_images": processed,
        **provenance,
        "s2_config": config.__dict__,
        "global_audit": finalize_transition_audit(total),
        "event_audits": {event: finalize_transition_audit(audit) for event, audit in sorted(events.items())},
        "direction_policy": policy,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"processed": processed, "accepted_directions": [key for key, value in policy["policy"].items() if value]}, indent=2))
    print(f"Saved: {output}")
    print("Validation/official instances/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
