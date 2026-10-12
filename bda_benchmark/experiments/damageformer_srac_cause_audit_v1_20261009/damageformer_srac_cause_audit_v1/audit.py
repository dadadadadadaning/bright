"""Read-only full-Val diagnostics using a stable intermediate or final snapshot."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    digest, lists, loader, new_output, parse_batch, save_json, seed_everything,
)
from .core import MODES, capture, gt_mixing_diagnostics, interventions, sharpen, split_ids
from .diagnostics import DiagnosticAccumulator
from .support import diagnostic_source_hash, load_snapshot, module_fingerprint, verify_splits


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    return p


def check_reproduction(result, payload, smoke):
    if smoke:
        return dict(passed=None, reason="2-image smoke cannot reproduce full-Val metrics")
    expected_a1 = payload["load_report"]["expected_valid_miou"]
    expected_region = payload["valid_miou"]
    observed = result["metrics"]
    checks = {}
    for name, expected in (("a1", expected_a1), ("region", expected_region)):
        actual = observed[name]["valid_miou"]
        checks[name] = dict(expected=expected, actual=actual, absolute_error=abs(actual-expected), passed=abs(actual-expected) <= 1e-5)
    return dict(passed=result["image_count"] == 450 and all(c["passed"] for c in checks.values()), tolerance=1e-5, checks=checks)


@torch.inference_mode()
def run(args):
    if args.workers < 0 or not 0 < args.threshold < 1 or not math.isfinite(args.power) or args.power <= 1:
        raise ValueError("Invalid diagnostic settings")
    _, val = lists(args.train_list, args.val_list)
    if Path(args.output).exists():
        raise ValueError("Preserve existing output and choose a new directory: " + args.output)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if not (Path(args.checkpoint).parent / "snapshot.json").is_file():
        raise ValueError("Run the snapshot command first; auditing a live best.pth is not supported")
    model, payload = load_snapshot(args.checkpoint, args.a1_checkpoint, device)
    verify_splits(payload, args.train_list, args.val_list)
    if payload["smoke"] and not args.smoke:
        raise ValueError("A smoke-trained front cannot be used as formal evidence")
    expected = payload.get("load_report", {}).get("expected_valid_miou")
    if type(expected) not in (int, float) or not math.isfinite(expected) or not 0 <= expected <= 1:
        raise ValueError("Missing valid audited A1 baseline; no images evaluated")
    output = new_output(args.output)
    identity_before = module_fingerprint(model)
    save_json(output / "manifest.json", dict(
        arguments=vars(args), checkpoint_sha256=digest(args.checkpoint), checkpoint_step=payload["step"],
        parent_complete=bool(payload.get("run_complete")), parent_smoke=payload["smoke"], a1_sha256=payload["a1_sha256"],
        source_hash=payload["source_hash"], train_sha256=digest(args.train_list), val_sha256=digest(args.val_list),
        diagnostic_source_hash=diagnostic_source_hash(),
        train_images_accessed=False, test_accessed=False, weights_updated=False, smoke=args.smoke,
        intervention="Fixed features, classifier and native structural adapter; only specified token aggregation/read routing changes",
    ))
    data = loader(args.dataset, val[:2] if args.smoke else val, workers=args.workers, seed=args.seed)
    diagnostics = DiagnosticAccumulator(MODES, args.threshold)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)  # model.to(device) has initialized CUDA
    started = time.perf_counter()
    inference_seconds = 0.0
    for batch in data:
        pre, post, _loc, gt, names = parse_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        native, aux = capture(model, pre, post)
        probabilities, building, routes = interventions(model, native, aux, gt.shape[-2:], args.power, args.threshold)
        predictions = {key: value.argmax(1) for key, value in probabilities.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter()-begin
        q, ids, count, shape = native["q"], native["ids"], native["count"], native["shape"]
        native_mix, buckets = gt_mixing_diagnostics(gt, q, ids, count, shape)
        sharp_mix, _ = gt_mixing_diagnostics(gt, sharpen(q, args.power), ids, count, shape)
        split_index, split_count = split_ids(ids, building, count, args.threshold)
        split_mix, _ = gt_mixing_diagnostics(gt, q, split_index, split_count, shape)
        diagnostics.update(gt, predictions, building, native["entropy_map"], buckets, names[0],
                           dict(native=native_mix, sharp=sharp_mix, predicted_split=split_mix), routes)
        if diagnostics.images and len(diagnostics.images) % 10 == 0:
            print(f"cause audit {len(diagnostics.images)}/{len(data)}", flush=True)
    result = diagnostics.finalize()
    reproduction = check_reproduction(result, payload, args.smoke)
    unchanged = identity_before == module_fingerprint(model)
    result.update(
        complete=unchanged and (args.smoke or reproduction["passed"]), smoke=args.smoke,
        checkpoint_step=payload["step"], weights_and_buffers_unchanged=unchanged,
        native_Q_unchanged=True, reproduction=reproduction, test_accessed=False,
        timing=dict(wall_seconds=time.perf_counter()-started, forward_all_interventions_seconds=inference_seconds,
                    first_image_included=True, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    warning="Audit with multiple interventions/diagnostics; not deployment throughput"),
        interpretation="Inference interventions and error associations are hypotheses, not retrained ablations or proof about frozen encoders",
    )
    save_json(output / "audit.json", result)  # preserve expensive evidence even if reproduction fails
    print(json.dumps({key: value["valid_miou"] for key, value in result["metrics"].items()}, indent=2), flush=True)
    if not result["complete"]:
        raise RuntimeError("Evidence saved to audit.json but native reproduction/buffer gate FAILED; do not interpret intervention gains")
    print("CAUSE AUDIT COMPLETE. No weights or buffers updated. No Test.", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
