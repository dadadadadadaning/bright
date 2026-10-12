"""Read-only boundary-channel interventions from one stable frozen snapshot."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.core import capture
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    lists, loader, new_output, parse_batch, save_json, save_pt,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities

from .boundary import boundary_maps
from .evidence import Evidence, LABELS
from .probability_check import ProbabilityContractError, projection_diagnostics, validate_probability
from .runtime import load_fixed, reproduction_gate


MODES = ("a1", "region", "edge_zero", "edge_mean", "edge_change")
PAIRS = (("a1", "region"), ("region", "edge_zero"), ("region", "edge_mean"), ("region", "edge_change"), ("edge_mean", "edge_change"))
PROBABILITY_TOLERANCE = 2e-5


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--start_index", type=int, default=0, help="Zero-based Val start; nonzero is a diagnostic subset, NOT a resumed/full audit")
    p.add_argument("--max_images", type=int, default=0, help="0 uses the remaining list; positive values are diagnostic subsets only")
    return p


@torch.no_grad()
def boundary_interventions(model, native, auxiliary, output_size):
    """Recompute adapter -> token embedding -> classifier for each E channel.

    F, Q, B and H are held fixed, along with weights and candidate IDs. Tokens
    are deliberately recomputed: fixing tokens would sever E's prediction path.
    There is no ground-truth argument anywhere in this prediction function.
    """
    q, ids, shape, count = (native[key] for key in ("q", "ids", "shape", "count"))
    fixed = dict(Q=q, ids=ids, F=auxiliary["fused_roi"], B=auxiliary["p_building"], H=native["entropy_map"])
    before = {key: value.clone() for key, value in fixed.items()}
    maps = boundary_maps(q, ids, shape)
    recomputed, region_logits = {}, {}
    for name, edge in maps.items():
        adapted, _ = model.adapter(fixed["F"], fixed["B"], edge, fixed["H"])
        tokens, _mass, _geometry, _valid = model.embedding(adapted, fixed["F"], q, ids, shape, count)
        region_logits[name] = model.region_classifier(tokens)
        recomputed[name] = project_region_probabilities(region_logits[name], q, ids, shape, output_size)
    max_difference = float((recomputed["old"] - native["prob"]).abs().max())
    native_edge_difference = float((maps["old"] - native["soft_boundary"]).abs().max())
    unchanged = {key: bool(torch.equal(value, before[key])) for key, value in fixed.items()}
    probabilities = dict(a1=native["old_pixel"].float().softmax(1), region=native["prob"])
    probabilities.update({"edge_" + key: recomputed[key] for key in ("zero", "mean", "change")})
    probability_checks = {}
    for name, probability in probabilities.items():
        try:
            probability_checks[name] = validate_probability(probability, name)
        except ProbabilityContractError as error:
            # Replays diagnose the failure. They never replace a prediction,
            # relax the original tolerance, or silently renormalize outputs.
            if name.startswith("edge_"):
                try:
                    error.diagnostics["projection_replays"] = projection_diagnostics(
                        region_logits[name.removeprefix("edge_")], q, ids, shape, output_size, probability,
                    )
                except Exception as replay_error:
                    error.diagnostics["projection_replay_error"] = str(replay_error)
            error.diagnostics["fixed_tensors_unchanged"] = unchanged
            error.diagnostics["previous_probability_checks"] = probability_checks
            raise
    building = F.interpolate(fixed["B"].detach().float(), shape, mode="bilinear", align_corners=False)
    checks = dict(
        old_probability_max_abs_difference=max_difference,
        old_probability_reproduced=math.isfinite(max_difference) and max_difference <= PROBABILITY_TOLERANCE,
        native_edge_max_abs_difference=native_edge_difference,
        native_edge_reproduced=math.isfinite(native_edge_difference) and native_edge_difference <= PROBABILITY_TOLERANCE,
        fixed_tensors_unchanged=unchanged,
        probability_checks=probability_checks,
    )
    return probabilities, building, maps, checks


def selected_validation(args, val):
    """A late-image probe is never reported as full-Val or resumed evidence."""
    start, maximum = getattr(args, "start_index", 0), getattr(args, "max_images", 0)
    if start < 0 or maximum < 0 or start >= len(val):
        raise ValueError("Invalid Val subset: start_index must be within Val and max_images nonnegative")
    if args.smoke and (start or maximum):
        raise ValueError("Use either smoke or a targeted Val probe, not both")
    subset = bool(args.smoke or start or maximum)
    selected = val[:2] if args.smoke else val[start: start + maximum if maximum else None]
    scope = dict(full_validation=not subset, diagnostic_subset=subset, start_index_zero_based=start,
                 requested_max_images=maximum, selected_images=len(selected), resumed=False,
                 warning="Subset results cannot replace full-Val450 evidence" if subset else None)
    return selected, scope


def runtime_diagnostics(device):
    source = inspect.getsource(project_region_probabilities)
    return dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), device=str(device),
                projection_file=inspect.getsourcefile(project_region_probabilities),
                projection_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                projection_source=source,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


class BoundaryDistributions:
    """Area-weighted feature-grid E distributions, evaluated after predictions."""

    def __init__(self):
        self.sums, self.images = {}, []
        self.edges = np.linspace(0., 1., 21)

    def update(self, maps, gt, name):
        y = gt[0].detach().long().cpu()
        valid = (y >= 0) & (y < 4)
        shape = maps["old"].shape[-2:]
        fractions = F.one_hot(y.clamp(0, 3), 4).permute(2, 0, 1).float() * valid[None]
        boundary = boundary_mask(y)
        original = dict(
            GT_class_area_fractions=fractions[None],
            GT_boundary_area_fraction=boundary[None, None].float(),
            GT_interior_area_fraction=(valid & ~boundary)[None, None].float(),
            GT_class_boundary_area_fractions=(fractions * boundary[None])[None],
            GT_class_interior_area_fractions=(fractions * (~boundary)[None])[None],
        )
        area_maps = {key: F.interpolate(value, shape, mode="area") for key, value in original.items()}
        weights = dict(all_feature_grid=np.ones(shape, dtype=np.float64))
        for index, label in enumerate(LABELS):
            weights[label] = area_maps["GT_class_area_fractions"][0, index].numpy()
            weights[label + "__boundary"] = area_maps["GT_class_boundary_area_fractions"][0, index].numpy()
            weights[label + "__interior"] = area_maps["GT_class_interior_area_fractions"][0, index].numpy()
        weights["boundary"] = area_maps["GT_boundary_area_fraction"][0, 0].numpy()
        weights["interior"] = area_maps["GT_interior_area_fraction"][0, 0].numpy()
        values = {key: value[0, 0].detach().float().cpu().numpy().astype(np.float64) for key, value in maps.items()}
        image = dict(image=str(name), feature_shape=list(shape), distributions={})
        for map_name, value in values.items():
            image["distributions"][map_name] = dict(
                mean=float(value.mean()), std=float(value.std()),
                minimum=float(value.min()), maximum=float(value.max()),
                quantiles_05_50_95=np.quantile(value, [.05, .5, .95]).tolist(),
            )
            for group, weight in weights.items():
                key = (map_name, group)
                dest = self.sums.setdefault(key, dict(weight=0., total=0., squared=0., histogram=np.zeros(20), minimum=None, maximum=None))
                mass = float(weight.sum())
                dest["weight"] += mass
                dest["total"] += float((value * weight).sum())
                dest["squared"] += float((value * value * weight).sum())
                dest["histogram"] += np.histogram(value, bins=self.edges, weights=weight)[0]
                if mass:
                    selected = value[weight > 0]
                    lower, upper = float(selected.min()), float(selected.max())
                    dest["minimum"] = lower if dest["minimum"] is None else min(dest["minimum"], lower)
                    dest["maximum"] = upper if dest["maximum"] is None else max(dest["maximum"], upper)
        image["GT_class_feature_area"] = {label: float(weights[label].sum()) for label in LABELS}
        self.images.append(image)
        return {key: value.detach().cpu().clone() for key, value in maps.items()}, area_maps

    def finalize(self):
        distributions = {}
        for (map_name, group), row in self.sums.items():
            weight = row["weight"]
            mean = row["total"] / weight if weight else None
            std = math.sqrt(max(0., row["squared"] / weight - mean * mean)) if weight else None
            distributions.setdefault(map_name, {})[group] = dict(
                area_weight_feature_pixels=weight, mean=mean, std=std,
                minimum=row["minimum"], maximum=row["maximum"],
                histogram_area_weight=row["histogram"].tolist(),
                histogram_fraction=(row["histogram"] / weight).tolist() if weight else [None] * 20,
            )
        return dict(
            feature_grid_distributions=distributions, histogram_edges=self.edges.tolist(), per_image=self.images,
            definitions=dict(
                old="max over adjacent neighbors of 1 - aligned global-region probability dot product",
                change="same neighbor aggregation of 0.5 * squared L2 change between global-region distributions",
                mean="Per-image spatial constant equal to unweighted old-channel feature-grid mean",
                class_weights="Original GT one-hot/valid masks area-resampled onto Q grid; ignored area contributes zero",
                boundary_weights="Original radius-2 GT boundary and interior masks area-resampled; both class sides retained",
                histograms="Area-weighted E histogram, left inclusive; final bin includes 1",
                scale_confound="E_change removes uncertainty-related baseline and changes scale/distribution. Its inference delta is not isolated boundary-quality evidence. E_mean controls old-channel mean without spatial variation; zero/mean are inference interventions, not retrained ablations.",
            ),
        )


@torch.inference_mode()
def run(args):
    if args.workers < 0 or not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        raise ValueError("Invalid boundary audit settings")
    if Path(args.output).exists():
        raise ValueError("Preserve existing output and choose a new directory: " + str(args.output))
    model, payload, manifest = load_fixed(args)
    _, val = lists(args.train_list, args.val_list)
    selected, scope = selected_validation(args, val)
    output = new_output(args.output)
    identity_before = module_fingerprint(model)
    manifest.update(
        train_images_accessed=False, test_accessed=False, weights_updated=False,
        modes=list(MODES), pairs=[list(pair) for pair in PAIRS],
        intervention="Fixed F,Q,B,H and all weights; replace E, then recompute adapter, embedding, region head and same-Q projection",
        controls="Zero and per-image old mean are inference interventions only; not retrained ablations",
        map_examples="First two predeclared Val images, not selected by outcomes; feature-grid maps include GT only for offline inspection",
        audit_scope=scope,
    )
    save_json(output / "manifest.json", manifest)
    device = torch.device(args.device)
    environment = runtime_diagnostics(device)
    save_json(output / "runtime.json", environment)
    data = loader(args.dataset, selected, workers=args.workers, seed=args.seed)
    evidence = Evidence(MODES, PAIRS, threshold=args.threshold)
    distributions = BoundaryDistributions()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    inference_seconds, checks, map_files = 0., [], []
    for index, batch in enumerate(data):
        pre, post, _loc, gt, names = parse_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        val_index = scope["start_index_zero_based"] + index
        try:
            native, auxiliary = capture(model, pre, post)
            probabilities, building, maps, check = boundary_interventions(model, native, auxiliary, gt.shape[-2:])
        except Exception as error:
            failure = dict(complete=False, smoke=args.smoke, audit_scope=scope,
                image=str(names[0]), val_index_zero_based=val_index, val_position_one_based=val_index + 1,
                processed_images=index, exception_type=type(error).__name__, message=str(error),
                probability_diagnostics=getattr(error, "diagnostics", None),
                runtime=environment, weights_updated=False, test_accessed=False,
                prediction_policy="No fallback, renormalization, dropped image, or tolerance relaxation",
                source_checkpoint_sha256=manifest.get("checkpoint_sha256"))
            save_json(output / "failure.json", failure)
            # Completed-image counts are retained for debugging only. The failed
            # image is NOT skipped and the run never receives complete=True.
            partial = evidence.finalize()
            partial.update(complete=False, smoke=args.smoke, test_accessed=False, audit_scope=scope,
                           failure_image=str(names[0]), weights_updated=False,
                           interpretation="Incomplete prefix only; not full-Val evidence or a resume checkpoint")
            save_json(output / "partial_audit.json", partial)
            print(f"FAILED Val position {val_index + 1}, image={names[0]}; diagnostics: {output / 'failure.json'}", flush=True)
            raise
        predictions = {key: value.argmax(1) for key, value in probabilities.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - begin
        checks.append(dict(image=str(names[0]), **check))
        evidence.update(gt, predictions, building, names[0])
        saved_maps, area_maps = distributions.update(maps, gt, names[0])
        if val_index < 2:
            filename = f"feature_maps_{index:02d}.pt"
            save_pt(output / filename, dict(image=str(names[0]), E=saved_maps, building=building.cpu(), **area_maps))
            map_files.append(filename)
        if (index + 1) % 10 == 0:
            print(f"boundary audit {index + 1}/{len(data)}", flush=True)
        if scope["diagnostic_subset"]:
            print(f"boundary probe Val position {val_index + 1}: {names[0]} OK", flush=True)
    result = evidence.finalize()
    reproduction = (dict(passed=None, reason="Diagnostic subset only; full-Val reproduction not evaluated")
                    if scope["diagnostic_subset"] else reproduction_gate(result, payload, False))
    unchanged = identity_before == module_fingerprint(model)
    native_pass = bool(checks) and all(check["old_probability_reproduced"] and check["native_edge_reproduced"] for check in checks)
    fixed_pass = bool(checks) and all(all(check["fixed_tensors_unchanged"].values()) for check in checks)
    result.update(
        complete=unchanged and native_pass and fixed_pass and (scope["diagnostic_subset"] or reproduction["passed"]),
        audit_scope=scope,
        smoke=args.smoke, checkpoint_step=payload["step"], parent_complete=bool(payload.get("run_complete")),
        diagnostic_only=True, reusable_as_parent=False, weights_and_buffers_unchanged=unchanged,
        native_Q_unchanged=fixed_pass, fixed_F_Q_B_H_unchanged=fixed_pass, reproduction=reproduction,
        native_recomputation=dict(passed=native_pass, tolerance=PROBABILITY_TOLERANCE, per_image=checks),
        boundary_statistics=distributions.finalize(), representative_map_files=map_files,
        train_images_accessed=False, test_accessed=False, weights_updated=False,
        timing=dict(
            wall_seconds=time.perf_counter() - started, forward_all_interventions_seconds=inference_seconds,
            first_image_included=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            warning="Multiple interventions plus offline diagnostics; not deployment throughput",
        ),
        interpretation="Frozen inference interventions only. E_change changes uncertainty baseline and scale; compare with E_mean/zero controls. Gains do not prove a retrained boundary method or encoder limit.",
    )
    save_json(output / "audit.json", result)
    print(json.dumps({key: value["valid_miou"] for key, value in result["metrics"].items()}, indent=2), flush=True)
    if not result["complete"]:
        raise RuntimeError("Evidence saved to audit.json, but reproduction/immutability gate FAILED; do not interpret intervention gains")
    label = "BOUNDARY SUBSET PROBE" if scope["diagnostic_subset"] else "BOUNDARY AUDIT"
    print(label + " COMPLETE. No training, weight changes, or Test access.", flush=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
