"""Fresh same-pass Val audit of Q mixing and frozen J1/J2/J3 Memory effects.

E0 fixes the source and numerical protocol, but its Top-8 IDs are not an
equality gate here. Each new prediction and its Q/Top-8 come from one forward.
The original, fixed 1e-5 reproduction check remains the formal gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .gates import final_evidence_gate
from bda_benchmark.experiments.damageformer_srac_q_memory_diagnostic_v1.stats import (
    ARMS, CLASS_PURITY_BINS, FOREGROUND_BINS,
    ExposureAccumulator, StratifiedAccumulator, classify_regions,
    route_maps, soft_route_exposure,
)


FORMAT = "SRAC_Q_MEMORY_SAMEPASS_V1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed full Val450 legacy/native E0")
    p.add_argument("--output", required=True, help="Fresh output directory")
    p.add_argument("--evidence", help="Optional passed E1 evidence")
    p.add_argument("--limit", type=int, default=0,
                   help="First N Val images for debugging; 0 means formal Val450")
    return p


def own_source_hash():
    """Track this package and the two shared, new-only diagnostic helpers."""
    root = Path(__file__).parent
    external = root.parent / "damageformer_srac_q_memory_diagnostic_v1"
    paths = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    paths += list(root.glob("*.sh"))
    paths += [external / "run.py", external / "stats.py"]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(root.parent)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _inventory_add(inventory, classification, labelled_mass):
    import numpy as np
    mass = np.asarray(labelled_mass, dtype=np.float64)
    for field, names, bins in (
        ("foreground_fraction", FOREGROUND_BINS, classification["foreground_bin"]),
        ("foreground_class_purity", CLASS_PURITY_BINS, classification["class_purity_bin"]),
    ):
        dest = inventory.setdefault(field, {
            name: {"regions": 0, "labelled_grid_mass": 0.0} for name in names
        })
        for index, name in enumerate(names):
            selected = bins == index
            dest[name]["regions"] += int(selected.sum())
            dest[name]["labelled_grid_mass"] += float(mass[selected].sum())


def _post_prediction_q_statistics(target, result, prediction, event_name,
                                  stratified, exposure_totals, inventory):
    """GT appears only here, after all five predictions and both Top-8 sets."""
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import region_targets

    distribution, labelled_mass = region_targets(target, result)
    d = distribution[0].detach().cpu().numpy()
    mass = labelled_mass[0].detach().cpu().numpy()
    classification = classify_regions(d, mass)
    q = result["q"][0].detach().cpu().numpy()
    ids = result["ids"].detach().cpu().numpy()
    shape = target.shape[-2:]
    maps = route_maps(q, ids, {
        "foreground_bin": classification["foreground_bin"],
        "class_purity_bin": classification["class_purity_bin"],
    }, result["shape"], shape)
    candidates = {arm: prediction[arm][0] for arm in ARMS}
    gt = target[0].detach().cpu().numpy()
    focus = stratified.add(event_name, gt, prediction["j1"][0], candidates, maps)
    flags = {
        "background_foreground_mixed": classification["foreground_bin"] == 1,
        "foreground_class_mixed": (
            (classification["foreground_bin"] == 2)
            & (classification["class_purity_bin"] == 1)
        ),
    }
    exposures = {}
    for name, marked in flags.items():
        grid = soft_route_exposure(q, ids, marked, result["shape"])
        full = F.interpolate(torch.from_numpy(grid)[None, None].float(),
                             size=shape, mode="bilinear", align_corners=False)
        exposures[name] = full[0, 0].numpy()
    exposure_totals.add(event_name, gt, prediction["j1"][0], candidates, exposures)
    _inventory_add(inventory, classification, mass)
    return focus, len(d), d, mass


def run(cli):
    import torch
    # E0 protocol.run is decorated with @torch.no_grad(). Match it.
    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch
    from bda_benchmark.experiments.damageformer_srac_q_memory_diagnostic_v1.run import (
        _checked_evidence, _checked_protocol,
    )
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, load_verified_bundle, new_output, safe_child, verify_unchanged,
        write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, event, full_image_loader, parse_batch,
        select_ids, validate_selected, gather_support, decoder_forward,
        upsample_probability, probability_prediction, confusion_update, metrics,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report

    if not 0 <= cli.limit <= 450:
        raise ValueError("--limit must be 0..450")
    protocol_path, protocol = _checked_protocol(cli.protocol)
    output = new_output(cli.output)
    started, phase, image_name = time.perf_counter(), "initialize", None
    capture = None
    source_before = own_source_hash()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, test_accessed=False, training=False, phase=phase)
    write_json(output / "summary.json", summary)
    try:
        args = argparse.Namespace(**protocol["arguments"])
        args.protocol, args.output = str(protocol_path), str(output)
        bundle = load_verified_bundle(args)
        evidence = _checked_evidence(cli.evidence, bundle) if cli.evidence else None
        names = bundle["val_names"]
        if names != protocol["selected_images"]:
            raise ValueError("Current Val image order differs from E0")
        selected = names[:cli.limit] if cli.limit else names
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        arms = ("a1_pixel", "direct_region", "j1", "j2", "j3")
        cms = {arm: torch.zeros(4, 4, dtype=torch.long) for arm in arms}
        by_event = {}
        stratified, exposure_totals, inventory = StratifiedAccumulator(), ExposureAccumulator(), {}
        prediction_dir = output / "predictions"
        prediction_dir.mkdir()
        phase = "same_pass_validation"
        with PoolGuard(output / "pool_debug", mode=args.pool_mode,
                       capture="failure", check_reference=False) as guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                    image_name = str(image_names[0])
                    if len(pre) != 1 or index >= len(selected) or image_name != selected[index]:
                        raise ValueError("Unexpected Val image/order/batch")
                    guard.set_image(image_name)
                    # Exactly one frozen front forward feeds fresh Q, Top-8 and predictions.
                    result, pixel_feature = capture(pre, post)
                    bank = bundle["bank"]
                    event_name = event(image_name)
                    event_id = (bank["event_names"].index(event_name)
                                if event_name in bank["event_names"] else -1)
                    query = result["tokens"][0]
                    selected_ids = {
                        "j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                        "j3": select_ids(query, bank, event_id, -1, bundle["topk"],
                                         projector=bundle["metric"]),
                    }
                    for ids in selected_ids.values():
                        validate_selected(ids, bank, event_id, -1)
                    support = {arm: gather_support(ids, bank)
                               for arm, ids in selected_ids.items()}
                    probability = {
                        "a1_pixel": result["old_pixel"].float().softmax(1),
                        "direct_region": result["prob"],
                    }
                    for arm in ("j1", "j2", "j3"):
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  pixel_feature, result,
                                                  support["j2"], support["j3"])
                        probability[arm] = upsample_probability(decoded["logits"],
                                                                target.shape[-2:])
                    prediction = {arm: probability_prediction(value, target.shape[-2:])
                                  for arm, value in probability.items()}
                    # No GT-dependent region label, routing decision or cohort exists above.
                    focus, region_count, region_distribution, region_mass = _post_prediction_q_statistics(
                        target, result,
                        {arm: value.detach().cpu().numpy() for arm, value in prediction.items()},
                        event_name, stratified, exposure_totals, inventory,
                    )
                    image_cms = {}
                    arrays = {
                        "target": target.detach().cpu().numpy().astype(np.int16),
                        "q": result["q"].detach().cpu().numpy(),
                        "region_ids": result["ids"].detach().cpu().numpy(),
                        "grid_shape": np.asarray(result["shape"], dtype=np.int32),
                        "region_distribution": region_distribution,
                        "region_labelled_mass": region_mass,
                    }
                    for arm, value in prediction.items():
                        cm = torch.zeros(4, 4, dtype=torch.long)
                        confusion_update(cm, target, value)
                        cms[arm].add_(cm)
                        by_event.setdefault(event_name, {}).setdefault(
                            arm, torch.zeros_like(cm)).add_(cm)
                        image_cms[arm] = cm.tolist()
                        arrays["prediction_" + arm] = value.detach().cpu().numpy().astype(np.uint8)
                        arrays["confusion_" + arm] = cm.numpy()
                    for arm, ids in selected_ids.items():
                        arrays["support_" + arm] = ids.detach().cpu().numpy()
                    prediction_file = prediction_dir / f"{index:04d}.npz"
                    np.savez_compressed(prediction_file, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=image_name, event=event_name,
                        region_count=region_count, confusion=image_cms, focus=focus,
                        prediction_file=prediction_file.relative_to(output).as_posix(),
                        prediction_sha256=digest(prediction_file),
                    ), ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    summary["image_count"] = index + 1
                    if (index + 1) % 10 == 0 or index + 1 == len(selected):
                        print(f"Q x Memory same-pass {index + 1}/{len(selected)}", flush=True)
        phase = "final_reproduction_and_integrity"
        journal_rows = [json.loads(line) for line in
                        (output / "images.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        if (len(journal_rows) != len(selected) or
                [row.get("image") for row in journal_rows] != selected or
                [row.get("index") for row in journal_rows] != list(range(len(selected)))):
            raise ValueError("Same-pass per-image journal is missing or out of order")
        for row in journal_rows:
            file = safe_child(output, row["prediction_file"])
            if digest(file) != row["prediction_sha256"]:
                raise ValueError("Same-pass prediction artifact hash mismatch: " + str(file))
        actual = {arm: metrics(cm) for arm, cm in cms.items()}
        smoke = len(selected) != 450
        reproduction = reproduction_report(actual, bundle["reference"],
                                           smoke=smoke, image_count=len(selected))
        # Informative only: two independent 1e-5-valid replays need not be
        # within 1e-5 of one another, and Top-8 row IDs need not match E0.
        e0_metric_comparison = reproduction_report(actual, protocol["metrics"],
                                                   smoke=smoke, image_count=len(selected))
        immutable = verify_unchanged(bundle)
        immutable["own_source"] = source_before == own_source_hash()
        pool_ok = bool(guard.restored and not guard.failures)
        formal = final_evidence_gate(len(selected), len(names), reproduction,
                                    immutable, pool_ok)
        complete = (len(selected) == summary["image_count"]
                    and all(immutable.values()) and pool_ok
                    and (smoke or reproduction["passed"] is True))
        report = dict(
            format=FORMAT, complete=complete, full_validation_evidence=formal,
            image_count=len(selected), expected_image_count=len(names),
            reproduction=reproduction, e0_metric_comparison=e0_metric_comparison,
            metrics=actual,
            per_event={name: {arm: metrics(cm) for arm, cm in values.items()}
                       for name, values in by_event.items()},
            immutable=immutable, pool_guard=guard.report(),
            own_source_sha256=source_before,
            protocol_path=str(protocol_path.resolve()), protocol_sha256=digest(protocol_path),
            e1_evidence=evidence,
            numerical_profile=protocol["profile"], pool_mode=protocol["pool_mode"],
            support_id_rule="Fresh same-pass J2/J3 Top-8; E0 Top-8 equality is not required",
            gt_rule="Val GT is used only after all five predictions and support IDs are fixed",
            diagnostic_caveat="Q strata show association, not causal effect or building-instance counts",
            foreground_bins=list(FOREGROUND_BINS), purity_bins=list(CLASS_PURITY_BINS),
            region_inventory=inventory, strata=stratified.report(),
            soft_q_mixed_exposure=exposure_totals.report(),
            test_accessed=False, training=False,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "audit.json", report)
        summary.update(complete=complete, full_validation_evidence=formal,
                       reproduction_passed=reproduction["passed"], immutable=immutable,
                       phase="complete" if complete else "failed_reproduction_or_integrity",
                       audit_file="audit.json")
        write_json(output / "summary.json", summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Same-pass reproduction/immutability failed; 1e-5 unchanged")
        return report
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False, phase=phase,
                       image=image_name,
                       error=type(exc).__name__ + ": " + str(exc))
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
