"""Recheck a verified legacy/native E0 and stratify fixed Memory effects by Q mixing.

This never changes a model, bank, checkpoint, neighbor selection, or prediction.
Val GT is used only after frozen predictions and Top-8 IDs have been replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .stats import (
    ARMS,
    CLASS_PURITY_BINS,
    FOREGROUND_BINS,
    ExposureAccumulator,
    StratifiedAccumulator,
    classify_regions,
    route_maps,
    soft_route_exposure,
)


FORMAT = "SRAC_Q_MEMORY_STRATA_V1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        required=True,
        help="Existing passed E0 protocol directory or protocol.json",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Fresh directory; never overwrites an old result",
    )
    p.add_argument(
        "--evidence", help="Optional passed E1 evidence directory or evidence.json"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="First N images for debugging; 0 means formal Val450",
    )
    return p


def own_source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).parent / "run.sh"
    ]:
        if path.name.startswith("test_"):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checked_protocol(path):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (  # noqa: E501
        read_json,
    )

    path = Path(path)
    path = path / "protocol.json" if path.is_dir() else path
    record = read_json(path)
    if (
        record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
        or record.get("complete") is not True
        or record.get("full_validation_evidence") is not True
        or record.get("image_count") != 450
        or record.get("reproduction", {}).get("passed") is not True
        or record.get("scope") != "all"
        or record.get("profile") != "legacy"
        or record.get("pool_mode") != "native"
    ):
        raise ValueError("Requires the passed full-Val450 legacy/native all-path E0")
    for key in (
        "input_files_unchanged",
        "code_unchanged",
        "frozen_evidence_unchanged",
        "numerical_unchanged",
    ):
        if record.get(key) is not True:
            raise ValueError("E0 immutability failed or is absent: " + key)
    names = record.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 selected images are incomplete or duplicated")
    return path, record


def _checked_rows(path, names):
    root = path.parent.resolve()
    log = root / "images.jsonl"
    rows = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(names) or [row.get("image") for row in rows] != names:
        raise ValueError("E0 per-image artifact rows are missing or out of order")
    return rows, log


def _artifact(root, row):
    import numpy as np
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (  # noqa: E501
        digest,
        safe_child,
    )

    relative = row.get("prediction_file")
    if not isinstance(relative, str):
        raise ValueError("Missing E0 prediction path")
    path = safe_child(root, relative)
    if digest(path) != row.get("prediction_sha256"):
        raise ValueError("E0 prediction artifact SHA256 mismatch: " + str(path))
    return np.load(path, allow_pickle=False)


def _checked_evidence(path, bundle):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (  # noqa: E501
        check_artifact,
        digest,
        read_json,
    )

    path = Path(path)
    path = path / "evidence.json" if path.is_dir() else path
    record = read_json(path)
    if (
        record.get("format") != "SRAC_HARDQUERY_EVIDENCE_V1"
        or record.get("image_count") != 450
        or record.get("full_validation_evidence") is not True
        or record.get("reproduction", {}).get("passed") is not True
    ):
        raise ValueError("Optional E1 evidence has not passed full Val450")
    check_artifact(record, bundle)
    immutable = record.get("immutable", {})
    for key in (
        "files",
        "inherited_code",
        "new_code",
        "frozen_models",
        "runtime",
        "protocol",
    ):
        if immutable.get(key) is not True:
            raise ValueError("Optional E1 immutable gate failed: " + key)
    return dict(path=str(path.resolve()), sha256=digest(path))


def _region_inventory_update(inventory, classification, labelled_mass):
    import numpy as np

    mass = np.asarray(labelled_mass, dtype=np.float64)
    for key, names in (
        ("foreground_fraction", FOREGROUND_BINS),
        ("foreground_class_purity", CLASS_PURITY_BINS),
    ):
        bins = classification[
            "foreground_bin" if key == "foreground_fraction" else "class_purity_bin"
        ]
        dest = inventory.setdefault(
            key, {name: {"regions": 0, "labelled_grid_mass": 0.0} for name in names}
        )
        for index, name in enumerate(names):
            selected = bins == index
            dest[name]["regions"] += int(selected.sum())
            dest[name]["labelled_grid_mass"] += float(mass[selected].sum())


def _same_saved_predictions(result, target, artifact):
    import numpy as np
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (  # noqa: E501
        probability_prediction,
    )

    replay = {
        "a1_pixel": probability_prediction(
            result["old_pixel"].float().softmax(1), target.shape[-2:]
        ),
        "direct_region": probability_prediction(result["prob"], target.shape[-2:]),
    }
    for arm, tensor in replay.items():
        if not np.array_equal(
            tensor.detach().cpu().numpy(), artifact["prediction_" + arm]
        ):
            raise ValueError("Frozen front prediction changed since E0: " + arm)


def _verify_confusions(artifact, target, names, totals):
    import numpy as np

    target = np.asarray(target)
    valid = (target >= 0) & (target < 4)
    expected = {
        "target",
        *["prediction_" + arm for arm in names],
        *["confusion_" + arm for arm in names],
        "support_j2",
        "support_j3",
    }
    if not expected.issubset(set(artifact.files)):
        raise ValueError(
            "E0 per-image artifact is missing prediction/confusion/Top-8 fields"
        )
    predictions = {}
    for arm in names:
        pred = np.asarray(artifact["prediction_" + arm])
        if pred.shape != target.shape or ((pred[valid] < 0) | (pred[valid] >= 4)).any():
            raise ValueError("E0 prediction shape/class mismatch: " + arm)
        confusion = np.bincount(
            (target[valid].astype(np.int64) * 4 + pred[valid].astype(np.int64)),
            minlength=16,
        ).reshape(4, 4)
        if not np.array_equal(confusion, artifact["confusion_" + arm]):
            raise ValueError("E0 stored per-image confusion is inconsistent: " + arm)
        totals[arm] += confusion
        predictions[arm] = pred[0]
    return predictions


def _support_mismatch(e0_ids, replay_ids, scores, allowed, topk, max_examples=5):
    """Explain a failed exact-ID replay using scores from the current query.

    The scores are *not* E0 scores: E0 saved IDs, but did not save query keys or
    similarity matrices. This function does not change the equality gate.
    """
    import numpy as np

    e0_ids = np.asarray(e0_ids, dtype=np.int64)
    replay_ids = np.asarray(replay_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    if (
        e0_ids.shape != replay_ids.shape
        or e0_ids.ndim != 2
        or scores.shape != (len(e0_ids), allowed.size)
        or topk != e0_ids.shape[1]
    ):
        raise ValueError("Cannot explain Top-8 mismatch with inconsistent shapes")
    changed = np.any(e0_ids != replay_ids, axis=1)
    set_changed = [
        set(e0_ids[i][e0_ids[i] >= 0].tolist())
        != set(replay_ids[i][replay_ids[i] >= 0].tolist())
        for i in np.flatnonzero(changed)
    ]

    def current_scores(row, ids):
        return [
            None if j < 0 else float(scores[row, j]) for j in ids.tolist()
        ]

    examples = []
    for i in np.flatnonzero(changed)[:max_examples]:
        valid_scores = np.sort(scores[i, allowed])[::-1]
        eighth = float(valid_scores[topk - 1]) if len(valid_scores) >= topk else None
        ninth = float(valid_scores[topk]) if len(valid_scores) > topk else None
        examples.append(
            dict(
                query_index=int(i),
                e0_ids=e0_ids[i].tolist(),
                replay_ids=replay_ids[i].tolist(),
                e0_ids_scores_under_replay_query=current_scores(i, e0_ids[i]),
                replay_ids_scores_under_replay_query=current_scores(i, replay_ids[i]),
                replay_rank_k_score=eighth,
                replay_rank_k_plus_1_score=ninth,
                replay_boundary_margin=(
                    None if eighth is None or ninth is None else eighth - ninth
                ),
            )
        )
    return dict(
        queries=int(len(e0_ids)),
        changed_slots=int(np.count_nonzero(e0_ids != replay_ids)),
        queries_with_rank_change=int(changed.sum()),
        queries_with_set_change=int(sum(set_changed)),
        examples=examples,
        score_basis="Current replay query and unchanged bank; E0 scores were not saved",
    )


def _write_support_mismatch(
    output, arm, image_name, image_index, e0_ids, replay_ids,
    query, bank, event_id, image_id, topk, projector,
    capture, pre, post, first_result,
):
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import write_json
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import project_batches

    query_key, bank_key = project_batches(projector, query, bank["tokens"])
    scores = query_key @ bank_key.T
    allowed = ((bank["events"] != event_id) & (bank["images"] != image_id))
    detail = _support_mismatch(
        e0_ids, replay_ids, scores.detach().cpu().numpy(),
        allowed.detach().cpu().numpy(), topk,
    )
    record = dict(
        image=image_name,
        image_index=image_index,
        arm=arm,
        exact_id_replay_passed=False,
        detail=detail,
        same_process_front_repeat={"status": "pending"},
        interpretation=(
            "A small score margin or a nonzero repeat difference suggests "
            "numerical sensitivity, but cannot identify the cause. "
            "The diagnostic remains failed."
        ),
    )
    # Preserve the actual ID/score evidence even if a second GPU forward fails.
    write_json(output / "mismatch.json", record)
    del query_key, bank_key, scores
    repeat = {}
    try:
        second_result, _ = capture(pre, post)
        for key in ("tokens", "q"):
            first = first_result[key]
            second = second_result[key]
            repeat[key + "_max_abs_difference"] = float(
                (first.float() - second.float()).abs().max().item()
            )
            repeat[key + "_equal"] = bool(torch.equal(first, second))
    except Exception as exc:
        repeat["error"] = type(exc).__name__ + ": " + str(exc)
    record["same_process_front_repeat"] = repeat
    write_json(output / "mismatch.json", record)


def run(cli):
    import torch

    # E0 protocol.run uses @torch.no_grad(); preserve that execution context.
    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (  # noqa: E501
        digest,
        load_verified_bundle,
        new_output,
        verify_unchanged,
        write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import (  # noqa: E501
        PoolGuard,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (  # noqa: E501
        FrozenFrontCapture,
        event,
        full_image_loader,
        parse_batch,
        select_ids,
        validate_selected,
    )
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
        region_targets,
    )

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    path, protocol = _checked_protocol(cli.protocol)
    rows, image_log = _checked_rows(path, protocol["selected_images"])
    output = new_output(cli.output)
    phase, image_name, capture = "initialize", None, None
    started = time.perf_counter()
    initial_source = own_source_hash()
    image_log_hash = digest(image_log)
    write_json(
        output / "summary.json",
        dict(
            format=FORMAT,
            complete=False,
            full_validation_evidence=False,
            phase=phase,
            test_accessed=False,
        ),
    )
    try:
        args = argparse.Namespace(**protocol["arguments"])
        args.protocol = str(path)
        args.output = str(output)
        bundle = load_verified_bundle(
            args
        )  # validates E0 source/runtime/files; no bypass on mismatch
        evidence = _checked_evidence(cli.evidence, bundle) if cli.evidence else None
        names = bundle["val_names"]
        if names != protocol["selected_images"]:
            raise ValueError("Current Val order differs from E0")
        selected = names[: cli.limit] if cli.limit else names
        data = full_image_loader(
            args.dataset, selected, False, args.workers, args.seed + 2
        )
        capture = FrozenFrontCapture(bundle["front"])
        arms = ("a1_pixel", "direct_region", "j1", "j2", "j3")
        confusions = {arm: np.zeros((4, 4), dtype=np.int64) for arm in arms}
        stratified = StratifiedAccumulator()
        exposure_totals = ExposureAccumulator()
        inventory = {}
        report = dict(
            format=FORMAT,
            complete=False,
            full_validation_evidence=False,
            image_count=0,
            test_accessed=False,
            training=False,
            protocol_path=str(path.resolve()),
            protocol_sha256=digest(path),
            protocol_reproduction=protocol["reproduction"],
            source_sha256=initial_source,
            evidence=evidence,
            runtime_settings=bundle["runtime_settings"],
            numerical_profile=protocol["profile"],
            pool_mode=protocol["pool_mode"],
            definitions=dict(
                foreground_bins=list(FOREGROUND_BINS),
                purity_bins=list(CLASS_PURITY_BINS),
                foreground_fraction="y_Intact+y_Damaged+y_Destroyed",
                foreground_class_purity=(
                    "max(y_Intact,y_Damaged,y_Destroyed)/foreground_fraction; "
                    "undefined at zero foreground"
                ),
                cohort_mapping=(
                    "argmax-Q region on feature grid, then nearest-neighbor "
                    "diagnostic resize to GT grid"
                ),
                exposure=(
                    "Q-soft mass for mixed regions, bilinearly resized "
                    "for association statistics"
                ),
                limitation=(
                    "Hard-route bins and soft exposure are associations, not "
                    "a causal decomposition of Memory backprojection; "
                    "GT only defines offline strata"
                ),
            ),
        )
        write_json(output / "summary.json", report)
        phase = "replay_frozen_front"
        with PoolGuard(
            output / "pool_debug",
            mode=args.pool_mode,
            capture="failure",
            check_reference=False,
        ) as guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    pre, post, _, target, image_names = parse_batch(
                        batch, bundle["device"]
                    )
                    image_name = str(image_names[0])
                    if (
                        len(pre) != 1
                        or image_name != selected[index]
                        or rows[index]["image"] != image_name
                    ):
                        raise ValueError("Val loader identity/order differs from E0")
                    if rows[index].get("event") != event(image_name):
                        raise ValueError("E0 event identity mismatch")
                    guard.set_image(image_name)
                    result, _feature = capture(pre, post)
                    # Match E0 order: select neighbors directly after capture,
                    # before reading saved predictions or synchronizing to CPU.
                    bank = bundle["bank"]
                    event_name = event(image_name)
                    event_id = (
                        bank["event_names"].index(event_name)
                        if event_name in bank["event_names"]
                        else -1
                    )
                    query = result["tokens"][0]
                    selected_ids = {
                        "j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                        "j3": select_ids(
                            query,
                            bank,
                            event_id,
                            -1,
                            bundle["topk"],
                            projector=bundle["metric"],
                        ),
                    }
                    for ids in selected_ids.values():
                        validate_selected(ids, bank, event_id, -1)
                    with _artifact(path.parent, rows[index]) as artifact:
                        if not np.array_equal(
                            artifact["target"], target.detach().cpu().numpy()
                        ):
                            raise ValueError("Val GT differs from E0: " + image_name)
                        _same_saved_predictions(result, target, artifact)
                        for arm, ids in selected_ids.items():
                            if not np.array_equal(
                                ids.detach().cpu().numpy(), artifact["support_" + arm]
                            ):
                                _write_support_mismatch(
                                    output, arm, image_name, index,
                                    artifact["support_" + arm],
                                    ids.detach().cpu().numpy(), query, bank,
                                    event_id, -1, bundle["topk"],
                                    bundle["metric"] if arm == "j3" else None,
                                    capture, pre, post, result,
                                )
                                raise ValueError(
                                    "Top-8 differs from E0: " + arm + " / " + image_name
                                )
                        predictions = _verify_confusions(
                            artifact, artifact["target"], arms, confusions
                        )
                        # GT first enters Q labels after predictions and IDs are fixed.
                        distribution, labelled_mass = region_targets(target, result)
                        d = distribution[0].detach().cpu().numpy()
                        mass = labelled_mass[0].detach().cpu().numpy()
                        classification = classify_regions(d, mass)
                        q_np = result["q"][0].detach().cpu().numpy()
                        ids_np = result["ids"].detach().cpu().numpy()
                        maps = route_maps(
                            q_np,
                            ids_np,
                            {
                                "foreground_bin": classification["foreground_bin"],
                                "class_purity_bin": classification["class_purity_bin"],
                            },
                            result["shape"],
                            target.shape[-2:],
                        )
                        candidates = {arm: predictions[arm] for arm in ARMS}
                        image_focus = stratified.add(
                            event_name,
                            artifact["target"][0],
                            predictions["j1"],
                            candidates,
                            maps,
                        )
                        mixed_flags = {
                            "background_foreground_mixed": classification[
                                "foreground_bin"
                            ]
                            == 1,
                            "foreground_class_mixed": (
                                classification["foreground_bin"] == 2
                            )
                            & (classification["class_purity_bin"] == 1),
                        }
                        exposure = {}
                        for key, flags in mixed_flags.items():
                            grid = soft_route_exposure(
                                q_np, ids_np, flags, result["shape"]
                            )
                            full = F.interpolate(
                                torch.from_numpy(grid)[None, None].float(),
                                size=target.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            exposure[key] = full[0, 0].numpy()
                        exposure_totals.add(
                            event_name,
                            artifact["target"][0],
                            predictions["j1"],
                            candidates,
                            exposure,
                        )
                        _region_inventory_update(inventory, classification, mass)
                        row = dict(
                            index=index,
                            image=image_name,
                            event=event_name,
                            artifact_sha256=rows[index]["prediction_sha256"],
                            region_count=len(d),
                            focus=image_focus,
                        )
                        journal.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                    report["image_count"] = index + 1
                    if (index + 1) % 10 == 0 or index + 1 == len(selected):
                        print(
                            f"Q x Memory diagnosis {index + 1}/{len(selected)}",
                            flush=True,
                        )
        phase = "final_integrity_checks"
        immutable = verify_unchanged(bundle)
        immutable["own_code"] = own_source_hash() == initial_source
        immutable["e0_image_log"] = digest(image_log) == image_log_hash
        immutable["pool_guard"] = guard.restored and not guard.failures
        if not all(immutable.values()):
            raise RuntimeError("Q diagnosis immutability failed: " + str(immutable))
        if len(selected) == 450:
            for arm in arms:
                if not np.array_equal(
                    confusions[arm], np.asarray(protocol["confusion"][arm])
                ):
                    raise ValueError("Aggregate confusion differs from E0: " + arm)
        report.update(
            complete=True,
            full_validation_evidence=len(selected) == 450,
            image_count=len(selected),
            immutable=immutable,
            region_inventory=inventory,
            strata=stratified.report(),
            soft_q_mixed_exposure=exposure_totals.report(),
            e0_confusions_verified=len(selected) == 450,
            e0_metrics=protocol["metrics"],
            e0_per_event=protocol["per_event"],
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "diagnostic.json", report)
        write_json(
            output / "summary.json",
            {
                key: report[key]
                for key in (
                    "format",
                    "complete",
                    "full_validation_evidence",
                    "image_count",
                    "immutable",
                    "protocol_path",
                    "protocol_sha256",
                    "elapsed_seconds",
                    "test_accessed",
                )
            },
        )
        return report
    except Exception as exc:
        write_json(
            output / "summary.json",
            dict(
                format=FORMAT,
                complete=False,
                full_validation_evidence=False,
                image_count=locals().get("report", {}).get("image_count", 0),
                phase=phase,
                image=image_name,
                error=type(exc).__name__ + ": " + str(exc),
                test_accessed=False,
            ),
        )
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
