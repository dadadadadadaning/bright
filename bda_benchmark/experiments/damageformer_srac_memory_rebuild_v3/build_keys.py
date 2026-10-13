"""Rebuild paired FP32 Train keys from one current frozen forward per image.

The archived bank supplies member identities only. Its token and label values
never enter extraction or a numerical acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from .paired import extract_pair
from .shared import (
    FORMAT, load_bundle, load_paired_bank, membership_digest, member_plan, provenance,
    source_hash, validate_member_identity, verify_all, write_rebuild_protocol,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed E0 source anchor; no archived token replay")
    p.add_argument("--output", required=True, help="New result directory; existing paths are refused")
    p.add_argument("--limit-images", type=int, default=0,
                   help="0: all fixed support members; positive: explicit smoke image subset")
    return p


def tensor_digest(value):
    """Canonical dtype/shape/value hash, independent of torch serialization."""
    import torch
    if not torch.is_tensor(value):
        raise TypeError("Expected a tensor for content hashing")
    value = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fresh_bank(original, rows, event_names):
    import torch
    count = len(rows)
    bank = {
        "t128": torch.zeros(count, 128, dtype=torch.float32),
        "f256": torch.zeros(count, 256, dtype=torch.float32),
        "distribution": torch.zeros(count, 4, dtype=torch.float32),
        "mass": torch.zeros(count, dtype=torch.float32),
        "label_mass": torch.zeros(count, dtype=torch.float32),
        "active": torch.zeros(count, dtype=torch.bool),
        "eligible": torch.zeros(count, dtype=torch.bool),
        "original_rows": rows.cpu().clone(),
        "member_ids": torch.arange(count, dtype=torch.long),
        "event_names": list(event_names),
    }
    for field in ("events", "images", "regions"):
        value = original[field]
        if (not torch.is_tensor(value)
                or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("Original member identity must be integer: " + field)
        bank[field] = value[rows].detach().cpu().long().clone()
    return bank


def _store_members(bank, destination, region_ids, result, pair):
    import torch
    count = result["count"]
    if (region_ids.dtype != torch.long or bool((region_ids < 0).any())
            or bool((region_ids >= count).any())):
        raise ValueError("Fixed support region outside current frozen Q grid")
    selection = region_ids.to(pair["mass"].device)
    current_valid = result["valid_regions"]
    if (current_valid.dtype != torch.bool or tuple(current_valid.shape) != (1, count)
            or not bool(current_valid[0, selection.to(current_valid.device)].all())):
        raise ValueError("A fixed support is not a valid current region")
    if bool((pair["mass"][selection] <= 0).any()):
        raise ValueError("A fixed support has zero current Q mass")
    if bool((pair["label_mass"][selection] <= 0).any()):
        raise ValueError("A fixed support has zero valid Train label mass")
    widths = {"t128": 128, "f256": 256, "distribution": 4}
    for field in ("t128", "f256", "distribution", "mass", "label_mass"):
        value = pair[field]
        expected_shape = (count, widths[field]) if field in widths else (count,)
        if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
            raise ValueError("Expected actual FP32 paired value with current region shape: " + field)
        chosen = value[selection.to(value.device)].detach().cpu()
        if not bool(torch.isfinite(chosen).all()):
            raise FloatingPointError("Nonfinite selected paired value: " + field)
        bank[field][destination] = chosen
    if pair["active"].dtype != torch.bool or tuple(pair["active"].shape) != (count,):
        raise ValueError("Expected paired active mask for all current regions")
    active = pair["active"][selection.to(pair["active"].device)].detach().cpu()
    # Zero-norm keys remain stored as original members. Both comparisons use
    # the same joint eligibility mask, so neither representation gets an
    # easier support set by silently dropping its own failed rows.
    eligible = (active & (bank["t128"][destination].norm(dim=1) > 1e-8)
                & (bank["f256"][destination].norm(dim=1) > 1e-8))
    bank["active"][destination] = eligible
    bank["eligible"][destination] = eligible


def _roundtrip_bank(bank, path):
    import torch
    with Path(path).open("xb") as stream:
        torch.save(bank, stream)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    if set(restored) != set(bank):
        raise ValueError("Saved paired bank field set differs after reopening")
    for field, expected in bank.items():
        actual = restored[field]
        if torch.is_tensor(expected):
            if (not torch.is_tensor(actual) or actual.dtype != expected.dtype
                    or actual.shape != expected.shape or not torch.equal(actual, expected)):
                raise ValueError("Saved paired bank changed after reopening: " + field)
        elif actual != expected:
            raise ValueError("Saved paired bank metadata changed after reopening: " + field)
    return restored


def run(cli):
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, new_output, write_json
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import NativeA1LogitsCapture, native_grid_probq
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import FrozenFrontCapture, full_image_loader, parse_batch
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths

    if cli.limit_images < 0:
        raise ValueError("--limit-images must be nonnegative")
    output = new_output(cli.output)
    summary = dict(format=FORMAT, complete=False, smoke=cli.limit_images > 0,
                   train_only=True, test_accessed=False, training=False,
                   full_validation_evidence=False, phase="load", image_count=0,
                   extraction_forward_count=0, archived_replay_required=False,
                   historical_replay_required=False,
                   historical_fp32_reproduction_claim=False)
    write_json(output / "summary.json", summary)
    capture = native_capture = guard = forward_handle = None
    manifest = None
    phase, current_image = "load", None
    started, own = time.perf_counter(), source_hash()
    forward_count = [0]
    try:
        bundle = load_bundle(cli.protocol, output)
        args, original = bundle["args"], bundle["bank_cpu"]
        if args.pool_mode != "native":
            raise ValueError("Fresh paired protocol preserves the source native pooling mode")
        plan = member_plan(original, bundle["train_names"], bundle["val_names"], cli.limit_images)
        rows, images, names = plan["rows"], plan["images"], plan["names"]
        if (not torch.is_tensor(rows) or rows.dtype != torch.long or rows.ndim != 1
                or len(rows) == 0 or not names or len(images) != len(names)):
            raise ValueError("Empty or invalid fixed Train member plan")
        bank = _fresh_bank(original, rows, plan["event_names"])
        written = torch.zeros(len(rows), dtype=torch.bool)
        image_ledger = {path.resolve(): digest(path) for name in names
                        for path in sample_paths(args.dataset, name).values()}
        manifest = dict(**summary, **provenance(bundle), support_count=len(rows),
                        original_support_count=len(original["images"]),
                        expected_source_images=len(names), selected_source_images=list(names),
                        selected_train_image_ids=list(images), membership_sha256=membership_digest(bank),
                        key_rule="same current frozen forward: native T128 and Q-weighted mean of native F256",
                        t128_rule="fresh native FP32 result tokens; historical token values unused",
                        f256_rule="fresh actual FP32 classifier input pooled with the same frozen Q",
                        t128_storage="torch.float32", f256_storage="torch.float32",
                        label_storage="torch.float32", query_dtype="float32",
                        label_rule="fresh four-class Train area fractions weighted by the same current Q",
                        label_semantics="classes 0/1/2/3; invalid GT excluded; positive valid-label mass required",
                        support_eligibility_rule="same joint finite, valid, positive-Q-mass, nonzero T128 and F256 mask",
                        selection_rule="fixed archived identities only; no labels, values or Val select members",
                        extraction_context="torch.no_grad; frozen eval front; unchanged native runtime",
                        source_image_fingerprints={str(path): sha for path, sha in image_ledger.items()})
        write_json(output / "manifest.json", manifest)
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 1)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        def count_forward(_module, _inputs, _output):
            forward_count[0] += 1
        forward_handle = bundle["front"].register_forward_hook(count_forward)
        guard = PoolGuard(output / "pool_debug", mode="native", capture="failure", check_reference=False)
        phase = "Train_paired_rebuild"
        native_max = 0.0
        with torch.no_grad(), guard, (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if (len(pre) != 1 or len(image_names) != 1 or index >= len(names)
                        or current_image != names[index]):
                    raise ValueError("Train extraction image/batch/order differs from fixed plan")
                guard.set_image(current_image)
                native_capture.clear()
                before = forward_count[0]
                result, feature = capture(pre, post)
                summary["extraction_forward_count"] = forward_count[0]
                if forward_count[0] - before != 1:
                    raise RuntimeError("Expected exactly one current front forward per support source image")
                logits = native_capture.take()
                _grid, native_check = native_grid_probq(result, feature, logits, target.shape[-2:])
                if native_check.get("passed") is not True:
                    raise ValueError("Same-pass native A1 gate failed: " + current_image)
                native_max = max(native_max, native_check["max_abs"])
                summary["raw_token_dtype"] = str(result["tokens"].dtype)
                if result["tokens"].dtype != torch.float32:
                    raise ValueError("Expected actual FP32 front tokens; refusing to upcast " + str(result["tokens"].dtype))
                pair = extract_pair(result, feature, target)
                if pair["q_check"].get("passed") is not True or pair["label_check"].get("passed") is not True:
                    raise ValueError("Fresh paired Q or Train label conservation check failed")
                destination = (bank["images"] == images[index]).nonzero().flatten()
                if not len(destination) or bool(written[destination].any()):
                    raise ValueError("Duplicate or missing fixed members for current source image")
                region_ids = bank["regions"][destination]
                _store_members(bank, destination, region_ids, result, pair)
                written[destination] = True
                frame = dict(index=index, frame_id=index, image=current_image,
                             train_image_id=images[index], member_ids=bank["member_ids"][destination].tolist(),
                             original_rows=bank["original_rows"][destination].tolist(),
                             region_ids=region_ids.tolist(), rows=len(destination),
                             forward_count=forward_count[0], extraction_forward_count=1,
                             native_a1=native_check,
                             q_check=pair["q_check"], label_check=pair["label_check"],
                             q_sha256=tensor_digest(result["q"]), ids_sha256=tensor_digest(result["ids"]),
                             paired_sha256={field: tensor_digest(bank[field][destination])
                                            for field in ("t128", "f256", "distribution", "mass", "label_mass")},
                             joint_ineligible_rows=int((~bank["eligible"][destination]).sum()))
                journal.write(json.dumps(frame, allow_nan=False) + "\n")
                journal.flush()
                summary.update(phase=phase, image_count=index + 1, current_image=current_image)
                write_json(output / "summary.json", summary)
                if (index + 1) % 25 == 0 or index + 1 == len(names):
                    print(f"Fresh FP32 paired Train rebuild {index + 1}/{len(names)}", flush=True)
        phase = "final_checks"
        if (not bool(written.all()) or summary["image_count"] != len(names)
                or forward_count[0] != len(names)):
            raise ValueError("Not every fixed support received paired keys from exactly one forward per image")
        validate_member_identity(bank, manifest, original)
        if not bool(guard.restored and not guard.failures):
            raise RuntimeError("Pool guard failed or did not restore original operators")
        immutable = verify_all(bundle, own, image_ledger)
        restored = _roundtrip_bank(bank, output / "paired_bank.pt")
        validate_member_identity(restored, manifest, original)
        manifest.update(paired_bank_sha256=digest(output / "paired_bank.pt"),
                        immutable=immutable, image_count=len(names),
                        extraction_forward_count=forward_count[0], native_a1_max_abs=native_max,
                        joint_eligible_count=int(bank["eligible"].sum()),
                        joint_ineligible_count=int((~bank["eligible"]).sum()),
                        zero_norm_t128_count=int((bank["t128"].norm(dim=1) == 0).sum()),
                        zero_norm_f256_count=int((bank["f256"].norm(dim=1) == 0).sum()),
                        small_norm_t128_count=int((bank["t128"].norm(dim=1) <= 1e-8).sum()),
                        small_norm_f256_count=int((bank["f256"].norm(dim=1) <= 1e-8).sum()),
                        fresh_label_checks_passed=True, same_pass_q_checks_passed=True,
                        label_reference_passed=True, q_normalization_passed=True,
                        native_a1_passed=True, pool_guard_passed=True,
                        images_journal_sha256=digest(output / "images.jsonl"),
                        saved_fp32_roundtrip_exact=True, pool_ok=True, pool_guard=guard.report(),
                        elapsed_seconds=time.perf_counter() - started)
        manifest.update(write_rebuild_protocol(output, bundle, restored, manifest))
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
        summary.update(complete=True, phase="complete", support_count=len(rows), immutable=immutable,
                       membership_sha256=manifest["membership_sha256"],
                       paired_bank_sha256=manifest["paired_bank_sha256"],
                       rebuild_protocol_sha256=manifest["rebuild_protocol_sha256"],
                       joint_eligible_count=manifest["joint_eligible_count"],
                       joint_ineligible_count=manifest["joint_ineligible_count"],
                       saved_fp32_roundtrip_exact=True, elapsed_seconds=manifest["elapsed_seconds"])
        write_json(output / "summary.json", summary)
        # Exercise the same independent protocol, provenance, journal and
        # checksum reader that later Val diagnostics will use. A failed reopen
        # marks both records incomplete in the exception handler below.
        load_paired_bank(output, bundle, allow_smoke=cli.limit_images > 0)
        print("FRESH PAIRED TRAIN BANK COMPLETE. No training. No Test. No historical replay claim.", flush=True)
        return summary
    except Exception as error:
        summary.update(complete=False, phase=phase, image=current_image,
                       error=type(error).__name__ + ": " + str(error))
        write_json(output / "summary.json", summary)
        if manifest is not None:
            manifest.update(complete=False, error=summary["error"])
            write_json(output / "manifest.json", manifest)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if forward_handle is not None:
            forward_handle.remove()
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
