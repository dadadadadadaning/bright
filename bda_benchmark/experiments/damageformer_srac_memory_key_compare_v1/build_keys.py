"""Extract Q-mean F256 for exactly the existing Train support members."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .shared import FORMAT, load_bundle, provenance, source_hash, verify_all, validate_member_identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit-images", type=int, default=0,
                   help="0: all original members; positive: smoke subset of member source images")
    return p


def run(cli):
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, new_output, write_json
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import NativeA1LogitsCapture, native_grid_probq
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import FrozenFrontCapture, full_image_loader, parse_batch, region_targets
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import mean_pool
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths

    if cli.limit_images < 0:
        raise ValueError("--limit-images must be nonnegative")
    output = new_output(cli.output)
    summary = dict(format=FORMAT, complete=False, smoke=cli.limit_images > 0,
                   train_only=True, test_accessed=False, training=False,
                   full_validation_evidence=False, phase="load", image_count=0)
    write_json(output / "summary.json", summary)
    capture = native_capture = guard = None
    phase, current_image = "load", None
    started, own = time.perf_counter(), source_hash()
    try:
        bundle = load_bundle(cli.protocol, output)
        args, original = bundle["args"], bundle["bank_cpu"]
        all_images = original["images"].long().unique(sorted=True).tolist()
        selected_images = all_images[:cli.limit_images] if cli.limit_images else all_images
        if not selected_images:
            raise ValueError("Original bank has no Train support images")
        selected = torch.zeros(len(original["tokens"]), dtype=torch.bool)
        for image in selected_images:
            selected |= original["images"] == image
        rows = selected.nonzero().flatten()
        storage_dtype = original["tokens"].dtype
        if storage_dtype not in (torch.float16, torch.float32):
            raise ValueError("Original T128 supports must have FP16 or FP32 storage")
        # Both support representations have the SAME original stored precision;
        # T128 and all labels/metadata remain byte-identical to original rows.
        bank = {"t128": original["tokens"][rows].cpu().clone(),
                "f256": torch.empty(len(rows), 256, dtype=storage_dtype),
                "original_rows": rows.clone(), "event_names": list(original["event_names"])}
        for key in ("distribution", "events", "images", "regions"):
            bank[key] = original[key][rows].cpu().clone()
        written = torch.zeros(len(rows), dtype=torch.bool)
        names = [bundle["train_names"][i] for i in selected_images]
        image_ledger = {path.resolve(): digest(path) for name in names
                        for path in sample_paths(args.dataset, name).values()}
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 1)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode, capture="failure", check_reference=False)
        manifest = dict(**summary, **provenance(bundle), support_count=len(rows),
                        original_support_count=len(original["tokens"]),
                        expected_source_images=len(names), key_rule="mean_Q of same-pass native F256",
                        t128_rule="unchanged original bank payload", f256_storage=str(storage_dtype),
                        query_dtype="float32", label_rule="unchanged original four-class Train soft distribution",
                        comparison_tolerance=dict(atol=2e-5, rtol=2e-5),
                        source_image_fingerprints={str(path): sha for path, sha in image_ledger.items()})
        write_json(output / "manifest.json", manifest)
        phase = "Train_support_extraction"
        native_max = label_max = token_max = 0.0
        with torch.no_grad(), guard, (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            import json
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if len(pre) != 1 or index >= len(names) or current_image != names[index]:
                    raise ValueError("Train extraction image/batch/order differs")
                guard.set_image(current_image)
                native_capture.clear()
                result, feature = capture(pre, post)
                logits = native_capture.take()
                _grid, replay = native_grid_probq(result, feature, logits, target.shape[-2:])
                if replay.get("passed") is not True:
                    raise ValueError("Same-pass native A1 gate failed: " + current_image)
                native_max = max(native_max, replay["max_abs"])
                if feature.shape[1] != 256:
                    raise ValueError("Expected native F256")
                means, mass = mean_pool(feature.float(), result["q"], result["ids"], result["count"])
                destination = (bank["images"] == selected_images[index]).nonzero().flatten()
                region_ids = bank["regions"][destination].long()
                if bool((region_ids < 0).any()) or bool((region_ids >= result["count"]).any()):
                    raise ValueError("Original member region outside frozen Q grid")
                if bool((mass[0, region_ids.to(mass.device)] <= 0).any()):
                    raise ValueError("An original support now has zero Q mass")
                # Compare replayed T128 at the original bank storage precision;
                # keep original bank keys rather than replacing a failed replay.
                replay_tokens = result["tokens"][0, region_ids.to(result["tokens"].device)]
                replay_tokens = replay_tokens.to(storage_dtype).float()
                expected_tokens = bank["t128"][destination].to(replay_tokens.device).float()
                token_agreement = close_report(replay_tokens, expected_tokens)
                if token_agreement.get("passed") is not True:
                    raise ValueError("Original T128 supports differ on replay: " + str(token_agreement))
                token_max = max(token_max, token_agreement["max_abs"])
                regenerated, _weight = region_targets(target, result)
                expected = bank["distribution"][destination].to(regenerated.device)
                agreement = close_report(regenerated[0, region_ids.to(regenerated.device)], expected)
                if agreement.get("passed") is not True:
                    raise ValueError("Original support labels differ from current Q: " + str(agreement))
                label_max = max(label_max, agreement["max_abs"])
                values = means[0, region_ids.to(means.device)].cpu().to(storage_dtype)
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("F256 becomes nonfinite after support storage conversion")
                bank["f256"][destination] = values
                written[destination] = True
                row = dict(index=index, image=current_image, train_image_id=selected_images[index],
                           rows=len(destination), native_a1=replay, soft_label_agreement=agreement,
                           stored_precision_t128_replay=token_agreement)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                summary.update(phase=phase, image_count=index + 1, current_image=current_image)
                write_json(output / "summary.json", summary)
                if (index + 1) % 25 == 0 or index + 1 == len(names):
                    print(f"Paired key extraction {index + 1}/{len(names)}", flush=True)
        phase = "final_checks"
        if not bool(written.all()) or summary["image_count"] != len(names):
            raise ValueError("Not every original support received an F256 key")
        validate_member_identity(bank, manifest, original)
        pool_ok = bool(guard.restored and not guard.failures)
        if not pool_ok:
            raise RuntimeError("Pool guard failed or did not restore the original operators")
        immutable = verify_all(bundle, own, image_ledger)
        torch.save(bank, output / "paired_bank.pt")
        manifest.update(complete=True, paired_bank_sha256=digest(output / "paired_bank.pt"),
                        immutable=immutable, image_count=len(names), native_a1_max_abs=native_max,
                        support_label_max_abs=label_max, t128_replay_max_abs=token_max,
                        pool_ok=pool_ok, pool_guard=guard.report(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        summary.update(complete=True, phase="complete", support_count=len(rows), immutable=immutable,
                       paired_bank_sha256=manifest["paired_bank_sha256"], elapsed_seconds=manifest["elapsed_seconds"])
        write_json(output / "summary.json", summary)
        print("PAIRED TRAIN KEYS COMPLETE. No training. No Test. This is not Val evidence.", flush=True)
        return summary
    except Exception as error:
        summary.update(complete=False, phase=phase, image=current_image,
                       error=type(error).__name__ + ": " + str(error))
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
