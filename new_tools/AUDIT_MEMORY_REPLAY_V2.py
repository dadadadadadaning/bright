#!/usr/bin/env python3
"""Read-only archive identity and controlled one-image Memory replay audit.

This external probe does not modify any experiment package, bank or model.
Its reports are observational diagnostics, never formal Val/Train evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path


# Importing existing experiments must not create bytecode files in them.
sys.dont_write_bytecode = True

FORMAT = "SRAC_MEMORY_REPLAY_AUDIT_V2"
STAGES = ("F256", "ids", "q", "adapted_content", "main_pool_input",
          "stats_mean_input", "stats_std_input", "mass", "geometry_input", "tokens")
ATOL = RTOL = 2e-5


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def _byte_equal(a, b):
    import torch
    return (torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),
                            b.detach().cpu().contiguous().view(torch.uint8)))


def tensor_digest(tensor):
    """Hash actual input dtype, shape and bytes without retaining a full copy."""
    import torch
    result = hashlib.sha256(json.dumps(dict(dtype=str(tensor.dtype), shape=list(tensor.shape)),
                                      sort_keys=True).encode())
    result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def verify_ledger(ledger):
    if any(digest(path) != expected for path, expected in ledger.items()):
        raise RuntimeError("Archive source file changed during the audit")
    return True


def validate_archive(protocol_path, evidence_path, evidence_sha256):
    """CPU-only identity checks. Any failure precedes model/CUDA imports."""
    import torch

    protocol_path, evidence_path = Path(protocol_path).resolve(), Path(evidence_path).resolve()
    if protocol_path.is_dir():
        protocol_path /= "protocol.json"
    if not re.fullmatch(r"[0-9a-fA-F]{64}", evidence_sha256):
        raise ValueError("An explicit 64-character evidence SHA256 is required")
    if digest(evidence_path) != evidence_sha256.lower():
        raise ValueError("Evidence SHA256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
                    full_validation_evidence=True, image_count=450, scope="all",
                    smoke=False, profile="legacy", pool_mode="native", test_accessed=False,
                    input_files_unchanged=True, code_unchanged=True,
                    frozen_evidence_unchanged=True, numerical_unchanged=True)
    if any(protocol.get(k) != value or type(protocol.get(k)) is not type(value)
           for k, value in required.items()) or protocol.get("reproduction", {}).get("passed") is not True:
        raise ValueError("Requires the passed immutable legacy/native full-Val450 E0 protocol")
    names = protocol.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 selected_images must contain 450 unique names")
    source = protocol.get("source_hashes", {})
    fingerprints = protocol.get("input_fingerprints", {})
    input_paths = protocol.get("input_paths", {})
    paths, ledger = {}, {protocol_path: digest(protocol_path), evidence_path: digest(evidence_path)}
    for key, source_key in (("cache_manifest", "cache_manifest_sha256"), ("bank", "bank_sha256")):
        if not input_paths.get(key):
            raise ValueError("E0 lacks its recorded input path: " + key)
        path = Path(input_paths[key]).resolve()
        sha = digest(path)
        if sha != source.get(source_key) or sha != fingerprints.get(key):
            raise ValueError("E0 source/input fingerprint mismatch: " + key)
        paths[key], ledger[path] = path, sha
    manifest = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or manifest.get("split") != "train"
            or manifest.get("smoke") is not False or manifest.get("test_accessed") is not False
            or manifest.get("bank_sha256") != ledger[paths["bank"]]):
        raise ValueError("Original Train cache manifest/bank binding failed")
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=True)
    if (evidence.get("format") != "SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2"
            or evidence.get("protocol_sha256") != ledger[protocol_path]
            or evidence.get("original_bank_sha256") != ledger[paths["bank"]]):
        raise ValueError("Evidence format, E0 or original bank identity mismatch")
    image, image_id = evidence.get("image"), evidence.get("train_image_id")
    if (not isinstance(image, str) or not image or type(image_id) is not int or image_id < 0):
        raise ValueError("Invalid evidence Train image identity")
    rows, regions = evidence.get("original_rows"), evidence.get("region_ids")
    for name, value in (("original_rows", rows), ("region_ids", regions)):
        if (not torch.is_tensor(value) or value.dtype != torch.long or value.ndim != 1
                or not value.numel() or bool((value < 0).any()) or len(value.unique()) != len(value)):
            raise ValueError("Invalid evidence " + name)
    if rows.shape != regions.shape:
        raise ValueError("Evidence row and region counts differ")
    bank = torch.load(paths["bank"], map_location="cpu", weights_only=True)
    if (bool((rows >= len(bank["tokens"])).any())
            or not torch.equal(bank["images"][rows].long(), torch.full_like(rows, image_id))
            or not torch.equal(bank["regions"][rows].long(), regions)):
        raise ValueError("Original bank image/region mapping differs from evidence")
    shards = manifest.get("shards", [])
    if image_id >= len(shards):
        raise ValueError("Evidence Train image ID is outside manifest shards")
    entry = shards[image_id]
    root = paths["cache_manifest"].parent
    shard_path = (root / entry["path"]).resolve()
    if root != shard_path and root not in shard_path.parents:
        raise ValueError("Shard path escapes its original cache")
    if digest(shard_path) != entry.get("sha256"):
        raise ValueError("Original shard SHA256 mismatch")
    paths["shard"], ledger[shard_path] = shard_path, digest(shard_path)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    if (shard.get("name") != image or shard.get("image") != image_id
            or bool((regions >= len(shard["tokens"])).any())
            or not torch.equal(shard.get("region_ids"), torch.arange(len(shard["tokens"])))):
        raise ValueError("Original shard image/name/region IDs mismatch")
    if ("event" in shard and (not bool((bank["events"][rows] == shard["event"]).all()))):
        raise ValueError("Original bank/shard event mapping mismatch")
    archived = evidence.get("archived_tokens")
    if (not _byte_equal(bank["tokens"][rows], archived)
            or not _byte_equal(shard["tokens"][regions], archived)):
        raise ValueError("Archived bank/shard/evidence token dtype or bytes differ")
    if archived.dtype not in (torch.float16, torch.float32) or not bool(torch.isfinite(archived).all()):
        raise ValueError("Invalid archived token precision or values")
    replayed = evidence.get("replayed_fp32")
    if (not torch.is_tensor(replayed) or replayed.dtype != torch.float32
            or replayed.shape != archived.shape or not bool(torch.isfinite(replayed).all())):
        raise ValueError("Evidence must retain finite raw FP32 replay tokens")
    for field in ("distribution", "geometry"):
        if field in bank and field in shard and not _byte_equal(bank[field][rows], shard[field][regions]):
            raise ValueError("Original bank/shard " + field + " bytes differ")
    if input_paths.get("train_list"):
        train_path = Path(input_paths["train_list"]).resolve()
        sha = digest(train_path)
        if sha != fingerprints.get("train_list") or sha != manifest.get("train_sha256"):
            raise ValueError("Original Train list fingerprint mismatch")
        train_names = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if image_id >= len(train_names) or train_names[image_id] != image:
            raise ValueError("Evidence image differs from recorded Train list")
        paths["train_list"], ledger[train_path] = train_path, sha
    return dict(protocol=protocol, evidence=evidence, manifest=manifest, shard=shard, bank=bank,
                original_rows=rows, region_ids=regions, train_image_id=image_id, image=image,
                paths=paths, ledger=ledger, protocol_path=protocol_path,
                checks=dict(evidence_sha256=True, e0_source_binding=True,
                            shard_identity=True, original_member_mapping=True,
                            archived_token_dtype_and_bytes=True, bank_shard_payload=True))


def compare_tensor(before, after):
    """CPU observations, including exact differences and the unchanged tolerance."""
    import torch

    result = dict(shape_equal=before.shape == after.shape, dtype_equal=before.dtype == after.dtype,
                  shapes=[list(before.shape), list(after.shape)], dtypes=[str(before.dtype), str(after.dtype)],
                  finite=bool(torch.isfinite(before).all() and torch.isfinite(after).all()),
                  exact_equal=False, max_abs=None, differing_elements=None,
                  outside_tolerance=None, atol=ATOL, rtol=RTOL)
    if not result["shape_equal"]:
        return result
    result["exact_equal"] = result["finite"] and _byte_equal(before, after)
    result["differing_elements"] = int((before != after).sum())
    if result["finite"]:
        delta = (before.double() - after.double()).abs()
        result["max_abs"] = float(delta.max()) if delta.numel() else 0.0
        result["outside_tolerance"] = int((delta > ATOL + RTOL * before.double().abs()).sum())
    return result


def compare_stages(before, after):
    rows = {name: compare_tensor(before[name], after[name]) for name in STAGES}
    return dict(stages=rows,
                first_recorded_exact_difference=next((name for name in STAGES if not rows[name]["exact_equal"]), None),
                first_recorded_tolerance_difference=next((name for name in STAGES
                    if rows[name]["outside_tolerance"] is None or rows[name]["outside_tolerance"] > 0), None),
                interpretation="First observed recorded difference is not a causal attribution")


class StageCapture:
    """Hooks retain tensor references only; CPU copies happen after front returns."""
    def __init__(self, front):
        self.refs, self.counts, self.handles = {}, {}, []
        for name, module in (("F256", front.base.clf_2), ("main_pool_input", front.embedding.main[0]),
                             ("stats_mean_std_input", front.embedding.stats[0]),
                             ("geometry_input", front.embedding.geometry[0])):
            def capture(_module, inputs, key=name):
                if len(inputs) != 1:
                    raise RuntimeError("Unexpected capture input: " + key)
                self.refs[key] = inputs[0]
                self.counts[key] = self.counts.get(key, 0) + 1
            self.handles.append(module.register_forward_pre_hook(capture))

    def clear(self):
        self.refs.clear()
        self.counts.clear()

    def snapshot(self, result):
        if any(self.counts.get(key) != 1 for key in
               ("F256", "main_pool_input", "stats_mean_std_input", "geometry_input")):
            raise RuntimeError("Every stage hook must fire exactly once per front forward")
        refs = {**self.refs, **{key: result[key] for key in
                ("q", "ids", "adapted_content", "mass", "tokens")}}
        snapshot = {name: tensor.detach().cpu().clone() for name, tensor in refs.items()}
        mean, std = snapshot["stats_mean_std_input"].chunk(2, -1)
        snapshot.update(stats_mean_input=mean, stats_std_input=std)
        return snapshot

    def close(self):
        for handle in self.handles:
            handle.remove()


def _outliers(raw, archived, regions, replay_module):
    """Fixed-tolerance coordinates outside archived rounding cells, without repair."""
    import torch
    actual, stored = raw.double(), archived.double()
    if archived.dtype == torch.float16:
        lower, upper = replay_module._half_rounding_cell(archived)
    else:
        lower = upper = stored
    distance = torch.maximum(lower - actual, actual - upper).clamp_min(0)
    tolerance = ATOL + RTOL * stored.abs()
    return [dict(member=int(row), region_id=int(regions[row]), channel=int(channel),
                 current_fp32=float(actual[row, channel]), stored=float(stored[row, channel]),
                 lower=float(lower[row, channel]), upper=float(upper[row, channel]),
                 interval_distance=float(distance[row, channel]), fixed_tolerance=float(tolerance[row, channel]))
            for row, channel in (distance > tolerance).nonzero().tolist()]


def run(cli):
    import torch

    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(format=FORMAT, audit_complete=False, complete=False,
                   full_validation_evidence=False, training=False, test_accessed=False,
                   observational_only=True, gpu_attempted=False, gpu_ran=False,
                   phase="CPU_archive_identity", samples=[], comparisons=[],
                   script_sha256=digest(__file__), arguments=vars(cli),
                   limitations="No historical FP32 or input image hash reconstruction; observed differences do not establish cause")
    write_json(output / "summary.json", summary)
    capture = None
    try:
        if cli.repeats != 2:
            raise ValueError("This controlled probe fixes --repeats=2")
        archive = validate_archive(cli.protocol, cli.evidence, cli.evidence_sha256)
        summary.update(archive_identity=archive["checks"], image=archive["image"],
                       train_image_id=archive["train_image_id"],
                       original_rows=archive["original_rows"].tolist(), region_ids=archive["region_ids"].tolist(),
                       prior_report=archive["evidence"].get("report"),
                       input_ledger={str(path): sha for path, sha in archive["ledger"].items()})
        if cli.cpu_only:
            summary["archive_files_unchanged"] = verify_ledger(archive["ledger"])
            summary.update(audit_complete=True, phase="CPU_only_complete", gpu_ran=False)
            write_json(output / "summary.json", summary)
            return summary
        repo = Path(cli.repo).resolve()
        if not (repo / "bda_benchmark").is_dir():
            raise ValueError("--repo must identify the existing BRIGHT repository")
        sys.path.insert(0, str(repo))
        from bda_benchmark.experiments.damageformer_srac_memory_key_compare_v2 import shared, replay
        from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
            full_image_loader, parse_batch, region_targets)
        from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths
        summary["phase"] = "load_E0_bound_bundle"
        own = shared.source_hash()
        if archive["evidence"].get("source_sha256") != own:
            raise ValueError("Current V2 source differs from the saved failed replay evidence")
        summary["gpu_attempted"] = True
        bundle = shared.load_bundle(archive["protocol_path"], output)
        args, front = bundle["args"], bundle["front"]
        if args.pool_mode != "native" or bundle["device"].type != "cuda":
            raise ValueError("Controlled GPU audit requires the recorded native CUDA E0")
        models = [front, bundle["metric"], *bundle["decoders"].values()]
        if (any(module.training for model in models for module in model.modules())
                or any(p.requires_grad for model in models for p in model.parameters())):
            raise ValueError("Every loaded model must be eval with frozen parameters")
        if bundle["train_names"][archive["train_image_id"]] != archive["image"]:
            raise ValueError("Current E0 Train image identity differs from archive")
        ledger = dict(archive["ledger"])
        ledger[Path(__file__).resolve()] = summary["script_sha256"]
        ledger.update({path.resolve(): digest(path) for path in
                       sample_paths(args.dataset, archive["image"]).values()})
        shared.verify_all(bundle, own, ledger)
        loader = full_image_loader(args.dataset, [archive["image"]], False, args.workers, args.seed + 1)
        batches = iter(loader)
        pre, post, _loc, target, names = parse_batch(next(batches), bundle["device"])
        if len(pre) != 1 or list(names) != [archive["image"]]:
            raise ValueError("Full-image loader did not return the exact requested Train image")
        if next(batches, None) is not None:
            raise ValueError("Unexpected additional Train image")
        summary.update(gpu_ran=True, runtime_settings=bundle["runtime_settings"],
                       input_shapes=dict(pre=list(pre.shape), post=list(post.shape)),
                       input_tensor_sha256=dict(pre=tensor_digest(pre), post=tensor_digest(post)),
                       current_image_fingerprints={str(path): sha for path, sha in ledger.items()
                                                   if path not in archive["ledger"] and path != Path(__file__).resolve()})
        capture = StageCapture(front)
        snapshots = {}
        for condition in ("native_inference", "native_no_grad", "guarded_no_grad"):
            for repeat in range(2):
                summary["phase"] = f"{condition}_{repeat + 1}_forward"
                write_json(output / "summary.json", summary)
                context = torch.inference_mode() if condition == "native_inference" else torch.no_grad()
                guard = (PoolGuard(output / f"pool_{repeat + 1}", mode="native", capture="failure",
                                   check_reference=False) if condition == "guarded_no_grad" else None)
                capture.clear()
                with context, (guard if guard is not None else nullcontext()):
                    if guard is not None:
                        guard.set_image(archive["image"])
                    result = front(pre, post)
                    labels, _weights = region_targets(target, result)
                    snap = capture.snapshot(result)
                    selected = archive["region_ids"].to(result["tokens"].device)
                    slim = dict(format=FORMAT, condition=condition, repeat=repeat + 1,
                                image=archive["image"], original_rows=archive["original_rows"],
                                region_ids=archive["region_ids"],
                                tokens=snap["tokens"][0, archive["region_ids"]].clone(),
                                distribution=labels[0, selected].detach().cpu().clone(),
                                mass=snap["mass"][0, archive["region_ids"]].clone(),
                                geometry=snap["geometry_input"][0, archive["region_ids"]].clone())
                if slim["tokens"].dtype != torch.float32:
                    raise ValueError("Replay must retain actual raw FP32 tokens")
                artifact = output / f"{condition}_{repeat + 1}.pt"
                torch.save(slim, artifact)
                shard, regions = archive["shard"], archive["region_ids"]
                row = dict(condition=condition, repeat=repeat + 1, artifact=artifact.name,
                           artifact_sha256=digest(artifact),
                           selected_storage_report=replay.stored_token_replay(slim["tokens"], archive["evidence"]["archived_tokens"]),
                           selected_vs_failed_V2_fp32=compare_tensor(archive["evidence"]["replayed_fp32"], slim["tokens"]),
                           labels_vs_shard=compare_tensor(shard["distribution"][regions], slim["distribution"]),
                           mass_vs_shard=compare_tensor(shard["mass"][regions], slim["mass"]),
                           geometry_storage_report=replay.stored_token_replay(slim["geometry"].float(), shard["geometry"][regions]),
                           outliers=_outliers(slim["tokens"], archive["evidence"]["archived_tokens"], regions, replay),
                           pool_guard=guard.report() if guard is not None else None)
                for (previous_condition, previous_repeat), previous in snapshots.items():
                    if (previous_condition == condition and previous_repeat == 0
                            or previous_condition != condition and previous_repeat == repeat):
                        summary["comparisons"].append(dict(before=f"{previous_condition}_{previous_repeat + 1}",
                            after=f"{condition}_{repeat + 1}", **compare_stages(previous, snap)))
                snapshots[(condition, repeat)] = snap
                summary["samples"].append(row)
                write_json(output / "summary.json", summary)
                print(f"Replay audit {condition} {repeat + 1}/2", flush=True)
        summary["phase"] = "immutable_checks"
        summary["input_tensor_sha256_after"] = dict(pre=tensor_digest(pre), post=tensor_digest(post))
        summary["input_tensor_bytes_unchanged"] = (summary["input_tensor_sha256_after"]
                                                    == summary["input_tensor_sha256"])
        if not summary["input_tensor_bytes_unchanged"]:
            raise RuntimeError("Actual pre/post input tensor bytes changed during the six forwards")
        summary["immutable"] = shared.verify_all(bundle, own, ledger)
        summary.update(audit_complete=True, phase="observational_audit_complete")
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        # After a CUDA assertion, do not touch CUDA tensors or GPU fingerprints.
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summary.update(audit_complete=False, complete=False, full_validation_evidence=False,
                       error=f"{type(error).__name__}: {error}", traceback_file="traceback.txt",
                       failure_note="Preserve this failure; use a fresh process after any CUDA device error")
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--evidence-sha256", required=True)
    p.add_argument("--output", required=True, help="New directory; existing paths are never overwritten")
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--repeats", type=int, default=2, choices=(2,))
    p.add_argument("--cpu-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
