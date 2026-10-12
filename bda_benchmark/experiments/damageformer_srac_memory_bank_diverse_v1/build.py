"""Build a same-capacity, quality-eligible, diverse SRAC Train Memory.

The source E0 protocol fixes checkpoints, cache, split, device and numerical
conditions.  This program never processes Val/Test images and never changes Q,
the retrieval projection, or the Reader.  Train soft labels rank support rows.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.metadata import disaster_id
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event, save_pt
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    digest, load_verified_bundle, read_json, safe_child, verify_unchanged, write_json,
)

from .bank import (FORMAT, FIELDS, bounded_push, bucket_counts, choose_diverse_bucket,
                   key_redundancy, purity_summary, quality_key, source_code_hash,
                   validate_shard)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed full legacy/native E0 directory or protocol.json")
    p.add_argument("--output", required=True, help="Fresh output directory; source cache is read-only")
    p.add_argument("--seed", type=int, default=3407, help="Stable tie breaker only")
    p.add_argument("--per_image_cap", type=int, default=4,
                   help="Preferred maximum rows from one Train image per event/class bucket")
    p.add_argument("--candidate_multiplier", type=int, default=4,
                   help="Bounded candidate heap is this many times each original bucket quota")
    p.add_argument("--quality_weight", type=float, default=0.2,
                   help="Fixed quality weight in projected-cosine coverage score")
    return p


def _source_arguments(protocol_path: Path, output: Path):
    protocol = read_json(protocol_path)
    if (protocol.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1" or protocol.get("scope") != "all"
            or protocol.get("complete") is not True or protocol.get("full_validation_evidence") is not True
            or protocol.get("image_count") != 450 or protocol.get("reproduction", {}).get("passed") is not True
            or protocol.get("profile") != "legacy" or protocol.get("pool_mode") != "native"
            or protocol.get("test_accessed") is not False):
        raise ValueError("Requires passed formal legacy/native E0 with full Val450 and original 1e-5 gate")
    source = protocol.get("arguments", {})
    required = ("joint_run", "a1_checkpoint", "dataset", "train_list", "val_list", "device",
                "profile", "pool_mode", "seed")
    if any(key not in source for key in required):
        raise ValueError("E0 arguments omit required source identity")
    args = argparse.Namespace(**source)
    args.protocol = str(protocol_path)
    args.output = str(output)
    args.smoke = False
    return args, protocol


def _read_shard(bundle, image_id: int):
    record = bundle["cache"]["shards"][image_id]
    path = safe_child(bundle["cache_root"], record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"Train shard SHA256 mismatch at image {image_id}")
    shard = torch.load(path, map_location="cpu", weights_only=False)
    name = bundle["train_names"][image_id]
    expected_event = bundle["bank_cpu"]["event_names"].index(event(name))
    eligible, dominant, purity = validate_shard(
        shard, image_id=image_id, image_name=name, event_id=expected_event,
        expected_dim=bundle["bank_cpu"]["tokens"].shape[1],
        min_purity=float(bundle["cache"]["min_purity"]),
    )
    return path, shard, eligible, dominant, purity


def _candidate_scan(bundle, quotas: list[list[int]], seed: int,
                    per_image_cap: int, multiplier: int):
    cap_heaps, unrestricted_heaps = {}, {}
    capacities = {}
    for e, row in enumerate(quotas):
        for c, quota in enumerate(row):
            if quota:
                key = (e, c)
                cap_heaps[key], unrestricted_heaps[key] = [], []
                capacities[key] = max(quota, multiplier * quota)
    seen = defaultdict(int)
    source_shards = []
    for image_id, record in enumerate(bundle["cache"]["shards"]):
        path, shard, eligible, dominant, purity = _read_shard(bundle, image_id)
        bundle["input_paths"][f"train_shard_{image_id:05d}"] = path
        bundle["inputs_before"][f"train_shard_{image_id:05d}"] = record["sha256"]
        source_shards.append(dict(path=record["path"], sha256=record["sha256"],
                                  name=shard["name"], event=int(shard["event"])))
        e = int(shard["event"])
        for c in range(4):
            key = (e, c)
            if key not in capacities:
                continue
            image_heap = []
            ids = (eligible & (dominant == c)).nonzero().flatten().tolist()
            seen[key] += len(ids)
            for region_id in ids:
                rank = quality_key(float(purity[region_id]), float(shard["mass"][region_id]),
                                   seed, shard["name"], region_id)
                item = (rank, image_id, region_id)
                bounded_push(unrestricted_heaps[key], item, capacities[key])
                bounded_push(image_heap, item, per_image_cap)
            for item in image_heap:
                bounded_push(cap_heaps[key], item, capacities[key])
        if (image_id + 1) % 100 == 0 or image_id + 1 == len(bundle["cache"]["shards"]):
            print(f"Verified and ranked Train shards {image_id + 1}/{len(bundle['cache']['shards'])}", flush=True)
    return cap_heaps, unrestricted_heaps, seen, source_shards


def _candidate_keys(bundle, cap_heaps, unrestricted_heaps):
    """Project only the bounded Train candidate pool with frozen J3 metric."""
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import encode_keys

    identities = {(item[1], item[2]) for heap in (*cap_heaps.values(), *unrestricted_heaps.values())
                  for item in heap}
    by_image = defaultdict(list)
    for image_id, region_id in identities:
        by_image[image_id].append(region_id)
    ordered_ids, raw = [], []
    with torch.inference_mode():
        for image_id, regions in sorted(by_image.items()):
            _, shard, eligible, _, _ = _read_shard(bundle, image_id)
            regions = sorted(regions)
            if any(not bool(eligible[region]) for region in regions):
                raise ValueError("A bounded Train candidate became ineligible")
            raw.append(shard["tokens"][regions].clone())
            ordered_ids.extend((image_id, region) for region in regions)
        keys = {}
        if raw:
            tokens = torch.cat(raw)
            for start in range(0, len(tokens), 4096):
                chunk = tokens[start:start + 4096].to(bundle["device"])
                encoded = encode_keys(chunk, bundle["metric"]).cpu()
                if encoded.shape != chunk.shape or not bool(torch.isfinite(encoded).all()):
                    raise ValueError("Frozen metric produced invalid candidate keys")
                for identity, key in zip(ordered_ids[start:start + len(encoded)], encoded):
                    keys[identity] = key
    if len(keys) != len(identities):
        raise ValueError("Projected Train candidate ledger is incomplete")
    return keys


def _project_bank(bundle, bank):
    """Diagnostic keys only; the saved Reader payload remains original tokens."""
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import encode_keys

    pieces = []
    with torch.inference_mode():
        for start in range(0, len(bank["tokens"]), 4096):
            tokens = bank["tokens"][start:start + 4096].to(bundle["device"])
            pieces.append(encode_keys(tokens, bundle["metric"]).cpu())
    return torch.cat(pieces)


def _chosen_rows(quotas, cap_heaps, unrestricted_heaps, keys,
                 per_image_cap, quality_weight):
    selected = []
    stats = {}
    for e, row in enumerate(quotas):
        for c, quota in enumerate(row):
            key = (e, c)
            picked = choose_diverse_bucket(
                cap_heaps.get(key, []), unrestricted_heaps.get(key, []), quota,
                keys, per_image_cap, quality_weight)
            for item, fallback in picked:
                selected.append(dict(event=e, dominant_class=c, image=item[1], region=item[2],
                                     purity=item[0][0], q_mass_capped=item[0][1],
                                     fallback_from_image_cap=fallback))
            stats[f"{e}:{c}"] = dict(quota=quota, selected=len(picked),
                                     relaxed=int(sum(fallback for _item, fallback in picked)),
                                     preferred=int(sum(not fallback for _item, fallback in picked)),
                                     mean_purity=(sum(item[0][0] for item, _ in picked) / quota
                                                  if quota else None))
    if len({(row["image"], row["region"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate Train row selected into bank")
    return selected, stats


def _materialize(bundle, selected):
    by_image = defaultdict(list)
    for position, row in enumerate(selected):
        by_image[row["image"]].append((position, row))
    materialized = [None] * len(selected)
    for image_id, choices in sorted(by_image.items()):
        _, shard, eligible, dominant, _purity = _read_shard(bundle, image_id)
        for position, row in choices:
            r = row["region"]
            if (r < 0 or r >= len(eligible) or not bool(eligible[r]) or
                    int(dominant[r]) != row["dominant_class"] or
                    int(shard["event"]) != row["event"]):
                raise ValueError("Selected source row changed or became ineligible")
            materialized[position] = dict(tokens=shard["tokens"][r].clone(),
                                          distribution=shard["distribution"][r].clone(),
                                          geometry=shard["geometry"][r].clone())
    if not materialized or any(row is None for row in materialized):
        raise ValueError("Could not materialize every selected Train row")
    event_names = bundle["bank_cpu"]["event_names"]
    bank = dict(tokens=torch.stack([row["tokens"] for row in materialized]),
                distribution=torch.stack([row["distribution"] for row in materialized]),
                geometry=torch.stack([row["geometry"] for row in materialized]),
                events=torch.tensor([row["event"] for row in selected], dtype=torch.long),
                images=torch.tensor([row["image"] for row in selected], dtype=torch.long),
                regions=torch.tensor([row["region"] for row in selected], dtype=torch.long),
                disasters=torch.tensor([disaster_id(event_names[row["event"]]) for row in selected], dtype=torch.long),
                event_names=event_names,
                checkpoint_sha256=bundle["bank_cpu"]["checkpoint_sha256"])
    return bank


def _verify_schema(old, new):
    if set(old) != set(new) or set(old) != set(FIELDS) | {"event_names", "checkpoint_sha256"}:
        raise ValueError("New bank tensor field schema differs from original bank")
    for key in FIELDS:
        if (old[key].dtype != new[key].dtype or old[key].shape[1:] != new[key].shape[1:]):
            raise ValueError("New bank tensor dtype/trailing shape differs: " + key)
    if old["event_names"] != new["event_names"] or old["checkpoint_sha256"] != new["checkpoint_sha256"]:
        raise ValueError("New bank metadata differs from old bank")
    if not torch.isfinite(new["tokens"]).all() or not torch.isfinite(new["geometry"]).all():
        raise ValueError("New bank contains non-finite features")


def run(cli):
    if (cli.per_image_cap < 1 or cli.candidate_multiplier < 1
            or not math.isfinite(cli.quality_weight) or cli.quality_weight < 0):
        raise ValueError("per_image_cap, candidate_multiplier and quality_weight must be valid")
    protocol_path = Path(cli.protocol)
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    protocol_path = protocol_path.resolve()
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(format=FORMAT, complete=False, train_only=True, test_accessed=False,
                  training=False, val_labels_used_for_selection=False)
    write_json(output / "summary.json", status)
    try:
        source_args, _protocol = _source_arguments(protocol_path, output)
        bundle = load_verified_bundle(source_args)
        if len(bundle["cache"].get("shards", [])) != 2918 or len(bundle["train_names"]) != 2918:
            raise ValueError("Formal bank requires exactly Train2918 source shards")
        if bundle["cache"].get("smoke") is not False or bundle["cache"].get("test_accessed") is not False:
            raise ValueError("Source Train cache is not a formal Train-only cache")
        source_code_before = source_code_hash()
        old = bundle["bank_cpu"]
        quotas = bucket_counts(old, len(old["event_names"]))
        cap_heaps, unrestricted_heaps, seen, source_shards = _candidate_scan(
            bundle, quotas, cli.seed, cli.per_image_cap, cli.candidate_multiplier)
        for e, row in enumerate(quotas):
            for c, quota in enumerate(row):
                if seen[(e, c)] < quota:
                    raise ValueError(f"Train bucket {(e, c)} has {seen[(e, c)]} eligible rows, needs {quota}")
        candidate_keys = _candidate_keys(bundle, cap_heaps, unrestricted_heaps)
        selected, fallback_stats = _chosen_rows(
            quotas, cap_heaps, unrestricted_heaps, candidate_keys,
            cli.per_image_cap, cli.quality_weight)
        new = _materialize(bundle, selected)
        _verify_schema(old, new)
        new_counts = bucket_counts(new, len(old["event_names"]))
        if new_counts != quotas or len(new["tokens"]) != len(old["tokens"]):
            raise ValueError("Exact event×class quota or total capacity changed")
        code_after = source_code_hash()
        if code_after != source_code_before:
            raise ValueError("New bank construction code changed during build")
        immutable = verify_unchanged(bundle)
        save_pt(output / "bank.pt", new)
        write_json(output / "selected_rows.json", selected)
        field_schema = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                        for key, value in new.items() if torch.is_tensor(value)}
        per_bucket_quality = {}
        old_keys, new_keys = _project_bank(bundle, old), _project_bank(bundle, new)
        for e, row in enumerate(quotas):
            for c, _quota in enumerate(row):
                old_mask = (old["events"] == e) & (old["distribution"].argmax(-1) == c)
                new_mask = (new["events"] == e) & (new["distribution"].argmax(-1) == c)
                per_bucket_quality[f"{e}:{c}"] = dict(old=purity_summary(old["distribution"][old_mask]),
                                                       new=purity_summary(new["distribution"][new_mask]),
                                                       old_key_redundancy=key_redundancy(old_keys[old_mask]),
                                                       new_key_redundancy=key_redundancy(new_keys[new_mask]),
                                                       selection=fallback_stats[f"{e}:{c}"],
                                                       eligible_seen=seen[(e, c)])
        manifest = dict(format=FORMAT, complete=True, bank_file="bank.pt", train_only=True,
                        training=False, test_accessed=False, val_labels_used_for_selection=False,
                        source_protocol_sha256=digest(protocol_path), protocol_sha256=digest(protocol_path),
                        source_protocol_path=str(protocol_path),
                        old_bank_sha256=bundle["hashes"]["bank_sha256"],
                        source_bank_sha256=bundle["hashes"]["bank_sha256"],
                        frozen_metric_checkpoint_sha256=bundle["hashes"]["metric_sha256"],
                        old_bank_path=str(bundle["input_paths"]["bank"]),
                        new_bank_sha256=digest(output / "bank.pt"), bank_sha256=digest(output / "bank.pt"),
                        cache_manifest_sha256=bundle["hashes"]["cache_manifest_sha256"],
                        source_cache_manifest_sha256=bundle["hashes"]["cache_manifest_sha256"],
                        source_code_sha256=source_code_before, new_code_sha256=source_code_before,
                        train_shards=len(source_shards), train_shard_records=source_shards,
                        train_shard_manifest_sha256=digest(bundle["input_paths"]["cache_manifest"]),
                        event_names=old["event_names"], capacity=len(new["tokens"]),
                        old_capacity=len(old["tokens"]), bank_regions=len(new["tokens"]),
                        event_class_counts=new_counts, old_event_class_counts=quotas,
                        field_schema=field_schema, selection=dict(seed=cli.seed,
                            preferred_per_image_cap=cli.per_image_cap,
                            candidate_multiplier=cli.candidate_multiplier,
                            quality_weight=cli.quality_weight,
                            candidate_pool="Train purity-ranked rows within exact event x class quota; Q mass capped at 16 resolves ties",
                            priority="Highest-quality seed, then greedy projected-cosine coverage plus fixed quality term",
                            metric="Frozen original J3 projector; original bank payload/checkpoint untouched",
                            cap_relaxation="Relax per-image cap only if every remaining candidate exceeds it; exact quota is unchanged"),
                        bucket_quality=per_bucket_quality, selected_rows_sha256=digest(output / "selected_rows.json"),
                        original_immutable=immutable)
        write_json(output / "manifest.json", manifest)
        status.update(complete=True, train_shards=2918, capacity=len(new["tokens"]),
                      exact_event_class_counts=True, immutable=immutable,
                      manifest_sha256=digest(output / "manifest.json"))
        write_json(output / "summary.json", status)
        print(json.dumps(dict(capacity=len(new["tokens"]), selected_rows=len(selected),
                              relaxed=sum(v["relaxed"] for v in fallback_stats.values()),
                              all_checks_passed=True)), flush=True)
    except Exception as exc:
        status["failure"] = dict(type=type(exc).__name__, message=str(exc))
        write_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    run(parser().parse_args())
