"""E2: matched bank-query and full-Train difficult-query search projectors.

Both arms start from the identity and have the same optimizer/budget/reliable
support subset. Neither fitting nor checkpoint selection uses Val labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ..damageformer_srac_qmetric_v1.metric import (
    RetrievalProjector, balanced_anchor_indices, cross_event_contrastive_loss,
)
from ..damageformer_srac_transformer_v1.common import event, save_pt
from .io_utils import (
    add_source_arguments, artifact_identity, check_artifact, digest, fail_summary,
    load_verified_bundle, new_output, read_json, safe_child, verify_unchanged, write_json,
)
from .metric_learning import HardQuerySampler, soft_cross_event_loss, validate_distribution

METRIC_FORMAT = "SRAC_HARDQUERY_METRIC_V1"
QUERY_FORMAT = "SRAC_HARDQUERY_QUERIES_V1"


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument("--protocol", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--arms", nargs="+", choices=("bank", "hard"), default=["bank", "hard"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--per_class", type=int, default=16)
    p.add_argument("--purity", type=float, default=.8)
    p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hard_fraction", type=float, default=.5)
    p.add_argument("--hard_quantile", type=float, default=.75)
    return p


class QueryStore:
    """Verified Train-only shard mappings plus compact CPU sampling metadata.

    Token storage is memory-mapped on CPU and only selected rows reach the GPU.
    This avoids allocating an extra dense copy of the full Train feature cache.
    """

    def __init__(self, path, bundle, smoke):
        path = Path(path)
        if path.is_dir():
            path = path / "query_manifest.json"
        self.path, self.root = path.resolve(), path.parent.resolve()
        record = read_json(path)
        check_artifact(record, bundle, allow_smoke=smoke)
        if record.get("format") != QUERY_FORMAT or record.get("train_only") is not True:
            raise ValueError("Expected completed Train-only hard-query cache")
        if record.get("smoke") is not bool(smoke):
            raise ValueError("Do not mix smoke/formal query caches")
        shards = record.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("No query shards")
        if (not smoke and len(shards) != 2918) or (smoke and len(shards) != 4):
            raise ValueError("Expected all 2918 Train images, or four-event smoke subset")
        names = bundle["train_names"]
        if len(set(names)) != len(names) or set(names) & set(bundle.get("val_names", [])):
            raise ValueError("Train list contains duplicate images or overlaps Val")
        if not smoke and len(shards) != len(names):
            raise ValueError("Full Train list/query cache count disagreement")
        event_names = bundle["bank_cpu"]["event_names"]
        self.identity = artifact_identity(bundle)
        self.manifest_sha256 = digest(path)
        self.shards, self.fields = [], {}
        collected = {key: [] for key in ("distribution", "base_probability", "events", "images", "shard", "row")}
        image_ids, event_ids, valid_counts = [], [], []
        dimension = int(bundle["bank_cpu"]["tokens"].shape[-1])
        bundle["input_paths"]["hardquery_manifest"] = path
        bundle["inputs_before"]["hardquery_manifest"] = self.manifest_sha256
        for index, item in enumerate(shards):
            shard_path = safe_child(self.root, item["path"])
            checksum = digest(shard_path)
            if checksum != item["sha256"]:
                raise ValueError("Query shard hash mismatch: " + str(shard_path))
            # Every tensor in these locally produced torch.save archives is read-only.
            shard = torch.load(str(shard_path), map_location="cpu", weights_only=False, mmap=True)
            image_id, event_id = int(shard["image"]), int(shard["event"])
            if image_id < 0 or image_id >= len(names) or shard["name"] != names[image_id]:
                raise ValueError("Query image does not belong to the declared Train row")
            if not smoke and image_id != index:
                raise ValueError("Full query cache must retain ordered complete Train list")
            expected_event = event_names.index(event(names[image_id]))
            if event_id != expected_event:
                raise ValueError("Query event ID does not match Train/bank vocabulary")
            for key in ("name", "image", "event"):
                if item.get(key) != shard[key]:
                    raise ValueError("Manifest/shard identity mismatch: " + key)
            if image_id in image_ids:
                raise ValueError("Duplicate Train image in query cache")
            image_ids.append(image_id)
            event_ids.append(event_id)
            tokens = shard["tokens"]
            if tokens.dtype != torch.float32 or tokens.ndim != 2 or tokens.shape[-1] != dimension:
                raise ValueError("Queries require full-float frozen tokens with the bank dimension")
            count = len(tokens)
            if not torch.isfinite(tokens).all():
                raise ValueError("Nonfinite query tokens")
            for key in ("distribution", "base_probability"):
                if shard[key].shape != (count, 4) or shard[key].dtype != torch.float32:
                    raise ValueError("Invalid query field: " + key)
                if not torch.isfinite(shard[key]).all() or (shard[key] < 0).any():
                    raise ValueError("Nonfinite/negative query labels or probabilities")
            for key in ("weights", "mass"):
                values = shard[key]
                if values.shape != (count,) or not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid query field: " + key)
            regions = shard["region_ids"]
            if (regions.shape != (count,) or regions.is_floating_point() or (regions < 0).any()
                    or len(regions.unique()) != count):
                raise ValueError("Invalid or duplicate per-image region IDs")
            valid = (shard["weights"] > 0) & (shard["mass"] > 1e-6)
            rows = torch.where(valid)[0]
            distribution = validate_distribution(shard["distribution"][rows], "Train distribution")
            probability = validate_distribution(shard["base_probability"][rows], "frozen J1 probability")
            for key, value in (
                ("distribution", distribution), ("base_probability", probability),
                ("events", torch.full((len(rows),), event_id, dtype=torch.long)),
                ("images", torch.full((len(rows),), image_id, dtype=torch.long)),
                ("shard", torch.full((len(rows),), index, dtype=torch.long)), ("row", rows),
            ):
                collected[key].append(value)
            self.shards.append(shard)
            valid_counts.append(len(rows))
            ledger = f"hardquery_shard_{index:05d}"
            bundle["input_paths"][ledger] = shard_path
            bundle["inputs_before"][ledger] = checksum
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(shards):
                print(f"Verified Train query shards {index + 1}/{len(shards)}", flush=True)
        if smoke and len(set(event_ids)) != 4:
            raise ValueError("Smoke queries must cover exactly four distinct Train events")
        self.fields = {key: torch.cat(values) for key, values in collected.items()}
        if not len(self.fields["events"]):
            raise ValueError("Query cache has no valid Train regions")
        self.inventory = dict(images=len(shards), valid_regions=sum(valid_counts),
                              events=len(set(event_ids)), storage="read-only CPU memory-mapped shards")
        self.dim = dimension

    def batch(self, indices, device):
        indices = indices.cpu().long()
        shard_ids = self.fields["shard"][indices]
        tokens = torch.empty((len(indices), self.dim), dtype=torch.float32)
        for shard_id in shard_ids.unique().tolist():
            destination = torch.where(shard_ids == shard_id)[0]
            rows = self.fields["row"][indices[destination]]
            tokens[destination] = self.shards[shard_id]["tokens"][rows]
        return dict(tokens=tokens.to(device), **{
            key: self.fields[key][indices].to(device)
            for key in ("distribution", "base_probability", "events", "images")})


def reliable_support(bank, purity, device):
    distribution = validate_distribution(bank["distribution"], "bank distribution")
    eligible = distribution.max(-1).values >= purity
    if not eligible.any():
        raise ValueError("No reliable supports; the purity threshold will not be relaxed")
    selected = {key: bank[key][eligible].detach().to(device) for key in (
        "tokens", "distribution", "events", "images")}
    selected["tokens"] = selected["tokens"].float()
    selected["distribution"] = selected["distribution"].float()
    selected["classes"] = selected["distribution"].argmax(-1)
    coverage = {}
    for class_id in range(4):
        mask = selected["classes"] == class_id
        coverage[str(class_id)] = dict(count=int(mask.sum()),
                                      events=len(selected["events"][mask].unique()))
    return selected, coverage


def train_arm(args, arm, support, store, sampler, output, identity, coverage):
    device = support["tokens"].device
    # Constructor has an exact identity reset; every arm gets identical parameters.
    projector = RetrievalProjector(support["tokens"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed + 1)
    active_updates = used_anchors = zero_batches = no_positive = 0
    sampled_by_class = [0] * 4
    bank_labels = support["classes"].cpu()
    bank_events, bank_images = support["events"].cpu(), support["images"].cpu()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if arm == "bank":
            indices, sampling = balanced_anchor_indices(
                bank_labels, bank_events, bank_images, args.per_class, generator=generator)
            loss, statistics = cross_event_contrastive_loss(
                support["tokens"], support["classes"], support["events"], support["images"],
                projector=projector, temperature=args.temperature, anchor_indices=indices.to(device))
            sampled_classes = bank_labels[indices]
        else:
            indices, sampling = sampler.sample(args.per_class, generator, args.hard_fraction)
            query = store.batch(indices, device)
            loss, statistics = soft_cross_event_loss(
                query["tokens"], query["distribution"], query["events"], query["images"],
                support["tokens"], support["distribution"], support["events"], support["images"],
                projector=projector, temperature=args.temperature)
            sampled_classes = sampler.classes[indices]
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0, error_if_nonfinite=True)
        factor = .5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.step()
        active_updates += int(float(grad) > 0)
        used_anchors += statistics["used_anchor_count"]
        zero_batches += int(statistics["zero_loss"])
        no_positive += statistics["skipped_selected_no_positive"]
        class_counts = torch.bincount(sampled_classes, minlength=4).tolist()
        sampled_by_class = [a + b for a, b in zip(sampled_by_class, class_counts)]
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = dict(arm=arm, step=step, loss=float(loss.detach()), grad_norm=float(grad),
                       lr=args.lr * factor, sampling=sampling, objective=statistics)
            with (output / f"training_{arm}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    if not args.smoke and (active_updates == 0 or used_anchors == 0):
        raise RuntimeError(f"{arm}: no metric learning occurred; inspect Train support coverage")
    summary = dict(arm=arm, optimizer_updates=args.steps, nonzero_gradient_updates=active_updates,
                   used_anchors=used_anchors, zero_anchor_batches=zero_batches,
                   skipped_no_positive=no_positive, sampled_by_class=sampled_by_class)
    payload = dict(
        format=METRIC_FORMAT, arm=arm, complete=True, smoke=args.smoke,
        step=args.steps, dim=projector.dim, model=projector.state_dict(),
        train_only=True, test_accessed=False, arguments=vars(args),
        query_manifest_sha256=store.manifest_sha256, support_coverage=coverage,
        initialization="identity", selection="fixed-budget last; no Val selection",
        objective="legacy_cross_event_supcon" if arm == "bank" else "normalized_soft_overlap_cross_entropy",
        support_subset="parent bank purity >= threshold; inference bank unchanged",
        training_summary=summary, **identity)
    return payload, summary


def run(args):
    args.steps = (20 if args.smoke else 2000) if args.steps is None else args.steps
    if args.steps != (20 if args.smoke else 2000):
        raise ValueError("The matched budget is exactly 2000 updates (20 in smoke)")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Each arm may run only once")
    if (args.per_class < 2 or not 0 <= args.purity <= 1 or not 0 < args.hard_fraction < 1
            or not 0 < args.hard_quantile < 1):
        raise ValueError("Invalid purity, per-class batch size, or hard fraction")
    for name in ("temperature", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    bundle = load_verified_bundle(args)
    identity = artifact_identity(bundle)
    store = QueryStore(args.queries, bundle, args.smoke)
    support, coverage = reliable_support(bundle["bank_cpu"], args.purity, torch.device(args.device))
    if not args.smoke and any(row["events"] < 2 for row in coverage.values()):
        raise ValueError("Formal metric training requires each class in >=2 support events")
    sampler = HardQuerySampler(store.fields["distribution"], store.fields["base_probability"],
                               hard_quantile=args.hard_quantile)
    output = new_output(args.output)
    manifest = dict(format=METRIC_FORMAT, complete=False, smoke=args.smoke,
                    arguments=vars(args), query_manifest_sha256=store.manifest_sha256,
                    support_coverage=coverage, query_inventory=store.inventory,
                    query_strata=sampler.inventory, train_only=True, test_accessed=False,
                    fixed="front, Q, cached features, bank membership, support payload, source J1/J2/J3",
                    learned="identity-initialized 128x128 retrieval projector only",
                    runtime_settings=bundle["runtime_settings"], **identity)
    write_json(output / "manifest.json", manifest)
    try:
        payloads, results = {}, {}
        for arm in args.arms:
            payloads[arm], results[arm] = train_arm(
                args, arm, support, store, sampler, output, identity, coverage)
        checks = verify_unchanged(bundle)
        checkpoint_sha256 = {}
        for arm, payload in payloads.items():
            payload["immutable_checks"] = checks
            checkpoint_path = output / f"metric_{arm}_last.pth"
            save_pt(checkpoint_path, payload)
            checkpoint_sha256[arm] = digest(checkpoint_path)
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", dict(
            complete=True, arms=results, steps=args.steps, checkpoint_sha256=checkpoint_sha256,
            smoke=args.smoke, test_accessed=False,
            immutable_checks=checks, query_manifest_sha256=store.manifest_sha256,
            full_validation_evidence=False, **identity))
    except Exception as error:
        fail_summary(output, error)
        raise
    print("METRIC TRAINING COMPLETE. Both arms are fixed-budget last checkpoints; no Val selection.", flush=True)
    return results


if __name__ == "__main__":
    run(parser().parse_args())
