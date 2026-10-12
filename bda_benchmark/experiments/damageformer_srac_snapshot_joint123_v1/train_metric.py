"""Fit only the retrieval metric on a completed snapshot Train bank.

Uses the original J3 objective and budget. The parent checkpoint does not need
to complete its declared frontend budget; that fact is explicitly recorded.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import (
    RetrievalProjector, balanced_anchor_indices, cross_event_contrastive_loss,
)
from .common import (
    METRIC_FORMAT, digest, load_front, new_output, read_cache,
    save_json, save_pt, seed_everything, source_hash,
)


def parser():
    p = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "a1_checkpoint", "cache", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--per_class", type=int, default=16)
    p.add_argument("--purity", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    return p


def run(args):
    if args.steps < 1 or args.per_class < 1 or not 0 <= args.purity <= 1 or args.lr <= 0:
        raise ValueError("Invalid metric-training parameters")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not args.smoke and args.steps != 2000:
        raise ValueError("Formal preregistered metric budget is 2000 updates")
    root, cache, bank = read_cache(args.cache, args.checkpoint)
    if bool(cache["smoke"]) != bool(args.smoke):
        raise ValueError("Do not mix smoke and formal bank artifacts")
    front, front_payload = load_front(args.checkpoint, "cpu", args.a1_checkpoint)
    front.eval().requires_grad_(False)
    if front_payload.get("smoke") is not False:
        raise ValueError("Snapshot metric requires a formal parent, even for metric smoke")
    before = module_fingerprint(front)
    seed_everything(args.seed)
    device = torch.device(args.device)
    distribution = bank["distribution"].float()
    purity, labels = distribution.max(-1)
    eligible = (purity >= args.purity) & (distribution.sum(-1) > 0)
    tokens = bank["tokens"][eligible].float().to(device)
    classes = labels[eligible].to(device)
    events = bank["events"][eligible].long().to(device)
    images = bank["images"][eligible].long().to(device)
    if not len(tokens):
        raise ValueError("No Train bank tokens satisfy metric purity; do not silently lower threshold")
    coverage = {str(c): dict(tokens=int((classes == c).sum()),
                            distinct_events=int(events[classes == c].unique().numel())) for c in range(4)}
    if not args.smoke and any(coverage[str(c)]["distinct_events"] < 2 for c in range(4)):
        raise ValueError("Cannot train cross-event metric for all four classes: " + str(coverage))
    output = new_output(args.output)
    projector = RetrievalProjector(tokens.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed + 1)
    manifest = dict(format=METRIC_FORMAT, arguments=vars(args), source_hash=source_hash(),
                    snapshot_protocol=True,
                    parent_run_complete=bool(front_payload.get("run_complete", False)),
                    parent_checkpoint_step=front_payload["step"],
                    declared_parent_steps=front_payload["arguments"]["steps"],
                    front_sha256=digest(args.checkpoint), bank_sha256=cache["bank_sha256"],
                    cache_manifest_sha256=digest(root / "manifest.json"), dim=tokens.shape[-1],
                    coverage=coverage, train_only=True, test_accessed=False, smoke=args.smoke,
                    source_fingerprint_before=before,
                    source_weights_and_buffers_unchanged=None,
                    fixed="front/Q/all tokens/full candidate bank/support payload/reader",
                    learned="identity-initialized search projection ONLY",
                    selection="fixed-budget last; no Val checkpoint or hyperparameter selection",
                    additional_optimizer_updates=args.steps)
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    total_active = 0
    for step in range(1, args.steps + 1):
        indices, sample_stats = balanced_anchor_indices(
            classes.cpu(), events.cpu(), images.cpu(), args.per_class, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, statistics = cross_event_contrastive_loss(
            tokens, classes, events, images, projector=projector,
            temperature=args.temperature, anchor_indices=indices.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite metric loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0, error_if_nonfinite=True)
        factor = 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.step()
        total_active += int(float(grad) > 0)
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad),
                       lr=args.lr * factor, sampling=sample_stats, contrastive=statistics)
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    if not args.smoke and total_active == 0:
        raise RuntimeError("No metric learning occurred; inspect cross-event support")
    after = module_fingerprint(front)
    unchanged = before == after
    manifest.update(source_fingerprint_after=after, source_weights_and_buffers_unchanged=unchanged)
    save_json(output / "manifest.json", manifest)
    if not unchanged:
        raise RuntimeError("Frozen snapshot weights/buffers changed; metric remains incomplete")
    save_pt(output / "metric_last.pth", dict(**manifest, complete=True, step=args.steps,
                                             model=projector.state_dict()))
    save_json(output / "summary.json", dict(complete=True, steps=args.steps, coverage=coverage,
                                             nonzero_gradient_updates=total_active,
                                             source_weights_and_buffers_unchanged=unchanged,
                                             smoke=args.smoke, test_accessed=False))
    print("SNAPSHOT METRIC FIT COMPLETE. Bank unchanged. No Val/Test fitted or selected.", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
