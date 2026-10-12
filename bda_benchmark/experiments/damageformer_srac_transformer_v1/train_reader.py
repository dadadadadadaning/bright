"""Stage C: matched direct no-Memory/Memory Transformer readers on frozen tokens."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .build_neighbors import gather_support, open_table
from .common import (
    READER_FORMAT,
    device_bank,
    digest,
    evaluate_front,
    lists,
    load_front,
    loader,
    new_output,
    read_cache,
    save_json,
    save_pt,
    seed_everything,
    source_hash,
)
from .retrieval import JointMemoryReader


def _sample_query(shard, batch, generator, policy):
    eligible = (shard["weights"] > 0).nonzero().flatten()
    if not len(eligible):
        return None
    if policy == "uniform":
        return eligible[torch.randint(len(eligible), (batch,), generator=generator)]
    if policy != "class_balanced":
        raise ValueError(policy)
    dominant = shard["distribution"].argmax(-1)
    chosen = []
    for _ in range(batch):
        class_index = int(torch.randint(4, (1,), generator=generator))
        candidates = eligible[dominant[eligible] == class_index]
        if not len(candidates):
            candidates = eligible
        chosen.append(candidates[torch.randint(len(candidates), (), generator=generator)])
    return torch.stack(chosen).long()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "cache", "dataset", "train_list", "val_list", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--a1_checkpoint")
    parser.add_argument("--mode", choices=("memory", "nomemory"), required=True)
    parser.add_argument("--neighbors")
    parser.add_argument("--retrieval_policy", choices=("global", "per_class"), default="global")
    parser.add_argument("--exclusion", choices=("same_image", "cross_event"), default="cross_event")
    parser.add_argument("--query_sampling", choices=("uniform", "class_balanced"), default="uniform")
    for name, default in dict(steps=5000, eval_every=500, batch=128, topk=8, workers=4, seed=3407).items():
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.batch, args.topk) < 1:
        parser.error("Positive sizes required")
    if not args.smoke and args.steps != 5000:
        parser.error("Formal reader comparison requires exactly 5000 updates")
    if args.retrieval_policy == "per_class" and args.topk % 4:
        parser.error("per_class topk must be divisible by four")
    if args.mode == "memory" and not args.neighbors:
        parser.error("Memory mode requires an immutable neighbor table")
    train_names, val_names = lists(args.train_list, args.val_list)
    cache_root, cache, bank_cpu = read_cache(args.cache, args.checkpoint)
    if cache["train_sha256"] != digest(args.train_list) or cache["val_sha256"] != digest(args.val_list):
        parser.error("Split hash mismatch")
    if bool(cache["smoke"]) != args.smoke:
        parser.error("Smoke/formal cache mismatch")
    for record in cache["shards"]:
        if digest(cache_root / record["path"]) != record["sha256"]:
            raise ValueError("Shard hash mismatch: " + record["path"])

    table = table_manifest = None
    if args.mode == "memory":
        table, table_manifest = open_table(
            args.neighbors,
            cache_root,
            cache["bank_sha256"],
            args.topk,
            args.retrieval_policy,
            args.exclusion,
        )

    device = torch.device(args.device)
    seed_everything(args.seed)
    model, front = load_front(args.checkpoint, device, args.a1_checkpoint)
    model.eval().requires_grad_(False)
    bank = device_bank(bank_cpu, device)
    reader = JointMemoryReader(cache["config"]["dim"], len(cache["event_names"]), slots=args.topk).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=5e-3)
    generator = torch.Generator().manual_seed(args.seed + 1)
    validation_data = loader(
        args.dataset,
        val_names[:2] if args.smoke else val_names,
        workers=args.workers,
        seed=args.seed + 2,
    )
    output = new_output(args.output)
    manifest = dict(
        arguments=vars(args),
        format=READER_FORMAT,
        config=cache["config"],
        source_hash=source_hash(),
        checkpoint_sha256=digest(args.checkpoint),
        bank_sha256=cache["bank_sha256"],
        mode=args.mode,
        topk=args.topk,
        retrieval_policy=args.retrieval_policy,
        exclusion=args.exclusion,
        query_sampling=args.query_sampling,
        neighbor_table_sha256=None if table_manifest is None else table_manifest["table_sha256"],
        classifier="Pre-LN Cross-Attention + FFN direct four-class region logits; no A1-logit residual and no query-as-support fallback",
        no_memory_control="same reader parameters and same Top-R slot count; learned supports replace historical regions",
        trainable_parameters=sum(parameter.numel() for parameter in reader.parameters()),
        loss="area-weighted Train region soft-label CE",
        test_accessed=False,
        smoke=args.smoke,
    )
    save_json(output / "manifest.json", manifest)
    best = -1.0
    for step in range(1, args.steps + 1):
        for _attempt in range(200):
            shard_index = int(torch.randint(len(cache["shards"]), (1,), generator=generator))
            shard = torch.load(cache_root / cache["shards"][shard_index]["path"], map_location="cpu", weights_only=False)
            ids = _sample_query(shard, args.batch, generator, args.query_sampling)
            if ids is not None:
                break
        else:
            raise RuntimeError("Could not sample a labeled Train region")
        query = shard["tokens"][ids].float().to(device)
        target = shard["distribution"][ids].float().to(device)
        weights = shard["weights"][ids].float().to(device)
        reader.train()
        if args.mode == "memory":
            offset = table_manifest["offsets"][shard_index]["start"]
            selected = torch.from_numpy(table[offset + ids.numpy()].copy()).long().to(device)
            support, distribution, valid, metadata = gather_support(selected, bank)
            logits = reader(query, support, distribution, valid, mode="memory", metadata=metadata)
            coverage = float(valid.any(1).float().mean())
        else:
            logits = reader(query, mode="nomemory")
            coverage = None
        loss = (-(target * F.log_softmax(logits.float(), dim=-1)).sum(-1) * weights).sum()
        loss = loss / weights.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite reader loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0, error_if_nonfinite=True)
        factor = 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.step()
        if step == 1 or step % 20 == 0:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                support_coverage=coverage,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=float(gradient_norm),
            )
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_front(
                model,
                validation_data,
                device,
                reader,
                bank,
                args.mode,
                args.topk,
                retrieval_policy=args.retrieval_policy,
                exclusion=args.exclusion,
            )
            score = result["metrics"]["final"]["valid_miou"]
            with (output / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(step=step, **result)) + "\n")
            state = dict(
                model=reader.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                valid_miou=score,
                **manifest,
            )
            if score > best:
                best = score
                save_pt(output / "best.pth", state)
            if step == args.steps:
                save_pt(output / "last.pth", state)
            print(f"VAL {args.mode} step={step} foreground_mIoU={score:.6f}", flush=True)
    for checkpoint_name in ("best.pth", "last.pth"):
        checkpoint = torch.load(output / checkpoint_name, map_location="cpu", weights_only=False)
        checkpoint.update(run_complete=True, completed_steps=args.steps)
        save_pt(output / checkpoint_name, checkpoint)
    save_json(output / "summary.json", dict(complete=True, best_valid_miou=best, test_accessed=False, smoke=args.smoke))


if __name__ == "__main__":
    main()

