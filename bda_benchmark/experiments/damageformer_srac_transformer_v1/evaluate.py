"""Read-only Val450 evaluation and Memory interventions; Test is unsupported."""
from __future__ import annotations

import argparse

import torch

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
    source_hash,
)
from .retrieval import JointMemoryReader


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "dataset", "train_list", "val_list", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--a1_checkpoint")
    parser.add_argument("--cache")
    parser.add_argument("--head")
    parser.add_argument(
        "--intervention",
        choices=("normal", "random", "shuffle_labels", "off", "no_metadata"),
        default="normal",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _train_names, val_names = lists(args.train_list, args.val_list)
    device = torch.device(args.device)
    model, front = load_front(args.checkpoint, device, args.a1_checkpoint)
    if bool(front["smoke"]) != args.smoke:
        parser.error("Smoke/formal mismatch")
    if front["val_sha256"] != digest(args.val_list):
        parser.error("Val split changed")

    reader = bank = None
    mode = "memory"
    topk = 8
    retrieval_policy = "global"
    exclusion = "cross_event"
    if args.head:
        if not args.cache:
            parser.error("A reader checkpoint requires its matching Train cache")
        _root, cache, bank = read_cache(args.cache, args.checkpoint)
        checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != READER_FORMAT or checkpoint.get("source_hash") != source_hash():
            parser.error("Wrong reader source or format")
        if not checkpoint.get("run_complete") or checkpoint.get("completed_steps") != checkpoint["arguments"]["steps"]:
            parser.error("Reader training did not complete its declared budget")
        if checkpoint["checkpoint_sha256"] != digest(args.checkpoint) or checkpoint["bank_sha256"] != cache["bank_sha256"]:
            parser.error("Reader/front/bank mismatch")
        reader = JointMemoryReader(
            cache["config"]["dim"], len(cache["event_names"]), slots=checkpoint["topk"]
        ).to(device)
        reader.load_state_dict(checkpoint["model"], strict=True)
        bank = device_bank(bank, device)
        mode = checkpoint["mode"]
        topk = checkpoint["topk"]
        retrieval_policy = checkpoint["retrieval_policy"]
        exclusion = checkpoint["exclusion"]
        if mode == "nomemory" and args.intervention != "normal":
            parser.error("Memory interventions do not apply to the no-Memory reader")
    elif args.intervention != "normal":
        parser.error("Memory intervention requires a Memory reader")

    output = new_output(args.output)
    result = evaluate_front(
        model,
        loader(args.dataset, val_names[:2] if args.smoke else val_names, workers=args.workers),
        device,
        reader,
        bank,
        mode,
        topk,
        args.intervention,
        retrieval_policy=retrieval_policy,
        exclusion=exclusion,
    )
    result.update(
        complete=True,
        arguments=vars(args),
        checkpoint_sha256=digest(args.checkpoint),
        source_hash=source_hash(),
        scope="all valid GT pixels; GT never filters predictions or retrieval",
        retrieval_policy=retrieval_policy,
        exclusion=exclusion,
    )
    save_json(output / "summary.json", result)
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    main()

