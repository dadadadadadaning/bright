from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .verifier import SingleEvidenceVerifier, standardized_inputs


PROTOCOL = "SPI_S2_SINGLE_EVIDENCE_VERIFIER_TRAIN_ONLY_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def choose_train_threshold(model, inputs, labels, mean, std, device, batch_size=65536):
    scores = []
    for start in range(0, inputs.shape[0], batch_size):
        batch = ((inputs[start:start + batch_size].to(device) - mean) / std)
        scores.append(torch.sigmoid(model(batch)).cpu())
    scores = torch.cat(scores)
    best = None
    for threshold in torch.linspace(0.05, 0.95, 91):
        prediction = scores >= threshold
        positive = labels == 1
        negative = labels == 0
        tpr = float((prediction & positive).sum()) / max(int(positive.sum()), 1)
        tnr = float((~prediction & negative).sum()) / max(int(negative.sum()), 1)
        row = {
            "threshold": float(threshold),
            "balanced_accuracy": 0.5 * (tpr + tnr),
            "positive_recall": tpr,
            "negative_recall": tnr,
        }
        if best is None or (row["balanced_accuracy"], -abs(row["threshold"] - 0.5)) > (
            best["balanced_accuracy"], -abs(best["threshold"] - 0.5)
        ):
            best = row
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train one admitted Evidence Verifier V2")
    parser.add_argument("--group", choices=("relation", "matching", "slic"), required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    admission = json.loads(Path(args.admission_json).read_text(encoding="utf-8"))
    if admission.get("protocol") != "SPI_S2_PARAMETER_FREE_EVIDENCE_AUDIT_v2":
        raise RuntimeError("unexpected admission protocol")
    if args.group not in admission.get("admitted_groups", ()):
        raise RuntimeError(f"{args.group} did not pass parameter-free admission")
    cache_sha = sha256_file(args.train_cache)
    if cache_sha != admission.get("train_cache_sha256"):
        raise RuntimeError("train cache does not match admitted audit")
    cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    if cache.get("split") != "train" or cache.get("validation_accessed", True):
        raise RuntimeError("verifier training requires a Train-only cache")
    labels = cache["selector_label"].long()
    valid = labels >= 0
    labels = labels[valid]
    inputs = standardized_inputs(cache, args.group)[valid]
    mean = inputs.mean(0).to(device)
    std = inputs.std(0, unbiased=False).clamp_min(1e-5).to(device)
    model = SingleEvidenceVerifier(inputs.shape[1], args.hidden).to(device)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    pos_weight = torch.tensor(negative / max(positive, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "verifier_last.pth"
    step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("group") != args.group or saved.get("train_cache_sha256") != cache_sha:
            raise RuntimeError("existing verifier checkpoint provenance mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        step = int(saved["optimizer_step"])
        if step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        else:
            if not saved.get("threshold_selection"):
                model.eval()
                saved["threshold_selection"] = choose_train_threshold(
                    model, inputs, labels, mean, std, device
                )
                saved["normalizer_mean"] = mean.cpu()
                saved["normalizer_std"] = std.cpu()
                atomic_save(saved, checkpoint_path)
                print(
                    "Recovered missing Train-selected threshold: "
                    f"{saved['threshold_selection']}",
                    flush=True,
                )
            print(f"Verifier already complete at step={step}", flush=True)
            return 0

    dataset = TensorDataset(inputs, labels.float())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        drop_last=False,
    )
    print(
        f"Evidence Verifier V2 | group={args.group} | samples={len(dataset)} | "
        f"inputs={inputs.shape[1]} | positive/negative={positive}/{negative}",
        flush=True,
    )
    model.train()
    while step < args.optimizer_steps:
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            normalized = (batch_inputs - mean) / std
            loss = loss_fn(model(normalized), batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite verifier loss at step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "optimizer_step": step,
                            "loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(gradient_norm),
                        }
                    ),
                    flush=True,
                )
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save(
                    {
                        "protocol": PROTOCOL,
                        "group": args.group,
                        "optimizer_step": step,
                        "input_channels": int(inputs.shape[1]),
                        "hidden": args.hidden,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "normalizer_mean": mean.cpu(),
                        "normalizer_std": std.cpu(),
                        "train_cache_sha256": cache_sha,
                        "admission_json": str(Path(args.admission_json).resolve()),
                        "admission_json_sha256": sha256_file(args.admission_json),
                        "threshold_selection": None,
                        "validation_accessed": False,
                        "official_instances_accessed": False,
                        "test_accessed": False,
                    },
                    checkpoint_path,
                )
            if step >= args.optimizer_steps:
                break

    model.eval()
    threshold = choose_train_threshold(model, inputs, labels, mean, std, device)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved["threshold_selection"] = threshold
    atomic_save(saved, checkpoint_path)
    print(f"Train-selected threshold: {threshold}", flush=True)
    print(f"Saved: {checkpoint_path}", flush=True)
    print("Validation/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
