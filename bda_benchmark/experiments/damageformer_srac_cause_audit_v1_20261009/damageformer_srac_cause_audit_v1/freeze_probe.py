"""Optional paired fusion-adaptation probe with fixed predicted B and soft Q.

F trains copied adapter/embedding/classifier modules. U starts from identical
copies and additionally trains a copy of A1 fusion_layer_3. Both reuse one frozen
source forward per batch, including its RGB/SAR features, ROI, B and Q. This
tests fusion adaptation only; neither arm unfreezes the Swin encoders.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import validate_batch_labels
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities


FORMAT = "SRAC_CAUSE_FIXED_Q_FUSION_PROBE_V1"
READOUT_NAMES = ("adapter", "embedding", "region_classifier")


@dataclass(frozen=True)
class FixedBatch:
    rgb_feature: torch.Tensor
    sar_feature: torch.Tensor
    roi_weight: torch.Tensor
    building: torch.Tensor
    q: torch.Tensor
    ids: torch.Tensor
    shape: tuple
    count: int
    boundary: torch.Tensor
    entropy: torch.Tensor
    output_size: tuple
    normal_probability: torch.Tensor
    a1_logits: torch.Tensor


class FrozenSource:
    """Capture A1 auxiliaries during exactly one ordinary parent forward."""

    def __init__(self, model):
        self.model = model.requires_grad_(False).eval()

    def __call__(self, pre, post):
        captured = []

        def capture(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Parent must call A1 with return_aux=True")
            captured.append(output[2])

        self.model.eval()
        hook = self.model.base.register_forward_hook(capture)
        # Explicitly exit any enclosing inference_mode: downstream trainable
        # convolutions must be allowed to save these frozen tensors for backward.
        try:
            with torch.inference_mode(False), torch.no_grad():
                result = self.model(pre, post)
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one A1 forward per source batch")
                aux = captured[0]
                owned = lambda value: value.detach().clone()
                fixed = FixedBatch(
                    rgb_feature=owned(aux["rgb_feature"]),
                    sar_feature=owned(aux["sar_feature"]),
                    roi_weight=owned(aux["roi_weight"]),
                    building=owned(aux["p_building"]),
                    q=owned(result["q"]),
                    ids=owned(result["ids"]),
                    shape=tuple(result["shape"]),
                    count=int(result["count"]),
                    boundary=owned(result["soft_boundary"]),
                    entropy=owned(result["entropy_map"]),
                    output_size=tuple(pre.shape[-2:]),
                    normal_probability=owned(result["prob"]),
                    a1_logits=owned(result["old_pixel"]),
                )
        finally:
            hook.remove()
        return fixed


class FusionReadoutArm(nn.Module):
    """Independent copies only; never registers or updates the source model."""

    def __init__(self, source, train_fusion: bool):
        super().__init__()
        self.train_fusion = bool(train_fusion)
        self.fusion_layer_3 = copy.deepcopy(source.base.fusion_layer_3)
        self.fusion_layer_3.requires_grad_(self.train_fusion)
        for name in READOUT_NAMES:
            setattr(self, name, copy.deepcopy(getattr(source, name)).requires_grad_(True))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Keep the entire fusion block in eval in BOTH arms. Gradients still
        # update U's convolution and BN affine parameters; running stats never
        # update. No dropout or BN-mode difference can explain a paired effect.
        self.fusion_layer_3.eval()
        return self

    def forward(self, fixed: FixedBatch):
        raw = self.fusion_layer_3(torch.cat((fixed.rgb_feature, fixed.sar_feature), dim=1))
        raw = raw * fixed.roi_weight
        adapted, _ = self.adapter(raw, fixed.building, fixed.boundary, fixed.entropy)
        tokens, mass, geometry, valid = self.embedding(
            adapted, raw, fixed.q, fixed.ids, fixed.shape, fixed.count
        )
        logits = self.region_classifier(tokens)
        probability = project_region_probabilities(
            logits, fixed.q, fixed.ids, fixed.shape, fixed.output_size
        )
        return dict(prob=probability, region_logits=logits, tokens=tokens,
                    mass=mass, geometry=geometry, valid_regions=valid, q=fixed.q)

    def readout_parameters(self):
        return [parameter for name in READOUT_NAMES for parameter in getattr(self, name).parameters()]


def independent_parameter_contract(source, arms):
    source_ids = {id(parameter) for parameter in source.parameters()}
    source_ptrs = {parameter.data_ptr() for parameter in source.parameters()}
    seen_ids, seen_ptrs = set(source_ids), set(source_ptrs)
    for name, arm in arms.items():
        ids = {id(parameter) for parameter in arm.parameters()}
        ptrs = {parameter.data_ptr() for parameter in arm.parameters()}
        if ids & seen_ids or ptrs & seen_ptrs:
            raise RuntimeError(f"{name} shares parameter storage with source or other arm")
        seen_ids.update(ids)
        seen_ptrs.update(ptrs)
    if any(parameter.requires_grad for parameter in source.parameters()):
        raise RuntimeError("Every source parameter, including decoder/partition, must be frozen")
    return dict(parameter_objects_disjoint=True, parameter_storage_disjoint=True,
                source_all_frozen=True)


def bn_buffers(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.named_buffers()}


def buffers_unchanged(module, before):
    after = dict(module.named_buffers())
    return set(after) == set(before) and all(
        torch.equal(after[name].detach().cpu(), value) for name, value in before.items()
    )


def gradient_l1(module):
    return sum(float(parameter.grad.detach().abs().sum())
               for parameter in module.parameters() if parameter.grad is not None)


def clip_gradients(arm, max_norm=1.0):
    """Match readout clipping across arms; clip U's extra fusion separately."""
    norms = dict(readout=float(torch.nn.utils.clip_grad_norm_(
        arm.readout_parameters(), max_norm, error_if_nonfinite=True
    )), fusion=None)
    if arm.train_fusion:
        norms["fusion"] = float(torch.nn.utils.clip_grad_norm_(
            arm.fusion_layer_3.parameters(), max_norm, error_if_nonfinite=True
        ))
    return norms


def cosine_warmup(step, steps):
    warmup = min(1000, max(1, steps // 10))
    return step / warmup if step <= warmup else 0.5 * (
        1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))
    )


def objective(probability, target):
    # Import lazily so small synthetic tests do not construct the training app.
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

    if not (target != 255).any():
        return probability.sum() * 0
    return region_nll(probability, target) + lovasz_softmax(
        probability, target, classes="present", ignore=255
    )


@torch.no_grad()
def evaluate(source, arms, data, device, check_initial=False):
    labels = ("normal", "F", "U", "a1_pixel")
    confusions = {name: torch.zeros(4, 4, dtype=torch.long) for name in labels}
    events = {}
    errors = {"F_vs_normal": 0.0, "U_vs_normal": 0.0, "F_vs_U": 0.0}
    images = 0
    for arm in arms.values():
        arm.eval()
    for batch in data:
        pre, post, _loc, gt, names = common.parse_batch(batch, device)
        fixed = source(pre, post)
        probabilities = {name: arm(fixed)["prob"] for name, arm in arms.items()}
        if check_initial:
            for comparison, a, b in (
                ("F_vs_normal", probabilities["F"], fixed.normal_probability),
                ("U_vs_normal", probabilities["U"], fixed.normal_probability),
                ("F_vs_U", probabilities["F"], probabilities["U"]),
            ):
                error = float((a - b).abs().max())
                errors[comparison] = max(errors[comparison], error)
                if not torch.isfinite(a).all() or error > 1e-6:
                    raise RuntimeError(f"Initial paired-output mismatch: {comparison}={error}")
        predictions = {name: probability.argmax(1) for name, probability in probabilities.items()}
        predictions.update(normal=fixed.normal_probability.argmax(1), a1_pixel=fixed.a1_logits.argmax(1))
        for name, prediction in predictions.items():
            common.confusion_update(confusions[name], gt, prediction)
            for index, image_name in enumerate(names):
                event_id = common.event(image_name)
                events.setdefault(event_id, {key: torch.zeros(4, 4, dtype=torch.long) for key in labels})
                common.confusion_update(events[event_id][name], gt[index], prediction[index])
        images += len(pre)
        if images % 25 == 0:
            print(f"freeze_probe validation {images}", flush=True)
    return dict(metrics={name: common.metrics(cm) for name, cm in confusions.items()},
                per_event={event: {name: common.metrics(cm) for name, cm in by_arm.items()}
                           for event, by_arm in events.items()},
                image_count=images, initial_max_probability_error=errors if check_initial else None,
                class_order=["Background", "Intact", "Damaged", "Destroyed"], test_accessed=False)


def reproduction_gate(initial, payload, smoke):
    expected = float(payload["valid_miou"])
    observed = initial["metrics"]["normal"]["valid_miou"]
    a1_expected = float(payload["load_report"]["expected_valid_miou"])
    a1_observed = initial["metrics"]["a1_pixel"]["valid_miou"]
    if any(value is None or not math.isfinite(value)
           for value in (expected, observed, a1_expected, a1_observed)):
        raise ValueError("Checkpoint, normal and A1 evaluation must contain finite valid_miou")
    full_val = initial["image_count"] == 450
    normal_pass = abs(observed - expected) <= 1e-5 if full_val else None
    a1_pass = abs(a1_observed - a1_expected) <= 1e-5 if full_val else None
    passed = normal_pass and a1_pass if full_val else None
    if not smoke and not full_val:
        raise RuntimeError("Formal fusion probe must verify all 450 Val images before training")
    return dict(expected_valid_miou=expected, observed_valid_miou=observed,
                tolerance=1e-5, full_val=full_val, reproduction_pass=passed,
                normal_reproduction_pass=normal_pass,
                a1_expected_valid_miou=a1_expected, a1_observed_valid_miou=a1_observed,
                a1_reproduction_pass=a1_pass,
                smoke_only_unverified=bool(smoke and not full_val))


def arm_checkpoint(arm, arm_name, step, score, manifest):
    names = READOUT_NAMES + (("fusion_layer_3",) if arm.train_fusion else ())
    state = {name: {key: value.detach().cpu().clone()
                    for key, value in getattr(arm, name).state_dict().items()}
             for name in names}
    return dict(format=FORMAT, arm=arm_name, step=step, valid_miou=score,
                modules=state, source_checkpoint_sha256=manifest["source_checkpoint_sha256"],
                a1_sha256=manifest["a1_sha256"], source_checkpoint=manifest["source_checkpoint"],
                a1_checkpoint=manifest["a1_checkpoint"], run_complete=False,
                smoke=manifest["smoke"], test_accessed=False,
                interpretation="fixed-Q fusion adaptation only; Swin encoders remain frozen")


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def parser():
    result = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        result.add_argument("--" + name, required=True)
    for name, default in dict(steps=3000, eval_every=500, batch=2, accum=8,
                              workers=4, seed=3407, crop=512).items():
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--lr_readout", type=float, default=1e-4)
    result.add_argument("--lr_fusion", type=float, default=1e-5)
    result.add_argument("--weight_decay", type=float, default=5e-3)
    result.add_argument("--amp", choices=("none", "bf16"), default="none")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--smoke", action="store_true", help="20 updates; Val first 2; not formal evidence")
    return result


def main():
    run_started = time.perf_counter()
    args = parser().parse_args()
    if args.smoke:
        args.steps, args.eval_every = 20, 20
    if min(args.steps, args.eval_every, args.batch, args.accum, args.crop) < 1:
        raise SystemExit("Step counts and dimensions must be positive")
    if args.workers < 0 or args.lr_readout <= 0 or args.lr_fusion <= 0 or args.weight_decay < 0:
        raise SystemExit("Invalid workers, learning rate or weight decay")
    if args.batch * args.accum != 16:
        raise SystemExit("Both arms require effective batch 16")
    requested_output = Path(args.output)
    # A smoke command cannot consume the formal destination accidentally.
    output_path = requested_output.with_name(requested_output.name + "_smoke") if args.smoke else requested_output
    if output_path.exists():
        raise SystemExit("Output exists; preserve it and choose a new directory")
    if not (Path(args.checkpoint).parent / "snapshot.json").is_file():
        raise SystemExit("Freeze probe requires a stable diagnostic snapshot with adjacent snapshot.json; run the snapshot stage first")
    train_names, val_names = common.lists(args.train_list, args.val_list)
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise SystemExit("BF16 is unavailable on this device")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    from .support import diagnostic_source_hash, load_snapshot, module_fingerprint, verify_splits

    common.seed_everything(args.seed)
    checkpoint_sha256 = common.digest(args.checkpoint)
    model, payload = load_snapshot(args.checkpoint, args.a1_checkpoint, device)
    if common.digest(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed while loading; use a stable diagnostic snapshot")
    if not args.smoke and payload.get("smoke"):
        raise RuntimeError("A smoke source cannot seed a formal paired fusion probe")
    if type(payload.get("valid_miou")) not in (int, float) or not math.isfinite(payload["valid_miou"]):
        raise ValueError("Source checkpoint must record a finite valid_miou for the step-0 gate")
    a1_expected = payload.get("load_report", {}).get("expected_valid_miou")
    if type(a1_expected) not in (int, float) or not math.isfinite(a1_expected):
        raise ValueError("Source checkpoint must record A1 expected_valid_miou for the step-0 gate")
    verify_splits(payload, args.train_list, args.val_list)
    source = FrozenSource(model)
    arms = {"F": FusionReadoutArm(model, False), "U": FusionReadoutArm(model, True)}
    contract = independent_parameter_contract(model, arms)
    source_before = module_fingerprint(model)
    before = {name: {part: module_fingerprint(getattr(arm, part))
                     for part in READOUT_NAMES + ("fusion_layer_3",)}
              for name, arm in arms.items()}
    if before["F"] != before["U"]:
        raise RuntimeError("Arms must start with identical copied weights and buffers")
    buffer_before = {name: bn_buffers(arm.fusion_layer_3) for name, arm in arms.items()}
    optimizer = {}
    for name, arm in arms.items():
        groups = [dict(params=arm.readout_parameters(), lr=args.lr_readout,
                       initial_lr=args.lr_readout, role="readout")]
        if arm.train_fusion:
            groups.append(dict(params=list(arm.fusion_layer_3.parameters()), lr=args.lr_fusion,
                               initial_lr=args.lr_fusion, role="fusion"))
        optimizer[name] = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    train_data = common.loader(args.dataset, train_names, train=True, batch=args.batch,
                               workers=args.workers, crop=args.crop, seed=args.seed + 2)
    val_data = common.loader(args.dataset, val_names[:2] if args.smoke else val_names,
                             workers=args.workers, seed=args.seed + 3)
    output = common.new_output(output_path)
    manifest = dict(format=FORMAT, arguments=vars(args), output=str(output.resolve()),
                    source_checkpoint=str(Path(args.checkpoint).resolve()),
                    source_checkpoint_sha256=checkpoint_sha256,
                    a1_checkpoint=str(Path(args.a1_checkpoint).resolve()),
                    a1_sha256=common.digest(args.a1_checkpoint),
                    probe_code_sha256=common.digest(__file__),
                    diagnostic_source_hash=diagnostic_source_hash(),
                    parent_source_hash=payload.get("source_hash"),
                    train_sha256=common.digest(args.train_list), val_sha256=common.digest(args.val_list),
                    source_model_fingerprint=source_before, initial_module_fingerprints=before,
                    independent_parameter_contract=contract, train_source_count=len(train_names),
                    validation_count=2 if args.smoke else 450, effective_batch=args.batch * args.accum,
                    steps_are="optimizer updates per arm; one shared batch and crop stream",
                    expected_crops_per_arm=args.steps * args.batch * args.accum,
                    trainable_parameters={name: sum(p.numel() for p in arm.parameters() if p.requires_grad)
                                          for name, arm in arms.items()},
                    trainable_names={name: [key for key, p in arm.named_parameters() if p.requires_grad]
                                     for name, arm in arms.items()},
                    scope="fusion_layer_3 adaptation only; both Swin encoders and all source weights stay frozen",
                    fixed="one shared frozen source provides RGB/SAR, predicted B, roi_weight, soft Q, boundary, entropy",
                    batchnorm="fusion eval in both arms; U trains affine weights only, running buffers fixed",
                    loss="existing region_nll + Lovasz-Softmax, unweighted; no auxiliary loss or Memory",
                    lr_schedule="identical cosine/warmup factor for both arms; separately recorded fusion rate",
                    gradient_clipping="L2 max_norm=1.0 on readouts independently in each arm; separate L2 max_norm=1.0 on U fusion; no combined readout/fusion norm",
                    checkpoint_storage="copied readouts and U fusion only; source and A1 referenced by SHA256",
                    smoke=args.smoke, test_accessed=False, complete=False)
    common.save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    initial = evaluate(source, arms, val_data, device, check_initial=True)
    gate = reproduction_gate(initial, payload, args.smoke)
    common.save_json(output / "step0.json", dict(**initial, **gate))
    if gate["reproduction_pass"] is False:
        raise RuntimeError("Normal or A1 checkpoint reproduction exceeds 1e-5; no optimizer update performed")

    initial_audit_seconds = time.perf_counter() - run_started
    training_started = time.perf_counter()
    iterator = iter(train_data)
    best = {name: -1.0 for name in arms}
    gradient_proof = None
    crop_hash = hashlib.sha256()
    observed_crops = 0
    for step in range(1, args.steps + 1):
        for name, arm in arms.items():
            arm.train()
            optimizer[name].zero_grad(set_to_none=True)
            for group in optimizer[name].param_groups:
                group["lr"] = group["initial_lr"] * cosine_warmup(step, args.steps)
        logged_loss = {name: 0.0 for name in arms}
        for microstep in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_data)
                batch = next(iterator)
            label_report = validate_batch_labels(batch, step, microstep)
            if not label_report["valid"]:
                common.save_json(output / "invalid_labels.json", label_report)
                raise ValueError("Invalid labels; no remapping is permitted")
            pre, post, _loc, target, names = common.parse_batch(batch, device)
            observed_crops += len(pre)
            crop_hash.update(json.dumps([step, microstep, names]).encode())
            # Use FP32 for the frozen source, matching original evaluation even
            # when optional BF16 is used for the trainable copies.
            fixed = source(pre, post)
            q_before = fixed.q.clone()
            first = step == 1 and microstep == 0
            f_grad_before_u = None
            for name, arm in arms.items():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                    result = arm(fixed)
                    loss = objective(result["prob"], target)
                    scaled_loss = loss / args.accum
                if not torch.isfinite(scaled_loss):
                    raise RuntimeError(f"Non-finite loss in arm {name}")
                scaled_loss.backward()
                logged_loss[name] += float(loss.detach()) / args.accum
                if first and name == "F":
                    if any(p.grad is not None for p in arms["U"].parameters()):
                        raise RuntimeError("F backward populated U gradients")
                    f_grad_before_u = {key: p.grad.detach().clone() for key, p in arm.named_parameters()
                                       if p.grad is not None}
                elif first and name == "U":
                    unchanged = all(torch.equal(dict(arms["F"].named_parameters())[key].grad, grad)
                                    for key, grad in f_grad_before_u.items())
                    if not unchanged:
                        raise RuntimeError("U backward changed F gradients")
                    gradient_proof = dict(F_backward_left_U_gradients_empty=True,
                                          U_backward_left_F_gradients_identical=unchanged,
                                          source_gradients_empty=all(p.grad is None for p in model.parameters()),
                                          fusion_gradient_l1={key: gradient_l1(value.fusion_layer_3)
                                                              for key, value in arms.items()},
                                          readout_gradient_l1={key: {part: gradient_l1(getattr(value, part))
                                                                     for part in READOUT_NAMES}
                                                               for key, value in arms.items()})
                    common.save_json(output / "independent_gradients.json", gradient_proof)
            if not torch.equal(fixed.q, q_before):
                raise RuntimeError("Shared fixed Q changed during paired backward")
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen source accumulated a gradient")
        norms = {}
        for name, arm in arms.items():
            norms[name] = clip_gradients(arm)
            optimizer[name].step()
            if not buffers_unchanged(arm.fusion_layer_3, buffer_before[name]):
                raise RuntimeError(f"{name} fusion running buffers changed")
        row = dict(step=step, crops_per_arm=observed_crops, loss=logged_loss, grad_norm=norms,
                   lr={name: {group["role"]: group["lr"] for group in opt.param_groups}
                       for name, opt in optimizer.items()}, shared_batch_stream=True)
        append_json(output / "training.jsonl", row)
        if step == 1 or step % 20 == 0:
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(source, arms, val_data, device)
            append_json(output / "validation.jsonl", dict(step=step, crops_per_arm=observed_crops, **result))
            for name, arm in arms.items():
                score = result["metrics"][name]["valid_miou"]
                state = arm_checkpoint(arm, name, step, score, manifest)
                if score > best[name]:
                    best[name] = score
                    common.save_pt(output / f"best_{name}.pth", state)
                if step == args.steps:
                    common.save_pt(output / f"last_{name}.pth", state)
            print(json.dumps(dict(step=step, valid_miou={name: result["metrics"][name]["valid_miou"]
                                                        for name in ("normal", "F", "U")})), flush=True)

    if module_fingerprint(model) != source_before:
        raise RuntimeError("Frozen source parameter or buffer changed")
    after = {name: {part: module_fingerprint(getattr(arm, part))
                    for part in READOUT_NAMES + ("fusion_layer_3",)}
             for name, arm in arms.items()}
    if before["F"]["fusion_layer_3"] != after["F"]["fusion_layer_3"]:
        raise RuntimeError("F fusion weights changed")
    for name in arms:
        for checkpoint_name in (f"best_{name}.pth", f"last_{name}.pth"):
            state = torch.load(output / checkpoint_name, map_location="cpu", weights_only=False)
            state.update(run_complete=True, completed_steps=args.steps)
            common.save_pt(output / checkpoint_name, state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = dict(device=str(device), torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   elapsed_seconds=time.perf_counter() - run_started,
                   loading_and_initial_audit_seconds=initial_audit_seconds,
                   paired_training_and_periodic_validation_seconds=time.perf_counter() - training_started,
                   device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    summary = dict(complete=True, smoke=args.smoke, test_accessed=False, runtime=runtime,
                   step0_gate=gate, best_valid_miou=best, final_validation=result,
                   frozen_source_unchanged=True, F_fusion_unchanged=True,
                   fusion_buffers_unchanged=True, initial_module_fingerprints=before,
                   gradient_clipping=manifest["gradient_clipping"],
                   final_module_fingerprints=after, independent_gradients=gradient_proof,
                   observed_crops_per_arm=observed_crops, crop_name_stream_sha256=crop_hash.hexdigest(),
                   parameter_groups_changed={name: {part: before[name][part] != after[name][part]
                                                     for part in before[name]} for name in arms},
                   interpretation="Compare U against F under fixed Q/B. Evidence concerns only fusion adaptation, not full-backbone unfreezing.")
    common.save_json(output / "summary.json", summary)
    manifest["complete"] = True
    common.save_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
