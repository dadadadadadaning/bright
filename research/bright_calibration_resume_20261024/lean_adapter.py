"""Calibration execution variant: detach its zero-coefficient evidence only.

Forward readouts, state_dict and the original Reader remain unchanged. Other
modes retain the published behavior. This does not repair indices or establish
a CUDA fault cause. Finite-input CPU equivalence is covered separately.

Explicit Train-memory evidence for conditional damage classification.

This extends the source-checked readout adapter without changing its visual
Reader, candidate search, routes or Train-only label bank.  Query labels are
not accepted.  The three foreground logits are corrected while retaining
their original log-sum-exp.  Therefore background probability and total
foreground probability are preserved up to floating-point error.  The binary
mask obtained by merging a FOUR-CLASS ARGMAX can nevertheless change.

The coefficient is signed, bounded and initialized at zero. A negative learned
coefficient is anti-evidence, not proof of beneficial label transfer.  The
calibration control learns the same three class offsets without memory gain.
"""
from __future__ import annotations

import inspect

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("original", "calibration", "severity_memory", "shuffled_memory")
SHUFFLE_ALGORITHM = "cpu_generator_seed_selected_parent_valid_slots_randperm_v2"
COEFFICIENT_BOUND = 2.0


def shuffled_label_copy(labels, valid, support_ids, seed):
    """Permute labels only inside this example's selected legal parent pool.

    Only the added probability readout receives this copy. The base Reader's
    label embedding continues to use the original labels.  This local CPU RNG
    does not consume the training or CUDA RNG streams. Donors are restricted to
    valid slots belonging to the unique nonpadding parents in ``support_ids``.
    Unselected bank rows cannot donate labels: these can include the current
    training query or same-event exclusions. Their contents remain unchanged.
    The same number of selected valid slots and seed give the same permutation;
    this is a deterministic correspondence control, not independent labels.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if labels.shape[:-1] != valid.shape or labels.shape[-1] != 4 or valid.dtype != torch.bool:
        raise ValueError("Expected labels [M,2,4] and boolean valid [M,2]")
    if support_ids.dtype != torch.long or support_ids.ndim != 2:
        raise ValueError("support_ids must be int64 [R,K]")
    if support_ids.device != labels.device or valid.device != labels.device:
        raise ValueError("labels, valid and support_ids must share a device")
    if bool(((support_ids < -1) | (support_ids >= labels.shape[0])).any()):
        raise ValueError("support_ids out of range; only -1 padding is allowed")
    flat = labels.detach().reshape(-1, 4)
    selected_parents = support_ids[support_ids >= 0].unique(sorted=True)
    selected_valid = torch.zeros_like(valid)
    selected_valid[selected_parents] = valid[selected_parents]
    indices = selected_valid.reshape(-1).nonzero(as_tuple=False).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(indices.numel(), generator=generator, device="cpu")
    result = flat.clone()
    if indices.numel():
        result[indices] = flat[indices[order.to(indices.device)]]
    return result.reshape_as(labels)


def preserve_foreground_mass(logits, correction):
    """Apply a three-class logit correction without moving background mass.

    No special zero branch is used: at zero the function is bitwise identical
    to its input while its derivative with respect to correction remains live.
    """
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("logits must be [B,4,H,W]")
    if correction.shape != logits[:, 1:].shape:
        raise ValueError("correction must be [B,3,H,W]")
    foreground = logits[:, 1:]
    adjusted = foreground + correction
    offset = torch.logsumexp(adjusted, dim=1, keepdim=True) - torch.logsumexp(
        foreground, dim=1, keepdim=True)
    return torch.cat((logits[:, :1], adjusted - offset), dim=1)


def make_classifier(base_class, old_adapter_module, slots_module, mode):
    """Factory for the existing batch-one runtime; state_dict is versioned.

    Constructor accepts original model options plus ``shuffle_seed=3407``.
    All non-original modes add exactly four zero-initialized scalar parameters.
    Adding these constants consumes no random numbers, preserving the original
    initialization for the common parameters. Do not strict-load an old state
    dictionary into a new mode without explicitly supplying the new zeros.
    """
    if mode not in MODES:
        raise ValueError("Unknown severity mode: " + str(mode))
    observed_class = old_adapter_module.make_classifier(
        base_class, "original" if mode == "original" else "observe")
    forward_signature = inspect.signature(base_class.forward)

    class SeverityClassifier(observed_class):
        severity_mode = mode
        shuffle_algorithm = SHUFFLE_ALGORITHM

        def __init__(self, *args, shuffle_seed=3407, **kwargs):
            if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int) or shuffle_seed < 0:
                raise ValueError("shuffle_seed must be a nonnegative integer")
            super().__init__(*args, **kwargs)
            self.shuffle_seed = shuffle_seed
            if mode != "original":
                self.severity_bias_raw = nn.Parameter(torch.zeros(3))
                self.memory_gain_raw = nn.Parameter(torch.zeros(()))

        def _read(self, query, support_ids, table_k, table_v, table_valid,
                  query_valid, chunk_size):
            if mode != "shuffled_memory":
                return super()._read(query, support_ids, table_k, table_v,
                                     table_valid, query_valid, chunk_size)
            labels = self._readout_labels
            try:
                self._readout_labels = shuffled_label_copy(
                    labels, table_valid, support_ids, self.shuffle_seed)
                return super()._read(query, support_ids, table_k, table_v,
                                     table_valid, query_valid, chunk_size)
            finally:
                self._readout_labels = labels

        def forward(self, *args, **kwargs):
            if mode == "original":
                return super().forward(*args, **kwargs)
            bound = forward_signature.bind(self, *args, **kwargs)
            result = super().forward(*args, **kwargs)
            common = bound.arguments
            pixel_features = common["pixel_features"]
            height, width = pixel_features.shape[-2:]
            read_probability = result["read_probability"]
            slots = read_probability.reshape(1, -1, 4)
            vote_grid = slots_module.sparse_unpool(
                slots, common["route_q"], common["route_ids"]
            ).reshape(1, 4, height, width)
            votes = F.interpolate(vote_grid, size=result["logits"].shape[-2:],
                                  mode="bilinear", align_corners=False)
            foreground_votes = votes[:, 1:]
            foreground_mass = foreground_votes.sum(dim=1, keepdim=True)
            # Centering does not change a conditional softmax. It makes the
            # bounded, zero-mean evidence visible and gives exact zero when
            # no legal support contributes. Background mass attenuates rather
            # than being amplified by division near zero.
            evidence = foreground_votes - foreground_mass / 3.0
            bias = COEFFICIENT_BOUND * self.severity_bias_raw.tanh()
            gain = COEFFICIENT_BOUND * self.memory_gain_raw.tanh()
            if mode == "calibration":
                gain = gain * 0.0
                # The evidence has zero coefficient in this control. Keep the
                # gain connection (zero gradient, not None) for AdamW state
                # compatibility, but exclude unused evidence from backward.
                correction_evidence = evidence.detach()
            else:
                correction_evidence = evidence
            correction = bias[None, :, None, None] + gain * correction_evidence
            logits = preserve_foreground_mass(result["logits"], correction)
            result["logits"] = logits
            result["probabilities"] = logits.softmax(1)
            # Only compact, detached summaries are added. Existing readout
            # outputs remain caller-owned; the model retains no input graph.
            result["severity_bias"] = bias.detach()
            result["memory_gain"] = gain.detach()
            result["severity_evidence_l1"] = evidence.detach().abs().mean()
            result["severity_memory_mass"] = foreground_mass.detach().mean()
            return result

    SeverityClassifier.__name__ = "Severity_" + mode + "_" + base_class.__name__
    SeverityClassifier.__qualname__ = SeverityClassifier.__name__
    return SeverityClassifier
