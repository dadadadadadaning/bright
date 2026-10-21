"""Opt-in raw-slot cosine prior for the inspected BRIGHT joint classifier.

Integration component, NOT a training launcher. Does not alter base modules,
retrieval, labels, routing, loss or checkpoints. Save prior_config separately
in every checkpoint: state_dict intentionally has the original parameter ABI.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import random

import torch
from torch.nn import functional as F

SUPPORTED_READER_SHA256 = "e2ed6c959391f5f1e929a74c2f95ad15daf6775359161b5eb5639e25e6aeb020"


def shuffled_valid_prior(prior, valid, *, image_key, start_region, seed):
    """Permute only prior values, independently per query slot; preserve masks.

    Deterministic per image/region/query-role, independent of chunk size and
    process hash randomization. Private Python RNG; no training RNG consumed.
    Candidate IDs, keys, values and labels are never permuted.
    """
    result = prior.clone()
    for row in range(prior.shape[0]):
        positions = valid[row].nonzero(as_tuple=False).flatten().tolist()
        if len(positions) < 2:
            continue
        for role in range(prior.shape[1]):
            token = f"{seed}|{image_key}|{start_region + row}|{role}".encode()
            private = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))
            donors = positions.copy()
            private.shuffle(donors)
            result[row, role, positions] = prior[row, role, donors]
    return result


def classifier_with_prior(base_class):
    required = ("pixel_features", "slot_f", "support_f", "support_valid", "slot_valid")
    signature = inspect.signature(base_class.forward)
    if not all(k in signature.parameters for k in required) or not hasattr(base_class, "_read"):
        raise TypeError("Expected the inspected JointRegionMemoryClassifier ABI")
    read_params = list(inspect.signature(base_class._read).parameters)
    if read_params != ["self", "query", "support_ids", "table_k", "table_v", "table_valid", "query_valid", "chunk_size"]:
        raise TypeError("Reader ABI changed; adapt and rerun compatibility checks first")
    read_sha = hashlib.sha256(inspect.getsource(base_class._read).encode()).hexdigest()
    if read_sha != SUPPORTED_READER_SHA256:
        raise ValueError("Original Reader implementation differs from the inspected version: " + read_sha
                         + "; compare the implementation before integrating the prior")

    class PriorClassifier(base_class):
        def __init__(self, *args, prior_beta=0.0, prior_mode="cosine", prior_seed=3407, **kwargs):
            super().__init__(*args, **kwargs)
            if not math.isfinite(prior_beta) or prior_beta < 0:
                raise ValueError("prior_beta must be finite and nonnegative")
            if prior_mode not in ("cosine", "shuffled"):
                raise ValueError("prior_mode must be cosine or shuffled")
            self.prior_beta = float(prior_beta)
            self.prior_mode = prior_mode
            self.prior_seed = int(prior_seed)
            self._raw_prior_inputs = None

        def prior_config(self):
            return dict(format="BRIGHT_RAW_SLOT_COSINE_PRIOR_V1", beta=self.prior_beta,
                        mode=self.prior_mode, seed=self.prior_seed,
                        feature="frozen raw 256D slot_f/support_f, detached",
                        normalization="L2 cosine, eps=1e-12; no per-row z-score",
                        base_reader_sha256=hashlib.sha256(inspect.getsource(base_class._read).encode()).hexdigest())

        def forward(self, *args, prior_key=None, **kwargs):
            # Bypass even feature preparation, preserving the original beta=0 path.
            if self.prior_beta == 0:
                return super().forward(*args, **kwargs)
            if self._raw_prior_inputs is not None:
                raise RuntimeError("Use sequential forwards on each model instance")
            if self.prior_mode == "shuffled" and (not isinstance(prior_key, str) or not prior_key):
                raise ValueError("C2 requires prior_key=example.image for deterministic shuffling")
            arguments = signature.bind(self, *args, **kwargs).arguments
            self._raw_prior_inputs = (arguments["slot_f"], arguments["support_f"], prior_key)
            try:
                # All original validation, slot encoders, label values and unpooling remain active.
                return super().forward(*args, **kwargs)
            finally:
                self._raw_prior_inputs = None

        def _read(self, query, support_ids, table_k, table_v, table_valid, query_valid, chunk_size):
            if self.prior_beta == 0:
                return super()._read(query, support_ids, table_k, table_v, table_valid, query_valid, chunk_size)
            if table_k.shape[0] == 0:
                return query * 0.0
            if self._raw_prior_inputs is None:
                raise RuntimeError("Nonzero prior requires the full forward input context")
            raw_q, raw_s, image_key = self._raw_prior_inputs
            # No GT-dependent filtering, gradients, or in-place edits to runtime tensors.
            with torch.no_grad():
                fq = F.normalize(torch.where(query_valid[..., None], raw_q[0].detach(), 0.0).float(), dim=-1, eps=1e-12)
                fs = F.normalize(torch.where(table_valid[..., None], raw_s.detach(), 0.0).float(), dim=-1, eps=1e-12)
            regions, _, width = query.shape
            dimension = width // self.heads
            outputs = []
            for start in range(0, regions, chunk_size):
                end = min(start + chunk_size, regions)
                fathers = support_ids[start:end]
                safe = fathers.clamp_min(0)
                valid = ((fathers[..., None] >= 0) & table_valid[safe]).flatten(1)
                k = table_k[safe].reshape(end-start, -1, self.heads, dimension)
                v = table_v[safe].reshape(end-start, -1, self.heads, dimension)
                q = self.query_projection(query[start:end]).reshape(end-start, 2, self.heads, dimension)
                scores = torch.einsum("rqhd,rkhd->rqhk", q, k) * (dimension ** -0.5)
                candidates = fs[safe].reshape(end-start, -1, fs.shape[-1])
                prior = torch.einsum("rqd,rkd->rqk", fq[start:end], candidates).clamp(-1, 1)
                prior = torch.where(valid[:, None, :], prior, 0.0)
                if self.prior_mode == "shuffled":
                    prior = shuffled_valid_prior(prior, valid, image_key=image_key,
                                                 start_region=start, seed=self.prior_seed)
                scores = scores + self.prior_beta * prior[:, :, None, :].to(scores.dtype)
                mask = valid[:, None, None, :]
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
                attention = torch.softmax(scores, -1) * mask.to(scores.dtype)
                denominator = attention.sum(-1, keepdim=True)
                attention = attention / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
                context = torch.einsum("rqhk,rkhd->rqhd", attention, v).reshape(end-start, 2, width)
                context = self.context_projection(context)
                outputs.append(torch.where(query_valid[start:end, :, None], context, 0.0))
            return torch.cat(outputs, 0)

    return PriorClassifier
