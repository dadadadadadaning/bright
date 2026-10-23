"""Small, source-checked extensions of the published BRIGHT joint classifier.

No labels from the query image are accepted here.  Query GT aggregation belongs
in the training loss, outside the forward/retrieval path.  This adapter does not
repair routes, overwrite source modules, add trainable parameters, or persist an
attention graph on the model.  Use ``state_dict`` checkpoints, not model pickles.

``observe`` and ``read_aux`` have identical forwards; the caller decides whether
to add a training loss.  New outputs are bounded summaries, not full attention:
  read_probability:    [1, regions, 2, 4], differentiable through Q/K attention
  read_valid:          [1, regions, 2], valid query and at least one legal key
  candidate_class_max: [1, regions, 2, 4], largest candidate label mass per class
The last quantity is a componentwise attainable bound, not an oracle mIoU and
not a claim that all four component maxima are jointly attainable.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import torch
from torch import Tensor


MODES = ("original", "observe", "no_reader", "no_label", "read_aux")
SUPPORTED_READ_AST_SHA256 = "e65f1e24cba9da45f1fe661317b1c07f10c85e8e28c75480d3fecd02f04078ee"


def reader_source_fingerprint(base_class) -> str:
    """Ignore whitespace/comments; fail if source is unavailable or differs."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(base_class._read)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise ValueError("Cannot inspect the base Reader; supply the published Python source") from exc
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()


def make_classifier(base_class, mode: str = "original"):
    """Return a subclass with the original constructor and state_dict schema.

    Intended use::

        ReadClassifier = make_classifier(JointRegionMemoryClassifier, "read_aux")
        model = ReadClassifier(chunk_size=32)
        model.load_state_dict(original_state, strict=True)
        out = model(**example.common, **support)

    ``no_reader`` means the memory-context tensor is zero from training start;
    all local branches and the fusion width remain present. ``no_label`` means
    only label_embedding's output is zero, matching the published N ablation.
    A temporary evaluation intervention is not a from-start training ablation.
    The wrapper is deliberately non-reentrant and is for the existing batch-one
    single-device runner, not concurrent calls sharing one model instance.
    """
    if mode not in MODES:
        raise ValueError("Unknown readout mode: " + str(mode))
    if reader_source_fingerprint(base_class) != SUPPORTED_READ_AST_SHA256:
        raise ValueError("Base Reader implementation differs from the verified source; review before adapting")
    forward_signature = inspect.signature(base_class.forward)
    if "support_labels" not in forward_signature.parameters:
        raise ValueError("The base classifier does not expose support_labels")

    class ReadoutClassifier(base_class):
        readout_mode = mode

        def forward(self, *args, **kwargs):
            # Exact original path: no copies, extra kernels, hooks, or outputs.
            if mode == "original":
                return super().forward(*args, **kwargs)
            if self.__dict__.get("_readout_active", False):
                raise RuntimeError("Readout adapter does not support reentrant forwards")
            self._readout_active = True
            hook = None
            try:
                if mode in ("observe", "read_aux"):
                    bound = forward_signature.bind(self, *args, **kwargs)
                    # Detach the extra label-readout target; the published
                    # feature/value forward remains unchanged.
                    self._readout_labels = bound.arguments["support_labels"].detach()
                elif mode == "no_label":
                    hook = self.label_embedding.register_forward_hook(
                        lambda module, inputs, output: torch.zeros_like(output)
                    )
                result = super().forward(*args, **kwargs)
                if mode in ("observe", "read_aux"):
                    readout = self.__dict__.get("_readout_result")
                    if readout is None:
                        raise RuntimeError("Base forward did not execute the expected Reader")
                    result.update(readout)
                return result
            finally:
                if hook is not None:
                    hook.remove()
                # Delete, rather than keeping detached or graph-bearing copies.
                for name in ("_readout_active", "_readout_labels", "_readout_result"):
                    self.__dict__.pop(name, None)

        def _read(self, query: Tensor, support_ids: Tensor, table_k: Tensor,
                  table_v: Tensor, table_valid: Tensor, query_valid: Tensor,
                  chunk_size: int) -> Tensor:
            if mode == "no_reader":
                return query * 0.0
            if mode not in ("observe", "read_aux"):
                return super()._read(query, support_ids, table_k, table_v,
                                     table_valid, query_valid, chunk_size)
            if not self.__dict__.get("_readout_active", False):
                raise RuntimeError("Use forward() to obtain readout summaries")
            regions, _, width = query.shape
            if table_k.shape[0] == 0:
                zero = query.sum(-1, keepdim=True).expand(-1, -1, 4) * 0.0
                self._readout_result = {
                    "read_probability": zero[None],
                    "read_valid": torch.zeros_like(query_valid)[None],
                    "candidate_class_max": torch.zeros_like(zero)[None],
                }
                return query * 0.0
            clean_labels = torch.where(table_valid[..., None], self._readout_labels, 0.0)
            dimension = width // self.heads
            outputs, probabilities, valid_queries, maxima = [], [], [], []
            for start in range(0, regions, chunk_size):
                end = min(start + chunk_size, regions)
                fathers = support_ids[start:end]
                # Exactly the original -1 structural-padding gather convention.
                # Positive IDs are validated in the unmodified base forward.
                safe = fathers.clamp_min(0)
                valid = (fathers[..., None] >= 0) & table_valid[safe]
                valid = valid.flatten(1)
                k = table_k[safe].reshape(end-start, -1, self.heads, dimension)
                v = table_v[safe].reshape(end-start, -1, self.heads, dimension)
                q = self.query_projection(query[start:end]).reshape(end-start, 2, self.heads, dimension)
                scores = torch.einsum("rqhd,rkhd->rqhk", q, k) * (dimension ** -0.5)
                mask = valid[:, None, None, :]
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
                attention = torch.softmax(scores, -1) * mask.to(scores.dtype)
                denominator = attention.sum(-1, keepdim=True)
                attention = attention / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
                context = torch.einsum("rqhk,rkhd->rqhd", attention, v).reshape(end-start, 2, width)
                context = self.context_projection(context)
                outputs.append(torch.where(query_valid[start:end, :, None], context, 0.0))

                candidate_labels = clean_labels[safe].reshape(end-start, -1, 4)
                candidate_labels = torch.where(valid[..., None], candidate_labels, 0.0)
                has_read = query_valid[start:end] & valid.any(-1)[:, None]
                p_read = torch.einsum("rsk,rkc->rsc", attention.mean(2), candidate_labels)
                probabilities.append(torch.where(has_read[..., None], p_read, 0.0))
                candidate_max = candidate_labels.amax(dim=1)[:, None, :].expand(-1, 2, -1)
                maxima.append(torch.where(has_read[..., None], candidate_max, 0.0))
                valid_queries.append(has_read)
            self._readout_result = {
                "read_probability": torch.cat(probabilities, 0)[None],
                "read_valid": torch.cat(valid_queries, 0)[None],
                "candidate_class_max": torch.cat(maxima, 0)[None],
            }
            return torch.cat(outputs, 0)

    ReadoutClassifier.__name__ = "Readout_" + mode + "_" + base_class.__name__
    ReadoutClassifier.__qualname__ = ReadoutClassifier.__name__
    return ReadoutClassifier
