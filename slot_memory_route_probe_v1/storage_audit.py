"""Metadata-only checks for frozen inputs sharing mutable training storage."""
from __future__ import annotations


def _metadata(torch, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Storage audit requires tensor inputs")
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("Storage audit requires materialized strided tensors")
    storage = tensor.untyped_storage()
    start, nbytes = int(storage.data_ptr()), int(storage.nbytes())
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    element_size, offset = tensor.element_size(), tensor.storage_offset()
    low = high = offset
    for size, stride in zip(shape, strides):
        distance = (size - 1) * stride
        low += min(0, distance)
        high += max(0, distance)
    empty = tensor.numel() == 0
    if empty:
        low = high = offset
    byte_start = start + low * element_size
    byte_end = byte_start if empty else start + (high + 1) * element_size
    # Dense transposes are exact byte intervals too. Sliced/interleaved views
    # can contain holes, so their bounding intervals are conservative evidence.
    dense = True
    expected_stride = 1
    for stride, size in sorted(
        (abs(stride), size) for size, stride in zip(shape, strides) if size > 1
    ):
        if stride != expected_stride:
            dense = False
            break
        expected_stride *= size
    try:
        version = int(tensor._version)
    except RuntimeError:
        version = None  # Inference tensors have no version counter.
    return dict(
        tensor_id=id(tensor), device=str(tensor.device),
        device_key=[tensor.device.type, tensor.device.index],
        dtype=str(tensor.dtype), shape=list(shape), stride=list(strides),
        numel=tensor.numel(), element_size=element_size,
        storage_offset=offset, data_ptr=int(tensor.data_ptr()),
        storage_start=start, storage_end=start + nbytes, storage_nbytes=nbytes,
        byte_span_start=byte_start, byte_span_end=byte_end,
        byte_span_is_dense=bool(dense or empty),
        requires_grad=bool(tensor.requires_grad), version=version,
    )


def _alias(left, right):
    """Classify allocation overlap; byte spans may conservatively include holes."""
    if left["device_key"] != right["device_key"]:
        return None
    storage_start = max(left["storage_start"], right["storage_start"])
    storage_end = min(left["storage_end"], right["storage_end"])
    if storage_start >= storage_end:
        return None
    byte_start = max(left["byte_span_start"], right["byte_span_start"])
    byte_end = min(left["byte_span_end"], right["byte_span_end"])
    overlap = byte_start < byte_end
    confirmed = (
        bool(overlap) if not overlap or (
            left["byte_span_is_dense"] and right["byte_span_is_dense"]
        ) else None
    )
    return dict(
        storage_ranges_overlap=True,
        same_storage_start=left["storage_start"] == right["storage_start"],
        storage_overlap_start=storage_start, storage_overlap_end=storage_end,
        byte_spans_overlap=overlap,
        byte_overlap_start=byte_start if overlap else None,
        byte_overlap_end=byte_end if overlap else None,
        tensor_overlap_confirmed=confirmed,
        relation=(
            "overlapping_tensor_byte_spans" if overlap
            else "disjoint_tensor_byte_spans_in_shared_storage"
        ),
    )


def _state_tensors(torch, value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = str(key) if not path else path + "." + str(key)
            yield from _state_tensors(torch, item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _state_tensors(torch, item, "%s[%s]" % (path, index))


def audit_storage(torch, watched, heads, optimizers):
    """Report aliases without reading tensor values or touching training state.

    Any shared storage with a mutable owner appears in ``unsafe_aliases``,
    including disjoint views. ``tensor_overlap_confirmed`` distinguishes proved
    byte overlap from conservative bounding intervals on non-dense views.
    Frozen input aliases are reported separately and are not marked unsafe.
    """
    watch_records = [
        dict(owner_type="watched", arm=None, name=str(name),
             metadata=_metadata(torch, tensor))
        for name, tensor in watched.items()
    ]
    owners = []
    parameter_names = {}
    for arm, head in heads.items():
        parameter_names[arm] = {}
        for name, parameter in head.named_parameters():
            parameter_names[arm][id(parameter)] = name
            if not parameter.requires_grad:
                continue
            owners.append(dict(
                owner_type="parameter", arm=str(arm), name=name,
                metadata=_metadata(torch, parameter),
            ))
            if parameter.grad is not None:
                owners.append(dict(
                    owner_type="gradient", arm=str(arm), name=name,
                    metadata=_metadata(torch, parameter.grad),
                ))
    for arm, optimizer in optimizers.items():
        names = parameter_names.get(arm, {})
        for index, (parameter, state) in enumerate(optimizer.state.items()):
            parameter_name = names.get(id(parameter), "unmapped_parameter_%s" % index)
            for path, tensor in _state_tensors(torch, state):
                owners.append(dict(
                    owner_type="optimizer_state", arm=str(arm),
                    name=parameter_name + "." + path,
                    parameter_name=parameter_name, state_path=path,
                    metadata=_metadata(torch, tensor),
                ))

    unsafe, frozen_aliases = [], []
    for watch in watch_records:
        for owner in owners:
            relation = _alias(watch["metadata"], owner["metadata"])
            if relation is not None:
                unsafe.append(dict(watched=watch, owner=owner, **relation))
    for index, left in enumerate(watch_records):
        for right in watch_records[index + 1:]:
            relation = _alias(left["metadata"], right["metadata"])
            if relation is not None:
                frozen_aliases.append(dict(left=left, right=right, **relation))
    counts = {}
    for owner in owners:
        kind = owner["owner_type"]
        counts[kind] = counts.get(kind, 0) + 1
    return dict(
        metadata_only=True, values_read=False, watched=watch_records,
        owned_tensor_count=len(owners), owned_tensor_counts=counts,
        unsafe_aliases=unsafe, watch_aliases=frozen_aliases,
        policy="Any watched/mutable-owner storage overlap is conservatively unsafe.",
        interval_note=(
            "Addresses are device-specific, half-open byte intervals. Non-dense "
            "tensor spans can include holes; only dense intersecting spans prove "
            "tensor-byte overlap. Disjoint spans can still share storage."
        ),
    )
