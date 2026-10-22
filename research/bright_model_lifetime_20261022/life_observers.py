"""Disable added wrappers in minimal; retain exact original model validators."""
import contextlib
import gc
import weakref

from ab_observers import ScalarOnly, hook_ids
from evaluation_observer import track_evaluation
from passive_routes import PassiveRoutes
from evaluate_checkpoint import write_json
from grid_probe import read_json


@contextlib.contextmanager
def execution(torch, runtime, factory, evaluator, output, position, names,
              state, observer, classifier, mode, trace):
    if mode not in ('minimal', 'instrumented'):
        raise ValueError('Unknown lifetime case')
    front = runtime.bundle['front']
    before = hook_ids(front)
    originals = [(obj, name, name in vars(obj), vars(obj).get(name), getattr(obj, name))
                 for obj, name in ((runtime, 'prepare'), (factory, 'prepare'),
                                   (factory, 'forward'), (evaluator, '_validate_probability'),
                                   (state, 'restore_rng'))]
    try:
        with contextlib.ExitStack() as stack:
            if mode == 'instrumented':
                stack.enter_context(PassiveRoutes(torch, classifier, output, position))
                stack.enter_context(track_evaluation(runtime, factory, evaluator, names, state, observer))
                stack.enter_context(ScalarOnly(runtime, factory, position, trace))
            # minimal leaves original runtime/factory methods and validators bound.
            yield
    finally:
        restored = all((name in vars(obj)) == owned and
                       (vars(obj).get(name) is value if owned else getattr(obj, name) == original)
                       for obj, name, owned, value, original in originals)
        write_json(output / 'lifetime_switch.json', dict(mode=mode,
            added_success_path_wrappers=mode == 'instrumented',
            added_forward_telemetry_hooks=mode == 'instrumented',
            extra_route_snapshot_observers=False, original_validators_preserved=True,
            original_feature_capture_hooks_preserved=True,
            hooks_restored=before == hook_ids(front), bindings_restored=restored,
            tensor_values_modified=False))


def python_tensor_inventory(torch, device_type='cuda'):
    """Owned metadata + weak references only; release gc.get_objects before GC.

    This sees Python-exposed tensors, not all C++ allocations. Storage sizes are
    counted once per storage pointer; no device values or tensor copies read.
    """
    references, storages, largest = [], {}, []
    requires_grad = 0
    for obj in gc.get_objects():
        if not issubclass(type(obj), torch.Tensor) or obj.device.type != device_type:
            continue
        references.append(weakref.ref(obj))
        requires_grad += int(obj.requires_grad)
        if obj.layout == torch.strided:
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr())
            size = storage.nbytes()
            storages[key] = max(storages.get(key, 0), size)
            largest.append(dict(shape=list(obj.shape), dtype=str(obj.dtype),
                                storage_bytes=size, requires_grad=obj.requires_grad))
    # obj/storage/list of gc objects are locals and die when this function returns.
    return references, dict(python_tensor_wrappers=len(references),
        python_requires_grad_wrappers=requires_grad,
        python_visible_unique_storage_bytes=sum(storages.values()),
        python_visible_unique_storages=len(storages),
        largest_python_tensors=sorted(largest, key=lambda r:r['storage_bytes'], reverse=True)[:12],
        coverage='Python-visible tensor wrappers only; not all CUDA/C++ allocations')


def probe_collection(torch, trace, output):
    if trace.failed:
        raise ValueError('No GC/CUDA diagnostic after execution failure')
    trace.sample('before_tensor_inventory')
    before = dict(trace.last)
    refs, inventory = python_tensor_inventory(torch)
    trace.sample('before_explicit_gc')
    pre_gc = dict(trace.last)
    collected = gc.collect()
    # Deliberately no empty_cache, synchronize, dtype/route correction or new forward.
    trace.sample('after_explicit_gc')
    after = dict(trace.last)
    alive = sum(r() is not None for r in refs)
    r = dict(format='BRIGHT_MODEL_GC_OBSERVATION_V1',
        before_inventory=before, before_gc=pre_gc, after_gc=after,
        python_inventory=inventory, python_weakrefs_alive_after_gc=alive,
        python_weakrefs_released=len(refs)-alive, gc_collected_objects=collected,
        allocated_reduction_bytes=pre_gc.get('allocated_bytes', 0)-after.get('allocated_bytes', 0),
        gc_after_scoring_only=True, explicit_gc_changes_execution=True,
        empty_cache_called=False, tensor_values_modified=False,
        historical_failure_resolved=False, training_performed=False, optimizer_updates=0)
    write_json(output / 'gc_observation.json', r)
    return r


def validate_evidence(output, names, mode):
    r = read_json(output / 'lifetime_switch.json')
    if (r.get('mode') != mode or r.get('added_success_path_wrappers') != (mode == 'instrumented')
            or r.get('hooks_restored') is not True or r.get('bindings_restored') is not True
            or r.get('original_validators_preserved') is not True
            or r.get('tensor_values_modified') is not False):
        raise ValueError('Lifetime wrapper closure incomplete')
    g = read_json(output / 'gc_observation.json')
    if (g.get('gc_after_scoring_only') is not True or g.get('empty_cache_called') is not False
            or g.get('optimizer_updates') != 0 or g.get('historical_failure_resolved') is not False):
        raise ValueError('Lifetime GC observation missing/invalid')
    r['gc_observation'] = g
    return r
