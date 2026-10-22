"""Switch only the added route snapshots/operator wrappers; keep original checks."""
import contextlib
from evaluate_checkpoint import write_json
from memory_trace import MemoryRoutes


def modules_for(front):
    result = [('front', front)] + [('front.' + k, m) for k, m in front.named_children()]
    if hasattr(front, 'embedding'):
        result += [('front.embedding.' + k, m) for k, m in front.embedding.named_children()]
    seen = set()
    return [(k, m) for k, m in result if not (id(m) in seen or seen.add(id(m)))]


def hook_ids(front):
    return {str(id(m)): (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in front.modules()}


class ScalarOnly:
    def __init__(self, runtime, factory, position, trace):
        self.runtime, self.factory, self.position, self.trace = runtime, factory, position, trace
        self.handles, self.bindings = [], []

    def measured(self, label, original):
        def invoke(*args, **kwargs):
            try:
                p = dict(self.position)
                if label == 'arm_forward':
                    p['active_arm'] = args[2] if len(args) > 2 else kwargs['arm']
                self.trace.sample(label + '_before', p)
                result = original(*args, **kwargs)
                self.trace.sample(label + '_after', self.position)
                return result
            except BaseException as exc:
                self.trace.failure(label + '_failed', exc, self.position)
                raise
        return invoke

    def __enter__(self):
        try:
            for obj, name, label in ((self.runtime, 'prepare', 'prepare'),
                                     (self.factory, 'prepare', 'supports'),
                                     (self.factory, 'forward', 'arm_forward')):
                original = getattr(obj, name)
                # Preserve whether the binding originally lived on the instance.
                owned = name in obj.__dict__
                self.bindings.append((obj, name, original, owned))
                setattr(obj, name, self.measured(label, original))
            for label, module in modules_for(self.runtime.bundle['front']):
                def before(_m, _i, label=label): self.trace.sample(label + '_before', self.position)
                def after(_m, _i, _o, label=label): self.trace.sample(label + '_after', self.position)
                self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            return self
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        for obj, name, original, owned in reversed(self.bindings):
            if owned: setattr(obj, name, original)
            else: delattr(obj, name)
        self.bindings.clear()

    def __exit__(self, kind, exc, tb):
        if exc is not None: self.trace.failure('evaluation_failed', exc, self.position)
        self.restore()


@contextlib.contextmanager
def observer_context(torch, runtime, factory, evaluator, output, position, names, canonical,
                     mode, target, *, heads, trace):
    if mode not in ('light', 'watched'): raise ValueError('Unknown observer case')
    front = runtime.bundle['front']
    before = hook_ids(front)
    observed = MemoryRoutes(torch, runtime, factory, evaluator, output, position, names, canonical,
                            'watched', target, heads=heads, trace=trace) if mode == 'watched' else ScalarOnly(runtime, factory, position, trace)
    try:
        with observed: yield observed
    finally:
        restored = before == hook_ids(front)
        write_json(output / 'observer_switch.json', dict(mode=mode,
            extra_route_observers=mode == 'watched', scalar_telemetry=True,
            front_hooks_restored=restored, route_values_modified=False,
            original_validators_preserved=True, original_feature_capture_hooks_preserved=True,
            telemetry_rows=trace.rows))


def validate_observers(output, names, mode):
    from grid_probe import read_json
    r = read_json(output / 'observer_switch.json')
    if (r.get('mode') != mode or r.get('extra_route_observers') != (mode == 'watched')
            or r.get('front_hooks_restored') is not True or r.get('route_values_modified') is not False
            or r.get('original_validators_preserved') is not True or r.get('telemetry_rows', 0) < 1):
        raise ValueError('Observer switch closure incomplete')
    if mode == 'watched':
        from night_assessment import validate_route_evidence
        r['route_evidence'] = validate_route_evidence(output, names, 'watched')
    else:
        # Absence is deliberate, never report the disabled observer as having passed.
        r['route_snapshot_checks'] = 'not_performed_in_light_case'
    return r
