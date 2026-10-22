"""R operation windows plus independent parent-index mirrors. Never repair input."""
import json
from evaluate_checkpoint import write_json
from sweep_routes import StageRoutes
from night_routes import metadata


class Journal:
    def __init__(self, stream, target):
        self.stream, self.target = stream, target
    def write(self, line):
        row = json.loads(line); p = row['position']
        if row['changed_keys'] or p.get('image') == self.target:
            self.stream.write(line)
        elif p.get('watch_stage') in ('extract_after', 'image_complete'):
            self.stream.write(json.dumps(dict(position=p, changed_keys=[],
                route_sha256={','.join(g['keys']):g['statistics']['sha256'] for g in row['groups']}))+'\n')
    def flush(self): self.stream.flush()
    def close(self): self.stream.close()


class CorroboratedRoutes(StageRoutes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_gpu = self.reference_cpu = None
        self.mirror_images = self.device_comparisons = 0
        self.last_device_equal = None
        self.confirmed = False
        self.corroboration_error = None

    def corroborate(self, reason):
        if self.confirmed: return
        self.confirmed = True
        record = dict(position=dict(self.position), reason=reason,
            compared_on_device=str(self.example.common['ids'].device),
            device_equal_before_cpu_snapshot=self.last_device_equal,
            route_values_modified=False, observations_synchronize=True,
            causal_conclusion='unresolved; these are observations at different instants', reads=[])
        payload = dict(reference_cpu=self.reference_cpu,
                       first_snapshot=self.latest_cpu.get('parent.ids'))
        try:
            parent = self.example.common['ids']
            record['parent_metadata'] = metadata(parent)
            record['device_equal_after_detection'] = bool(self.torch.equal(parent, self.reference_gpu))
            mirror = self.reference_gpu.detach().to(device='cpu', copy=True).contiguous()
            payload['mirror_cpu'] = mirror
            record['mirror_cpu_equals_initial'] = bool(self.torch.equal(mirror, self.reference_cpu))
            for index in range(2):
                value = parent.detach().to(device='cpu', copy=True).contiguous()
                payload['repeat_cpu_%s' % index] = value
                record['reads'].append(dict(index=index,
                    equals_initial=bool(self.torch.equal(value, self.reference_cpu)),
                    changed_elements=int((value != self.reference_cpu).sum())))
            record['repeat_copies_equal'] = bool(self.torch.equal(payload['repeat_cpu_0'], payload['repeat_cpu_1']))
        except BaseException as exc:
            # If a corroboration read faults, stop GPU access immediately.
            record['confirmation_error'] = type(exc).__name__ + ': ' + str(exc)
        try:
            self.torch.save(payload, self.output / 'parent_ids_corroboration_cpu.pt')
            write_json(self.output / 'parent_ids_corroboration.json', record)
        except BaseException as exc:
            self.corroboration_error = type(exc).__name__ + ': ' + str(exc)

    def snapshot(self, stage):
        if not self.tensors(): return super().snapshot(stage)
        self.position['watch_stage'] = stage
        if stage == 'extract_after':
            self.reference_gpu = self.reference_cpu = None
            self.last_device_equal = None
            super().snapshot(stage)
            parent = self.example.common['ids']
            self.reference_cpu = self.baseline['parent.ids'].clone()
            self.reference_gpu = parent.detach().clone()
            copied = self.reference_gpu.to(device='cpu', copy=True).contiguous()
            self.mirror_images += 1
            if not self.torch.equal(copied, self.reference_cpu):
                self.corroborate('New device mirror differs from initial CPU snapshot')
                raise ValueError('Initial parent.ids device mirror/CPU snapshot disagree')
            return
        if self.reference_gpu is not None:
            self.last_device_equal = bool(self.torch.equal(self.example.common['ids'], self.reference_gpu))
            self.device_comparisons += 1
        try:
            super().snapshot(stage)
        except ValueError as exc:
            # Only the explicit CPU-byte mismatch gets additional device reads.
            # CUDA exceptions and arbitrary validation exceptions do not.
            if str(exc).startswith('Frozen routing tensors changed') and 'parent.ids' in (self.latest or {}).get('changed_keys', []):
                self.corroborate(str(exc))
            raise
        if self.last_device_equal is False:
            self.corroborate('Device mirror comparison differed while CPU snapshot matched baseline')
            raise ValueError('Parent.ids device mirror comparison failed')

    def wrap(self, label, original):
        def invoke(*args, **kwargs):
            detailed = self.position.get('active_arm') == 'r' or self.position.get('image') == self.target
            if not self.in_forward or not detailed: return original(*args, **kwargs)
            self.boundary_calls += 1
            previous = self.position.get('operator')
            self.position['operator'] = label
            self.position['operator_sequence'] = self.boundary_calls
            try:
                self.snapshot(label+'_before')
                value = original(*args, **kwargs)
                self.snapshot(label+'_after')
                return value
            except BaseException as exc:
                self.record(exc)
                raise
            finally:
                if previous is None: self.position.pop('operator', None)
                else: self.position['operator'] = previous
                self.position.pop('operator_sequence', None)
        return invoke

    def __enter__(self):
        super().__enter__()
        self.journal = Journal(self.journal.stream, self.target)
        return self

    def __exit__(self, *args):
        try: return super().__exit__(*args)
        finally:
            path = self.output / 'operator_observer_summary.json'
            row = json.loads(path.read_text())
            row.update(detailed_arm_default='r', all_arm_detailed_target=self.target)
            write_json(path, row)
            write_json(self.output / 'corroboration_observer_summary.json', dict(
                mirror_images=self.mirror_images, device_comparisons=self.device_comparisons,
                corroboration_triggered=self.confirmed, evidence_error=self.corroboration_error,
                route_values_modified=False, bindings_restored=True,
                compared_on_device='same device as parent.ids', observations_synchronize=True))
