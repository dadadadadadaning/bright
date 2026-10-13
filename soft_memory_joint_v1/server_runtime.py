"""A/B/C/D bridge using the verified E0 source and fixed V3 parent retrieval.

All source hooks and integrity gates are inherited from ``base_runtime``. This
module does not patch or bypass native pooling. A native CUDA failure still
stops the run and preserves its evidence. Only child value summaries are new;
parent keys/member identities and same-image/event exclusion remain fixed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

try:
    from .base_runtime import Runtime as BaseRuntime, local_support_ids, validate_raw_labels
    from .common import Example, digest, source_hash, write_json
    from .slots import prepare_slots, slot_pool
except ImportError:
    from base_runtime import Runtime as BaseRuntime, local_support_ids, validate_raw_labels
    from common import Example, digest, source_hash, write_json
    from slots import prepare_slots, slot_pool

RULES = ('capacity_control', 'soft')
VALUES_FORMAT = 'SRAC_SOFT_MEMORY_VALUES_V1'
VALUE_SCHEMA = dict(version=1, parent_identity='unchanged V3 row identity and keys',
    rules=list(RULES), order=['background_or_spatial0', 'building_or_spatial1'],
    feature='FP32 Q-route weighted mean of native F256',
    parent_context='immutable original V3 parent T128, not a child T128',
    route='capacity_control:4x4-grid-checkerboard;soft:Q*(1-B),Q*B;no floor',
    storage='all original parent rows; smoke has explicit written mask')
LABEL_SCHEMA = dict(version=1, classes=['background','intact','damaged','destroyed'],
    rule='area-resize valid one-hot Train GT; route-weighted counts; divide by valid label mass',
    invalid='255 excluded; zero-label-mass child has zero distribution and is masked',
    local_rule='same-forward native grid softmax weighted by identical routes; no query GT',
    reference='independent CPU FP64 area interpolation then all-route scatter-add')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def tree_hash(value):
    """Stable tensor-content hash; independent of torch serialization metadata."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous()
            h.update(b'tensor'); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
            h.update(x.view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(b'list')
            for item in x: visit(item)
        else:
            h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def validate_values(value, parent, *, smoke):
    """Validate identities and actual FP32 contents without silently repairing."""
    n = len(parent['t128'])
    if value.get('format') != VALUES_FORMAT or set(value.get('rules', {})) != set(RULES):
        raise ValueError('Unknown child values format/rules')
    if value.get('value_schema_sha256') != json_hash(VALUE_SCHEMA) or value.get('label_schema_sha256') != json_hash(LABEL_SCHEMA):
        raise ValueError('Child value/label schema mismatch')
    identity = value.get('identity', {})
    for key in ('member_ids','original_rows','images','events','regions'):
        expected = parent[key].detach().cpu()
        actual = identity.get(key)
        if (not torch.is_tensor(actual) or actual.dtype != expected.dtype or
                actual.shape != expected.shape or not torch.equal(actual.cpu(), expected)):
            raise ValueError('Child bank parent identity mismatch: ' + key)
    if identity.get('event_names') != parent['event_names']:
        raise ValueError('Child bank event names mismatch')
    written = value.get('written')
    if not torch.is_tensor(written) or written.dtype != torch.bool or written.shape != (n,):
        raise ValueError('Missing full-parent-row written mask')
    if not bool(written.any()) or (not smoke and not bool(written.all())):
        raise ValueError('Formal child values require every original parent row')
    written = written.cpu()
    for rule in RULES:
        record = value['rules'][rule]
        shapes = {'slot_f':(n,2,256), 'slot_mass':(n,2), 'slot_fraction':(n,2),
                  'slot_labels':(n,2,4), 'label_mass':(n,2), 'slot_valid':(n,2)}
        if set(record) != set(shapes):
            raise ValueError('Unexpected child value fields: ' + rule)
        for name, shape in shapes.items():
            item = record[name]
            dtype = torch.bool if name == 'slot_valid' else torch.float32
            if not torch.is_tensor(item) or item.dtype != dtype or tuple(item.shape) != shape:
                raise ValueError('Child value dtype/shape mismatch: ' + rule + '/' + name)
            if dtype != torch.bool and not bool(torch.isfinite(item).all()):
                raise ValueError('Nonfinite child values: ' + rule + '/' + name)
            if bool((item.cpu()[~written] != 0).any()):
                raise ValueError('Unwritten smoke rows must stay exactly zero')
        mass = record['slot_mass'].cpu(); labels = record['slot_labels'].cpu()
        lm = record['label_mass'].cpu(); frac = record['slot_fraction'].cpu()
        valid = record['slot_valid'].cpu()
        if any(bool((x < 0).any()) for x in (mass, labels, lm, frac)) or bool((frac > 1 + 2e-5).any()):
            raise ValueError('Negative mass/labels or invalid child fraction')
        if bool((lm > mass + 2e-5 + 2e-5*mass).any()):
            raise ValueError('Valid-label mass exceeds assignment mass')
        if not torch.equal(valid, (mass > 0) & (lm > 0)):
            raise ValueError('Child support validity must be positive feature and valid-label mass')
        parent_mass = mass.sum(-1)
        expected_fraction = mass / torch.where(parent_mass > 0, parent_mass, 1)[:,None]
        if not torch.allclose(frac, expected_fraction, atol=2e-5, rtol=2e-5):
            raise ValueError('Child fractions do not sum to parent mass')
        if bool((parent_mass[written] <= 0).any()):
            raise ValueError('Selected parent became empty in fresh extraction')
        if bool((lm.sum(-1)[written] <= 0).any()):
            raise ValueError('A fixed parent lost all valid Train label mass')
        if not torch.allclose(labels.sum(-1)[valid], torch.ones_like(lm[valid]), atol=2e-5, rtol=2e-5):
            raise ValueError('Valid child labels must sum to one')
        if bool((labels[~valid] != 0).any()):
            raise ValueError('Invalid child labels must be exactly zero')
    if not torch.allclose(value['rules'][RULES[0]]['slot_mass'].sum(-1),
                          value['rules'][RULES[1]]['slot_mass'].sum(-1), atol=2e-5, rtol=2e-5):
        raise ValueError('Capacity and soft child masses do not conserve the same parent Q mass')
    return dict(identity=True, fp32=True, labels=True, written=True, mass_conservation=True)


class BuildingCapture:
    """Capture exact B passed to frozen partition; no upsampled-logit inversion."""
    def __init__(self, front):
        self.value = None
        self.calls = 0
        self.handle = front.partition.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if len(inputs) != 2:
            raise RuntimeError('Unexpected frozen partition inputs')
        self.calls += 1
        self.value = inputs[1].detach()

    def clear(self):
        self.value = None
        self.calls = 0

    def take(self):
        if self.calls != 1 or self.value is None:
            raise RuntimeError('Expected exactly one frozen partition building input')
        value = self.value
        self.clear()
        if value.dtype != torch.float32 or value.ndim != 4 or value.shape[:2] != (1,1):
            raise ValueError('Native building probability must be actual FP32 [1,1,H,W]')
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError('Native building probability out of range')
        return value

    def close(self):
        self.handle.remove()


class FrozenSlotRuntime(BaseRuntime):
    """Shared exact forward for build and training; builder needs no child bank."""
    def __enter__(self):
        try:
            super().__enter__()
            self.building_capture = BuildingCapture(self.bundle['front'])
            return self
        except BaseException:
            self.close(*sys.exc_info())
            raise

    def source_batches(self, names):
        """Train-only fixed source order, inference transforms, no shuffle."""
        names = list(names)
        if not names or len(names) != len(set(names)) or any(n not in self._train_ids for n in names):
            raise ValueError('Child value build sources must be unique verified Train images')
        for name in names:
            for p in self.deps.sample_paths(self.bundle['args'].dataset, name).values():
                p = Path(p).resolve()
                if p not in self.input_ledger: self.input_ledger[p] = digest(p)
        return self.deps.full_image_loader(self.bundle['args'].dataset, names, False,
                                          self.args.workers, self.args.seed)

    @torch.no_grad()
    def extract(self, batch):
        try:
            validate_raw_labels(batch)
        except ValueError as error:
            write_json(self.output / 'label_failure.json', dict(phase='cpu_label_validation',
                       error=str(error), prepare_count=self.prepare_count, complete=False))
            raise
        d = self.deps
        pre, post, _loc, target, names = d.parse_batch(batch, self.device)
        if len(pre) != 1 or len(names) != 1:
            raise ValueError('Frozen extraction requires exactly one image')
        name = str(names[0])
        if name not in self._train_ids and name not in self._val_set:
            raise ValueError('Image outside verified Train/Val')
        for p in d.sample_paths(self.bundle['args'].dataset, name).values():
            p = Path(p).resolve()
            if p not in self.input_ledger: self.input_ledger[p] = digest(p)
        self.guard.set_image(name)
        self.native_capture.clear(); self.building_capture.clear()
        result, feature = self.capture(pre, post)
        logits = self.native_capture.take(); building = self.building_capture.take()
        baseline, check = d.native_grid_probq(result, feature, logits, target.shape[-2:])
        if check.get('passed') is not True:
            raise ValueError('Same-pass native probability check failed: ' + name)
        pair = d.extract_pair(result, feature)  # NO query target in keys or features.
        if pair['q_check'].get('passed') is not True:
            raise ValueError('Current frozen Q check failed: ' + name)
        slots = {rule: prepare_slots(feature, result['q'], result['ids'], result['count'],
                                     tuple(result['shape']), building, rule) for rule in RULES}
        self.prepare_count += 1
        return dict(name=name, event=d.event(name), target=target, result=result, feature=feature,
                    pair=pair, native_logits=logits, building=building, slots=slots,
                    base_probability_full=baseline, native_check=check)

    def close(self, exc_type=None, exc=None, tb=None):
        capture = getattr(self, 'building_capture', None)
        if capture is not None:
            capture.close(); self.building_capture = None
        super().close(exc_type, exc, tb)


def validate_selected(ids, bank, *, event_id, image_id):
    n = len(bank['t128'])
    if ids.dtype != torch.long or ids.ndim != 2 or bool(((ids < -1) | (ids >= n)).any()):
        raise ValueError('Parent support IDs out of range')
    chosen = ids[ids >= 0]
    if not bool(bank['eligible'][chosen].all()):
        raise ValueError('An unavailable parent was selected')
    for field, query in (('events', event_id), ('images', image_id)):
        if query >= 0 and bool((bank[field][chosen] == query).any()):
            raise ValueError('Selected a same-event or same-image parent: ' + field)


class Runtime(FrozenSlotRuntime):
    def __enter__(self):
        try:
            super().__enter__()
            if getattr(self.args, 'mode', None) == 'train' and not getattr(self.args, 'smoke', False):
                try:
                    from .gates import validate_baseline
                except ImportError:
                    from gates import validate_baseline
                evidence = validate_baseline(self.args)
                previous = getattr(self.args, 'baseline_evidence', None)
                if previous is not None and previous != evidence:
                    raise ValueError('Baseline evidence changed since runner validation')
                self.args.baseline_evidence = evidence
            evidence = getattr(self.args, 'baseline_evidence', None)
            if evidence is not None:
                for field in ('source_protocol_sha256', 'paired_bank_sha256'):
                    if evidence.get(field) != self.protocol_info[field]:
                        raise ValueError('Baseline evidence uses a different frozen source: ' + field)
                self.protocol_info['baseline_evidence'] = evidence
                for path, sha in evidence['files'].items():
                    path = Path(path).resolve()
                    if digest(path) != sha: raise ValueError('Baseline file changed: ' + str(path))
                    self.input_ledger[path] = sha
            key = getattr(self.args, 'key_name', 't128')
            if key not in ('t128','f256'): raise ValueError('key_name must be t128 or f256')
            self.values_root = Path(self.args.values).expanduser().resolve()
            manifest_path = self.values_root / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('format') != VALUES_FORMAT or manifest.get('complete') is not True:
                raise ValueError('Completed child values manifest required')
            smoke = manifest.get('smoke') is True
            if smoke and not getattr(self.args, 'allow_smoke_values', False):
                raise ValueError('Smoke values require --allow-smoke-values and cannot enter formal training')
            if smoke and not getattr(self.args, 'smoke', False):
                raise ValueError('Formal training cannot use smoke values')
            expected = dict(source_protocol_sha256=self.protocol_info['source_protocol_sha256'],
                paired_bank_sha256=self.protocol_info['paired_bank_sha256'],
                rebuild_protocol_sha256=self.protocol_info['rebuild_protocol_sha256'],
                membership_sha256=self.protocol_info['membership_sha256'],
                code_sha256=source_hash(), value_schema_sha256=json_hash(VALUE_SCHEMA),
                label_schema_sha256=json_hash(LABEL_SCHEMA))
            for field, value in expected.items():
                if manifest.get(field) != value: raise ValueError('Child bank provenance mismatch: ' + field)
            for field in ('label_reference_passed','roundtrip_exact','source_integrity_passed','pool_guard_passed'):
                if manifest.get(field) is not True: raise ValueError('Child bank build gate failed: ' + field)
            self.values_ledger = {}
            for name, sha in manifest.get('file_sha256', {}).items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe child artifact ledger path')
                path = self.values_root / relative
                if digest(path) != sha: raise ValueError('Child artifact changed: ' + name)
                self.values_ledger[path] = sha
            if 'slot_bank.pt' not in manifest.get('file_sha256', {}) or 'images.jsonl' not in manifest.get('file_sha256', {}):
                raise ValueError('Child values ledger lacks required artifacts')
            self.values_ledger[manifest_path] = digest(manifest_path)
            fingerprints = self.values_root / 'input_fingerprints.json'
            if fingerprints.name not in manifest.get('file_sha256', {}):
                raise ValueError('Child bank lacks source image fingerprint ledger')
            for path, sha in json.loads(fingerprints.read_text()).items():
                path = Path(path).resolve()
                if digest(path) != sha: raise ValueError('Child value source image changed: ' + str(path))
                self.input_ledger[path] = sha
            self.values = torch.load(self.values_root / 'slot_bank.pt', map_location=self.device, weights_only=False)
            validate_values(self.values, self.bank, smoke=smoke)
            written = self.values['written']
            if (manifest.get('support_count') != len(self.bank['t128'])
                    or manifest.get('stored_parent_count') != int(written.sum())
                    or manifest.get('image_count') != len(self.values['identity']['images'][written].unique())):
                raise ValueError('Child manifest counts disagree with saved parent identities')
            self.values_before = tree_hash(self.values)
            self.retrieval_bank = dict(self.bank)
            self.retrieval_bank['eligible'] = self.bank['eligible'] & self.values['written'].to(self.device)
            self.protocol_info.update(child_values=str(self.values_root), child_values_sha256=digest(self.values_root/'slot_bank.pt'),
                child_manifest_sha256=digest(manifest_path), child_values_smoke=smoke,
                child_value_schema=VALUE_SCHEMA, child_label_schema=LABEL_SCHEMA, parent_key=key,
                selection_rule='C/D identical parent IDs; A/B identical centroid IDs',
                child_context='original parent T128; fresh child F256',
                support_slot_order=['background_or_spatial0','building_or_spatial1'],
                no_memory_rejection_gate=True)
            return self
        except BaseException:
            self.close(*sys.exc_info())
            raise

    def smoke_names(self):
        old = self.bank
        try:
            self.bank = self.retrieval_bank
            return super().smoke_names()
        finally:
            self.bank = old

    @torch.no_grad()
    def prepare(self, batch):
        data = self.extract(batch)
        d = self.deps; result = data['result']; pair = data['pair']
        name, event = data['name'], data['event']; active = pair['active']
        h,w = result['shape']
        y,x = torch.meshgrid(torch.arange(h,device=self.device,dtype=torch.float32),
                             torch.arange(w,device=self.device,dtype=torch.float32),indexing='ij')
        coords = torch.stack(((y+.5)/h,(x+.5)/w))[None]
        centroid,_ = d.mean_pool(coords,result['q'],result['ids'],result['count'])
        local = local_support_ids(centroid[0],active,self.args.topk)
        eid = self.bank['event_names'].index(event) if event in self.bank['event_names'] else -1
        iid = self._train_ids.get(name,-1)
        key = getattr(self.args,'key_name','t128')
        ids,_ = d.select_neighbors(pair[key].masked_fill(~active[:,None],0), self.retrieval_bank,
            key_name=key,event_id=eid,image_id=iid,topk=self.args.topk,policy='global',chunk_size=self.args.chunk_size)
        ids = ids.masked_fill(~active[:,None],-1)
        validate_selected(ids,self.retrieval_bank,event_id=eid,image_id=iid)
        native_probability = data['native_logits'].float().softmax(1)
        supports = {}; diagnostics = {}
        for rule, local_arm, memory_arm in (('capacity_control','a','c'),('soft','b','d')):
            slot = data['slots'][rule]
            local_labels,_ = slot_pool(native_probability,slot['route_q'],slot['route_ids'],slot['slot_count'])
            query = {k:slot[k] for k in ('slot_f','slot_fraction','slot_valid','route_q','route_ids')}
            supports[local_arm] = dict(**query,support_ids=local,support_f=slot['slot_f'][0],
                support_t=pair['t128'],support_labels=local_labels[0],support_fraction=slot['slot_fraction'][0],
                support_valid=slot['slot_valid'][0] & active[:,None])
            table = self.values['rules'][rule]
            supports[memory_arm] = dict(**query,support_ids=ids,support_f=table['slot_f'],support_t=self.bank['t128'],
                support_labels=table['slot_labels'],support_fraction=table['slot_fraction'],support_valid=table['slot_valid'])
            for arm, chosen in ((local_arm,local),(memory_arm,ids)):
                valid_child = supports[arm]['support_valid']
                selected_child = valid_child[chosen.clamp_min(0)] & (chosen>=0)[...,None]
                diagnostics[arm] = dict(supported_queries=int(selected_child.any(-1).any(-1).sum()),
                    parent_supported_queries=int((chosen>=0).any(-1).sum()), valid_parent_slots=int((chosen>=0).sum()),
                    valid_slots=int(selected_child.sum()), total_parent_slots=chosen.numel(),
                    query_nonempty_slots=int(slot['slot_valid'].sum()))
        common = dict(pixel_features=data['feature'].detach(),region_f=pair['f256'][None],region_t=pair['t128'][None],
            q=result['q'].detach(),ids=result['ids'].detach(),base_probability_full=data['base_probability_full'].detach())
        diagnostic = dict(image=name,event=event,event_id=eid,image_id=iid,active_queries=int(active.sum()),
            total_queries=len(active),native_a1=data['native_check'],q_check=pair['q_check'],source_forward_count=1,
            supports=diagnostics,parent_key=key,query_GT_used_for_supports=False)
        return Example(name,event,data['target'],common,supports,diagnostic)

    def verify(self):
        checks = super().verify()
        checks['child_files'] = all(digest(path)==sha for path,sha in self.values_ledger.items())
        checks['child_values_in_memory'] = tree_hash(self.values)==self.values_before
        if not all(v is True for v in checks.values()):
            raise RuntimeError('Child values changed during training: ' + str(checks))
        return checks
