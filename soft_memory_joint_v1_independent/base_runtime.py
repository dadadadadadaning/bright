"""Server bridge to the unchanged V3 frozen source and completed paired Train bank.

The module can be imported without BRIGHT/CUDA dependencies. Heavy project
imports and source protocol validation occur only on Runtime.__enter__.
No archived source file, checkpoint, bank or failed diagnostic ledger is edited.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from .common import Example, digest, write_json
except ImportError:
    from common import Example, digest, write_json

SOURCE_PROTOCOL = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json')


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def source_protocol(args):
    """The keys/protocol.json is a new-bank record, never the E0 source anchor."""
    if getattr(args, 'protocol', None):
        path = Path(args.protocol).expanduser()
    else:
        ledger = Path(args.v3_root).resolve() / 'pipeline_state.json'
        record = _read(ledger) if ledger.is_file() else {}
        path = Path(record.get('protocol') or SOURCE_PROTOCOL)
    path = path / 'protocol.json' if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError('Source E0 protocol is missing; use --protocol: ' + str(path))
    return path.resolve()


def _dependencies(repo):
    sys.path.insert(1, str(repo))
    prefix = 'bda_benchmark.experiments.'
    def get(package):
        return importlib.import_module(prefix + package)
    shared = get('damageformer_srac_memory_rebuild_v3.shared')
    if not Path(shared.__file__).resolve().is_relative_to(repo):
        raise ValueError('Imported V3 source is outside --repo')
    native = get('damageformer_srac_memory_metric_eval_v2.native')
    audit = get('damageformer_srac_stage_diagnostics_v1.memory_audit')
    feature = get('damageformer_srac_stage_diagnostics_v1.feature_core')
    pool = get('damageformer_srac_evidence_followup_v1.native_pool')
    paired = get('damageformer_srac_memory_rebuild_v3.paired')
    core = get('damageformer_srac_memory_rebuild_v3.core')
    region = get('damageformer_srac_region_test_v1.core')
    losses = get('damageformer_srac_fourstage_v2.losses')
    train = importlib.import_module('bda_benchmark.script.standard_ML.train_S5_S8_GPU')
    return SimpleNamespace(shared=shared, NativeA1LogitsCapture=native.NativeA1LogitsCapture,
        native_grid_probq=native.native_grid_probq, FrozenFrontCapture=audit.FrozenFrontCapture,
        parse_batch=audit.parse_batch, full_image_loader=audit.full_image_loader,
        event=audit.event, mean_pool=feature.mean_pool, PoolGuard=pool.PoolGuard,
        extract_pair=paired.extract_pair, select_neighbors=core.select_neighbors,
        sample_paths=region.sample_paths, region_nll=losses.region_nll,
        lovasz_softmax=train.lovasz_softmax)


@torch.no_grad()
def local_support_ids(centroids, active, topk):
    """Nearest current-image regions by Q centroid, including self; stable ID ties.

    Centroids [R,2] and active [R] are feature/Q metadata, never ground truth.
    Inactive queries/supports are excluded; an active query itself is eligible.
    Empty positions are -1 so gather_support returns zero, masked padding.
    """
    if centroids.ndim != 2 or centroids.shape[1] != 2 or centroids.dtype != torch.float32:
        raise ValueError('centroids must be actual FP32 [R,2]')
    count = centroids.shape[0]
    if (active.shape != (count,) or active.dtype != torch.bool
            or active.device != centroids.device or count < 1
            or not bool(torch.isfinite(centroids).all())
            or type(topk) is not int or topk < 1):
        raise ValueError('Invalid centroid/active/TopK inputs')
    distance = (centroids[:, None] - centroids[None, :]).square().sum(-1)
    allowed = active[:, None] & active[None, :]
    distance = distance.masked_fill(~allowed, torch.inf)
    selected = torch.argsort(distance, dim=1, stable=True)[:, :min(topk, count)]
    valid = distance.gather(1, selected).isfinite()
    result = torch.full((count, topk), -1, device=centroids.device, dtype=torch.long)
    result[:, :selected.shape[1]] = torch.where(valid, selected, -1)
    return result


@torch.no_grad()
def gather_support(ids, f256, t128, labels, *, eligible=None,
                   events=None, images=None, event_id=None, image_id=None):
    """Gather paired candidate values and verify selection before any indexing.

    Same IDs gather both features and soft labels. No query target is accepted.
    Train/Val identity exclusion is checked a second time at the reader boundary.
    """
    if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
        raise ValueError('Support IDs must be nonempty long [R,K]')
    n = f256.shape[0]
    if (n < 1 or f256.shape != (n, 256) or t128.shape != (n, 128)
            or labels.shape != (n, 4)):
        raise ValueError('Paired support feature/label shapes differ')
    if any(value.device != ids.device for value in (f256, t128, labels)):
        raise ValueError('IDs and support tensors must share device')
    if any(value.dtype != torch.float32 for value in (f256, t128, labels)):
        raise ValueError('Support values must remain actual FP32')
    if bool((ids < -1).any()) or bool((ids >= n).any()):
        raise ValueError('Support ID outside bank; only -1 padding is allowed')
    valid = ids >= 0
    selected = ids[valid]
    if eligible is not None and selected.numel() and not bool(eligible[selected].all()):
        raise ValueError('An ineligible support was selected')
    for metadata, query_id, name in ((events, event_id, 'event'), (images, image_id, 'image')):
        if metadata is not None and query_id is not None and query_id >= 0:
            if bool((metadata[selected] == query_id).any()):
                raise ValueError('Selected support shares query ' + name)
    safe = ids.clamp_min(0)
    def gather(value):
        return torch.where(valid[..., None], value[safe], 0).unsqueeze(0)
    output = dict(support_f=gather(f256), support_t=gather(t128),
                  support_labels=gather(labels), support_valid=valid.unsqueeze(0))
    if any(not bool(torch.isfinite(output[name]).all())
           for name in ('support_f', 'support_t', 'support_labels')):
        raise ValueError('Nonfinite valid support content')
    distributions = output['support_labels'][output['support_valid']]
    if (bool((distributions < 0).any()) or not torch.allclose(distributions.sum(-1),
            torch.ones_like(distributions[:, 0]), atol=2e-5, rtol=2e-5)):
        raise ValueError('Selected support label distribution is invalid')
    return output


def validate_raw_labels(batch):
    """Reject fractional/overflowing labels before inherited long conversion.

    This validates label *format*, never uses labels to construct a feature,
    select a support or decide which valid images enter training/evaluation.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) < 5:
        raise ValueError('Expected BRIGHT pre/post/location/damage/names batch')
    names = batch[4]
    for index, field, classes in ((2, 'location', 2), (3, 'damage', 4)):
        value = torch.as_tensor(batch[index]).detach().cpu()
        if value.ndim != 3 or value.shape[0] != len(names):
            raise ValueError('Expected raw ' + field + ' label [B,H,W]')
        if value.is_complex():
            raise ValueError('Complex raw ' + field + ' label')
        finite = torch.isfinite(value)
        integral = value == value.round() if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
        legal = finite & integral & (((value >= 0) & (value < classes)) | (value == 255))
        if not bool(legal.all()):
            invalid = torch.unique(value[~legal])[:20].tolist()
            raise ValueError('Invalid raw ' + field + ' labels before integer cast: ' + repr(invalid))
    return True


def _bank_fingerprint(bank):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(bank):
        value = bank[key]
        h.update(key.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


class Runtime:
    """One frozen-source lifetime shared by all three separately trained heads."""
    def __init__(self, args, output):
        self.args, self.output = args, Path(output).resolve()
        self.bundle = self.guard = self.capture = self.native_capture = None
        self.guard_report = None
        self._entered = False
        self.input_ledger = {}
        self.prepare_count = 0

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Runtime cannot be entered twice')
        self._entered = True
        try:
            repo = Path(self.args.repo).expanduser().resolve()
            self.deps = _dependencies(repo)
            d = self.deps
            protocol = source_protocol(self.args)
            record = _read(protocol)
            requested_device = torch.device(self.args.device)
            actual_device = torch.device(record['arguments']['device'])
            def device_identity(device):
                return device.type, (device.index or 0) if device.type == 'cuda' else None
            if device_identity(requested_device) != device_identity(actual_device):
                raise ValueError('--device must match the immutable source E0 protocol')
            if self.args.workers < 0 or self.args.topk not in (8, 64) or self.args.chunk_size < 1:
                raise ValueError('Require workers >= 0, TopK 8 or 64, chunk_size > 0')
            self.own_before = d.shared.source_hash()
            print('Loading and verifying frozen E0 source:', protocol, flush=True)
            self.bundle = d.shared.load_bundle(protocol, self.output)
            self.device = self.bundle['device']
            self.train_names = list(self.bundle['train_names'])
            self.val_names = list(self.bundle['val_names'])
            self._train_ids = {name: i for i, name in enumerate(self.train_names)}
            self._val_set = set(self.val_names)
            self.key_root = Path(self.args.v3_root).expanduser().resolve() / 'keys'
            self.extra_ledger = d.shared.paired_ledger(self.key_root)
            # A completed formal bank is a standalone artifact. Old diagnose
            # ledger status is recorded, never edited or claimed as passed.
            print('Verifying completed V3 FP32 bank:', self.key_root, flush=True)
            self.bank, manifest = d.shared.load_paired_bank(self.key_root, self.bundle, allow_smoke=False)
            print('V3 bank verified: Train=%d Val=%d supports=%d; source files unchanged.' %
                  (len(self.train_names), len(self.val_names), len(self.bank['t128'])), flush=True)
            self.bank_before = _bank_fingerprint(self.bank)
            observations = {}
            for rel in ('pipeline_state.json', 'diagnose/summary.json', 'diagnose/result.json'):
                path = Path(self.args.v3_root) / rel
                if path.is_file():
                    data = _read(path)
                    observations[rel] = {key: data[key] for key in (
                        'complete', 'execution_complete', 'full_validation_evidence',
                        'phase', 'current', 'image_count', 'error') if key in data}
                else:
                    observations[rel] = {'exists': False}
            self.protocol_info = dict(
                source_protocol=str(protocol), source_protocol_sha256=digest(protocol),
                source_runtime_settings=self.bundle['runtime_settings'],
                source_hashes=self.bundle['hashes'], source_head_hashes=self.bundle['head_hashes'],
                inherited_source_sha256=self.bundle['own_source_sha256'], v3_source_sha256=self.own_before,
                paired_bank=str(self.key_root / 'paired_bank.pt'),
                paired_bank_sha256=manifest['paired_bank_sha256'],
                rebuild_protocol_sha256=manifest['rebuild_protocol_sha256'],
                membership_sha256=manifest['membership_sha256'], support_count=manifest['support_count'],
                support_source_image_count=manifest['image_count'],
                train_count=len(self.train_names), val_count=len(self.val_names),
                historical_reproduction_claim=False,
                key_storage='torch.float32', retrieval='global_cosine_exclude_same_image_and_event',
                r0_rule='nearest active Q-centroids including self; squared unit-grid distance, stable region-ID ties',
                r0_include_self=True,
                r0_labels='same-forward native A1 grid softmax Q-weighted mean; no query GT',
                train_query_rule='entire Train list, frozen full-image loader; no crop/augmentation',
                test_accessed=False)
            self.diagnostic_status = observations
            self.capture = d.FrozenFrontCapture(self.bundle['front'])
            self.native_capture = d.NativeA1LogitsCapture(self.bundle['front'])
            self.guard = d.PoolGuard(self.output / 'pool_debug', mode=self.bundle['args'].pool_mode,
                                     capture='failure', check_reference=False)
            self.guard.__enter__()
            return self
        except BaseException:
            self.close(*sys.exc_info())
            raise

    def batches(self, names, train=False, seed=3407):
        names = list(names)
        if not names or len(names) != len(set(names)):
            raise ValueError('Loader names must be a nonempty unique list')
        allowed = self._train_ids if train else self._val_set
        if any(name not in allowed for name in names):
            raise ValueError('Requested loader crosses the Train/Val split')
        # Bind image contents before the loader can read them; retain one hash
        # per first observation and verify all images again after training.
        for name in names:
            for path in self.deps.sample_paths(self.bundle['args'].dataset, name).values():
                path = Path(path).resolve()
                if path not in self.input_ledger:
                    self.input_ledger[path] = digest(path)
        return self.deps.full_image_loader(self.bundle['args'].dataset, names,
            train, self.args.workers, seed)

    def smoke_names(self):
        """Pick distinct events with legal Train supports, without reading GT."""
        def legal(name):
            event = self.deps.event(name)
            eid = self.bank['event_names'].index(event) if event in self.bank['event_names'] else -1
            mask = self.bank['eligible'] & (self.bank['events'] != eid)
            if name in self._train_ids:
                mask &= self.bank['images'] != self._train_ids[name]
            return bool(mask.any())
        def choose(names, count):
            chosen, events = [], set()
            for name in names:
                event = self.deps.event(name)
                if event not in events and legal(name):
                    chosen.append(name); events.add(event)
                if len(chosen) == count:
                    return chosen
            raise ValueError('Smoke requires legal cross-event supports and distinct source events')
        return choose(self.train_names, 4), choose(self.val_names, 2)

    @torch.no_grad()
    def prepare(self, batch):
        d = self.deps
        try:
            validate_raw_labels(batch)
        except ValueError as error:
            write_json(self.output / 'label_failure.json', dict(phase='cpu_label_validation',
                       error=str(error), prepare_count=self.prepare_count, complete=False))
            raise
        pre, post, _loc, target, names = d.parse_batch(batch, self.device)
        if len(pre) != 1 or len(names) != 1:
            raise ValueError('The frozen paired extraction requires image batch size 1')
        name = str(names[0])
        if name not in self._train_ids and name not in self._val_set:
            raise ValueError('Query is absent from verified Train/Val lists')
        # Direct prepare is supported, but its source must also be fingerprinted.
        for path in d.sample_paths(self.bundle['args'].dataset, name).values():
            path = Path(path).resolve()
            if path not in self.input_ledger:
                self.input_ledger[path] = digest(path)
        self.guard.set_image(name)
        self.native_capture.clear()
        result, feature = self.capture(pre, post)
        native_logits = self.native_capture.take()
        baseline, native_check = d.native_grid_probq(result, feature, native_logits, target.shape[-2:])
        if native_check.get('passed') is not True:
            raise ValueError('Same-pass native A1 probability check failed: ' + name)
        pair = d.extract_pair(result, feature)  # Deliberately no query target.
        if pair['q_check'].get('passed') is not True:
            raise ValueError('Current Q/paired extraction check failed: ' + name)
        active = pair['active']
        height, width = result['shape']
        # Q-weighted geometric centroids and current-image predictions share
        # the established exact-positive-mass grouped mean operator.
        y, x = torch.meshgrid(torch.arange(height, device=self.device, dtype=torch.float32),
                             torch.arange(width, device=self.device, dtype=torch.float32), indexing='ij')
        coordinates = torch.stack(((y + .5) / height, (x + .5) / width), dim=0).unsqueeze(0)
        centroid, _ = d.mean_pool(coordinates, result['q'], result['ids'], result['count'])
        local_labels, _ = d.mean_pool(native_logits.float().softmax(1), result['q'], result['ids'], result['count'])
        local_ids = local_support_ids(centroid[0], active, self.args.topk)
        supports = {'r0': gather_support(local_ids, pair['f256'], pair['t128'], local_labels[0], eligible=active)}
        event_name = d.event(name)
        event_id = self.bank['event_names'].index(event_name) if event_name in self.bank['event_names'] else -1
        image_id = self._train_ids.get(name, -1)  # -1 is verified held-out Val.
        selected = {'r0': local_ids}
        for arm, key in (('rt', 't128'), ('rf', 'f256')):
            query = pair[key].masked_fill(~active[:, None], 0)
            ids, _scores = d.select_neighbors(query, self.bank, key_name=key, event_id=event_id,
                image_id=image_id, topk=self.args.topk, policy='global', chunk_size=self.args.chunk_size)
            ids = ids.masked_fill(~active[:, None], -1)
            supports[arm] = gather_support(ids, self.bank['f256'], self.bank['t128'], self.bank['distribution'],
                eligible=self.bank['eligible'], events=self.bank['events'], images=self.bank['images'],
                event_id=event_id, image_id=image_id)
            selected[arm] = ids
        self.prepare_count += 1
        diagnostics = dict(image=name, event=event_name, event_id=event_id, image_id=image_id,
            active_queries=int(active.sum()), total_queries=len(active), native_a1=native_check,
            q_check=pair['q_check'], source_forward_count=1,
            supports={arm: dict(supported_queries=int((ids >= 0).any(-1).sum()),
                valid_slots=int((ids >= 0).sum()), total_slots=ids.numel()) for arm, ids in selected.items()})
        common = dict(pixel_features=feature.detach(), region_f=pair['f256'].unsqueeze(0),
            region_t=pair['t128'].unsqueeze(0), q=result['q'].detach(), ids=result['ids'].detach(),
            base_probability_full=baseline.detach())
        return Example(name, event_name, target, common, supports, diagnostics)

    def loss(self, probability, target):
        if not bool((target != 255).any()):
            return probability.sum() * 0
        loss = self.deps.region_nll(probability, target)
        return loss + self.deps.lovasz_softmax(probability, target, classes='present', ignore=255)

    def verify(self):
        checks = self.deps.shared.verify_all(self.bundle, self.own_before,
                    {**self.extra_ledger, **self.input_ledger})
        checks['paired_bank_in_memory'] = _bank_fingerprint(self.bank) == self.bank_before
        checks['pool_guard'] = not self.guard.failures and not self.guard.restore_conflicts
        if not all(value is True for value in checks.values()):
            raise RuntimeError('Joint training changed frozen inputs/runtime: ' + str(checks))
        return checks

    def close(self, exc_type=None, exc=None, tb=None):
        failures = []
        try:
            if self.guard is not None and getattr(self.guard, '_entered', False) and not self.guard.restored:
                try:
                    self.guard.__exit__(exc_type, exc, tb)
                except Exception as error:
                    failures.append('PoolGuard close: ' + repr(error))
                self.guard_report = self.guard.report()
        finally:
            for name in ('native_capture', 'capture'):
                handle = getattr(self, name, None)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception as error:
                        failures.append(name + ' close: ' + repr(error))
                    finally:
                        setattr(self, name, None)
        if failures:
            self.guard_report = {**(self.guard_report or {}), 'cleanup_failures': failures}
            if exc_type is None:
                raise RuntimeError('Runtime cleanup failed: ' + '; '.join(failures))

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type, exc, tb)
        if exc_type is None and self.guard_report is not None:
            if self.guard_report.get('restored') is not True or self.guard_report.get('restore_conflicts'):
                raise RuntimeError('PoolGuard did not restore inherited runtime hooks')
        return False
