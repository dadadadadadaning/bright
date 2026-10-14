"""Server bridge to a freshly declared Q source and paired Train bank.

The module can be imported without BRIGHT/CUDA dependencies. Heavy project
imports and source protocol validation occur only on Runtime.__enter__.
No source checkpoint, bank, or failed diagnostic ledger is edited.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from .common import Example, digest, source_hash, write_json
except ImportError:
    from common import Example, digest, source_hash, write_json


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def source_protocol(args):
    """Require an explicit, newly issued Q-source protocol."""
    if not getattr(args, 'protocol', None):
        raise ValueError('A fresh Q-source --protocol is required')
    path = Path(args.protocol).expanduser()
    path = path / 'protocol.json' if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError('Fresh Q-source protocol is missing: ' + str(path))
    return path.resolve()


def _local_module(name):
    return importlib.import_module('.' + name, __package__) if __package__ else importlib.import_module(name)


def _dependencies(repo):
    sys.path.insert(1, str(repo))
    prefix = 'bda_benchmark.experiments.'
    def get(package):
        return importlib.import_module(prefix + package)
    native = get('damageformer_srac_memory_metric_eval_v2.native')
    audit = get('damageformer_srac_stage_diagnostics_v1.memory_audit')
    feature = get('damageformer_srac_stage_diagnostics_v1.feature_core')
    pool = get('damageformer_srac_evidence_followup_v1.native_pool')
    paired = get('damageformer_srac_memory_rebuild_v3.paired')
    core = get('damageformer_srac_memory_rebuild_v3.core')
    region = get('damageformer_srac_region_test_v1.core')
    losses = get('damageformer_srac_fourstage_v2.losses')
    train = importlib.import_module('bda_benchmark.script.standard_ML.train_S5_S8_GPU')
    for module in (native, audit, feature, pool, paired, core, region, losses):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise ValueError('Imported BRIGHT source is outside --repo: ' + str(module.__file__))
    return SimpleNamespace(NativeA1LogitsCapture=native.NativeA1LogitsCapture,
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


def _model_fingerprint(model):
    """Include frozen parameters and buffers, even nonpersistent buffers."""
    h = hashlib.sha256()
    for kind, items in (('parameter', model.named_parameters()),
                        ('buffer', model.named_buffers())):
        for name, value in sorted(items):
            tensor = value.detach().cpu().contiguous()
            h.update(json.dumps((kind, name, str(tensor.dtype), list(tensor.shape)),
                                separators=(',', ':')).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _execution_settings():
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'))


def _source_ledger(bundle):
    """Bind every file the fresh Q loader used, with no inherited E0 lookup."""
    paths, fingerprints = bundle['input_paths'], bundle['input_fingerprints']
    if (not isinstance(paths, dict) or not paths or not isinstance(fingerprints, dict)
            or set(paths) != set(fingerprints)):
        raise ValueError('Fresh Q source input paths/fingerprints are incomplete')
    ledger = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).expanduser().resolve()
        expected = fingerprints[name]
        if (not isinstance(expected, str) or len(expected) != 64 or not path.is_file()
                or digest(path) != expected):
            raise ValueError('Fresh Q source file changed: ' + str(name))
        if path in ledger and ledger[path] != expected:
            raise ValueError('Conflicting fresh Q source fingerprints: ' + str(path))
        ledger[path] = expected
    return ledger


def _bank_ledger(root):
    ledger = {}
    for name in ('paired_bank.pt', 'manifest.json', 'summary.json', 'protocol.json', 'images.jsonl'):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('Fresh paired-bank file missing: ' + str(path))
        ledger[path] = digest(path)
    return ledger


def _allow_smoke_parent(args):
    """Only the smoke run and four-image child build may read a smoke bank."""
    mode = getattr(args, 'mode', None) or getattr(args, 'command', None)
    if mode in ('train', 'full'):
        return False
    return (mode == 'smoke' or getattr(args, 'smoke', False) is True
            or (mode == 'build' and type(getattr(args, 'limit_images', None)) is int
                and args.limit_images == 4))


def _validate_parent_bank(bank, train_names, device):
    required = ('t128', 'f256', 'distribution', 'eligible', 'images', 'events',
                'regions', 'member_ids', 'original_rows')
    if any(key not in bank for key in required):
        raise ValueError('Fresh paired bank lacks a required parent field')
    count = len(bank['t128'])
    if count < 1 or bank['t128'].shape != (count, 128) or bank['f256'].shape != (count, 256) \
            or bank['distribution'].shape != (count, 4):
        raise ValueError('Fresh paired-bank tensor shapes differ')
    for key in ('t128', 'f256', 'distribution'):
        item = bank[key]
        if item.device != device or item.dtype != torch.float32 or not bool(torch.isfinite(item).all()):
            raise ValueError('Fresh parent values must be finite actual FP32: ' + key)
    for key in ('images', 'events', 'regions', 'member_ids', 'original_rows'):
        item = bank[key]
        if item.device != device or item.dtype != torch.long or item.shape != (count,):
            raise ValueError('Fresh parent identity must be long rows: ' + key)
    eligible = bank['eligible']
    names = bank.get('event_names')
    if (eligible.device != device or eligible.dtype != torch.bool
            or eligible.shape != (count,) or not bool(eligible.any())
            or not isinstance(names, list) or not names or len(names) != len(set(names))
            or bool(((bank['images'] < 0) | (bank['images'] >= len(train_names))).any())
            or bool(((bank['events'] < 0) | (bank['events'] >= len(names))).any())):
        raise ValueError('Invalid fresh parent eligibility or Train/event identities')
    labels = bank['distribution'][eligible]
    if (bool((labels < 0).any()) or not torch.allclose(labels.sum(-1),
            torch.ones(len(labels), device=labels.device), atol=2e-5, rtol=2e-5)):
        raise ValueError('Eligible fresh parent labels are invalid')
    return count


class Runtime:
    """One frozen Q-source lifetime shared by the four matched heads."""
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
            if self.args.workers < 0 or self.args.topk not in (8, 64) or self.args.chunk_size < 1:
                raise ValueError('Require workers >= 0, TopK 8 or 64, chunk_size > 0')
            source_module = _local_module('fresh_source')
            bank_module = _local_module('fresh_parent_bank')
            print('Loading and verifying fresh frozen Q source:', protocol, flush=True)
            self.bundle = source_module.load_q_source(protocol, self.output)
            if not isinstance(self.bundle, dict):
                raise ValueError('Fresh Q source loader must return a bundle')
            if self.bundle['source_protocol_sha256'] != digest(protocol):
                raise ValueError('Fresh Q source protocol checksum differs from loaded file')
            self.device = torch.device(self.bundle['device'])
            requested_device = torch.device(self.args.device)
            def device_identity(device):
                return device.type, (device.index or 0) if device.type == 'cuda' else None
            if device_identity(requested_device) != device_identity(self.device):
                raise ValueError('--device must match the fresh Q source protocol')
            source_args = self.bundle['args']
            if not getattr(source_args, 'dataset', None) or getattr(source_args, 'pool_mode', None) != 'native':
                raise ValueError('Fresh Q source requires an explicit dataset and native pooling')
            front = self.bundle['front']
            if (any(module.training for module in front.modules())
                    or any(parameter.requires_grad for parameter in front.parameters())):
                raise ValueError('Fresh Q front must be frozen and in eval mode')
            self.front_before = _model_fingerprint(front)
            self.source_code_before = source_hash()
            self.execution_before = _execution_settings()
            self.input_ledger = _source_ledger(self.bundle)
            self.input_ledger[protocol] = digest(protocol)
            self.train_names = list(self.bundle['train_names'])
            self.val_names = list(self.bundle['val_names'])
            if (len(self.train_names) != 2918 or len(self.val_names) != 450
                    or len(set(self.train_names)) != 2918 or len(set(self.val_names)) != 450
                    or set(self.train_names) & set(self.val_names)):
                raise ValueError('Fresh Q source requires disjoint Train2918/Val450')
            self._train_ids = {name: i for i, name in enumerate(self.train_names)}
            self._val_set = set(self.val_names)
            bank_root = getattr(self.args, 'bank_root', None)
            if bank_root is None:
                raise ValueError('A fresh paired-bank --bank-root is required')
            self.key_root = Path(bank_root).expanduser().resolve()
            print('Verifying fresh FP32 parent bank:', self.key_root, flush=True)
            allow_smoke = _allow_smoke_parent(self.args)
            self.bank, manifest = bank_module.load_bank(self.key_root, self.bundle,
                                                        allow_smoke=allow_smoke)
            if (manifest.get('source_protocol_sha256') != digest(protocol)
                    or manifest.get('complete') is not True
                    or type(manifest.get('smoke')) is not bool
                    or (manifest['smoke'] and not allow_smoke)):
                raise ValueError('Fresh parent bank does not bind this complete Q source')
            self.extra_ledger = _bank_ledger(self.key_root)
            self.input_ledger.update(self.extra_ledger)
            source_images = manifest.get('source_image_fingerprints')
            if not isinstance(source_images, dict) or not source_images:
                raise ValueError('Fresh parent bank has no Train source-image ledger')
            for raw_path, expected in source_images.items():
                path = Path(raw_path).expanduser().resolve()
                if not isinstance(expected, str) or digest(path) != expected:
                    raise ValueError('Fresh parent source image changed: ' + str(path))
                self.input_ledger[path] = expected
            archived = manifest.get('archived_files_sha256')
            if not isinstance(archived, dict) or not archived:
                raise ValueError('Fresh parent bank lacks archived member-identity ledger')
            archive_root = Path(manifest['archived_membership_root']).expanduser().resolve()
            if (archive_root / 'keys').is_dir() and not (archive_root / 'paired_bank.pt').is_file():
                archive_root /= 'keys'
            for name, expected in archived.items():
                if Path(name).name != name or not isinstance(expected, str):
                    raise ValueError('Invalid archived member-identity ledger entry')
                path = archive_root / name
                if not path.is_file() or digest(path) != expected:
                    raise ValueError('Archived member-identity artifact changed: ' + str(path))
                self.input_ledger[path] = expected
            support_count = _validate_parent_bank(self.bank, self.train_names, self.device)
            if manifest.get('support_count') != support_count:
                raise ValueError('Fresh parent support count differs from manifest')
            print('Fresh bank verified: Train=%d Val=%d supports=%d; Q source files unchanged.' %
                  (len(self.train_names), len(self.val_names), len(self.bank['t128'])), flush=True)
            self.bank_before = _bank_fingerprint(self.bank)
            self.protocol_info = dict(
                source_protocol=str(protocol), source_protocol_sha256=digest(protocol),
                source_provenance=self.bundle['source_provenance'],
                source_checkpoint_sha256=self.bundle['source_checkpoint_sha256'],
                original_front_sha256=self.bundle['original_front_sha256'],
                a1_sha256=self.bundle['a1_sha256'],
                execution_settings=self.execution_before,
                runtime_source_sha256=self.source_code_before,
                paired_bank=str(self.key_root / 'paired_bank.pt'),
                paired_bank_sha256=manifest['paired_bank_sha256'],
                rebuild_protocol_sha256=manifest['rebuild_protocol_sha256'],
                membership_sha256=manifest['membership_sha256'], support_count=manifest['support_count'],
                support_source_image_count=manifest['image_count'],
                archived_membership_sha256=manifest['archived_membership_sha256'],
                parent_bank_smoke=manifest['smoke'],
                train_count=len(self.train_names), val_count=len(self.val_names),
                historical_e0_reproduction_claim=False,
                old_three_arm_baseline='not_comparable_new_Q_source',
                key_storage='torch.float32', retrieval='global_cosine_exclude_same_image_and_event',
                r0_rule='nearest active Q-centroids including self; squared unit-grid distance, stable region-ID ties',
                r0_include_self=True,
                r0_labels='same-forward native A1 grid softmax Q-weighted mean; no query GT',
                train_query_rule='entire Train list, frozen full-image loader; no crop/augmentation',
                test_accessed=False)
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
        checks = dict(
            source_files=all(path.is_file() and digest(path) == sha
                             for path, sha in self.input_ledger.items()),
            source_code=source_hash() == self.source_code_before,
            frozen_front=_model_fingerprint(self.bundle['front']) == self.front_before,
            frozen_front_eval=all(not module.training for module in self.bundle['front'].modules()),
            frozen_front_no_grad=all(not parameter.requires_grad
                                     for parameter in self.bundle['front'].parameters()),
            execution_settings=_execution_settings() == self.execution_before)
        checks['paired_bank_in_memory'] = _bank_fingerprint(self.bank) == self.bank_before
        checks['pool_guard'] = not self.guard.failures and not self.guard.restore_conflicts
        if not all(value is True for value in checks.values()):
            raise RuntimeError('Q-rebased training changed frozen inputs/runtime: ' + str(checks))
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
