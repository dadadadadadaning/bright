"""Strict provenance and shared I/O; never mutate source runs or access Test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from bda_benchmark.experiments.damageformer_roi_exchange_v1 import core as old
from bda_benchmark.experiments.damageformer_roi_exchange_v1.audit import validate_checkpoints, load_heads
from bda_benchmark.experiments.damageformer_roi_exchange_v1.train import prepare_device, state_digest
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import (
    load_fixed, lists, loader, parse_batch, metrics, cm_update, seed_everything,
    new_output, save_json, save_pt, digest, source_hash, event,
)
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective
from .routing import route_from_q

capture = old.capture


def experiment_hash():
    h = hashlib.sha256(old.implementation_hash().encode())
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file() and p.suffix in ('.py', '.sh'):
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()


def add_arguments(parser, training=False):
    for key in ('checkpoint', 'run_root', 'dataset', 'train_list', 'val_list', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--source_head', choices=('hard', 'exchange'), default='hard')
    parser.add_argument('--pool_mode', choices=('hard', 'exchange'), default='hard')
    parser.add_argument('--read_mode', choices=('hard', 'exchange'), default='hard')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    if training:
        parser.add_argument('--steps', type=int, default=5000)
        parser.add_argument('--eval_every', type=int, default=500)
        parser.add_argument('--batch', type=int, default=2)
        parser.add_argument('--accum', type=int, default=8)
        parser.add_argument('--seed', type=int, default=3407)
        parser.add_argument('--lr', type=float, default=1e-4)
    return parser


def validate_training_args(args):
    if min(args.steps, args.eval_every, args.batch, args.accum) <= 0 or args.batch*args.accum != 16:
        raise ValueError('Positive budgets and single-device batch*accum=16 required')
    if not math.isfinite(args.lr) or args.lr <= 0 or args.workers < 0:
        raise ValueError('Invalid lr/workers')
    if args.smoke and (args.steps != 20 or args.eval_every != 20):
        raise ValueError('Smoke requires steps=20 and eval_every=20')


def load_source(args, names=None):
    """Load immutable SAME-STEP final heads of a completed ROI_EXCHANGE_V1 run.

    This intentionally rejects historical banks, A1 weights in place of the
    V2 front, best checkpoints from different steps, and modified source code.
    """
    if args.workers < 0:
        raise ValueError('Workers must be nonnegative')
    names = list(names or [args.source_head])
    if not names or len(set(names)) != len(names) or any(n not in ('hard', 'exchange') for n in names):
        raise ValueError('Expected unique hard/exchange source heads')
    root = Path(args.run_root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'ROI_EXCHANGE_V1':
        raise ValueError('Use completed ROI_EXCHANGE_V1 run, not old Memory/factorial cache')
    if bool(manifest.get('smoke')) != bool(summary.get('smoke')):
        raise ValueError('Inconsistent source smoke markers')
    if manifest.get('smoke') and not args.smoke:
        raise ValueError('Cannot use smoke-trained source for a formal experiment')
    train, val = lists(args.train_list, args.val_list)
    paths = {n: root/(n+'_last.pth') for n in names}
    file_hashes = {str(p): digest(p) for p in [Path(args.checkpoint), root/'manifest.json', root/'summary.json', Path(args.train_list), Path(args.val_list), *paths.values()]}
    hashes = dict(checkpoint_sha=digest(args.checkpoint), train_sha=digest(args.train_list),
                  val_sha=digest(args.val_list), frontend_hash=source_hash(), experiment_hash=old.implementation_hash())
    states = {n: torch.load(p, map_location='cpu', weights_only=False) for n, p in paths.items()}
    final_step = validate_checkpoints(manifest, summary, states, **hashes)
    device = prepare_device(args.device)
    front, ckpt, sha = load_fixed(args.checkpoint, device)
    if sha != hashes['checkpoint_sha'] or ckpt['step'] != manifest['checkpoint_step']:
        raise ValueError('Front identity mismatch')
    if any(front.config[k] != manifest[k] for k in ('dim', 'stride')):
        raise ValueError('Front configuration mismatch')
    heads = load_heads(states, manifest['dim'], device, old.Readout)
    before = dict(front=state_digest(front), **{n: state_digest(h) for n, h in heads.items()})
    if before['front'] != manifest['initial_front_state_sha256']:
        raise ValueError('Front state differs from source training')
    provenance = dict(source_run=str(root), source_head_step=final_step,
                      source_hashes=hashes, source_file_sha256=file_hashes,
                      head_file_sha256={n: digest(p) for n,p in paths.items()},
                      before_state_sha256=before, followup_hash=experiment_hash(),
                      test_accessed=False)
    return dict(front=front, heads=heads, manifest=manifest, summary=summary,
                provenance=provenance, train=train, val=val, device=device)


def routes(reference, manifest, pool_mode, read_mode):
    args = manifest['arguments']
    def one(mode):
        if mode not in ('hard', 'exchange'):
            raise ValueError(mode)
        return route_from_q(reference['q'], reference['ids'], reference['tokens'].shape[1],
                            reference['shape'], reference['building'],
                            epsilon=0. if mode == 'hard' else args['epsilon'],
                            threshold=args['threshold'])
    return one(pool_mode), one(read_mode)


def source_identity(bundle, args):
    p = bundle['provenance']
    return dict(front_sha256=p['source_hashes']['checkpoint_sha'],
                readout_sha256=p['head_file_sha256'][args.source_head],
                source_head=args.source_head, source_step=p['source_head_step'],
                source_front_hash=p['source_hashes']['frontend_hash'],
                source_experiment_hash=p['source_hashes']['experiment_hash'],
                followup_hash=p['followup_hash'],
                train_sha256=p['source_hashes']['train_sha'], val_sha256=p['source_hashes']['val_sha'],
                pool_mode=args.pool_mode, read_mode=args.read_mode,
                epsilon=bundle['manifest']['arguments']['epsilon'],
                threshold=bundle['manifest']['arguments']['threshold'],
                dim=bundle['manifest']['dim'], stride=bundle['manifest']['stride'],
                feature_source='original pixel-head input including existing SoftROI',
                bank_extraction_resolution=1024, train_crop_resolution=512,
                gradient_policy='front and chosen region embedding frozen; fresh classifiers only')


def assert_source(bundle):
    p = bundle['provenance']
    modules = dict(front=bundle['front'], **bundle['heads'])
    for name, model in modules.items():
        if model.training or any(x.requires_grad or x.grad is not None for x in model.parameters()):
            raise RuntimeError('Source model not frozen: '+name)
        if state_digest(model) != p['before_state_sha256'][name]:
            raise RuntimeError('Source parameters/buffers changed: '+name)
    for path, expected in p['source_file_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source file changed: '+path)
    if source_hash() != p['source_hashes']['frontend_hash'] or old.implementation_hash() != p['source_hashes']['experiment_hash'] or experiment_hash() != p['followup_hash']:
        raise RuntimeError('Code changed during experiment; do not bypass provenance checks')


def finish_output(out, summary):
    save_json(Path(out)/'summary.json', dict(summary, complete=True, test_accessed=False))
