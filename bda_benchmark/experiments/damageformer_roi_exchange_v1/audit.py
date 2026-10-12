"""Frozen, same-step validation audit of last readouts; no training or Test access."""
import argparse
import json
import math
from pathlib import Path
import torch
from .diagnostics import DiagnosticAccumulator
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1 import core as factorial
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.train import prepare_device, state_digest, assert_front
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import (
    load_fixed, lists, loader, parse_batch, new_output, save_json, digest, source_hash,
)


def implementation(manifest):
    if manifest.get('format') == 'ROI_PARTITION_FACTORIAL_V1':
        return factorial
    if manifest.get('format') == 'ROI_EXCHANGE_V1':
        from . import core
        return core
    raise ValueError('Unsupported run format')


def validate_checkpoints(manifest, summary, states, checkpoint_sha, train_sha, val_sha, frontend_hash, experiment_hash):
    """Pure validation, also exercised on CPU without the dataset or front."""
    if manifest.get('format') not in ('ROI_PARTITION_FACTORIAL_V1', 'ROI_EXCHANGE_V1'):
        raise ValueError('Unsupported manifest format')
    if summary.get('complete') is not True or summary.get('test_accessed') is not False or manifest.get('test_accessed') is not False:
        raise ValueError('Require complete Train/Val-only run')
    for key, expected in dict(checkpoint_sha256=checkpoint_sha, train_sha256=train_sha,
                              val_sha256=val_sha, frontend_source_hash=frontend_hash,
                              experiment_hash=experiment_hash).items():
        if manifest.get(key) != expected:
            raise ValueError('Provenance mismatch: '+key)
    if not states or not isinstance(summary.get('final_step'), int) or summary['final_step'] <= 0:
        raise ValueError('Missing same-step final readouts')
    for name, state in states.items():
        if state.get('name') != name or state.get('step') != summary['final_step']:
            raise ValueError('Readout name or final step mismatch: '+name)
        if state.get('manifest') != manifest:
            raise ValueError('Embedded head manifest differs from run manifest: '+name)
        weights = state.get('head')
        if not isinstance(weights, dict) or not weights:
            raise ValueError('Missing readout state: '+name)
        if any(not torch.is_tensor(t) or not torch.isfinite(t).all() for t in weights.values()):
            raise ValueError('Invalid readout tensor: '+name)
    return summary['final_step']


def load_heads(states, dim, device, readout_cls=None):
    cls = readout_cls or factorial.Readout
    heads = {}
    for name, state in states.items():
        head = cls(dim).to(device)
        head.load_state_dict(state['head'], strict=True)
        head.requires_grad_(False).eval()
        heads[name] = head
    return heads


def verify_reproduction(actual, expected, tolerance=1e-5):
    for name, values in actual.items():
        if name not in expected:
            raise ValueError('Missing final reference metrics: '+name)
        for key in ('valid_miou', 'iou', 'precision', 'recall'):
            observed, reference = values[key], expected[name][key]
            a = observed if isinstance(observed,list) else [observed]
            b = reference if isinstance(reference,list) else [reference]
            if len(a)!=len(b):
                raise ValueError('Reference metric shape mismatch')
            for x,y in zip(a,b):
                if x is None or y is None:
                    if x is not None or y is not None:
                        raise ValueError('Reference absent class mismatch: '+name)
                elif not math.isfinite(x) or not math.isfinite(y) or abs(x-y)>tolerance:
                    raise ValueError('Final metric reproduction failed: '+name+'/'+key)


@torch.no_grad()
def evaluate(model, heads, data, device, manifest, module):
    threshold = manifest['arguments']['threshold']
    pairs = [('s0', 's2'), ('hard', 'exchange')]
    pairs = [(a,b) for a,b in pairs if a in heads and b in heads]
    pairs += [('old_pixel', k) for k in heads]
    accumulator = DiagnosticAccumulator(threshold, pairs=pairs)
    model.eval()
    for head in heads.values():
        head.eval()
    for batch in data:
        pre, post, _, gt, names = parse_batch(batch, device)
        reference = factorial.capture(model, pre, post)
        predictions = {'old_pixel': reference['pixel'].argmax(1), 'old_region': reference['prob'].argmax(1)}
        for name, head in heads.items():
            kwargs = dict(threshold=threshold, stride=model.config['stride'])
            if manifest['format'] == 'ROI_EXCHANGE_V1':
                kwargs['epsilon'] = manifest['arguments']['epsilon']
            routing = module.partition(reference, name, **kwargs)
            predictions[name] = head(reference['features']['roi'], routing, gt.shape[-2:]).argmax(1)
        accumulator.update(gt, predictions, reference['building'], names)
        if len(accumulator.images)%25==0:
            print('audit evaluate',len(accumulator.images),flush=True)
    return accumulator.finalize()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'run_root', 'output', 'dataset', 'train_list', 'val_list'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--names', nargs='+', default=['s0','s2'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 0 or not args.names or len(set(args.names)) != len(args.names):
        parser.error('Invalid workers or duplicate names')
    if any(not name.replace('_','').isalnum() for name in args.names):
        parser.error('Names must be simple identifiers')
    root = Path(args.run_root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    module = implementation(manifest)
    allowed = ('hard','exchange') if manifest['format']=='ROI_EXCHANGE_V1' else factorial.VARIANTS
    if any(name not in allowed for name in args.names):
        parser.error('Names do not belong to this experiment format')
    _, val = lists(args.train_list, args.val_list)
    hashes = dict(checkpoint_sha=digest(args.checkpoint), train_sha=digest(args.train_list),
                  val_sha=digest(args.val_list), frontend_hash=source_hash(), experiment_hash=module.implementation_hash())
    audit_files = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    paths = {name:root/(name+'_last.pth') for name in args.names}
    head_files = {name:digest(path) for name,path in paths.items()}
    states = {name:torch.load(path,map_location='cpu',weights_only=False) for name,path in paths.items()}
    final_step = validate_checkpoints(manifest,summary,states,**hashes)
    if manifest.get('smoke') and not args.smoke:
        raise ValueError('Smoke-trained heads cannot produce a full-run audit')
    device = prepare_device(args.device)
    model, front_checkpoint, sha = load_fixed(args.checkpoint, device)
    if sha != hashes['checkpoint_sha'] or front_checkpoint['step'] != manifest['checkpoint_step']:
        raise ValueError('Front checkpoint changed or step mismatch')
    if any(model.config[k] != manifest[k] for k in ('dim','stride')):
        raise ValueError('Front configuration mismatch')
    heads = load_heads(states,manifest['dim'],device,module.Readout)
    before = dict(front=state_digest(model), **{k:state_digest(h) for k,h in heads.items()})
    if before['front'] != manifest['initial_front_state_sha256']:
        raise ValueError('Frozen front state differs from training')
    buffers = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    out = new_output(args.output)
    record = dict(format='ROI_EXCHANGE_AUDIT_V1',arguments=vars(args),source_run_manifest=manifest,
                  source_run_manifest_sha256=digest(root/'manifest.json'),source_summary_sha256=digest(root/'summary.json'),
                  checkpoint_sha256=sha,head_file_sha256=head_files,final_step=final_step,
                  source_hashes=hashes,before_state_sha256=before,smoke=args.smoke,test_accessed=False,
                  audit_source_sha256=audit_files,
                  gt_used_for_routing=False,weights_updated=False,buffers_updated=False,
                  stores_predictions=False,stores_feature_cache=False)
    save_json(out/'manifest.json',record)
    save_json(out/'summary.json',dict(complete=False,smoke=args.smoke,test_accessed=False))
    data = loader(args.dataset,val[:2] if args.smoke else val,workers=args.workers)
    result = evaluate(model,heads,data,device,manifest,module)
    assert_front(model,buffers)
    after = dict(front=state_digest(model), **{k:state_digest(h) for k,h in heads.items()})
    if before != after or any(h.training or any(p.requires_grad or p.grad is not None for p in h.parameters()) for h in heads.values()):
        raise RuntimeError('Evaluation changed model state or trainability')
    if digest(args.checkpoint)!=sha or any(digest(paths[k])!=v for k,v in head_files.items()):
        raise RuntimeError('Input checkpoint files changed')
    if source_hash()!=hashes['frontend_hash'] or module.implementation_hash()!=hashes['experiment_hash']:
        raise RuntimeError('Implementation changed during audit')
    if {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')} != audit_files:
        raise RuntimeError('Audit implementation changed during evaluation')
    if digest(root/'manifest.json')!=record['source_run_manifest_sha256'] or digest(root/'summary.json')!=record['source_summary_sha256']:
        raise RuntimeError('Source run metadata changed during audit')
    if digest(args.train_list)!=hashes['train_sha'] or digest(args.val_list)!=hashes['val_sha']:
        raise RuntimeError('Data split lists changed during audit')
    if result['image_count'] != (2 if args.smoke else 450):
        raise RuntimeError('Incomplete validation traversal')
    if not args.smoke:
        expected = summary.get('final_metrics')
        if expected is None:
            log_path = root/'validation.jsonl'
            rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            matching = [row for row in rows if row.get('step')==final_step]
            if len(matching)!=1:
                raise ValueError('Require one final-step reference validation row')
            expected = matching[0]['metrics']
            record['validation_log_sha256'] = digest(log_path)
        verify_reproduction(result['groups']['all']['metrics'],expected)
    record.update(immutable_front_verified=True,immutable_readouts_verified=True,
                  source_integrity_verified=True,same_step_verified=True,after_state_sha256=after,
                  final_metrics_reproduced=not args.smoke,reproduction_tolerance=1e-5)
    result['provenance'] = record
    save_json(out/'audit_final.json',result)
    save_json(out/'manifest.json',record)
    save_json(out/'summary.json',dict(complete=True,final_step=final_step,image_count=result['image_count'],
                                    smoke=args.smoke,test_accessed=False,immutable_front_verified=True,
                                    immutable_readouts_verified=True,audit_sha256=digest(out/'audit_final.json'),
                                    manifest_sha256=digest(out/'manifest.json'),diagnostics_file='audit_final.json'))
    print('Frozen same-step audit complete:',out)


if __name__ == '__main__':
    main()
