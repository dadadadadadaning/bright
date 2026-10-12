"""Recompute Train-only query features and fixed-J1 difficulty without changing the bank."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .io_utils import (add_source_arguments, load_verified_bundle, new_output, artifact_identity,
                       verify_unchanged, write_json, digest, safe_child, fail_summary)


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument('--protocol', required=True)
    return p


@torch.no_grad()
def pool_base_probability(probability, result):
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting
    grid = F.interpolate(probability.float(), size=result['shape'], mode='area')
    route = GroupedRouting(result['ids'], result['tokens'].shape[1])
    pooled, mass = route.pool(grid, result['q'], result['ids'], result['tokens'].shape[1])
    return pooled[0], mass[0]


def choose_smoke_names(names):
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event
    selected, events = [], set()
    for name in names:
        key = event(name)
        if key not in events:
            events.add(key)
            selected.append(name)
        if len(selected) == 4:
            return selected
    raise ValueError('Smoke needs four distinct Train events')


def run(args):
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import decoder_forward, upsample_probability
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import parse_batch, region_targets, event
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    output = new_output(args.output)
    capture = None
    try:
        bundle = load_verified_bundle(args)
        names = choose_smoke_names(bundle['train_names']) if args.smoke else bundle['train_names']
        index_by_name = {name: i for i, name in enumerate(bundle['train_names'])}
        if len(index_by_name) != len(bundle['train_names']):
            raise ValueError('Duplicate Train names')
        if set(names) & set(bundle['val_names']):
            raise ValueError('Train/Val image overlap')
        output.joinpath('shards').mkdir()
        manifest = dict(format='SRAC_HARDQUERY_QUERIES_V1', complete=False, smoke=args.smoke,
                        train_only=True, test_accessed=False, **artifact_identity(bundle),
                        arguments=vars(args), train_count=len(names), shards=[],
                        difficulty='1 - dot(Train soft label, fixed J1 same-Q mean probability)',
                        query_origin='Recomputed FP32 full-image frozen front; original bank unchanged',
                        hardness_limit='J1/front saw Train: descriptive errors, not out-of-fold generalization estimates')
        write_json(output / 'query_manifest.json', manifest)
        capture = FrozenFrontCapture(bundle['front'])
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed)
        with PoolGuard(output / 'pool_debug', mode=args.pool_mode) as guard, torch.no_grad():
            for batch in data:
                pre, post, _, target, image_names = parse_batch(batch, bundle['device'])
                name = str(image_names[0])
                if name not in index_by_name or name not in names:
                    raise ValueError('Non-Train query encountered')
                image_id = index_by_name[name]
                event_id = bundle['bank']['event_names'].index(event(name))
                # Also verify the existing shard belongs to this exact Train image;
                # its quantized features are not silently substituted for live queries.
                old_record = bundle['cache']['shards'][image_id]
                old_path = safe_child(bundle['cache_root'], old_record['path'])
                if digest(old_path) != old_record['sha256']:
                    raise ValueError('Original Train shard hash mismatch')
                old = torch.load(old_path, map_location='cpu', weights_only=False)
                if old['name'] != name or int(old['image']) != image_id or int(old['event']) != event_id:
                    raise ValueError('Original Train shard provenance mismatch')
                bundle['input_paths']['query_source_%05d' % image_id] = old_path
                bundle['inputs_before']['query_source_%05d' % image_id] = old_record['sha256']
                guard.set_image(name)
                result, feature = capture(pre, post)
                decoded = decoder_forward(bundle['decoders']['j1'], 'j1', feature, result, None, None)
                probability = upsample_probability(decoded['logits'], target.shape[-2:])
                base, mass = pool_base_probability(probability, result)
                distribution, weights = region_targets(target, result)
                valid = (weights[0] > 0) & (mass > 1e-6) & result['valid_regions'][0]
                weights = weights[0] * valid.to(weights.dtype)
                payload = dict(tokens=result['tokens'][0].float().cpu(),
                               distribution=distribution[0].float().cpu(), weights=weights.float().cpu(),
                               mass=mass.float().cpu(), base_probability=base.float().cpu(),
                               event=event_id, image=image_id, name=name,
                               region_ids=torch.arange(len(base)))
                if not all(torch.isfinite(payload[key]).all() for key in
                           ('tokens', 'distribution', 'weights', 'mass', 'base_probability')):
                    raise ValueError('Nonfinite Train query')
                relative = 'shards/%05d.pt' % image_id
                path = output / relative
                torch.save(payload, path)
                manifest['shards'].append(dict(path=relative, sha256=digest(path), name=name,
                                               event=event_id, image=image_id))
                write_json(output / 'query_manifest.json', manifest)
                if len(manifest['shards']) % 25 == 0 or len(manifest['shards']) == len(names):
                    print('Train query cache %d/%d' % (len(manifest['shards']), len(names)), flush=True)
        manifest.update(complete=True, immutable=verify_unchanged(bundle),
                        original_shards_sha256={k: v for k, v in bundle['inputs_before'].items()
                                               if k.startswith('query_source_')})
        write_json(output / 'query_manifest.json', manifest)
        write_json(output / 'summary.json', dict(complete=True, smoke=args.smoke, train_only=True,
                   image_count=len(names), test_accessed=False, full_validation_evidence=False))
        return manifest
    except Exception as error:
        fail_summary(output, error)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == '__main__':
    run(parser().parse_args())
