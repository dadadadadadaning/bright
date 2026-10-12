"""Replay one recorded failed feature-audit image, or its exact Val prefix.

This is a new read-only audit, not a resume operation. The failed run remains
incomplete and is never changed. Model/data provenance must match the old run;
the diagnostic implementation is allowed to change and records its new hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest, lists


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed_run', required=True, help='Existing failed feature-audit directory')
    p.add_argument('--output', required=True, help='New, nonexistent output directory outside the failed run')
    p.add_argument('--prefix', action='store_true',
                   help='Replay Val from index zero through the recorded failure, inclusive')
    return p


def _object(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: '+str(path))
    return value


def load_recheck_arguments(args):
    """Validate provenance without inference, creating directories, or changing files."""
    failed = Path(args.failed_run).resolve()
    output = Path(args.output).resolve()
    if not failed.is_dir():
        raise ValueError('Failed-run directory is missing: '+str(failed))
    if output == failed or failed in output.parents:
        raise ValueError('Recheck output must be outside the immutable failed-run directory')
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError('Output exists; preserve it and choose a new output: '+str(output))

    summary_path, manifest_path = failed/'summary.json', failed/'manifest.json'
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError('Failed run requires both summary.json and manifest.json')
    source_hashes = {'summary_sha256': digest(summary_path), 'manifest_sha256': digest(manifest_path)}
    summary, manifest = _object(summary_path), _object(manifest_path)
    if summary.get('complete') is not False:
        raise ValueError('Expected complete:false; this entry does not replay completed runs')
    if summary.get('audit_type') != 'feature' or manifest.get('audit_type') != 'feature':
        raise ValueError('Only a failed feature audit can be rechecked here')
    failure = summary.get('failure')
    image = failure.get('image') if isinstance(failure, dict) else None
    if not isinstance(image, str) or not image:
        raise ValueError('failure.image is missing; do not guess an image index from progress logs')

    original = manifest.get('arguments')
    required = ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output',
                'device', 'workers', 'seed', 'smoke', 'start_index', 'max_images',
                'embedding_interventions')
    if not isinstance(original, dict) or any(key not in original for key in required):
        raise ValueError('Manifest arguments are incomplete; original execution settings are required')
    if Path(original['output']).resolve() != failed:
        raise ValueError('Manifest output does not identify the supplied failed-run directory')
    if (type(original['smoke']) is not bool or type(original['embedding_interventions']) is not bool
            or type(original['start_index']) is not int or type(original['max_images']) is not int
            or original['start_index'] < 0 or original['max_images'] < 0):
        raise ValueError('Original smoke/subset/embedding settings are invalid')

    # Do not silently substitute a newer checkpoint or a different split.
    checks = {}
    for argument, field in (('checkpoint', 'checkpoint_sha256'), ('a1_checkpoint', 'a1_sha256'),
                            ('train_list', 'train_sha256'), ('val_list', 'val_sha256')):
        expected = manifest.get(field)
        if not isinstance(expected, str) or not expected:
            raise ValueError('Missing original provenance field: '+field)
        actual = digest(original[argument])
        if actual != expected:
            raise ValueError('Original input hash mismatch: '+argument)
        checks[field] = actual
    if 'snapshot_metadata_sha256' in manifest:
        actual = digest(Path(original['checkpoint']).parent/'snapshot.json')
        if actual != manifest['snapshot_metadata_sha256']:
            raise ValueError('Original input hash mismatch: snapshot.json')
        checks['snapshot_metadata_sha256'] = actual

    _, val = lists(original['train_list'], original['val_list'])
    if image not in val:
        raise ValueError('Recorded failure image is not in the original Val list: '+image)
    start, maximum = original['start_index'], original['max_images']
    if start >= len(val) or (original['smoke'] and (start or maximum)):
        raise ValueError('Original smoke/subset settings disagree with Val')
    expected_selected = val[:2] if original['smoke'] else val[start:start+maximum if maximum else None]
    selected = manifest.get('selected_images')
    if not isinstance(selected, list) or selected != expected_selected or image not in selected:
        raise ValueError('Original selected_images are missing or inconsistent with the failure and arguments')
    index = val.index(image)
    if digest(summary_path) != source_hashes['summary_sha256'] or digest(manifest_path) != source_hashes['manifest_sha256']:
        raise RuntimeError('Failed-run metadata changed while it was being inspected')

    updated = dict(original)
    updated.update(output=str(output), smoke=False, start_index=0 if args.prefix else index,
                   max_images=index+1 if args.prefix else 1)
    updated['recheck_origin'] = dict(
        failed_run=str(failed), **source_hashes, image=image, original_val_index_zero_based=index,
        original_selected_index_zero_based=selected.index(image),
        mode='prefix' if args.prefix else 'single_image', original_input_hashes=checks,
        original_stage_source_sha256=manifest.get('stage_source_sha256'),
        original_execution=manifest.get('execution'),
        failed_run_modified=False, not_training=True,
        source_check='Original model/data hashes enforced; changed diagnostic source is recorded by the new run, not required to equal the old hash.',
        full_val_warning='Single-image/prefix diagnostics are not full-Val evidence unless the existing audit independently validates all 450 images.',
    )
    return argparse.Namespace(**updated)


def main():
    args = load_recheck_arguments(parser().parse_args())
    origin = args.recheck_origin
    print('Recorded failed image: '+origin['image'], flush=True)
    print('Original Val position (one based): '+str(origin['original_val_index_zero_based']+1), flush=True)
    print('Recheck mode: '+origin['mode'], flush=True)
    print('New output: '+args.output, flush=True)
    print('No training, no CPU fallback/retry, no old-result overwrite. This does not repair the old report.', flush=True)
    from .feature_audit import run
    run(args)


if __name__ == '__main__':
    main()
