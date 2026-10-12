"""Read-only provenance helpers for the follow-up, not historical source edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def code_hash():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob('*.py')) + sorted(root.glob('*.sh')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def select_images(names, smoke=False, start_index=0, max_images=0):
    if (start_index < 0 or max_images < 0 or start_index >= len(names)
            or (smoke and (start_index or max_images))):
        raise ValueError('Choose smoke OR a valid nonnegative Val subset')
    selected = names[:2] if smoke else names[start_index:start_index+max_images if max_images else None]
    return selected, bool(smoke or selected != names)


def validate_failed_memory(args):
    """Resolve original arguments/hashes; never rename a failed run as complete."""
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest, lists
    failed = Path(args.failed_run).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output == failed or failed in output.parents:
        raise ValueError('Choose a new output outside the preserved failed run')
    summary = read_object(failed / 'summary.json')
    manifest = read_object(failed / 'manifest.json')
    if summary.get('complete') is not False or manifest.get('format') != 'SRAC_STAGE_MEMORY_DIAGNOSTIC_V1':
        raise ValueError('Requires an incomplete stage Memory audit')
    image = (summary.get('failure') or {}).get('image')
    original = dict(manifest['arguments'])
    # Verify the exact files recorded by the original audit, including moved
    # overrides. An unrelated/new checkpoint must never become a failure replay.
    paths = manifest.get('all_input_paths', {})
    hashes = manifest.get('input_fingerprints_before', {})
    if not paths or set(paths) != set(hashes):
        raise ValueError('Missing original immutable-file ledger')
    for name, path in paths.items():
        if digest(path) != hashes[name]:
            raise ValueError('Failed-run input changed: ' + name)
    _, val = lists(original['train_list'], original['val_list'])
    if image not in val:
        raise ValueError('Failure image missing from original Val list')
    index = val.index(image)
    old_names, _ = select_images(val, original.get('smoke', False),
                                original.get('start_index', 0), original.get('max_images', 0))
    if image not in old_names:
        raise ValueError('Failure is outside original image selection')
    original.update(output=str(output), smoke=False, start_index=0 if args.prefix else index,
                    max_images=index+1 if args.prefix else 1, inventory_shards=False,
                    pool_mode=args.pool_mode, pool_capture=args.pool_capture,
                    pool_check_reference=args.pool_check_reference)
    original['recheck_origin'] = dict(failed_run=str(failed), image=image,
        original_val_index=index, mode='prefix' if args.prefix else 'single',
        summary_sha256=digest(failed/'summary.json'), manifest_sha256=digest(failed/'manifest.json'),
        old_result_modified=False)
    return original
