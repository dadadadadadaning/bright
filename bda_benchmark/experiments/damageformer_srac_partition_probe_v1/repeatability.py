"""Single-image native/replay repetitions; no training and no full-Val claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit import TOLERANCE, capture_inputs, cpu_copy, compare_tensor, source_digest
from .core import run_partition
from .numeric_runtime import configure_determinism, assert_determinism, q_difference
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.runtime import load_fixed
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    lists, loader, parse_batch, new_output, save_json,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--image', default='ukraine-conflict_00000089')
    p.add_argument('--repeats', type=int, default=10)
    p.set_defaults(smoke=False)
    return p


@torch.no_grad()
def run(args):
    if args.repeats < 2:
        raise ValueError('Repeatability requires at least two repetitions')
    _, val = lists(args.train_list, args.val_list)
    if args.image not in val:
        raise ValueError('Requested image is not in the audited Val list: ' + args.image)
    out = new_output(args.output)
    status = dict(mode='repeatability', complete=False, gates_passed=False,
                  image=args.image, val_index_one_based=val.index(args.image)+1,
                  repeats_requested=args.repeats, repeats_completed=0,
                  diagnostic_subset=True, full_validation_evidence=False,
                  test_accessed=False, weights_updated=False, memory_accessed=False,
                  q_replay_tolerance=TOLERANCE, self_repeat_exact=True, checks_max_abs={})
    save_json(out/'summary.json', status)
    try:
        execution = configure_determinism()
        status['execution'] = execution
        save_json(out/'summary.json', status)
        model, _, manifest = load_fixed(args)
        model.requires_grad_(False).eval()
        if assert_determinism() != execution:
            raise RuntimeError('Numerical execution settings changed while loading the source')
        manifest.update(mode='repeatability', execution=execution, probe_source_sha256=source_digest(),
                        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                        full_validation=False, q_replay_tolerance=TOLERANCE,
                        repeats=args.repeats, selected_images=[args.image])
        save_json(out/'manifest.json', manifest)
        print('Strict numerical execution: ' + json.dumps(execution), flush=True)
        before_state = module_fingerprint(model)
        device = torch.device(args.device)
        data = loader(args.dataset, [args.image], False, 1, args.workers, seed=args.seed)
        pre, post, _, _, names = parse_batch(next(iter(data)), device)
        if str(names[0]) != args.image:
            raise RuntimeError('Loader returned a different image')
        all_passed = True
        first_native = first_replay = None
        with torch.autocast(device_type=device.type, enabled=False):
            captured, fixed = capture_inputs(model, pre.float(), post.float())
            # Own CPU snapshots of ALL evidence, not just partition inputs.
            snapshots = {key: cpu_copy(value) for key, value in fixed.items()}
            capture_q, ids = cpu_copy(captured['q']), cpu_copy(captured['ids'])
            with (out/'repeats.jsonl').open('x', encoding='utf-8') as journal:
                for repeat in range(args.repeats):
                    if assert_determinism() != execution:
                        raise RuntimeError('Numerical execution settings changed during repetitions')
                    native = model.partition(fixed['structural'], fixed['B'])
                    replay = run_partition(model.partition, fixed['structural'], fixed['B'])
                    ids_match = all(torch.equal(cpu_copy(part['ids']), ids) for part in (native, replay))
                    nq, rq = cpu_copy(native['q']), cpu_copy(replay['q'])
                    if first_native is None:
                        first_native, first_replay = nq, rq
                    checks = dict(
                        capture_vs_native=q_difference(capture_q, nq, ids, TOLERANCE),
                        native_vs_replay=q_difference(nq, rq, ids, TOLERANCE),
                        native_first_vs_current=q_difference(first_native, nq, ids, TOLERANCE),
                        replay_first_vs_current=q_difference(first_replay, rq, ids, TOLERANCE),
                    )
                    self_exact = (checks['native_first_vs_current']['bitwise_equal'] and
                                  checks['replay_first_vs_current']['bitwise_equal'])
                    immutable = {key: compare_tensor(value, snapshots[key]) for key, value in fixed.items()}
                    settings_match = assert_determinism() == execution
                    passed = (ids_match and self_exact and settings_match and
                              all(row['passed'] for row in checks.values()) and
                              all(row['passed'] for row in immutable.values()))
                    all_passed = all_passed and passed
                    status['self_repeat_exact'] = status['self_repeat_exact'] and self_exact
                    for name, row in checks.items():
                        if row['max_abs'] is not None:
                            status['checks_max_abs'][name] = max(
                                status['checks_max_abs'].get(name, 0.), row['max_abs'])
                    record = dict(repeat=repeat+1, passed=passed, ids_match=ids_match,
                                  self_repeat_exact=self_exact, checks=checks, fixed_unchanged=immutable)
                    journal.write(json.dumps(record, allow_nan=False)+'\n')
                    journal.flush()
                    status['repeats_completed'] = repeat+1
                    save_json(out/'summary.json', status)
                    print(json.dumps(dict(repeat=repeat+1, passed=passed,
                                          self_repeat_exact=self_exact, checks=checks)), flush=True)
                    # Never continue with changed inputs or different candidate IDs.
                    if not ids_match or not all(row['passed'] for row in immutable.values()):
                        raise RuntimeError('Fixed evidence or candidate IDs changed; see repeats.jsonl')
                    # Finite numeric gate failures are collected for all ten runs,
                    # not retried until a lucky run passes.
                    if not all(row['finite'] for row in checks.values()):
                        raise RuntimeError('Nonfinite Q found; see repeats.jsonl')
        unchanged = module_fingerprint(model) == before_state
        status['weights_buffers_unchanged'] = unchanged
        status['execution_unchanged'] = assert_determinism() == execution
        passed = all_passed and unchanged and status['execution_unchanged']
        status.update(complete=passed, gates_passed=passed)
        save_json(out/'summary.json', status)
        if not passed:
            raise RuntimeError('Repeatability gate FAILED; all repetitions are retained, no tolerance relaxed')
        print('REPEATABILITY COMPLETE. Single image only; full Val is still required. No training/Test.', flush=True)
    except Exception as error:
        status.update(complete=False, gates_passed=False,
                      failure=dict(type=type(error).__name__, message=str(error)))
        save_json(out/'summary.json', status)
        raise


if __name__ == '__main__':
    run(parser().parse_args())
