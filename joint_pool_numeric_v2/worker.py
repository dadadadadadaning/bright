"""Isolated saved-input numerical audit. This module cannot start training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from pool_ops import (compare_results, limited_cpu_threads, restore_layout,
                      tensor_layout, validate_layout)
from numeric_core import compare, cpu_reference64, pool_literal, validate_snapshot
from replay_worker import apply_runtime, current_runtime, read_runtime, save, sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('cpu', 'native', 'contiguous', 'compare'), required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected-sha256')
    p.add_argument('--expected-image')
    p.add_argument('--expected-call', type=int)
    p.add_argument('--runtime-json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--audit-root')
    return p


def finite(outputs):
    return all(bool(torch.isfinite(x).all()) for x in outputs.values())


def compact(record):
    return {name: {key: value.get(key) for key in
                  ('passed', 'finite', 'max_abs', 'elements_outside_tolerance')}
            for name, value in record.items() if isinstance(value, dict)}


def localize_tokens(actual, reference, snapshot):
    """Small B*K*C diagnostics: validity, worst coordinates, raw-moment halves."""
    aa, rr = actual['tokens'], reference['tokens']
    diff = (aa.double() - rr.double()).abs()
    bound = 2e-5 + 2e-5 * rr.double().abs()
    bad = diff > bound
    valid_a, valid_r = actual['mass'] > 1e-6, reference['mass'] > 1e-6
    result = dict(elements_outside_tolerance=int(bad.sum()),
                  distinct_tokens_outside_tolerance=int(bad.any(-1).sum()),
                  valid_token_elements_outside=int((bad & valid_r[..., None]).sum()),
                  invalid_token_elements_outside=int((bad & ~valid_r[..., None]).sum()),
                  validity_disagreements=int((valid_a != valid_r).sum()), worst=[])
    scaled = (diff / bound).flatten()
    for index in torch.topk(scaled, min(8, scaled.numel())).indices.tolist():
        channel = index % aa.shape[-1]
        token = (index // aa.shape[-1]) % aa.shape[1]
        batch = index // (aa.shape[1] * aa.shape[2])
        pos = (batch, token, channel)
        result['worst'].append(dict(coordinate=list(pos), actual=float(aa[pos]),
            reference=float(rr[pos]), normalized_error=float(scaled[index]),
            mass_actual=float(actual['mass'][batch, token]),
            mass_reference=float(reference['mass'][batch, token]),
            reference_valid=bool(valid_r[batch, token])))
    content = snapshot['content']
    # Only infer mean/second-moment semantics when the saved input proves it.
    is_stats = content.shape[1] == 512 and torch.equal(content[:, :256].square(), content[:, 256:])
    result['saved_input_is_raw_and_fp32_square'] = bool(is_stats)
    if is_stats:
        result['mean_channels_outside'] = int(bad[..., :256].sum())
        result['second_moment_channels_outside'] = int(bad[..., 256:].sum())
        ma, sa = aa.chunk(2, dim=-1)
        mr, sr = rr.chunk(2, dim=-1)
        std_a = (sa - ma.square()).clamp_min(0).add(1e-6).sqrt()
        std_r = (sr - mr.square()).clamp_min(0).add(1e-6).sqrt()
        dd = (std_a.double() - std_r.double()).abs()
        result['derived_std'] = dict(max_abs=float(dd.max()), mean_abs=float(dd.mean()),
            elements_outside_tolerance=int((dd > 2e-5 + 2e-5 * std_r.double().abs()).sum()),
            diagnostic_only=True)
        result['downstream_use'] = ('This stats call uses tokens/mass and derived mean/std; '
                                    'its reconstruction is not consumed by SuperpixelEmbedding.')
    return result


def load_bound_snapshot(args):
    digest = sha256(args.input)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError('Saved input checksum changed')
    snapshot = torch.load(args.input, map_location='cpu', weights_only=True)
    check = validate_snapshot(snapshot)
    if any(snapshot[k].dtype != torch.float32 for k in ('content', 'weights')):
        raise ValueError('This comparison requires the saved FP32 content and weights')
    if args.expected_image is not None and snapshot.get('image') != args.expected_image:
        raise ValueError('Saved input image differs from failure.json')
    if args.expected_call is not None and snapshot.get('call') != args.expected_call:
        raise ValueError('Saved input call differs from failure.json')
    return snapshot, digest, check


def classify(cpu, native, contiguous, paired):
    """Evidence-based hints, never permission to train or a root-cause verdict."""
    if not cpu.get('execution_complete'):
        return 'CPU controls incomplete; no numerical attribution.'
    if not cpu['comparisons']['cpu64_scatter_vs_grouped']['passed']:
        return 'Independent FP64 controls disagree; inspect the references before interpreting CUDA.'
    if not native.get('execution_complete') or not contiguous.get('execution_complete'):
        return 'At least one CUDA worker did not finish; inspect its failure and log.'
    nn, cc = native['repeats'], contiguous['repeats']
    if not nn or not cc or not all(x['vs_first']['passed'] for x in nn + cc):
        return 'At least one backend is not repeat-stable at the fixed bound; inspect its repeat records.'
    native32 = all(x['original_cpu32_gate']['passed'] for x in nn)
    candidate32 = all(x['original_cpu32_gate']['passed'] for x in cc)
    if native32 and not candidate32:
        return 'Only contiguous failed the original CPU32 gate; investigate the candidate execution.'
    if not native32 and not candidate32 and paired and paired['passed']:
        if all(x['vs_cpu64']['passed'] for x in nn + cc):
            return ('Both CUDA variants fail CPU32; their first outputs agree in both directions, '
                    'each backend is repeat-stable, and all repeats meet the same bound '
                    'against CPU64; this supports a CPU32/reference-precision explanation for this input. '
                    'It does not validate full training or justify silently changing the original gate.')
        return ('Both CUDA variants fail CPU32 but agree with each other; inspect FP64 error and '
                'repeat stability. This alone does not establish acceptable numerical accuracy.')
    if native32 and candidate32:
        return ('The original CPU32 mismatch did not recur in these isolated replays. '
                'A full-sequence difference remains possible; do not claim the original failure is fixed.')
    return 'The controls show mixed results; inspect all pairwise comparisons and repeat stability.'


def run_comparison(args, output, summary):
    root = Path(args.audit_root).resolve()
    records, missing = {}, []
    for name in ('cpu', 'native', 'contiguous'):
        path = root / name / 'summary.json'
        records[name] = json.loads(path.read_text()) if path.is_file() else {}
        if not records[name].get('execution_complete'):
            missing.append(name)
        else:
            artifact = root / name / 'outputs.pt'
            if (records[name].get('input_sha256') != summary['input_sha256']
                    or not artifact.is_file()
                    or sha256(artifact) != records[name].get('outputs_sha256')):
                raise ValueError('Completed control has changed input/output binding: ' + name)
    if not missing:
        paths = [root / name / 'outputs.pt' for name in ('native', 'contiguous')]
        if all(p.is_file() for p in paths):
            for name, path in zip(('native', 'contiguous'), paths):
                if (records[name].get('input_sha256') != summary['input_sha256']
                        or sha256(path) != records[name].get('outputs_sha256')):
                    raise ValueError('CUDA result checksum/input binding differs: ' + name)
            native, candidate = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
            summary['comparisons']['contiguous_vs_native'] = compare(candidate, native)
            summary['comparisons']['native_vs_contiguous'] = compare(native, candidate)
            del native, candidate
        else:
            missing.append('finite_cuda_outputs')
    summary.update(execution_complete=True, diagnostic_complete=not missing,
                   finite_outputs=False, missing_workers=missing,
                   worker_summaries=records,
                   interpretation=classify(records['cpu'], records['native'], records['contiguous'],
                                           ({'passed': all(value['passed'] for value in
                                                           summary['comparisons'].values())}
                                            if summary['comparisons'] else None)))
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


@torch.no_grad()
def run(args):
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(format='JOINT_POOL_NUMERIC_WORKER_V2', backend=args.backend,
                   execution_complete=False, finite_outputs=False, diagnostic_complete=False,
                   training_allowed=False, training=False, test_accessed=False,
                   full_validation_evidence=False, original_gate_relaxed=False,
                   cpu_thread_policy='CPU controls/comparisons use min(current_threads, 4); restored after each scope',
                   tolerance={'atol': 2e-5, 'rtol': 2e-5}, comparisons={}, repeats=[], failure=None)
    save(output / 'summary.json', summary)
    phase = 'load_saved_input'
    try:
        snapshot, digest, check = load_bound_snapshot(args)
        summary.update(input=str(Path(args.input).resolve()), input_sha256=digest,
                       image=snapshot.get('image'), call=snapshot.get('call'), input_check=check)
        save(output / 'summary.json', summary)
        if args.backend == 'compare':
            with limited_cpu_threads():
                return run_comparison(args, output, summary)
        x, q, ids, count = (snapshot[k] for k in ('content', 'weights', 'ids', 'token_count'))
        if args.backend == 'cpu':
            summary['runtime_actual'] = current_runtime(cpu_only=True)
            with limited_cpu_threads():
                phase = 'cpu32_original'
                print('CPU32 original scatter', flush=True)
                cpu32 = pool_literal(x, q, ids, count, precision='fp32')
                phase = 'cpu64_scatter'
                print('CPU64 scatter: promoting saved FP32 inputs, preserving their values', flush=True)
                cpu64 = pool_literal(x, q, ids, count, precision='fp64')
                phase = 'cpu64_independent_grouped'
                print('CPU64 independent grouped reference', flush=True)
                grouped = cpu_reference64(snapshot)
                if not all(finite(y) for y in (cpu32, cpu64, grouped)):
                    raise FloatingPointError('Nonfinite CPU reference output')
                phase = 'cpu_comparisons'
                summary['comparisons']['cpu64_scatter_vs_grouped'] = compare(cpu64, grouped)
                summary['comparisons']['cpu32_vs_cpu64'] = compare(cpu32, grouped)
                summary['cpu32_vs_cpu64_token_detail'] = localize_tokens(cpu32, grouped, snapshot)
                # CPU64 diagnostics permit mixed output dtypes. They are not a
                # replacement for the original FP32 reference gate.
                summary['fp64_reference_policy'] = ('Exact saved FP32 content/weights promoted before '
                    'arithmetic; independent grouped accumulation; duplicate routes and clamp1e-6 retained')
                del cpu64
                torch.save({'cpu32': cpu32, 'cpu64': grouped}, output / 'outputs.pt')
                summary.update(execution_complete=True, finite_outputs=True)
        else:
            phase = 'load_independent_cpu_controls'
            controls_path = output.parent / 'cpu/outputs.pt'
            controls_summary = json.loads((output.parent / 'cpu/summary.json').read_text())
            if (controls_summary.get('input_sha256') != digest
                    or not controls_summary.get('execution_complete')):
                raise ValueError('CPU controls do not bind this completed saved-input audit')
            if sha256(controls_path) != controls_summary.get('outputs_sha256'):
                raise ValueError('Saved CPU control outputs changed')
            controls = torch.load(controls_path, map_location='cpu', weights_only=True)
            layout = snapshot.get('layout')
            if not isinstance(layout, dict) or set(('content', 'weights', 'ids')) - set(layout):
                raise ValueError('Captured layout is required for CUDA replay')
            for key in ('content', 'weights', 'ids'):
                validate_layout(snapshot[key], layout[key])
            if not args.runtime_json:
                raise ValueError('Original runtime settings are required for CUDA')
            phase = 'runtime_configuration'
            requested = read_runtime(args.runtime_json)
            summary['runtime_requested'] = requested
            summary['runtime_actual'] = apply_runtime(requested, args.device)
            save(output / 'summary.json', summary)
            device = torch.device(args.device)
            if device.type != 'cuda':
                raise ValueError('CUDA workers require an explicit CUDA device')
            phase = 'restore_captured_layout'
            tensors = [restore_layout(snapshot[k], layout[k], device) for k in ('content', 'weights', 'ids')]
            summary['restored_layout'] = {k: tensor_layout(v) for k, v in zip(('content', 'weights', 'ids'), tensors)}
            first = None
            for repeat in range(args.repeats):
                phase = 'cuda_' + args.backend + '_repeat_' + str(repeat + 1)
                print('START ' + phase, flush=True)
                result = pool_literal(*tensors, count, index_layout=args.backend, precision='fp32')
                torch.cuda.synchronize(device)
                host = {k: v.detach().cpu().clone() for k, v in result.items()}
                if not finite(host):
                    raise FloatingPointError('Nonfinite CUDA result; worker stops')
                del result
                phase = 'compare_successful_cpu_outputs'
                with limited_cpu_threads():
                    if first is None:
                        first = host
                        torch.save(first, output / 'outputs.pt')
                    row = dict(repeat=repeat + 1,
                               original_cpu32_gate=compare_results(host, controls['cpu32']),
                               vs_cpu32=compare(host, controls['cpu32']),
                               vs_cpu64=compare(host, controls['cpu64']),
                               vs_first=compare(host, first))
                    for key in ('tokens', 'mass', 'reconstruction'):
                        row['original_cpu32_gate'][key]['total_elements'] = host[key].numel()
                    row['cpu32_token_detail'] = localize_tokens(host, controls['cpu32'], snapshot)
                    row['cpu64_token_detail'] = localize_tokens(host, controls['cpu64'], snapshot)
                summary['repeats'].append(row)
                save(output / 'summary.json', summary)
                print(json.dumps({'repeat': repeat + 1,
                                  'original_cpu32_gate': compact(row['original_cpu32_gate']),
                                  'vs_cpu64': compact(row['vs_cpu64']),
                                  'vs_first': compact(row['vs_first'])}), flush=True)
                # A numerical difference is recorded and may be compared in this
                # diagnostic. It never enables training or changes a gate.
            summary.update(execution_complete=True, finite_outputs=True)
        summary['outputs_sha256'] = sha256(output / 'outputs.pt')
        save(output / 'summary.json', summary)
        print('DIAGNOSTIC WORKER COMPLETE; training_allowed=false', flush=True)
        return summary
    except Exception as error:
        # CPU file operations only: do not probe/synchronize/access CUDA after
        # any device fault. Each GPU backend is in a separate OS process.
        summary['failure'] = dict(phase=phase, type=type(error).__name__, message=str(error))
        save(output / 'summary.json', summary)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    run(parser().parse_args())
