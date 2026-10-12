"""Train paired region-logit residuals: frozen hard source, with/without Train Memory."""
import argparse
import math
import time
from pathlib import Path

import torch

from .common import (FORMAT, source, add_arguments, require_hard, implementation_hash, baseline_policy,
                     initial_gate, assert_unchanged, OutputLock)
from .core import make_heads, classify_regions, add_statistics, finish_statistics
from .diagnostics import retrieval_quality, add_quality, finish_quality
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import read_bank, device_bank
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions, decode_regions
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator


@torch.no_grad()
def evaluate(bundle, heads, bank, data, args, detailed=False, initial=False):
    front, source_head, device = bundle['front'], bundle['heads']['hard'], bundle['device']
    front.eval(); source_head.eval()
    for head in heads.values():
        head.eval()
    cases = [('nomemory', 'nomemory', 'normal'), ('memory', 'memory', 'normal')]
    if detailed:
        cases += [('memory_'+x, 'memory', x) for x in ('random', 'shuffle_labels', 'off')]
    names_all = ['old_pixel', 'old_region', 'source_hard']+[name for name, _, _ in cases]
    matrices = {name: torch.zeros(4, 4, dtype=torch.long) for name in names_all}
    statistics = {name: {} for name, _, _ in cases}
    changed_pixels = {name: 0 for name, _, _ in cases}
    pairs = [('nomemory', 'memory'), ('source_hard', 'nomemory'), ('source_hard', 'memory')]
    if detailed:
        pairs += [('memory', 'memory_'+x) for x in ('random', 'shuffle_labels', 'off')]
    diagnostic = DiagnosticAccumulator(threshold=bundle['manifest']['arguments']['threshold'],
                                      pairs=pairs) if detailed else None
    generators = {name: torch.Generator(device=device).manual_seed(args.seed+500+index)
                  for index, (name, _, _) in enumerate(cases)}
    count, valid_pixels, replay_max = 0, 0, 0.
    retrieval_evidence = {}
    for batch in data:
        pre, post, _, gt, names = source.parse_batch(batch, device)
        reference = source.capture(front, pre, post)
        pool_route, read_route = source.routes(reference, bundle['manifest'], 'hard', 'hard')
        encoded = encode_regions(source_head, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base_logits = source_head.classifier(tokens)
        base_prob = decode_regions(base_logits, read_route, mass, gt.shape[-2:])
        legacy = source_head(reference['features']['roi'], pool_route, gt.shape[-2:])
        replay_max = max(replay_max, float((base_prob-legacy).abs().max()))
        torch.testing.assert_close(base_prob, legacy, atol=2e-5, rtol=2e-4,
                                   msg='Separated frozen hard readout replay')
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1), source_hard=base_prob.argmax(1))
        valid = (gt >= 0) & (gt < 4)
        for name, mode, intervention in cases:
            logits, stats = classify_regions(heads[mode], tokens, mass, base_logits, bank, names,
                [source.event(x) for x in names], mode, args.topk, args.query_chunk,
                not args.allow_same_event, intervention, generators[name])
            if initial and not torch.equal(logits, base_logits):
                raise RuntimeError('Zero residual must exactly reproduce source region logits at step0: '+name)
            probability = decode_regions(logits, read_route, mass, gt.shape[-2:])
            if not torch.isfinite(probability).all():
                raise RuntimeError('Nonfinite validation probability: '+name)
            predictions[name] = probability.argmax(1)
            changed_pixels[name] += int(((predictions[name] != predictions['source_hard']) & valid).sum())
            add_statistics(statistics[name], stats)
        for name, prediction in predictions.items():
            source.cm_update(matrices[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], names)
            # GT is consumed only by this offline diagnostic AFTER feature-only
            # retrieval; it never enters the classifier, neighbor rank, or Q.
            add_quality(retrieval_evidence, retrieval_quality(tokens, mass, pool_route, gt,
                bank, names, [source.event(x) for x in names], args.topk, args.query_chunk,
                not args.allow_same_event))
        count += len(names)
        valid_pixels += int(valid.sum())
        if count % 25 == 0:
            print('evaluate memory_v2', count, flush=True)
    if count != (2 if args.smoke else 450):
        raise RuntimeError('Expected smoke Val2 or complete Val450')
    result = dict(metrics={name: source.metrics(cm) for name, cm in matrices.items()},
        support={name: finish_statistics(row) for name, row in statistics.items()},
        changed_pixel_fraction_vs_source={name: value/max(1, valid_pixels) for name, value in changed_pixels.items()},
        image_count=count, hard_replay_max_abs=replay_max, zero_residual_identity_checked=initial,
        test_accessed=False)
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
        result['retrieval_quality'] = finish_quality(retrieval_evidence)
    return result


def schedule_factor(step, steps):
    warm = min(100, max(1, steps//10))
    return step/warm if step <= warm else .5*(1+math.cos(math.pi*(step-warm)/max(1, steps-warm)))


def run(args):
    require_hard(args)
    source.validate_training_args(args)
    begin = time.monotonic()
    source.seed_everything(args.seed)
    code_hash = implementation_hash()
    bundle = source.load_source(args)
    # Policy is fixed BEFORE inference. Never tune tolerance using this run.
    policy = baseline_policy(bundle, args)
    identity = source.source_identity(bundle, args)
    bank_cpu, bank_manifest = read_bank(args.bank, identity, args.smoke, bundle['train'])
    if not args.smoke and set(bank_cpu['image_names']) != set(bundle['train']):
        raise ValueError('Formal bank must have visited all Train2918 images')
    device = bundle['device']
    front, source_head = bundle['front'], bundle['heads']['hard']
    bank = device_bank(bank_cpu, device)
    heads = make_heads(bank['tokens'].shape[1], device)
    initial_hash = {name: source.state_digest(head) for name, head in heads.items()}
    if len(set(initial_hash.values())) != 1:
        raise RuntimeError('Matched heads require identical initialization')
    if set(map(id, heads['nomemory'].parameters())) & set(map(id, heads['memory'].parameters())):
        raise RuntimeError('Matched heads must not share trainable tensors')
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.005)
                  for name, head in heads.items()}
    bank_files = {str(Path(args.bank)/name): source.digest(Path(args.bank)/name)
                  for name in ('bank.pt', 'manifest.json')}
    out = source.new_output(args.output)
    manifest = dict(format=FORMAT, arguments=vars(args), identity=identity, code_hash=code_hash,
        source_hash=source.experiment_hash(), bank_file_sha256=bank_files,
        bank_regions=len(bank['tokens']), bank_limit=bank_manifest['bank_limit'],
        head_initial_sha256=initial_hash, head_parameters={n: sum(p.numel() for p in h.parameters()) for n,h in heads.items()},
        smoke=args.smoke, test_accessed=False, frozen_front=True, frozen_building=True,
        frozen_q=True, frozen_embedding=True, frozen_source_classifier=True,
        output='frozen hard REGION logits + zero-initialized learned REGION correction; unchanged hard read route',
        control='identical learned tensors and initialization; nomemory learned class slots versus Train retrieval',
        exclusion='same image forbidden; '+('same event allowed (explicit ablation)' if args.allow_same_event else 'same event forbidden'),
        loss='same pixel NLL+Lovasz after regional probability unpool; no new loss or class reweighting',
        protocol='paired crops; batch*accum16; AdamW wd.005, warmup100/cosine, clip1, FP32',
        primary_endpoint='same-step final memory minus independently trained nomemory; compare source_hard too',
        checkpoints='best+last correction heads ONLY; no front/optimizer/dense cache copy; not exact-resume capable',
        interventions='final-only trained Memory random, fixed-per-image shuffle_labels, off; NOT trained baselines',
        bank_context_mismatch='full Train1024 supports versus random Train512 queries; fixed encoder does not remove context shift',
        first_step_gradient_note='zero output layer blocks earlier adapter gradients at step1 only; monitor subsequent steps')
    source.save_json(out/'manifest.json', manifest)
    source.save_json(out/'baseline_policy.json', policy)
    source.save_json(out/'summary.json', dict(complete=False, phase='initial_evaluation',
        optimizer_steps=0, smoke=args.smoke, test_accessed=False))
    train_names = bank['image_names'] if args.smoke else bundle['train']
    training = source.loader(args.dataset, train_names, True, args.batch, args.workers, seed=args.seed+2)
    validation = source.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
                               workers=args.workers, seed=args.seed+3)
    if not len(training):
        raise ValueError('Empty training loader; smoke batch <=2 required')
    print('REGION MEMORY V2: fixed hard source, paired residual heads; source step',
          bundle['provenance']['source_head_step'], flush=True)
    print('Parameters:', manifest['head_parameters'], 'bank regions:', len(bank['tokens']), flush=True)
    initial_result = evaluate(bundle, heads, bank, validation, args, initial=True)
    gate = initial_gate(bundle, initial_result, args, out, policy)
    assert_unchanged(bundle, code_hash, bank_files)
    print('STEP0', {n: m['valid_miou'] for n,m in initial_result['metrics'].items()}, flush=True)
    if device.type == 'cuda':
        # load_source has explicitly prepared/initialized this device already.
        torch.cuda.reset_peak_memory_stats(device)
    best = {n: -1. for n in heads}
    best_steps = {n: 0 for n in heads}
    grad_steps = {n: 0 for n in heads}
    module_grad_steps = {n: {k: 0 for k in ('labels','attention','norm','ffn')} for n in heads}
    iterator = iter(training)
    for step in range(1, args.steps+1):
        front.eval(); source_head.eval()
        factor = schedule_factor(step, args.steps)
        losses = {n: 0. for n in heads}
        statistics = {n: {} for n in heads}
        for name, head in heads.items():
            head.train(); optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr*factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training); batch = next(iterator)
            pre, post, _, gt, names = source.parse_batch(batch, device)
            with torch.no_grad():
                reference = source.capture(front, pre, post)
                pool_route, read_route = source.routes(reference, bundle['manifest'], 'hard', 'hard')
                encoded = encode_regions(source_head, reference['features']['roi'], pool_route)
                base_logits = source_head.classifier(encoded['tokens'])
            for name, head in heads.items():
                logits, stats = classify_regions(head, encoded['tokens'], encoded['mass'], base_logits,
                    bank, names, [source.event(x) for x in names], name, args.topk, args.query_chunk,
                    not args.allow_same_event)
                probability = decode_regions(logits, read_route, encoded['mass'], gt.shape[-2:])
                loss = source.objective(probability, gt)/args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward(); losses[name] += float(loss.detach())
                add_statistics(statistics[name], stats)
        grad_norm = {}
        for name, head in heads.items():
            for module_name in module_grad_steps[name]:
                parameters = getattr(head, module_name).parameters()
                has_gradient = any(p.grad is not None and bool((p.grad != 0).any()) for p in parameters)
                module_grad_steps[name][module_name] += int(has_gradient)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            grad_norm[name] = float(norm)
            grad_steps[name] += int(float(norm) > 0)
            optimizers[name].step()
        if any(p.requires_grad or p.grad is not None for m in (front, source_head) for p in m.parameters()):
            raise RuntimeError('Frozen front/embedding/classifier gradient boundary violated')
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step*16, losses=losses, lr=args.lr*factor, grad_norm=grad_norm,
                support={n: finish_statistics(s) for n,s in statistics.items()}, elapsed_seconds=time.monotonic()-begin)
            source.append_json(out/'training.jsonl', row)
            print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, heads, bank, validation, args, detailed=step == args.steps)
            assert_unchanged(bundle, code_hash, bank_files)
            if 'diagnostics' in result:
                source.save_json(out/'audit_final.json', result.pop('diagnostics'))
                source.save_json(out/'retrieval_quality.json', result['retrieval_quality'])
            source.append_json(out/'validation.jsonl', dict(step=step, **result))
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                checkpoint = dict(format=FORMAT, name=name, step=step, head=head.state_dict(),
                                  manifest=manifest, metrics=result['metrics'][name])
                if score > best[name]:
                    best[name], best_steps[name] = score, step
                    source.save_pt(out/(name+'_best.pth'), checkpoint)
                if step == args.steps:
                    source.save_pt(out/(name+'_last.pth'), checkpoint)
            source.save_json(out/'summary.json', dict(complete=False, phase='training', optimizer_steps=step,
                best=best, best_steps=best_steps, smoke=args.smoke, test_accessed=False))
            print('VAL memory_v2', step, {n:m['valid_miou'] for n,m in result['metrics'].items()}, flush=True)
    assert_unchanged(bundle, code_hash, bank_files)
    changed = {n: source.state_digest(h) != initial_hash[n] for n,h in heads.items()}
    if not all(changed.values()) or not all(grad_steps.values()):
        raise RuntimeError('A trained correction head never changed or received gradients')
    source.save_json(out/'summary.json', dict(complete=True, phase='complete', final_step=args.steps,
        optimizer_steps=args.steps, smoke=args.smoke, test_accessed=False, best=best, best_steps=best_steps,
        final_metrics=result['metrics'], support=result['support'],
        changed_pixel_fraction_vs_source=result['changed_pixel_fraction_vs_source'],
        final_memory_minus_nomemory=result['metrics']['memory']['valid_miou']-result['metrics']['nomemory']['valid_miou'],
        baseline_reproduction_pass=gate['passed'], frozen_source_verified=True,
        training_verification=dict(nonzero_gradient_steps=grad_steps, module_nonzero_gradient_steps=module_grad_steps,
                                   weights_changed=changed),
        elapsed_seconds=time.monotonic()-begin,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    print('PAIRED RESIDUAL MEMORY COMPLETE. Train/Val only. No source weights changed.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    add_arguments(parser, training=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--allow_same_event', action='store_true', help='Explicit ablation; same image always forbidden')
    args = parser.parse_args()
    if min(args.topk, args.query_chunk) < 1 or (args.smoke and args.batch > 2):
        parser.error('Positive topk/query_chunk, smoke batch<=2 required')
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
