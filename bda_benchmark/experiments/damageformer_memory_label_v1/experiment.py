"""Fixed Train bank -> explicit label evidence. No new Q/B/embedding learning."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .core import LabelMixer, select_neighbors, shuffled_distribution, weighted_labels, mix_fixed
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    source, add_arguments, require_hard, baseline_policy, initial_gate, OutputLock,
    implementation_hash as memory_code_hash,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import (
    read_bank, device_bank, target_distribution,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions, valid_read_route
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import unpool
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator

FORMAT = 'REGION_LABEL_MEMORY_V1'
CLASS_NAMES = ('Background', 'Intact', 'Damaged', 'Destroyed')


def implementation_hash():
    digest = hashlib.sha256(memory_code_hash().encode())
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix in ('.py', '.sh'):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_frozen(bundle, code_hash, bank_files):
    source.assert_source(bundle)
    if implementation_hash() != code_hash:
        raise RuntimeError('New experiment code changed during this run')
    for path, expected in bank_files.items():
        if source.digest(path) != expected:
            raise RuntimeError('Source bank changed: ' + path)


def decode(probability, route, mass, output_size):
    """Input is probabilities, NOT logits; never apply a second softmax."""
    route = valid_read_route(route, mass)
    return unpool(probability, route['q'], route['ids'], route['shape'], output_size)


@torch.no_grad()
def extract(bundle, pre, post):
    reference = source.capture(bundle['front'], pre, post)
    pool_route, read_route = source.routes(reference, bundle['manifest'], 'hard', 'hard')
    encoded = encode_regions(bundle['heads']['hard'], reference['features']['roi'], pool_route)
    base = bundle['heads']['hard'].classifier(encoded['tokens']).float().softmax(-1)
    return reference, pool_route, read_route, encoded, base


@torch.no_grad()
def records(tokens, mass, bank, names, args, intervention='normal'):
    """Retrieval completed before any query GT is consumed. Small sparse cache."""
    rows = []
    for image_index, name in enumerate(names):
        image_id = bank['image_lookup'].get(name, -1)
        event_id = bank['event_lookup'].get(source.event(name), -1)
        seed = (args.seed + int(hashlib.sha256((name+intervention).encode()).hexdigest()[:8], 16)) % (2**63-1)
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        distribution = (shuffled_distribution(bank, event_id, image_id, generator)
                        if intervention == 'shuffle_labels' else None)
        active = (mass[image_index] > 0).nonzero().flatten()
        for begin in range(0, len(active), args.query_chunk):
            selected = active[begin:begin+args.query_chunk]
            support = select_neighbors(tokens[image_index, selected], bank, event_id, image_id,
                args.topk, intervention, generator, distribution=distribution)
            rows.append((image_index, selected, support))
    return rows


def read_probabilities(base, rows, head=None, temperature=1., alpha=.1, only=False):
    outputs = list(base.unbind(0))
    for index, selected, support in rows:
        current = base[index, selected]
        if head is not None:
            prediction = head(current, support['scores'], support['labels'], support['valid'])
        else:
            evidence, supported = weighted_labels(support['scores'], support['labels'], support['valid'], temperature)
            prediction = mix_fixed(current, evidence, supported, 1. if only else alpha)
        outputs[index] = outputs[index].index_copy(0, selected, prediction)
    return torch.stack(outputs)


@torch.no_grad()
def add_neighbor_quality(total, rows, base, route, gt, bank, names, temperature):
    """Offline region-majority diagnostic, NOT pixel accuracy/instances/upper bound."""
    distribution, labelled_mass = target_distribution(gt, route)
    for index, selected, support in rows:
        truth = distribution[index, selected].argmax(-1)
        base_correct = base[index, selected].argmax(-1) == truth
        evidence, supported = weighted_labels(support['scores'], support['labels'], support['valid'], temperature)
        labelled = labelled_mass[index, selected] > 0
        image_id = bank['image_lookup'].get(names[index], -1)
        event_id = bank['event_lookup'].get(source.event(names[index]), -1)
        allowed = (bank['images'] != image_id) & (bank['events'] != event_id)
        prior = bank['distribution'][allowed].mean(0) if allowed.any() else None
        for cls, name in enumerate(CLASS_NAMES):
            for condition, condition_mask in (('base_correct', base_correct), ('base_wrong', ~base_correct)):
                mask = labelled & (truth == cls) & condition_mask
                used = mask & supported
                row = total.setdefault(name, {}).setdefault(condition,
                    dict(queries=0, supported=0, correct_neighbor_votes=0, true_class_mass_sum=0., eligible_prior_sum=0.))
                row['queries'] += int(mask.sum())
                row['supported'] += int(used.sum())
                row['correct_neighbor_votes'] += int((used & (evidence.argmax(-1) == cls)).sum())
                row['true_class_mass_sum'] += float(evidence[used, cls].double().sum())
                if prior is not None:
                    row['eligible_prior_sum'] += int(used.sum()) * float(prior[cls])


def finish_neighbor_quality(total):
    for conditions in total.values():
        for row in conditions.values():
            n = row['supported']
            row['coverage'] = n/row['queries'] if row['queries'] else None
            row['vote_agreement'] = row['correct_neighbor_votes']/n if n else None
            row['true_class_mass'] = row['true_class_mass_sum']/n if n else None
            row['eligible_class_prior'] = row['eligible_prior_sum']/n if n else None
    return dict(by_class=total, weighting='one labelled occupied region, majority GT; stratified by ORIGINAL hard region correctness',
        caution='Offline GT diagnostic only, NOT building instances or full-image mIoU; no GT enters selection or prediction',
        test_accessed=False)


def add_pixel_complementarity(total, gt, base, neighbor_only, candidate):
    """Observe aligned label evidence AFTER prediction; never choose corrections with GT."""
    valid = (gt >= 0) & (gt < 4)
    bc, nc, cc = base == gt, neighbor_only == gt, candidate == gt
    masks = {'all':valid, 'foreground':valid & (gt>0)}
    masks.update({name:valid & (gt==cls) for cls,name in enumerate(CLASS_NAMES)})
    conditions = dict(base_wrong_neighbor_correct=~bc & nc,
        supported_rescue=~bc & nc & cc, unused_neighbor_opportunity=~bc & nc & ~cc,
        aligned_harm=bc & ~nc & ~cc, rescued=~bc & cc, harmed=bc & ~cc)
    for name, mask in masks.items():
        row = total.setdefault(name, {key:0 for key in conditions})
        for key, condition in conditions.items():
            row[key] += int((mask & condition).sum())


@torch.no_grad()
def evaluate(bundle, bank, data, args, heads=None, detailed=False, initial=False):
    for model in (bundle['front'], bundle['heads']['hard'], *(heads or {}).values()):
        model.eval()
    primary = ['nomemory', 'memory'] if heads is not None else ['knn_only', 'knn_mix']
    candidate = 'memory' if heads is not None else 'knn_mix'
    prefix = 'memory' if heads is not None else 'knn'
    cases = primary + ([prefix+'_random', prefix+'_shuffle_labels', prefix+'_off'] if detailed else [])
    if detailed and heads is not None:
        cases += ['knn_only']
    names_all = ['old_pixel', 'old_region', 'source_hard'] + cases
    matrices = {name: torch.zeros(4, 4, dtype=torch.long) for name in names_all}
    pairs = [('source_hard', name) for name in cases]
    if heads is not None:
        pairs += [('nomemory', 'memory')]
    if detailed:
        pairs += [(candidate, prefix+'_'+key) for key in ('random', 'shuffle_labels', 'off')]
    diagnostics = DiagnosticAccumulator(bundle['manifest']['arguments']['threshold'], pairs) if detailed else None
    quality, complementarity, image_count, replay_max = {}, {}, 0, 0.
    for batch in data:
        pre, post, _, gt, names = source.parse_batch(batch, bundle['device'])
        reference, pool_route, read_route, encoded, base = extract(bundle, pre, post)
        rows = records(encoded['tokens'], encoded['mass'], bank, names, args)
        base_pixel = decode(base, read_route, encoded['mass'], gt.shape[-2:])
        legacy = bundle['heads']['hard'](reference['features']['roi'], pool_route, gt.shape[-2:])
        replay_max = max(replay_max, float((base_pixel-legacy).abs().max()))
        torch.testing.assert_close(base_pixel, legacy, atol=2e-5, rtol=2e-4)
        predictions = dict(old_pixel=reference['pixel'].argmax(1), old_region=reference['prob'].argmax(1),
                           source_hard=base_pixel.argmax(1))
        temperature = heads['memory'].values()['temperature'] if heads is not None else args.temperature
        for name in cases:
            if name == prefix+'_off':
                probability = base
            elif name in ('nomemory', 'memory'):
                probability = read_probabilities(base, rows, head=heads[name])
            elif name == 'knn_only':
                probability = read_probabilities(base, rows, temperature=temperature, only=True)
            elif name == 'knn_mix':
                probability = read_probabilities(base, rows, temperature=temperature, alpha=args.mix_weight)
            else:
                intervention = name[len(prefix)+1:]
                alternative = records(encoded['tokens'], encoded['mass'], bank, names, args, intervention)
                probability = read_probabilities(base, alternative,
                    head=heads['memory'] if heads is not None else None, temperature=temperature, alpha=args.mix_weight)
            if not torch.isfinite(probability).all():
                raise RuntimeError('Nonfinite probability: '+name)
            if initial and heads is not None:
                torch.testing.assert_close(probability, base, atol=1e-7, rtol=1e-6)
            output = decode(probability, read_route, encoded['mass'], gt.shape[-2:])
            predictions[name] = output.argmax(1)
        for name, prediction in predictions.items():
            source.cm_update(matrices[name], gt, prediction)
        if diagnostics is not None:
            diagnostics.update(gt, predictions, reference['building'], names)
            add_neighbor_quality(quality, rows, base, pool_route, gt, bank, names, temperature)
            add_pixel_complementarity(complementarity, gt, predictions['source_hard'],
                                     predictions['knn_only'], predictions[candidate])
        image_count += len(names)
        if image_count == 1 or image_count % 25 == 0:
            print('evaluate label_memory', image_count, flush=True)
    if image_count != (2 if args.smoke else 450):
        raise RuntimeError('Expected smoke Val2 or formal Val450')
    result = dict(metrics={name: source.metrics(cm) for name, cm in matrices.items()},
        image_count=image_count, hard_replay_max_abs=replay_max, test_accessed=False)
    if diagnostics is not None:
        result.update(diagnostics=diagnostics.finalize(), retrieval_quality=finish_neighbor_quality(quality),
            label_evidence=dict(counts=complementarity,
                caution='Paired pixel association, not causal proof; GT read only after predictions; no GT gate',
                candidate=candidate, reference='source_hard'))
    return result


def persist_details(out, result):
    if 'diagnostics' in result:
        source.save_json(out/'audit_final.json', result.pop('diagnostics'))
        source.save_json(out/'retrieval_quality.json', result.pop('retrieval_quality'))
        source.save_json(out/'label_evidence.json', result.pop('label_evidence'))


def run(args):
    require_hard(args)
    source.validate_training_args(args)
    source.seed_everything(args.seed)
    begin = time.monotonic()
    code_hash = implementation_hash()
    bundle = source.load_source(args)
    policy = baseline_policy(bundle, args)  # must precede current inference
    identity = source.source_identity(bundle, args)
    bank_metadata = json.loads((Path(args.bank)/'manifest.json').read_text(encoding='utf-8'))
    bank_smoke = bool(bank_metadata.get('smoke'))
    if bank_smoke and not args.smoke:
        raise ValueError('A smoke bank cannot be used for formal evaluation/training')
    bank_cpu, bank_manifest = read_bank(args.bank, identity, bank_smoke, bundle['train'])
    if not bank_smoke and set(bank_cpu['image_names']) != set(bundle['train']):
        raise ValueError('Full bank must have visited all Train2918')
    bank = device_bank(bank_cpu, bundle['device'])
    bank_files = {str(Path(args.bank)/name): source.digest(Path(args.bank)/name) for name in ('bank.pt', 'manifest.json')}
    out = source.new_output(args.output)
    heads = {name: LabelMixer(name).to(bundle['device']) for name in ('nomemory', 'memory')} if args.mode == 'train' else None
    manifest = dict(format=FORMAT, arguments=vars(args), identity=identity, code_hash=code_hash,
        bank_file_sha256=bank_files, bank_smoke=bank_smoke, bank_regions=len(bank['tokens']),
        smoke=args.smoke, test_accessed=False, frozen=['front','B','Q','embedding','base_classifier','bank'],
        learned='memory: temperature and mixture; nomemory: temperature and foreground logit bias' if heads else 'none',
        parameter_count={n: sum(p.numel() for p in h.parameters()) for n,h in (heads or {}).items()},
        initialization='alpha=0, log_temperature=0 for BOTH; probability identity within FP32 precision',
        bounds='memory lambda[0,.5]; nomemory foreground bias[-.5,.5]; both temperature[.02,4]',
        comparison='equal scalar count and initialization, NOT identical functions or compute',
        exclusion='same image AND same event, including random and eligible-only per-image label shuffle',
        input='same frozen ROI content, hard building split and soft Q within groups',
        objective='same original pixel NLL+Lovasz; no new loss, no class weighting, no learned key',
        primary='same-step memory minus nomemory AND source_hard; paired all/foreground/Damaged rescue minus harm',
        selection='fixed final-step primary; best-Val secondary only; no outcome-guaranteeing tuning',
        interventions='same scalar reader; random legal supports use actual cosine, shuffled eligible labels; off is exact source',
        context='bank full Train1024, query random Train512, Val1024; context mismatch unchanged',
        checkpoint='two scalar readers only, no front copy/optimizer/exact resume')
    source.save_json(out/'manifest.json', manifest)
    source.save_json(out/'baseline_policy.json', policy)
    source.save_json(out/'summary.json', dict(format=FORMAT, complete=False, mode=args.mode,
        phase='initial_evaluation', smoke=args.smoke, test_accessed=False))
    validation = source.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
        workers=args.workers, seed=args.seed+3)
    print('LABEL MEMORY', args.mode, 'bank:', args.bank, 'parameters:', manifest['parameter_count'], flush=True)
    initial = evaluate(bundle, bank, validation, args, heads, detailed=args.mode == 'audit', initial=heads is not None)
    gate = initial_gate(bundle, initial, args, out, policy)
    verify_frozen(bundle, code_hash, bank_files)
    if args.mode == 'audit':
        persist_details(out, initial)
        source.save_json(out/'summary.json', dict(format=FORMAT, complete=True, mode='audit', final_step=0,
            image_count=initial['image_count'], final_metrics=initial['metrics'],
            smoke=args.smoke, test_accessed=False, baseline_reproduction_pass=gate['passed'],
            frozen_source_verified=True, elapsed_seconds=time.monotonic()-begin))
        print('LABEL AUDIT COMPLETE. No parameters updated. No Test.', flush=True)
        return
    train_names = bank['image_names'][:2] if args.smoke else bundle['train']
    training = source.loader(args.dataset, train_names, True, args.batch, args.workers, seed=args.seed+2)
    if not len(training):
        raise ValueError('Empty training loader; smoke batch<=2 required')
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=args.lr) for name,head in heads.items()}
    best, best_step = {n: -1. for n in heads}, {n: 0 for n in heads}
    gradients = {n: 0 for n in heads}
    iterator = iter(training)
    if bundle['device'].type == 'cuda':
        torch.cuda.reset_peak_memory_stats(bundle['device'])
    for step in range(1, args.steps+1):
        # A small fixed scalar-calibration budget, paired exact same batches.
        factor = .5*(1+math.cos(math.pi*(step-1)/max(1,args.steps-1)))
        losses = {n: 0. for n in heads}
        for name, head in heads.items():
            head.train(); optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr*factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training); batch = next(iterator)
            pre, post, _, gt, names = source.parse_batch(batch, bundle['device'])
            reference, pool_route, read_route, encoded, base = extract(bundle, pre, post)
            rows = records(encoded['tokens'], encoded['mass'], bank, names, args)
            for name, head in heads.items():
                region_prob = read_probabilities(base, rows, head=head)
                probability = decode(region_prob, read_route, encoded['mass'], gt.shape[-2:])
                loss = source.objective(probability, gt)/args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward(); losses[name] += float(loss.detach())
        norms = {}
        for name, head in heads.items():
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            norms[name] = float(norm); gradients[name] += int(float(norm)>0)
            optimizers[name].step(); head.project_()
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step*args.batch*args.accum, losses=losses,
                controls={n:h.values() for n,h in heads.items()}, grad_norm=norms,
                lr=args.lr*factor, elapsed_seconds=time.monotonic()-begin)
            source.append_json(out/'training.jsonl', row); print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, bank, validation, args, heads, detailed=step == args.steps)
            verify_frozen(bundle, code_hash, bank_files)
            persist_details(out, result)
            source.append_json(out/'validation.jsonl', dict(step=step, **result))
            for name, head in heads.items():
                checkpoint = dict(format=FORMAT, name=name, step=step, head=head.state_dict(),
                    manifest=manifest, metrics=result['metrics'][name])
                score = result['metrics'][name]['valid_miou']
                if score > best[name]:
                    best[name], best_step[name] = score, step
                    source.save_pt(out/(name+'_best.pth'), checkpoint)
                if step == args.steps:
                    source.save_pt(out/(name+'_last.pth'), checkpoint)
            source.save_json(out/'summary.json', dict(format=FORMAT, complete=False, mode='train',
                phase='training', optimizer_steps=step, smoke=args.smoke, test_accessed=False))
            print('VAL LABEL', step, {n:m['valid_miou'] for n,m in result['metrics'].items()}, flush=True)
    verify_frozen(bundle, code_hash, bank_files)
    source.save_json(out/'summary.json', dict(format=FORMAT, complete=True, mode='train', final_step=args.steps,
        image_count=result['image_count'], final_metrics=result['metrics'], best=best, best_steps=best_step,
        controls={n:h.values() for n,h in heads.items()}, nonzero_gradient_steps=gradients,
        note='Zero learned mixture is a VALID negative result; no minimum gain enforced',
        smoke=args.smoke, test_accessed=False, baseline_reproduction_pass=gate['passed'], frozen_source_verified=True,
        elapsed_seconds=time.monotonic()-begin,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(bundle['device']) if bundle['device'].type=='cuda' else None))
    print('LABEL TRAINING COMPLETE. Frozen source/bank unchanged. Inspect rescue AND harm; benefit is not guaranteed.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    add_arguments(parser, training=True)
    parser.set_defaults(steps=1000, eval_every=250, lr=.01)
    parser.add_argument('--mode', choices=('audit','train'), required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=.1, help='Fixed audit only; not selected from current Val')
    parser.add_argument('--mix_weight', type=float, default=.1, help='Fixed audit only; no automatic Val sweep')
    args = parser.parse_args()
    if (min(args.topk,args.query_chunk)<1 or not math.isfinite(args.temperature) or args.temperature<=0
            or not math.isfinite(args.mix_weight) or not 0<=args.mix_weight<=.5 or (args.smoke and args.batch>2)):
        parser.error('Invalid retrieval/mixture/smoke batch setting')
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
