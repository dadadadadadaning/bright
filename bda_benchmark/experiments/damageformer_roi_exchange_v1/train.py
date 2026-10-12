"""Frozen V2, exact original SoftROI content; matched hard versus bounded-exchange readouts."""
import argparse
import hashlib
import json
import math
import time
import torch
from .core import DEFAULT_VARIANTS, VARIANTS, capture, partition, make_heads, routing_stats, implementation_hash, verify_hard_parity
from .diagnostics import DiagnosticAccumulator
from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import (
    load_fixed, lists, loader, parse_batch, metrics, cm_update, seed_everything,
    new_output, save_json, save_pt, source_hash, digest,
)
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective


def prepare_device(spec):
    """Initialize CUDA before allocator statistics (required by torch 2.2.x).

    Unlike model.to(cuda), reset_peak_memory_stats does not reliably trigger
    lazy initialization. Fail explicitly if CUDA was requested but unavailable.
    This changes resource bookkeeping only, not RNG, model or optimizer setup.
    """
    device = torch.device(spec)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; check interpreter, driver and visible GPUs')
        torch.cuda.init()
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def state_digest(model):
    """Hash parameters and buffers without retaining a second front checkpoint."""
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def assert_front(model, buffers):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen front is training, trainable, or has gradients')
    for name, value in model.named_buffers():
        if not torch.equal(value.detach().cpu(), buffers[name]):
            raise RuntimeError('Frozen buffer changed: '+name)


def route(reference, name, model, args):
    return partition(reference, name, epsilon=args.epsilon, threshold=args.threshold, stride=model.config['stride'])


@torch.no_grad()
def evaluate(model, heads, data, device, args, detailed=False, check_parity=False):
    model.eval()
    for head in heads.values():
        head.eval()
    cms = {name: torch.zeros(4, 4, dtype=torch.long) for name in ('old_pixel', 'old_region', *heads)}
    stats = {name: {} for name in heads}
    pairs = [('hard', 'exchange')] if 'hard' in heads and 'exchange' in heads else []
    pairs += [('old_pixel', name) for name in heads]
    diagnostic = DiagnosticAccumulator(threshold=args.threshold, pairs=pairs) if detailed else None
    count, identity_error, replay_error = 0, 0., 0.
    parity_report = None
    for batch in data:
        pre, post, _, gt, names = parse_batch(batch, device)
        reference = capture(model, pre, post)
        if check_parity and parity_report is None:
            parity_head = heads.get('hard', next(iter(heads.values())))
            before = state_digest(parity_head)
            # The verifier uses independent copies. Explicitly allow its backward
            # check despite evaluation's no_grad; never train the real front/head.
            with torch.enable_grad():
                parity_report = verify_hard_parity(
                    reference['features']['roi'], reference, parity_head, gt.shape[-2:],
                    stride=model.config['stride'], threshold=args.threshold)
            if state_digest(parity_head) != before or any(p.grad is not None for p in parity_head.parameters()):
                raise RuntimeError('Hard-parity verifier mutated live head state/gradients')
        identity_error = max(identity_error, reference['identity_max_abs'])
        replay_error = max(replay_error, verify_original_readout(model, reference, gt.shape[-2:]))
        predictions = {'old_pixel': reference['pixel'].argmax(1), 'old_region': reference['prob'].argmax(1)}
        for name, head in heads.items():
            routing = route(reference, name, model, args)
            predictions[name] = head(reference['features']['roi'], routing, gt.shape[-2:]).argmax(1)
            for key, value in routing_stats(routing, reference['building']).items():
                stats[name][key] = stats[name].get(key, 0.) + float(value) * len(pre)
        for name, prediction in predictions.items():
            cm_update(cms[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], names)
        count += len(pre)
        if count % 25 == 0:
            print('evaluate', count, flush=True)
    result = dict(metrics={name: metrics(cm) for name, cm in cms.items()}, image_count=count,
                  routing={name: {key: value/max(1,count) for key,value in row.items()} for name,row in stats.items()},
                  identity_max_abs=identity_error, region_replay_max_abs=replay_error, test_accessed=False)
    if check_parity:
        if parity_report is None:
            raise RuntimeError('Initial validation empty; hard/S2 parity was not checked')
        result['hard_s2_parity'] = parity_report
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
    return result


def append_json(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()


def main():
    wall_start = time.monotonic()
    parser = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=.05)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.eval_every,args.batch,args.accum) <= 0 or args.batch*args.accum != 16:
        parser.error('Positive settings and single-device batch*accum=16 required')
    if not math.isfinite(args.lr) or args.lr <= 0 or args.workers < 0:
        parser.error('Invalid lr/workers')
    if not 0 < args.threshold < 1 or len(set(args.variants)) != len(args.variants):
        parser.error('Invalid threshold or duplicate variants')
    if not math.isfinite(args.epsilon) or not 0 <= args.epsilon <= .5:
        parser.error('epsilon must be finite in [0, 0.5]')
    if args.smoke and args.steps != 20:
        parser.error('Smoke is fixed to 20 optimizer steps; pass --steps 20')
    train, val = lists(args.train_list, args.val_list)
    seed_everything(args.seed)
    device = prepare_device(args.device)
    model, checkpoint, sha = load_fixed(args.checkpoint, device)
    out = new_output(args.output)
    heads = make_heads(args.variants, model.config['dim'], device)
    initial_head_hashes = {name:state_digest(head) for name,head in heads.items()}
    if len(set(initial_head_hashes.values())) != 1:
        raise RuntimeError('Readouts did not initialize identically')
    parameter_sets = [set(id(p) for p in head.parameters()) for head in heads.values()]
    if any(a & b for i,a in enumerate(parameter_sets) for b in parameter_sets[i+1:]):
        raise RuntimeError('Readouts share parameters')
    optimizers = {name:torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.005) for name,head in heads.items()}
    buffers = {name:value.detach().cpu().clone() for name,value in model.named_buffers()}
    front_hash = state_digest(model)
    manifest = dict(arguments=vars(args), format='ROI_EXCHANGE_V1',
                    checkpoint_sha256=sha, checkpoint_step=checkpoint['step'],
                    frontend_source_hash=source_hash(), experiment_hash=implementation_hash(),
                    initial_front_state_sha256=front_hash, initial_head_sha256=initial_head_hashes,
                    train_sha256=digest(args.train_list), val_sha256=digest(args.val_list),
                    feature_source='exact original pixel-head input including existing SoftROI',
                    frozen_front=True,frozen_building=True,frozen_base_q=True,learned_partition=False,
                    test_accessed=False,smoke=args.smoke,dim=model.config['dim'],stride=model.config['stride'],
                    torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                    resource_scope='combined frozen front and all selected independent heads; not per-method cost',
                    parameters={name:sum(p.numel() for p in head.parameters()) for name,head in heads.items()},
                    variants=dict(hard='predicted building/background split, epsilon=0',
                                  exchange='same hard groups with fixed bounded cross-group exchange'),
                    effective_epsilon={name:0. if name=='hard' else args.epsilon for name in heads},
                    epsilon=args.epsilon,
                    routing_contract='both variants use 18 candidate entries and 2K slots; frozen Q/B',
                    loss='region NLL+Lovasz; no new auxiliary loss or Memory',
                    primary_endpoint='same-step final exchange-hard contrast; best is secondary',
                    budget_caution='Equal slot ceilings do not imply equal occupied region counts',
                    checkpoint_policy='best+last readout-only; no optimizer/front/cache copy',
                    schedule='AdamW weight_decay=.005; 100-step warmup (short smoke adapts); cosine; clip=1')
    save_json(out/'manifest.json', manifest)
    save_json(out/'summary.json', dict(complete=False,smoke=args.smoke,test_accessed=False))
    print(json.dumps(manifest,indent=2),flush=True)
    training = loader(args.dataset,train,True,args.batch,args.workers,seed=args.seed+2)
    validation = loader(args.dataset,val[:2] if args.smoke else val,workers=args.workers,seed=args.seed+3)
    initial = evaluate(model,heads,validation,device,args,check_parity=True)
    assert_front(model,buffers)
    save_json(out/'initial.json',initial)
    reproduced = initial['metrics']['old_region']['valid_miou']
    if not args.smoke and (initial['image_count'] != 450 or not math.isfinite(reproduced) or
                           abs(reproduced-checkpoint['valid_miou']) > 1e-5):
        raise RuntimeError('Full Val checkpoint reproduction failed; do not bypass')
    best = {name:-1. for name in heads}
    best_steps = {name:0 for name in heads}
    iterator = iter(training)
    start = time.monotonic()
    for step in range(1,args.steps+1):
        model.eval()
        warm = min(100,max(1,args.steps//10))
        factor = step/warm if step <= warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,args.steps-warm)))
        totals = {name:0. for name in heads}
        for name,head in heads.items():
            head.train()
            optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr*factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training)
                batch = next(iterator)
            pre,post,_,gt,_ = parse_batch(batch,device)
            with torch.no_grad():
                reference = capture(model,pre,post)
            for name,head in heads.items():
                routing = route(reference,name,model,args)
                loss = objective(head(reference['features']['roi'],routing,gt.shape[-2:]),gt)/args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward()
                totals[name] += float(loss.detach())
        for name,head in heads.items():
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizers[name].step()
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen front has gradients')
        if step == 1 or step % 20 == 0:
            row = dict(step=step,crops=step*16,losses=totals,lr=args.lr*factor,elapsed_seconds=time.monotonic()-start)
            append_json(out/'training.jsonl',row)
            print(row,flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model,heads,validation,device,args,detailed=step==args.steps)
            assert_front(model,buffers)
            if result['image_count'] != (2 if args.smoke else 450):
                raise RuntimeError('Validation image count changed; expected Val450 or smoke Val2')
            if 'diagnostics' in result:
                save_json(out/'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            append_json(out/'validation.jsonl',dict(step=step,**result))
            for name,head in heads.items():
                value = result['metrics'][name]['valid_miou']
                state = dict(head=head.state_dict(),step=step,name=name,metrics=result['metrics'][name],manifest=manifest)
                if value > best[name]:
                    best[name],best_steps[name] = value,step
                    save_pt(out/(name+'_best.pth'),state)
                if step == args.steps:
                    save_pt(out/(name+'_last.pth'),state)
            print('VAL',step,{name:m['valid_miou'] for name,m in result['metrics'].items()},flush=True)
            print('ROUTING',result['routing'],flush=True)
    final_hash = state_digest(model)
    if final_hash != front_hash or digest(args.checkpoint) != sha:
        raise RuntimeError('Front state or input checkpoint changed during experiment')
    if source_hash() != manifest['frontend_source_hash'] or implementation_hash() != manifest['experiment_hash']:
        raise RuntimeError('Implementation changed during experiment')
    save_json(out/'summary.json',dict(complete=True,final_step=args.steps,final_metrics=result['metrics'],
                                    best=best,best_steps=best_steps,smoke=args.smoke,test_accessed=False,
                                    final_front_state_sha256=final_hash,immutable_front_verified=True,
                                    elapsed_seconds=time.monotonic()-wall_start,
                                    train_and_periodic_eval_seconds=time.monotonic()-start,
                                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                                    resource_scope=manifest['resource_scope'],diagnostics_file='audit_final.json'))
    print('ROI EXCHANGE COMPLETE. Fixed-step primary; best secondary. Train/Val only.',flush=True)


if __name__ == '__main__':
    main()


