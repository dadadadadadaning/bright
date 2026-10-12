"""E0/E1 encoder training, E2/E3 paired heads, E4/E5 interventions. No pixel writeback."""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import Pooler, SupportHead, batch, fingerprint, load_records, metrics, retrieve, select_bank


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


@torch.no_grad()
def encode(model, data, sources, device, size):
    model.eval()
    z, preds = [], []
    for start in range(0, len(data['labels']), size):
        ids = torch.arange(start, min(start + size, len(data['labels'])))
        features, logits = model(*batch(data, ids, sources, device))
        z.append(features.cpu()); preds.append(logits.argmax(1).cpu())
    return torch.cat(z), torch.cat(preds)


def report(data, pred):
    y = data['labels'] - 1
    result = metrics(y, pred)
    result['per_event'] = {}
    for event in sorted(set(data['events'])):
        ids = torch.tensor([i for i, e in enumerate(data['events']) if e == event])
        result['per_event'][event] = metrics(y[ids], pred[ids])
    result['event_macro_f1'] = sum(v['macro_f1'] for v in result['per_event'].values()) / len(result['per_event'])
    if 'p3_region_labels' in data:
        base = data['p3_region_labels'].long() - 1
        correct, new = base == y, pred == y
        result['base_region_comparison'] = dict(
            baseline=metrics(y, base.clamp(0, 2)) if (base >= 0).all() else None,
            rescued=int((~correct & new).sum()), harmed=int((correct & ~new).sum()),
            base_wrong_count=int((~correct).sum()), base_correct_count=int(correct.sum()),
            net_correction=int(new.sum() - correct.sum()))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--train', required=True); p.add_argument('--val', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sources', default='base_change,explicit_diff')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--per_bucket', type=int, default=64)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.per_bucket, args.topk, args.repeats, args.eval_every) < 1 or args.dim % 4:
        p.error('Positive counts and dim divisible by four required')
    for path in (args.train, args.val):
        if not Path(path).is_file():
            p.error(f'Missing region cache: {path}; generate matching Train/Val caches first')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive ownership protects shared caches/checkpoints from duplicate launchers.
    with (out / 'RUN_STARTED.json').open('x', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    sources = args.sources.split(',')
    train = load_records(args.train, 'train', sources)
    val = load_records(args.val, 'validation', sources)
    if set(train['images']) & set(val['images']):
        raise ValueError('Train/Val image overlap')
    dims = [train['tokens'][s].shape[-1] for s in sources]
    if dims != [val['tokens'][s].shape[-1] for s in sources]:
        raise ValueError('Train/Val channel mismatch')
    for field in ('base_checkpoint', 'base_sha256', 'base_model', 'roi_alpha', 'roi_variant', 'region_mode', 'segments', 'max_region_tokens', 'token_sampling', 'feature_source'):
        a, b = train.get('metadata', {}).get(field), val.get('metadata', {}).get(field)
        if a != b:
            raise ValueError(f'Cache provenance mismatch: {field}')
    manifest = dict(protocol='SRAC_SUPPORT_ABLATION_V1', arguments=vars(args),
                    train_sha256=fingerprint(args.train), val_sha256=fingerprint(args.val),
                    train_metadata=train.get('metadata', {}), val_metadata=val.get('metadata', {}),
                    metric_scope='sampled-region classification; NOT pixel mIoU',
                    test_accessed=False, exclude_same_event=True, exclude_same_image=True)
    save_json(out / 'manifest.json', manifest)
    summary = {}
    def emit(name, pred, extra=None):
        value = dict(variant=name, **report(val, pred), test_accessed=False,
                     metric_scope=manifest['metric_scope'], **(extra or {}))
        save_json(out / f'{name}.json', value)
        torch.save(pred, out / f'{name}_predictions.pt')
        summary[name] = value
        print(f'{name}: accuracy={value["accuracy"]:.6f} macro_f1={value["macro_f1"]:.6f}', flush=True)
    n = len(train['labels'])
    for name, weighted in [('E0', False), ('E1', True)]:
        torch.manual_seed(args.seed)
        model = Pooler(sum(dims), args.dim, weighted).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        gen = torch.Generator().manual_seed(args.seed)
        best = -1.
        for step in range(1, args.steps + 1):
            model.train()
            ids = torch.randint(n, (args.batch_size,), generator=gen)
            _, logits = model(*batch(train, ids, sources, device))
            loss = F.cross_entropy(logits, (train['labels'][ids] - 1).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); schedule.step()
            if step == 1 or step % 100 == 0:
                print(f'{name} {step}/{args.steps} loss={loss.item():.5f}', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                _, pred = encode(model, val, sources, device, args.batch_size)
                score = metrics(val['labels'] - 1, pred)['macro_f1']
                print(f'{name} validation step={step} macro_f1={score:.6f}', flush=True)
                state = dict(model=model.state_dict(), step=step, score=score, args=vars(args))
                if score > best:
                    best = score; torch.save(state, out / f'{name}_best.pth')
                if step == args.steps:
                    torch.save(state, out / f'{name}_last.pth')
        model.load_state_dict(torch.load(out / f'{name}_best.pth', map_location=device, weights_only=False)['model'])
        _, pred = encode(model, val, sources, device, args.batch_size)
        emit(name, pred, {'parameters': sum(p.numel() for p in model.parameters())})
    # Frozen E1 embeddings shared by E2 and E3. Supports never depend on Val labels.
    tz, _ = encode(model, train, sources, device, args.batch_size)
    vz, _ = encode(model, val, sources, device, args.batch_size)
    bank_ids = select_bank(train, args.per_bucket, args.seed)
    bank = tz[bank_ids].to(device)
    labels = (train['labels'][bank_ids] - 1).to(device)
    be = [train['events'][i] for i in bank_ids.tolist()]
    bi = [train['images'][i] for i in bank_ids.tolist()]
    torch.save(dict(ids=bank_ids, features=bank.cpu(), labels=labels.cpu(), events=be, images=bi,
                    source_sha256=manifest['train_sha256']), out / 'train_memory.pt')
    def neighbors(z, data, random_mode=False, seed=0):
        result = []
        g = torch.Generator().manual_seed(seed)
        for start in range(0, len(z), args.batch_size):
            stop = min(start + args.batch_size, len(z))
            result.append(retrieve(z[start:stop].to(device), bank, data['events'][start:stop],
                                  data['images'][start:stop], be, bi, args.topk,
                                  random_mode, g).cpu())
            if start == 0 or stop == len(z) or (start // args.batch_size) % 25 == 0:
                print(f'neighbors split={data["split"]} random={random_mode} {stop}/{len(z)}', flush=True)
        return torch.cat(result)
    print('Building cross-event neighbor tables', flush=True)
    with torch.no_grad():
        tn, vn = neighbors(tz, train), neighbors(vz, val)
    torch.save(dict(train=tn, validation=vn), out / 'neighbor_ids.pt')
    torch.manual_seed(args.seed)
    initial = SupportHead(args.dim).state_dict()
    @torch.no_grad()
    def predict(head, indices=None, bank_labels=None):
        head.eval(); result = []
        for start in range(0, len(vz), args.batch_size):
            q = vz[start:start + args.batch_size].to(device)
            if indices is None:
                logits = head(q)
            else:
                ix = indices[start:start + args.batch_size].to(device)
                logits = head(q, bank[ix], bank_labels[ix])
            result.append(logits.argmax(1).cpu())
        return torch.cat(result)
    for name in ('E2', 'E3'):
        head = SupportHead(args.dim).to(device); head.load_state_dict(copy.deepcopy(initial))
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        gen = torch.Generator().manual_seed(args.seed); best = -1.
        for step in range(1, args.steps + 1):
            head.train(); ids = torch.randint(n, (args.batch_size,), generator=gen)
            q = tz[ids].to(device); ix = tn[ids].to(device)
            logits = head(q) if name == 'E2' else head(q, bank[ix], labels[ix])
            loss = F.cross_entropy(logits, (train['labels'][ids] - 1).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            opt.step(); schedule.step()
            if step == 1 or step % 100 == 0:
                print(f'{name} {step}/{args.steps} loss={loss.item():.5f}', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                pred = predict(head, vn if name == 'E3' else None, labels)
                score = metrics(val['labels'] - 1, pred)['macro_f1']
                print(f'{name} validation step={step} macro_f1={score:.6f}', flush=True)
                state = dict(model=head.state_dict(), step=step, score=score, args=vars(args))
                if score > best:
                    best = score; torch.save(state, out / f'{name}_best.pth')
                if step == args.steps:
                    torch.save(state, out / f'{name}_last.pth')
        head.load_state_dict(torch.load(out / f'{name}_best.pth', map_location=device, weights_only=False)['model'])
        emit(name, predict(head, vn if name == 'E3' else None, labels),
             {'parameters': sum(p.numel() for p in head.parameters()), 'encoder_frozen': True})
    for repeat in range(args.repeats):
        seed = args.seed + 10000 + repeat
        rn = neighbors(vz, val, True, seed)
        emit(f'E4_random_{repeat}', predict(head, rn, labels), {'intervention_seed': seed})
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(labels), generator=g).to(device)
        shuffled = labels[perm]
        emit(f'E5_labels_{repeat}', predict(head, vn, shuffled),
             {'intervention_seed': seed, 'changed_label_fraction': float((shuffled != labels).float().mean())})
    summary['comparisons_pp'] = {f'{a}_minus_{b}': 100 * (summary[a]['macro_f1'] - summary[b]['macro_f1'])
                                for a, b in [('E1','E0'), ('E2','E1'), ('E3','E2')]}
    for prefix in ('E4_random', 'E5_labels'):
        scores = torch.tensor([summary[f'{prefix}_{i}']['macro_f1'] for i in range(args.repeats)])
        summary[prefix + '_aggregate'] = dict(mean=float(scores.mean()), std=float(scores.std(unbiased=False)),
                                             e3_minus_mean_pp=100 * (summary['E3']['macro_f1'] - float(scores.mean())))
    save_json(out / 'summary.json', summary)
    print('E0-E5 ALL COMPLETE. Region metrics only. Test access: FORBIDDEN.', flush=True)


if __name__ == '__main__':
    main()
