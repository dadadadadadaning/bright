"""One paired optimizer engine; the classifier, frozen runtime and original loss are reused."""
from __future__ import annotations
import copy
import math
import torch
from anchor import anchor_terms, serialized_anchor_terms, gradient_alignment


def fork_pair(prototype, optimizer, state):
    """Independent parameter AND Adam moment storage; no model implementation copy."""
    heads, optimizers = {}, {}
    expected_head = state.state_fingerprint(prototype.state_dict())
    expected_optimizer = state.state_fingerprint(optimizer.state_dict())
    records = {}
    for name in ('control', 'anchor'):
        head = copy.deepcopy(prototype)
        opt = torch.optim.AdamW(head.parameters(), lr=optimizer.param_groups[0]['lr'])
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        heads[name], optimizers[name] = head, opt
        actual = (state.state_fingerprint(head.state_dict()), state.state_fingerprint(opt.state_dict()))
        if actual != (expected_head, expected_optimizer):
            raise ValueError('Pair restoration differs: ' + name)
        records[name] = dict(head_sha256=actual[0], optimizer_sha256=actual[1])
    for left, right in zip(heads['control'].parameters(), heads['anchor'].parameters()):
        if left.data_ptr() == right.data_ptr():
            raise ValueError('Pair shares model storage')
        ls, rs = optimizers['control'].state.get(left, {}), optimizers['anchor'].state.get(right, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in ls and ls[key].data_ptr() == rs[key].data_ptr():
                raise ValueError('Pair shares Adam moment storage')
    return heads, optimizers, records


def train_step(heads, optimizers, example, runtime, *, step, arm, state,
               maximum_lr=1e-4, lr_multiplier=.2, anchor_weight=.1,
               diagnose=False, prefix=False):
    """Same frozen example and RNG for each branch; runtime.loss remains unchanged.

    prefix=True updates one shared original-objective head before branching.
    Step is the actual Adam update count, not a restarted suffix counter.
    """
    expected = {'control'} if prefix else {'control', 'anchor'}
    if set(heads) != expected or set(optimizers) != expected:
        raise ValueError('Wrong paired/prefix model set')
    if arm not in ('b', 'd') or not 1 <= step <= 1500:
        raise ValueError('Invalid arm or update index')
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError('Invalid anchor weight')
    before = state.capture_rng()
    after, rows = None, {}
    for name, head in heads.items():
        state.restore_rng(before)
        optimizer = optimizers[name]
        head.train()
        lr = state.lr_at_step(step, maximum=maximum_lr, horizon=5000,
                              multiplier=lr_multiplier)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        result = head(**example.common, **example.supports[arm])
        original = runtime.loss(result['probabilities'], example.target)
        if original.ndim or not bool(torch.isfinite(original)):
            raise ValueError('Nonfinite/nonscalar inherited loss')
        # Score the SAME candidate objective on both branches, but control never
        # differentiates it. No ground-truth-dependent acceptance/filtering.
        with torch.set_grad_enabled((name == 'anchor' and anchor_weight > 0) or diagnose):
            terms = anchor_terms(result['probabilities'], example.common['base_probability_full'],
                                 example.target, weight=anchor_weight, student_logits=result['logits'])
        loss = original if name == 'control' else original + terms['weighted_kl']
        alignment = (gradient_alignment(original, terms['weighted_kl'], result['logits'], space='logits')
                     if diagnose and anchor_weight > 0 else None)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite combined loss')
        loss.backward()
        parameters = list(head.parameters())
        old = [p.detach().clone() for p in parameters] if diagnose else None
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        reader_gradient = sum(float(p.grad.detach().abs().sum())
                              for key, p in head.named_parameters()
                              if p.grad is not None and key.startswith(('query_projection', 'key_projection',
                                  'value_projection', 'label_embedding', 'context_projection')))
        optimizer.step()
        actual = [int(v['step'].item()) if torch.is_tensor(v['step']) else int(v['step'])
                  for v in optimizer.state.values() if 'step' in v]
        if not actual or any(count != step for count in actual):
            raise ValueError('Adam update count mismatch')
        if not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError('Updated weights are not finite')
        update = (math.sqrt(sum(float((p.detach().double()-q.double()).square().sum())
                               for p, q in zip(parameters, old))) if diagnose else None)
        rows[name] = dict(original_loss=float(original.detach()),
                          scored_candidate_loss=float((original.detach()+terms['weighted_kl'].detach())),
                          training_loss=float(loss.detach()),
                          trained_objective='original' if name == 'control' else 'original_plus_grid_KL',
                          anchor=serialized_anchor_terms(terms), lr=lr, optimizer_step=step,
                          grad_norm_preclip=float(norm), clip_threshold=1.,
                          reader_grad_abs_sum=reader_gradient,
                          parameter_update_l2=update, gradient_alignment=alignment)
        current = state.capture_rng()
        if after is not None and state.state_fingerprint(current) != state.state_fingerprint(after):
            raise ValueError('Paired branches consumed different random streams')
        after = current
    state.restore_rng(after)
    return rows
