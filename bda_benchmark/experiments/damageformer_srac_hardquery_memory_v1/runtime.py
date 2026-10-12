"""Shared full-image evaluation on one frozen J1 cohort for every candidate."""
from __future__ import annotations

import torch

from .reader import retrieve_evidence
from .prepare_queries import pool_base_probability
from .io_utils import append_json, read_json, digest, check_artifact
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import unpool_region_tokens
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import decoder_forward, upsample_probability
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support, RetrievalProjector
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    parse_batch, event, confusion_update, region_targets)
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
    transition_counts, transition_report, reproduction_report, project_label_evidence,
    cohort_evidence, region_retrieval_report, add_nested_counts)
from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.cohorts import difficult_retrieval


BASE_NAMES = ('a1_pixel', 'direct_region', 'j1', 'j2', 'j3')
TRAIN_ARMS = ('query_ungated', 'query_gated', 'bank_ungated', 'bank_gated', 'hard_ungated', 'hard_gated')


def query_only_evidence(evidence):
    """Same availability mask, but no historical labels, scores or quality features.

    This capacity control keeps query/base inputs and identical head structure.
    Apply AFTER support dropout so recomputation cannot restore Memory features.
    """
    result = dict(evidence)
    result['posterior'] = torch.zeros_like(evidence['posterior'])
    result['quality'] = torch.zeros_like(evidence['quality'])
    return result


def corrected_logits(base_logits, region_delta, result):
    delta = unpool_region_tokens(region_delta[None], result['q'], result['ids'], result['shape'])
    return base_logits + delta


def load_new_metrics(directory, bundle, smoke=False):
    from pathlib import Path
    root = Path(directory)
    summary = read_json(root / 'summary.json')
    check_artifact(summary, bundle, allow_smoke=smoke)
    expected = 20 if summary.get('smoke') else 2000
    if summary.get('steps') != expected or set(summary.get('arms', [])) != {'bank', 'hard'}:
        raise ValueError('Reader comparison requires completed equal-budget bank and hard metrics')
    output, ledger = {}, {}
    for arm in ('bank', 'hard'):
        path = root / ('metric_' + arm + '_last.pth')
        checksum = digest(path)
        # A final metric checkpoint is bound to its manifest/summary, not merely
        # accepted because a tensor with the right shape can be loaded.
        recorded = summary.get('checkpoint_sha256', {}).get(arm)
        if recorded != checksum:
            raise ValueError('Metric checkpoint hash mismatch: ' + arm)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        check_artifact(payload, bundle, allow_smoke=smoke)
        if (payload.get('format') != 'SRAC_HARDQUERY_METRIC_V1' or payload.get('arm') != arm
                or payload.get('step') != expected):
            raise ValueError('Wrong metric format/arm/budget')
        projector = RetrievalProjector(int(payload['dim'])).to(bundle['device'])
        projector.load_state_dict(payload['model'], strict=True)
        output[arm] = projector.eval().requires_grad_(False)
        ledger[arm] = checksum
    return output, ledger


@torch.no_grad()
def evaluate(bundle, capture, data, guard, *, projectors=None, heads=None,
             temperature=.1, image_log=None, smoke=False):
    projectors, heads = projectors or {}, heads or {}
    for head in heads.values():
        head.eval()
    cm, by_event, transitions, event_transitions = {}, {}, {}, {}
    cohorts, retrieval, difficult, event_cohorts = {}, {}, {}, {}
    maxima = {name: 0.0 for name in heads}
    gate_totals = {name: dict(sum=0., count=0, supported=0) for name in heads}
    count = 0
    for batch in data:
        pre, post, _, target, names = parse_batch(batch, bundle['device'])
        name = str(names[0])
        guard.set_image(name)
        result, feature = capture(pre, post)
        event_name = event(name)
        events = bundle['bank']['event_names']
        event_id = events.index(event_name) if event_name in events else -1
        evidence = {
            'cosine': retrieve_evidence(result['tokens'][0], bundle['bank'], event_id, -1,
                                       bundle['topk'], None, temperature),
            'old_metric': retrieve_evidence(result['tokens'][0], bundle['bank'], event_id, -1,
                                           bundle['topk'], bundle['metric'], temperature)}
        cosine_support = gather_support(evidence['cosine']['selected'], bundle['bank'])
        metric_support = gather_support(evidence['old_metric']['selected'], bundle['bank'])
        probability = dict(a1_pixel=result['old_pixel'].softmax(1), direct_region=result['prob'])
        base_logits = None
        for arm in ('j1', 'j2', 'j3'):
            decoded = decoder_forward(bundle['decoders'][arm], arm, feature, result, cosine_support, metric_support)
            probability[arm] = upsample_probability(decoded['logits'], target.shape[-2:])
            if arm == 'j1':
                base_logits = decoded['logits']
        for key, projector in projectors.items():
            evidence[key] = retrieve_evidence(result['tokens'][0], bundle['bank'], event_id, -1,
                                              bundle['topk'], projector, temperature)
        base_region, _ = pool_base_probability(probability['j1'], result)
        for arm, head in heads.items():
            key, mode = arm.rsplit('_', 1)
            item = query_only_evidence(evidence['bank']) if key == 'query' else evidence[key]
            delta, gate, _ = head(result['tokens'][0], base_region, item, mode=mode)
            logits = corrected_logits(base_logits, delta, result)
            probability[arm] = upsample_probability(logits, target.shape[-2:])
            maxima[arm] = max(maxima[arm], float((probability[arm] - probability['j1']).abs().max()))
            totals = gate_totals[arm]
            totals['sum'] += float(gate.sum())
            totals['count'] += gate.numel()
            totals['supported'] += int(item['supported'].sum())
        projected = {}
        for key, item in evidence.items():
            cond, support_mass, prediction = project_label_evidence(
                item['posterior'], item['supported'], result['q'], result['ids'], result['shape'], target.shape[-2:])
            projected[key] = (cond, support_mass, prediction)
            # Retain original J1 where Q has no usable evidence. This is an
            # explicit fixed diagnostic mixture, not a calibrated probability model.
            probability['label_' + key] = cond * support_mass[:, None] + probability['j1'] * (1 - support_mass[:, None])
        predictions = {key: value.argmax(1) for key, value in probability.items()}
        per_image = {}
        for key, prediction in predictions.items():
            value = probability[key]
            if not torch.isfinite(value).all() or not torch.allclose(value.sum(1), torch.ones_like(value[:, 0]), atol=2e-5, rtol=0):
                raise ValueError('Invalid evaluation probability: ' + key)
            current = torch.zeros(4, 4, dtype=torch.long)
            confusion_update(current, target, prediction)
            cm.setdefault(key, torch.zeros_like(current)).add_(current)
            by_event.setdefault(key, {}).setdefault(event_name, torch.zeros_like(current)).add_(current)
            per_image[key] = current.tolist()
            if key != 'j1':
                change = transition_counts(target, predictions['j1'], prediction)
                transitions.setdefault(key, torch.zeros_like(change)).add_(change)
                event_transitions.setdefault(key, {}).setdefault(event_name, torch.zeros_like(change)).add_(change)
        # GT only enters retrieval reporting after every prediction is fixed.
        distribution, weight = region_targets(target, result)
        for key, item in evidence.items():
            add_nested_counts(retrieval.setdefault(key, {}), region_retrieval_report(
                item['selected'], bundle['bank'], distribution[0], weight[0]))
            add_nested_counts(difficult.setdefault(key, {}), difficult_retrieval(
                target, predictions['j1'], item['selected'], bundle['bank'], result['q'], result['ids'], result['shape']))
            candidate_names = ['label_' + key]
            if key == 'cosine':
                candidate_names += ['j2']
            if key == 'old_metric':
                candidate_names += ['j3']
            candidate_names += [arm for arm in heads if arm.startswith(key + '_')]
            cond, support_mass, evidence_pred = projected[key]
            for arm in candidate_names:
                row = cohort_evidence(target, predictions['j1'], predictions[arm], evidence_pred, support_mass, cond)
                add_nested_counts(cohorts.setdefault(arm, {}), row)
                add_nested_counts(event_cohorts.setdefault(arm, {}).setdefault(event_name, {}), row)
        for arm in heads:
            if arm.startswith('query_'):
                no_mass = torch.zeros_like(probability['j1'][:, 0])
                no_probability = torch.zeros_like(probability['j1'])
                no_prediction = predictions['j1'].new_full(predictions['j1'].shape, -1)
                row = cohort_evidence(target, predictions['j1'], predictions[arm], no_prediction, no_mass, no_probability)
                add_nested_counts(cohorts.setdefault(arm, {}), row)
                add_nested_counts(event_cohorts.setdefault(arm, {}).setdefault(event_name, {}), row)
        count += 1
        if image_log is not None:
            append_json(image_log, dict(index=count - 1, name=name, event=event_name, confusion=per_image))
        if count % 25 == 0:
            print('Validation %d' % count, flush=True)
    actual = {key: metrics(value) for key, value in cm.items()}
    reproduction = reproduction_report(actual, bundle['reference'], smoke=smoke, image_count=count)
    return dict(image_count=count, metrics=actual, reproduction=reproduction,
                per_event={k: {e: metrics(v) for e, v in rows.items()} for k, rows in by_event.items()},
                transitions={k: transition_report(v) for k, v in transitions.items()},
                per_event_transitions={k: {e: transition_report(v) for e, v in rows.items()} for k, rows in event_transitions.items()},
                difficult_cohorts=cohorts, per_event_difficult_cohorts=event_cohorts,
                retrieval=retrieval, difficult_retrieval=difficult,
                max_abs_probability_to_fixed_j1=maxima, gate_totals=gate_totals,
                reference_cohort='same immutable J1 for all arms and all training steps',
                evidence_definition='Similarity-weighted Train soft labels, not Reader attention or calibrated posterior',
                test_accessed=False)
