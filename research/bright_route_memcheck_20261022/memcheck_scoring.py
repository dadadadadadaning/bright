"""Explicit held-out Test inference. No loss, calibration, updates or checkpoint selection."""
import json
from pathlib import Path
import re

from grid_probe import sha256
from evaluate_checkpoint import write_json, ARMS


def read_names(path):
    names = [s.split()[0] for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names):
        raise ValueError('Invalid or duplicate split identities: ' + str(path))
    return names


def declare_test(runtime, manifest, *, repo):
    """Admit canonical Test queries only in this independent, explicitly marked process."""
    split = Path(repo) / 'bda_benchmark/dataset/splitname/standard_ML'
    files = {key: split / (key + '_set.txt') for key in ('train', 'val', 'test')}
    names = {k: read_names(p) for k, p in files.items()}
    for key in ('train', 'val'):
        if set(names[key]) != set(manifest[key + '_names']):
            raise ValueError('Canonical ' + key + ' split differs from training manifest')
    if len(names['test']) != 878 or set(names['test']) & (set(names['train']) | set(names['val'])):
        raise ValueError('Require all878 Test images, disjoint from Train/Val')
    allowed = {str(p.resolve()): sha256(p) for p in files.values()}
    dataset = runtime.bundle['args'].dataset
    print('TEST preflight: hashing canonical878 full-image inputs', flush=True)
    for name in names['test']:
        paths = runtime.deps.sample_paths(dataset, name)
        if len(paths) != 3:
            raise ValueError('Expected pre/post/target input paths')
        for path in paths.values():
            p = Path(path).resolve()
            allowed[str(p)] = sha256(p)
    # Original extractor checks _val_set for evaluation-query admission.
    # val_names remains Val450; the inherited evaluator is NOT relabelled as Test.
    runtime._val_set = set(runtime._val_set) | set(names['test'])
    runtime.protocol_info = {**runtime.protocol_info, 'test_accessed': True,
                             'evaluation_split': 'test', 'exploratory_test': True}
    runtime.input_ledger.update({Path(p): h for p, h in allowed.items()})
    return names['test'], allowed, {k: {'path': str(p), 'sha256': sha256(p)} for k, p in files.items()}


def verify_scope(runtime, inherited, allowed):
    for p, h in runtime.input_ledger.items():
        p = str(p)
        if p in inherited:
            if inherited[p] != h:
                raise ValueError('Inherited input changed: ' + p)
        elif allowed.get(p) != h:
            raise ValueError('Undeclared Test input: ' + p)


def metric(matrix, original_train):
    result = original_train._metrics(matrix)
    result['miou4'] = sum(x if x is not None else 0. for x in result['iou']) / 4
    result['overall_accuracy'] = float(matrix.trace() / matrix.sum()) if matrix.sum() else None
    return result


def rebuild(journal, names, paths, comparisons):
    import numpy as np
    cm = {p: np.zeros((4,4),dtype=np.int64) for p in paths}
    paired = {k: np.zeros((4,4,4),dtype=np.int64) for k in comparisons}
    events, event_paired, counts = {}, {}, {}
    seen = []
    with Path(journal).open() as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            if i >= len(names) or row['index'] != i or row['image'] != names[i]:
                raise ValueError('Test journal identity/order mismatch')
            if set(row['confusion']) != set(paths) or set(row['paired_counts']) != set(comparisons):
                raise ValueError('Test journal arm/pair mismatch')
            event = row['event']
            events.setdefault(event, {p: np.zeros((4,4),dtype=np.int64) for p in paths})
            event_paired.setdefault(event, {k: np.zeros((4,4,4),dtype=np.int64) for k in comparisons})
            counts[event] = counts.get(event,0) + 1
            rows = {}
            for p in paths:
                a = np.asarray(row['confusion'][p])
                if a.shape != (4,4) or a.dtype.kind not in 'iu' or (a < 0).any() or int(a.sum()) != row['valid_pixels']:
                    raise ValueError('Invalid Test confusion')
                rows[p] = a
                cm[p] += a
                events[event][p] += a
            if any(not np.array_equal(a.sum(1), rows[paths[0]].sum(1)) for a in rows.values()):
                raise ValueError('Test GT totals differ between arms')
            for k, (before,after) in comparisons.items():
                a = np.asarray(row['paired_counts'][k])
                if (a.shape != (4,4,4) or a.dtype.kind not in 'iu' or (a < 0).any()
                    or not np.array_equal(a.sum(2),rows[before]) or not np.array_equal(a.sum(1),rows[after])):
                    raise ValueError('Invalid paired Test transitions')
                paired[k] += a
                event_paired[event][k] += a
            seen.append(row['image'])
    if seen != list(names):
        raise ValueError('Incomplete Test journal')
    return cm, paired, events, event_paired, counts


def evaluate_test(runtime, heads, names, folder, *, seed, original_train, legacy, factory, evaluator, batches, step):
    import numpy as np
    import torch
    if tuple(heads) != tuple(ARMS) or set(names) & (set(runtime.train_names) | set(runtime.val_names)):
        raise ValueError('Test arms/split mismatch')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    paths = (evaluator.BASELINE,) + tuple(heads)
    comparisons = evaluator.comparison_pairs(heads)
    cm = {p: np.zeros((4,4),dtype=np.int64) for p in paths}
    state = legacy.state
    rng = state.capture_rng()
    modes = evaluator._capture_modes(runtime,heads)
    try:
        for module, _ in modes: module.training = False
        with torch.no_grad(), (folder/'images.jsonl').open('x') as stream:
            for index,batch in enumerate(batches()):
                example = runtime.prepare(batch)
                if index >= len(names) or example.image != names[index]:
                    raise ValueError('Test loader identity/order mismatch')
                original_train._assert_example(example)
                legacy.study.validate_example(example,runtime,original_train)
                support = factory.prepare(example,tuple(heads))
                if set(support) != set(heads): raise ValueError('Test supports mismatch')
                target = example.target.detach().cpu().numpy()
                row = dict(index=index,image=example.image,event=example.event,
                           valid_pixels=int((target != 255).sum()),confusion={},paired_counts={})
                predictions = {}
                for arm in paths:
                    prob = (example.common['base_probability_full'] if arm == evaluator.BASELINE else
                            factory.forward(heads[arm],example,arm,support[arm])['probabilities'])
                    evaluator._validate_probability(prob,example.target,arm)
                    pred = prob.argmax(1).detach().cpu().numpy()
                    matrix = original_train._confusion(target,pred)
                    cm[arm] += matrix
                    row['confusion'][arm] = matrix.tolist()
                    predictions[arm] = pred
                for k,(before,after) in comparisons.items():
                    row['paired_counts'][k] = original_train._transition(target,predictions[before],predictions[after]).tolist()
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                if (index+1) % 10 == 0 or index+1 == len(names):
                    print('MEMCHECK PREFIX step=%s %s/%s' % (step,index+1,len(names)),flush=True)
    finally:
        for module,mode in modes: module.training = mode
        # The external EvaluationObserver preserves an active CUDA exception
        # before attempting RNG restoration or any further CUDA operation.
        state.restore_rng(rng)
    rebuilt,pairs,events,event_pairs,counts = rebuild(folder/'images.jsonl',names,paths,comparisons)
    if any(not np.array_equal(cm[p],rebuilt[p]) for p in paths):
        raise ValueError('Test journal and in-memory confusion differ')
    metrics = {p: metric(cm[p],original_train) for p in paths}
    report = dict(format='BRIGHT_PARENT_IDS_PREFIX_DIAGNOSTIC_V1', step=step,split='test',
        phase='evaluation_complete_integrity_pending',complete=False,execution_complete=True,
        final_integrity_pending=True,full_validation_evidence=False,full_test_evidence=False,
        training=False,optimizer_updates=0,test_accessed=True,exploratory_test=True,
        test_used_for_selection=False,checkpoint_selection='diagnostic_fixed1500_canonical_test_prefix356',
        class_names=['background','intact','damaged','destroyed'],arms=list(heads),image_count=len(names),
        names_sha256=evaluator.names_hash(sorted(names)),order_sha256=evaluator.names_hash(names),
        journal_sha256=sha256(folder/'images.jsonl'),metrics=metrics,
        per_event={e:{p:metric(a,original_train) for p,a in tables.items()} for e,tables in events.items()},
        per_event_image_count=counts,
        transitions={k:evaluator._paired_report(a,*comparisons[k],original_train) for k,a in pairs.items()},
        per_event_transitions={e:{k:evaluator._paired_report(a,*comparisons[k],original_train)
                                 for k,a in tables.items()} for e,tables in event_pairs.items()},
        comparisons_miou4_pp={k:100*(metrics[b]['miou4']-metrics[a]['miou4']) for k,(a,b) in comparisons.items()},
        metric_definition='Pixel confusion, rows GT/columns prediction, ignore255; fixed four-class mean, absent IoU zero',
        test_labels_used_for='scoring only; no query labels enter support selection or prediction')
    write_json(folder/'result.json',report)
    return report
