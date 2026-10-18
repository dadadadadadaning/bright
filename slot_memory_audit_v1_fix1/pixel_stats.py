"""Paired, GT-stratified pixel errors. Counts are pixels, never buildings."""
import numpy as np
import math


PATHS = ('grid_probq', 'b', 'd')
PAIRS = {'b_vs_grid': ('grid_probq', 'b'), 'd_vs_grid': ('grid_probq', 'd'),
         'd_vs_b': ('b', 'd')}


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    union = gt + pred - tp
    div = lambda a, b: [float(x/y) if y else None for x, y in zip(a, b)]
    iou = div(tp, union)
    return dict(confusion=cm.tolist(), iou=iou,
                foreground_miou=sum(v or 0. for v in iou[1:])/3,
                precision=div(tp, pred), recall=div(tp, gt),
                valid_pixels=int(cm.sum()),
                foreground_average='fixed classes1/2/3; absent union contributes zero')


def paired_metrics(counts):
    a = np.asarray(counts, dtype=np.int64)
    rescue = [int(a[c, :, c].sum()-a[c,c,c]) for c in range(4)]
    harm = [int(a[c,c,:].sum()-a[c,c,c]) for c in range(4)]
    return dict(counts=a.tolist(), axes=['GT','reference','candidate'],
                rescue_by_gt_class=rescue, harm_by_gt_class=harm,
                intact_correct_to_damaged=int(a[1,1,2]),
                damaged_background_to_correct=int(a[2,0,2]),
                damaged_correct_to_background=int(a[2,2,0]),
                damaged_intact_to_correct=int(a[2,1,2]),
                damaged_correct_to_intact=int(a[2,2,1]))


class PixelAccumulator:
    def __init__(self):
        self.rows = {}

    def add(self, target, predictions, masks):
        target = np.asarray(target)
        if target.ndim != 2 or not set(PATHS) <= set(predictions):
            raise ValueError('Expected 2D labels and all three predictions')
        valid = target != 255
        if np.any((target[valid]<0)|(target[valid]>3)):
            raise ValueError('Invalid GT class')
        for pred in predictions.values():
            if pred.shape != target.shape or np.any((pred<0)|(pred>3)):
                raise ValueError('Invalid prediction')
        for name, mask in masks.items():
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != target.shape:
                raise ValueError('Invalid posthoc pixel mask')
            chosen = mask & valid
            if name not in self.rows:
                self.rows[name] = dict(cm={p:np.zeros((4,4),dtype=np.int64) for p in PATHS},
                                       paired={p:np.zeros((4,4,4),dtype=np.int64) for p in PAIRS})
            row = self.rows[name]
            y = target[chosen].astype(np.int64)
            for p in PATHS:
                code = y*4 + predictions[p][chosen]
                row['cm'][p] += np.bincount(code,minlength=16).reshape(4,4)
            for p,(before,after) in PAIRS.items():
                code = y*16+predictions[before][chosen]*4+predictions[after][chosen]
                row['paired'][p] += np.bincount(code,minlength=64).reshape(4,4,4)

    def report(self):
        return {name:dict(metrics={k:metrics(cm) for k,cm in row['cm'].items()},
                          paired={k:paired_metrics(v) for k,v in row['paired'].items()})
                for name,row in self.rows.items()}


def check_replay(actual, reference, tolerance_pp=.02):
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError('Replay tolerance must be finite and nonnegative')

    def validate(row):
        iou = row['iou']
        if len(iou) != 4:
            raise ValueError('Replay requires exactly four class IoUs')
        numbers = [x for x in iou if x is not None] + [row['foreground_miou']]
        try:
            valid = all(math.isfinite(x) and 0 <= x <= 1 for x in numbers)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('Replay metrics must be finite probabilities or absent-class None')
        cm = np.asarray(row['confusion'])
        if cm.shape != (4, 4) or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0):
            raise ValueError('Replay confusion must be nonnegative integer [4,4]')
        return cm.tolist()

    rows = {}
    for name in PATHS:
        a,b = actual[name],reference['metrics'][name]
        a_cm, b_cm = validate(a), validate(b)
        values = []
        for x,y in zip(a['iou'],b['iou']):
            if (x is None)!=(y is None):
                raise ValueError('Class presence differs on replay')
            if x is not None: values.append(abs(100*(x-y)))
        values.append(abs(100*(a['foreground_miou']-b['foreground_miou'])))
        rows[name] = dict(max_iou_delta_pp=max(values),passed=max(values)<=tolerance_pp,
                          exact_confusion=a_cm==b_cm)
    return dict(passed=all(r['passed'] for r in rows.values()), tolerance_pp=tolerance_pp,paths=rows)
