"""Final-Val-only paired evidence. GT never enters the model or routing.

All matrices use rows=GT, columns=prediction. Conditional scores are not
full-image mIoU. Components are foreground proxies, NOT building instances.
"""
import torch
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics, size_masks
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.common import event

CLASS_NAMES = ('Background', 'Intact', 'Damaged', 'Destroyed')


def paired_names(names):
    pairs = [('g0', 'g1'), ('s0', 's2'), ('g1', 's2'), ('g0', 's0'),
             ('h1', 's2'), ('g1', 'h1'), ('s0', 'h0'), ('h0', 'h1')]
    pairs += [('old_pixel', k) for k in names if k not in ('old_pixel', 'old_region')]
    return [(a, b) for a, b in pairs if a in names and b in names]


def new_bucket(names, pairs):
    return {'cm': {k: torch.zeros(4, 4, dtype=torch.long) for k in names},
            'joint': {a+'__'+b: torch.zeros(4, 4, 4, dtype=torch.long)
                      for a, b in pairs}}


def accumulate(bucket, gt, predictions, mask, pairs):
    truth = gt[mask]
    selected = {k: p[mask] for k, p in predictions.items()}
    for name, pred in selected.items():
        bucket['cm'][name] += torch.bincount(truth*4+pred, minlength=16).reshape(4, 4)
    for a, b in pairs:
        bucket['joint'][a+'__'+b] += torch.bincount(
            truth*16 + selected[a]*4 + selected[b], minlength=64).reshape(4, 4, 4)


def finalize_bucket(bucket):
    comparisons = {}
    for pair, joint in bucket['joint'].items():
        harm = torch.zeros(4, 4, dtype=torch.long)
        rescue = torch.zeros_like(harm)
        correct = sum(int(joint[c, c, :].sum()) for c in range(4))
        count = int(joint.sum())
        for c in range(4):
            for other in range(4):
                if c != other:
                    harm[c, other] = joint[c, c, other]
                    rescue[c, other] = joint[c, other, c]
        harmed, rescued = int(harm.sum()), int(rescue.sum())
        comparisons[pair] = dict(count=count, base_correct=correct, base_wrong=count-correct,
            base_correct_by_class=[int(joint[c,c,:].sum()) for c in range(4)],
            base_wrong_by_class=[int(joint[c].sum()-joint[c,c,:].sum()) for c in range(4)],
            harmed=harmed, rescued=rescued, net=rescued-harmed,
            harm_rate=harmed/correct if correct else None,
            recovery_rate=rescued/(count-correct) if count>correct else None,
            damaged_to_intact_harm=int(harm[2, 1]), intact_to_damaged_rescue=int(rescue[2, 1]),
            harm_gt_to_wrong=harm.tolist(), rescue_gt_from_wrong=rescue.tolist())
    rows = {}
    for k, cm in bucket['cm'].items():
        rows[k] = metrics(cm)
        rows[k].update(tp=cm.diag().tolist(), fn=(cm.sum(1)-cm.diag()).tolist(), fp=(cm.sum(0)-cm.diag()).tolist())
    return dict(metrics=rows, comparisons=comparisons)


class DiagnosticAccumulator:
    def __init__(self, threshold=.5, pairs=None):
        if not 0 < threshold < 1:
            raise ValueError('Invalid building threshold')
        self.threshold = threshold
        self.pairs = None if pairs is None else tuple(tuple(p) for p in pairs)
        self.prediction_names = None
        self.groups, self.events, self.images = {}, {}, []
        self.component_counts = {}

    @torch.no_grad()
    def update(self, gt, predictions, building, names):
        if self.prediction_names is None:
            self.prediction_names = tuple(predictions)
            if self.pairs is None:
                self.pairs = tuple(paired_names(predictions)) + tuple((a, b) for a, b in [('hard', 'exchange')] if a in predictions and b in predictions)
            else:
                self.pairs = self.pairs + tuple(('old_pixel', k) for k in predictions if k not in ('old_pixel', 'old_region') and ('old_pixel', k) not in self.pairs and 'old_pixel' in predictions)
            if len(set(self.pairs)) != len(self.pairs) or any(len(p) != 2 or p[0] == p[1] or any(k not in predictions for k in p) for p in self.pairs):
                raise ValueError('Invalid comparison pairs')
        elif tuple(predictions) != self.prediction_names:
            raise ValueError('Prediction names/order changed')
        gt = gt.detach().long().cpu()
        predictions = {k: p.detach().long().cpu() for k, p in predictions.items()}
        if gt.ndim != 3 or len(names) != len(gt) or building.shape[:2] != (len(gt), 1):
            raise ValueError('Diagnostics require BHW targets, B1hw building and sample names')
        if any(p.shape != gt.shape or (p<0).any() or (p>3).any() for p in predictions.values()):
            raise ValueError('Prediction shape/range mismatch')
        # Nearest mapping preserves the ACTUAL native-grid group decision.
        bmap = F.interpolate(building.detach().float().cpu(), gt.shape[-2:], mode='nearest')[:, 0]
        if not torch.isfinite(bmap).all() or (bmap<0).any() or (bmap>1).any():
            raise ValueError('Invalid building probability')
        for j, name in enumerate(names):
            y = gt[j]; preds = {k: p[j] for k, p in predictions.items()}
            valid = (y>=0)&(y<4); foreground = valid&(y>0)
            predicted_building = bmap[j] >= self.threshold
            edge = boundary_mask(y)
            masks = dict(all=valid, boundary=edge, interior=valid&~edge,
                         foreground_interior=foreground&~edge,
                         roi_true_positive=foreground&predicted_building,
                         roi_false_negative=foreground&~predicted_building,
                         roi_false_positive=(y==0)&predicted_building,
                         roi_true_negative=(y==0)&~predicted_building,
                         building_prob_mid=valid&(bmap[j]>=.4)&(bmap[j]<=.6))
            for c, label in enumerate(CLASS_NAMES):
                masks[label+'__boundary'] = valid&(y==c)&edge
                masks[label+'__interior'] = valid&(y==c)&~edge
            sizes, numbers = size_masks(y)
            masks.update(sizes)
            for k, value in numbers.items():
                self.component_counts[k] = self.component_counts.get(k, 0)+value
            for lo in range(10):
                lower, upper = lo / 10, (lo + 1) / 10
                band = valid & (bmap[j] >= lower) & ((bmap[j] <= upper) if lo == 9 else (bmap[j] < upper))
                for roi in ('true_positive', 'false_negative', 'false_positive', 'true_negative'):
                    masks[f'b_bin_{lo}__roi_{roi}'] = band & masks['roi_' + roi]
            for key, mask in masks.items():
                bucket = self.groups.setdefault(key, new_bucket(preds, self.pairs))
                accumulate(bucket, y, preds, mask, self.pairs)
            ev = event(name)
            accumulate(self.events.setdefault(ev, new_bucket(preds, self.pairs)), y, preds, valid, self.pairs)
            row = new_bucket(preds, self.pairs)
            accumulate(row, y, preds, valid, self.pairs)
            self.images.append(dict(image=str(name), event=ev, **finalize_bucket(row)))

    def finalize(self):
        return dict(image_count=len(self.images), test_accessed=False,
            groups={k: finalize_bucket(v) for k, v in self.groups.items()},
            per_event={k: finalize_bucket(v) for k, v in self.events.items()},
            per_image=self.images, component_counts=self.component_counts,
            protocol=dict(reference='old_pixel is frozen V2 snapshot output, NOT historical A1',
                boundary='GT four-class transitions; radius2 original pixels; ignore excluded',
                roi='predicted native B >= threshold; nearest-expanded to output grid; GT evaluation only',
                threshold=self.threshold, mid_probability_band=[.4,.6],
                b_bins=[i/10 for i in range(11)],
                b_bin_endpoints='[lower,upper), except last [0.9,1]; native B nearest mapped; bins count output pixels',
                pairs=list(self.pairs or []),
                sizes='8-connected GT foreground proxies; small<256, medium256..1023, large>=1024 original pixels',
                warning='Conditional scores are NOT full-image mIoU; components are NOT building instances',
                absent_classes='fixed three-class mIoU zero for zero union; per-class undefined ratios null',
                statistics='per-image/event confusion saved for paired resampling; no pixel-iid CI claimed'))
