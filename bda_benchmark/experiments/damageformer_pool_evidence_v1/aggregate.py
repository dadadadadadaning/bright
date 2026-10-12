"""Associations, not causal attribution. GT is used after inference only."""
import torch
from torch.nn import functional as F
from .core import bin_exposure


def summarize_joint(joint):
    j = joint.double()
    correct = sum(float(j[c, c, :].sum()) for c in range(4))
    count = float(j.sum())
    harm = torch.zeros(4, 4, dtype=torch.float64)
    rescue = torch.zeros_like(harm)
    for c in range(4):
        for other in range(4):
            if c != other:
                harm[c, other] = j[c, c, other]
                rescue[c, other] = j[c, other, c]
    h, r = float(harm.sum()), float(rescue.sum())
    return dict(weighted_pixels=count, base_correct=correct, base_wrong=count-correct,
                harmed=h, rescued=r, net=r-h,
                harm_rate=h/correct if correct else None,
                rescue_rate=r/(count-correct) if count > correct else None,
                gt_support=j.sum((1, 2)).tolist(),
                harm_gt_to_wrong=harm.tolist(), rescue_gt_from_wrong=rescue.tolist(),
                damaged_to_intact_harm=float(harm[2, 1]),
                intact_to_damaged_rescue=float(rescue[2, 1]))


class PoolAccumulator:
    def __init__(self, threshold):
        self.threshold = threshold
        self.joints, self.slots, self.path_totals = {}, {}, {}

    @torch.no_grad()
    def update(self, evidence, route, gt, baseline, candidate, building):
        valid = (gt >= 0) & (gt < 4)
        bmap = F.interpolate(building.float(), gt.shape[-2:], mode='nearest')[:, 0]
        b = bmap >= self.threshold
        fg = valid & (gt > 0)
        index = (gt.clamp(0, 3)*16 + baseline*4 + candidate).flatten()
        # Four DISJOINT ROI groups; obtain foreground/all by addition. Chunked
        # histograms reduce CUDA atomic contention from dominant background.
        roi = torch.where(fg, torch.where(b, 0, 1), torch.where(b, 2, 3)).flatten()
        block = torch.arange(index.numel(), device=gt.device)//2048
        histogram_index = block*256 + roi*64 + index
        blocks = (index.numel()+2047)//2048
        for path, stats in evidence.items():
            totals = self.path_totals.setdefault(path, dict(images=0, allocated_slots=0,
                occupied_slots=0, empty_slots=0, positive_guard_slots=0, no_valid_gt_slots=0))
            mass = stats['mass']
            totals['images'] += len(gt)
            totals['allocated_slots'] += mass.numel()
            totals['occupied_slots'] += int((mass > 0).sum())
            totals['empty_slots'] += int((mass == 0).sum())
            totals['positive_guard_slots'] += int(((mass > 0) & (mass < 1e-6)).sum())
            totals['no_valid_gt_slots'] += int(((mass > 0) & (stats['valid_mass'] == 0)).sum())
            slot_masks = dict(stats['slot_masks'])
            # An offline stratum only. NEVER used to choose model outputs.
            slot_masks['clean_supported'] = ((stats['valid_mass'] > 0) &
                (stats['purity'] >= .95) & (stats['mass'] >= 1) & (stats['ess'] >= 2))
            slot_masks['clean_supported_foreground'] = (slot_masks['clean_supported'] &
                (stats['foreground_fraction'] >= .95))
            for key, mask in slot_masks.items():
                label = path + '/' + key
                sums = self.slots.setdefault(label, dict(slots=0, mass=0., valid_mass=0.,
                    foreground_mass=0., purity_sum=0., purity_valid_mass_sum=0.,
                    foreground_purity_fg_mass_sum=0., ess_sum=0.))
                fg_mass = stats['class_mass'][..., 1:].sum(-1)
                sums['slots'] += int(mask.sum())
                for name, values in (
                    ('mass', stats['mass']), ('valid_mass', stats['valid_mass']),
                    ('foreground_mass', fg_mass), ('purity_sum', stats['purity']),
                    ('purity_valid_mass_sum', stats['purity']*stats['valid_mass']),
                    ('foreground_purity_fg_mass_sum', stats['foreground_purity']*fg_mass),
                    ('ess_sum', stats['ess'])):
                    sums[name] += float(values[mask].double().sum())
            keys = list(slot_masks)
            for start in range(0, len(keys), 4):
                subset = keys[start:start+4]
                maps = bin_exposure(route, {k: slot_masks[k] for k in subset}, gt.shape[-2:])
                for j, key in enumerate(subset):
                    exposure = maps[:, j].flatten()
                    histogram = torch.zeros(blocks*256, device=gt.device, dtype=torch.float64)
                    histogram.scatter_add_(0, histogram_index, (exposure*valid.flatten()).double())
                    roi_joints = histogram.reshape(blocks, 4, 4, 4, 4).sum(0).cpu()
                    joints = dict(roi_true_positive=roi_joints[0], roi_false_negative=roi_joints[1],
                                  roi_false_positive=roi_joints[2], roi_true_negative=roi_joints[3],
                                  foreground=roi_joints[:2].sum(0), all=roi_joints.sum(0))
                    for context, joint in joints.items():
                        label = path + '/' + key + '/' + context
                        if label not in self.joints:
                            self.joints[label] = torch.zeros(4, 4, 4, dtype=torch.float64)
                        self.joints[label] += joint

    def finalize(self):
        return dict(slots=self.slots, path_totals=self.path_totals,
                    associations={k: summarize_joint(v) for k, v in self.joints.items()},
                    protocol=dict(
                        statistical_unit='weighted output pixels, NOT independent samples or building instances',
                        exposure='regional stratum indicators -> SAME soft Q read -> bilinear output grid; no channel renormalization',
                        overlapping='Different criteria and ROI/foreground contexts overlap; do not add across criteria',
                        clean_supported='offline purity>=.95, native mass>=1 and ESS>=2; not an inference gate',
                        clean_supported_foreground='clean_supported AND GT foreground fraction>=.95; prevents pure background from implying pure building evidence',
                        interpretation='Associations only; neither mixture purity nor exposure proves causal harm',
                        foreground_purity='dominant damage grade within foreground GT support; NOT whole-slot purity'))
