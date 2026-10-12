"""Frozen A1 pixel path -> sampled region tokens. No ChangeMamba or bag inference."""
import argparse
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F


@torch.inference_mode()
def main(argv=None, *, allow_test=False):
    p = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'dataset', 'data_list', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--split', choices=['test'] if allow_test else ['train', 'validation'], required=True)
    p.add_argument('--max_images', type=int, default=0)
    p.add_argument('--expected_images', type=int, default=0)
    p.add_argument('--segments', type=int, default=400)
    p.add_argument('--max_tokens', type=int, default=12)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--roi_alpha', type=float, default=0.1)
    p.add_argument('--expected_miou', type=float, default=-1)
    p.add_argument('--tolerance_pp', type=float, default=0.02)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args(argv)
    if not allow_test and 'test' in Path(a.data_list).name.lower():
        p.error('Test access forbidden')
    out = Path(a.output)
    if out.exists():
        p.error('Output exists; refusing overwrite')
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent, sha256_file, parse_batch, event_name
    from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import read_list
    from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import make_region_ids, split_regions_by_building
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist
    names = read_list(a.data_list)
    if a.max_images:
        names = names[:a.max_images]
    if a.expected_images and len(names) != a.expected_images:
        raise ValueError('Unexpected image count')
    model = DamageFormer_S8_MICB_DSMIL(roi_alpha=a.roi_alpha, mil_residual_scale=0).to(a.device).eval()
    report = load_parent(model, a.checkpoint)
    if report['unexpected_keys']:
        raise ValueError('Unexpected checkpoint keys: use compatible A1, not a modified residual model')
    model.requires_grad_(False)
    loader = DataLoader(MultimodalDamageAssessmentDatset(a.dataset, names, crop_size=1024, type='test'),
                        batch_size=1, shuffle=False, num_workers=a.workers)
    rows, masks, labels, images, events, base_labels = [], [], [], [], [], []
    hist = torch.zeros(4, 4, dtype=torch.float64)
    event_hist = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    for n, batch in enumerate(loader, 1):
        pre, post, loc, target, sample_names = parse_batch(batch, torch.device(a.device))
        _, logits, aux = model(pre, post, return_aux=True)
        feat = aux['fused_roi'][0]
        size = feat.shape[-2:]
        ids = split_regions_by_building(make_region_ids(pre, size, mode='slico', n_segments=a.segments, compactness=10), aux['p_building'])[0]
        gt = F.interpolate(target[:, None].float(), size=size, mode='nearest')[0, 0].long()
        prob = F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=False).softmax(1)[0]
        building = aux['p_building'][0, 0]
        prediction = F.interpolate(logits.float(), size=target.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
        cm = fast_hist(prediction[0], target[0], 4, 255).cpu().double()
        name = str(sample_names[0]); event = event_name(name)
        hist += cm; event_hist[event] += cm
        flat = feat.flatten(1).T
        for rid in ids.unique():
            mask = ids == rid
            positions = mask.flatten().nonzero().flatten()
            if len(positions) < 4 or building[mask].mean() < 0.5:
                continue
            foreground = gt[mask]; foreground = foreground[(foreground >= 1) & (foreground <= 3)]
            if not len(foreground):
                continue
            # Uniform spatial sampling: identical for E0-E5; no GT or anomaly ranking.
            k = min(a.max_tokens, len(positions))
            selected = positions[torch.linspace(0, len(positions)-1, k, device=positions.device).long()]
            tokens = torch.zeros(a.max_tokens, feat.shape[0], device=feat.device)
            tokens[:k] = flat[selected]
            rows.append(tokens.half().cpu())
            masks.append(torch.arange(a.max_tokens) < k)
            labels.append(int(torch.bincount(foreground, minlength=4)[1:4].argmax())+1)
            base_labels.append(int(prob[:, mask].mean(1).argmax()))
            images.append(name); events.append(event)
        if n % 10 == 0 or n == len(names):
            print(f'records {n}/{len(names)} regions={len(labels)}', flush=True)
    iou = hist.diag() / (hist.sum(0)+hist.sum(1)-hist.diag()).clamp_min(1)
    miou = float(iou[1:].mean())
    print(f'Base Valid mIoU={miou:.8f}; mIoU4={float(iou.mean()):.8f}', flush=True)
    if a.expected_miou >= 0 and abs(miou-a.expected_miou)*100 > a.tolerance_pp:
        raise RuntimeError('Base reproduction failed; cache NOT saved')
    if not rows:
        raise RuntimeError('No eligible regions')
    meta = dict(base_model='DamageFormer_A1_pixel_path', base_checkpoint=str(Path(a.checkpoint).resolve()),
                base_sha256=report['sha256'], roi_alpha=a.roi_alpha, region_mode='slico', segments=a.segments,
                max_region_tokens=a.max_tokens, token_sampling='uniform_spatial', feature_source='fused_roi',
                image_count=len(names), data_list_sha256=sha256_file(a.data_list), base_load=report,
                base_valid_miou=miou, test_accessed=allow_test,
                scope='Predicted-ROI regions with GT foreground; NOT full-image segmentation',
                gt_role='Foreground eligibility and majority labels only; no GT partition or feature input')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(protocol='SRAC_A1_RECORDS_V1', split=a.split, allowed_for_training=a.split=='train',
                   tokens={'fused_roi': torch.stack(rows)}, token_mask=torch.stack(masks), labels=torch.tensor(labels),
                   images=images, events=events, p3_region_labels=torch.tensor(base_labels), base_hist=hist,
                   event_base_hist=dict(event_hist), metadata=meta, test_accessed=allow_test)
    temporary = out.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(out)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
