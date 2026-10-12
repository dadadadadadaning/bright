"""Explicitly authorized, inference-only Test evaluation. Never selects checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from .core import Pooler, SupportHead, fingerprint, batch, retrieve
from .run import encode, report, save_json


def pixel_metrics(hist):
    h = hist.double()
    tp = h.diag()
    iou = tp / (h.sum(0) + h.sum(1) - tp).clamp_min(1)
    f1 = 2 * tp / (h.sum(0) + h.sum(1)).clamp_min(1)
    return dict(valid_miou=float(iou[1:].mean()), miou4=float(iou.mean()),
                oa=float(tp.sum()/h.sum().clamp_min(1)), iou=iou.tolist(), f1=f1.tolist(),
                class_order=['Background', 'Intact', 'Damaged', 'Destroyed'], confusion=h.long().tolist())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('run_dir', 'dataset', 'test_list', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--confirm_test_access', action='store_true', required=True)
    a = p.parse_args()
    root, out = Path(a.run_dir), Path(a.output)
    manifest = json.loads((root/'manifest.json').read_text())
    cfg, meta = manifest['arguments'], manifest['train_metadata']
    if cfg['sources'] != 'fused_roi' or meta['base_model'] != 'DamageFormer_A1_pixel_path':
        raise ValueError('Only verified A1 fused_roi run supported')
    if meta['region_mode'] != 'slico' or meta['token_sampling'] != 'uniform_spatial':
        raise ValueError('Unsupported partition/sampling')
    base = Path(meta['base_checkpoint'])
    if fingerprint(base) != meta['base_sha256']:
        raise ValueError('A1 checkpoint hash mismatch')
    # Check list overlap before any Test image access, including images without regions.
    names = Path(a.test_list).read_text().split()
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate Test list')
    from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import read_list
    for split, expected in [('train', manifest['train_metadata']), ('val', manifest['val_metadata'])]:
        path = Path(a.dataset)/'splitname'/'standard_ML'/('train_set.txt' if split == 'train' else 'val_set.txt')
        if fingerprint(path) != expected['data_list_sha256']:
            raise ValueError('Train/Val list provenance mismatch')
        if set(names) & set(read_list(str(path))):
            raise ValueError('Test overlaps Train/Val')
    artifacts = {name: root/f'{name}_best.pth' for name in ['E0','E1','E2','E3']}
    artifacts['memory'] = root/'train_memory.pt'
    hashes = {k: fingerprint(v) for k,v in artifacts.items()}
    memory = torch.load(artifacts['memory'], map_location='cpu', weights_only=False)
    if memory['source_sha256'] != manifest['train_sha256']:
        raise ValueError('Memory Train provenance mismatch')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'frozen_protocol.json', dict(arguments=vars(a), training_manifest=manifest,
              artifact_sha256=hashes, test_list_sha256=fingerprint(a.test_list), test_image_count=len(names),
              test_access_authorized=True, training=False, checkpoint_selection='Val best, fixed before Test'))
    from .build_a1_records import main as build
    cache = out/'test_records.pt'
    build(['--checkpoint', str(base), '--dataset', a.dataset, '--data_list', a.test_list,
           '--output', str(cache), '--split', 'test', '--expected_images', str(len(names)),
           '--segments', str(meta['segments']), '--max_tokens', str(meta['max_region_tokens']),
           '--roi_alpha', str(meta['roi_alpha']), '--workers', str(a.workers), '--device', a.device], allow_test=True)
    data = torch.load(cache, map_location='cpu', weights_only=False, mmap=True)
    assert data['split'] == 'test' and data['test_accessed'] and not data['allowed_for_training']
    device = torch.device(a.device)
    size, dim = cfg['batch_size'], cfg['dim']
    summary = dict(test_accessed=True, scope='A1 full-image pixels; E0-E3 GT-foreground-eligible regions only; NO pixel writeback',
                   A1_pixel=pixel_metrics(data['base_hist']))
    def emit(name, pred):
        result = dict(**report(data, pred), test_accessed=True, training=False,
                      metric_scope='conditional region classification, NOT pixel mIoU')
        save_json(out/f'{name}.json', result)
        torch.save(pred, out/f'{name}_predictions.pt')
        summary[name] = result
        print(f'{name} Test accuracy={result["accuracy"]:.6f} macro_f1={result["macro_f1"]:.6f}', flush=True)
    for name in ['E0','E1']:
        encoder = Pooler(data['tokens']['fused_roi'].shape[-1], dim, name=='E1').to(device).eval()
        encoder.load_state_dict(torch.load(artifacts[name], map_location=device, weights_only=False)['model'])
        z, pred = encode(encoder, data, ['fused_roi'], device, size)
        emit(name, pred)
    bank, labels = memory['features'].to(device), memory['labels'].to(device)
    for name in ['E2','E3']:
        head = SupportHead(dim).to(device).eval()
        head.load_state_dict(torch.load(artifacts[name], map_location=device, weights_only=False)['model'])
        predictions = []
        for start in range(0, len(z), size):
            stop = min(start+size, len(z)); q = z[start:stop].to(device)
            if name == 'E2':
                logits = head(q)
            else:
                ix = retrieve(q, bank, data['events'][start:stop], data['images'][start:stop],
                              memory['events'], memory['images'], cfg['topk'])
                logits = head(q, bank[ix], labels[ix])
            predictions.append(logits.argmax(1).cpu())
        emit(name, torch.cat(predictions))
    if any(fingerprint(artifacts[k]) != v for k,v in hashes.items()):
        raise RuntimeError('Source artifacts changed during Test evaluation')
    save_json(out/'summary.json', summary)
    print('FROZEN TEST COMPLETE. No training. No Memory updates. No pixel writeback.', flush=True)


if __name__ == '__main__':
    main()
