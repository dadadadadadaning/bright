"""Export measured V2/V3 regional evidence. Never creates synthetic attention.

Capture in existing inference code:
    loc, damage, aux = model(pre, post, return_aux=True)
    save_aux_npz(aux, 'query_A.npz', sample=0,
                 metadata={'sample_id': name, 'checkpoint': ckpt})
Render:
    python tools/export_region_figure.py --rgb raw_query.png --aux query_A.npz \
        --out figure_A --building-id 1 --region-id 12

RGB must be the same crop/orientation as the model input before normalization.
Dependencies: numpy, Pillow, matplotlib. Torch required only for model inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps, pyplot as plt


def save_aux_npz(aux, path, sample=0, metadata=None):
    """Capture one sample from the verified V2/V3 aux contract, no training changes."""
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(dest)
    arrays = {}
    for key in ('building_ids', 'candidate_ids', 'association_probabilities', 'candidate_valid'):
        value = aux[key][sample]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    arrays['metadata_json'] = np.array(json.dumps(metadata or {}, ensure_ascii=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)


def decode(building, ids, prob, valid):
    building = np.asarray(building)
    if building.ndim != 2:
        raise ValueError('building_ids must be H x W for one sample')
    h, w = building.shape
    if ids.shape != prob.shape or valid.shape != prob.shape:
        raise ValueError('candidate IDs, validity and probabilities must have identical shapes')
    if prob.ndim not in (2, 3) or np.prod(prob.shape[1:]) != h*w:
        raise ValueError('Expected candidate-first arrays K x (H*W) or K x H x W')
    ids = ids.reshape(-1, h, w).astype(np.int64)
    prob = prob.reshape(ids.shape).astype(float)
    valid = valid.reshape(ids.shape).astype(bool) & (ids >= 0)
    if not np.isfinite(prob).all() or (prob < -1e-6).any():
        raise ValueError('Invalid association probabilities')
    prob = np.where(valid, prob, 0)
    fg = building > 0
    if not np.allclose(prob.sum(0)[fg], 1, atol=2e-3):
        raise ValueError('Building-pixel association weights must sum to one')
    if not valid.any(0)[fg].all():
        raise ValueError('Building pixels without valid candidates')
    winners = np.where(valid, prob, -np.inf).argmax(0)
    hard = np.take_along_axis(ids, winners[None], axis=0)[0]
    hard[~fg] = -1
    return hard, ids, prob


def edges(ids):
    e = np.zeros(ids.shape, bool)
    e[1:] |= ids[1:] != ids[:-1]
    e[:-1] |= ids[1:] != ids[:-1]
    e[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    e[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    return e


def nearest(a, size):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, Image.Resampling.NEAREST))


def grid_control(mask, target):
    yy, xx = np.indices(mask.shape)
    best = None
    for cell in range(1, max(mask.shape)+1):
        ids = (yy//cell)*(int(np.ceil(mask.shape[1]/cell))+1)+xx//cell
        n = len(np.unique(ids[mask]))
        if best is None or abs(n-target) < best[0]:
            best = (abs(n-target), cell, n, np.where(mask, ids, -1))
        if n == target:
            break
    return best[3], best[1], best[2]


def render(rgb_path, aux_path, out, building_id=None, region_id=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    with np.load(aux_path, allow_pickle=False) as z:
        building = z['building_ids']
        hard, ids, prob = decode(building, z['candidate_ids'], z['association_probabilities'], z['candidate_valid'])
        meta = json.loads(str(z['metadata_json'])) if 'metadata_json' in z else {}
    labels, counts = np.unique(building[building>0], return_counts=True)
    if not len(labels):
        raise ValueError('No predicted buildings; cannot render building evidence')
    bid = int(labels[counts.argmax()]) if building_id is None else building_id
    mask = building == bid
    if not mask.any():
        raise ValueError(f'Building {bid} absent; available IDs: {labels.tolist()}')
    regions, mass = np.unique(hard[mask], return_counts=True)
    rid = int(regions[mass.argmax()]) if region_id is None else region_id
    if rid not in regions:
        raise ValueError(f'Region {rid} absent in building {bid}; choose {regions.tolist()}')
    # Sparse local association slots must be mapped back to the global token ID.
    q = np.sum(np.where(ids == rid, prob, 0), axis=0)
    q[~mask] = 0
    grid, cell, grid_n = grid_control(mask, len(regions))
    size = (rgb.shape[1], rgb.shape[0])
    m = nearest(mask, size)>0
    hs = nearest(np.where(mask, hard, -1), size).astype(int)
    gs = nearest(grid, size).astype(int)
    qs = nearest(q, size)
    base = rgb.copy()
    base[~m] = (0.23*base[~m]+0.77*255).astype(np.uint8)
    out.mkdir(parents=True)
    def save(a, name):
        Image.fromarray(a.astype(np.uint8)).save(out/name)
    outline = rgb.copy(); outline[edges(m.astype(int))] = (255,220,80)
    save(outline, '01_building_outline.png')
    for values, name in [(gs,'02_grid_control.png'), (hs,'03_content_regions.png')]:
        overlay = base.copy(); overlay[edges(values)&m] = (30,190,220)
        save(overlay,name)
    selected = (hs == rid)&m
    selected_img = (0.20*rgb+0.80*255).astype(np.uint8)
    selected_img[selected] = rgb[selected]
    selected_img[edges(selected.astype(int))] = (220,70,150)
    save(selected_img, '04_selected_region.png')
    colour = colormaps['viridis'](np.clip(qs,0,1))[...,:3]*255
    overlay = base.copy(); overlay[m] = (0.45*rgb[m]+0.55*colour[m]).astype(np.uint8)
    save(overlay, '05_assignment_overlay.png')
    fig, ax = plt.subplots(figsize=(3.5,0.65))
    fig.subplots_adjust(bottom=.5,top=.78)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='viridis'),cax=ax,orientation='horizontal',label='Assignment weight')
    fig.savefig(out/'assignment_colorbar.svg',bbox_inches='tight');plt.close(fig)
    np.save(out/'assignment_values.npy',q)
    report = {'provenance':meta, 'rgb':str(rgb_path), 'aux':str(aux_path),
              'building_id':bid,'region_id':rid,'available_regions':regions.tolist(),
              'model_shape':list(building.shape),'display_shape':list(rgb.shape[:2]),
              'resampling':'nearest; no inferred registration or crop correction',
              'content_region_count':len(regions),'grid_region_count':grid_n,'grid_cell':cell,
              'grid_count_exact_match':grid_n==len(regions),
              'warning':'Q is assignment, not damage or Memory attention; no retrieval outputs created.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


def self_test():
    building=np.array([[1,1],[0,2]])
    ids=np.array([[[9,9],[0,15]],[[12,12],[0,16]]])
    prob=np.array([[[.8,.2],[1,.4]],[[.2,.8],[0,.6]]])
    hard,ci,q=decode(building,ids,prob,np.ones_like(ids,bool))
    assert hard.tolist()==[[9,12],[-1,16]], hard
    assert np.allclose(np.where(ci==12,q,0).sum(0)[0],[.2,.8])
    try:
        decode(building,ids,prob*2,np.ones_like(ids,bool))
    except ValueError:
        pass
    else:
        raise AssertionError('Bad normalization was accepted')
    print('PASS: sparse-ID mapping, background masking, normalization guard')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rgb');ap.add_argument('--aux');ap.add_argument('--out')
    ap.add_argument('--building-id',type=int);ap.add_argument('--region-id',type=int)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:
        self_test()
    elif not all([a.rgb,a.aux,a.out]):
        ap.error('--rgb --aux --out are required')
    else:
        render(a.rgb,a.aux,a.out,a.building_id,a.region_id)
