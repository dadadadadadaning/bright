"""Run the user-authorized 18-candidate Test preview with six frozen models.

Authorization to infer is NOT a certification that each crop is cloud-free.
Uses existing server manifest; outputs to a fresh directory, no training.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED
import compare_bright_checkpoints as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(c.ROOT/'bda_benchmark/dataset'))
    p.add_argument('--manifest',default='/root/shared-nvme/BRIGHT-A1-compare-v1/models.json')
    p.add_argument('--selection',default=str(Path(__file__).parent/'BRIGHT_test_candidates_v2/selection.json'))
    p.add_argument('--work',default='/root/shared-nvme/BRIGHT-test-sixmodels-v2')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import torch
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable here. Run this script in your bright4090 server environment.')
    selection=c.load_json(a.selection)
    if selection.get('split')!='test':p.error('Expected a Test selection, not Val.')
    rows=selection['samples']
    if not rows:p.error('No candidates')
    valid=set((c.ROOT/'bda_benchmark/dataset/splitname/standard_ML/test_set.txt').read_text().split())
    models=c.load_json(a.manifest)
    required=['UNet','DeepLabV3Plus','SiamAttnUNet','SiamCRNN','DamageFormer','Ours_A1']
    entries=[]
    for name in required:
        matches=[e for e in models if e['name']==name and e.get('enabled',True)]
        if len(matches)!=1:p.error(f'Missing/duplicate enabled model: {name}')
        if not Path(matches[0]['checkpoint']).is_file():p.error(f'Missing checkpoint: {matches[0]["checkpoint"]}')
        entries.append(matches[0])
    ours=entries[-1]
    ours['expected_sha256']='cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add'
    ours['expected_variant']='a1'
    for r in rows:
        sid=r['sample_id']
        if sid not in valid:p.error(f'Not a standard_ML Test sample: {sid}')
        for folder,suffix in [('pre-event','pre_disaster'),('post-event','post_disaster'),('target','building_damage')]:
            f=Path(a.data)/folder/f'{sid}_{suffix}.tif'
            if not f.is_file():p.error(f'Missing input/GT: {f}')
        if r.get('approved') is not True:
            r['approved']=True
            r['review_status']='inference_authorized_cloud_review_PENDING'
    work=Path(a.work);work.mkdir(parents=True,exist_ok=False)
    selection['note']='User authorized inference on all candidates. This is NOT cloud-free certification. Selected Test examples only, not full Test metrics.'
    c.write_json(work/'selection_inference.json',selection)
    c.write_json(work/'models_frozen.json',entries)
    print(f'Running {len(rows)} Test examples, six models. No training. Cloud review remains pending.',flush=True)
    c.run(SimpleNamespace(data=a.data,manifest=str(work/'models_frozen.json'),
        selection=str(work/'selection_inference.json'),output=str(work/'predictions'),
        device=a.device,models=None,split='test',allow_test=True))
    with ZipFile(work/'BRIGHT_test_sixmodels_images.zip','x',ZIP_DEFLATED) as z:
        for f in sorted(work.rglob('*')):
            if f.is_file() and f.suffix.lower() in ('.png','.pdf','.json'):
                z.write(f,f.relative_to(work))
    print('RESULTS:',work/'BRIGHT_test_sixmodels_images.zip',flush=True)


if __name__=='__main__':main()
