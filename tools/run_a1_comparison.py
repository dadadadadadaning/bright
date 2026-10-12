"""Confirmed A1 checkpoint comparison: prepare -> review -> run -> ZIP figures."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import zipfile

import compare_bright_checkpoints as c

A1_HASH='cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run'])
    p.add_argument('--work',required=True,help='New run directory for prepare; same directory for run')
    p.add_argument('--data',default='/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset')
    p.add_argument('--checkpoints',default='/root/shared-nvme/BRIGHT-checkpoints')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--changemamba-cfg',help='Exact training YAML; otherwise explicitly omitted')
    p.add_argument('--accept-auto-preview',action='store_true',help='Run automatically screened candidates as DRAFT, not manually verified shadow-free figures')
    args=p.parse_args()
    work=Path(args.work)
    if args.action=='prepare':
        work.mkdir(parents=True,exist_ok=False)
        c.init(SimpleNamespace(checkpoints=args.checkpoints,output=str(work/'models.json')))
        entries=c.load_json(work/'models.json')
        ours=next(e for e in entries if e['name']=='Ours')
        ours.update(name='Ours_A1',
            factory='bda_benchmark.model.DamageFormer_S8_MICB_DSMIL:DamageFormer_S8_MICB_DSMIL',
            kwargs=dict(roi_alpha=.1,detach_building_prior=True,token_dim=128,mil_topk=8,
                        max_tokens_per_bag=256,mil_residual_scale=0.0),
            expected_sha256=A1_HASH,expected_variant='a1')
        if args.changemamba_cfg:
            cm=next(e for e in entries if e['name']=='ChangeMamba')
            cm.update(enabled=True,cfg=str(Path(args.changemamba_cfg).resolve()))
        else:
            print('ChangeMamba NOT included: exact training YAML is still required.',flush=True)
        c.write_json(work/'models.json',entries)
        for e in entries:
            if e['enabled'] and not Path(e['checkpoint']).is_file():
                raise FileNotFoundError(e['checkpoint'])
        c.write_json(work/'settings.json',dict(data=str(Path(args.data).resolve()),
            note='Existing checkpoints; not retrained under a matched training budget. A1 is not the proposed region-memory architecture.'))
        c.select(SimpleNamespace(data=args.data,output=str(work/'candidates'),crop=512,
            per_event=1,dark_threshold=40.,max_dark=.15,max_invalid=.01,min_building_pixels=1000))
        print('Review candidates, set approved=true, then run this script with action run.',flush=True)
        return
    selection=work/'candidates/selection.json'
    if args.accept_auto_preview:
        obj=c.load_json(selection)
        for s in obj['samples']:
            s['approved']=True
            s['review_status']='automatically_screened_NOT_manually_reviewed'
        selection=work/'selection_auto_preview.json'
        c.write_json(selection,obj)
        print('DRAFT preview: automatic brightness filtering is not verified shadow detection.',flush=True)
    settings=c.load_json(work/'settings.json')
    output=work/'predictions'
    c.run(SimpleNamespace(data=settings['data'],manifest=str(work/'models.json'),
        selection=str(selection),output=str(output),device=args.device,models=None))
    archive=work/'BRIGHT_A1_comparison_images.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob('*')):
            # Full probabilities remain on the server to keep the download small.
            if f.is_file() and f.suffix.lower() in ('.png','.pdf','.json'):
                z.write(f,f.relative_to(work))
        for f in (work/'models.json',work/'settings.json',selection):
            z.write(f,f.relative_to(work))
    print('DOWNLOAD THIS FILE:',archive,flush=True)


if __name__=='__main__':main()
