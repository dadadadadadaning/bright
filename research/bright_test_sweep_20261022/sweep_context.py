"""Five explicitly requested checkpoints from one six-arm trajectory."""
from pathlib import Path
from grid_probe import read_json,sha256
from evaluate_checkpoint import ARMS
from resume_train import verify_files
import checkpoint as checkpoint_io
import night_entry
from night_config import SOURCE

STEPS=(500,1000,1500,2000,2500)
SELECTION='requested_fixed_steps_500_1000_1500_2000_2500_all_six_arms'
BASE=SOURCE.parent.parent/'SRAC_SLOT_MEMORY_ABLATION_TEST_SWEEP_V1'
RESUMES={1500:SOURCE.parent/'resume_20261020_161047_4184092',
         2000:SOURCE.parent/'resume_20261021_112222_495713',2500:SOURCE}


def validate_sidecar(path,step,kind,c):
    meta=read_json(path.with_suffix('.pt.json'))
    if (meta.get('step')!=step or meta.get('roundtrip_verified') is not True
        or meta.get('sha256')!=sha256(path) or set(meta.get('head_hashes',{}))!=set(ARMS)):
        raise ValueError('Checkpoint hash/step/arms mismatch: '+str(path))
    final_prefix=checkpoint_io.journal_prefix(c['parent_journal'],step=step,
        plan=c['plan'],original_manifest=c['manifest'])
    own_journal=path.parent.parent/'train.jsonl'
    prefix=checkpoint_io.journal_prefix(own_journal,step=step,plan=c['plan'],original_manifest=c['manifest'])
    if prefix!=final_prefix:raise ValueError('Checkpoint does not share the final trajectory journal prefix: '+str(path))
    if kind=='resume':
        m=read_json(path.parent.parent/'manifest.json')
        if (meta.get('format')!=checkpoint_io.FORMAT or meta.get('lineage')!=c['lineage']
            or meta.get('updates_per_arm')!={a:step for a in ARMS}
            or m.get('lineage')!=c['lineage'] or m.get('original_manifest')!=c['manifest']
            or any(meta.get(k)!=v for k,v in prefix.items())):
            raise ValueError('Checkpoint lineage or update counts differ: '+str(path))
    return dict(path=str(path),kind=kind,step=step,sha256=meta['sha256'],journal=str(own_journal),
                head_hashes=meta['head_hashes'],journal_prefix=prefix)


def preflight():
    c=night_entry.preflight()
    records={}
    for step in STEPS:
        kind='original' if step<=1000 else 'resume'
        root=Path(c['train']) if kind=='original' else RESUMES[step]
        path=root/'checkpoints'/('checkpoint_%06d.pt'%step)
        if not path.is_file() or not path.with_suffix('.pt.json').is_file():
            raise ValueError('Missing requested checkpoint/sidecar; no substitution or training: '+str(path))
        records[str(step)]=validate_sidecar(path,step,kind,c)
        print('CHECKPOINT step=%s kind=%s sha256=%s path=%s'%(step,kind,records[str(step)]['sha256'],path),flush=True)
        paths=[path,path.with_suffix('.pt.json'),root/'train.jsonl',root/'manifest.json']
        c['evidence_files'].update({str(p):sha256(p) for p in paths})
    if records['1000']['sha256']!=c['lineage']['original_parent_checkpoint_sha256']:
        raise ValueError('Original1000 checkpoint identity differs')
    if records['2500']['sha256']!=c['parent']['sha256']:raise ValueError('Final2500 identity differs')
    verify_files(c['evidence_files']);c['checkpoints']=records
    return c


def for_step(c,step):
    if step not in STEPS:raise ValueError('Unsupported requested step')
    selected=c['checkpoints'][str(step)]
    return dict(c,parent=selected,parent_journal=Path(selected['journal']),selected_step=step)


def load_selected(torch,state,c):
    record=c['parent'];path=Path(record['path']);step=record['step']
    if sha256(path)!=record['sha256']:raise ValueError('Selected checkpoint changed')
    if record['kind']=='resume':
        return checkpoint_io.load(torch,path,state,plan=c['plan'],original_manifest=c['manifest'],
                                  lineage=c['lineage'],journal_path=c['parent_journal'])
    from probe import validate_checkpoint
    saved=torch.load(path,map_location='cpu',weights_only=False)
    meta=read_json(path.with_suffix('.pt.json'))
    validate_checkpoint(saved,meta,c['plan'],c['manifest'],state.state_fingerprint)
    checkpoint_io._states(torch,saved['heads'],saved['optimizers'],step)
    if saved.get('test_accessed') is not False or not saved.get('rng') or not saved.get('input_file_sha256'):
        raise ValueError('Original checkpoint lacks training-only RNG/input evidence')
    if saved.get('step')!=step or sha256(path)!=record['sha256']:raise ValueError('Original checkpoint changed')
    return saved,dict(meta,path=str(path),kind='original')
