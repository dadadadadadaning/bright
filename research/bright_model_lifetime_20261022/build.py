import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'bright_observer_ab_20261022/bright_observer_ab_v1.pyz'
EXPECTED = '49c6aead979cf22248ecea3a84e1a272b33fcfe0224d63b8e435216e09fac517'


def adapt(text, edits):
    for old,new in edits:
        if text.count(old) != 1: raise ValueError('Adapter mismatch: ' + old)
        text = text.replace(old,new)
    return text


def build():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED: raise ValueError('Base archive differs')
    with zipfile.ZipFile(BASE) as z: files={n:z.read(n) for n in z.namelist() if n!='__main__.py'}
    ledger={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    assessment = adapt(files['ab_assessment.py'].decode(), [
        ('from ab_context import TARGET, COUNT, SELECTION','from life_context import TARGET, COUNT, SELECTION'),
        ('from ab_observers import observer_context as StageRoutes, validate_observers',
         'from life_observers import execution, probe_collection, validate_evidence'),
        ('from ab_reports import finish_report','from life_reports import finish_report'),
        ("FORMAT='BRIGHT_OBSERVER_AB_CASE_V1'", "FORMAT='BRIGHT_MODEL_LIFETIME_CASE_V1'"),
        ('import ab_scoring as test_endpoint','import life_scoring as test_endpoint'),
        ("extra_route_observers=mode=='watched'", "extra_route_observers=False, added_success_path_wrappers=mode=='instrumented', explicit_gc_after_scoring=True"),
        ("test_adapter='ab_scoring.py'", "test_adapter='life_scoring.py'"),
        ("            with PassiveRoutes(torch,bindings.original_train.JointRegionMemoryClassifier,output,position), \\\n                 track_evaluation(runtime,factory,evaluator,names,state,observer), \\\n                 StageRoutes(torch,runtime,factory,evaluator,output,position,names,canonical,mode,TARGET,heads=heads,trace=memory):",
         "            with execution(torch,runtime,factory,evaluator,output,position,names,state,observer,\n                           bindings.original_train.JointRegionMemoryClassifier,mode,memory):"),
        ('context[\'seed\']),step=step)', 'context[\'seed\']),step=step,position=position,mode=mode)'),
        ("            status['observer_evidence']=validate_observers(output,names,mode)",
         "            status['gc_observation']=probe_collection(torch,memory,output)\n            status['observer_evidence']=validate_evidence(output,names,mode)"),
        ("print('OBSERVER AB CASE RESULT '", "print('MODEL LIFETIME CASE RESULT '")])
    scoring = adapt(files['ab_scoring.py'].decode(), [
        ('import re\n','import re\nimport sys\n'),
        ('evaluator, batches, step):', 'evaluator, batches, step, position=None, mode=None):'),
        ('                example = runtime.prepare(batch)',
         "                if position is not None and mode == 'minimal':\n                    position.update(index=index,image=names[index],active_arm=None,stage='prepare')\n                example = runtime.prepare(batch)"),
        ('                support = factory.prepare(example,tuple(heads))',
         "                if position is not None: position.update(stage='support_prepare',active_arm=None)\n                support = factory.prepare(example,tuple(heads))"),
        ('                for arm in paths:\n', "                for arm in paths:\n                    if position is not None: position.update(active_arm=arm,stage='forward')\n"),
        ('                    evaluator._validate_probability(prob,example.target,arm)',
         "                    if position is not None: position['stage']='probability_validation'\n                    evaluator._validate_probability(prob,example.target,arm)"),
        ('        state.restore_rng(rng)',
         "        # Preserve the primary error; no CUDA RNG/hash/GC after a failed call.\n        if sys.exc_info()[1] is None: state.restore_rng(rng)"),
        ("format='BRIGHT_ONE_IMAGE_OBSERVER_AB_V1'", "format='BRIGHT_ONE_IMAGE_MODEL_LIFETIME_V1'"),
        ("checkpoint_selection='diagnostic_fixed1500_first_image_observer_ab'", "checkpoint_selection='diagnostic_fixed1500_first_image_model_lifetime'"),
        ("print('OBSERVER AB step=%s %s/%s'", "print('MODEL LIFETIME step=%s %s/%s'")])
    reports=adapt(files['ab_reports.py'].decode(),[
        ('from ab_context import COUNT, STEP, SELECTION','from life_context import COUNT, STEP, SELECTION'),
        ('from ab_observers import validate_observers','from life_observers import validate_evidence'),
        ("    validate_observers(output,names,provenance['case'])\n    if provenance['extra_route_observers']:\n        validate_detailed(output)", "    validate_evidence(output,names,provenance['case'])"),
        ("format='BRIGHT_ONE_IMAGE_OBSERVER_AB_V1'", "format='BRIGHT_ONE_IMAGE_MODEL_LIFETIME_V1'")])
    reports=reports.split('\n\ndef validate_detailed')[0].rstrip()+'\n'
    entry=adapt(files['ab_entry.py'].decode(),[
        ('from ab_context import BASE, CASES, COUNT, FAILED, FORMAT, STEP, preflight',
         'from life_context import BASE, CASES, COUNT, PREVIOUS, FORMAT, STEP, preflight'),
        ("PREFIX = 'observer_ab1500_'", "PREFIX = 'model_lifetime1500_'"),
        ("        old = FAILED / 'memcheck/sanitizer.log'\n        old_titles = headings(old)\n        errors = [s for s in old_titles if 'ERROR SUMMARY' not in s]\n        if (len(errors) != 6 or sanitizer_summary(old)['error_counts'] != [6]\n                or any('Program hit cudaErrorMemoryAllocation' not in s for s in errors)):\n            raise ValueError('Prior sanitizer log is not the six-allocation-error case: ' + '\\n'.join(old_titles))",
         "        old_titles = {case: headings(PREVIOUS / case / 'sanitizer.log') for case in ('light','watched')}\n        if any(sanitizer_summary(PREVIOUS / case / 'sanitizer.log')['error_counts'] != [0] for case in old_titles):\n            raise ValueError('Prior observer sanitizer closure changed')"),
        ('from ab_assessment import run_assessment','from life_assessment import run_assessment'),
        ('from ab_observers import validate_observers','from life_observers import validate_evidence'),
        ("validate_observers(folder, request['names'], case)", "validate_evidence(folder, request['names'], case)"),
        ("completed_images=s.get('completed_images', 0), memory=memory_digest(folder))",
         "completed_images=s.get('completed_images', 0), memory=memory_digest(folder),\n        gc_observation=s.get('gc_observation'))"),
        ("conclusion='Both single-image observer cases completed. Compare memory traces; no historical fault resolution.'",
         "conclusion='Single-image wrapper/lifetime comparison only; inspect pre-GC memory and collection response. Historical route corruption unresolved.'")])
    entry=entry.replace('latest_observer_ab1500.txt','latest_model_lifetime1500.txt')
    entry=entry.replace('已后台启动单图观察器对照：两段均使用Sanitizer，先light再watched；零训练更新。',
                        '已后台启动真实模型生命周期诊断：两段均使用Sanitizer，先minimal再instrumented；每段一张图，推理后测量GC；零训练更新。')
    entry=entry.replace('OBSERVER AB','MODEL LIFETIME').replace('observer_pair_complete','model_lifetime_pair_complete')
    entry=entry.replace('One-image light/watched observer comparison under memcheck; zero training updates.',
                        'One-image minimal/instrumented model lifetime diagnostic; post-scoring GC, zero updates.')
    for name,data in [('life_assessment.py',assessment),('life_scoring.py',scoring),('life_reports.py',reports),('life_entry.py',entry)]:
        files[name]=data.encode();(HERE/name).write_text(data)
    for name in ('life_context','life_observers'): files[name+'.py']=(HERE/(name+'.py')).read_bytes()
    files['MODEL_LIFETIME_BASE.json']=json.dumps(dict(sha256=EXPECTED,files=ledger),indent=2).encode()
    files['__main__.py']=b'from life_entry import main\nraise SystemExit(main())\n'
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    out=HERE/'bright_model_lifetime_v1.pyz'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    h=hashlib.sha256(out.read_bytes()).hexdigest();(HERE/'SHA256SUMS').write_text(h+'  '+out.name+'\n')
    print(h,out.stat().st_size)


if __name__=='__main__':build()
