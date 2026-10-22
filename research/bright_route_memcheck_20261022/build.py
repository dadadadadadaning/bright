"""Keep the released sweep byte-for-byte; add an explicit diagnostic adapter."""
import hashlib
import json
from pathlib import Path
import zipfile
HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'bright_test_sweep_20261022/bright_test_sweep_v1.pyz'
EXPECTED = 'cb4dc39c75ab29971a9f1fe5d313ec7c1fd4b11dae3aaed5f74b561e9f7a9b70'


def adapt(source, replacements):
    for before, after in replacements:
        if source.count(before) != 1: raise ValueError('Adapter source mismatch: '+before)
        source = source.replace(before, after)
    return source


def build():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED: raise ValueError('Sweep base changed')
    with zipfile.ZipFile(BASE) as z: files = {n:z.read(n) for n in z.namelist() if n != '__main__.py'}
    ledger = {n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    assessment = adapt(files['sweep_assessment.py'].decode(), [
        ('from night_config import TARGET', 'from memcheck_context import TARGET, COUNT, SELECTION'),
        ('from sweep_routes import StageRoutes', 'from memcheck_routes import CorroboratedRoutes as StageRoutes'),
        ('from sweep_context import load_selected,SELECTION', 'from sweep_context import load_selected'),
        ('from sweep_reports import finish_report', 'from memcheck_reports import finish_report'),
        ("FORMAT='BRIGHT_FIVE_STEP_TEST878_CASE_V1'", "FORMAT='BRIGHT_PARENT_IDS_MEMCHECK_CASE_V1'"),
        ("requested,mode,full=context['canonical_test_names'],'watched',True", "requested,mode,full=context['canonical_test_names'][:COUNT],'watched',False"),
        ('import sweep_scoring as test_endpoint', 'import memcheck_scoring as test_endpoint'),
        ("names,mode,full=list(canonical),'watched',True", "names,mode,full=list(canonical[:COUNT]),'watched',False"),
        ("test_adapter='sweep_scoring.py'", "test_adapter='memcheck_scoring.py'"),
        ("split=split,test_accessed=split=='test',test_adapter=", "split=split,test_accessed=split=='test',sanitizer_tool='compute-sanitizer memcheck',instrumentation_changes_execution=True,test_adapter="),
        ("return 0 if status['complete'] else 2 if status.get('failure_phase')=='evaluation' else 3", "return 0 if status.get('diagnostic_complete') else 2 if status.get('failure_phase')=='evaluation' else 3"),
        ("print('TEST SWEEP CASE RESULT '", "print('MEMCHECK PREFIX RESULT '")])
    scoring = adapt(files['sweep_scoring.py'].decode(), [
        ("format='BRIGHT_FIVE_STEP_EXPLORATORY_TEST878_V1'", "format='BRIGHT_PARENT_IDS_PREFIX_DIAGNOSTIC_V1'"),
        ("checkpoint_selection='requested_fixed_steps_500_1000_1500_2000_2500_all_six_arms'", "checkpoint_selection='diagnostic_fixed1500_canonical_test_prefix356'"),
        ("print('TEST step=%s %s/%s'", "print('MEMCHECK PREFIX step=%s %s/%s'")])
    files['memcheck_assessment.py'] = assessment.encode(); files['memcheck_scoring.py'] = scoring.encode()
    for name in ('memcheck_context','memcheck_routes','memcheck_reports','memcheck_entry'):
        files[name+'.py'] = (HERE/(name+'.py')).read_bytes()
    files['MEMCHECK_BASE.json'] = json.dumps(dict(sha256=EXPECTED, files=ledger), indent=2).encode()
    files['__main__.py'] = b'from memcheck_entry import main\nraise SystemExit(main())\n'
    for name,data in files.items():
        if name.endswith('.py'): compile(data, name, 'exec')
    output = HERE/'bright_route_memcheck_v1.pyz'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,22,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16
            z.writestr(info,data)
    for name,data in [('memcheck_assessment.py',assessment),('memcheck_scoring.py',scoring)]: (HERE/name).write_text(data)
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    (HERE/'SHA256SUMS').write_text(checksum+'  '+output.name+'\n'); print(checksum,output)
    return output


if __name__=='__main__': build()
