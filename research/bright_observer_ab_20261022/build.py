import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'bright_memcheck_memory_20261022/bright_memcheck_memory_v1.pyz'
EXPECTED = '9c46b2f2137184a73391dd9213d224de5966606b28a149adb3a995eb093f8554'


def adapt(text, edits):
    for old, new in edits:
        if text.count(old) != 1: raise ValueError('Adapter mismatch: ' + old)
        text = text.replace(old, new)
    return text


def build():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED: raise ValueError('Base archive differs')
    with zipfile.ZipFile(BASE) as z: files = {n: z.read(n) for n in z.namelist() if n != '__main__.py'}
    ledger = {n: hashlib.sha256(b).hexdigest() for n,b in files.items()}
    assessment = adapt(files['memory_assessment.py'].decode(), [
        ('from memory_context import TARGET, COUNT, SELECTION', 'from ab_context import TARGET, COUNT, SELECTION'),
        ('from memory_trace import MemoryRoutes as StageRoutes, MemoryTrace', 'from memory_trace import MemoryTrace\nfrom ab_observers import observer_context as StageRoutes, validate_observers'),
        ('from memory_reports import finish_report', 'from ab_reports import finish_report'),
        ("FORMAT='BRIGHT_MEMCHECK_MEMORY_CASE_V1'", "FORMAT='BRIGHT_OBSERVER_AB_CASE_V1'"),
        ("requested,mode,full=context['canonical_test_names'][:COUNT],'watched',False", "requested,mode,full=context['canonical_test_names'][:COUNT],assessment,False"),
        ('import memory_scoring as test_endpoint', 'import ab_scoring as test_endpoint'),
        ("names,mode,full=list(canonical[:COUNT]),'watched',False", "names,mode,full=list(canonical[:COUNT]),assessment,False"),
        ("sanitizer_tool=('compute-sanitizer memcheck' if assessment=='memcheck' else None)", "sanitizer_tool='compute-sanitizer memcheck',extra_route_observers=mode=='watched'"),
        ("test_adapter='memory_scoring.py'", "test_adapter='ab_scoring.py'"),
        ("status['route_evidence']=validate_route_evidence(output,names,mode)", "status['observer_evidence']=validate_observers(output,names,mode)"),
        ("print('MEMORY CASE RESULT '", "print('OBSERVER AB CASE RESULT '")])
    scoring = adapt(files['memory_scoring.py'].decode(), [
        ("format='BRIGHT_TWO_IMAGE_MEMORY_DIAGNOSTIC_V1'", "format='BRIGHT_ONE_IMAGE_OBSERVER_AB_V1'"),
        ("checkpoint_selection='diagnostic_fixed1500_canonical_prefix2_memory'", "checkpoint_selection='diagnostic_fixed1500_first_image_observer_ab'"),
        ("print('MEMORY PREFIX step=%s %s/%s'", "print('OBSERVER AB step=%s %s/%s'")])
    reports = adapt(files['memory_reports.py'].decode(), [
        ('from memory_context import COUNT, STEP, PREFIX_LAST, SELECTION', 'from ab_context import COUNT, STEP, SELECTION\nfrom ab_observers import validate_observers'),
        (' or names[-1] != PREFIX_LAST', ''),
        ("    operators = read_json(output / 'operator_observer_summary.json')", "    validate_observers(output,names,provenance['case'])\n    if provenance['extra_route_observers']:\n        validate_detailed(output)\n    final = copy.deepcopy(report)\n    final.update(format='BRIGHT_ONE_IMAGE_OBSERVER_AB_V1', phase='diagnostic_complete',\n        complete=False, diagnostic_complete=True, full_test_evidence=False, full_validation_evidence=False,\n        final_integrity_pending=False, standalone_integrity_checks=checks,\n        standalone_provenance=provenance, historical_failure_resolved=False, sanitizer_check_pending=True)\n    write_json(path, final)\n    path.with_name('result.json.sha256').write_text(sha256(path)+'  result.json\\n')\n    return final\n\n\ndef validate_detailed(output):\n    operators = read_json(output / 'operator_observer_summary.json')"),
    ])
    # The original detailed-observer checks remain; remove its superseded report writer.
    marker = "    final = copy.deepcopy(report)"
    first, last = reports.find(marker), reports.rfind(marker)
    if first == last: raise ValueError('Expected one replaced finalizer and one old tail')
    reports = reports[:last]
    entry = files['memory_entry.py'].decode()
    # Some control-flow substitutions intentionally occur more than once.
    entry = adapt(entry, [
        ('from memory_context import BASE, CASES, COUNT, FAILED, FORMAT, STEP, preflight', 'from ab_context import BASE, CASES, COUNT, FAILED, FORMAT, STEP, preflight'),
        ("PREFIX = 'memory1500_'", "PREFIX = 'observer_ab1500_'"),
        ("old = FAILED / 'prefix1500_memcheck/sanitizer.log'", "old = FAILED / 'memcheck/sanitizer.log'"),
        ('from memory_assessment import run_assessment', 'from ab_assessment import run_assessment'),
        ("        from night_assessment import validate_route_evidence\n        validate_route_evidence(folder, request['names'], 'watched')", "        from ab_observers import validate_observers\n        validate_observers(folder, request['names'], case)"),
        ("conclusion='Both short cases passed. This does not reproduce the index355 incident or clear hardware.'", "conclusion='Both single-image observer cases completed. Compare memory traces; no historical fault resolution.'"),
        ("if case == 'plain':\n                        status['memcheck_skipped'] = 'Plain two-image execution did not pass; no instrumented retry'", "status['remaining_cases_skipped'] = 'First unsuccessful case stops the pair; no retries'"),
    ])
    for old,new,n in [("if case == 'memcheck':", "if case in CASES:", 3),
                      ('latest_memory1500.txt','latest_observer_ab1500.txt',2)]:
        if entry.count(old)!=n: raise ValueError('Repeated adapter mismatch: '+old)
        entry=entry.replace(old,new)
    entry=entry.replace('已后台启动两图显存对照：普通进程通过后，再运行Sanitizer进程。', '已后台启动单图观察器对照：两段均使用Sanitizer，先light再watched；零训练更新。')
    entry=entry.replace('MEMORY HEARTBEAT','OBSERVER AB HEARTBEAT').replace('MEMORY COMPARISON RESULT','OBSERVER AB COMPARISON RESULT')
    entry=entry.replace('two_image_comparison_complete','observer_pair_complete')
    entry=entry.replace('Bounded two-image plain/memcheck allocator comparison; no training or repair.', 'One-image light/watched observer comparison under memcheck; zero training updates.')
    assessment=assessment.replace("observation_synchronizes=mode=='watched'", 'observation_synchronizes=True')
    for name,data in [('ab_assessment.py',assessment),('ab_scoring.py',scoring),('ab_reports.py',reports),('ab_entry.py',entry)]:
        files[name]=data.encode();(HERE/name).write_text(data)
    for name in ('ab_context','ab_observers'): files[name+'.py']=(HERE/(name+'.py')).read_bytes()
    files['AB_BASE.json']=json.dumps(dict(sha256=EXPECTED,files=ledger),indent=2).encode()
    files['__main__.py']=b'from ab_entry import main\nraise SystemExit(main())\n'
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    out=HERE/'bright_observer_ab_v1.pyz'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    h=hashlib.sha256(out.read_bytes()).hexdigest();(HERE/'SHA256SUMS').write_text(h+'  '+out.name+'\n')
    print(h,out.stat().st_size)


if __name__=='__main__': build()
