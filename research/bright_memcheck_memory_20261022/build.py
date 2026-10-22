"""Inherit the published memcheck bytes and add separately named two-image adapters."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'bright_route_memcheck_20261022/bright_route_memcheck_v1.pyz'
EXPECTED = '93d2a756138c496a8a3602c821c5de03c970a44137a0c18225c947c2c625a488'


def adapt(text, edits):
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError('Adapter source mismatch: ' + before)
        text = text.replace(before, after)
    return text


def build():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Released memcheck changed')
    with zipfile.ZipFile(BASE) as z:
        files = {n: z.read(n) for n in z.namelist() if n != '__main__.py'}
    ledger = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    assessment = adapt(files['memcheck_assessment.py'].decode(), [
        ('from memcheck_context import TARGET, COUNT, SELECTION', 'from memory_context import TARGET, COUNT, SELECTION'),
        ('from memcheck_routes import CorroboratedRoutes as StageRoutes', 'from memory_trace import MemoryRoutes as StageRoutes, MemoryTrace'),
        ('from memcheck_reports import finish_report', 'from memory_reports import finish_report'),
        ("FORMAT='BRIGHT_PARENT_IDS_MEMCHECK_CASE_V1'", "FORMAT='BRIGHT_MEMCHECK_MEMORY_CASE_V1'"),
        ('import memcheck_scoring as test_endpoint', 'import memory_scoring as test_endpoint'),
        ('observer = None', 'observer = None\n    memory = None'),
        ("bindings = bridge.load_runtime(source)", "memory = MemoryTrace(torch, output)\n        bindings = bridge.load_runtime(source)"),
        ('with bindings.Runtime(args,runtime_output) as runtime:', "with bindings.Runtime(args,runtime_output) as runtime:\n            memory.sample('runtime_ready')"),
        ("status['restoration'] = restore_training_state(torch,heads,optimizers,saved,state)", "status['restoration'] = restore_training_state(torch,heads,optimizers,saved,state)\n            memory.sample('heads_optimizers_restored')"),
        ('del saved', "del saved\n            memory.sample('checkpoint_payload_released')"),
        ("sanitizer_tool='compute-sanitizer memcheck'", "sanitizer_tool=('compute-sanitizer memcheck' if assessment=='memcheck' else None)"),
        ("test_adapter='memcheck_scoring.py'", "test_adapter='memory_scoring.py'"),
        ('mode,TARGET,heads=heads):', 'mode,TARGET,heads=heads,trace=memory):'),
        ("phase='integrity'", "phase='integrity'\n            memory.sample('evaluation_finished')"),
        ("status.update(phase='failed',complete=False,failure_phase=phase,", "if memory is not None: memory.failure(phase+'_failed',primary_error,position)\n        status.update(phase='failed',complete=False,failure_phase=phase,"),
        ('if observer is not None:\n        status.update', "if memory is not None:\n        memory.close()\n        status['memory_rows']=memory.rows\n    if observer is not None:\n        status.update"),
        ("print('MEMCHECK PREFIX RESULT '", "print('MEMORY CASE RESULT '")])
    scoring = adapt(files['memcheck_scoring.py'].decode(), [
        ("format='BRIGHT_PARENT_IDS_PREFIX_DIAGNOSTIC_V1'", "format='BRIGHT_TWO_IMAGE_MEMORY_DIAGNOSTIC_V1'"),
        ("checkpoint_selection='diagnostic_fixed1500_canonical_test_prefix356'", "checkpoint_selection='diagnostic_fixed1500_canonical_prefix2_memory'"),
        ("if (index+1) % 10 == 0 or index+1 == len(names):", 'if True:  # Only two diagnostic images; show each one.'),
        ("print('MEMCHECK PREFIX step=%s %s/%s'", "print('MEMORY PREFIX step=%s %s/%s'")])
    reports = adapt(files['memcheck_reports.py'].decode(), [
        ('from memcheck_context import COUNT, STEP, TARGET, SELECTION', 'from memory_context import COUNT, STEP, PREFIX_LAST, SELECTION'),
        ('names[-1] != TARGET', 'names[-1] != PREFIX_LAST'),
        ("format='BRIGHT_PARENT_IDS_DIAGNOSTIC_PREFIX_V1'", "format='BRIGHT_TWO_IMAGE_MEMORY_DIAGNOSTIC_V1'"),
        ('sanitizer_check_pending=True', "sanitizer_check_pending=provenance.get('sanitizer_tool') is not None")])
    for name, data in [('memory_assessment.py', assessment), ('memory_scoring.py', scoring), ('memory_reports.py', reports)]:
        files[name] = data.encode()
        (HERE / name).write_text(data)
    for name in ('memory_context', 'memory_trace', 'memory_entry'):
        files[name + '.py'] = (HERE / (name + '.py')).read_bytes()
    files['MEMORY_BASE.json'] = json.dumps(dict(sha256=EXPECTED, files=ledger), indent=2).encode()
    files['__main__.py'] = b'from memory_entry import main\nraise SystemExit(main())\n'
    for n, data in files.items():
        if n.endswith('.py'):
            compile(data, n, 'exec')
    output = HERE / 'bright_memcheck_memory_v1.pyz'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    h = hashlib.sha256(output.read_bytes()).hexdigest()
    (HERE / 'SHA256SUMS').write_text(h + '  ' + output.name + '\n')
    print(h, output)


if __name__ == '__main__':
    build()
