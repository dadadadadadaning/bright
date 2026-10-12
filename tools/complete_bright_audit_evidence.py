"""Extract auditable curve records and retry Windows long paths; no training."""
import csv
import json
import re
import os
import sys
import zipfile
import hashlib
import ast
from pathlib import Path

out = Path(sys.argv[1])
root = Path('D:/software/dowmload/BRIGHT-train-logs')
records = []
details = []
for relative, variant in [
    ('DAMAGEFORMER_JOINT_REGION_IMAGE_V2/t0.log', 't0'),
    ('DAMAGEFORMER_JOINT_REGION_IMAGE_V2/sequence_t1_t3.log', 'unknown'),
    ('DAMAGEFORMER_ITERATIVE_PARTITION_V1/iter3_priority.log', 'iter3'),
    ('DAMAGEFORMER_ITERATIVE_PARTITION_V1/sequence.log', 'iter_sequence')]:
    path = root / relative
    step = 0
    last = {}
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        start = re.search(r'===== (t\d) START', line)
        if start:
            variant = start[1]
            step = 0
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            item = {}
        if isinstance(item, dict) and 'step' in item:
            step = item['step']
            last = item
        match = re.search(r'Val loc_F1=([\d.]+)% foreground mIoU=([\d.]+)% best=([\d.]+)% step=(\d+)', line)
        if match:
            records.append(dict(file=str(path), line=line_no, variant=variant,
                observed_optimizer_step=step, loc_f1_percent=float(match[1]),
                foreground_miou_percent=float(match[2]), best_percent=float(match[3]),
                best_step=int(match[4])))
    details.append(dict(file=str(path), last_training_record=last))
with (out/'joint_and_iterative_validation_curves.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(out/'joint_and_iterative_log_tails.json').write_text(json.dumps(details, indent=2), encoding='utf-8')

retries = []
with (out/'file_inventory.csv').open(encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        if row['status'] != 'read_error':
            continue
        path = '\\\\?\\' + str(Path(row['absolute_path']).resolve())
        entry = dict(original=row['absolute_path'], extended_path=path)
        try:
            entry['bytes'] = os.stat(path).st_size
            with zipfile.ZipFile(path) as archive:
                entry['members'] = len(archive.infolist())
            entry['status'] = 'binary_container_inspected_no_unpickle_after_long_path_retry'
        except Exception as error:
            entry['error'] = str(error)
        retries.append(entry)
cache = r'\\?\D:\software\dowmload\BRIGHT-master\bda_benchmark\experiments\rgid_s9ic_clean\.pytest_cache'
try:
    retries.append(dict(path=cache, entries=os.listdir(cache)))
except Exception as error:
    retries.append(dict(path=cache, error=str(error)))
(out/'read_error_retries.json').write_text(json.dumps(retries, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(validation_rows=len(records), retries=retries), ensure_ascii=False))

# Include code backups and less common source formats omitted by the first pass.
# Store hashes/structure only, never dump environment-file values or secrets.
extra_ext = {'.sha256','.ndjson','.mjs','.cmd','.rels','.iml','.cjs','.ts',
    '.map','.conf','.lock','.cts','.svg','.markdown','.mts','.css','.license',
    '.webmanifest','.gypi','.ninja','.cc','.apache','.bsd','.typed','.cs',
    '.bnf','.bash','.fish','.zsh','.gyp','.h','.ejs','.env','.cuh','.cpp','.cu','.ipynb'}
extra = []
with (out/'file_inventory.csv').open(encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        if row['status'] != 'inventoried_binary_or_other_not_content_reviewed':
            continue
        path = Path(row['absolute_path'])
        if not (path.suffix in extra_ext or '.bak' in path.name or '.backup' in path.name
                or '.old_' in path.name or not path.suffix or path.suffix == '.0e-5'):
            continue
        item = dict(file=str(path))
        try:
            data = path.read_bytes()
            if b'\x00' in data:
                item['status'] = 'binary_detected_not_decoded'
            else:
                text = data.decode('utf-8-sig', errors='replace')
                item.update(status='supplemental_full_text_machine_scan', bytes=len(data),
                    lines=len(text.splitlines()), sha256=hashlib.sha256(data).hexdigest())
                if '.py.' in path.name:
                    try:
                        tree = ast.parse(text)
                        item['python_definitions'] = [n.name for n in ast.walk(tree)
                            if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
                    except Exception as error:
                        item['parse_error'] = str(error)
        except Exception as error:
            item['error'] = str(error)
        extra.append(item)
with (out/'supplemental_text_scan.jsonl').open('w',encoding='utf-8') as stream:
    for item in extra:
        stream.write(json.dumps(item,ensure_ascii=False)+'\n')
print(json.dumps(dict(supplemental_entries=len(extra),
    supplemental_texts=sum(x.get('status')=='supplemental_full_text_machine_scan' for x in extra))))
