"""Read-only source audit; writes only to the explicitly supplied output directory.

Text scanning is NOT a claim of manual semantic review. Never unpickles weights.
"""
import ast
import collections
import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

ROOTS = [Path('D:/software/dowmload') / x for x in
         ('BRIGHT-master', 'BRIGHT-eval-results', 'BRIGHT-train-logs', 'BRIGHT-checkpoints')]
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
TEXT = {'.py','.sh','.md','.txt','.json','.jsonl','.csv','.tsv','.log','.yaml','.yml','.toml','.ini','.cfg','.xml','.html','.js','.ps1','.bat','.latest','.pid'}
seen = {}
counts = collections.Counter()
errors = []
streams = {k: (OUT / (k + '.jsonl')).open('w', encoding='utf-8') for k in
           ('text_scan','code_structure','log_scan','json_scan','checkpoint_containers')}
def emit(k, v):
    streams[k].write(json.dumps(v, ensure_ascii=False, default=str) + '\n')
def walk_json(v, path='$'):
    if isinstance(v,dict):
        for k,x in v.items():
            loc=path+'.'+str(k)
            if not isinstance(x,(dict,list)) and re.search(r'miou|iou|f1|decision|pass$|step$|seed$|test_access|processed_images|checkpoint|parent|freeze|batch|accum|crop|learning_rate|epochs|variant|protocol|split|class_count',str(k),re.I):
                yield [loc,x]
            if isinstance(x,(dict,list)): yield from walk_json(x,loc)
    elif isinstance(v,list):
        for i,x in enumerate(v):
            if isinstance(x,(dict,list)): yield from walk_json(x,path+'['+str(i)+']')
with (OUT/'file_inventory.csv').open('w',newline='',encoding='utf-8-sig') as dest:
    w=csv.DictWriter(dest,fieldnames=['root','relative_path','absolute_path','size','mtime','type','status','sha256','duplicate_of'])
    w.writeheader()
    for root in ROOTS:
        print('START',root,flush=True)
        for folder,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
            for name in list(dirs):
                p=Path(folder)/name
                if p.is_symlink():
                    dirs.remove(name); errors.append('Directory link not followed: '+str(p))
            for name in files:
                p=Path(folder)/name
                row={'root':root.name,'relative_path':str(p.relative_to(root)),'absolute_path':str(p),'size':'','mtime':'','type':p.suffix.lower(),'status':'','sha256':'','duplicate_of':''}
                counts[root.name+':files']+=1
                try:
                    st=p.stat();row.update(size=st.st_size,mtime=st.st_mtime)
                    counts[root.name+':bytes']+=st.st_size
                    ext=p.suffix.lower()
                    if ext in TEXT or name in {'AGENTS.md','LICENSE','Dockerfile'}:
                        data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest();row['sha256']=sha
                        if sha in seen:
                            row.update(status='exact_duplicate_text',duplicate_of=seen[sha]);counts['duplicate_text']+=1
                        else:
                            seen[sha]=str(p)
                            try:t=data.decode('utf-8-sig'); encoding='utf-8-sig'
                            except UnicodeDecodeError:
                                try:t=data.decode('gb18030');encoding='gb18030'
                                except UnicodeDecodeError:t=data.decode('utf-8',errors='replace');encoding='utf-8-replacement'
                            row['status']='full_text_machine_scanned';counts['unique_text']+=1
                            emit('text_scan',{'file':str(p),'bytes':len(data),'lines':len(t.splitlines()),'encoding':encoding,'sha256':sha,'semantic_review':'not_implied'})
                            if ext=='.py':
                                try:
                                    a=ast.parse(t); entries=[]
                                    for n in ast.walk(a):
                                        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                                            entries.append({'name':n.name,'line':n.lineno,'end':n.end_lineno,'kind':type(n).__name__})
                                    emit('code_structure',{'file':str(p),'definitions':entries,'imports':[ast.unparse(n) for n in ast.walk(a) if isinstance(n,(ast.Import,ast.ImportFrom))]})
                                except Exception as e:emit('code_structure',{'file':str(p),'parse_error':str(e)})
                            if ext in {'.log','.jsonl'}:
                                hits=[]
                                for i,line in enumerate(t.splitlines(),1):
                                    if re.search(r'valid_miou|best.*miou|Training complete|Traceback|Error:|Terminated|Test monitor|test_accessed.*true|nan\b',line,re.I): hits.append([i,line])
                                emit('log_scan',{'file':str(p),'matches':hits,'tail':t[-1800:]})
                            if ext=='.json':
                                try:
                                    v=json.loads(t);emit('json_scan',{'file':str(p),'top_keys':list(v) if isinstance(v,dict) else [],'fields':list(walk_json(v))})
                                except Exception as e:emit('json_scan',{'file':str(p),'parse_error':str(e)})
                    elif ext in {'.pth','.pt','.ckpt'}:
                        row['status']='binary_container_inspected_no_unpickle'
                        info={'file':str(p),'size':st.st_size,'tensor_values_read':False}
                        try:
                            if zipfile.is_zipfile(p):
                                with zipfile.ZipFile(p) as z:
                                    entries=z.infolist();info.update(format='zip',entries=len(entries),uncompressed_bytes=sum(x.file_size for x in entries),metadata_names=[x.filename for x in entries if not re.search(r'/data/\d+$',x.filename)][:30])
                            else:info['format']='nonzip_not_deserialized'
                        except Exception as e:info['container_error']=str(e)
                        emit('checkpoint_containers',info)
                    else:row['status']='inventoried_binary_or_other_not_content_reviewed'
                except Exception as e:
                    row['status']='read_error';errors.append(str(p)+': '+str(e))
                w.writerow(row);counts['status:'+row['status']]+=1
        print('DONE',root.name,counts[root.name+':files'],flush=True)
for f in streams.values():f.close()
(OUT/'coverage_summary.json').write_text(json.dumps({'counts':counts,'errors':errors,'scope':'All filesystem entries reached without following directory symlinks. Exact text duplicates scanned once. Machine scan is not semantic review. Binary tensors/images not fully interpreted.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'counts':counts,'error_count':len(errors)},ensure_ascii=False),flush=True)
