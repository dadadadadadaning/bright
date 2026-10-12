import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const rt='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=rt+'/node/node_modules';process.env.RUNTIME_NODE=rt+'/node/bin/node.exe';
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(rt+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const work='D:/software/dowmload/BRIGHT-master/research_reports/test_reselected_examples';
const rows=JSON.parse(await fs.readFile(work+'/build/candidates.json','utf8'));
const chosen=[rows[2],rows[5]];
const p=Presentation.create({slideSize:{width:1800,height:1120}});
function txt(s,v,x,y,w,h,size=24,bold=false,color='#252525'){
 const e=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});e.text=v;e.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};
}
function box(s,x,y,w,h,color='#167DB0',width=3){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width}});}
async function img(s,path,x,y,size){s.images.add({blob:new Uint8Array(await fs.readFile(path)),contentType:'image/png',fit:'contain',alt:path.split('/').at(-1),position:{left:x,top:y,width:size,height:size}});}
const pct=n=>(n*100).toFixed(1);
for(let i=0;i<chosen.length;i++){
 const r=chosen[i],m=r.local,s=p.slides.add();s.background.fill='#FFFFFF';
 txt(s,i===0?'Recovered Damaged region in Turkey':'Recovered Damaged regions in Myanmar',45,20,1710,55,38,true);
 txt(s,r.sample_id+'   /   '+(i===0?'Earthquake':'Hurricane')+'   /   Selected Test example',45,80,1710,40,25);
 const ids=[0,1,2,7,8],labels=['Pre-event RGB','Post-event SAR','Reference GT','DamageFormer','Ours (A1)'];
 for(let j=0;j<5;j++){
  let x=45+j*347;txt(s,labels[j],x,142,325,42,28,true);
  await img(s,r.paths[ids[j]],x,193,320);box(s,x,193,320,320,'#C4C4C4',1);
  const [xx,yy,w,h]=m.zoom_xywh,f=320/r.xywh[2];box(s,x+xx*f,193+yy*f,w*f,h*f);
  await img(s,r.zoom_paths[j],x,580,320);box(s,x,580,320,320);
 }
 txt(s,`Same spatial window in every column (${m.zoom_xywh[2]} x ${m.zoom_xywh[3]} source pixels)`,45,531,1710,40,26,true,'#167DB0');
 txt(s,`Damaged IoU in the enlarged window: ${pct(m.iou_b)}% to ${pct(m.iou_a)}%`,45,924,1710,42,29,true);
 txt(s,i===0?'Remaining error: some GT Destroyed pixels still receive the Damaged label.':'Remaining error: a small intact prediction persists and some background receives building labels.',45,972,1710,38,23);
 txt(s,'Classes: white = background, green = intact, tan = damaged, red = destroyed.',45,1018,1710,33,23);
 txt(s,'Selected after inspecting predictions. These local examples do not establish full-Test gains or region/Memory causality.',45,1063,1710,34,20,false,'#555555');
 s.speakerNotes.textFrame.setText('Sources: BRIGHT_test_sixmodels_images.zip and original paired images/GT in F:/建筑损毁分类数据集. Existing fixed A1 and DamageFormer predictions. No retraining or threshold tuning. Candidates searched only inside the previous 18 selected Test crops. Recovered GT Damaged connected components used to locate diagnostic windows. Components may not equal building instances. Local metrics cover the whole displayed zoom window, including false positives. Original prediction pixels unchanged. '+JSON.stringify({sample_id:r.sample_id,crop:r.xywh,window:m.zoom_xywh,local_metrics:m}));
}
await fs.mkdir(work+'/output',{recursive:true});
const candidate=work+'/build/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<2;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:.8});await fs.writeFile(work+'/build/slide'+(i+1)+'.png',new Uint8Array(await b.arrayBuffer()));}
await finalizePresentation({workspaceDir:work,candidatePath:candidate,finalPath:work+'/output/BRIGHT_A1_reselected_improvements_v2.pptx',pythonExecutable:rt+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','17145000,10668000'],fontPolicy:{basis:'design',families:['Arial']},explicitTotalSlideCount:2,verifyArtifactToolImport:true,receiptPath:work+'/build/validation.json'});
console.log('DONE');
