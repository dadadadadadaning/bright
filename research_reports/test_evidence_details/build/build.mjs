import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';process.env.RUNTIME_NODE=runtime+'/node/bin/node.exe';
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const work='D:/software/dowmload/BRIGHT-master/research_reports/test_evidence_details';
const d=JSON.parse(await fs.readFile(work+'/build/analysis.json','utf8'));await fs.mkdir(work+'/output',{recursive:true});
const W=1600,H=1080;
function txt(s,v,x,y,w,h,size=22,bold=false,color='#222222'){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=v;a.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'left',verticalAlignment:'middle',autoFit:'none'};return a;
}
function rect(s,x,y,w,h,color='#147DA8',width=3){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width}});}
async function pic(s,file,x,y,w,h){s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',fit:'contain',alt:file.split('/').at(-1),position:{left:x,top:y,width:w,height:h}});}
const pct=v=>(100*v).toFixed(2);const delta=v=>(v>=0?'+':'')+(100*v).toFixed(2);
function notes(s,r){s.speakerNotes.textFrame.setText('Source: BRIGHT_test_sixmodels_images.zip, label PNGs, run_manifest.json, and matched BRIGHT GT. Metrics recomputed from label IDs and verified against the saved confusion matrices. Selected Test crops only. No equal-training-budget claim. This is A1, not the proposed region-memory architecture. Case selection was diagnostic after observing predictions, showing both gains and regressions. '+(r?JSON.stringify({sample:r.sample_id,xywh:r.xywh,zoom:r.zoom_xywh,rule:r.zoom_rule}):''));}
const zoom=Presentation.create({slideSize:{width:W,height:H}});
let s=zoom.slides.add();s.background.fill='#FFFFFF';
txt(s,'A1 gains and failures on 18 selected Test crops',60,30,1480,60,34,true);
txt(s,'Pooled pixel metrics, not full-Test performance. DamageFormer is the reference checkpoint.',60,98,1480,40,23);
const values=[['Model','Foreground mIoU','Damaged IoU','Destroyed IoU','Damaged F1']];
for(const [name,v] of Object.entries(d.summary))values.push([name,pct(v.miou3)+'%',pct(v.iou[2])+'%',pct(v.iou[3])+'%',pct(v.damaged_f1)+'%']);
const table=s.tables.add({rows:7,columns:5,left:60,top:175,width:1480,height:420,columnWidths:[340,300,280,280,280],values});
table.borders.assign({fill:'#D5DCE2',width:1});
for(let i=0;i<7;i++)for(let j=0;j<5;j++){const cell=table.getCell(i,j);cell.fill=i===0?'#E8EDF1':i===6?'#ECF4F8':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:23,bold:i===0||i===6,color:'#222222'};}
const a=d.summary.Ours_A1,b=d.summary.DamageFormer;
txt(s,`12 crops improve and 6 decline in foreground mIoU.`,60,640,1480,45,28,true);
txt(s,`Destroyed IoU: ${pct(b.iou[3])}% to ${pct(a.iou[3])}%.  Damaged IoU: ${pct(b.iou[2])}% to ${pct(a.iou[2])}%.`,60,705,1480,45,26);
txt(s,`Damaged predicted Intact: ${b.damaged_as_intact.toLocaleString()} to ${a.damaged_as_intact.toLocaleString()} pixels.`,60,765,1480,45,26);
txt(s,`Intact predicted Damaged: ${b.intact_as_damaged.toLocaleString()} to ${a.intact_as_damaged.toLocaleString()} pixels.`,60,825,1480,45,26);
txt(s,'Improved recall comes with more intact-building false alarms. ROI causality requires the matched A0 control.',60,905,1480,70,23);
txt(s,'Crop sizes vary. Larger crops contribute more pixels. No confidence intervals or full-Test claims.',60,1000,1480,35,18,false,'#555555');notes(s);
for(const r of d.cases){
 s=zoom.slides.add();s.background.fill='#FFFFFF';
 txt(s,r.case_kind,60,24,1480,52,32,true);
 txt(s,`${r.sample_id}     Crop: ${r.xywh[2]} px     Foreground mIoU change: ${delta(r.delta_miou3)} pp     Damaged IoU change: ${delta(r.delta_damaged_iou)} pp`,60,85,1480,48,21);
 const indexes=[0,2,7,8],labels=['Pre-event RGB','Reference GT','DamageFormer','Ours (A1)'];
 for(let j=0;j<4;j++){
  const x=85+j*385;txt(s,labels[j],x,142,310,35,24,true);
  await pic(s,r.paths[indexes[j]],x,190,300,300);rect(s,x,190,300,300,'#B8B8B8',.8);
  const [xx,yy,ww,hh]=r.zoom_xywh;const f=300/r.xywh[2];rect(s,x+xx*f,190+yy*f,ww*f,hh*f);
  await pic(s,r.zoom_paths[indexes[j]],x,570,300,300);rect(s,x,570,300,300);
 }
 txt(s,`Identical blue-box crop in all columns, enlarged ${r.xywh[2]/r.zoom_xywh[2]}x`,85,510,1450,40,25,true,'#147DA8');
 txt(s,'GT and predictions: background white, intact green, damaged tan, destroyed red.',60,910,1480,35,21);
 txt(s,'Boxes mark error changes for diagnosis. They are not learned regions or an attention map.',60,954,1480,35,21);
 txt(s,'Metrics above refer to the full displayed crop, not just the blue box. Case selection uses observed prediction differences.',60,1005,1480,35,18,false,'#555555');notes(s,r);
}
const heat=Presentation.create({slideSize:{width:W,height:H}});
for(const r of d.cases){
 const s=heat.slides.add();s.background.fill='#FFFFFF';
 txt(s,'Damaged-class probability across six models',50,20,1270,50,32,true);
 txt(s,r.sample_id+'   '+r.case_kind,50,83,1240,40,23);
 txt(s,'Same crop and 0–1 probability scale in every panel',50,126,1240,30,20);
 await pic(s,r.paths[2],1420,15,125,125);txt(s,'Reference GT',1400,143,175,30,16);
 for(let j=0;j<6;j++)await pic(s,r.heatmap_paths[j],50+(j%3)*515,180+Math.floor(j/3)*410,460,414);
 txt(s,'P(Damaged), not attention. PNG heatmaps retain the exported color bars. Raw probability arrays were not supplied.',50,1020,1500,35,18,false,'#555555');notes(s,r);
}
async function save(p,name,tableOwners=[]){
 const candidate=work+'/build/'+name+'.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
 for(let i=0;i<p.slides.items.length;i++){
  const img=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(work+'/build/'+name+'_'+(i+1)+'.png',new Uint8Array(await img.arrayBuffer()));
 }
 await finalizePresentation({workspaceDir:work,candidatePath:candidate,finalPath:work+'/output/'+name+'.pptx',pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:['Arial']},requiredNativeTableOwnerSlides:tableOwners,explicitTotalSlideCount:p.slides.items.length,verifyArtifactToolImport:true,receiptPath:work+'/build/'+name+'.validation.json'});console.log('DONE',name);
}
await save(zoom,'BRIGHT_Test_A1_error_analysis_zoom',[1]);
await save(heat,'BRIGHT_Test_Damaged_heatmaps');
