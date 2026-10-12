import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
process.env.RUNTIME_NODE=runtime+'/node/bin/node.exe';
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const work='D:/software/dowmload/BRIGHT-master/research_reports/test_comparison_ppt';
const data=JSON.parse(await fs.readFile(work+'/build/data.json','utf8'));
await fs.mkdir(work+'/output',{recursive:true});
const headers=['Pre-event\nRGB','Post-event\nSAR','Reference\nGT','UNet','DeepLabV3+','SiamAttnUNet','SiamCRNN','DamageFormer','Ours (A1)'];
function text(slide,txt,x,y,w,h,size=16,bold=false,color='#222222'){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=txt;s.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',autoFit:'none'};return s;
}
async function page(p,rows,W,H,size,subtitle){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'BRIGHT building damage assessment',20,15,W-40,32,24,true);
 text(s,subtitle,20,51,W-40,24,14,false,'#555555');
 const gap=3,left=W-24-9*size-8*gap,top=130;
 for(let j=0;j<9;j++)text(s,headers[j],left+j*(size+gap),83,size,40,size>140?17:13,j===8,j===8?'#9C2831':'#222222');
 for(let i=0;i<rows.length;i++){
  const r=rows[i];const y=top+i*(size+gap);
  text(s,r.event.replaceAll('-','\n')+'\n#'+r.sample_id.split('_').at(-1)+'\n'+r.xywh[2]+' px',5,y,left-9,size,size>140?15:12);
  for(let j=0;j<9;j++){
    s.images.add({blob:new Uint8Array(await fs.readFile(r.paths[j])),contentType:'image/png',alt:r.sample_id+' '+headers[j].replace('\n',' '),fit:'contain',position:{left:left+j*(size+gap),top:y,width:size,height:size}});
    s.shapes.add({geometry:'rect',position:{left:left+j*(size+gap),top:y,width:size,height:size},fill:'none',line:{fill:'#AAAAAA',width:.45}});
  }
 }
 const y=top+rows.length*(size+gap)+12;
 text(s,'Background (white)     Intact (green)     Damaged (tan)     Destroyed (red)',left,y,W-left-20,24,14);
 text(s,'Selected Test crops with four GT classes. A1 step 15000. Cloud screening is heuristic.',20,y+28,W-40,25,13,false,'#555555');
 text(s,'Different crop sizes share the same display size. These examples do not represent full-Test performance.',20,y+53,W-40,25,13,false,'#555555');
 s.speakerNotes.textFrame.setText('Source: user-provided BRIGHT_test_sixmodels_images.zip, predictions/run_manifest.json and selection.json. RGB/SAR/GT extracted from local original BRIGHT archives, matched by sample ID and xywh. Verified GT class counts against the selection record. Prediction masks come from the uploaded run only. Six checkpoints use the same display crop per row. Ours_A1 SHA256 cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add. Automatically screened Test examples, not full-Test performance and not evidence of a matched retraining budget. 7 hazard types, 10 of 14 events. Missing Beirut, Haiti, Morocco and Marshall. ChangeOS and ChangeMamba are not included.\n'+rows.map(r=>`${r.sample_id}: crop=${r.xywh}; GT counts=${r.class_pixels}`).join('\n'));
 return s;
}
async function deliver(name,W,H,batches,size){
 const p=Presentation.create({slideSize:{width:W,height:H}});
 for(let i=0;i<batches.length;i++)await page(p,batches[i],W,H,size,batches.length===1?'18 Test cases from 10 events across 7 hazard types':`Test comparison ${i+1} of ${batches.length} (18 cases in total)`);
 const candidate=work+'/build/'+name+'.pptx';await (await PresentationFile.exportPptx(p)).save(candidate);
 for(let i=0;i<p.slides.items.length;i++){
   const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
   await fs.writeFile(work+'/build/'+name+'_'+(i+1)+'.png',new Uint8Array(await png.arrayBuffer()));
 }
 await finalizePresentation({workspaceDir:work,candidatePath:candidate,finalPath:work+'/output/'+name+'.pptx',pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`],fontPolicy:{basis:'design',families:['Arial']},explicitTotalSlideCount:batches.length,verifyArtifactToolImport:true,receiptPath:work+'/build/'+name+'.validation.json'});
 console.log('DONE',name);
}
await deliver('BRIGHT_Test_18cases_overview',1200,2330,[data.rows],112);
await deliver('BRIGHT_Test_18cases_enlarged',1600,1080,[data.rows.slice(0,5),data.rows.slice(5,10),data.rows.slice(10,14),data.rows.slice(14)],160);
