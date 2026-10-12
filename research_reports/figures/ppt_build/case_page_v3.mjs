import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/梁嘎嘎/Desktop/BRIGHT_evidence_compact_v2.pptx'));const s=p.slides.items[0];s.shapes.deleteAll();
const ink='#25313F',pink='#BC7479',blue='#55799C';
function t(v,x,y,w,h=28,sz=21,b=false,color=ink,align='left'){const a=s.shapes.add({name:v,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={typeface:'Arial',fontSize:sz,bold:b,color,wrap:'none',alignment:align,verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};}
function box(x,y,w,h,fill='none',stroke='#C6CED5',dash=false){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1,style:dash?'dash':'solid'}});}
function slot(x,y,w,h,title,color='#F6F8FA'){box(x,y,w,h,color,'#9AA9B7',true);t(title,x+5,y+h/2-14,w-10,28,20,false,'#657384','center');}
function line(x,y,w){box(x,y,w,1,'#D7DDE3','#D7DDE3');}
function circle(x,y,n,col){s.shapes.add({geometry:'ellipse',position:{left:x,top:y,width:29,height:29},fill:col,line:{fill:col,width:0}});t(n,x,y,29,29,17,true,'#FFFFFF','center');}
t('Query regions and retrieved local evidence',35,22,1370,46,34,true);t('Case-study layout • Query A is shared with the prediction comparison',35,75,1340,28,20);
line(35,117,1370);
// The query is a stable left-hand anchor, not repeated across every row.
box(35,142,324,641,'#FFF9F8','#F0DCDA');box(35,142,6,641,pink,pink);
t('a  Current query',54,155,282,34,25,true,pink);t('Query A',54,199,282,28,22,true);
slot(55,250,135,145,'RGB','#FBEEEE');slot(204,250,135,145,'SAR','#FBEEEE');
t('Matched scene crops',55,401,280,25,18);t('Selected region r*',55,445,280,29,22,true);
slot(55,489,284,176,'Actual region overlay','#FBEEEE');
t('Query ID / event: [fill]',55,687,282,27,18);t('GT excluded from retrieval',55,721,282,27,18);
// Region diagnosis is subordinate: compact, parallel comparisons without process arrows.
t('b  Region inspection',393,149,990,34,25,true);const rx=[394,731,1068];
['Grid control','Content regions','Soft assignment'].forEach((q,i)=>{t(q,rx[i],197,310,29,22,true);slot(rx[i],239,310,112,i===2?'Measured Q(:, r*)':'Same query building');});
t('Matched coverage / region count',394,357,652,25,18);t('Assignment scale required',1068,357,310,25,18);
line(393,402,1012);
// Per-building identity is visible; metadata stays attached to its own support.
t('c  Retrieved training supports',393,419,1000,33,25,true,blue);const sx=[394,731,1068];
sx.forEach((x,i)=>{circle(x,468,''+(i+1),blue);t('Support '+(i+1),x+40,466,250,32,23,true,blue);slot(x,514,147,111,'RGB','#F0F4F9');slot(x+162,514,147,111,'SAR','#F0F4F9');t('Label / event: [fill]',x,631,312,24,18);t('Source ID / score: [fill]',x,658,312,24,18);slot(x,696,309,86,'Actual local reading weights','#F0F4F9');});
// Visual grammar: colours denote provenance, not building damage categories.
box(36,808,16,16,'#F5C9C9',pink);t('Current query',61,801,169,28,18);box(252,808,16,16,'#B6C8E4',blue);t('Training support',277,801,190,28,18);
box(503,808,27,17,'none','#647589',true);t('Missing data slot',540,801,180,28,18);
t('Reading weights: normalize within each support',764,801,630,28,18);
t('Template only. No measured maps are shown. Attention indicates reading behaviour, not proven causal importance.',35,850,1370,27,18);
s.speakerNotes.textFrame.setText('Redesigned first page only; second-page comparison preserved. Visual priority: left current Query A with matched RGB/SAR and selected region r*, upper-right secondary partition controls, lower-right primary support cases. Pink represents query provenance; blue represents training-support provenance; neither is a damage class. Numbered circles are support rank, not confidence. Dashed boxes are missing real evidence. No artificial buildings, attention heat maps, predictions or values were generated. Region control images use the same crop and matched count/coverage. Q(:,r*) is assignment weight, not damage probability. Per-building local attention weights need layer/head averaging stated, colour bar and numeric range when filled; independently normalized weights cannot compare total support importance. Source ID, event, building label and retrieval score stay with each training support. Labels are not local damage labels. Exact retrieved IDs must be used. Query A links to second page, where predictions belong. Main architecture formulas/attention computation/database are deliberately absent. All elements editable.');
const candidate=root+'/ppt_build/case_page_v3_candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<2;i++)await fs.writeFile(root+'/ppt_output/BRIGHT_evidence_refined_v3_'+(i+1)+'.png',new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
const sk='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_evidence_refined_v3.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:sk+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:sk+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:2,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/case_page_v3_validation.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_evidence_refined_v3.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_evidence_refined_v3.pptx');console.log('Finished first page redesign');
