import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=Presentation.create({slideSize:{width:1440,height:900}});let s;
const C={ink:'#25313F',line:'#607184',pink:'#F5C9C9',blue:'#B6C8E4',yellow:'#FFD982',green:'#C8DDB0',paper:'#F6F8FA'};
function tx(t,x,y,w,h=30,size=23,bold=false,align='left'){let a=s.shapes.add({name:t,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'Arial',fontSize:size,bold,color:C.ink,wrap:'none',alignment:align,verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};return a;}
function rect(x,y,w,h,fill='none',dash=false){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1,style:dash?'dash':'solid'}});}
function ln(pts,fill='none',close=false){const x=Math.min(...pts.map(q=>q[0])),y=Math.min(...pts.map(q=>q[1])),w=Math.max(.1,Math.max(...pts.map(q=>q[0]))-x),h=Math.max(.1,Math.max(...pts.map(q=>q[1]))-y);const cmds=pts.map((q,i)=>({[i?'lineTo':'moveTo']:{x:q[0]-x,y:q[1]-y}}));if(close)cmds.push({close:{}});s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1.2},customPaths:[{width:w,height:h,commands:cmds}]});}
function arrow(x,y,xx,yy){ln([[x,y],[xx,yy]]);const a=Math.atan2(yy-y,xx-x);ln([[xx,yy],[xx-9*Math.cos(a)+4*Math.sin(a),yy-9*Math.sin(a)-4*Math.cos(a)],[xx-9*Math.cos(a)-4*Math.sin(a),yy-9*Math.sin(a)+4*Math.cos(a)]],C.line,true);}
function slot(x,y,w,h,label,sub='Insert measured output',tint=C.paper){rect(x,y,w,h,tint,true);tx(label,x+8,y+h/2-25,w-16,29,21,false,'center');tx(sub,x+8,y+h/2+7,w-16,26,17,false,'center');}
function page(n,title,sub){s=p.slides.add();s.background.fill='#FFFFFF';tx(n,34,24,55,48,38,true);tx(title,93,24,1300,48,34,true);tx(sub,93,79,1290,29,20);ln([[35,122],[1405,122]]);}
function roof(x,y,w,col=C.pink){ln([[x,y],[x+w,y],[x+w,y+w*.62],[x+w*.68,y+w*.62],[x+w*.68,y+w],[x,y+w]],col,true);ln([[x+w*.42,y],[x+w*.42,y+w*.4],[x+w*.68,y+w*.62],[x+w,y+w*.3]]);ln([[x,y+w*.4],[x+w*.42,y+w*.4]]);ln([[x,y+w*.73],[x+w*.3,y+w*.73],[x+w*.68,y+w*.62]]);}
function bars(x,y,col=C.pink,n=3){for(let i=0;i<n;i++)rect(x,y+19*i,52,11,col);}
function footer(t){tx(t,35,849,1370,29,18);}
// Figure 2: actual-data slots above, explicit conceptual mechanism below.
page('02','Region formation and pixel association','Query A throughout the figure. Dashed frames are data slots, not experimental observations.');
const xs=[45,390,735,1080];const headings=['a  Pre-event RGB','b  Predicted buildings','c  Region boundaries','d  Building close-up'];
headings.forEach((t,i)=>tx(t,xs[i],140,310,31,24,true));
slot(45,181,300,225,'RGB crop','Same coordinates as SAR');
slot(390,181,300,225,'P + centres','Predicted outline, not GT');
slot(735,181,300,225,'Boundary overlay','Hard display of soft assignment');
slot(1080,181,300,225,'Selected building','Crop from panel c');
for(let i=0;i<3;i++)arrow(xs[i]+305,288,xs[i+1]-12,288);
tx('One scene, one crop, one selected building',45,416,790,27,20);
ln([[35,459],[1405,459]]);
tx('e  Partition controls',45,476,580,30,24,true);tx('f  Assignment and token interpretation',740,476,650,30,24,true);
slot(45,522,265,190,'Grid partition','Match region count and coverage');slot(340,522,265,190,'Content regions','Actual model output');
tx('Use the same predicted building mask',45,724,570,29,20);
roof(755,540,110);tx('Region r',755,665,110,28,21,false,'center');arrow(880,593,938,593);bars(955,565);tx('Region token',917,665,129,28,21,false,'center');
slot(1090,522,290,190,'Soft assignment Q(:, r)','Insert weights with a labelled scale');
tx('Schematic only',756,707,274,26,18,false,'center');
tx('Colour identifies regions, not damage classes',740,758,650,28,20);
footer('Template only. Replace dashed frames with matched outputs. No model-generated region boundaries are shown yet.');
s.speakerNotes.textFrame.setText('Figure 2 template. Supply query RGB crop, building probability P, predicted connected components and centres, soft assignment Q, hard argmax display and identical-coordinate grid control. Remove empty regions before plotting. For a chosen region r, show Q[:,r] with its real numeric scale. Region colours are identifiers, not damage labels. Hard boundaries are for visualization only. The roof/token illustration is a conceptual schematic, not a learned output. Keep model/version/query ID and crop coordinates in the final caption.');
// Figure 3: true records and attention slots, no invented scores.
page('03','Retrieved buildings and local support reading','One query and three displayed supports. Retrieve buildings first, then inspect local evidence.');
tx('a  Current query',45,141,320,32,24,true);tx('b  Retrieved training supports',465,141,880,32,24,true);
slot(45,189,170,175,'RGB','Query A');slot(230,189,170,175,'SAR','Query A');
const sx=[475,785,1095];sx.forEach((x,i)=>{tx('Rank '+(i+1),x,179,285,28,22,true,'center');slot(x,215,135,130,'RGB','Train support', '#F0F4F9');slot(x+147,215,135,130,'SAR','Same support','#F0F4F9');tx('Building label: [fill]',x,351,290,27,19);tx('Event / ID: [fill]',x,380,290,27,19);tx('Retrieval score: [fill]',x,409,290,27,19);});
tx('Query ID / event: [fill]',45,377,360,26,19);tx('Query GT is for evaluation only',45,413,390,26,19);
ln([[35,458],[1405,458]]);tx('c  Reading from a selected query region',45,475,1050,30,24,true);
slot(45,520,355,155,'Selected region r*','Actual query partition');
sx.forEach((x)=>slot(x,520,282,155,'Local attention weights','Normalize within this support','#F0F4F9'));
tx('State the layer and head averaging rule. Attention weights are not retrieval scores.',45,685,1320,28,20);
tx('d  Query prediction',45,744,295,30,24,true);
slot(350,733,300,78,'GT','Actual query labels');slot(680,733,300,78,'Without Memory','Same query and crop');slot(1010,733,370,78,'With Memory','Same query and crop');
footer('Template only. Fill supports from real retrieval IDs. Building labels do not label every local token.');
s.speakerNotes.textFrame.setText('Figure 3 template. Use the same Query A as Figure 2 and a corresponding row of Figure 4. Obtain actual top-K support IDs from a train-only memory using the fixed source exclusion protocol. Include actual RGB/SAR pairs for each support, event/source IDs, building labels and exact retrieval score definition. Do not use validation/test images as supports. Display one selected query region and actual per-support attention weights; document layer/head averaging and local normalization. Do not compare maxima from independent softmax distributions as total support importance. Add learned support aggregation weights only if explicitly available from the implementation. Duplicate this page only later if a verified failure example is required.');
// Figure 4: aligned comparison sheet.
page('04','Qualitative damage assessment','Matched crops across every column. Populate all methods with predictions from the same evaluation protocol.');
const cx=[205,405,605,805,1005,1205];const heads=['Pre-event RGB','Post-event SAR','Ground truth','Pixel baseline','+ Regions','+ Regions + Memory'];
heads.forEach((t,i)=>tx(t,cx[i],142,180,48,i===5?19:21,true,'center'));
const rows=[['Query A','Damaged miss'],['Query B','Region effect'],['Query C','Memory effect'],['Query D','Failure case']];
rows.forEach(([q,purpose],r)=>{const y=209+r*139;tx(q,40,y+24,150,30,23,true);tx(purpose,40,y+57,155,30,18);cx.forEach((x,c)=>slot(x,y,180,116,c<2?'Image crop':c===2?'GT map':'Prediction','Awaiting data',c===5?'#F3F7EF':C.paper));});
ln([[35,784],[1405,784]]);tx('Class legend',40,800,150,25,20,true);const lc=['#FFFFFF','#6F9F9B','#E5B95F','#BE7980'];['Background','Intact','Damaged','Destroyed'].forEach((t,i)=>{rect(220+i*235,805,20,20,lc[i]);tx(t,250+i*235,798,190,31,20);});
footer('Template only. Row descriptions are selection targets, not findings. Use verified cases and one shared class palette.');
s.speakerNotes.textFrame.setText('Figure 4 comparison template, not experimental results. Intended row types are damaged false negative, regional effect, added memory effect, and failure; these types are not claims until real cases are found. Change or drop row purposes that are unsupported. All six columns use the same geographic crop and output resolution per row. Keep model checkpoint/protocol consistent with the reported ablation. Use actual GT and predictions, never recolour source-paper reference maps as model outputs. Legend is a proposed consistent palette and must be mapped to the actual class IDs. If magnified insets are added, use identical positions and scales across columns. Qualitative cases do not replace aggregate Damaged precision/recall/F1 and event-level evaluation.');
const candidate=root+'/ppt_build/BRIGHT_evidence_templates_candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<3;i++)await fs.writeFile(root+'/ppt_output/BRIGHT_evidence_fig'+(i+2)+'.png',new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_figures_234_editable_v1.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:3,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/evidence_templates_validation.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_figures_234_editable_v1.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_figures_234_editable_v1.pptx');console.log('Finished 3 editable figure templates');
