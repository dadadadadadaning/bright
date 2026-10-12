import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=Presentation.create({slideSize:{width:1440,height:900}});let s;
function tx(t,x,y,w,h=28,size=21,bold=false,align='left'){let a=s.shapes.add({name:t,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'Arial',fontSize:size,bold,color:'#25313F',wrap:'none',alignment:align,verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};return a;}
function box(x,y,w,h,fill='#F6F8FA',dash=true){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'#9BA9B6',width:1,style:dash?'dash':'solid'}});}
function rule(y){box(35,y,1370,1,'#CDD4DB',false);}
function slot(x,y,w,h,title,sub='',fill='#F6F8FA'){box(x,y,w,h,fill);tx(title,x+5,y+h/2-23,w-10,28,20,false,'center');if(sub)tx(sub,x+5,y+h/2+6,w-10,25,17,false,'center');}
function page(title,subtitle){s=p.slides.add();s.background.fill='#FFFFFF';tx(title,35,25,1370,45,34,true);tx(subtitle,35,79,1370,27,20);rule(122);}
page('Regional evidence and retrieved support cases','Query A links this mechanism figure to the prediction comparison. Layout template, awaiting measured data.');
tx('a  Region inspection',40,136,600,31,24,true);
const xs=[45,390,735,1080];const titles=['RGB + predicted outline','Grid control','Content regions','Soft assignment'];
titles.forEach((t,i)=>{tx(t,xs[i],175,300,28,21,true,'center');slot(xs[i],210,300,120,i===0?'Query A crop':i===1?'Matched region count':i===2?'Actual partition':'Selected region weights',i===3?'Assignment scale required':'Insert matched crop');});
tx('Same building and crop for all four views. Region colours identify regions, not damage classes.',45,340,1320,26,19);
rule(382);
tx('b  Query and retrieved training buildings',40,397,1320,32,24,true);
const cols=[45,390,735,1080];['Current query A','Support rank 1','Support rank 2','Support rank 3'].forEach((t,i)=>tx(t,cols[i],441,300,30,23,true,'center'));
cols.forEach((x,i)=>{const tint=i===0?'#FCF0EF':'#F0F4F9';slot(x,481,143,139,'RGB','Actual crop',tint);slot(x+157,481,143,139,'SAR','Paired crop',tint);if(i===0){tx('Query ID / event: [fill]',x,630,310,25,18);tx('Query GT excluded from retrieval',x,658,320,25,18);}else{tx('Building label / event: [fill]',x,629,310,25,18);tx('Source ID / score: [fill]',x,657,310,25,18);}});
cols.forEach((x,i)=>slot(x,698,300,104,i===0?'Selected query region':'Local reading weights',i===0?'Actual partition':'Within-support normalization',i===0?'#FCF0EF':'#F0F4F9'));
tx('State the attention layer and head averaging. Attention weights do not establish causal importance.',45,813,1330,25,19);
tx('Dashed frames require actual outputs. Supports must come from the training memory. No predictions appear on this page.',35,855,1370,24,18);
s.speakerNotes.textFrame.setText('机制案例图模板，原区域页压缩为上方四列，原Memory页保留查询与支持材料。删除分区到token示意、流程箭头、计算框、公式和预测前后对比。图a使用同一Query A建筑的RGB轮廓、网格、内容区域、某区域软归属。网格与内容区域数量及覆盖需匹配。软分配硬化边界只用于显示。图b使用真实训练支持ID，标签是建筑级，不是局部损毁GT。读取权重须记录层和头的平均方法，各支持独立归一化时不能用最大权重比较整栋贡献。不把注意力图解释为因果或真实损毁位置。案例页没有任何真实实验素材，全部虚线框待填。不重新绘制主模型结构。Query A与下一页同名样本严格对应。');
page('Qualitative prediction comparison','One crop per row. Identical evaluation protocol across model variants. Layout template, awaiting predictions.');
const cx=[205,405,605,805,1005,1205];['Pre-event RGB','Post-event SAR','Ground truth','Pixel baseline','+ Regions','+ Regions + Memory'].forEach((t,i)=>tx(t,cx[i],141,180,49,i===5?18:20,true,'center'));
for(let r=0;r<4;r++){let y=209+139*r;tx('Query '+String.fromCharCode(65+r),40,y+23,155,29,23,true);tx('Event / ID: [fill]',40,y+59,160,27,17);cx.forEach((x,c)=>slot(x,y,180,116,c<2?'Image crop':c===2?'GT map':'Prediction','Awaiting data'));}
rule(784);tx('Class legend',40,799,160,28,20,true);['Background','Intact','Damaged','Destroyed'].forEach((t,i)=>{box(220+i*235,805,20,20,['#FFFFFF','#6F9F9B','#E5B95F','#BE7980'][i],false);tx(t,250+i*235,798,180,30,20);});
tx('Use verified cases, including failure cases. Neutral query labels make no claim about module effectiveness.',35,855,1370,25,18);
s.speakerNotes.textFrame.setText('预测对比模板。仅保留RGB、SAR、GT、像素基线、加区域、加区域和Memory六列。与机制页共享Query A。其余Query B-D根据实际结果选择，至少纳入真实失败或退化案例，不事先设定哪个模块一定有效。删除原模板Region effect/Memory effect等先验结论性行标签。每行同裁剪范围和尺度。模型间预算/父权重/评估协议需要可比，否则只能称版本差异，不能作模块因果归因。预测图必须来自真实模型输出，不使用参考论文GT冒充。本页不显示检索案例、分区过程、注意力或结构模块，避免与机制页及主图重复。类别配色需与实际class ID对应。');
const candidate=root+'/ppt_build/evidence_compact_v2_candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<2;i++)await fs.writeFile(root+'/ppt_output/BRIGHT_evidence_compact_v2_'+(i+1)+'.png',new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_evidence_compact_v2.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:2,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/evidence_compact_v2_validation.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_evidence_compact_v2.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_evidence_compact_v2.pptx');console.log('Finished two-page evidence layout');
