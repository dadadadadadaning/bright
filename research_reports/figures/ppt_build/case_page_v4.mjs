import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/梁嘎嘎/Desktop/BRIGHT_evidence_refined_v3.pptx'));const s=p.slides.items[0];s.shapes.deleteAll();
function t(v,x,y,w,h=26,sz=20,b=false,col='#25313F',align='left'){const a=s.shapes.add({name:v,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={typeface:'Arial',fontSize:sz,bold:b,color:col,wrap:'none',alignment:align,verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};}
function box(x,y,w,h,fill='none',stroke='#BCC7D1',dash=false){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1,style:dash?'dash':'solid'}});}
function slot(x,y,w,label,sub,fill='#F7F8FA'){box(x,y,w,w,fill,'#A5B2BF',true);t(label,x+4,y+w*.38,w-8,28,20,false,'#596878','center');t(sub,x+4,y+w*.38+32,w-8,45,16,false,'#596878','center');}
function rule(y){box(35,y,1370,1,'#D5DDE3','#D5DDE3');}
function bar(x,y,w,title){t(title,x,y-28,w,25,18);const colors=['#440154','#482878','#3E4989','#31688E','#26828E','#1F9E89','#35B779','#6DCD59','#B4DE2C','#FDE725'];for(let i=0;i<10;i++)box(x+i*w/10,y,w/10,12,colors[i],colors[i]);t('0',x,y+14,35,22,16);t('1',x+w-30,y+14,30,22,16,false,'#25313F','right');}
t('What does the model read?',35,23,1370,45,34,true);t('Query A: one selected region and its local reading maps on retrieved training buildings',35,78,1350,28,21);rule(119);
t('a  Matched query crops',40,138,330,31,23,true,'#AA686F');t('b  Same-building region inspection',391,138,970,31,23,true);
slot(40,197,145,'RGB','Pre-event','#FCF0EF');slot(200,197,145,'SAR','Post-event','#FCF0EF');t('Same scene and crop',40,351,310,26,18);t('Query ID / event: [fill]',40,381,320,26,18);
const xx=[390,635,880,1125];['Selected region r*','Grid control','Content regions','Soft assignment'].forEach((a,i)=>t(a,xx[i],178,210,28,21,true));
slot(390,215,200,'RGB + boundaries','Highlight r*\nFade other regions','#FCF0EF');slot(635,215,200,'RGB + grid','Matched coverage\nand region count');slot(880,215,200,'RGB + boundaries','Actual model partition\nSame scale as grid');slot(1125,215,200,'Q(:, r*) overlay','Measured weights\nNot damage probability');
t('Boundary colour denotes region identity',390,422,700,27,18);bar(1125,462,200,'Assignment weight');
rule(507);t('c  Local reading from the same query region r*',40,518,1040,31,24,true,'#55799C');t('2 of K supports shown',1130,520,270,28,20);
// Square measured-map slots have priority over the number of support examples.
for(let j=0;j<2;j++){
 const x=40+j*700;t('Rank '+(j+1)+'  /  training building',x,562,650,30,23,true,'#55799C');
 slot(x,611,115,'RGB','Actual support','#F0F4F9');slot(x+130,611,115,'SAR','Paired crop','#F0F4F9');
 t('Building label: [fill]',x,737,265,25,18);t('Event / source ID: [fill]',x,765,270,25,18);t('Retrieval score: [fill]',x,793,265,25,18);
 slot(x+285,607,225,'Reading from r*','RGB + weight overlay\nActual support regions','#F0F4F9');
 // Vertical colour scale uses absolute attention weights, independently normalized per support.
 const colors=['#FDE725','#B4DE2C','#6DCD59','#35B779','#1F9E89','#26828E','#31688E','#3E4989','#482878','#440154'];
 for(let k=0;k<10;k++)box(x+525,625+k*16,14,16,colors[k],colors[k]);t('1',x+546,619,25,24,17);t('0',x+546,769,25,24,17);
 t('Reading weight',x+285,838,280,24,18);
}
t('Template only: dashed slots await measured data. Reading weights normalize within each support; they do not measure causal importance.',35,875,1370,21,16);
s.speakerNotes.textFrame.setText('第一页重新优化，第二页不变。六个上排素材：Query A真实配对RGB/SAR（相同裁剪）；选中区域r*的RGB叠加边界（其余区域淡化）；同建筑网格；同建筑内容分区；Q[:,r*]软归属。网格和内容区域同底图同尺度，数量与覆盖匹配。下排只展示两栋真实Top-K训练支持，每栋配对RGB/SAR、建筑标签、事件/来源ID、检索分数，以及225x225的方形局部读取权重叠加图。两个读取图均由同一个r*产生。色条采用顺序Viridis采样颜色，定义0到1的权重映射，不表示已有数据或观测范围；实际权重可落在其子区间，不应按各图极值独立拉伸。软归属与局部读取是不同量，分别标Assignment weight和Reading weight。读取权重在每栋支持内独立归一化，必须在填图后注明attention层和head平均规则，不能从最大值比较两栋支持总贡献。粉蓝只表示query/support来源，不能用作损毁类别。两个案例是展示数量，不是K=2。所有素材仍待填，未伪造建筑图、GT、注意力或实验输出。');
const candidate=root+'/ppt_build/case_page_v4_candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);for(let i=0;i<2;i++)await fs.writeFile(root+'/ppt_output/BRIGHT_evidence_square_v4_'+(i+1)+'.png',new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
const sk='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_evidence_square_v4.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:sk+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:sk+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:2,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/case_page_v4_validation.json'});await fs.copyFile(root+'/ppt_output/BRIGHT_evidence_square_v4.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_evidence_square_v4.pptx');console.log('Finished square-map version');
