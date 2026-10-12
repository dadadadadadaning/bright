import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures',data=root+'/seven_disasters_v2',sk='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const records=JSON.parse(await fs.readFile(data+'/manifest.json','utf8'));
if(records.length!==7)throw Error('Seven events required');
const p=Presentation.create({slideSize:{width:1800,height:1900}}),s=p.slides.add();s.background.fill='#FFFFFF';
function t(v,x,y,w,h,size=23,bold=false,color='#263343'){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={typeface:'Microsoft YaHei',fontSize:size,bold,color,autoFit:'none'};}
t('七类灾害的建筑区域对照',30,25,1740,60,42,true);
t('真实配对影像与 GT 约束分区。当前使用模型：DamageFormer（预测待填）',30,93,1740,40,27);
const cols=['灾前 RGB','灾后 SAR','GT 范围','规则网格','RGB SLIC','选中区域','DamageFormer'];
cols.forEach((v,c)=>t(v,200+c*225,158,219,42,c===6?23:27,true,c===6?'#A84D57':'#263343'));
const cn=['地震','爆炸','火山','野火','洪水','冲突','飓风'];
const places=['土耳其','巴塔','刚果','夏威夷','利比亚','乌克兰','缅甸'];
for(let r=0;r<7;r++){
 const q=records[r],y=225+r*215;
 t(cn[r],30,y+24,160,43,32,true);t(places[r],30,y+76,165,37,25);
 t('样本 '+q.sample_id.split('_').at(-1),30,y+121,165,30,18,false,'#596777');
 t(`网格 ${q.grid_regions} / SLIC ${q.slic_regions}`,30,y+157,165,30,17,false,'#596777');
 for(let c=0;c<6;c++)s.images.add({blob:new Uint8Array(await fs.readFile(`${data}/${q.sample_id}/${String(c+1).padStart(2,'0')}.png`)),contentType:'image/png',position:{left:200+c*225,top:y,width:200,height:200},alt:q.sample_id+' '+cols[c]});
 s.shapes.add({geometry:'rect',position:{left:1550,top:y,width:200,height:200},fill:'#FAFAFA',line:{fill:'#BC7880',width:1}});
 t('待填真实预测',1564,y+65,180,36,23,false,'#805329');t('尚未运行推理',1564,y+105,180,34,20,false,'#697580');
}
t('用途：比较不同灾害场景下的区域形态、碎片与边界。SLIC 仅作对照，不代表学习区域或损毁位置。',30,1745,1740,40,25,true);
t('判断效果需同预算模型消融和 Damaged 指标。DamageFormer 保留原名，暂不标为原创 Ours。',30,1793,1740,37,24);
t('乌克兰、缅甸采用像素 GT 的连通域作为范围，其余采用实例标注。所有区域数如实列出，未严格匹配。SAR 仅作显示拉伸。',30,1840,1740,38,21,false,'#596777');
s.speakerNotes.textFrame.setText('编辑依据：C:/Users/梁嘎嘎/Desktop/BRIGHT_real_regions_onepage_v3.pptx，保留六列区域对照，增加灾害类型与当前模型列。来源 F:/建筑损毁分类数据集。预览用于区域形态检查，未核验样本属于训练还是验证划分，不应作为模型泛化证据。图像为真实数据，SLIC为GT辅助算法对照。DamageFormer预测格全部待填。乌克兰和缅甸无实例JSON，采用target.zip中前景连通域，可能合并相邻建筑。'+JSON.stringify(records));
const candidate=root+'/ppt_build/seven_disasters_candidate.pptx',final=root+'/ppt_output/BRIGHT_regions_seven_disasters_v2.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:sk+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:sk+'/container_tools/inspect_presentation_layout_geometry.py',fontPolicy:{basis:'design',families:['Microsoft YaHei']},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/seven_disasters_validation_v2.json'});
const png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(root+'/ppt_output/BRIGHT_regions_seven_disasters_v1.png',new Uint8Array(await png.arrayBuffer()));
await fs.copyFile(final,'C:/Users/梁嘎嘎/Desktop/BRIGHT_regions_seven_disasters_v2.pptx');console.log('Finished');
