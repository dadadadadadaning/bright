import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures',sk='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const p=Presentation.create({slideSize:{width:1800,height:820}}),s=p.slides.add();s.background.fill='#FFFFFF';
function txt(v,x,y,w,h,size=23,bold=false,color='#263343'){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={typeface:'Microsoft YaHei',fontSize:size,bold,color,autoFit:'none'};}
txt('建筑损毁预测对比（待填预测）',30,20,1740,55,38,true);
txt('DamageFormer 暂列为当前使用模型。所有模型采用相同输入、裁剪与类别配色',30,83,1740,38,25);
const names=['灾前 RGB','灾后 SAR','参考标注 GT','UNet','DeepLabV3+','SiamAttnUNet','SiamCRNN','ChangeMamba','DamageFormer'];
names.forEach((v,i)=>txt(v,30+i*196,140,194,37,i>3?21:23,true,i===8?'#A84D57':'#263343'));
txt('当前使用模型',1598,177,190,30,19,false,'#A84D57');
const records=JSON.parse(await fs.readFile(root+'/real_region_preview_v2/manifest.json','utf8')).slice(0,2);
for(let r=0;r<2;r++){
 const q=records[r],top=250+r*240;
 txt((r===0?'土耳其地震':'巴塔爆炸')+'  '+q.sample_id,30,top-35,1500,30,21,false,'#596777');
 for(let c=0;c<9;c++){
  const x=30+c*196;
  if(c<3){const file=c===2?`${root}/comparison_gt_v1/${q.sample_id}.png`:`${root}/real_region_preview_v2/${q.sample_id}/${c===0?'01':'02'}.png`;
   s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',position:{left:x,top,width:180,height:180},alt:q.sample_id+' '+names[c]});
  }else{
   s.shapes.add({geometry:'rect',position:{left:x,top,width:180,height:180},fill:'#FAFAFA',line:{fill:c===8?'#BC7880':'#C6CDD3',width:1}});
   txt('待填真实预测',x+12,top+64,166,37,21,false,'#697580');
   txt('尚未执行推理',x+16,top+103,160,30,18,false,'#697580');
  }
 }
}
txt('类别：白色＝背景    绿色＝完整    浅褐色＝损毁    红色＝毁坏',30,705,1740,36,24);
txt('注意：这两处裁剪的 GT 仅含背景与完整建筑，可检查误报，不能验证 Damaged 漏检改善。',30,751,1740,34,23,true,'#805329');
s.speakerNotes.textFrame.setText('真实素材来自 F:/建筑损毁分类数据集 的 pre-event.zip、post-event.zip、target.zip。GT与输入使用相同裁剪。'+JSON.stringify(records)+'\n所有预测格尚未执行推理。DamageFormer是现有方法，不应在论文中冒充原创模型或61.8959%的Memory组合。当前两个裁剪GT仅有0、1类。');
const draft=root+'/ppt_build/comparison_layout_candidate.pptx',final=root+'/ppt_output/BRIGHT_comparison_DamageFormer_v1.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:sk+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:sk+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/comparison_layout_validation.json'});
const png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(root+'/ppt_output/BRIGHT_comparison_DamageFormer_v1.png',new Uint8Array(await png.arrayBuffer()));
await fs.copyFile(final,'C:/Users/梁嘎嘎/Desktop/BRIGHT_comparison_DamageFormer_v1.pptx');console.log('Finished');
