import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';
const sk='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const data=root+'/real_region_preview_v2';
const rows=JSON.parse(await fs.readFile(data+'/manifest.json','utf8')).slice(0,2);
const p=Presentation.create({slideSize:{width:1440,height:1000}}),s=p.slides.add();
s.background.fill='#FFFFFF';
function t(v,x,y,w,h,size=23,bold=false,color='#263343'){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=v;a.text.style={typeface:'Microsoft YaHei',fontSize:size,bold,color,autoFit:'none'};
}
t('建筑内区域划分：真实影像对照与评价边界',30,22,1380,55,36,true);
t('目的：检查固定网格与内容分区如何组织同一栋建筑，为后续区域聚合选择表示方式',30,83,1380,40,23);
const labels=['灾前 RGB','灾后 SAR','GT 建筑轮廓','规则网格','RGB SLIC 分区','选中区域'];
for(let c=0;c<6;c++)t(labels[c],30+c*233,135,218,36,24,true);
for(let r=0;r<2;r++){
 const row=rows[r],y=210+r*265;
 t(`${r===0?'土耳其地震':'巴塔爆炸'}  ${row.sample_id}`,30,y-35,950,32,20,false,'#596777');
 t(`网格 ${row.grid_regions} 区 / SLIC ${row.slic_regions} 区`,1110,y-35,300,32,20,false,'#596777');
 for(let c=0;c<6;c++)s.images.add({blob:new Uint8Array(await fs.readFile(`${data}/${row.sample_id}/${String(c+1).padStart(2,'0')}.png`)),contentType:'image/png',position:{left:30+c*233,top:y,width:210,height:210},alt:row.sample_id+' '+labels[c]});
}
t('当前能看出什么',30,715,420,38,27,true);
t('SLIC 随 RGB 外观形成不规则分区\n部分边界细碎，需检查稳定性\n粉色突出选中区域，不代表损毁位置',30,760,435,135,21);
t('怎样检查区域质量',505,715,430,38,27,true);
t('检查跨楼、背景混入与碎片比例\n控制区域数量，检查连通性与稳定性\n此处用 GT 限制范围，不能验证定位',505,760,435,135,21);
t('怎样证明你的设计有效',980,715,430,38,27,true);
t('同预算对比像素、网格与学习区域\n评价 Damaged 的 IoU、精确率和召回率\n检查各事件收益，单独消融 Memory',980,760,430,135,21);
t('说明：此图为 GT 轮廓约束的 SLIC 对照，尚未运行你的学习区域模块，不能据此宣称本方法有效。',30,908,1380,34,21,true,'#805329');
t('同坐标裁剪，未额外配准。SAR 仅做 1–99% 显示拉伸。巴塔两种分区数量不同，暂不构成严格等数量对照。',30,953,1380,30,18,false,'#596777');
s.speakerNotes.textFrame.setText('来源：F:/建筑损毁分类数据集。'+JSON.stringify(rows)+'\n用途：区域表示诊断，不是模型效果图。没有真实内部损毁边界标注，不能用这些图验证局部损毁定位准确性。GT辅助区域限制避免跨楼是构造结果，不能作为定位模块成功的证据。颜色同质性只是辅助诊断，不能证明损毁判别更好。建议固定训练预算与父权重比较像素、网格、SLIC及模型学习区域，再对Memory作独立消融。');
const draft=root+'/ppt_build/real_pairs_onepage_candidate.pptx',final=root+'/ppt_output/BRIGHT_real_regions_onepage_v3.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:sk+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:sk+'/container_tools/inspect_presentation_layout_geometry.py',fontPolicy:{basis:'design',families:['Microsoft YaHei']},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/real_pairs_onepage_validation_v3.json'});
const png=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(root+'/ppt_output/BRIGHT_real_regions_onepage_v2.png',new Uint8Array(await png.arrayBuffer()));
await fs.copyFile(final,'C:/Users/梁嘎嘎/Desktop/BRIGHT_real_regions_onepage_v3.pptx');
console.log('Finished');
