import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures',build=root+'/ppt_build';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/ppt_output/BRIGHT_model_detailed_v6.pptx'));
const s=p.slides.items[0];
const C={pink:'#F5C9C9',blue:'#B6C8E4',yellow:'#FFD982',green:'#C8DDB0',gray:'#F0F0F0',ink:'#25313F',line:'#424242'};
function box(x,y,w,h,fill=C.gray){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1}});}
function txt(t,x,y,w,h=25,size=20){let a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'Arial',fontSize:size,color:C.ink,wrap:'none',alignment:'center',verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};return a;}
function line(pts,fill='none',closed=false){const x=Math.min(...pts.map(a=>a[0])),y=Math.min(...pts.map(a=>a[1])),w=Math.max(.1,Math.max(...pts.map(a=>a[0]))-x),h=Math.max(.1,Math.max(...pts.map(a=>a[1]))-y);const commands=pts.map((a,i)=>({[i?'lineTo':'moveTo']:{x:a[0]-x,y:a[1]-y}}));if(closed)commands.push({close:{}});return s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1},customPaths:[{width:w,height:h,commands}]});}
function ellipse(x,y,w,h,fill){return s.shapes.add({geometry:'ellipse',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1}});}
function clear(x,y,w,h){for(const a of [...s.shapes.items]){const z=a.position;if(z.left>=x-.2&&z.top>=y-.2&&z.left+z.width<=x+w+.2&&z.top+z.height<=y+h+.2)a.delete();}}
function bars(x,y,col,w=33,n=3,gap=12){for(let i=0;i<n;i++)box(x,y+i*gap,w,7,col);}
function roof(x,y,size,col){line([[x,y],[x+size,y],[x+size,y+size*.64],[x+size*.66,y+size*.64],[x+size*.66,y+size],[x,y+size]],col,true);line([[x+size*.4,y],[x+size*.4,y+size*.4],[x+size*.66,y+size*.64],[x+size,y+size*.36]]);line([[x,y+size*.4],[x+size*.4,y+size*.4]]);}
function card(x,y,w=72,h=60){box(x,y,w,h,'#FFFFFF');roof(x+5,y+7,30,C.blue);bars(x+42,y+8,C.blue,w-48,3,11);}
// Feature tensors: stacked planes, not a measured heat map.
clear(392,182,51,51);for(let i=0;i<3;i++)box(393+i*8,180+i*9,42,42,i===2?C.pink:'#F1E3E3');
// Memory cylinder: supports remain training-only, retrieval remains online.
clear(421,424,213,67);ellipse(433,472,190,25,C.blue);box(433,434,190,50,C.blue);ellipse(433,422,190,25,'#D9E3F0');txt('Memory bank',445,445,166,26,22);txt('Train only',453,469,150,19,16);
// Explicit retrieval, with query and database input arrows unchanged.
clear(677,425,232,65);box(677,425,232,65,C.yellow);ellipse(691,438,24,24,'#FFF2CB');line([[712,458],[724,470]]);for(let i=0;i<3;i++)box(739+i*21,439,15,22,C.blue);txt('Valid Top-K',802,433,101,28,17);txt('Source filter + dedup',716,466,183,22,17);
// A retrieved support is a building plus a SET of tokens, not one prototype.
clear(950,425,280,65);for(let i=0;i<3;i++)card(962+i*84,425,75,59);txt('V_j, y_j, ID_j; valid mask m_j',950,489,285,22,16);
for(let j=0;j<3;j++){const y=608+j*73;clear(902,y,123,61);card(902,y,123,61);txt('V_'+(j+1),911,y+38,95,21,17);}
// Pool and unpool use reciprocal visual vocabulary: many pixels / few tokens.
clear(171,710,180,66);box(171,710,180,66,C.yellow);for(let j=0;j<2;j++)for(let i=0;i<3;i++)box(183+i*10,720+j*10,7,7,C.pink);for(let j=0;j<3;j++)line([[215,724+j*7],[256,730]]);box(259,725,34,10,C.pink);txt('Pool(F, Q)',184,747,150,24,19);
clear(488,710,202,66);box(488,710,202,66,C.yellow);const nodes=[[515,727],[551,720],[590,734],[624,721]];for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++)line([nodes[i],nodes[j]]);for(const [x,y]of nodes)ellipse(x-4,y-4,8,8,C.pink);txt('Attention + FFN',500,747,180,24,19);
clear(289,839,169,66);box(289,839,169,66,C.green);bars(301,850,C.pink,25,2,10);for(let j=0;j<3;j++)line([[329,859],[369,847+j*10]]);for(let j=0;j<3;j++)for(let i=0;i<3;i++)box(372+i*8,846+j*8,6,6,C.pink);txt('Project / Q-unpool',292,878,163,23,17);
// Attach explicit Q/K/V roles without replacing attention with decorative wires.
for(const a of s.shapes.items){if(a.text?.toString()==='Cross-attention'){a.text='Cross-attention';}}
txt('Q',1044,592,22,18,14);txt('K,V',1028,646,36,18,14);
// Bottom disclosure makes abstraction status unambiguous.
for(const a of s.shapes.items){const t=a.text?.toString()??'';if(t.startsWith('*Candidate architecture.'))a.text='*Candidate architecture. Building partitions, token links and support cards are schematic. Output is an example.';}
s.speakerNotes.textFrame.setText('Candidate inference-retrieval architecture. Original paired RGB/SAR and example map retained from v6. Example output is not established as the user model prediction. All new building support cards, feature planes, token links, and partitions are editable schematic diagrams, not data visualizations. Drawing references: Superpixel Transformers for Efficient Semantic Segmentation, Fig. 2, https://arxiv.org/html/2309.16889v2 (pixel/region association and unfolding; its hard drawing stands for soft assignments). RAC, CVPR 2022, Fig. 1, https://arxiv.org/pdf/2202.11233 (external memory and query/support separation). We borrow visual vocabulary only, not their model operations. Existing computational dependencies retained: F/R/P to regions, T_A to key and attention query, valid Top-K on train-only memory, labels/masks to support aggregation, Q to unpool. Shared attention per support; 3 supports illustrate arbitrary K. No support gives zero update. Dashed arrows denote offline construction. RGB/SAR provenance inherited: BRIGHT ESSD 2025 Fig.6, Turkey example, imagery credits require original-source acknowledgement.');
const candidate=build+'/icons_candidate_v8.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(root+'/ppt_output/BRIGHT_model_icons_v8.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_model_icons_v8.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:build+'/icons_validation_v8.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_model_icons_v8.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_model_icons_v8.pptx');
console.log('Finished icons v8');
