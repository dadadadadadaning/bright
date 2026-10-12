import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures',build=root+'/ppt_build';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/梁嘎嘎/Desktop/BRIGHT_model_MoAT_palette_v4.pptx'));
const s=p.slides.items[0];s.shapes.deleteAll();s.background.fill='#FFFFFF';
const C={pink:'#F5C9C9',blue:'#B6C8E4',yellow:'#FFD982',green:'#C8DDB0',gray:'#F0F0F0',ink:'#25313F',line:'#424242'};
function box(x,y,w,h,fill=C.gray,name=''){return s.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1}});}
function txt(t,x,y,w,h=29,size=22,bold=false,align='left'){let sh=s.shapes.add({name:t,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=t;sh.text.style={typeface:'Arial',fontSize:size,bold,color:C.ink,wrap:'none',alignment:align,verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};return sh;}
function op(t,x,y,w,h=48,fill=C.yellow,size=21){box(x,y,w,h,fill,t);txt(t,x+6,y+2,w-12,h-4,size,false,'center');}
function poly(pts,fill='none',closed=false,dash=false){const x=Math.min(...pts.map(q=>q[0])),y=Math.min(...pts.map(q=>q[1])),w=Math.max(.1,Math.max(...pts.map(q=>q[0]))-x),h=Math.max(.1,Math.max(...pts.map(q=>q[1]))-y);const cmd=pts.map((q,i)=>({[i?'lineTo':'moveTo']:{x:q[0]-x,y:q[1]-y}}));if(closed)cmd.push({close:{}});return s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:C.line,width:1.3,style:dash?'dash':'solid'},customPaths:[{width:w,height:h,commands:cmd}]});}
function arrow(pts,dash=false){poly(pts,'none',false,dash);const [x,y]=pts.at(-1),[a,b]=pts.at(-2),th=Math.atan2(y-b,x-a),l=8,v=3.5;poly([[x,y],[x-l*Math.cos(th)+v*Math.sin(th),y-l*Math.sin(th)-v*Math.cos(th)],[x-l*Math.cos(th)-v*Math.sin(th),y-l*Math.sin(th)+v*Math.cos(th)]],C.line,true);}
function plus(x,y){s.shapes.add({geometry:'ellipse',position:{left:x-16,top:y-16,width:32,height:32},fill:'#FFFFFF',line:{fill:C.line,width:1.2}});txt('+',x-13,y-17,26,34,27,true,'center');}
function tokens(x,y,col=C.pink,n=3,w=45){for(let i=0;i<n;i++)box(x,y+i*18,w,11,col,'Region token');}
function grid(x,y,col=C.pink){for(let j=0;j<3;j++)for(let i=0;i<3;i++)box(x+i*17,y+j*17,14,14,col,'Feature entry');}
function building(x,y,col=C.pink,centres=false){poly([[x,y],[x+65,y],[x+65,y+45],[x+43,y+45],[x+43,y+66],[x,y+66]],centres?'#FFFFFF':col,true);if(centres){for(const [dx,dy] of [[16,15],[46,19],[21,48]])s.shapes.add({geometry:'ellipse',position:{left:x+dx,top:y+dy,width:6,height:6},fill:C.ink,line:{fill:C.ink,width:0}});}else{poly([[x+27,y],[x+27,y+26],[x+44,y+43],[x+65,y+25]]);poly([[x,y+26],[x+27,y+26]]);poly([[x,y+49],[x+17,y+49],[x+44,y+43]]);}}
// a: retained inputs and output image from the user's source deck.
txt('a  Building retrieval and local evidence reading',25,8,980,37,28,true);
txt('Pre-event RGB',10,162,183);txt('Post-event SAR',7,289,187);
box(205,76,278,235,C.gray);txt('DamageFormer',231,89,235,32,27,true,'center');
op('RGB encoder',220,137,132,44,C.gray,20);op('SAR encoder',220,238,132,44,C.gray,20);grid(393,183);txt('Fusion',386,244,90);
arrow([[152,120],[181,120],[181,159],[220,159]]);arrow([[152,247],[220,247]]);
arrow([[352,159],[373,159],[373,194],[391,194]]);arrow([[352,260],[373,260],[373,222],[391,222]]);
box(545,76,248,235,'#FBE2DF');txt('Regional context (b)',556,92,231);building(568,139);tokens(708,149);arrow([[636,172],[690,172]]);txt('Soft regions',554,218,141,26,20);txt('Tokens',707,218,78,26,20);txt('Pool + interact',586,266,189);
arrow([[483,198],[545,198]]);txt('F, R, P',489,158,100,27,20);
box(850,76,260,235,'#EDF2F8');txt('Support reading (c)',862,92,235);tokens(871,149);tokens(1034,149,C.blue);
txt('Query',867,122,91,23,19);txt('Support',1022,122,87,23,19);
for(let i=0;i<3;i++)poly([[916,155+i*18],[1034,155+(2-i)*18]]);
arrow([[978,207],[978,235]]);txt('Read + aggregate',875,263,225);
arrow([[793,198],[850,198]]);txt('T_A',801,160,55,25,21);
arrow([[780,76],[780,58],[1174,58],[1174,178]]);txt('Pixel pathway F_A',850,26,300);
arrow([[1110,198],[1158,198]]);txt('ΔF_mem',1114,235,105,26,20);plus(1174,198);
arrow([[1190,198],[1211,198],[1211,118],[1240,118]]);op('Pixel classifier',1240,86,230,61,C.green,24);
arrow([[1355,147],[1355,164]]);txt('Example map*',1270,276,217);
txt('F: pixels    R: structure    P: building probability',211,315,520,25,20);
// Explicit query-key formation and legal retrieval, below the main route.
op('Building pool + projection',535,358,265,42,C.green,20);arrow([[672,311],[672,348],[670,348],[670,358]]);txt('T_A',684,336,55,24,19);
txt('Training pairs',27,411,150,25,21);grid(55,443,'#DFE3E8');
op('Matched feature\npipeline',183,425,197,65,'#F3B488',21);arrow([[107,467],[183,467]],true);
op('Memory bank\nTrain entries only',422,425,211,65,C.blue,21);arrow([[380,457],[422,457]],true);
op('Valid Top-K\nSource filter + dedup',677,425,232,65,C.yellow,21);arrow([[633,457],[677,457]]);
arrow([[800,379],[823,379],[823,425]]);txt('Key k_b',830,368,94,25,19);
op('Supports {V_j, y_j, ID_j}\n+ valid mask m_j',950,425,280,65,'#EDF2F8',20);arrow([[909,457],[950,457]]);
arrow([[1090,425],[1090,347],[1020,347],[1020,311]]);txt('Local support sets',1103,355,223,26,20);
txt('Y = h(F_A + ΔF_mem)',1250,447,245,33,22);
// b: grouping, soft assignment, regional update, and two explicit outputs.
box(20,517,713,448,'#FAFAFA');txt('b  Building-constrained regional update',38,529,683,35,25,true);
txt('P',41,586,27);arrow([[62,604],[94,604]]);building(98,581,C.pink,true);
txt('Outline + centres',67,650,175,25,19);
op('Same-building\ncandidates',211,584,151,64,C.yellow,20);arrow([[164,609],[211,609]]);
op('Soft association',397,584,185,64,C.yellow,20);arrow([[362,613],[397,613]]);
txt('R',466,555,28,24,20);arrow([[479,579],[479,584]]);
arrow([[582,613],[619,613]]);grid(624,589);txt('Q',643,647,30,25,21);
grid(50,722);txt('Pixels F',38,774,105,25,21);
op('Weighted pool\nT = Pool(F,Q)',171,710,180,66,C.yellow,21);arrow([[99,744],[171,744]]);
arrow([[650,639],[650,685],[268,685],[268,710]]);
tokens(392,721);txt('T',401,773,35,25,21);arrow([[351,744],[391,744]]);
op('Region attention\n+ FFN',488,710,202,66,C.yellow,21);arrow([[437,744],[488,744]]);
arrow([[587,776],[587,818],[370,818],[370,839]]);txt('ΔT_local',515,786,120,25,21);
op('Project +\nQ-unpool',289,839,169,66,C.green,22);txt('Q',339,930,25,24,21);arrow([[351,931],[351,905]]);
plus(227,872);arrow([[289,872],[243,872]]);arrow([[50,744],[30,744],[30,872],[211,872]]);arrow([[227,888],[227,931],[280,931]]);txt('F_A = F + ΔF_local',65,901,220,25,20);
op('T_A = T + ΔT_local',488,879,217,50,C.pink,21);arrow([[587,818],[587,879]]);txt('T',460,887,27,27,21);arrow([[478,904],[488,904]]);
txt('Updated query tokens',494,933,220,24,19);
// c: per-building attention, shared parameters, label/mask aggregation, Q-unpool.
box(753,517,725,448,'#FAFAFA');txt('c  Per-building support reading',771,529,690,35,25,true);
txt('Shared weights; j = 1, …, K (3 shown)',988,565,470,26,20);
tokens(779,699);txt('Query T_A',763,758,132,25,20);
for(let j=0;j<3;j++){
 const y=608+j*73;
 box(902,y,123,61,'#FFFFFF');tokens(917,y+8,C.blue,3,47);txt('V_'+(j+1),968,y+18,53,25,20);
 op('Cross-attention',1070,y+7,181,48,C.yellow,20);arrow([[1025,y+31],[1070,y+31]]);
 arrow([[824,723],[861,723],[861,y-10],[1160,y-10],[1160,y+7]]);
 tokens(1290,y+10,C.blue,2,44);arrow([[1251,y+31],[1290,y+31]]);txt('U_'+(j+1),1342,y+17,70,26,21);
 poly([[1410,y+31],[1450,y+31],[1450,869]]);
}
op('Label embedding\ne(y_j)',777,836,185,61,C.blue,20);arrow([[962,868],[1030,868]]);
op('Support aggregation',1030,841,285,51,C.yellow,21);arrow([[1450,869],[1315,869]]);
txt('Valid mask m_j',1251,809,204,26,20);arrow([[1327,835],[1327,847],[1315,847]]);
op('Project + Q-unpool',1186,915,250,36,C.green,21);arrow([[1170,892],[1170,933],[1186,933]]);
txt('Q',1388,886,38,26,21);arrow([[1400,912],[1400,915]]);
txt('ΔT_mem',1058,901,127,24,20);txt('ΔF_mem',1058,937,120,24,20);arrow([[1436,933],[1455,933],[1455,959],[1165,959]]);
txt('No valid support: ΔF_mem = 0',776,915,365,26,20);
// Caption remains outside the computation panels.
txt('Pink: current regions    Blue: support cases    Yellow: interaction    Green: projection / decoding',25,978,1450,26,21);
txt('*Candidate architecture. Region icons and example map are illustrative. Dashed arrows: offline construction.',25,1007,1450,25,20);
s.speakerNotes.textFrame.setText('Expanded candidate model based on the user-provided BRIGHT_model_MoAT_palette_v4.pptx. Source images retained. BRIGHT inputs: Chen et al., ESSD 2025, doi:10.5194/essd-17-6217-2025, Fig. 6 Turkey event, imagery credits Table 2. Output image provenance as a model prediction is not established, so it remains Example map. Q is pixel-to-region soft assignment; T=Pool(F,Q); local interaction produces deltaT_local and T_A=T+deltaT_local. Projection and Q-unpool produce pixel-space deltaF_local, then F_A=F+deltaF_local. Building pool and projection produce key k_b. Top-K searches only eligible training supports, enforcing source restrictions and deduplication. During inference the train-only memory remains active. V_j is the support-local token set (attention keys and values); T_A is the query. Attention weights are shared over j. U_j is query-aligned local evidence. Support label embedding e(y_j) and mask m_j enter support aggregation, not local ground truth. Source IDs are used for eligibility filtering. Projection and Q-unpool produce deltaF_mem. Zero legal support produces zero increment. Memory refresh, differentiability of retrieval, training exclusion, and choice of event restrictions must be defined in the implementation. No new performance or novelty claim is made.');
await(await PresentationFile.exportPptx(p)).save(build+'/detail_candidate.pptx');
await fs.writeFile(root+'/ppt_output/BRIGHT_model_detailed_v6.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:build+'/detail_candidate.pptx',finalPath:root+'/ppt_output/BRIGHT_model_detailed_v6.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:build+'/detail_validation_v6.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_model_detailed_v6.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_model_detailed_v6.pptx');
console.log('Finished detailed editable deck');
