import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/梁嘎嘎/Desktop/BRIGHT_model_icons_v8.pptx'));const s=p.slides.items[0];
function text(t,x,y,w,h,size=21,font='Cambria Math'){const a=s.shapes.add({name:t,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:font,fontSize:size,color:'#25313F',wrap:'none',alignment:'center',verticalAlignment:'middle',insets:{top:0,right:0,bottom:0,left:0}};return a;}
// Native editable text pieces give explicit sub/superscript placement, no literal underscores.
function math(str,x,y,w,h=29,size=23){const parts=[];const re=/([_^])\{([^}]+)\}|([_^])([A-Za-z0-9]+)/g;let pos=0,m;while((m=re.exec(str))){if(m.index>pos)parts.push([str.slice(pos,m.index),0]);parts.push([m[2]??m[4],(m[1]??m[3])==='_'?1:-1]);pos=re.lastIndex;}if(pos<str.length)parts.push([str.slice(pos),0]);function width(t,sz){return [...t].reduce((v,c)=>v+(/[ilj1., ]/.test(c)?.28:/[MWΔΣ]/.test(c)?.82:.56)*sz,0)+2;}let widths=parts.map(([t,k])=>width(t,size*(k?.69:1)));let total=widths.reduce((a,b)=>a+b,0);if(total>w){size*=w/total;widths=parts.map(([t,k])=>width(t,size*(k?.69:1)));total=widths.reduce((a,b)=>a+b,0);}let cx=x+(w-total)/2;parts.forEach(([t,k],i)=>{text(t,cx,y+(k===1?size*.31:k===-1?-size*.23:0),widths[i]+2,h,size*(k?.69:1));cx+=widths[i];});}
const replacements=new Map([
 ['Y = h(F_A + ΔF_mem)','Ŷ = h(F_{A} + ΔF_{mem})'],
 ['T_A = T + ΔT_local','T_{A} = T + ΔT_{local}'],
 ['F_A = F + ΔF_local','F_{A} = F + ΔF_{local}'],
 ['ΔT_local','ΔT_{local}'],['ΔT_mem','ΔT_{mem}'],['ΔF_mem','ΔF_{mem}'],
 ['T_A','T_{A}'],['Pixel pathway F_A','Pixel pathway  F_{A}'],['Key k_b','Key  k_{b}'],
 ['Query T_A','Query  T_{A}'],['Valid mask m_j','Valid mask  m_{j}'],
 ['No valid support: ΔF_mem = 0','No valid support: ΔF_{mem} = 0'],
 ['V_j, y_j, ID_j; valid mask m_j','V_{j}, y_{j}, ID_{j}, m_{j}'],
 ['V_1','V_{1}'],['V_2','V_{2}'],['V_3','V_{3}'],['U_1','U_{1}'],['U_2','U_{2}'],['U_3','U_{3}']
]);
for(const a of [...s.shapes.items]){const t=a.text?.toString()??'';if(replacements.has(t)){const z=a.position;a.delete();math(replacements.get(t),z.left,z.top,z.width,z.height,t.startsWith('No valid')?19:22);}}
for(const a of [...s.shapes.items]){const t=a.text?.toString()??'';const z=a.position;
 if(t==='Project / Q-unpool'){a.delete();math('Q(ΔT_{local}W_{local})',z.left-3,z.top-1,z.width+6,z.height,21);}
 if(t==='Project + Q-unpool'){a.delete();math('ΔF_{mem} = Q(ΔT_{mem}W_{mem})',z.left,z.top,z.width,z.height,20);}
 if(t==='Pool(F, Q)'){a.delete();math('T = Pool(F,Q)',z.left-8,z.top,z.width+16,z.height,21);}
 if(t==='Label embedding\ne(y_j)'){a.delete();text('Label embedding',z.left,z.top,z.width,25,19,'Arial');math('e(y_{j})',z.left,z.top+26,z.width,26,23);}
 if(t==='Train only'){a.text='Train entries';}
 if(t==='Shared weights; j = 1, …, K (3 shown)'){a.text='Shared weights  (3 of K supports shown)';}
 if(t==='Q' && z.top===592){a.text='Query';a.position={...z,left:1028,width:52};}
}
// Clarify that Q denotes association, while attention Q is labelled Query.
for(const a of [...s.shapes.items]){const t=a.text?.toString()??'';if(t==='K,V')a.text='K, V';}
const notes=`Formula convention for the candidate architecture (not a claim about existing code): F has N pixel rows and C channels. Q has N pixel rows and M regional columns, with nonnegative same-building assignment weights. Valid building pixels have row sum 1; background rows are zero. Empty columns are removed. D=diag(Q^T 1_N); T=D^{-1}Q^T F, dimension M by C. Local interaction gives deltaT_local, also M by C; T_A=T+deltaT_local. W_local and W_mem are bias-free C by C output projections. deltaF_local=Q(deltaT_local W_local); F_A=F+deltaF_local. Building k_b=Normalize(Proj(Pool_b(T_A))). Each support j provides regional set V_j and building label y_j, source ID_j and mask m_j. U_j=CrossAttn(T_A,V_j,V_j), independently per current building, with parameters shared across supports. Learned support aggregation takes {U_j,e(y_j),m_j} and produces deltaT_mem, dimension M by C. Invalid supports contribute zero; when none are valid, deltaT_mem is explicitly zero (avoid all-masked softmax). deltaF_mem=Q(deltaT_mem W_mem). Yhat=h(F_A+deltaF_mem) denotes four-class pixel scores; argmax yields the discrete map. Q is always pixel-to-region association, not the attention query variable. These equations define a consistent candidate using regional width C; if another width d is used, an explicit input projection is required before the token residual. Native editable math labels use Cambria Math with positioned subscripts. Region/support icons and output example remain schematic. Image provenance and figure inspirations: BRIGHT ESSD 2025 Fig6; Superpixel Transformers https://arxiv.org/html/2309.16889v2 and RAC https://arxiv.org/pdf/2202.11233. No performance claim.`;
s.speakerNotes.textFrame.setText(notes);
const candidate=root+'/ppt_build/formulas_candidate_v9.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(root+'/ppt_output/BRIGHT_model_formulas_v9.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_model_formulas_v9.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/formulas_v9_validation.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_model_formulas_v9.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_model_formulas_v9.pptx');console.log('Finished formula revision');
