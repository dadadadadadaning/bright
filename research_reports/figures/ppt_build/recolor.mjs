import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures',build=root+'/ppt_build';
const deps='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES=deps;
const require=createRequire(import.meta.url),Zip=require(deps+'/jszip'),xml=require(deps+'/xml-js');
const source=build+'/desktop_source.pptx';
const zip=await Zip.loadAsync(await fs.readFile(source));
const doc=xml.xml2js(await zip.file('ppt/slides/slide1.xml').async('string'),{compact:true});
const arr=v=>Array.isArray(v)?v:v?[v]:[];
const meta=new Map(arr(doc['p:sld']['p:cSld']['p:spTree']['p:sp']).map(s=>[s['p:nvSpPr']['p:cNvPr']._attributes.id,s]));
const p=await PresentationFile.importPptx(await FileBlob.load(source)),s=p.slides.items[0];
const palette={'263f50':'263544','445d6d':'566675',f3f6f8:'F2F4F6',bccbd4:'BDC6CF',eef5fa:'EEF2F6','9abbd0':'7799BA',eef6fa:'EAF2F9',c4deeb:'B8D2EA','8fbad2':'6FA0CC',fff4e5:'FFF1E5',cca36c:'C88750',fffdfa:'FFFFFF',d4b182:'C88750',efdabd:'F2CBA9',dcb17a:'DA9862',f6e7d3:'FBE7D7',bf9155:'AC612A',e6ba7d:'E0A271',f0d2a6:'F2CBA9',f3dfc2:'FBE7D7',d19b57:'BE7137',b77e39:'B46932'};
function recol(c){return '#'+(palette[c?.toLowerCase()]||c||'FFFFFF');}
for(const sh of s.shapes.items){
 const z=meta.get(sh.id),pr=z?.['p:spPr'];
 const f=pr?.['a:solidFill']?.['a:srgbClr']?._attributes.val;
 const line=pr?.['a:ln'],lc=line?.['a:solidFill']?.['a:srgbClr']?._attributes.val;
 if(f)sh.fill=recol(f);
 if(lc)sh.line={fill:recol(lc),width:+(line._attributes?.w||12700)/9525,style:line['a:prstDash']?._attributes.val==='dash'?'dash':'solid'};
 if(sh.text.toString())sh.text.color='#263544';
 if(sh.name==='Editable 1 rect')sh.position={left:0,top:0,width:1500,height:1040};
 const n=+(sh.name?.match(/^Editable (\d+)/)||[])[1];
 if([30,114].includes(n)){sh.fill='#F4F8FC';sh.line={fill:'#8AAFD0',width:1.5};}
 if([43,115].includes(n)){sh.fill='#FFFAF5';sh.line={fill:'#D5A67D',width:1.5};}
}
function find(t){return s.shapes.items.find(v=>v.text.toString()===t);}
function edit(old,t,x,y,w,size=24){const v=find(old);if(!v)return;v.text=t;if(x!==undefined)v.position={left:x,top:y,width:w,height:32};v.text.style={typeface:'Arial',fontSize:size,color:'#263544',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};}
function label(t,x,y,w,size=20,c='#496071'){const v=s.shapes.add({name:'Label '+t,geometry:'textbox',position:{left:x,top:y,width:w,height:size+7},fill:'none',line:{fill:'none',width:0}});v.text=t;v.text.style={typeface:'Arial',fontSize:size,color:c,wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};}
edit('Memory','Memory bank',565,393,165);
edit('Damage map','Example map*',1268,275,210);
edit('Building prior','Predicted building',43,562,220,23);
edit('Q','Soft assignment Q',385,686,200,21);
edit('Local region sets','Support = local tokens',1165,352,310,23);
edit('+ building labels','+ building label + source ID',1165,384,330,22);
label('Soft regions',559,226,145,19,'#376B99');label('Tokens',704,226,90,19,'#376B99');
label('Query',865,125,85,18,'#376B99');label('Support',1025,125,105,18,'#A55F2F');
label('Pixel features',365,222,145,19);
label('Addition',1130,220,95,18);
label('V₁',974,585,38,19,'#A55F2F');label('V₂',974,684,38,19,'#A55F2F');label('V₃',974,783,38,19,'#A55F2F');
label('Query tokens',698,766,135,18,'#376B99');
label('Readouts',1228,568,112,19,'#A55F2F');
edit('Rectangles: operations   ·   Bars/grids: features   ·   ⊕: addition   ·   Dashed arrows: offline construction','Blue: current regions     Orange: historical support     Gray: base network     Dashed: offline',25,930,1440,23);
edit('Candidate model: regions, support cards and output map are schematic, not measured results.','*Example map and region/support icons illustrate the workflow, not validated model outputs.',25,963,1440,23);
const notes=s.speakerNotes.textFrame;
notes.setText('Source: user-provided Desktop/BRIGHT_model_editable_v1.pptx. Retains user image edits and model topology. Inputs: Chen et al., BRIGHT (ESSD 2025), doi:10.5194/essd-17-6217-2025, Figure 6 Turkey event; imagery credits Table 2. The output image is retained from the supplied deck and is labelled Example map because its provenance as a model prediction has not been established. Blue denotes current-region representations, orange historical support, and gray base-network components. V_j: local-token set of support building j. U_j: current-query-aligned readout from that support. Q: soft pixel-to-region assignment. Icon colours do not represent measured feature values.');
await(await PresentationFile.exportPptx(p)).save(build+'/recolored_candidate.pptx');
await fs.writeFile(build+'/recolored_preview.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
await finalizePresentation({workspaceDir:root,candidatePath:build+'/recolored_candidate.pptx',finalPath:root+'/ppt_output/BRIGHT_model_colors_labels_v3.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:build+'/colors_validation_v3.json'});
await fs.copyFile(root+'/ppt_output/BRIGHT_model_colors_labels_v3.pptx','C:/Users/梁嘎嘎/Desktop/BRIGHT_model_colors_labels_v3.pptx');
console.log('Finished');
