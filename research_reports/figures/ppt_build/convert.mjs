import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {Presentation,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const require=createRequire(import.meta.url);
process.env.RUNTIME_NODE_MODULES='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const sharp=require(process.env.RUNTIME_NODE_MODULES+'/sharp');
const {chromium}=require('C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='D:/software/dowmload/BRIGHT-master/research_reports/figures';
const skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const page=await browser.newPage({viewport:{width:1500,height:1040}});
await page.setContent('<body style="margin:0">'+await fs.readFile(root+'/BRIGHT_real_inputs_overview_v2.svg','utf8')+'</body>');
const elements=await page.evaluate(()=>{
 const root=document.querySelector('svg'); const out=[];
 for(const e of root.querySelectorAll('rect,circle,ellipse,path,text,svg')){
  if(e.closest('defs')||e.ownerSVGElement!==root)continue;
  const s=getComputedStyle(e),m=e.getCTM(),b=e.getBoundingClientRect();
  const o={type:e.tagName,x:b.x,y:b.y,w:b.width,h:b.height,fill:s.fill,stroke:s.stroke,sw:parseFloat(s.strokeWidth),dash:s.strokeDasharray};
  if(e.tagName==='svg'){o.type='image';o.index=out.filter(v=>v.type==='image').length;Object.assign(o,{x:+e.getAttribute('x'),y:+e.getAttribute('y'),w:+e.getAttribute('width'),h:+e.getAttribute('height')});out.push(o);continue;}
  if(e.tagName==='text'){Object.assign(o,{text:e.textContent,size:parseFloat(s.fontSize),bold:parseInt(s.fontWeight)>=600});out.push(o);continue;}
  if(e.tagName==='path'){
   o.parts=[];
   for(const d of e.getAttribute('d').split(/(?=M)/).filter(Boolean)){
    const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',d);
    const len=p.getTotalLength(),pts=[];
    for(let i=0,n=Math.max(2,Math.ceil(len/2));i<=n;i++){const q=p.getPointAtLength(len*i/n);const t=new DOMPoint(q.x,q.y).matrixTransform(m);pts.push({x:t.x,y:t.y});}
    o.parts.push({pts,closed:/z/i.test(d)});
   }
   o.arrow=s.markerEnd!=='none';
  }
  out.push(o);
 } return out;
});
const imgs=[];for(const left of [198,291])imgs.push(await sharp(root+'/../../tmp/pdfs/bright_figure_review/page-12.png').extract({left,top:806,width:86,height:87}).png().toBuffer());
await browser.close();
const p=Presentation.create({slideSize:{width:1500,height:1040}}),slide=p.slides.add();
slide.background.fill='#ffffff';
const color=s=>s==='none'?'none':s.startsWith('rgb')?'#'+s.match(/\d+/g).slice(0,3).map(v=>(+v).toString(16).padStart(2,'0')).join(''):s;
let count=0;
function poly(pts,fill,stroke,sw,closed,name,dash=false){
 const x=Math.min(...pts.map(q=>q.x)),y=Math.min(...pts.map(q=>q.y));
 const w=Math.max(.1,Math.max(...pts.map(q=>q.x))-x),h=Math.max(.1,Math.max(...pts.map(q=>q.y))-y);
 const cmd=pts.map((q,i)=>({[i?'lineTo':'moveTo']:{x:q.x-x,y:q.y-y}}));if(closed)cmd.push({close:{}});
 slide.shapes.add({name,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:color(fill),line:{fill:color(stroke),width:sw,style:dash?'dash':'solid'},customPaths:[{width:w,height:h,commands:cmd}]});
}
for(const e of elements){
 const name='Editable '+(++count)+' '+(e.text||e.type);
 if(e.type==='image'){slide.images.add({blob:new Uint8Array(imgs[e.index]),contentType:'image/png',position:{left:e.x,top:e.y,width:e.w,height:e.h},alt:e.index?'Post-event SAR, BRIGHT Fig. 6 Turkey':'Pre-event optical, BRIGHT Fig. 6 Turkey'});continue;}
 if(e.type==='text'){
  const t=slide.shapes.add({name,geometry:'textbox',position:{left:e.x,top:e.y-1,width:Math.min(1500-e.x,e.w+12),height:e.h+6},fill:'none',line:{fill:'none',width:0}});
  t.text=e.text;t.text.style={fontSize:e.size,typeface:'Arial',bold:e.bold,color:color(e.fill),wrap:'none',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};continue;
 }
 if(e.type==='path'){
  for(const part of e.parts)poly(part.pts,e.fill,e.stroke,e.sw,part.closed,name,e.dash!=='none');
  if(e.arrow){const pts=e.parts.at(-1).pts,q=pts.at(-1),a=pts.at(-2),ang=Math.atan2(q.y-a.y,q.x-a.x),l=10,w=4.5;
   poly([q,{x:q.x-l*Math.cos(ang)+w*Math.sin(ang),y:q.y-l*Math.sin(ang)-w*Math.cos(ang)},{x:q.x-l*Math.cos(ang)-w*Math.sin(ang),y:q.y-l*Math.sin(ang)+w*Math.cos(ang)}],e.stroke,'none',0,true,name+' arrow');
  }continue;
 }
 slide.shapes.add({name,geometry:e.type==='rect'?'rect':'ellipse',position:{left:e.x,top:e.y,width:e.w,height:e.h},fill:color(e.fill),line:{fill:color(e.stroke),width:e.sw}});
}
slide.speakerNotes.textFrame.setText('Editable reconstruction of BRIGHT_real_inputs_overview_v2.svg. Input illustrations reproduced from Chen et al., BRIGHT, ESSD 17, 6217–6253 (2025), Fig. 6, Turkey-EQ-2023. DOI:10.5194/essd-17-6217-2025. See Table 2 for imagery providers. Regions, memory examples and output map are schematic. Candidate inference retrieval architecture, not validated performance. Text, boxes and freeform paths are native editable objects. Arrows are freeform objects and do not automatically reroute. RGB/SAR are embedded raster images.');
const candidate=root+'/ppt_build/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
const png=await p.export({slide,format:'png',scale:1});await fs.writeFile(root+'/ppt_build/preview.png',new Uint8Array(await png.arrayBuffer()));
await fs.mkdir(root+'/ppt_output',{recursive:true});
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/ppt_output/BRIGHT_model_editable_v1.pptx',pythonExecutable:'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','14287500,9906000'],fontPolicy:{basis:'design',families:['Arial']},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/ppt_build/validation.json'});
console.log(JSON.stringify(result));
