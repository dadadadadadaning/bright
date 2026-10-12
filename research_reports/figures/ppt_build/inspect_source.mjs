import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const dir='D:/software/dowmload/BRIGHT-master/research_reports/figures/ppt_build';
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'/desktop_source.pptx'));
console.log((await p.inspect({kind:'slide,textbox',maxChars:16000})).ndjson);
const s=p.slides.items[0];
await fs.writeFile(dir+'/desktop_preview.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(dir+'/source_shapes.json',JSON.stringify(s.shapes.items.map(v=>({id:v.id,name:v.name,pos:v.position,fill:v.fill,line:v.line,text:v.text.toString()})),null,2));
