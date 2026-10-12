import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/梁嘎嘎/Desktop/BRIGHT_model_MoAT_palette_v4.pptx'));
const s=p.slides.items[0];
console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(s.shapes)));console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(s.shapes.items[0])));
await fs.writeFile('D:/software/dowmload/BRIGHT-master/research_reports/figures/ppt_build/v4_source.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
console.log((await p.inspect({kind:'slide',maxChars:500})).ndjson);
