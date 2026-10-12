import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations/container_tools/artifact_tool_utils.mjs';
const work='D:/software/dowmload/BRIGHT-master/research_reports/merged_figures', skill='C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations',runtime='C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';process.env.RUNTIME_NODE=runtime+'/node/bin/node.exe';
await fs.mkdir(work+'/output',{recursive:true});
let p;try{p=await PresentationFile.importPptx(await FileBlob.load(work+'/build/candidate.pptx'));}catch(e){console.log(String(e));process.exit(1);}
console.log('imported',p.slides.items.length);
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:.8});await fs.writeFile(work+'/build/slide'+(i+1)+'.png',new Uint8Array(await b.arrayBuffer()));console.log('rendered',i+1);
}
await finalizePresentation({workspaceDir:work,candidatePath:work+'/build/candidate.pptx',finalPath:work+'/output/BRIGHT_combined_figures_v1.pptx',pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','15240000,10287000'],explicitTotalSlideCount:7,verifyArtifactToolImport:true,receiptPath:work+'/build/validation.json'});
console.log('DONE');
