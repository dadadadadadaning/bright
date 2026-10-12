import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const skillDir = process.env.SKILL_DIR;
const sourcePath = process.env.SOURCE_PPTX;
const outputDir = process.env.TMP_DIR;
if (!skillDir || !sourcePath || !outputDir) throw new Error("Missing SKILL_DIR, SOURCE_PPTX or TMP_DIR");

const { importRuntimeModule } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/runtime_helpers.mjs")).href
);
const { FileBlob, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,text,textPreview,bbox,bboxUnit,alt,isPlaceholder,placeholders",
  maxChars: 100000,
});
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(path.join(outputDir, "source-inspect.ndjson"), snapshot.ndjson, "utf8");

const slide = presentation.slides.getItem(0);
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(outputDir, "source-slide-1.layout.json"), await layout.text(), "utf8");

const summary = {
  slideCount: presentation.slides.count,
  slideSize: presentation.slideSize,
  masters: presentation.masters.items.map((item) => ({ id: item.id, name: item.name })),
  layouts: presentation.layouts.items.map((item) => ({ id: item.id, name: item.name })),
};
await fs.writeFile(path.join(outputDir, "source-summary.json"), JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify(summary, null, 2));
