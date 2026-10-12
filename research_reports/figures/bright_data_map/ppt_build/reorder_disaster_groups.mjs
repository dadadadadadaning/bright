import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import JSZip from "jszip";

const workspaceDir = "D:/software/dowmload/BRIGHT-master";
const skillDir = "C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations";
const sourcePath = "C:/Users/梁嘎嘎/Desktop/新图.pptx";
const buildDir = path.join(workspaceDir, "research_reports/figures/bright_data_map/ppt_build");
const stagingDir = path.join(workspaceDir, ".codex-finalizer-bright-map-order");
const candidatePath = path.join(buildDir, "new_map_disaster_order_candidate.pptx");
const finalPath = path.join(workspaceDir, "research_reports/figures/bright_data_map/新图_灾害顺序调整.pptx");
const runtimePython = "C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";

const desiredOrder = [
  "Earthquake",
  "Armed conflict",
  "Volcano eruption",
  "Wildfire",
  "Hurricane",
  "Flood",
  "Explosion",
];

function requireCondition(ok, message) {
  if (!ok) throw new Error(message);
}

function itemText(xml) {
  return [...xml.matchAll(/<a:t>([\s\S]*?)<\/a:t>/g)].map(match => match[1]).join(" ");
}

function firstOffset(xml) {
  const match = xml.match(/<a:off x="(-?\d+)" y="(-?\d+)"\s*\/>/);
  requireCondition(match, "A shape has no native position");
  return { x: Number(match[1]), y: Number(match[2]) };
}

function shiftFirstOffset(xml, dx, dy) {
  let found = false;
  const result = xml.replace(/<a:off x="(-?\d+)" y="(-?\d+)"\s*\/>/, (_, x, y) => {
    found = true;
    return `<a:off x="${Number(x) + dx}" y="${Number(y) + dy}"/>`;
  });
  requireCondition(found, "Unable to move a native shape");
  return result;
}

function reorderSlideMap(xml) {
  const nodes = [...xml.matchAll(/<p:(sp|pic)\b[\s\S]*?<\/p:\1>/g)];
  requireCondition(nodes.length === 107, `Unexpected slide-4 shape count: ${nodes.length}`);
  const originals = nodes.map(node => node[0]);
  const replacements = [...originals];

  // Seven top groups; each has a coloured star, heading, two photos and two frames.
  const topStart = 1;
  const topLength = 6;
  const topNames = desiredOrder.map((_, slot) => itemText(originals[topStart + slot * topLength + 1]));
  requireCondition(new Set(topNames).size === 7, `Unexpected top categories: ${topNames}`);
  const topX = topNames.map((_, slot) => firstOffset(originals[topStart + slot * topLength]).x);
  for (let newSlot = 0; newSlot < desiredOrder.length; newSlot++) {
    const oldSlot = topNames.indexOf(desiredOrder[newSlot]);
    requireCondition(oldSlot >= 0, `Missing top group: ${desiredOrder[newSlot]}`);
    const dx = topX[newSlot] - topX[oldSlot];
    for (let j = 0; j < topLength; j++) {
      replacements[topStart + newSlot * topLength + j] = shiftFirstOffset(
        originals[topStart + oldSlot * topLength + j], dx, 0,
      );
    }
  }

  // Seven left-bottom legend rows; each is a star and an editable text label.
  const legendStart = 93;
  const legendNames = desiredOrder.map((_, slot) => itemText(originals[legendStart + slot * 2 + 1]));
  requireCondition(new Set(legendNames).size === 7, `Unexpected legend categories: ${legendNames}`);
  const legendY = legendNames.map((_, slot) => firstOffset(originals[legendStart + slot * 2]).y);
  for (let newSlot = 0; newSlot < desiredOrder.length; newSlot++) {
    const oldSlot = legendNames.indexOf(desiredOrder[newSlot]);
    requireCondition(oldSlot >= 0, `Missing legend row: ${desiredOrder[newSlot]}`);
    const dy = legendY[newSlot] - legendY[oldSlot];
    for (let j = 0; j < 2; j++) {
      replacements[legendStart + newSlot * 2 + j] = shiftFirstOffset(
        originals[legendStart + oldSlot * 2 + j], 0, dy,
      );
    }
  }

  let result = "";
  let lastEnd = 0;
  nodes.forEach((node, index) => {
    result += xml.slice(lastEnd, node.index) + replacements[index];
    lastEnd = node.index + node[0].length;
  });
  result += xml.slice(lastEnd);

  const finalNodes = [...result.matchAll(/<p:(sp|pic)\b[\s\S]*?<\/p:\1>/g)].map(match => match[0]);
  requireCondition(finalNodes.length === 107, "Changed slide shape count");
  const finalTop = desiredOrder.map((_, slot) => itemText(finalNodes[topStart + slot * topLength + 1]));
  const finalLegend = desiredOrder.map((_, slot) => itemText(finalNodes[legendStart + slot * 2 + 1]));
  requireCondition(JSON.stringify(finalTop) === JSON.stringify(desiredOrder), "Top group order did not match");
  requireCondition(JSON.stringify(finalLegend) === JSON.stringify(desiredOrder), "Legend order did not match");

  return { xml: result, beforeTop: topNames, beforeLegend: legendNames, after: desiredOrder };
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
const sourceBytes = await fs.readFile(sourcePath);
const archive = await JSZip.loadAsync(sourceBytes);
const slideEntry = archive.file("ppt/slides/slide4.xml");
requireCondition(slideEntry, "Source slide 4 is missing");
const slideXml = await slideEntry.async("string");
const change = reorderSlideMap(slideXml);
archive.file("ppt/slides/slide4.xml", change.xml);
const candidate = await archive.generateAsync({ type: "nodebuffer", compression: "DEFLATE", compressionOptions: { level: 6 } });
await fs.writeFile(candidatePath, candidate);

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "16002000,8382000"],
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "new_map_disaster_order.validation.json"),
});

console.log(JSON.stringify({ finalPath, ...change, xml: undefined }));
