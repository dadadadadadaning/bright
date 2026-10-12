import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "D:/software/dowmload/BRIGHT-master";
const SKILL_DIR = "C:/Users/梁嘎嘎/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations";
const buildDir = path.join(workspaceDir, "research_reports/figures/bright_data_map/ppt_build");
const outputDir = path.join(workspaceDir, "research_reports/figures/bright_data_map");
const FINAL_PPTX = path.join(outputDir, "BRIGHT_map_with_disaster_examples_locations_corrected.pptx");
const baseMapPath = path.join(outputDir, "BRIGHT_world_base_no_country_borders_for_ppt.png");
const RUNTIME_PYTHON = "C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";

const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const family = resolvePresentationFont();

const presentation = Presentation.create({ slideSize: { width: 1280, height: 840 } });
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

const MAP_SHIFT_Y = 145;
const mapFrame = { left: 24, top: 26 + MAP_SHIFT_Y, width: 1232, height: 574 };
const baseBytes = await fs.readFile(baseMapPath);
slide.images.add({
  blob: baseBytes,
  contentType: "image/png",
  alt: "World basemap with country boundaries and coordinate grid",
  fit: "contain",
  position: mapFrame,
});

const colors = {
  "Earthquake": "#E69F00",
  "Flood": "#0072B2",
  "Hurricane": "#56B4E9",
  "Volcano eruption": "#D55E00",
  "Wildfire": "#009E73",
  "Explosion": "#CC79A7",
  "Armed conflict": "#6A3D9A",
};

const exampleGroups = [
  { kind: "Earthquake", files: [
    "research_reports/test_comparison_ppt/build/assets/turkey-earthquake_00001074_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/noto-earthquake_00000049_RGB.png",
  ] },
  { kind: "Flood", files: [
    "research_reports/test_comparison_ppt/build/assets/libya-flood_00000021_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/libya-flood_00000021_SAR.png",
  ] },
  { kind: "Hurricane", files: [
    "research_reports/test_comparison_ppt/build/assets/myanmar-hurricane_00000107_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/mexico-hurricane_00000005_RGB.png",
  ] },
  { kind: "Volcano eruption", files: [
    "research_reports/test_comparison_ppt/build/assets/la_palma-volcano_00000597_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/congo-volcano_00000092_RGB.png",
  ] },
  { kind: "Wildfire", files: [
    "research_reports/test_comparison_ppt/build/assets/hawaii-wildfire_00000014_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/hawaii-wildfire_00000003_RGB.png",
  ] },
  { kind: "Explosion", files: [
    "research_reports/test_comparison_ppt/build/assets/bata-explosion_00000062_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/bata-explosion_00000049_RGB.png",
  ] },
  { kind: "Armed conflict", files: [
    "research_reports/test_comparison_ppt/build/assets/ukraine-conflict_00000354_RGB.png",
    "research_reports/test_comparison_ppt/build/assets/ukraine-conflict_00000315_RGB.png",
  ] },
];

const events = [
  { place: "Colorado, USA", detail: "Dec 30, 2021  |  WF  |  n=77", lon: -105.17, lat: 39.96, kind: "Wildfire", label: { x: 255, y: 151, w: 170, h: 41 } },
  { place: "Maui, Hawaii, USA", detail: "Aug 08, 2023  |  WF  |  n=65", lon: -156.33, lat: 20.80, kind: "Wildfire", label: { x: 83, y: 300, w: 190, h: 41 } },
  { place: "Acapulco, Mexico", detail: "Oct 25, 2023  |  HU  |  n=212", lon: -99.91, lat: 16.85, kind: "Hurricane", label: { x: 250, y: 283, w: 190, h: 41 } },
  { place: "Les Cayes, Haiti", detail: "Aug 14, 2021  |  EQ  |  n=73", lon: -73.75, lat: 18.20, kind: "Earthquake", label: { x: 397, y: 279, w: 175, h: 41 } },
  { place: "La Palma, Spain", detail: "Sep 19, 2021  |  VO  |  n=933", lon: -17.86, lat: 28.68, kind: "Volcano eruption", label: { x: 525, y: 269, w: 175, h: 41 } },
  { place: "Morocco", detail: "Sep 08, 2023  |  EQ  |  n=567", lon: -8.41, lat: 31.07, kind: "Earthquake", label: { x: 600, y: 191, w: 182, h: 41 } },
  { place: "Bata, Equatorial Guinea", detail: "Mar 07, 2021  |  EX  |  n=107", lon: 9.77, lat: 1.86, kind: "Explosion", label: { x: 594, y: 358, w: 190, h: 41 } },
  { place: "Goma, DR Congo", detail: "May 22, 2021  |  VO  |  n=123", lon: 29.22, lat: -1.66, kind: "Volcano eruption", label: { x: 750, y: 371, w: 175, h: 41 } },
  { place: "Derna, Libya", detail: "Sep 11, 2023  |  FL  |  n=124", lon: 22.64, lat: 32.77, kind: "Flood", label: { x: 660, y: 245, w: 170, h: 41 } },
  { place: "Beirut, Lebanon", detail: "Aug 04, 2020  |  EX  |  n=133", lon: 35.50, lat: 33.89, kind: "Explosion", label: { x: 777, y: 250, w: 185, h: 41 } },
  { place: "Türkiye", detail: "Feb 06, 2023  |  EQ  |  n=1,114", lon: 37.03, lat: 37.17, kind: "Earthquake", label: { x: 781, y: 193, w: 185, h: 41 } },
  { place: "Ukraine", detail: "2022  |  AC  |  n=513", lon: 37.80, lat: 48.40, kind: "Armed conflict", label: { x: 782, y: 141, w: 150, h: 41 } },
  { place: "Myanmar", detail: "May 14, 2023  |  HU  |  n=126", lon: 92.90, lat: 20.15, kind: "Hurricane", label: { x: 975, y: 277, w: 170, h: 41 } },
  { place: "Noto, Japan", detail: "Jan 01, 2024  |  EQ  |  n=79", lon: 137.30, lat: 37.50, kind: "Earthquake", label: { x: 1100, y: 202, w: 155, h: 41 } },
];

// Calibrated to the actual black axes frame inside the contained basemap image.
// This preserves the basemap/layout while placing every event at its true lon/lat.
const mapPlot = { left: 25.1, top: 56.4 + MAP_SHIFT_Y, width: 1229.9, height: 513.2 };
function project(lon, lat) {
  return {
    x: mapPlot.left + ((lon + 180) / 360) * mapPlot.width,
    y: mapPlot.top + ((85 - lat) / 145) * mapPlot.height,
  };
}

function addTextBox(text, position, options = {}) {
  const box = slide.shapes.add({
    geometry: options.geometry ?? "textbox",
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 12,
    bold: options.bold ?? false,
    color: options.color ?? "#20252A",
    autoFit: "shrinkText",
    verticalAlignment: options.verticalAlignment ?? "middle",
    alignment: options.alignment ?? "left",
  };
  return box;
}

// Two authentic BRIGHT RGB examples for each disaster type. Every thumbnail
// remains an independent, replaceable image object in PowerPoint.
const groupStartX = 36;
const groupWidth = 172;
for (let i = 0; i < exampleGroups.length; i += 1) {
  const group = exampleGroups[i];
  const x = groupStartX + i * groupWidth;
  slide.shapes.add({
    geometry: "star5",
    position: { left: x, top: 15, width: 15, height: 15 },
    fill: colors[group.kind],
    line: { fill: "#FFFFFF", width: 0.6 },
  });
  addTextBox(group.kind, { left: x + 19, top: 11, width: 143, height: 23 }, {
    fontSize: 11.2,
    bold: true,
  });
  for (let j = 0; j < group.files.length; j += 1) {
    const imagePath = path.join(workspaceDir, group.files[j]);
    const bytes = await fs.readFile(imagePath);
    const frame = { left: x + j * 82, top: 37, width: 78, height: 94 };
    slide.images.add({
      blob: bytes,
      contentType: "image/png",
      alt: `${group.kind} BRIGHT RGB example ${j + 1}`,
      fit: "cover",
      position: frame,
    });
    slide.shapes.add({
      geometry: "rect",
      position: frame,
      fill: "none",
      line: { fill: colors[group.kind], width: 1.25 },
    });
  }
}
slide.shapes.add({
  geometry: "line",
  position: { left: 30, top: 145, width: 1220, height: 1 },
  fill: "none",
  line: { fill: "#BFC6C9", width: 0.8 },
});

// Native coordinate labels.
for (let lon = -180; lon <= 180; lon += 30) {
  const p = project(lon, 85);
  const txt = lon === 0 ? "0°" : `${Math.abs(lon)}° ${lon < 0 ? "W" : "E"}`;
  addTextBox(txt, { left: p.x - 25, top: 4 + MAP_SHIFT_Y, width: 50, height: 20 }, { fontSize: 9.5, alignment: "center", color: "#2D3338" });
}
for (let lat = -60; lat <= 90; lat += 30) {
  const clamped = Math.min(lat, 85);
  const p = project(180, clamped);
  const txt = lat === 0 ? "0°" : `${Math.abs(lat)}° ${lat < 0 ? "S" : "N"}`;
  addTextBox(txt, { left: 1239, top: p.y - 9, width: 39, height: 18 }, { fontSize: 9.5, alignment: "center", color: "#2D3338" });
}

// Native, independently editable stars, connector lines and labels.
for (const event of events) {
  const p = project(event.lon, event.lat);
  const labelCenter = { x: event.label.x + event.label.w / 2, y: event.label.y + MAP_SHIFT_Y + event.label.h / 2 };
  const labelPosition = {
    left: event.label.x,
    top: event.label.y + MAP_SHIFT_Y,
    width: event.label.w,
    height: event.label.h,
  };
  slide.shapes.add({
    geometry: "line",
    position: {
      left: Math.min(p.x, labelCenter.x),
      top: Math.min(p.y, labelCenter.y),
      width: Math.max(1, Math.abs(labelCenter.x - p.x)),
      height: Math.max(1, Math.abs(labelCenter.y - p.y)),
    },
    fill: "none",
    line: { fill: "#687078", width: 0.7 },
  });
  slide.shapes.add({
    geometry: "star5",
    position: { left: p.x - 8.5, top: p.y - 8.5, width: 17, height: 17 },
    fill: colors[event.kind],
    line: { fill: "#FFFFFF", width: 1 },
  });
  slide.shapes.add({
    geometry: "rect",
    position: labelPosition,
    fill: "#FFFFFF",
    line: { fill: "#D7DDE0", width: 0.35 },
  });
  addTextBox(`${event.place}\n${event.detail.split("  |  ")[0]}`, labelPosition, {
    geometry: "textbox",
    fill: "none",
    line: { fill: "none", width: 0 },
    fontSize: 11.2,
    bold: true,
  });
}

// Editable disaster-type legend.
const legendX = 52;
const legendY = 423 + MAP_SHIFT_Y;
slide.shapes.add({
  geometry: "rect",
  position: { left: legendX - 13, top: legendY - 16, width: 154, height: 169 },
  fill: { color: "#FFFFFF", transparency: 4 },
  line: { fill: "#4B5157", width: 1 },
});
addTextBox("Disaster type", { left: legendX, top: legendY - 11, width: 120, height: 22 }, { fontSize: 11.5, bold: true });
let legendRow = 0;
for (const [kind, color] of Object.entries(colors)) {
  const y = legendY + 15 + legendRow * 18.2;
  slide.shapes.add({
    geometry: "star5",
    position: { left: legendX, top: y + 1, width: 12, height: 12 },
    fill: color,
    line: { fill: "#FFFFFF", width: 0.6 },
  });
  addTextBox(kind, { left: legendX + 18, top: y - 1, width: 112, height: 17 }, { fontSize: 10.5 });
  legendRow += 1;
}

// Editable symbol key.
slide.shapes.add({
  geometry: "rect",
  position: { left: 1080, top: 523 + MAP_SHIFT_Y, width: 145, height: 58 },
  fill: { color: "#FFFFFF", transparency: 4 },
  line: { fill: "#4B5157", width: 1 },
});
addTextBox("Map symbol", { left: 1093, top: 528 + MAP_SHIFT_Y, width: 116, height: 19 }, { fontSize: 11.5, bold: true });
slide.shapes.add({
  geometry: "star5",
  position: { left: 1093, top: 551 + MAP_SHIFT_Y, width: 15, height: 15 },
  fill: "#333333",
  line: { fill: "#FFFFFF", width: 0.7 },
});
addTextBox("Event location", { left: 1113, top: 547 + MAP_SHIFT_Y, width: 103, height: 22 }, { fontSize: 10.5 });

addTextBox(
  "BRIGHT standard split: 14 disaster events  |  4,246 paired samples  |  all locations use star symbols",
  { left: 155, top: 622 + MAP_SHIFT_Y, width: 970, height: 25 },
  { fontSize: 12, alignment: "center", color: "#454D54" },
);
addTextBox(
  "Representative event or city coordinates are shown. Points do not denote every image tile footprint.",
  { left: 175, top: 650 + MAP_SHIFT_Y, width: 930, height: 23 },
  { fontSize: 10.5, alignment: "center", color: "#69737B" },
);

slide.speakerNotes.textFrame.setText(
  "Event labels and counts come from the local BRIGHT standard split files. " +
  "Coordinates are representative event or city points. The basemap uses Natural Earth 1:110m land silhouettes without country borders. " +
  "All markers use the same five-point star; colour identifies disaster type. The top strip contains two local BRIGHT RGB examples per disaster type. " +
  exampleGroups.flatMap(group => group.files).join("; "),
);

const stagingDir = path.join(workspaceDir, ".codex-finalizer-bright-map");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate_map_with_examples.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const preview = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(buildDir, "slide-1-locations-corrected.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(buildDir, "slide-1.layout.json"), await layout.text());

await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,8001000",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "BRIGHT_map_locations_corrected.validation.json"),
});

console.log(JSON.stringify({ final: FINAL_PPTX, font: family, preview: path.join(buildDir, "slide-1-locations-corrected.png") }));
