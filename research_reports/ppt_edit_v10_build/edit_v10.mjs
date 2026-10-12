import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const skillDir = process.env.SKILL_DIR;
const workspaceDir = process.env.WORKSPACE_DIR;
const sourcePath = process.env.SOURCE_PPTX;
const tmpDir = process.env.TMP_DIR;
const finalPath = process.env.FINAL_PPTX;
const runtimePython = process.env.RUNTIME_PYTHON;
if (![skillDir, workspaceDir, sourcePath, tmpDir, finalPath, runtimePython].every(Boolean)) {
  throw new Error("Missing required environment paths");
}

const { importRuntimeModule } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/runtime_helpers.mjs")).href
);
const { FileBlob, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href
);

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
if (presentation.slides.count !== 1) throw new Error(`Expected one slide, found ${presentation.slides.count}`);
const slide = presentation.slides.getItem(0);

const NAVY = "#25313F";
const WHITE = "#FFFFFF";
const PINK = "#F5C9C9";
const BLUE = "#B6C8E4";
const YELLOW = "#FFD982";
const GREEN = "#C8DDB0";

function resolve(id) {
  const item = presentation.resolve(id);
  if (!item) throw new Error(`Could not resolve ${id}`);
  return item;
}

function setText(id, text, options = {}) {
  const shape = resolve(id);
  shape.text = text;
  shape.text.style = {
    typeface: options.typeface ?? "Arial",
    fontSize: options.fontSize ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? NAVY,
    alignment: options.alignment ?? "center",
    verticalAlignment: options.verticalAlignment ?? "middle",
    wrap: options.wrap ?? "square",
    autoFit: options.autoFit ?? "shrinkText",
    insets: options.insets ?? { top: 0, right: 2, bottom: 0, left: 2 },
  };
  if (options.position) shape.position = options.position;
  return shape;
}

function clearText(ids) {
  for (const id of ids) resolve(id).text = "";
}

function addText(name, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: options.typeface ?? "Arial",
    fontSize: options.fontSize ?? 17,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    color: options.color ?? NAVY,
    alignment: options.alignment ?? "center",
    verticalAlignment: options.verticalAlignment ?? "middle",
    wrap: options.wrap ?? "square",
    autoFit: options.autoFit ?? "shrinkText",
    insets: options.insets ?? { top: 0, right: 2, bottom: 0, left: 2 },
  };
  return shape;
}

function addStep(number, left, top) {
  const badge = slide.shapes.add({
    geometry: "ellipse",
    name: `Workflow step ${number}`,
    position: { left, top, width: 24, height: 24 },
    fill: NAVY,
    line: { style: "solid", fill: WHITE, width: 1.2 },
  });
  badge.text = String(number);
  badge.text.style = {
    typeface: "Arial",
    fontSize: 15,
    bold: true,
    color: WHITE,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "none",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  badge.bringToFront();
}

// Make the main residual route explicit.
setText("sh/32dg7u10", "Preserved pixel pathway  F_A", {
  fontSize: 17,
  position: { left: 875, top: 27, width: 235, height: 28 },
});

// Current-building query construction and building-level retrieval.
setText("sh/4bipkzix", "Building pooling + key projection", {
  fontSize: 17,
  position: { left: 541, top: 360, width: 253, height: 38 },
});
setText("sh/za5gzed4", "Shared feature + region\nencoder snapshot", {
  fontSize: 17,
  position: { left: 189, top: 427, width: 185, height: 61 },
});
setText("sh/658v690b", "Train only: k_j, V_j, y_j, source_j", {
  fontSize: 11,
  position: { left: 423, top: 468, width: 210, height: 20 },
});
setText("sh/t03q1oju", "Top-K buildings", {
  fontSize: 15,
  position: { left: 802, top: 431, width: 99, height: 27 },
});
setText("sh/szup8j29", "valid source + dedup", {
  fontSize: 14,
  position: { left: 716, top: 463, width: 183, height: 24 },
});

// Replace the fragmented support-set formula with one readable label.
clearText([
  "sh/o7214bul", "sh/36t0b6tg", "sh/25kj2lcv", "sh/94f2pkbq",
  "sh/8361gfu5", "sh/wva10bqp", "sh/hwjitg7u", "sh/utsjyl8j",
]);
addText(
  "Retrieved support record",
  "{V_j, y_j, s_j, m_j}",
  { left: 1025, top: 486, width: 185, height: 28 },
  { fontSize: 16, typeface: "Cambria Math" },
);

// Panel b: explicitly distinguish token update from the pixel residual.
clearText(["sh/vidgnmxw", "sh/gjmxgreh", "sh/hkvypwf2", "sh/il4fi1wn", "sh/3mdgrmx8"]);

// Panel c: labels are building-level annotations; support weights use retrieval
// similarity and the valid-support mask.
setText("sh/do321072", "Building-label embedding", {
  fontSize: 15,
  position: { left: 783, top: 838, width: 173, height: 25 },
});
setText("sh/kveps365", "Score s_j + valid mask m_j", {
  fontSize: 14,
  position: { left: 1240, top: 808, width: 184, height: 27 },
});
resolve("sh/lwnql8nq").text = "";
setText("sh/z6povmp8", "Support aggregation\nα_j = MaskedSoftmax(s_j, m_j)", {
  fontSize: 14,
  position: { left: 1036, top: 843, width: 273, height: 47 },
});

clearText(["sh/ip072xoj", "sh/3q98v254"]);
addText(
  "Support token aggregation formula",
  "ΔT_mem = Σ_j α_j U_j",
  { left: 1031, top: 893, width: 173, height: 28 },
  { fontSize: 14, typeface: "Cambria Math" },
);

clearText(["sh/budkfqtc", "sh/qtkj6lcr", "sh/p8bidgb6"]);
addText(
  "No support fallback",
  "No valid support: ΔT_mem = ΔF_mem = 0",
  { left: 805, top: 916, width: 285, height: 28 },
  { fontSize: 13, typeface: "Cambria Math" },
);

clearText([
  "sh/ojyl0zap", "sh/fu54nu98", "sh/etw3upsn", "sh/1gnmp4re",
  "sh/0velwzat", "sh/zq94ze9c", "sh/ypgn6t87",
]);
addText(
  "Memory pixel update formula",
  "ΔF_mem = Q(ΔT_mem W_mem)",
  { left: 1192, top: 916, width: 230, height: 32 },
  { fontSize: 14, typeface: "Cambria Math" },
);

// Clarify the colour semantics and the provisional status of the architecture.
setText(
  "sh/n6dsjqhc",
  "Pink: current query    Blue: retrieved support    Yellow: association / retrieval    Green: projection / prediction    Gray: reused backbone",
  { fontSize: 15, alignment: "left", position: { left: 25, top: 978, width: 1450, height: 26 } },
);
setText(
  "sh/o7m9cvyx",
  "*Candidate architecture. Memory contains training entries only; region links and the example output are schematic.",
  { fontSize: 14, alignment: "left", position: { left: 25, top: 1007, width: 1450, height: 25 } },
);

// Five numbered stages make the main workflow readable without adding prose.
addStep(1, 210, 82);
addStep(2, 762, 82);
addStep(3, 885, 415);
addStep(4, 1080, 82);
addStep(5, 1442, 88);

slide.speakerNotes.textFrame.setText([
  "流程说明：",
  "1. DamageFormer分别编码灾前RGB与灾后SAR，输出像素特征F、结构特征R和建筑概率P。",
  "2. P限定预测建筑，R计算建筑内部软关联Q；由F和Q形成区域token T，经区域交互得到ΔT_local，并回填为像素增量ΔF_local。",
  "3. 将更新后的区域token T_A汇总为建筑级查询key k_b，在仅由训练样本构建的Memory中执行合法Top-K检索，同时返回相似度s_j和有效掩码m_j。",
  "4. 当前区域token逐栋读取支持建筑的局部token V_j；利用s_j、m_j和建筑级标签编码聚合读取结果，得到ΔT_mem。无合法支持时令ΔT_mem和ΔF_mem为零。",
  "5. 将Memory增量回填至像素，并与保留的当前像素路径F_A相加，由像素分类头输出四分类损毁图。",
  "注意：图中标签y_j是支持建筑的训练集建筑级标签，不是区域级损毁真值。source_j只用于合法性筛选。",
].join("\n"));

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const stagingDir = path.join(tmpDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "reference",
  families: ["Arial", "Cambria Math"],
  referencePath: sourcePath,
  referenceSha256: "bee0fea679efad320388837aff7d6d27a5b850ac5f1dcdfcc4e50aac2a51bf09",
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "14287500,9906000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "BRIGHT_model_formulas_v10_release.validation.json"),
});

const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(path.join(tmpDir, "edited-slide-1.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(tmpDir, "edited-slide-1.layout.json"), await layout.text(), "utf8");
console.log(JSON.stringify({ finalPath, result }, null, 2));
