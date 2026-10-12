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
const sharpModule = await importRuntimeModule("sharp");
const sharp = sharpModule.default ?? sharpModule;
const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href
);

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
if (presentation.slides.count !== 1) throw new Error(`Expected one slide, found ${presentation.slides.count}`);
const slide = presentation.slides.getItem(0);

const NAVY = "#25313F";
const WHITE = "#FFFFFF";
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
  return shape;
}

function clearText(ids) {
  for (const id of ids) {
    const shape = resolve(id);
    if (shape.text) shape.text = "";
  }
}

function addCover(name, position, fill) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { fill: "none", width: 0 },
  });
  shape.name = name;
  return shape;
}

function formulaSvg(width, height, lines, fontSize = 16) {
  const lineGap = height / lines.length;
  const body = lines.map((line, i) => {
    const y = lineGap * (i + 0.68);
    return `<text x="${width / 2}" y="${y}" text-anchor="middle" font-family="Cambria Math" font-size="${line.fontSize ?? fontSize}" fill="${NAVY}" ${line.bold ? 'font-weight="700"' : ''}>${line.markup}</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
}

function v(text) { return `<tspan font-style="italic">${text}</tspan>`; }
function roman(text) { return `<tspan font-style="normal">${text}</tspan>`; }
function sub(text) { return `<tspan baseline-shift="sub" font-size="70%">${text}</tspan>`; }
function sup(text) { return `<tspan baseline-shift="super" font-size="70%">${text}</tspan>`; }

async function addFormula(name, position, lines, options = {}) {
  const width = Math.max(1, Math.round(position.width * 2));
  const height = Math.max(1, Math.round(position.height * 2));
  const svg = formulaSvg(width, height, lines, (options.fontSize ?? 15) * 2);
  const png = await sharp(Buffer.from(svg, "utf8"), { density: 192 }).png().toBuffer();
  const image = slide.images.add({
    blob: new Uint8Array(png),
    contentType: "image/png",
    alt: options.alt ?? name,
    fit: "contain",
    position,
  });
  image.name = name;
  return image;
}

// Remove draft-style underscore notation before placing typeset equations.
clearText([
  "sh/uh4fehwb", // Pool formula text
  "sh/z6povmp8", // Support aggregation text
  "sh/1kzil0z6", // Memory token aggregation
  "sh/0jq1cvy1", // Empty-support fallback
  "sh/zih0jqhg", // Memory unpool formula
  "sh/jqtgf2xo", // Retrieved support tuple
  "sh/lwzahs7q", // Query label above support block
  "sh/kbqtonq5", // K,V label above support block
  "sh/7654je1w", "sh/k3uloji5", "sh/l43mh4jq", "sh/y1c3m90z", "sh/j2l4fe1k", // Final output equation
]);

setText("sh/658v690b", "Train-only records", { fontSize: 11 });
setText("sh/kveps365", "Retrieval score and valid mask", { fontSize: 13 });
setText("sh/cbmd0jml", "Shared cross-attention parameters (3 of K supports shown)", { fontSize: 15 });
const pixelPathLabel = setText("sh/32dg7u10", "Preserved pixel pathway", { fontSize: 17, alignment: "right" });
pixelPathLabel.position = { left: 875, top: 27, width: 190, height: 28 };
clearText(["sh/o3mh0z2l"]);
await addFormula(
  "Preserved pixel feature",
  { left: 1068, top: 27, width: 38, height: 28 },
  [{ markup: `${v("F")}${sub("A")}` }],
  { fontSize: 16, alt: "Preserved pixel pathway feature F sub A" },
);

// Building-constrained normalized pooling:
// Q is N x M, D_Q normalizes the mass assigned to each region.
addCover("Pool formula cover", { left: 187, top: 740, width: 152, height: 33 }, YELLOW);
await addFormula(
  "Normalized region pooling",
  { left: 181, top: 738, width: 164, height: 36 },
  [
    { markup: `${v("D")}${sub("Q")} = ${roman("diag")}${v("(")}${v("Q")}${sup("⊤")}<tspan font-weight="700">1</tspan>${sub("N")}${v(")")}` },
    { markup: `${v("T")} = ${v("D")}${sub("Q")}${sup("−1")}${v("Q")}${sup("⊤")}${v("F")}` },
  ],
  { fontSize: 12, alt: "Normalized soft-region pooling equations" },
);

// Pixel-domain residual returned by the local regional interaction.
addCover("Local unpool formula cover", { left: 292, top: 873, width: 163, height: 29 }, GREEN);
await addFormula(
  "Local pixel residual",
  { left: 291, top: 873, width: 165, height: 28 },
  [{ markup: `Δ${v("F")}${sub("loc")} = ${v("Q")} Δ${v("T")}${sub("loc")} ${v("W")}${sub("loc")}` }],
  { fontSize: 12, alt: "Local regional residual unpooled to pixels" },
);

// The query reads each support building separately. V_j supplies both keys and values.
addCover("Cross attention equation cover", { left: 975, top: 588, width: 430, height: 24 }, WHITE);
await addFormula(
  "Per-support cross attention",
  { left: 982, top: 587, width: 416, height: 25 },
  [{ markup: `${v("U")}${sub("j")} = ${roman("CrossAttn")}${v("(")}${v("T")}${sub("A")}, ${v("V")}${sub("j")}, ${v("V")}${sub("j")}${v(")")},   ${v("j")} = 1,…,${v("K")}` }],
  { fontSize: 13, alt: "Per-building cross-attention readout" },
);

// Query-specific support records returned after building-level retrieval.
addCover("Retrieved support set cover", { left: 1015, top: 483, width: 215, height: 28 }, WHITE);
await addFormula(
  "Retrieved support set",
  { left: 1016, top: 482, width: 213, height: 29 },
  [{ markup: `{${v("V")}${sub("j")}, ${v("y")}${sub("j")}, ${v("s")}${sub("j")}, ${v("m")}${sub("j")}}${sub("j∈𝒩ₖ(b)")}` }],
  { fontSize: 13, alt: "Retrieved support records" },
);

// A support's building label is broadcast to its readout before masked aggregation.
addCover("Support aggregation formula cover", { left: 1036, top: 845, width: 272, height: 42 }, YELLOW);
await addFormula(
  "Label-conditioned support aggregation",
  { left: 1038, top: 844, width: 268, height: 44 },
  [
    { markup: `${v("Ũ")}${sub("j")} = [${v("U")}${sub("j")} ∥ <tspan font-weight="700">1</tspan>${sub("M")}${v("e")}${v("(")}${v("y")}${sub("j")}${v(")")}${sup("⊤")} ] ${v("W")}${sub("u")}` },
    { markup: `${v("α")}${sub("j")} = ${v("m")}${sub("j")} ${roman("exp")}${v("(")}${v("s")}${sub("j")}/${v("τ")}${v(")")} / Σ${sub("ℓ")}${v("m")}${sub("ℓ")} ${roman("exp")}${v("(")}${v("s")}${sub("ℓ")}/${v("τ")}${v(")")}` },
  ],
  { fontSize: 11.5, alt: "Label-conditioned support representation and masked weights" },
);

addCover("Memory token aggregation cover", { left: 1018, top: 891, width: 216, height: 25 }, WHITE);
await addFormula(
  "Memory token residual",
  { left: 1021, top: 891, width: 210, height: 25 },
  [{ markup: `Δ${v("T")}${sub("mem")} = Σ${sub("j∈𝒩ₖ(b)")} ${v("α")}${sub("j")} ${v("Ũ")}${sub("j")}` }],
  { fontSize: 12.5, alt: "Aggregated memory token residual" },
);

addCover("No support fallback cover", { left: 796, top: 914, width: 302, height: 25 }, WHITE);
await addFormula(
  "No valid support fallback",
  { left: 798, top: 914, width: 298, height: 25 },
  [{ markup: `Σ${sub("j")} ${v("m")}${sub("j")} = 0 ⇒ Δ${v("T")}${sub("mem")} = Δ${v("F")}${sub("mem")} = 0` }],
  { fontSize: 11.5, alt: "Zero residual when no valid support exists" },
);

addCover("Memory unpool formula cover", { left: 1190, top: 918, width: 242, height: 29 }, GREEN);
await addFormula(
  "Memory pixel residual",
  { left: 1191, top: 918, width: 240, height: 29 },
  [{ markup: `Δ${v("F")}${sub("mem")} = ${v("Q")} Δ${v("T")}${sub("mem")} ${v("W")}${sub("mem")}` }],
  { fontSize: 13, alt: "Memory residual unpooled to pixels" },
);

// The classifier produces probabilities; the displayed label map uses an argmax.
addCover("Final prediction equation cover", { left: 1268, top: 438, width: 210, height: 51 }, WHITE);
await addFormula(
  "Final prediction equations",
  { left: 1268, top: 439, width: 210, height: 49 },
  [
    { markup: `${v("P̂")} = ${roman("softmax")}${v("(")}${v("h")}${v("(")}${v("F")}${sub("A")} + Δ${v("F")}${sub("mem")}${v(")")}${v(")")}` },
    { markup: `${v("Ŷ")}${sub("n")} = ${roman("arg max")}${sub("c")} ${v("P̂")}${sub("n,c")}` },
  ],
  { fontSize: 14, alt: "Four-class probability and label-map equations" },
);

slide.speakerNotes.textFrame.setText([
  "流程与公式说明：",
  "1. DamageFormer输出像素特征 F∈R^{N×C}、结构特征 R 和建筑概率 P。对每栋预测建筑，Q∈[0,1]^{N×M}为像素到M个区域的软关联矩阵，并满足每个有效像素的关联权重和为1。",
  "2. 为避免区域大小改变token幅值，先令 D_Q=diag(Q^T 1_N)，再计算 T=D_Q^{-1}Q^T F。数值实现时应在D_Q对角线上加入ε，防止空区域导致除零。",
  "3. 区域交互产生增量 ΔT_loc，得到 T_A=T+ΔT_loc。局部增量通过 ΔF_loc=QΔT_locW_loc 回填，得到 F_A=F+ΔF_loc。",
  "4. 将T_A按建筑汇总并投影为查询key k_b。Memory仅存训练样本记录(k_j,V_j,y_j,source_j)。检索后得到支持局部token V_j、建筑级标签y_j、相似度s_j和有效掩码m_j。",
  "5. 对每栋支持建筑计算 U_j=CrossAttn(T_A,V_j,V_j)。将建筑标签编码广播到当前建筑的各区域后，与U_j拼接并投影为 Ũ_j。聚合权重 α_j 采用带有效掩码的归一化指数权重。",
  "6. ΔT_mem=Σ_j α_jŨ_j，ΔF_mem=QΔT_memW_mem。若Σ_jm_j=0，则令ΔT_mem=ΔF_mem=0。",
  "7. 分类器输出 P̂=softmax(h(F_A+ΔF_mem))，最终类别图由逐像素argmax得到。",
  "维度检查：F和ΔF均为N×C；T、T_A、ΔT均为M×d；W_loc与W_mem把区域维度d投影到像素通道C。",
  "注意：y_j是支持建筑的建筑级训练标签，不是局部区域标签；source_j只用于合法性筛选，不输入分类器。",
].join("\n"));

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const stagingDir = path.join(tmpDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v11.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
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
  fontPolicy: {
    basis: "reference",
    families: ["Arial", "Cambria Math"],
    referencePath: sourcePath,
    referenceSha256: "d964b186d9de1f6b656a9f1c64ef292e9a1d5b46cea0ab13bbb3709d14f23b8f",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "BRIGHT_model_formulas_v11_formula_final.validation.json"),
});

const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(path.join(tmpDir, "edited-slide-1.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(tmpDir, "edited-slide-1.layout.json"), await layout.text(), "utf8");
console.log(JSON.stringify({ finalPath, result }, null, 2));
