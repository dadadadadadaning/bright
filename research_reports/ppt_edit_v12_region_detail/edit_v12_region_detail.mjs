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
if (presentation.slides.count !== 1) throw new Error(`Expected one source slide, found ${presentation.slides.count}`);

const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

const NAVY = "#25313F";
const TEXT = "#344150";
const LINE = "#5F6872";
const SOFT_GRAY = "#F2F4F6";
const GRAY = "#DDE2E7";
const PINK = "#F5C9C9";
const PINK_BG = "#FFF0F0";
const BLUE = "#B6C8E4";
const YELLOW = "#FFD982";
const YELLOW_BG = "#FFF6DB";
const GREEN = "#C8DDB0";
const GREEN_BG = "#F0F7E8";
const WHITE = "#FFFFFF";
const LAYOUT_SCALE = 0.75;

function scalePosition(position) {
  if (!position) return position;
  const scaled = { ...position };
  for (const key of ["left", "top", "width", "height"]) {
    if (typeof scaled[key] === "number") scaled[key] *= LAYOUT_SCALE;
  }
  return scaled;
}

function addShape(spec) {
  return slide.shapes.add({ ...spec, position: scalePosition(spec.position) });
}

function addImage(spec) {
  return slide.images.add({ ...spec, position: scalePosition(spec.position) });
}

function addText(name, text, position, options = {}) {
  const s = addShape({
    geometry: "textbox",
    name,
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: options.typeface ?? "Arial",
    fontSize: (options.fontSize ?? 18) * LAYOUT_SCALE,
    bold: options.bold ?? false,
    color: options.color ?? TEXT,
    alignment: options.alignment ?? "center",
    verticalAlignment: options.verticalAlignment ?? "middle",
    wrap: options.wrap ?? "square",
    autoFit: options.autoFit ?? "shrinkText",
    insets: options.insets ?? { top: 2, right: 4, bottom: 2, left: 4 },
  };
  return s;
}

function addRect(name, position, fill, lineFill = LINE, radius = 10) {
  const s = addShape({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: 1.2 },
    borderRadius: radius,
  });
  return s;
}

function addBadge(n, left, top) {
  const s = addShape({
    geometry: "ellipse",
    name: `Step ${n}`,
    position: { left, top, width: 28, height: 28 },
    fill: NAVY,
    line: { fill: "none", width: 0 },
  });
  s.text = String(n);
  s.text.style = {
    typeface: "Arial", fontSize: 15 * LAYOUT_SCALE, bold: true, color: WHITE,
    alignment: "center", verticalAlignment: "middle", autoFit: "none",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
}

function addArrow(left, top, width = 24, height = 20) {
  addShape({
    geometry: "rightArrow",
    position: { left, top, width, height },
    fill: LINE,
    line: { fill: "none", width: 0 },
  });
}

function addLine(x1, y1, x2, y2, color = LINE, width = 1.4) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const lineWidth = Math.max(0.1, Math.abs(x2 - x1));
  const lineHeight = Math.max(0.1, Math.abs(y2 - y1));
  addShape({
    geometry: "line",
    position: {
      left,
      top,
      width: lineWidth,
      height: lineHeight,
      horizontalFlip: x2 < x1,
      verticalFlip: y2 < y1,
    },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
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
  const width = Math.max(1, Math.round(position.width * 3));
  const height = Math.max(1, Math.round(position.height * 3));
  const svg = formulaSvg(width, height, lines, (options.fontSize ?? 15) * 3);
  const png = await sharp(Buffer.from(svg, "utf8"), { density: 384 }).png().toBuffer();
  const image = addImage({
    blob: new Uint8Array(png),
    contentType: "image/png",
    alt: options.alt ?? name,
    fit: "contain",
    position,
  });
  image.name = name;
  return image;
}

function addFeatureStack(left, top, fill, label) {
  for (let i = 2; i >= 0; i--) {
    addShape({
      geometry: "rect",
      position: { left: left + i * 8, top: top - i * 7, width: 62, height: 42 },
      fill: i === 0 ? fill : "#FFFFFF",
      line: { style: "solid", fill: LINE, width: 1 },
    });
  }
  addText(`${label} label`, label, { left: left - 8, top: top + 47, width: 90, height: 24 }, { fontSize: 14, bold: true });
}

function addMatrix(left, top, rows, cols, fill) {
  const cell = 14;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      addShape({
        geometry: "rect",
        position: { left: left + c * (cell + 3), top: top + r * (cell + 3), width: cell, height: cell },
        fill: (r + c) % 3 === 0 ? fill : "#FFFFFF",
        line: { style: "solid", fill: LINE, width: 0.8 },
      });
    }
  }
}

// Slide header.
addText("Slide title", "Building-constrained pixel-preserving regional update", { left: 35, top: 18, width: 1160, height: 48 }, { fontSize: 30, bold: true, color: NAVY, alignment: "left" });
addText(
  "Slide subtitle",
  "P defines predicted buildings. R forms structural assignments. F carries RGB–SAR damage evidence.",
  { left: 38, top: 68, width: 1180, height: 28 },
  { fontSize: 16, color: "#66717D", alignment: "left" },
);
addText("Module note", "Inserted before the original pixel classifier", { left: 1215, top: 26, width: 245, height: 46 }, { fontSize: 14, bold: true, color: NAVY, fill: GREEN_BG, line: { style: "solid", fill: GREEN, width: 1 } });

// Five-stage editable diagram.
const stages = [
  { x: 35, w: 225, fill: SOFT_GRAY, title: "DamageFormer outputs" },
  { x: 285, w: 245, fill: YELLOW_BG, title: "Predicted buildings and centres" },
  { x: 555, w: 280, fill: YELLOW_BG, title: "Building-constrained assignment" },
  { x: 860, w: 300, fill: PINK_BG, title: "Pooling and regional interaction" },
  { x: 1185, w: 280, fill: GREEN_BG, title: "Pixel residual update" },
];
for (let i = 0; i < stages.length; i++) {
  const st = stages[i];
  addRect(`Stage ${i + 1}`, { left: st.x, top: 120, width: st.w, height: 360 }, st.fill, i === 0 ? "#A9B1B9" : i < 3 ? "#D1A848" : i === 3 ? "#D6A6A6" : "#91AD72", 12);
  addBadge(i + 1, st.x + 12, 132);
  addText(`Stage ${i + 1} title`, st.title, { left: st.x + 43, top: 128, width: st.w - 56, height: 44 }, { fontSize: 17, bold: true, color: NAVY, alignment: "left" });
  if (i < stages.length - 1) addArrow(st.x + st.w + 5, 286, 20, 18);
}

// Stage 1 icons and roles.
addFeatureStack(62, 206, GREEN, "P");
addFeatureStack(151, 206, BLUE, "R");
addFeatureStack(106, 322, PINK, "F");
addText("P role", "building probability", { left: 46, top: 280, width: 98, height: 24 }, { fontSize: 12 });
addText("R role", "RGB structure", { left: 145, top: 280, width: 98, height: 24 }, { fontSize: 12 });
addText("F role", "fused damage feature", { left: 75, top: 396, width: 145, height: 25 }, { fontSize: 12 });
addText("Stage 1 constraint", "Use features at one common spatial resolution", { left: 52, top: 435, width: 192, height: 31 }, { fontSize: 12, color: "#5E6872" });

// Stage 2 building and centres icon.
addShape({
  geometry: "custom",
  name: "Predicted building outline",
  position: { left: 326, top: 206, width: 154, height: 94 },
  fill: WHITE,
  line: { style: "solid", fill: NAVY, width: 1.4 },
  customPaths: [{
    width: 154, height: 94,
    commands: [
      { moveTo: { x: 12, y: 78 } }, { lineTo: { x: 20, y: 18 } },
      { lineTo: { x: 76, y: 10 } }, { lineTo: { x: 91, y: 35 } },
      { lineTo: { x: 139, y: 29 } }, { lineTo: { x: 144, y: 82 } },
      { lineTo: { x: 84, y: 87 } }, { lineTo: { x: 52, y: 68 } },
      { close: {} },
    ],
  }],
});
for (const [cx, cy] of [[350,236],[390,224],[436,251],[372,277]]) {
  addShape({ geometry: "ellipse", position: { left: cx, top: cy, width: 10, height: 10 }, fill: NAVY, line: { fill: "none", width: 0 } });
}
await addFormula(
  "Building instances",
  { left: 308, top: 312, width: 200, height: 34 },
  [{ markup: `${v("ℬ")} = ${roman("CC")}${v("(")}𝟙[${v("P")} &gt; ${v("τ")}${sub("b")}]${v(")")}` }],
  { fontSize: 15, alt: "Predicted building instances from thresholded building probability" },
);
await addFormula(
  "Adaptive centre count",
  { left: 302, top: 352, width: 214, height: 47 },
  [{ markup: `${v("K")}${sub("b")} = ${roman("clip")}${v("(")}⌈|${v("B")}${sub("b")}|/${v("A")}${sub("0")}⌉, 1, ${v("K")}${sub("max")}${v(")")}` }],
  { fontSize: 14, alt: "Adaptive number of region centres per predicted building" },
);
addText("Centre rule", "Initialise on a grid or by farthest-point sampling, then snap centres inside the predicted building", { left: 305, top: 405, width: 205, height: 55 }, { fontSize: 12, color: "#5E6872" });

// Stage 3 association illustration.
addMatrix(592, 205, 4, 4, PINK);
for (const [tx, ty] of [[723,218],[760,247],[717,282]]) {
  addShape({ geometry: "ellipse", position: { left: tx, top: ty, width: 12, height: 12 }, fill: NAVY, line: { fill: "none", width: 0 } });
}
for (const [x1,y1,x2,y2] of [[655,220,729,224],[655,237,766,253],[655,271,723,288],[638,254,729,224]]) addLine(x1,y1,x2,y2,"#A67C22",1.2);
await addFormula(
  "Association score",
  { left: 570, top: 316, width: 251, height: 55 },
  [{ markup: `${v("a")}${sub("ik")} = −‖${v("W")}${sub("r")}${v("R")}${sub("i")}−${v("μ")}${sub("k")}‖${sup("2")}/${v("τ")}${sub("r")} − ${v("λ")}${sub("x")}‖${v("x")}${sub("i")}−${v("c")}${sub("k")}‖${sup("2")}/${v("τ")}${sub("x")}` }],
  { fontSize: 12, alt: "Structure and spatial association score" },
);
await addFormula(
  "Soft assignment",
  { left: 588, top: 374, width: 215, height: 38 },
  [{ markup: `${v("Q")}${sub("ik")} = ${roman("softmax")}${sub("k∈𝒞ᵦ(i)")}${v("(")}${v("a")}${sub("ik")}${v(")")}` }],
  { fontSize: 14, alt: "Soft assignment within the same predicted building" },
);
addText("Assignment constraints", "Candidates remain inside the same predicted building. Background rows of Q are zero.", { left: 575, top: 420, width: 240, height: 42 }, { fontSize: 12, color: "#5E6872" });

// Stage 4 pooling and interaction icon.
addMatrix(890, 208, 3, 3, PINK);
for (let i = 0; i < 4; i++) {
  addShape({ geometry: "rect", position: { left: 993, top: 210 + i * 23, width: 54, height: 13 }, fill: PINK, line: { style: "solid", fill: LINE, width: 0.9 } });
}
for (const [cx,cy] of [[1087,211],[1128,231],[1098,275],[1049,255]]) {
  addShape({ geometry: "ellipse", position: { left: cx, top: cy, width: 13, height: 13 }, fill: PINK, line: { style: "solid", fill: LINE, width: 1 } });
}
for (const [x1,y1,x2,y2] of [[1093,217,1134,237],[1093,217,1055,261],[1134,237,1104,281],[1055,261,1104,281]]) addLine(x1,y1,x2,y2,"#7C8792",1.1);
await addFormula(
  "Normalized pooling",
  { left: 880, top: 315, width: 260, height: 62 },
  [
    { markup: `${v("D")}${sub("Q")} = ${roman("diag")}${v("(")}${v("Q")}${sup("⊤")}<tspan font-weight="700">1</tspan>${sub("N")}${v(")")} + ${v("εI")}` },
    { markup: `${v("T")} = ${v("D")}${sub("Q")}${sup("−1")}${v("Q")}${sup("⊤")}${v("F")}` },
  ],
  { fontSize: 14, alt: "Normalized soft-region pooling" },
);
await addFormula(
  "Regional interaction",
  { left: 900, top: 387, width: 220, height: 48 },
  [
    { markup: `${v("T")}${sub("A")} = ${v("g")}${sub("reg")}${v("(")}${v("T")}; ${v("A")}${sub("same")}${v(")")}` },
    { markup: `Δ${v("T")}${sub("loc")} = ${v("T")}${sub("A")} − ${v("T")}` },
  ],
  { fontSize: 13.5, alt: "Same-building regional attention and token residual" },
);
addText("Interaction rule", "One masked MHA + FFN block. Regions from different buildings never attend to each other.", { left: 884, top: 438, width: 250, height: 29 }, { fontSize: 12, color: "#5E6872" });

// Stage 5 residual unpooling with preserved pixel path.
addMatrix(1212, 202, 3, 3, PINK);
for (let i = 0; i < 3; i++) {
  addShape({ geometry: "rect", position: { left: 1323, top: 205 + i * 26, width: 52, height: 14 }, fill: PINK, line: { style: "solid", fill: LINE, width: 0.9 } });
}
for (const [x1,y1,x2,y2] of [[1270,216,1323,212],[1270,232,1323,238],[1270,249,1323,264]]) addLine(x1,y1,x2,y2,"#819265",1.1);
const residualAdd = addShape({
  geometry: "ellipse",
  position: { left: 1395, top: 224, width: 32, height: 32 },
  fill: WHITE,
  line: { style: "solid", fill: LINE, width: 1.2 },
});
residualAdd.text = "+";
residualAdd.text.style = {
  typeface: "Arial",
  fontSize: 18 * LAYOUT_SCALE,
  bold: true,
  color: NAVY,
  alignment: "center",
  verticalAlignment: "middle",
  autoFit: "none",
  insets: { top: 0, right: 0, bottom: 0, left: 0 },
};
addLine(1213,292,1411,292,LINE,1.5);
addLine(1411,292,1411,255,LINE,1.5);
addText("Preserved pixel path", "Original F", { left: 1205, top: 276, width: 90, height: 25 }, { fontSize: 12, bold: true, alignment: "left" });
await addFormula(
  "Pixel residual equations",
  { left: 1205, top: 324, width: 240, height: 71 },
  [
    { markup: `Δ${v("F")}${sub("loc")} = ${v("Q")} Δ${v("T")}${sub("loc")} ${v("W")}${sub("loc")}` },
    { markup: `${v("F")}${sub("A")} = ${v("F")} + Δ${v("F")}${sub("loc")}` },
    { markup: `${v("Z")} = ${v("h")}${v("(")}${v("F")}${sub("A")}${v(")")}` },
  ],
  { fontSize: 14, alt: "Regional residual unpooling and pixel classification feature" },
);
addText("Stage 5 explanation", "The region branch supplies an additive context residual. It never replaces the original pixel feature.", { left: 1207, top: 407, width: 238, height: 51 }, { fontSize: 12, color: "#5E6872" });

// Bottom implementation band tailored to the user's current evidence.
addRect("Implementation band", { left: 35, top: 515, width: 1430, height: 425 }, "#FAFBFC", "#C8CDD2", 10);
addText("Implementation heading", "Recommended first implementation for your current BRIGHT experiments", { left: 55, top: 532, width: 850, height: 34 }, { fontSize: 22, bold: true, color: NAVY, alignment: "left" });
addText("Candidate warning", "Candidate design. Validate the region mechanism before reconnecting inference Memory.", { left: 1000, top: 532, width: 430, height: 34 }, { fontSize: 14, bold: true, color: "#935F00", alignment: "right" });

addLine(515,580,515,900,"#D1D5D9",1);
addLine(1000,580,1000,900,"#D1D5D9",1);

addText("Roles heading", "Feature roles", { left: 60, top: 585, width: 410, height: 28 }, { fontSize: 18, bold: true, color: NAVY, alignment: "left" });
addText("Roles content", "P  groups predicted building instances\nR  determines structural and spatial assignments\nF  carries the fused RGB–SAR evidence used for classification\nQ  is generated without damage labels or test-time GT", { left: 60, top: 625, width: 410, height: 140 }, { fontSize: 15, alignment: "left", verticalAlignment: "top", insets: { top: 0, right: 5, bottom: 0, left: 0 } });
addText("Boundary note", "This module organises context inside predicted buildings. It does not perform RGB–SAR geometric alignment.", { left: 60, top: 790, width: 410, height: 70 }, { fontSize: 14, bold: true, color: "#8B3D3D", alignment: "left", fill: PINK_BG, line: { style: "solid", fill: PINK, width: 1 } });

addText("Config heading", "Starting configuration", { left: 545, top: 585, width: 410, height: 28 }, { fontSize: 18, bold: true, color: NAVY, alignment: "left" });
addText("Config content", "Choose A₀ so the median building yields about four regions\nKmax = 8\nFour candidate centres per pixel\nToken dimension d = 64\nOne attention block with four heads\nStop gradients through Q in the first controlled run\nZero or small initialise Wloc", { left: 545, top: 625, width: 410, height: 200 }, { fontSize: 15, alignment: "left", verticalAlignment: "top", insets: { top: 0, right: 5, bottom: 0, left: 0 } });
addText("Training sequence", "Training order: freeze the encoders first. Train the region block and output projection. Unfreeze the classifier and fusion layers only after an independent gain appears.", { left: 545, top: 835, width: 410, height: 63 }, { fontSize: 13, color: "#5E6872", alignment: "left" });

addText("Controls heading", "Required controls", { left: 1030, top: 585, width: 380, height: 28 }, { fontSize: 18, bold: true, color: NAVY, alignment: "left" });
const controls = [
  ["R0", "Parent pixel model"],
  ["R1", "Equal-capacity pixel residual block"],
  ["R2", "Matched grid regions + the same transformer"],
  ["R3", "Building-constrained content regions"],
];
for (let i = 0; i < controls.length; i++) {
  const y = 630 + i * 53;
  addText(`Control code ${controls[i][0]}`, controls[i][0], { left: 1030, top: y, width: 48, height: 32 }, { fontSize: 14, bold: true, color: NAVY, fill: i === 3 ? PINK : GRAY, line: { style: "solid", fill: LINE, width: 1 } });
  addText(`Control text ${controls[i][0]}`, controls[i][1], { left: 1090, top: y, width: 325, height: 32 }, { fontSize: 14, alignment: "left" });
}
addText("Success rule", "Keep the module only if R3 exceeds both R1 and R2, improves Damaged F1, and the best checkpoint is not step 0.", { left: 1030, top: 855, width: 385, height: 56 }, { fontSize: 14, bold: true, color: "#42602C", alignment: "left", fill: GREEN_BG, line: { style: "solid", fill: GREEN, width: 1 } });

addText("Footer", "Soft regions provide building context while the original pixel pathway preserves weak local evidence.", { left: 35, top: 958, width: 1430, height: 34 }, { fontSize: 15, bold: true, color: NAVY, alignment: "center" });
addText("Footnote", "All centre counts and dimensions are starting values for controlled validation, not established optima.", { left: 35, top: 997, width: 1430, height: 24 }, { fontSize: 12, color: "#68737E", alignment: "center" });

slide.speakerNotes.textFrame.setText([
  "本页根据当前DamageFormer实验情况补全区域模块。",
  "1. 输入职责分离：P只负责预测建筑实例，R只负责结构与空间软关联，F保留RGB-SAR融合后的损毁判别信息。建议R优先来自灾前RGB结构路径，避免SAR散斑主导区域边界。",
  "2. 验证和测试阶段不得使用GT建筑轮廓或GT损毁标签生成Q。预测建筑黏连或漏检属于定位误差，区域模块不能自动消除。",
  "3. K_b根据预测建筑面积自适应确定。A_0应根据F所在分辨率的训练集建筑面积统计选择，使中位数建筑约产生4个区域。K_max=8只是首轮起点。",
  "4. Q只连接同一预测建筑内部的候选中心。候选数建议先取4。背景像素的Q行全部为0。",
  "5. D_Q中加入εI，避免空区域除零。区域Transformer只允许同建筑区域交互。",
  "6. 区域模块只回填增量ΔF_loc，不能用QT_A直接替代像素特征F。输出投影W_loc采用零初始化或极小初始化，使训练起点接近父模型。",
  "7. 首轮不接Memory，不新增MIL、对比学习或长尾损失。先比较R0-R3，确认内容区域是否超过等容量像素块和规则网格。",
  "8. 本模块用于建筑内部上下文组织，不等价于RGB-SAR几何配准，也不能宣称消除了传感器差异。",
  "张量维度：F∈R^{N×C}，Q∈[0,1]^{N×M}，T和T_A∈R^{M×d}，W_loc∈R^{d×C}，ΔF_loc∈R^{N×C}。",
].join("\n"));

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const stagingDir = path.join(tmpDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v13.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  explicitTotalSlideCount: 2,
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
    referenceSha256: "91ef7d3614040b125e89e25cf913b9f8caa861aaf9d5768452c7361f45da07d0",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "BRIGHT_model_formulas_v13_region_complete.validation.json"),
});

for (let i = 0; i < presentation.slides.count; i++) {
  const s = presentation.slides.getItem(i);
  const preview = await presentation.export({ slide: s, format: "png", scale: 1.25 });
  await fs.writeFile(path.join(tmpDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await s.export({ format: "layout" });
  await fs.writeFile(path.join(tmpDir, `slide-${i + 1}.layout.json`), await layout.text(), "utf8");
}
console.log(JSON.stringify({ finalPath, result }, null, 2));
