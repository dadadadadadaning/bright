import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const skillDir = process.env.SKILL_DIR;
const workspaceDir = process.env.WORKSPACE_DIR;
const sourcePath = process.env.SOURCE_PPTX;
const tmpDir = process.env.TMP_DIR;
const finalPath = process.env.FINAL_PPTX;
const runtimePython = process.env.RUNTIME_PYTHON;
const regionRoot = process.env.REGION_ROOT;
const fullTileRoot = process.env.FULL_TILE_ROOT;
if (![skillDir, workspaceDir, sourcePath, tmpDir, finalPath, runtimePython, regionRoot, fullTileRoot].every(Boolean)) {
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
if (presentation.slides.count !== 1) throw new Error(`Expected one source slide, found ${presentation.slides.count}`);

const FONT = "Microsoft YaHei";
const NAVY = "#25313F";
const TEXT = "#344150";
const MUTED = "#657383";
const LINE = "#B6BEC7";
const PINK = "#D78383";
const PINK_BG = "#FFF1F1";
const BLUE = "#6D91BD";
const BLUE_BG = "#EEF4FA";
const YELLOW = "#D9A942";
const YELLOW_BG = "#FFF6DD";
const GREEN = "#799C5B";
const GREEN_BG = "#F1F7E9";
const WHITE = "#FFFFFF";

async function readPng(filePath) {
  return new Uint8Array(await fs.readFile(filePath));
}

function findExactText(slide, oldText) {
  const shape = slide.shapes.items.find((item) => item.text?.toString?.() === oldText);
  if (!shape) throw new Error(`Text not found: ${oldText}`);
  return shape;
}

function replaceExactText(slide, oldText, newText) {
  const shape = findExactText(slide, oldText);
  shape.text.replace(oldText, newText);
  return shape;
}

// Slide 1: retain the source comparison layout, but use two complete building
// crops with matched grid/SLIC region counts and no missing-coverage triangle.
const regionSlide = presentation.slides.getItem(0);
const sampleDirs = [
  path.join(regionRoot, "congo-volcano_00000013"),
  path.join(regionRoot, "bata-explosion_00000001"),
];
const regionImages = [];
for (const dir of sampleDirs) {
  for (let i = 1; i <= 6; i++) regionImages.push(await readPng(path.join(dir, `0${i}.png`)));
}
if (regionSlide.images.items.length < 12) throw new Error("The source region slide has fewer than 12 evidence images");
for (let i = 0; i < 12; i++) {
  const image = regionSlide.images.items[i];
  const frame = image.frame;
  image.replace({
    blob: regionImages[i],
    contentType: "image/png",
    fit: "contain",
    alt: `Region baseline evidence image ${i + 1}`,
  });
  image.frame = frame;
  image.fit = "contain";
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  image.geometry = "rect";
}

replaceExactText(regionSlide, "建筑内区域划分：真实影像对照与评价边界", "建筑内区域划分基线：规则网格与内容区域");
replaceExactText(regionSlide, "目的：检查固定网格与内容分区如何组织同一栋建筑，为后续区域聚合选择表示方式", "用途：比较固定网格与RGB内容分区的区域形态，为学习区域模块确定必要对照");
replaceExactText(regionSlide, "土耳其地震  turkey-earthquake_00000012", "刚果火山  congo-volcano_00000013");
replaceExactText(regionSlide, "网格 6 区 / SLIC 6 区", "网格 7 区 / SLIC 7 区");
replaceExactText(regionSlide, "巴塔爆炸  bata-explosion_00000007", "巴塔爆炸  bata-explosion_00000001");
replaceExactText(regionSlide, "网格 6 区 / SLIC 7 区", "网格 8 区 / SLIC 8 区");
replaceExactText(regionSlide, "当前能看出什么", "这张图能够回答什么");
replaceExactText(
  regionSlide,
  "SLIC 随 RGB 外观形成不规则分区\n部分边界细碎，需检查稳定性\n粉色突出选中区域，不代表损毁位置",
  "SLIC沿RGB外观形成不规则区域\n网格与SLIC均保持在同一建筑轮廓内\n粉色只标出一个区域，不表示真实损毁"
);
replaceExactText(regionSlide, "怎样检查区域质量", "这张图不能回答什么");
replaceExactText(
  regionSlide,
  "检查跨楼、背景混入与碎片比例\n控制区域数量，检查连通性与稳定性\n此处用 GT 限制范围，不能验证定位",
  "GT轮廓已经排除了定位误差\nSLIC没有参与模型训练或像素分类\n区域外观不能证明Damaged识别得到改善"
);
replaceExactText(regionSlide, "怎样证明你的设计有效", "你的学习区域怎样验证");
replaceExactText(
  regionSlide,
  "同预算对比像素、网格与学习区域\n评价 Damaged 的 IoU、精确率和召回率\n检查各事件收益，单独消融 Memory",
  "固定父模型与训练预算，对比R0至R3\n使用预测建筑生成Q，测试时不读取GT\n报告Damaged精确率、召回率与事件结果"
);
replaceExactText(
  regionSlide,
  "说明：此图为 GT 轮廓约束的 SLIC 对照，尚未运行你的学习区域模块，不能据此宣称本方法有效。",
  "说明：本页仍是GT轮廓约束的网格与SLIC基线，不是你的学习区域输出，也不用于证明精度提升。"
);
replaceExactText(
  regionSlide,
  "同坐标裁剪，未额外配准。SAR 仅做 1–99% 显示拉伸。巴塔两种分区数量不同，暂不构成严格等数量对照。",
  "两例均使用同坐标完整建筑裁剪；SAR仅做1–99%显示拉伸；网格与SLIC区域数在每个案例内保持一致。"
);
regionSlide.speakerNotes.textFrame.setText([
  "本页用于建立区域模块的可视化基线。",
  "案例1为 congo-volcano_00000013，规则网格和SLIC均为7个区域。",
  "案例2为 bata-explosion_00000001，规则网格和SLIC均为8个区域。",
  "所有分区使用GT建筑轮廓限制，仅用于观察区域形态。它们不是预测建筑、学习软关联Q或模型损毁输出。",
  "两例保留完整建筑裁剪，没有使用带大面积缺测黑区的案例。SAR仅为显示进行1至99百分位拉伸。",
  "第一轮学习区域实验必须使用预测建筑生成Q，并与等容量像素残差块和等数量规则网格比较。",
].join("\n"));

// Slide 2: actual local inference plus a concrete decision path.
const slide = presentation.slides.add();
slide.background.fill = WHITE;

function addShape(spec) {
  return slide.shapes.add(spec);
}

function addText(name, text, position, options = {}) {
  const shape = addShape({
    geometry: "textbox",
    name,
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? TEXT,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "middle",
    wrap: "square",
    autoFit: options.autoFit ?? "shrinkText",
    insets: options.insets ?? { top: 3, right: 5, bottom: 3, left: 5 },
  };
  return shape;
}

function addLine(left, top, width, height, fill = LINE, lineWidth = 1) {
  return addShape({
    geometry: "line",
    position: { left, top, width: Math.max(width, 0.1), height: Math.max(height, 0.1) },
    fill: "none",
    line: { style: "solid", fill, width: lineWidth },
  });
}

function addNumber(number, left, top, fill) {
  const badge = addShape({
    geometry: "ellipse",
    position: { left, top, width: 31, height: 31 },
    fill,
    line: { fill: "none", width: 0 },
  });
  badge.text = String(number);
  badge.text.style = {
    typeface: FONT,
    fontSize: 15,
    bold: true,
    color: WHITE,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "none",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
}

const rgb = await readPng(path.join(fullTileRoot, "testarea-disaster_00000085_rgb.png"));
const sar = await readPng(path.join(fullTileRoot, "testarea-disaster_00000085_sar.png"));
const location = await readPng(path.join(fullTileRoot, "testarea-disaster_00000085_location.png"));
const prediction = await readPng(path.join(fullTileRoot, "testarea-disaster_00000085_prediction.png"));

addText("Title", "你的已实现模型：完整瓦片输入与真实输出", { left: 30, top: 22, width: 1080, height: 55 }, { fontSize: 34, bold: true, color: NAVY });
addText(
  "Subtitle",
  "MyModel01已完成本地前向；本页用于确认运行链和输出形态，不用于证明候选区域或Memory有效",
  { left: 30, top: 83, width: 1180, height: 38 },
  { fontSize: 20, color: TEXT }
);
addText("Status", "已运行 checkpoint", { left: 1190, top: 35, width: 220, height: 42 }, { fontSize: 17, bold: true, color: "#46672E", alignment: "center", fill: GREEN_BG, line: { style: "solid", fill: GREEN, width: 1 } });

const panels = [
  { x: 30, title: "灾前RGB", subtitle: "完整1024×1024", bytes: rgb, color: PINK, alt: "Full pre-event RGB tile" },
  { x: 383, title: "灾后SAR", subtitle: "完整1024×1024", bytes: sar, color: BLUE, alt: "Full post-event SAR tile" },
  { x: 736, title: "建筑位置预测", subtitle: "MyModel01定位分支", bytes: location, color: YELLOW, alt: "Predicted building location mask" },
  { x: 1089, title: "四分类损毁图", subtitle: "MyModel01损毁分支", bytes: prediction, color: GREEN, alt: "Four-class damage prediction" },
];
for (let i = 0; i < panels.length; i++) {
  const panel = panels[i];
  addText(`Panel ${i + 1} title`, panel.title, { left: panel.x, top: 137, width: 321, height: 36 }, { fontSize: 23, bold: true, color: NAVY, alignment: "center" });
  addText(`Panel ${i + 1} subtitle`, panel.subtitle, { left: panel.x, top: 171, width: 321, height: 28 }, { fontSize: 14, color: MUTED, alignment: "center" });
  const image = slide.images.add({
    blob: panel.bytes,
    contentType: "image/png",
    alt: panel.alt,
    fit: "contain",
    position: { left: panel.x + 10, top: 205, width: 301, height: 301 },
  });
  image.fit = "contain";
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  image.geometry = "rect";
  addShape({
    geometry: "rect",
    position: { left: panel.x + 10, top: 205, width: 301, height: 301 },
    fill: "none",
    line: { style: "solid", fill: panel.color, width: 1.2 },
  });
}

// Compact prediction legend.
const legend = [
  ["背景", "#FFFFFF", "#AAB2BA"],
  ["完整", "#46B579", "#46B579"],
  ["损毁", "#E4BD8B", "#D39E57"],
  ["毁坏", "#B64645", "#B64645"],
];
for (let i = 0; i < legend.length; i++) {
  const x = 1099 + i * 78;
  addShape({ geometry: "rect", position: { left: x, top: 517, width: 18, height: 18 }, fill: legend[i][1], line: { style: "solid", fill: legend[i][2], width: 1 } });
  addText(`Legend ${i}`, legend[i][0], { left: x + 21, top: 512, width: 54, height: 28 }, { fontSize: 12, color: TEXT });
}
addText("Tile ID", "样本：testarea-disaster_00000085 · 无匹配公开GT", { left: 30, top: 525, width: 1000, height: 28 }, { fontSize: 14, color: MUTED });

addLine(30, 565, 1380, 0, "#CBD1D7", 1);
addText("Evidence heading", "当前证据", { left: 30, top: 585, width: 400, height: 38 }, { fontSize: 25, bold: true, color: NAVY });
addText(
  "Evidence body",
  "本地checkpoint能够在完整瓦片上输出建筑位置与四分类损毁图。该瓦片的Damaged预测仅1个像素，但没有GT，不能据此判断模型漏检，也不能把结果归因于尚未实现的学习区域或Memory。",
  { left: 30, top: 632, width: 410, height: 165 },
  { fontSize: 17, verticalAlignment: "top", insets: { top: 0, right: 6, bottom: 0, left: 0 } }
);
addText("Identity", "实际运行：MyModel01_complete_20260517_135420 / best_model.pth", { left: 30, top: 825, width: 410, height: 54 }, { fontSize: 14, bold: true, color: "#8B5A2B", fill: YELLOW_BG, line: { style: "solid", fill: YELLOW, width: 1 } });

addLine(467, 602, 0, 286, "#D4D8DD", 1);
addText("Plan heading", "下一步最小验证路线", { left: 500, top: 585, width: 875, height: 38 }, { fontSize: 25, bold: true, color: NAVY });
const steps = [
  {
    x: 500,
    color: PINK,
    title: "区域模块独立试验",
    body: "固定同一父模型与数据划分；先冻结编码器，只训练区域交互及输出投影。Q由预测建筑生成，Val/Test均不读取GT。",
  },
  {
    x: 795,
    color: BLUE,
    title: "同预算四组对照",
    body: "R0父像素模型；R1等容量像素残差；R2等数量规则网格；R3学习内容区域。各组训练步数、损失及初始化一致",
  },
  {
    x: 1090,
    color: GREEN,
    title: "通过条件与Memory",
    body: "R3同时超过R1和R2，Damaged F1提升，最佳检查点不为step 0。满足后再修复并接回Memory，否则停止该方向。",
  },
];
for (let i = 0; i < steps.length; i++) {
  const item = steps[i];
  addNumber(i + 1, item.x, 640, item.color);
  addText(`Step ${i + 1} title`, item.title, { left: item.x + 42, top: 634, width: 240, height: 42 }, { fontSize: 19, bold: true, color: NAVY });
  addText(`Step ${i + 1} body`, item.body, { left: item.x, top: 687, width: 275, height: 160 }, { fontSize: 16, verticalAlignment: "top", insets: { top: 0, right: 6, bottom: 0, left: 0 } });
  if (i < steps.length - 1) addLine(item.x + 280, 660, 15, 0, "#9AA2AA", 1.4);
}
addText(
  "Decision",
  "当前优先级：先证明区域交互有独立增益，再讨论推理Memory。",
  { left: 500, top: 858, width: 865, height: 55 },
  { fontSize: 18, bold: true, color: "#46672E", alignment: "center", fill: GREEN_BG, line: { style: "solid", fill: GREEN, width: 1 } }
);

addText("Footer", "区域外观图用于提出假设；同预算消融和完整验证结果才决定是否继续。", { left: 30, top: 942, width: 1380, height: 35 }, { fontSize: 16, bold: true, color: NAVY, alignment: "center" });

slide.speakerNotes.textFrame.setText([
  "本页补充用户已实现模型的实际本地推理结果。",
  "样本为BRIGHT DFC25 Track-2测试瓦片 testarea-disaster_00000085，RGB和SAR均为完整1024×1024画幅。",
  "权重为 saved_weights/BRIGHT_mymodel/MyModel01_complete_20260517_135420/best_model.pth。",
  "本次全图前向使用SAR频域滤波、统计风格对齐和融合路径；ROI门控交叉注意力不参与。",
  "预测类别像素数为background 748442、intact 271713、damaged 1、destroyed 28420。",
  "该匿名测试瓦片没有匹配的公开GT，所以只能确认运行链和输出形态，不能计算IoU或判断Damaged漏检。",
  "学习区域与推理Memory候选结构尚未由该checkpoint实现。下一步优先做R0至R3同预算区域实验。",
].join("\n"));

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const stagingDir = path.join(tmpDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-regions-own-v4.pptx");
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
    "--expected-slide-size-emu", "13716000,9525000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  fontPolicy: {
    basis: "reference",
    families: ["Microsoft YaHei"],
    referencePath: sourcePath,
    referenceSha256: "930f4569bd5933c9e1ab79e293d1f2997dccca369981cbf530498e70076e3b99",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "BRIGHT_regions_and_own_model_v4.validation.json"),
});

for (let i = 0; i < presentation.slides.count; i++) {
  const current = presentation.slides.getItem(i);
  const preview = await presentation.export({ slide: current, format: "png", scale: 1.25 });
  await fs.writeFile(path.join(tmpDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await current.export({ format: "layout" });
  await fs.writeFile(path.join(tmpDir, `slide-${i + 1}.layout.json`), await layout.text(), "utf8");
}

console.log(JSON.stringify({ finalPath, result }, null, 2));
