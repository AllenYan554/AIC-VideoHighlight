import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import {
  FileBlob,
  Presentation,
  PresentationFile,
} from "file:///C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const workspaceDir = "D:\\CDUT\\硕士\\2026-09 AIC\\AIC-VideoHighlight";
const SPEC_PATH = "D:\\CDUT\\硕士\\2026-09 AIC\\实验记录\\Stage7_VHiCraFTNet研发\\FTNet.md";
const SKILL_DIR = "C:\\Users\\lenovo\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.909.12148\\skills\\presentations";
const RUNTIME_NODE = "C:\\Users\\lenovo\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\bin\\node.exe";
const RUNTIME_PYTHON = "C:\\Users\\lenovo\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const RUNTIME_NODE_MODULES = "C:\\Users\\lenovo\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";
const RUNTIME_BIN_DIR = "C:\\Users\\lenovo\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\bin\\override";
process.env.RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;

const buildDir = path.join(workspaceDir, ".codex-build", "ftnet-flowchart-v2");
const stagingDir = path.join(workspaceDir, ".codex-finalizer", "ftnet-flowchart-v2");
const validatedDir = path.join(workspaceDir, ".codex-validated", "ftnet-flowchart-v2");
const outputDir = path.join(workspaceDir, "artifacts");
const FINAL_PPTX = path.join(outputDir, "FTNet_Flowchart_Editable.pptx");
const FINAL_PREVIEW = path.join(outputDir, "FTNet_Flowchart_Preview.png");
const FINAL_NATIVE_PREVIEW = path.join(outputDir, "FTNet_Native16_Preview.png");
const buildToken = String(Date.now());
const validatedPptx = path.join(validatedDir, `FTNet_Flowchart_Editable_${buildToken}.pptx`);

const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(validatedDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const FONT = "Segoe UI";
const CJK_FONT = "Noto Sans SC";
const C = {
  bg: "#F7F9FC",
  paper: "#FFFFFF",
  ink: "#10213A",
  muted: "#5D6C80",
  faint: "#8895A6",
  line: "#D7DFEA",
  navy: "#18345F",
  blue: "#356AE6",
  blueDark: "#214AA5",
  blueFill: "#EAF0FF",
  orange: "#D96C2E",
  orangeFill: "#FFF0E7",
  teal: "#16857B",
  tealFill: "#E7F6F3",
  purple: "#7455C3",
  purpleFill: "#F0ECFB",
  green: "#2E7D61",
  greenFill: "#EAF5EF",
  red: "#B84752",
  redFill: "#FBECEF",
  sand: "#A56A27",
  sandFill: "#FFF6E8",
};

function addText(slide, {
  name,
  x,
  y,
  w,
  h,
  text,
  fontSize = 16,
  color = C.ink,
  bold = false,
  align = "left",
  valign = "middle",
  typeface = FONT,
  insets = { top: 0, right: 0, bottom: 0, left: 0 },
}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface,
    fontSize,
    color,
    bold,
    alignment: align,
    verticalAlignment: valign,
    autoFit: "shrinkText",
    wrap: "square",
    insets,
  };
  return shape;
}

function addBox(slide, {
  name,
  x,
  y,
  w,
  h,
  fill = C.paper,
  stroke = C.line,
  strokeWidth = 1,
  radius = 12,
  text = "",
  fontSize = 15,
  color = C.ink,
  bold = false,
  align = "center",
  valign = "middle",
  typeface = FONT,
  dash = false,
  insets = { top: 8, right: 10, bottom: 8, left: 10 },
}) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: dash ? "dashed" : "solid", fill: stroke, width: strokeWidth },
    borderRadius: radius,
  });
  shape.text = text;
  shape.text.style = {
    typeface,
    fontSize,
    color,
    bold,
    alignment: align,
    verticalAlignment: valign,
    autoFit: "shrinkText",
    wrap: "square",
    insets,
  };
  return shape;
}

function addRule(slide, name, x, y, w, color = C.line, h = 2) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addDot(slide, name, x, y, size, fill) {
  return slide.shapes.add({
    geometry: "ellipse",
    name,
    position: { left: x, top: y, width: size, height: size },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function connect(slide, from, to, {
  kind = "straight",
  fromSide = "right",
  toSide = "left",
  color = C.navy,
  width = 2,
  dash = false,
  arrow = true,
} = {}) {
  return slide.shapes.connect(from, to, {
    kind,
    fromSide,
    toSide,
    line: { style: dash ? "dashed" : "solid", fill: color, width },
    head: { type: "none" },
    tail: arrow ? { type: "triangle", width: "sm", length: "sm" } : { type: "none" },
    cap: "round",
    join: "round",
  });
}

function addPageChrome(slide, page, title, subtitle, {
  titleWidth = 880,
  titleFontSize = 36,
} = {}) {
  slide.background.fill = C.bg;
  addText(slide, {
    name: `slide-${page}-title`,
    x: 38,
    y: 22,
    w: titleWidth,
    h: 42,
    text: title,
    fontSize: titleFontSize,
    color: C.ink,
    bold: true,
  });
  addText(slide, {
    name: `slide-${page}-subtitle`,
    x: 40,
    y: 65,
    w: 880,
    h: 24,
    text: subtitle,
    fontSize: 15,
    color: C.muted,
  });
  addText(slide, {
    name: `slide-${page}-page-number`,
    x: 1202,
    y: 681,
    w: 38,
    h: 20,
    text: `0${page}`,
    fontSize: 12,
    color: C.faint,
    bold: true,
    align: "right",
  });
}

// Slide 1: FTNet architecture
const slide1 = presentation.slides.add();
addPageChrome(
  slide1,
  1,
  "FTNet — Fine-grained Highlight Frame Selection Network",
  "Visual Appearance + Temporal Change + Structured Pipeline Context",
  { titleWidth: 850, titleFontSize: 30.5 },
);

addText(slide1, { name: "hypothesis-appearance", x: 906, y: 27, w: 92, h: 22, text: "Appearance", fontSize: 15, color: C.blueDark, bold: true, align: "center" });
addText(slide1, { name: "hypothesis-plus-1", x: 996, y: 27, w: 20, h: 22, text: "+", fontSize: 16, color: C.faint, bold: true, align: "center" });
addText(slide1, { name: "hypothesis-change", x: 1015, y: 27, w: 70, h: 22, text: "Change", fontSize: 15, color: C.orange, bold: true, align: "center" });
addText(slide1, { name: "hypothesis-plus-2", x: 1084, y: 27, w: 20, h: 22, text: "+", fontSize: 16, color: C.faint, bold: true, align: "center" });
addText(slide1, { name: "hypothesis-context", x: 1102, y: 27, w: 142, h: 22, text: "Structured Context", fontSize: 15, color: C.purple, bold: true, align: "center" });
addText(slide1, { name: "hypothesis-label", x: 917, y: 52, w: 318, h: 17, text: "FTNet three-branch research hypothesis", fontSize: 10.5, color: C.faint, align: "center" });
addRule(slide1, "slide-1-title-rule", 38, 93, 1204, C.line, 2);

addText(slide1, { name: "upstream-label", x: 40, y: 103, w: 190, h: 18, text: "UPSTREAM / PRECOMPUTED", fontSize: 11, color: C.faint, bold: true });
const upstreamSteps = [
  { label: "Video", note: "source", color: C.navy },
  { label: "Qwen", note: "coarse high-recall retrieval", color: C.navy },
  { label: "RT-DETR", note: "visual representation", color: C.blue },
  { label: "LOC", note: "primary subject", color: C.orange },
  { label: "CMP", note: "subject-aware focus", color: C.purple },
  { label: "TS", note: "temporal stabilization", color: C.teal },
];
const upstreamNodes = [];
const upstreamX = [48, 151, 281, 409, 514, 624];
for (let i = 0; i < upstreamSteps.length; i += 1) {
  const step = upstreamSteps[i];
  const x = upstreamX[i];
  const dot = addDot(slide1, `upstream-dot-${i}`, x, 133, 13, step.color);
  upstreamNodes.push(dot);
  addText(slide1, { name: `upstream-label-${i}`, x: x - 24, y: 121, w: 62, h: 18, text: step.label, fontSize: 12.5, color: C.ink, bold: true, align: "center" });
  addText(slide1, { name: `upstream-note-${i}`, x: x - 38, y: 149, w: 92, h: 26, text: step.note, fontSize: 8.8, color: C.muted, align: "center", valign: "top" });
  if (i > 0) connect(slide1, upstreamNodes[i - 1], dot, { color: C.faint, width: 1.4 });
}
addText(slide1, {
  name: "upstream-cmp-note",
  x: 710,
  y: 117,
  w: 522,
  h: 49,
  text: "CMP = Subject-Aware Composition / Subject Focus\nRT-DETR supplies visual features. LOC, CMP, and TS supply structured Native context.",
  fontSize: 12.5,
  color: C.muted,
  align: "right",
  typeface: CJK_FONT,
});

const branchStage = addBox(slide1, {
  name: "ftnet-branch-stage",
  x: 35,
  y: 193,
  w: 590,
  h: 312,
  fill: C.paper,
  stroke: C.line,
  strokeWidth: 1.2,
  radius: 18,
});
branchStage.sendToBack();
addText(slide1, { name: "branch-stage-label", x: 52, y: 203, w: 320, h: 20, text: "FTNET INPUT PROJECTIONS", fontSize: 11, color: C.faint, bold: true });

function addBranchLane({ y, fill, accent, name, eyebrow, question, input, transform, output, nativeGroups = "" }) {
  const lane = addBox(slide1, {
    name: `${name}-lane`, x: 50, y, w: 558, h: name === "native" ? 82 : 76,
    fill, stroke: fill, strokeWidth: 0.5, radius: 14,
  });
  lane.sendToBack();
  addRule(slide1, `${name}-accent`, 50, y, 6, accent, name === "native" ? 82 : 76);
  addText(slide1, { name: `${name}-eyebrow`, x: 70, y: y + 8, w: 152, h: 20, text: eyebrow, fontSize: 16, color: accent, bold: true });
  addText(slide1, { name: `${name}-question`, x: 70, y: y + 29, w: 152, h: 35, text: question, fontSize: 11.5, color: C.muted, valign: "top" });
  addText(slide1, { name: `${name}-input`, x: 225, y: y + 8, w: 90, h: 24, text: input, fontSize: 15, color: C.ink, bold: true, align: "center" });
  addText(slide1, { name: `${name}-transform`, x: 310, y: y + 6, w: 190, h: nativeGroups ? 38 : 60, text: transform, fontSize: nativeGroups ? 13 : 14, color: C.ink, bold: true, align: "center" });
  if (nativeGroups) {
    addText(slide1, { name: `${name}-groups`, x: 225, y: y + 45, w: 275, h: 29, text: nativeGroups, fontSize: 9.8, color: C.muted, align: "center", typeface: CJK_FONT });
  }
  return addBox(slide1, {
    name: `${name}-output`, x: 514, y: y + 17, w: 78, h: 42,
    fill: C.paper, stroke: accent, strokeWidth: 1.2, radius: 10,
    text: output, fontSize: 15.5, color: accent, bold: true,
    insets: { top: 4, right: 5, bottom: 4, left: 5 },
  });
}

const visualOut = addBranchLane({
  y: 226, fill: C.blueFill, accent: C.blueDark, name: "visual",
  eyebrow: "VISUAL", question: "What is in this frame?", input: "eₜ ∈ ℝ²⁵⁶",
  transform: "Linear 256→96\nLayerNorm · GELU", output: "vₜ\n96D",
});
const deltaOut = addBranchLane({
  y: 314, fill: C.orangeFill, accent: C.orange, name: "delta",
  eyebrow: "TEMPORAL CHANGE", question: "What changed?", input: "Δeₜ ∈ ℝ²⁵⁶",
  transform: "Linear 256→96\nLayerNorm · GELU", output: "dₜ\n96D",
});
addText(slide1, { name: "delta-definition", x: 224, y: 376, w: 276, h: 13, text: "Δeₜ = eₜ − eₜ₋₁ for true source adjacency; otherwise 0", fontSize: 8.8, color: C.muted, align: "center" });
const nativeOut = addBranchLane({
  y: 402, fill: C.purpleFill, accent: C.purple, name: "native",
  eyebrow: "STRUCTURED CONTEXT", question: "Where and how is this frame in the pipeline?", input: "nₜ ∈ ℝ¹⁶",
  transform: "Linear 16→32\nLayerNorm · GELU", output: "rₜ\n32D",
  nativeGroups: "Retrieval 5D   Subject / Geometry 6D   Composition 1D   Temporal Stability 4D",
});
addBox(slide1, {
  name: "native-required-tag", x: 71, y: 468, w: 90, h: 18,
  fill: C.purple, stroke: C.purple, radius: 9,
  text: "REQUIRED BRANCH", fontSize: 8.6, color: C.paper, bold: true,
  insets: { top: 2, right: 4, bottom: 2, left: 4 },
});

const mergeHub = addDot(slide1, "branch-merge-hub", 647, 345, 16, C.navy);
connect(slide1, visualOut, mergeHub, { kind: "elbow", color: C.blueDark, width: 1.8 });
connect(slide1, deltaOut, mergeHub, { color: C.orange, width: 1.8 });
connect(slide1, nativeOut, mergeHub, { kind: "elbow", color: C.purple, width: 1.8 });

const fusion = addBox(slide1, {
  name: "fusion", x: 681, y: 269, w: 142, h: 170,
  fill: C.paper, stroke: C.navy, strokeWidth: 1.8, radius: 16,
});
addText(slide1, { name: "fusion-title", x: 697, y: 282, w: 110, h: 22, text: "FUSION", fontSize: 17, color: C.navy, bold: true, align: "center" });
addText(slide1, { name: "fusion-concat", x: 694, y: 310, w: 116, h: 48, text: "Concat\n96 + 96 + 32", fontSize: 16, color: C.ink, bold: true, align: "center" });
addBox(slide1, { name: "fusion-dim", x: 720, y: 356, w: 64, h: 31, fill: C.navy, stroke: C.navy, radius: 15, text: "224D", fontSize: 16, color: C.paper, bold: true, insets: { top: 3, right: 4, bottom: 3, left: 4 } });
addText(slide1, { name: "fusion-projection", x: 694, y: 392, w: 116, h: 35, text: "Linear 224→128\nGELU · fₜ ∈ ℝ¹²⁸", fontSize: 12.5, color: C.navy, bold: true, align: "center" });
connect(slide1, mergeHub, fusion, { color: C.navy, width: 2.2 });

const tcn = addBox(slide1, {
  name: "tcn-backbone", x: 850, y: 226, w: 192, h: 178,
  fill: C.blueFill, stroke: C.blueDark, strokeWidth: 1.6, radius: 16,
});
addText(slide1, { name: "tcn-title", x: 868, y: 240, w: 156, h: 50, text: "5 × Dilated Depthwise\nSeparable TCN Blocks", fontSize: 15.5, color: C.blueDark, bold: true, align: "center" });
addText(slide1, { name: "tcn-channels", x: 872, y: 295, w: 148, h: 24, text: "Channels = 128", fontSize: 14, color: C.ink, bold: true, align: "center" });
addText(slide1, { name: "tcn-dilation", x: 868, y: 327, w: 156, h: 31, text: "Dilation\n1   2   4   8   16", fontSize: 13, color: C.muted, bold: true, align: "center" });
addBox(slide1, { name: "tcn-output", x: 902, y: 365, w: 88, h: 28, fill: C.blueDark, stroke: C.blueDark, radius: 14, text: "hₜ ∈ ℝ¹²⁸", fontSize: 14, color: C.paper, bold: true, insets: { top: 3, right: 5, bottom: 3, left: 5 } });
connect(slide1, fusion, tcn, { color: C.navy, width: 2.4 });

addBox(slide1, {
  name: "tcn-single-block-inset", x: 850, y: 419, w: 192, h: 82,
  fill: C.paper, stroke: C.line, strokeWidth: 1.1, radius: 13,
});
addText(slide1, { name: "tcn-inset-title", x: 865, y: 428, w: 162, h: 17, text: "SINGLE TCN BLOCK", fontSize: 10.5, color: C.faint, bold: true, align: "center" });
addText(slide1, { name: "tcn-inset-flow", x: 865, y: 449, w: 162, h: 38, text: "Depthwise Conv1D  →  Pointwise Conv1D\nLayerNorm  →  GELU  →  Dropout  →  Residual Add", fontSize: 10.7, color: C.ink, bold: true, align: "center" });

const yHead = addBox(slide1, {
  name: "y-head", x: 1067, y: 250, w: 128, h: 155,
  fill: C.paper, stroke: C.navy, strokeWidth: 1.8, radius: 16,
  text: "Y HEAD\n\nLinear 128→1\nlogit aₜ\nSigmoid\npₜ",
  fontSize: 15, color: C.navy, bold: true,
});
connect(slide1, tcn, yHead, { color: C.navy, width: 2.4 });
addText(slide1, {
  name: "probability-semantics", x: 1046, y: 413, w: 202, h: 62,
  text: "Probability that candidate frame t\nshould be retained in the final highlight video\n候选帧进入最终高光剪辑的保留概率",
  fontSize: 10.3, color: C.muted, bold: true, align: "center", valign: "top", typeface: CJK_FONT,
});

addText(slide1, { name: "outside-label", x: 1080, y: 478, w: 172, h: 17, text: "OUTSIDE FTNet", fontSize: 10.5, color: C.teal, bold: true, align: "center" });
const decoder = addBox(slide1, {
  name: "temporal-coherence-decoder", x: 1061, y: 497, w: 194, h: 76,
  fill: C.tealFill, stroke: C.teal, strokeWidth: 1.5, radius: 14,
  text: "Temporal Coherence Decoder\nContinuous Highlight Segments\n[start_time, end_time]",
  fontSize: 12.5, color: C.teal, bold: true,
});
const decoderRouteTop = addDot(slide1, "decoder-route-top", 1262, 325, 5, C.teal);
const decoderRouteBottom = addDot(slide1, "decoder-route-bottom", 1262, 532, 5, C.teal);
connect(slide1, yHead, decoderRouteTop, { fromSide: "right", toSide: "left", color: C.teal, width: 2.2, arrow: false });
connect(slide1, decoderRouteTop, decoderRouteBottom, { fromSide: "bottom", toSide: "top", color: C.teal, width: 2.2, arrow: false });
connect(slide1, decoderRouteBottom, decoder, { fromSide: "left", toSide: "right", color: C.teal, width: 2.2 });

const trainingStrip = addBox(slide1, {
  name: "training-strip", x: 35, y: 535, w: 987, h: 95,
  fill: C.sandFill, stroke: "#E7C89D", strokeWidth: 1.1, radius: 15,
});
trainingStrip.sendToBack();
addText(slide1, { name: "training-label", x: 52, y: 546, w: 130, h: 18, text: "TRAINING PROTOCOL", fontSize: 10.5, color: C.sand, bold: true });
const trainX = [54, 232, 390, 534, 676];
const trainW = [132, 108, 96, 104, 166];
const trainTexts = ["Human Highlight GT", "Y Loss", "Backward", "AdamW", "Update FTNet learnable core"];
const trainNodes = trainTexts.map((text, i) => addBox(slide1, {
  name: `training-step-${i}`, x: trainX[i], y: 570, w: trainW[i], h: 42,
  fill: C.paper, stroke: "#E2B97E", strokeWidth: 1, radius: 10,
  text, fontSize: 12, color: C.sand, bold: true,
  insets: { top: 4, right: 6, bottom: 4, left: 6 },
}));
for (let i = 1; i < trainNodes.length; i += 1) connect(slide1, trainNodes[i - 1], trainNodes[i], { color: C.sand, width: 1.5 });
addBox(slide1, { name: "stop-gradient", x: 847, y: 570, w: 103, h: 19, fill: C.redFill, stroke: C.red, radius: 9, text: "STOP GRADIENT", fontSize: 8.8, color: C.red, bold: true, insets: { top: 2, right: 3, bottom: 2, left: 3 } });
addText(slide1, { name: "stop-gradient-note", x: 844, y: 591, w: 110, h: 16, text: "at frozen RT-DETR", fontSize: 8.8, color: C.red, bold: true, align: "center" });
addText(slide1, { name: "training-config", x: 52, y: 613, w: 795, h: 14, text: "AdamW · LR 3e−4 · WD 1e−2 · 40 epochs · Batch 4 · Grad clip 1.0", fontSize: 9.2, color: C.muted });

addText(slide1, { name: "q-head-note", x: 1048, y: 590, w: 208, h: 33, text: "Q Head: optional future research\nNot used for Stage 7 Y selection", fontSize: 9.8, color: C.faint, align: "center" });
addText(slide1, { name: "slide-1-footer", x: 38, y: 682, w: 1135, h: 18, text: "Stage 7 reference architecture · Native branch required · Decoder remains outside FTNet", fontSize: 9.5, color: C.faint });

slide1.speakerNotes.textFrame.setText([
  `Source of truth: ${SPEC_PATH}`,
  "Architecture: Visual 256→96, Delta 256→96, Native 16→32, concat 224, fusion 224→128, TCN channels 128, Y Head 128→1.",
  "Native branch is required. Q Head is not used for Stage 7 Y selection. Decoder is outside FTNet.",
].join("\n"));

// Slide 2: Native 16D schema
const slide2 = presentation.slides.add();
addPageChrome(
  slide2,
  2,
  "FTNet Native Features — 16D Structured Pipeline Context",
  "Stage 7 Native Schema v1.1 — Reviewed Candidate",
  { titleWidth: 1150, titleFontSize: 31.5 },
);
addRule(slide2, "slide-2-title-rule", 38, 93, 1204, C.line, 2);

function addStatusText(x, w, label, value, color) {
  addText(slide2, { name: `status-${label}`, x, y: 102, w, h: 18, text: `${label}: ${value}`, fontSize: 10.2, color, bold: true, align: "center" });
}
addStatusText(50, 300, "Native dimension", "FROZEN (16)", C.navy);
addStatusText(390, 390, "Native field schema", "REVIEWED CANDIDATE", C.purple);
addStatusText(824, 400, "YouTube Highlights verification", "NOT YET COMPLETE", C.orange);

function addFeatureCard({ name, x, y, w, h, title, count, fill, accent, rows, rowHeight, bodySize = 10.5 }) {
  const card = addBox(slide2, { name: `${name}-card`, x, y, w, h, fill: C.paper, stroke: C.line, strokeWidth: 1.1, radius: 15 });
  card.sendToBack();
  addRule(slide2, `${name}-top-accent`, x, y, w, accent, 6);
  addText(slide2, { name: `${name}-title`, x: x + 18, y: y + 13, w: w - 100, h: 24, text: title, fontSize: 17, color: accent, bold: true });
  addBox(slide2, { name: `${name}-count`, x: x + w - 68, y: y + 13, w: 48, h: 22, fill, stroke: accent, strokeWidth: 1, radius: 11, text: count, fontSize: 11, color: accent, bold: true, insets: { top: 2, right: 4, bottom: 2, left: 4 } });
  const startY = y + 43;
  rows.forEach((row, index) => {
    const rowY = startY + index * rowHeight;
    if (index > 0) addRule(slide2, `${name}-row-rule-${index}`, x + 18, rowY - 2, w - 36, "#E9EDF3", 1);
    addText(slide2, { name: `${name}-row-label-${index}`, x: x + 18, y: rowY, w: w - 36, h: 17, text: `${row.id}. ${row.field} · ${row.zh}`, fontSize: 11.7, color: C.ink, bold: true, typeface: CJK_FONT });
    addText(slide2, { name: `${name}-row-body-${index}`, x: x + 18, y: rowY + 16, w: w - 36, h: rowHeight - 18, text: row.body, fontSize: bodySize, color: C.muted, valign: "top", typeface: CJK_FONT });
  });
}

const retrievalRows = [
  { id: 0, field: "retrieval_support_ratio", zh: "检索支持比例", body: "不同 Qwen chunk 对该帧属于候选高光的支持一致性。Conditional / variance gate。" },
  { id: 1, field: "candidate_time_from_start_log", zh: "距候选区间起点时间", body: "当前帧距离粗召回 candidate 起点的对数时间距离。" },
  { id: 2, field: "candidate_time_to_end_log", zh: "距候选区间终点时间", body: "当前帧距离 candidate 终点的对数时间距离。" },
  { id: 3, field: "candidate_duration_log", zh: "候选区间时长", body: "当前粗高光 candidate 的对数持续时间。" },
  { id: 4, field: "normalized_candidate_position", zh: "候选区间内归一化位置", body: "当前帧位于 candidate 内的位置。0 = 起点，1 = 终点。" },
];
const subjectRows = [
  { id: 5, field: "subject_track_present", zh: "主体轨迹存在标志", body: "当前帧是否存在有效的主要主体 / track。" },
  { id: 6, field: "subject_selection_confidence", zh: "主体选择置信度", body: "LOC 对当前主要主体选择的置信程度。" },
  { id: 7, field: "subject_selection_margin", zh: "主体选择置信裕度", body: "第一候选主体与第二候选主体之间的置信差距。" },
  { id: 8, field: "bbox_area_ratio", zh: "主体框面积比例", body: "主体 bbox 占原始画面面积的比例。" },
  { id: 9, field: "subject_scale_dynamics", zh: "主体尺度变化率", body: "连续帧间主体面积 / 尺度变化的时间动态。" },
  { id: 10, field: "subject_frame_offset", zh: "主体相对画面中心偏移", body: "主体中心距离原始 frame center 的归一化距离。" },
];
const compositionRows = [
  { id: 11, field: "geometric_context_retention", zh: "几何上下文保留比例", body: "TS 稳定后的 focus window 保留原始画面的几何范围。仅表示几何上下文，不表示 blur / attenuation 强度。" },
];
const temporalRows = [
  { id: 12, field: "bbox_continuity_iou", zh: "主体框连续性 IoU", body: "当前帧与前一真实连续帧主体 bbox 的 IoU。" },
  { id: 13, field: "continuity_valid", zh: "连续性有效标志", body: "bbox continuity 是否来自合法连续 source pair。" },
  { id: 14, field: "stabilized_focus_velocity", zh: "稳定聚焦中心运动速度", body: "TS 后 focus center 的归一化运动速度。" },
  { id: 15, field: "confidence_persistence", zh: "主体置信持续比例", body: "局部 ±0.5 秒窗口内存在有效主体的帧比例。" },
];

addFeatureCard({
  name: "retrieval", x: 35, y: 128, w: 597, h: 267,
  title: "A. Retrieval Context", count: "5D", fill: C.blueFill, accent: C.blueDark,
  rows: retrievalRows, rowHeight: 42, bodySize: 10.3,
});
addFeatureCard({
  name: "subject", x: 648, y: 128, w: 597, h: 267,
  title: "B. Subject / Geometry", count: "6D", fill: C.orangeFill, accent: C.orange,
  rows: subjectRows, rowHeight: 35.2, bodySize: 9.8,
});
addFeatureCard({
  name: "composition", x: 35, y: 410, w: 597, h: 132,
  title: "C. Composition", count: "1D", fill: C.purpleFill, accent: C.purple,
  rows: compositionRows, rowHeight: 58, bodySize: 10.3,
});
addText(slide2, { name: "cmp-scope-note", x: 54, y: 506, w: 558, h: 25, text: "Generic CMP supports crop, zoom, blur, attenuation, foreground emphasis, and fallback.", fontSize: 9.2, color: C.faint, align: "center" });
addFeatureCard({
  name: "temporal", x: 648, y: 410, w: 597, h: 165,
  title: "D. Temporal Stability", count: "4D", fill: C.tealFill, accent: C.teal,
  rows: temporalRows, rowHeight: 28.5, bodySize: 9.3,
});

const summaryBand = addBox(slide2, {
  name: "native-summary-band", x: 35, y: 590, w: 1210, h: 88,
  fill: C.navy, stroke: C.navy, strokeWidth: 0, radius: 16,
});
summaryBand.sendToBack();
addText(slide2, {
  name: "native-not-gt", x: 55, y: 601, w: 370, h: 56,
  text: "Native 16D does NOT represent ground truth.\n16维 Native 特征不是标签，而是 VHiCraFT pipeline 的结构化上下文。",
  fontSize: 13.2, color: C.paper, bold: true, typeface: CJK_FONT,
});
addText(slide2, {
  name: "native-summary-flow", x: 445, y: 598, w: 500, h: 63,
  text: "16D  →  Linear 16→32  →  Fusion with Visual96 + Delta96\n224D  →  128D temporal representation  →  Y Head",
  fontSize: 14, color: "#DCE7FF", bold: true, align: "center",
});
addText(slide2, {
  name: "native-summary-status", x: 970, y: 598, w: 250, h: 66,
  text: "Structure: FROZEN\nField schema: REVIEWED CANDIDATE\nReal-data verification: NOT YET COMPLETE",
  fontSize: 10.6, color: "#C8D6EA", bold: true, align: "right",
});

slide2.speakerNotes.textFrame.setText([
  `Source of truth: ${SPEC_PATH}`,
  "Native schema status: Stage7-NativeSchema-v1.1 REVIEWED_CANDIDATE.",
  "Native structure is frozen at dim=16 and 16→32, but YouTube Highlights real-data verification is not complete.",
  "Composition field 11 is geometric_context_retention only. It does not represent blur or attenuation strength.",
].join("\n"));

// Export, validate, reopen, render, and deliver
const candidatePath = path.join(stagingDir, "FTNet_Flowchart_candidate.pptx");
const inspectPath = path.join(buildDir, "FTNet_Flowchart.inspect.ndjson");
const layout1Path = path.join(buildDir, "FTNet_Flowchart.slide-1.layout.json");
const layout2Path = path.join(buildDir, "FTNet_Flowchart.slide-2.layout.json");

await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const initialInspect = await presentation.inspect({
  kind: "slide,textbox,shape,notes,layout",
  maxChars: 120000,
});
await fs.writeFile(inspectPath, initialInspect.ndjson, "utf8");
await fs.writeFile(layout1Path, await (await slide1.export({ format: "layout" })).text(), "utf8");
await fs.writeFile(layout2Path, await (await slide2.export({ format: "layout" })).text(), "utf8");

const requirements = {
  explicitTotalSlideCount: 2,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "design",
  families: [FONT, CJK_FONT],
  scriptFonts: { ea: CJK_FONT },
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: validatedPptx,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `FTNet_Flowchart_Editable_${buildToken}.pptx.validation.json`),
});

await fs.copyFile(validatedPptx, FINAL_PPTX);
await fs.unlink(validatedPptx);

const reopened = await PresentationFile.importPptx(await FileBlob.load(FINAL_PPTX));
const reopenedInspect = await reopened.inspect({
  kind: "slide,textbox,shape,notes,layout",
  maxChars: 140000,
});
const reopenedText = reopenedInspect.ndjson;
const requiredTokens = ["REQUIRED BRANCH", "Linear 16→32", "96 + 96 + 32", "Linear 224→128", "REVIEWED CANDIDATE"];
const bannedTokens = ["OPTIONAL NATIVE", "native_dim = 0", "fusion_dim = 192", "192 / 224", "p × q"];
for (const token of requiredTokens) {
  if (!reopenedText.includes(token)) throw new Error(`Required content missing after reopen: ${token}`);
}
for (const token of bannedTokens) {
  if (reopenedText.includes(token)) throw new Error(`Banned outdated content found after reopen: ${token}`);
}
await fs.writeFile(path.join(buildDir, "FTNet_Flowchart.reopened.inspect.ndjson"), reopenedText, "utf8");

const renderedFinalDir = path.join(buildDir, "rendered-final");
await fs.mkdir(renderedFinalDir, { recursive: true });
const renderResult = spawnSync(
  RUNTIME_PYTHON,
  [
    path.join(SKILL_DIR, "container_tools", "render_slides.py"),
    FINAL_PPTX,
    "--output_dir", renderedFinalDir,
    "--width", "2560",
    "--height", "1440",
  ],
  {
    shell: false,
    encoding: "utf8",
    env: {
      ...process.env,
      RUNTIME_NODE,
      RUNTIME_NODE_MODULES,
      RUNTIME_BIN_DIR,
    },
  },
);
if (renderResult.error || renderResult.signal || renderResult.status !== 0) {
  throw new Error(`Final PPTX render failed: ${renderResult.stderr || renderResult.stdout}`);
}
await fs.copyFile(path.join(renderedFinalDir, "slide-1.png"), FINAL_PREVIEW);
await fs.copyFile(path.join(renderedFinalDir, "slide-2.png"), FINAL_NATIVE_PREVIEW);

console.log(JSON.stringify({
  finalPptx: FINAL_PPTX,
  finalPreview: FINAL_PREVIEW,
  finalNativePreview: FINAL_NATIVE_PREVIEW,
  candidatePath,
  inspectPath,
  layout1Path,
  layout2Path,
  reopenedSlideCount: reopened.slides.items.length,
  reopenedAudit: { requiredTokens, bannedTokens, passed: true },
  renderOutput: renderResult.stdout.trim(),
  finalization: result,
}, null, 2));
