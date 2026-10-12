from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / ".patent_flowchart_work"
FIG_DIR = WORK_DIR / "figures"
OUTPUT = ROOT / "deliverables" / "按专利流程图样式修订版_建筑内局部证据与跨事件关系记忆建筑损毁评估.docx"

CANVAS_W = 2000
CANVAS_H = 2300
INK = "#000000"
PAPER = "#FFFFFF"


def font_path(*, bold: bool = False) -> str:
    candidates = (
        [Path(r"C:\Windows\Fonts\msyhbd.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]
        if bold
        else [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simsun.ttc")]
    )
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No Chinese font found")


REGULAR_FONT = font_path()
BOLD_FONT = font_path(bold=True)


def img_font(size: int, *, bold: bool = False):
    return ImageFont.truetype(BOLD_FONT if bold else REGULAR_FONT, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font, spacing: int = 12) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    return box[2] - box[0], box[3] - box[1]


def split_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """Wrap Chinese/Latin mixed text by rendered width without splitting logical tokens."""
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if current and text_size(draw, trial, font)[0] > max_width:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = trial
    if current:
        lines.append(current.rstrip())
    return "\n".join(lines)


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_xy: tuple[int, int, int, int],
    *,
    max_size: int = 44,
    min_size: int = 28,
    bold: bool = False,
) -> tuple[str, ImageFont.FreeTypeFont]:
    x1, y1, x2, y2 = box_xy
    inner_w = x2 - x1 - 90
    inner_h = y2 - y1 - 56
    for size in range(max_size, min_size - 1, -1):
        font = img_font(size, bold=bold)
        wrapped = split_text(draw, text, font, inner_w)
        tw, th = text_size(draw, wrapped, font)
        if tw <= inner_w and th <= inner_h:
            return wrapped, font
    font = img_font(min_size, bold=bold)
    return split_text(draw, text, font, inner_w), font


def draw_box(
    draw: ImageDraw.ImageDraw,
    box_xy: tuple[int, int, int, int],
    text: str,
    *,
    max_font: int = 44,
    min_font: int = 28,
    bold: bool = False,
    border_width: int = 5,
) -> None:
    draw.rectangle(box_xy, fill=PAPER, outline=INK, width=border_width)
    wrapped, font = fit_wrapped_text(
        draw, text, box_xy, max_size=max_font, min_size=min_font, bold=bold
    )
    tw, th = text_size(draw, wrapped, font)
    x1, y1, x2, y2 = box_xy
    draw.multiline_text(
        ((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 - 4),
        wrapped,
        font=font,
        fill=INK,
        spacing=12,
        align="center",
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int = 5,
    head: int = 24,
) -> None:
    draw.line([start, end], fill=INK, width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    left = (x2 - ux * head + px * head * 0.48, y2 - uy * head + py * head * 0.48)
    right = (x2 - ux * head - px * head * 0.48, y2 - uy * head - py * head * 0.48)
    draw.polygon([(x2, y2), left, right], fill=INK)


def draw_poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    width: int = 5,
) -> None:
    draw.line(points, fill=INK, width=width, joint="curve")
    draw_arrow(draw, points[-2], points[-1], width=width)


def cubic_bezier(p0, p1, p2, p3, count=36):
    points = []
    for i in range(count + 1):
        t = i / count
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def draw_step_label(
    draw: ImageDraw.ImageDraw,
    box_xy: tuple[int, int, int, int],
    step: str,
    *,
    label_x: int = 1860,
) -> None:
    x1, y1, x2, y2 = box_xy
    y = (y1 + y2) / 2
    start = (x2, y)
    gap = max(label_x - x2, 120)
    bend = (x2 + gap * 0.32, y)
    end_y = y - 55
    step_font = img_font(52)
    step_width = text_size(draw, step, step_font)[0]
    line_end_x = label_x - step_width / 2 - 24
    curve = cubic_bezier(
        bend,
        (x2 + gap * 0.60, y),
        (line_end_x - gap * 0.18, end_y),
        (line_end_x, end_y),
    )
    draw.line([start, bend], fill=INK, width=4)
    draw.line(curve, fill=INK, width=4)
    draw.text((label_x, end_y), step, font=step_font, fill=INK, anchor="mm")


def draw_vertical_steps(
    filename: str,
    steps: list[tuple[str, str]],
    *,
    top: int = 80,
    bottom: int = 90,
    box_left: int = 100,
    box_right: int = 1570,
    max_font: int = 44,
    min_font: int = 28,
) -> Path:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), PAPER)
    draw = ImageDraw.Draw(image)
    count = len(steps)
    gap = 62
    usable = CANVAS_H - top - bottom - gap * (count - 1)
    box_h = usable // count
    boxes: list[tuple[int, int, int, int]] = []
    y = top
    for step, text in steps:
        box_xy = (box_left, y, box_right, y + box_h)
        draw_box(draw, box_xy, text, max_font=max_font, min_font=min_font)
        draw_step_label(draw, box_xy, step)
        boxes.append(box_xy)
        y += box_h + gap
    for upper, lower in zip(boxes, boxes[1:]):
        x = (upper[0] + upper[2]) / 2
        draw_arrow(draw, (x, upper[3]), (x, lower[1] - 6))
    path = FIG_DIR / filename
    image.save(path, dpi=(300, 300), optimize=True)
    return path


def draw_figure_6() -> Path:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), PAPER)
    draw = ImageDraw.Draw(image)

    common = [
        ("S601", "获取当前疑似损毁分数、历史损毁支持信息、关系类型和基础预测不确定度"),
        ("S602", "估计候选修正相对于保持基础预测的残差效用"),
        ("S603", "判断所述残差效用是否满足预设接受条件"),
    ]
    boxes: list[tuple[int, int, int, int]] = []
    y_positions = [90, 420, 750]
    for (step, text), y in zip(common, y_positions):
        box_xy = (135, y, 1505, y + 230)
        draw_box(draw, box_xy, text, max_font=46, min_font=31)
        draw_step_label(draw, box_xy, step)
        boxes.append(box_xy)
    for upper, lower in zip(boxes, boxes[1:]):
        x = (upper[0] + upper[2]) / 2
        draw_arrow(draw, (x, upper[3]), (x, lower[1] - 6))

    reject_box = (80, 1220, 760, 1485)
    accept_box = (1010, 1220, 1690, 1485)
    correct_box = (1010, 1740, 1690, 2020)
    draw_box(draw, reject_box, "不满足接受条件时，保持基础预测结果不变", max_font=39, min_font=29)
    draw_box(draw, accept_box, "满足接受条件时，筛选目标区域内原预测为完整类别的像素", max_font=37, min_font=28)
    draw_box(draw, correct_box, "将筛选出的像素选择性修正为损毁类别，并保持其他类别不变", max_font=37, min_font=28)
    draw_step_label(draw, reject_box, "S604", label_x=910)
    draw_step_label(draw, accept_box, "S605", label_x=1870)
    draw_step_label(draw, correct_box, "S606", label_x=1870)

    decision = boxes[-1]
    center_x = (decision[0] + decision[2]) / 2
    split_y = 1095
    draw.line([(center_x, decision[3]), (center_x, split_y)], fill=INK, width=5)
    draw_poly_arrow(draw, [(center_x, split_y), (420, split_y), (420, reject_box[1] - 6)])
    draw_poly_arrow(draw, [(center_x, split_y), (1350, split_y), (1350, accept_box[1] - 6)])
    draw.text((485, 1125), "否", font=img_font(38), fill=INK, anchor="mm")
    draw.text((1285, 1125), "是", font=img_font(38), fill=INK, anchor="mm")
    draw_arrow(draw, (1350, accept_box[3]), (1350, correct_box[1] - 6))

    path = FIG_DIR / "figure_6_late_correction.png"
    image.save(path, dpi=(300, 300), optimize=True)
    return path


def draw_all_figures() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(
        draw_vertical_steps(
            "figure_1_overall.png",
            [
                ("S1", "获取目标区域的灾前光学遥感影像和灾后合成孔径雷达影像，并进行预处理"),
                ("S2", "将两类影像输入基础建筑损毁评估网络，获得建筑位置、基础损毁类别、光学结构特征和雷达响应特征"),
                ("S3", "根据预测建筑实例设置局部区域中心，并将每个建筑像素的候选中心限制在同一建筑内"),
                ("S4", "在建筑内搜索雷达局部响应，以幅值受限的残差修正光学结构关联并形成区域证据"),
                ("S5", "聚合区域证据构建建筑内损毁证据结构，并由条件损毁专家生成当前修正建议"),
                ("S6", "排除同一灾害事件后检索跨事件关系记忆，形成一致、仅当前、仅历史或冲突关系"),
                ("S7", "估计候选修正相对于保持基础预测的残差效用"),
                ("S8", "仅在残差效用满足条件时将目标区域内原完整类别修正为损毁类别，否则保持基础预测"),
            ],
            max_font=39,
            min_font=27,
        )
    )
    results.append(
        draw_vertical_steps(
            "figure_2_building_regions.png",
            [
                ("S201", "根据建筑位置概率获得预测建筑掩膜，并分离相互独立的建筑实例"),
                ("S202", "根据每一建筑实例的面积，自适应确定该建筑的局部区域中心数量 K"),
                ("S203", "将各局部区域中心吸附至对应建筑实例内部的有效位置"),
                ("S204", "为每个建筑像素建立仅包含同一建筑内附近区域中心的候选集合"),
                ("S205", "将跨建筑候选关联概率设置为零，并将面积过小的局部区域与邻接区域合并"),
            ],
            top=135,
            bottom=135,
            max_font=46,
            min_font=31,
        )
    )
    results.append(
        draw_vertical_steps(
            "figure_3_cross_modal.png",
            [
                ("S301", "根据光学结构特征计算像素与候选局部区域中心之间的基础结构关联得分 Lstruct"),
                ("S302", "在同一预测建筑实例与预设邻域的交集中搜索与光学结构位置对应的雷达局部响应"),
                ("S303", "根据光学特征和候选雷达特征生成跨模态关系特征、匹配置信度和匹配不确定度"),
                ("S304", "利用所述关系特征生成变化残差，并通过初始值为零且幅值受限的系数 α 进行门控"),
                ("S305", "根据 Lstruct 与 α 乘变化残差之和，获得像素与局部区域中心之间的软关联 Q"),
            ],
            top=120,
            bottom=120,
            max_font=43,
            min_font=29,
        )
    )
    results.append(
        draw_vertical_steps(
            "figure_4_evidence_structure.png",
            [
                ("S401", "按照软关联聚合同一预测建筑内各局部区域的光学结构特征和匹配后的雷达响应特征"),
                ("S402", "计算跨模态关系统计量、匹配质量、局部强响应和区域面积占比"),
                ("S403", "通过温度控制加权计算可微软峰值，保留小面积局部强损毁证据"),
                ("S404", "聚合整体统计量、内部异质性、前 M 个关键局部区域表示和区域邻接关系"),
                ("S405", "形成用于条件损毁专家判别的建筑内损毁证据结构"),
            ],
            top=135,
            bottom=135,
            max_font=45,
            min_font=30,
        )
    )
    results.append(
        draw_vertical_steps(
            "figure_5_specialist_memory.png",
            [
                ("S501", "从基础预测中选择被预测为完整类别或完整类别与损毁类别满足混淆条件的疑难建筑"),
                ("S502", "将建筑内损毁证据结构输入条件损毁专家，获得当前修正建议和疑似损毁分数"),
                ("S503", "在跨灾害事件关系记忆库中进行建筑级粗检索，并排除属于同一灾害事件的历史建筑"),
                ("S504", "利用当前建筑与历史建筑的关键局部区域表示进行细粒度重排序"),
                ("S505", "根据重排序结果获得历史修正建议和历史损毁支持信息"),
                ("S506", "将当前修正建议与历史修正建议编码为一致、仅当前、仅历史或冲突关系"),
            ],
            top=110,
            bottom=110,
            max_font=41,
            min_font=28,
        )
    )
    results.append(draw_figure_6())
    return results


def build_document() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import build_patent_from_template as patent_builder

    patent_builder.FIG_DIR = FIG_DIR
    patent_builder.OUTPUT = OUTPUT
    patent_builder.build()


if __name__ == "__main__":
    paths = draw_all_figures()
    build_document()
    for path in paths:
        print(path)
    print(OUTPUT)
