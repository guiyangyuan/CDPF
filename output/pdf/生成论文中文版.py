#!/usr/bin/env python3
"""生成 TextConsistency 论文的完整中文阅读版（Markdown + PDF）。"""

from pathlib import Path
from html import escape
import shutil

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, LongTable,
    Table, TableStyle, KeepTogether
)


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "论文中文版.pdf"
MD_PATH = ROOT / "论文中文版.md"
ASSET_DIR = ROOT / "中文版资源"

FONT_PATHS = [
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/PingFang.ttc"),
]
FONT_PATH = next((path for path in FONT_PATHS if path.exists()), None)
if FONT_PATH is None:
    raise FileNotFoundError("未找到可用的中文字体。")

pdfmetrics.registerFont(TTFont("中文", str(FONT_PATH), subfontIndex=0))

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 17 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
PALE_BLUE = colors.HexColor("#EEF4FA")
MID_GRAY = colors.HexColor("#667085")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
LINE_GRAY = colors.HexColor("#C9D1D9")
WHITE = colors.white

styles = getSampleStyleSheet()
STYLE_TITLE = ParagraphStyle(
    "中文标题", fontName="中文", fontSize=20, leading=29,
    alignment=TA_CENTER, textColor=NAVY, spaceAfter=14
)
STYLE_SUBTITLE = ParagraphStyle(
    "副标题", fontName="中文", fontSize=10.2, leading=16,
    alignment=TA_CENTER, textColor=MID_GRAY, spaceAfter=10
)
STYLE_H1 = ParagraphStyle(
    "一级标题", fontName="中文", fontSize=14.2, leading=21,
    textColor=NAVY, spaceBefore=10, spaceAfter=7, keepWithNext=True
)
STYLE_H2 = ParagraphStyle(
    "二级标题", fontName="中文", fontSize=11.3, leading=18,
    textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True
)
STYLE_BODY = ParagraphStyle(
    "正文", fontName="中文", fontSize=9.35, leading=15.2,
    alignment=TA_LEFT, firstLineIndent=18, textColor=colors.HexColor("#202124"),
    spaceAfter=5
)
STYLE_BODY_NO_INDENT = ParagraphStyle(
    "正文无缩进", parent=STYLE_BODY, firstLineIndent=0
)
STYLE_BULLET = ParagraphStyle(
    "项目", fontName="中文", fontSize=9.2, leading=14.8,
    leftIndent=16, firstLineIndent=-10, spaceAfter=4, textColor=colors.HexColor("#202124")
)
STYLE_ABSTRACT = ParagraphStyle(
    "摘要", fontName="中文", fontSize=9.45, leading=15.5,
    alignment=TA_LEFT, firstLineIndent=18, backColor=PALE_BLUE,
    borderColor=LIGHT_BLUE, borderWidth=0.6, borderPadding=9, spaceAfter=8
)
STYLE_EQUATION = ParagraphStyle(
    "公式", fontName="中文", fontSize=10.0, leading=17,
    alignment=TA_CENTER, backColor=LIGHT_GRAY, borderPadding=5,
    spaceBefore=4, spaceAfter=6
)
STYLE_CAPTION = ParagraphStyle(
    "图表标题", fontName="中文", fontSize=8.2, leading=12.5,
    alignment=TA_CENTER, textColor=MID_GRAY, spaceBefore=4, spaceAfter=9
)
STYLE_NOTE = ParagraphStyle(
    "注释", fontName="中文", fontSize=7.6, leading=11.5,
    textColor=MID_GRAY, spaceAfter=7
)
STYLE_REF = ParagraphStyle(
    "参考文献", fontName="中文", fontSize=7.7, leading=11.5,
    leftIndent=14, firstLineIndent=-14, spaceAfter=3, textColor=colors.HexColor("#30343B")
)


def txt(value):
    return escape(str(value)).replace("\n", "<br/>")


def para(text, style=STYLE_BODY):
    return Paragraph(txt(text), style)


def heading(text, level=1):
    return Paragraph(txt(text), STYLE_H1 if level == 1 else STYLE_H2)


def equation(text):
    return Paragraph(txt(text), STYLE_EQUATION)


def bullet(text):
    return Paragraph("•  " + txt(text), STYLE_BULLET)


def table(data, widths, font_size=6.5, header_rows=1, align="CENTER"):
    rendered = []
    for row_idx, row in enumerate(data):
        row_style = ParagraphStyle(
            f"表格-{font_size}-{row_idx}", fontName="中文", fontSize=font_size,
            leading=font_size + 2.2, alignment=TA_CENTER,
            textColor=WHITE if row_idx < header_rows else colors.HexColor("#202124")
        )
        rendered.append([Paragraph(txt(cell), row_style) for cell in row])
    t = LongTable(rendered, colWidths=widths, repeatRows=header_rows, hAlign=align)
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), WHITE),
        ("FONTNAME", (0, 0), (-1, -1), "中文"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE_GRAY),
    ]
    for row_idx in range(header_rows, len(data)):
        if (row_idx - header_rows) % 2 == 1:
            commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), LIGHT_GRAY))
    t.setStyle(TableStyle(commands))
    return t


def figure(filename, caption, width=CONTENT_W * 0.96):
    path = ASSET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    max_height = 108 * mm
    if img.drawHeight > max_height:
        scale = max_height / img.drawHeight
        img.drawWidth *= scale
        img.drawHeight *= scale
    img.hAlign = "CENTER"
    return [img, Paragraph(txt(caption), STYLE_CAPTION)]


SECTIONS = [
    ("1 引言", [
        "CLIP、ALIGN 等大规模视觉语言模型从海量图文数据中学习共享表示，并能通过文本提示完成开放词汇识别[1,10]。LiT 进一步表明，在保持图像编码器锁定的情况下调整文本塔，仍可保留很强的零样本迁移能力[11]。这类能力推动了参数高效的下游适配方法：冻结编码器，只优化少量提示参数。CoOp 用连续可学习向量替代人工上下文词，在少样本识别中获得显著提升[2]；但对已见基础类别的适配，有时会削弱对未见新类别的迁移。CoCoOp 通过图像条件提示缓解这一问题[3]，后续方法则引入提示分布、视觉—文本耦合提示、知识引导、自正则化或残差分类器[5–8,12]。",
        "学习提示与描述性语言提供了互补的决策信息。学习到的上下文能高效适配有标签的基础样本，自然语言描述则保留了显式的属性、部件、场景与功能语义。CuPL 证明，语言模型生成的类别特定描述能够提升零样本识别[4]。我们的实验得到一个更具体的结论：冻结描述分支单独使用时，平均性能并未超过 CoOp；但两者后验概率结合后，互补错误能够改善基础类别与新类别之间的平衡。而且，这一改进高于使用普通类别名称提示进行同样融合的结果，说明收益不能仅由普通分类器平均来解释。",
        "基于这一互补结构，我们提出 TextConsistency。对每个类别，用冻结的 CLIP 文本编码器编码一组固定自然语言描述；先对归一化描述嵌入取均值，再归一化为类别原型。推理时，一条分支产生 CoOp 学习提示的后验概率，另一条分支产生冻结描述原型的后验概率，最终输出二者的算术平均。所有划分、随机种子和数据集统一使用 0.5 的固定全局融合系数，不进行基于验证集或数据集特定的调参。",
        "这一设计把语义互补带来的收益与更大规模可训练结构带来的收益分离开来。描述嵌入可以提前缓存，因此 TextConsistency 不增加可训练参数，推理时只增加一次图像—类别原型矩阵乘法、一次 softmax 和一次概率平均。它在保持 CoOp 训练流程和参数效率的同时，提高了对新类别的迁移。",
    ]),
    ("2 相关工作", []),
]


REFERENCES = [
    "[1] A. Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML, 2021, pp. 8748–8763.",
    "[2] K. Zhou, J. Yang, C. C. Loy, Z. Liu, Learning to Prompt for Vision-Language Models, International Journal of Computer Vision 130 (2022) 2337–2348.",
    "[3] K. Zhou, J. Yang, C. C. Loy, Z. Liu, Conditional Prompt Learning for Vision-Language Models, CVPR, 2022, pp. 16816–16825.",
    "[4] S. Pratt, I. Covert, R. Liu, A. Farhadi, What Does a Platypus Look Like? Generating Customized Prompts for Zero-Shot Image Classification, ICCV, 2023, pp. 15691–15701.",
    "[5] Y. Lu, J. Liu, Y. Zhang, Y. Liu, X. Tian, Prompt Distribution Learning, CVPR, 2022, pp. 5206–5215.",
    "[6] M. U. Khattak et al., MaPLe: Multi-modal Prompt Learning, CVPR, 2023, pp. 19113–19122.",
    "[7] H. Yao, R. Zhang, C. Xu, Visual-Language Prompt Tuning with Knowledge-Guided Context Optimization, CVPR, 2023, pp. 6757–6767.",
    "[8] M. U. Khattak et al., Self-regulating Prompts: Foundational Model Adaptation without Forgetting, ICCV, 2023, pp. 15190–15200.",
    "[9] R. Zhang et al., Tip-Adapter: Training-Free Adaption of CLIP for Few-Shot Classification, ECCV, 2022, pp. 493–510.",
    "[10] C. Jia et al., Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision, ICML, 2021, pp. 4904–4916.",
    "[11] X. Zhai et al., LiT: Zero-Shot Transfer With Locked-Image Text Tuning, CVPR, 2022, pp. 18123–18133.",
    "[12] T. Yu et al., Task Residual for Tuning Vision-Language Models, CVPR, 2023, pp. 10899–10909.",
    "[13] B. Zhu et al., Prompt-aligned Gradient for Prompt Tuning, ICCV, 2023, pp. 15659–15669.",
    "[14] A. Bulat, G. Tzimiropoulos, LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision and Language Models, CVPR, 2023, pp. 23232–23241.",
    "[15] G. Chen et al., PLOT: Prompt Learning with Optimal Transport for Vision-Language Models, ICLR, 2023.",
    "[16] M. Shu et al., Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models, NeurIPS 35, 2022.",
    "[17] J. A. Samadh et al., Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization, NeurIPS 36, 2023.",
    "[18] Z. Li et al., PromptKD: Unsupervised Prompt Distillation for Vision-Language Models, CVPR, 2024, pp. 26617–26626.",
    "[19] V. Udandarao, A. Gupta, S. Albanie, SuS-X: Training-Free Name-Only Transfer of Vision-Language Models, ICCV, 2023, pp. 2725–2736.",
    "[20] Z. Guo et al., CALIP: Zero-Shot Enhancement of CLIP with Parameter-Free Attention, AAAI 37(1) (2023) 746–754.",
    "[21] S. Menon, C. Vondrick, Visual Classification via Description from Large Language Models, ICLR, 2023.",
    "[22] Z. Novack et al., CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets, ICML, 2023, pp. 26342–26362.",
    "[23] K. Roth et al., Waffling Around for Performance: Visual Classification with Random Words and Broad Concepts, ICCV, 2023, pp. 15746–15757.",
    "[24] C. Liao, T. Tsiligkaridis, B. Kulis, Descriptor and Word Soups: Overcoming the Parameter Efficiency Accuracy Tradeoff for Out-of-Distribution Few-Shot Learning, CVPR, 2024, pp. 27015–27025.",
    "[25] M. Maniparambil et al., Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts, ICCV Workshops, 2023, pp. 262–271.",
    "[26] M. U. Khattak et al., Learning to Prompt with Text Only Supervision for Vision-Language Models, AAAI 39 (2025) 4230–4238.",
    "[27] G. Wu et al., Cascade Prompt Learning for Vision-Language Model Adaptation, ECCV, 2024, pp. 304–321.",
    "[28] F. Hao et al., Task-Aware Clustering for Prompting Vision-Language Models, CVPR, 2025, pp. 14745–14755.",
    "[29] H. Li et al., DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models, CVPR, 2025, pp. 25623–25632.",
    "[30] J. Sun et al., Cross-Modal Feature Alignment and MMD Improve Robustness of Prompt Tuning, WACV, 2025, pp. 4714–4724.",
]


TABLE_LOCAL = [
    ["方法", "UCF", "DTD", "Aircraft", "EuroSAT", "Flowers", "Caltech", "Pets", "Food", "Cars", "SUN", "ImageNet", "平均"],
    ["CoOp", "74.42", "63.79", "34.26", "67.52", "81.07", "95.95", "96.39", "90.51", "72.35", "77.05", "72.39", "75.06"],
    ["TextConsistency", "79.47", "70.99", "38.46", "76.65", "85.06", "96.09", "96.63", "91.30", "73.16", "79.74", "74.05", "78.33"],
]

TABLE_AVG = [
    ["方法", "基础类别", "新类别", "调和平均 HM"],
    ["CoOp", "81.39", "70.34", "75.06"],
    ["TextConsistency", "81.13", "76.00", "78.33"],
]

TABLE_PUBLISHED = [
    ["方法", "UCF", "DTD", "Aircraft", "EuroSAT", "Flowers", "Caltech", "Pets", "Food", "Cars", "SUN", "平均"],
    ["CuPL†", "76.40", "61.30", "33.20", "61.80", "76.10", "95.60", "95.80", "90.60", "68.80", "75.70", "73.53"],
    ["CoCoOp†", "77.64", "64.85", "27.74", "71.21", "81.71", "95.84", "96.43", "90.99", "72.01", "78.27", "75.67"],
    ["KgCoOp†", "79.65", "64.35", "34.83", "73.48", "83.65", "96.03", "96.18", "91.09", "73.36", "78.36", "77.10"],
    ["MaPLe†", "80.77", "68.16", "36.50", "82.35", "82.56", "96.02", "96.58", "91.38", "73.47", "79.75", "78.75"],
    ["PromptSRC†", "82.74", "71.75", "40.15", "82.32", "85.95", "96.02", "96.30", "91.10", "76.58", "80.52", "80.34"],
    ["TextConsistency", "79.47", "70.99", "38.46", "76.65", "85.06", "96.09", "96.63", "91.30", "73.16", "79.74", "78.75"],
]

TABLE_COMPONENTS = [
    ["方法", "UCF", "DTD", "Aircraft", "EuroSAT", "Flowers", "Caltech", "Pets", "Food", "Cars", "SUN", "ImageNet", "平均"],
    TABLE_LOCAL[1],
    ["仅描述分支", "76.46", "63.30", "34.83", "69.99", "76.69", "96.29", "95.27", "90.58", "67.53", "76.73", "72.77", "74.59"],
    ["Logit 融合", "78.76", "70.75", "38.31", "75.69", "83.73", "96.09", "96.64", "91.30", "72.86", "79.54", "74.01", "77.97"],
    ["概率融合（本文）", "79.47", "70.99", "38.46", "76.65", "85.06", "96.09", "96.63", "91.30", "73.16", "79.74", "74.05", "78.33"],
]

TABLE_CLASS = [
    ["方法", "基础类别", "新类别", "HM"],
    ["CoOp", "81.39", "70.34", "75.06"],
    ["仅类别名", "69.49", "74.31", "71.71"],
    ["CoOp + 类别名", "80.71", "73.80", "76.84"],
    ["仅类别描述", "72.01", "77.78", "74.59"],
    ["TextConsistency", "81.13", "76.00", "78.33"],
]

TABLE_EFF = [
    ["方法", "批大小", "延迟（毫秒/批）", "吞吐（图/秒）", "相对开销"],
    ["CoOp", "1", "18.827", "53.12", "0.00%"],
    ["TextConsistency", "1", "18.904", "52.90", "0.41%"],
    ["CoOp", "64", "90.128", "710.10", "0.00%"],
    ["TextConsistency", "64", "90.796", "704.88", "0.74%"],
]

APPENDIX = [
    ["数据集", "方法", "基础类别", "新类别", "种子级 HM", "种子1", "种子2", "种子3"],
    ["UCF101", "CoOp", "83.56±0.45", "67.08±2.90", "74.40±1.94", "72.48", "76.36", "74.36"],
    ["UCF101", "TextConsistency", "82.70±0.43", "76.49±1.13", "79.47±0.80", "78.81", "80.36", "79.24"],
    ["DTD", "CoOp", "77.78±0.92", "54.07±0.77", "63.79±0.77", "63.60", "63.13", "64.64"],
    ["DTD", "TextConsistency", "77.93±0.48", "65.18±1.82", "70.98±1.02", "70.28", "70.51", "72.14"],
    ["FGVC Aircraft", "CoOp", "37.54±0.92", "31.51±1.84", "34.25±1.31", "35.36", "32.80", "34.57"],
    ["FGVC Aircraft", "TextConsistency", "37.47±0.09", "39.49±1.27", "38.45±0.56", "38.83", "37.80", "38.72"],
    ["EuroSAT", "CoOp", "84.38±4.27", "56.28±13.40", "66.64±8.04", "65.77", "75.08", "59.07"],
    ["EuroSAT", "TextConsistency", "84.56±1.65", "70.10±6.04", "76.52±2.87", "74.82", "79.83", "74.91"],
    ["OxfordFlowers", "CoOp", "97.02±0.43", "69.62±3.03", "81.04±1.97", "82.57", "81.74", "78.82"],
    ["OxfordFlowers", "TextConsistency", "95.73±0.41", "76.52±1.72", "85.05±0.98", "85.70", "85.52", "83.92"],
    ["Caltech101", "CoOp", "98.02±0.36", "93.96±1.26", "95.94±0.52", "96.53", "95.74", "95.56"],
    ["Caltech101", "TextConsistency", "98.00±0.13", "94.25±1.30", "96.08±0.63", "96.79", "95.85", "95.61"],
    ["OxfordPets", "CoOp", "95.57±0.26", "97.22±0.48", "96.39±0.26", "96.28", "96.69", "96.19"],
    ["OxfordPets", "TextConsistency", "95.61±0.20", "97.67±0.36", "96.63±0.14", "96.61", "96.77", "96.50"],
    ["Food101", "CoOp", "90.10±0.23", "90.94±0.75", "90.51±0.46", "90.68", "90.00", "90.87"],
    ["Food101", "TextConsistency", "90.65±0.16", "91.95±0.16", "91.30±0.13", "91.37", "91.15", "91.38"],
    ["StanfordCars", "CoOp", "74.33±0.98", "70.47±1.17", "72.34±0.45", "71.97", "72.85", "72.20"],
    ["StanfordCars", "TextConsistency", "72.11±1.14", "74.25±0.72", "73.16±0.35", "72.78", "73.21", "73.47"],
    ["SUN397", "CoOp", "80.54±0.16", "73.85±0.63", "77.04±0.29", "77.05", "76.75", "77.33"],
    ["SUN397", "TextConsistency", "81.07±0.15", "78.46±0.15", "79.74±0.07", "79.72", "79.68", "79.82"],
    ["ImageNet", "CoOp", "76.47±0.06", "68.72±0.21", "72.39±0.13", "72.24", "72.48", "72.44"],
    ["ImageNet", "TextConsistency", "76.66±0.07", "71.62±0.08", "74.05±0.02", "74.04", "74.05", "74.07"],
]


def add_cover(story):
    story.extend([
        Spacer(1, 21 * mm),
        Paragraph("TextConsistency", STYLE_TITLE),
        Paragraph("学习提示与类别描述的无参数概率融合<br/>用于基础类别到新类别识别", STYLE_TITLE),
        Spacer(1, 6 * mm),
        Paragraph("英文题目：Parameter-Free Probability Fusion of Learned Prompts and Class Descriptions for Base-to-Novel Recognition", STYLE_SUBTITLE),
        Spacer(1, 10 * mm),
    ])
    box_data = [[Paragraph("中文阅读版", ParagraphStyle("封面框", fontName="中文", fontSize=15, leading=22, alignment=TA_CENTER, textColor=NAVY))],
                [Paragraph("用于作者自查与导师审阅；正式投稿以英文版为准。", STYLE_SUBTITLE)],
                [Paragraph("作者、导师、学校与邮箱信息待按英文投稿版填写。", STYLE_SUBTITLE)]]
    box = Table(box_data, colWidths=[CONTENT_W * 0.78])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, LIGHT_BLUE),
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([box, Spacer(1, 24 * mm)])
    story.append(Paragraph("版本日期：2026 年 9 月 19 日", STYLE_SUBTITLE))
    story.append(PageBreak())


def build_story():
    s = []
    add_cover(s)
    s.append(heading("摘要"))
    s.append(Paragraph(txt(
        "提示学习可以高效地把视觉语言模型适配到下游识别任务，但仅在基础类别上训练的提示可能丢失冻结模型原有的部分语义覆盖。本文研究一种无需增加可训练参数的恢复方法。TextConsistency 将 CoOp 学习提示产生的后验概率，与冻结的类别特定自然语言描述产生的第二个后验概率结合。每个类别内的描述嵌入先取平均，两条分支共享冻结的 CLIP 图像编码器，再以固定权重 0.5 融合类别概率。描述分支提前计算，不改变提示训练过程。我们在标准 16-shot 基础类别到新类别协议下，使用 CLIP ViT-B/16、11 个数据集和 3 个随机种子进行评估。TextConsistency 的平均调和均值为 78.33，而匹配 CoOp 实验为 75.06，并在 33 个种子级配对比较中全部取得正增益。ImageNet 上，新类别准确率由 68.72 提高到 71.62，HM 由 72.39 提高到 74.05。11 数据集的类别名称对照达到 76.84 HM，说明收益不能用 CoOp 与普通零样本分类器的平均完全解释。在 RTX 4090 上，融合预测器不增加可训练参数，批大小 64 时推理延迟仅增加 0.74%。结果表明，固定概率融合能够高效结合任务适配信息与冻结语义先验。"
    ), STYLE_ABSTRACT))
    s.append(para("关键词：视觉语言模型；提示学习；少样本分类；基础类别到新类别泛化；类别描述；概率融合", STYLE_BODY_NO_INDENT))

    for sec, paragraphs in SECTIONS:
        s.append(heading(sec))
        for item in paragraphs:
            s.append(para(item))

    s.extend([
        heading("2.1 视觉语言提示学习", 2),
        para("CLIP 通过比较归一化图像表示与类别提示的文本表示完成分类[1]。CoOp 在冻结两个编码器的同时学习连续上下文词[2]，但基础类别优化可能导致提示特化，降低新类别表达能力。CoCoOp 将提示与输入图像关联，从而提高类别迁移和数据集迁移能力[3]。"),
        para("后续研究从多个方向正则化或扩展提示空间。ProDA 学习提示分布而非单一点估计[5]；PLOT 学习多个提示，并借助最优传输与视觉特征对齐[15]；ProGrad 仅接受不与预训练知识冲突的梯度[13]；LASP 通过文本到文本监督减轻基础类别过拟合[14]。MaPLe 在视觉和语言分支的多个深度加入耦合提示[6]；KgCoOp 以人工提示的知识约束可学习上下文[7]；PromptSRC 使用相互一致、自集成和文本多样性保留通用表示[8]；TaskRes 则在冻结文本分类器上增加任务残差[12]。PromptKD 利用存储的类别向量和无标签域图像，从更大的教师模型蒸馏知识[18]。"),
        para("近期工作进一步引入级联结构或额外监督。CasPL 分两个级联阶段学习增强提示和适配提示[27]；TAC 从少量聚类中心构建任务感知前置上下文[28]；DPC 用平行提示分别优化基础类别和新类别目标[29]；CRAFT 对齐跨模态相对表示，并用最大均值差异约束分布[30]；ProText 则只用语言模型文本监督学习可迁移提示[26]。与这些更强的学习式适配不同，TextConsistency 保持 CoOp 训练不变，不增加可训练参数，只在推理时加入一个缓存文本分类器。"),
        para("提示适配也可以在测试时进行。TPT 对单张测试图像的增强视图做熵最小化[16]，PromptAlign 在此基础上加入分布对齐[17]。这些方法面向测试时域偏移，需要逐样本优化。TextConsistency 面向有监督的基础到新类别设置，推理过程确定且无需更新。"),
        heading("2.2 类别描述与免训练适配", 2),
        para("自然语言能够表达短模板中缺失的类别属性。CuPL 用语言模型生成定制视觉描述完成零样本分类[4]；ProText 随后发布这些描述资源，并将其作为仅文本监督用于提示学习[26]。描述式分类直接预测可解释的视觉属性[21]，CHiLS 把粗粒度类别展开为分层标签集合[22]，GPT-4 生成的视觉描述也被用于构建细粒度 CLIP 分类器[25]。与此同时，WaffleCLIP 发现，即使替换为随机词和宽泛概念，描述集成的一部分增益仍然存在[23]；Descriptor/Word Soups 则研究如何从少样本数据中选择紧凑文本集成[24]。因此，本文设置直接的类别名称融合对照，不预设所有增益都来自描述语义。"),
        para("免训练适配提供了另一条保留预训练知识的路线。Tip-Adapter 使用少样本视觉特征缓存[9]，SuS-X 用支持集完成仅类别名称的迁移[19]，CALIP 引入无参数跨模态注意力[20]。TextConsistency 与它们可以互补：冻结分支来自文本，学习分支保持标准 CoOp，最后使用固定后验平均。"),
        heading("3 方法"),
        heading("3.1 问题设定", 2),
        para("将类别集合划分为互不重叠的基础类别 C_b 与新类别 C_n。每个基础类别使用 16 张有标签图像训练提示；测试时分别评估基础类别和新类别。报告基础准确率 A_b、新类别准确率 A_n 以及二者的调和平均："),
        equation("H = 2 A_b A_n / (A_b + A_n)"),
        para("调和平均会惩罚只提升一个划分、却严重损害另一个划分的模型。CLIP 包含图像编码器 E_v 与文本编码器 E_t。对图像 x，其归一化特征为："),
        equation("v(x) = E_v(x) / ‖E_v(x)‖₂"),
        para("全部实验中，两个编码器始终保持冻结。"),
        heading("3.2 学习提示分支", 2),
        para("可训练分支采用 CoOp。对类别 c，一个提示由 M 个可学习上下文向量与分词后的类别名称组成，其归一化文本表示为："),
        equation("tᵖ_c = E_t([q₁,…,q_M,c]) / ‖E_t([q₁,…,q_M,c])‖₂"),
        para("令冻结的 CLIP 温度尺度为 s = exp(γ)，学习提示分支的 logit 与后验概率为："),
        equation("zᵖ_c(x) = s · v(x)ᵀ tᵖ_c"),
        equation("pᵖ_c(x) = exp(zᵖ_c(x)) / Σₖ exp(zᵖ_k(x))"),
        para("仅上下文向量参与优化，训练目标为基础类别图像上的标准交叉熵。"),
        heading("3.3 冻结描述分支", 2),
        para("每个类别 c 使用固定的自然语言描述集合 D_c={d_c,1,…,d_c,m}。本文复用 ProText 官方仓库公开的文本文件[26]。除 EuroSAT 外，其余基准可追溯到 CuPL 的完整提示资源[4]。CuPL 报告使用 GPT-3 DaVinci-002：不同数据集采用 2–9 条视觉描述指令，每条指令生成 10 个结果，温度 0.99，最大 50 个 token，并以句号作为停止符。上游未公开确切生成日期。EuroSAT 文件与 ProText 版本逐字节一致，但上游没有记录其精确语言模型版本、温度、日期和采样数量。本文实验不再调用语言模型。"),
        para("处理时先规范类别键，保持源文件顺序，删除空字符串和完全重复字符串，每类最多保留前 20 条描述；经 CLIP 分词后完全相同的序列也会去重。本地只做了类别键规范、删除一个类似文件名的字符串、修正一个首字符笔误。任何描述都没有依据验证集或测试集准确率筛选。仓库提供文件哈希和机器可读的差异审计。"),
        para("每条描述先独立编码并归一化："),
        equation("u_c,j = E_t(d_c,j) / ‖E_t(d_c,j)‖₂"),
        para("随后对同类描述嵌入求平均，再归一化为类别原型："),
        equation("ū_c = [(1/m_c) Σⱼ u_c,j] / ‖(1/m_c) Σⱼ u_c,j‖₂"),
        para("描述分支的 logit 和后验概率为："),
        equation("zᵈ_c(x) = s · v(x)ᵀ ū_c"),
        equation("pᵈ_c(x) = exp(zᵈ_c(x)) / Σₖ exp(zᵈ_k(x))"),
        para("描述原型全部提前计算并缓存。该分支没有可训练参数，也不改变 CoOp 的训练过程。"),
        heading("3.4 固定概率融合", 2),
        para("最终后验为学习提示后验与描述后验的固定加权平均："),
        equation("pᵗᶜ_c(x) = (1−α) pᵖ_c(x) + α pᵈ_c(x)，其中 α = 0.5"),
        para("预测类别为使 pᵗᶜ_c(x) 最大的 c。所有划分、种子和数据集统一采用 0.5，不使用验证集选择权重，也不为单个数据集调参。相较 logit 平均，概率平均允许两条分支先各自形成归一化类别分布，再进行融合。"),
    ])
    s.extend(figure("图1-方法框架.png", "图 1  TextConsistency 框架。冻结的 CLIP 图像表示分别与学习得到的 CoOp 分类器、缓存的类别描述分类器计算相似度。只有 CoOp 上下文词参与训练，两个独立归一化的后验以固定系数 0.5 平均。"))
    s.extend([
        heading("3.5 参数与计算成本", 2),
        para("相较 CoOp，TextConsistency 增加 0 个可训练参数。在 ViT-B/16 实现中，CoOp 共训练 2,048 个上下文参数。在 UCF101 的效率测量设置中，缓存的类别描述张量占 0.0977 MiB。推理时额外操作仅为一次图像—原型矩阵乘法、一次 softmax 与一次概率平均；计时循环中不包含文本编码。"),
        heading("4 实验"),
        heading("4.1 数据集与评估协议", 2),
        para("实验覆盖标准 11 个基础到新类别基准：ImageNet、Caltech101、OxfordPets、StanfordCars、OxfordFlowers、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT 和 UCF101，涵盖通用物体、细粒度类别、食品、飞机、场景、纹理、遥感与人体动作。采用 CoOp/CoCoOp 公开类别划分[2,3]。ImageNet 审计确认训练集 1,281,167 张图像、验证集 50,000 张图像，基础/新类别各 500 类。"),
        para("所有可训练方法都使用每个基础类别 16 个样本，以及随机种子 1、2、3。先在三个种子上分别平均基础准确率与新类别准确率，再按调和平均公式得到数据集级 HM。跨数据集结果为 11 个数据集级指标的等权宏平均。冻结文本分支没有训练随机种子。"),
        heading("4.2 实现细节", 2),
        para("本地实验均使用 OpenAI CLIP ViT-B/16。CoOp 使用 4 个上下文词，批大小 16，SGD 学习率 0.002、动量 0.9，余弦调度，训练 20 个 epoch。输入图像调整为 224×224 并采用 CLIP 归一化。描述分支执行前述确定性过滤，每类最多使用 20 条描述，和学习分支共享冻结的 CLIP logit 尺度；描述嵌入只计算一次并缓存。融合系数始终为 0.5。"),
        heading("4.3 基线与报告边界", 2),
        para("11 数据集的严格匹配基线为 CoOp。为便于定位，另列 CuPL、CoCoOp、KgCoOp、MaPLe 和 PromptSRC 的论文报告结果[3,4,6–8]。文献数值用 † 标记。这些数值来自不同论文和实现，只用于文献定位，不能视为完全受控的复现排名。"),
        heading("5 结果"),
        heading("5.1 本地基础到新类别比较", 2),
        Paragraph("表 1  三种子匹配实验的逐数据集 HM（%）。", STYLE_CAPTION),
        table(TABLE_LOCAL, [53] + [36.7] * 12, 5.5),
        Spacer(1, 5),
        Paragraph("表 2  11 数据集的基础类别、新类别与 HM 宏平均（%）。", STYLE_CAPTION),
        table(TABLE_AVG, [105, 95, 95, 110], 7.4),
        para("TextConsistency 在 11 个数据集上取得 78.33 的平均 HM，高于 CoOp 的 75.06，提升 3.26 个百分点。ImageNet 上 HM 从 72.39 提高到 74.05，新类别准确率提高 2.90 点，基础类别准确率没有下降。"),
        para("从基础类别与新类别拆分看，基础类别宏平均由 81.39 轻微变化为 81.13，而新类别由 70.34 提高到 76.00。5.66 点的新类别增益是整体平衡改善的主要来源。11 个数据集和全部 33 个种子级配对比较均为正增益；EuroSAT、DTD、UCF101、FGVC Aircraft 与 OxfordFlowers 的改善最显著，而在已接近饱和的 Caltech101、OxfordPets、Food101 等数据集上增益较小。"),
    ])
    s.extend(figure("图2-十一数据集增益.png", "图 2  TextConsistency 相对匹配 CoOp 的逐数据集 HM 增益。所有柱均高于零；数值来自经过审计的三种子数据集汇总。", CONTENT_W * 0.91))
    s.extend([
        heading("5.2 与论文报告结果比较", 2),
        Paragraph("表 3  十个共同数据集上的论文报告 HM（%）。† 表示引用原论文数值。", STYLE_CAPTION),
        table(TABLE_PUBLISHED, [60] + [39] * 11, 5.35),
        para("在所有方法共同覆盖的十数据集上，TextConsistency 平均 HM 为 78.75，比 CuPL 高 5.22 点，比论文报告的 CoCoOp 高 3.09 点，比 KgCoOp 高 1.66 点；按表中精度与 MaPLe 的 78.75 持平，比 PromptSRC 低 1.59 点。由于这些对照来自不同实现，本表用于说明方法所处位置，不作为受控排名。"),
        heading("5.3 组件分析", 2),
        Paragraph("表 4  分支和融合方式消融的 HM（%）。", STYLE_CAPTION),
        table(TABLE_COMPONENTS, [64] + [35.8] * 12, 5.15),
        para("仅描述分支的平均 HM 为 74.59，低于 CoOp 的 75.06，说明它的价值来自互补性而非单分支更强。固定 logit 融合达到 77.97，表明合并两类语义信息本身已经有效；概率融合进一步达到 78.33，因此被采用为最终方法。所有数据集使用相同系数，差异不是由数据集特定权重搜索造成。描述分支对固定数据集和类别划分是确定性的，与三个 CoOp 检查点重复组合并不等同于三次独立的描述模型训练。"),
        PageBreak(),
        heading("5.4 类别名称对照与全局融合权重", 2),
        Paragraph("表 5  11 数据集的语义对照（宏平均，%）。", STYLE_CAPTION),
        table(TABLE_CLASS, [125, 90, 90, 90], 7.2),
        para("使用与主实验完全相同的检查点时，CoOp 与普通类别名称提示融合得到 76.84 HM，TextConsistency 达到 78.33，提升 1.49 点。描述融合在 11 个数据集中的 10 个更高，在 33 个数据集—种子配对中的 28 个更高。ImageNet 提升 1.10 点，StanfordCars 是唯一小幅下降的数据集（−0.18 点）。因此，类别描述在平均意义上提供了独立语义贡献，但不意味着每个数据集都必然改善。"),
    ])
    s.extend(figure("图4-语义对照与权重稳健性.png", "图 3  11 数据集上的语义对照与权重分析。（a）TextConsistency 相对 CoOp+类别名称概率平均的 HM 差值；（b）所有数据集共用同一个描述权重时的基础类别、新类别与 HM 宏平均。本文固定采用 α=0.5。"))
    s.extend([
        para("统一网格 α∈{0,0.25,0.5,0.75,1} 没有为单个数据集选择最优值。固定中点 α=0.5 取得最高宏平均 HM 78.33。两个中间混合点均优于任一单分支端点；随着 α 增大，基础类别准确率下降、新类别准确率提高，符合在任务适配与冻结语义覆盖之间取得平衡的设计目标。"),
        heading("5.5 跨种子稳定性", 2),
        para("每个种子都使用同一训练检查点比较 TextConsistency 与 CoOp，33 个配对 HM 差异全部为正。ImageNet 三个种子的增益分别为 +1.80、+1.57 和 +1.63。由于每个数据集仅有三个种子，本文把结果解释为方向稳定性的证据，不用小样本置信区间宣称很强的统计显著性。附录 A 给出完整的基础类别、新类别和种子级 HM。"),
    ])
    s.append(PageBreak())
    s.extend(figure("图3-十一数据集三种子配对增益.png", "图 4  11 个数据集、3 个种子的 TextConsistency−CoOp 配对 HM 增益。菱形为三个种子级 HM 差值的算术平均，竖线覆盖三个种子值。该均值可能与表 1 中“先平均基础/新类别准确率再计算 HM”的差值略有不同。全部 33 个比较均为正。", CONTENT_W * 0.89))
    s.extend([
        heading("5.6 推理效率", 2),
        Paragraph("表 6  单张 RTX 4090 上的 CUDA Event 推理测量。", STYLE_CAPTION),
        table(TABLE_EFF, [110, 70, 110, 100, 85], 7.0),
        para("每种设置先预热 20 次，再对相同的 UCF101 新类别检查点和输入张量计时 100 次。批大小为 64 时，TextConsistency 延迟从 90.128 毫秒增加到 90.796 毫秒，开销 0.74%。在当前分配器分辨率下，额外峰值显存没有变化。测量不包括模型加载、磁盘 I/O、CPU 预处理和一次性文本编码。"),
        heading("6 讨论与局限"),
        para("结果表明，固定后验融合能够恢复基础类别提示适配后未被充分利用的冻结语义信息。证据链包括：学习分支与冻结分支存在互补错误；描述融合把宏平均 HM 从 75.06 提高到 78.33；在同样全局权重与零新增可训练参数条件下，又比匹配的类别名称融合高 1.49 点。描述分支单独使用仍低于 CoOp，进一步说明收益来自互补，而非简单替换学习提示。"),
        para("本文仍有局限。第一，描述来自公开 CuPL/ProText 资源，尚未系统比较不同语言模型、生成指令和描述预算。第二，全局权重只在五个粗粒度点上分析，且仍使用主实验数据集，不能证明外部分布偏移下的校准稳健性。第三，效率仅在一张 RTX 4090 上测量，不能视为与硬件无关的延迟保证。"),
        para("公开仓库保存了固定数据划分、随机种子、配置、运行日志、源文件哈希、机器可读汇总和独立复算脚本。ImageNet 审计记录了评估器哈希，并核验训练集 1,281,167 张和验证集 50,000 张图像，使 11 数据集评估过程能够追踪。"),
        heading("7 结论"),
        para("本文提出 TextConsistency：对 CoOp 学习提示与冻结类别描述产生的类别概率进行平均，从而形成一个无额外可训练参数的扩展。该方法保持 CLIP 冻结，不改变提示训练，采用单一固定融合系数并缓存描述原型。在标准 11 个基础到新类别数据集上，它把匹配 CoOp 的平均 HM 从 75.06 提高到 78.33，33 个种子级比较均为正增益。组件消融和类别名称对照支持“描述语义互补”这一解释；RTX 4090 测量表明，批大小 64 时延迟开销低于 1%。因此，TextConsistency 提供了一种可复现、低成本的方式，在不扩大可训练模型的情况下改善新类别平衡。"),
        heading("数据与代码可用性"),
        para("实现、固定配置、结果表、审计脚本与实验汇总已发布于 GitHub：https://github.com/guiyangyuan/textconsistency_v2_tune 。仓库将本地实验结果与论文引用值分开，并提供机器可读的 CSV/JSON 汇总。类别描述文件可追溯到固定的 ProText 与 CuPL 提交，文件哈希、两处本地文本清理以及上游许可证位置记录在准备报告中。"),
        PageBreak(),
        heading("附录 A 逐数据集种子统计"),
        para("表 A1 给出主比较背后的完整三种子波动。标准差由每个数据集的三个匹配 CoOp 检查点计算。此处“种子级 HM”是三个原始 HM 的算术平均，可能与主表中“先平均基础/新类别准确率，再计算 HM”的结果略有不同；最后三列使每个配对比较均可直接核验。"),
        Paragraph("表 A1  CoOp 与 TextConsistency 的基础类别、新类别、种子级 HM（均值±标准差）及三个原始 HM。", STYLE_CAPTION),
        table(APPENDIX, [65, 82, 65, 65, 72, 50, 50, 50], 5.65),
        PageBreak(),
        heading("参考文献"),
    ])
    for ref in REFERENCES:
        s.append(Paragraph(txt(ref), STYLE_REF))
    return s


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LIGHT_BLUE)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, PAGE_H - 12 * mm, PAGE_W - RIGHT, PAGE_H - 12 * mm)
    canvas.setFont("中文", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(LEFT, PAGE_H - 9.5 * mm, "TextConsistency 中文阅读版")
    canvas.drawRightString(PAGE_W - RIGHT, 9 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def markdown_table(rows):
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def write_markdown():
    md = """# TextConsistency：学习提示与类别描述的无参数概率融合，用于基础类别到新类别识别

> 中文阅读版，用于作者自查与导师审阅；正式投稿以英文版为准。作者、导师、学校与邮箱信息待按英文投稿版填写。

## 摘要

提示学习可以高效地把视觉语言模型适配到下游识别任务，但仅在基础类别上训练的提示可能丢失冻结模型原有的部分语义覆盖。本文研究一种无需增加可训练参数的恢复方法。TextConsistency 将 CoOp 学习提示产生的后验概率，与冻结的类别特定自然语言描述产生的第二个后验概率结合。每个类别内的描述嵌入先取平均，两条分支共享冻结的 CLIP 图像编码器，再以固定权重 0.5 融合类别概率。描述分支提前计算，不改变提示训练过程。

我们在标准 16-shot 基础类别到新类别协议下，使用 CLIP ViT-B/16、11 个数据集和 3 个随机种子进行评估。TextConsistency 的平均调和均值为 78.33，而匹配 CoOp 实验为 75.06，并在 33 个种子级配对比较中全部取得正增益。ImageNet 上，新类别准确率由 68.72 提高到 71.62，HM 由 72.39 提高到 74.05。11 数据集的类别名称对照达到 76.84 HM，说明收益不能用 CoOp 与普通零样本分类器的平均完全解释。在 RTX 4090 上，融合预测器不增加可训练参数，批大小 64 时推理延迟仅增加 0.74%。

**关键词：** 视觉语言模型；提示学习；少样本分类；基础类别到新类别泛化；类别描述；概率融合

## 1 引言

CLIP、ALIGN 等大规模视觉语言模型从海量图文数据中学习共享表示，并能通过文本提示完成开放词汇识别[1,10]。LiT 进一步表明，在保持图像编码器锁定的情况下调整文本塔，仍可保留很强的零样本迁移能力[11]。CoOp 用连续可学习向量替代人工上下文词，在少样本识别中获得显著提升[2]；但对已见基础类别的适配，有时会削弱对未见新类别的迁移。CoCoOp 通过图像条件提示缓解这一问题[3]，后续方法则引入提示分布、视觉—文本耦合提示、知识引导、自正则化或残差分类器[5–8,12]。

学习提示与描述性语言提供互补信息。学习上下文适配基础样本，自然语言描述保留属性、部件、场景与功能语义。本文发现：冻结描述分支单独使用时平均性能并未超过 CoOp，但二者后验结合后能够改善基础/新类别平衡，而且高于普通类别名称提示的同类融合。

TextConsistency 对每个类别编码一组固定描述，平均归一化嵌入形成类别原型；推理时平均 CoOp 后验与描述后验。所有划分、种子和数据集统一使用 0.5，不进行验证集或数据集特定调参。描述嵌入可以缓存，因此不增加可训练参数，只增加一次矩阵乘法、softmax 和概率平均。

本文贡献包括：

- 提出学习提示后验与冻结类别描述后验的固定概率融合，无需增加 CoOp 之外的可训练参数。
- 通过组件与类别名称对照分离分支互补和普通类别名称集成；描述融合比类别名融合高 1.49 HM。
- 在标准 11 个数据集、3 个种子上统一使用 0.5，达到 78.33 平均 HM，33/33 配对增益为正。
- 报告 RTX 4090 实测推理开销，并区分本地结果与论文引用值。

## 2 相关工作

### 2.1 视觉语言提示学习

CoOp、CoCoOp、ProDA、PLOT、ProGrad、LASP、MaPLe、KgCoOp、PromptSRC、TaskRes 与 PromptKD 分别从可学习上下文、条件提示、提示分布、最优传输、梯度约束、文本监督、多模态深层提示、知识约束、自正则化、残差分类器和蒸馏等角度改进提示适配[2,3,5–8,12–15,18]。近期 CasPL、TAC、DPC、CRAFT 与 ProText 继续引入级联结构、任务感知聚类、双提示协作、跨模态对齐与纯文本监督[26–30]。TPT 和 PromptAlign 则在测试时逐样本更新提示[16,17]。TextConsistency 保持 CoOp 训练和推理的确定性，只加入一个缓存文本分类器。

### 2.2 类别描述与免训练适配

CuPL、描述式分类、CHiLS、GPT-4 描述分类器和 Descriptor/Word Soups 研究了丰富语言语义对零样本或少样本识别的作用[4,21,22,24,25]。WaffleCLIP 表明随机词也可能贡献部分集成收益[23]，因此本文必须加入类别名称融合对照。Tip-Adapter、SuS-X 与 CALIP 代表免训练适配路线[9,19,20]；TextConsistency 的冻结分支来自文本，并与标准 CoOp 互补。

## 3 方法

### 3.1 问题设定

类别集合划分为基础类别和新类别。每个基础类别使用 16 张有标签图像训练。报告基础准确率 A_b、新类别准确率 A_n 和调和平均：

`H = 2 A_b A_n / (A_b + A_n)`

图像归一化特征为 `v(x)=E_v(x)/‖E_v(x)‖₂`，视觉与文本编码器均保持冻结。

### 3.2 学习提示分支

CoOp 的类别文本表示为 `tᵖ_c=E_t([q₁,…,q_M,c])/‖E_t([q₁,…,q_M,c])‖₂`，对应 `zᵖ_c(x)=s·v(x)ᵀtᵖ_c`，再经 softmax 得到 `pᵖ_c(x)`。仅上下文向量通过基础类别交叉熵训练。

### 3.3 冻结描述分支

描述来自 ProText 官方仓库，除 EuroSAT 外可追溯到 CuPL。CuPL 使用 GPT-3 DaVinci-002、2–9 条数据集特定指令、每条 10 个生成、温度 0.99、50-token 上限和句号停止符。本文不再次调用语言模型。处理过程保持源顺序，删除空串、精确重复项和分词后重复项，每类最多取前 20 条；不按准确率筛选。

每条描述归一化后求均值并再次归一化：`ū_c=[(1/m_c)Σⱼu_c,j]/‖(1/m_c)Σⱼu_c,j‖₂`。类别原型提前缓存，描述分支没有可训练参数。

### 3.4 固定概率融合

`pᵗᶜ_c(x)=(1−α)pᵖ_c(x)+αpᵈ_c(x)，α=0.5`。全部实验共用这一固定全局系数。

![方法框架](中文版资源/图1-方法框架.png)

### 3.5 参数与计算成本

相较 CoOp，TextConsistency 增加 0 个可训练参数。ViT-B/16 的 CoOp 训练 2,048 个上下文参数；UCF101 测量中缓存描述张量占 0.0977 MiB。

## 4 实验设置

实验覆盖 ImageNet、Caltech101、OxfordPets、StanfordCars、OxfordFlowers、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT 和 UCF101。使用 16-shot、随机种子 1/2/3。先平均基础/新类别准确率，再计算数据集 HM；跨数据集使用等权宏平均。

本地实验均使用 OpenAI CLIP ViT-B/16。CoOp 使用 4 个上下文词、批大小 16、SGD 学习率 0.002、动量 0.9、余弦调度和 20 个 epoch。输入为 224×224。严格匹配基线是 CoOp；CuPL、CoCoOp、KgCoOp、MaPLe、PromptSRC 使用论文报告值并标记 †。

## 5 实验结果

### 5.1 本地基础到新类别比较

"""
    md += markdown_table(TABLE_LOCAL) + "\n\n"
    md += markdown_table(TABLE_AVG) + "\n\n"
    md += "TextConsistency 的 11 数据集平均 HM 为 78.33，相比 CoOp 的 75.06 提高 3.26 点。基础类别由 81.39 轻微变化为 81.13，新类别由 70.34 提高到 76.00；新类别 +5.66 是平衡改善的主要来源。11 个数据集、33 个种子配对全部为正增益。\n\n![逐数据集增益](中文版资源/图2-十一数据集增益.png)\n\n### 5.2 与论文报告结果比较\n\n"
    md += markdown_table(TABLE_PUBLISHED) + "\n\n"
    md += "十个共同数据集上，本文为 78.75，高于 CuPL、CoCoOp、KgCoOp，与 MaPLe 按表中精度持平，低于 PromptSRC 1.59 点。不同论文实现不完全一致，因此本表只用于定位。\n\n### 5.3 组件分析\n\n"
    md += markdown_table(TABLE_COMPONENTS) + "\n\n"
    md += "描述分支单独为 74.59，低于 CoOp；logit 融合为 77.97，概率融合为 78.33，说明收益来自互补且概率融合更合适。\n\n### 5.4 类别名称对照与全局权重\n\n"
    md += markdown_table(TABLE_CLASS) + "\n\n"
    md += "CoOp+类别名为 76.84，TextConsistency 为 78.33，差 1.49 点；描述融合在 10/11 数据集、28/33 数据集—种子配对中更高。统一 α 网格中，0.5 得到最高 HM 78.33。\n\n![语义对照与权重](中文版资源/图4-语义对照与权重稳健性.png)\n\n### 5.5 跨种子稳定性\n\n全部 33 个配对 HM 差异为正。ImageNet 三个种子增益为 +1.80、+1.57、+1.63。图中菱形是种子级 HM 差值平均，可能与主表先平均基础/新类别准确率再计算 HM 的差值略有不同。\n\n![三种子配对增益](中文版资源/图3-十一数据集三种子配对增益.png)\n\n### 5.6 推理效率\n\n"
    md += markdown_table(TABLE_EFF) + "\n\n"
    md += "批大小 64 时，延迟由 90.128 ms 增至 90.796 ms，开销 0.74%；没有新增可训练参数。\n\n## 6 讨论与局限\n\n证据支持固定概率融合能够恢复提示适配后未被充分利用的冻结语义：描述融合 HM 为 78.33，比 CoOp 高 3.26，也比同权重类别名融合高 1.49。局限包括：只使用公开 CuPL/ProText 描述；融合权重只做五点粗网格分析；效率只在一张 RTX 4090 上测量。\n\n公开仓库保存固定数据划分、随机种子、配置、日志、源文件哈希、机器可读汇总与独立复算脚本。\n\n## 7 结论\n\nTextConsistency 以固定权重平均学习提示与冻结类别描述的后验，不改变 CoOp 训练，也不增加可训练参数。11 数据集平均 HM 从 75.06 提高到 78.33，33/33 配对均为正；语义对照与组件消融支持描述互补这一解释。\n\n## 数据与代码可用性\n\n仓库：[guiyangyuan/textconsistency_v2_tune](https://github.com/guiyangyuan/textconsistency_v2_tune)。\n\n## 附录 A：逐数据集种子统计\n\n"
    md += markdown_table(APPENDIX) + "\n\n## 参考文献\n\n" + "\n\n".join(REFERENCES) + "\n"
    MD_PATH.write_text(md, encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    write_markdown()
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM, title="TextConsistency 论文中文版",
        author="TextConsistency authors", subject="中文阅读版"
    )
    doc.build(build_story(), onFirstPage=page_decor, onLaterPages=page_decor)
    test_dir = ROOT / "tmp"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print(f"已生成：{PDF_PATH}")
    print(f"已生成：{MD_PATH}")


if __name__ == "__main__":
    main()
