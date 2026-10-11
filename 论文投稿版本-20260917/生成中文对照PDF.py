#!/usr/bin/env python3
"""将中文对照稿生成一份可供阅读和核对的 A4 PDF。"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "论文投稿版_中文对照稿.md"
DEFAULT_BIB = ROOT / "最终提交源码" / "cas-refs.bib"
FIGURES = {
    "图1": ROOT / "图表预览" / "图1-方法框架.png",
    "图2": ROOT / "图表预览" / "图2-十一数据集增益.png",
    "图3": ROOT / "图表预览" / "图3-十一数据集消融稳定性效率.png",
}
CHINESE_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def inline_markup(text: str) -> str:
    """将本稿用到的少量 Markdown/LaTeX 行内语法转为 ReportLab 标记。"""
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(m.group(1).split(",")) + "]", text)
    text = text.replace(r"\%", "%").replace("&", "&amp;")
    text = html.escape(text, quote=False).replace("&amp;amp;", "&amp;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\$([^$]+)\$", r'<font name="Helvetica-Oblique">\1</font>', text)
    return text


def clean_tex(value: str) -> str:
    value = value.replace("{", "").replace("}", "")
    value = value.replace("--", "–").replace(r"\&", "&")
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    return " ".join(value.split())


def parse_bibtex(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"(?m)^@\w+\{([^,]+),", text))
    entries: list[dict[str, str]] = []
    for index, match in enumerate(starts):
        block = text[match.end() : starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        fields = {"key": match.group(1)}
        for field in ("author", "title", "booktitle", "journal", "year", "volume", "pages"):
            found = re.search(rf"(?ms)^\s*{field}\s*=\s*\{{(.*?)\}}\s*,?\s*$", block)
            if found:
                fields[field] = clean_tex(found.group(1))
        entries.append(fields)
    return entries


def make_styles():
    # 嵌入字体，避免在没有中文 CID 字体的 PDF 阅读器中丢字。
    pdfmetrics.registerFont(TTFont("CJK", CHINESE_FONT))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle", parent=base["Title"], fontName="CJK", fontSize=20,
            leading=29, alignment=TA_CENTER, textColor=colors.HexColor("#183153"),
            spaceAfter=9 * mm,
        ),
        "h1": ParagraphStyle(
            "ChineseH1", parent=base["Heading1"], fontName="CJK", fontSize=15,
            leading=21, textColor=colors.HexColor("#183153"), spaceBefore=7 * mm,
            spaceAfter=3 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ChineseH2", parent=base["Heading2"], fontName="CJK", fontSize=12,
            leading=17, textColor=colors.HexColor("#24527A"), spaceBefore=5 * mm,
            spaceAfter=2 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ChineseBody", parent=base["BodyText"], fontName="CJK", fontSize=9.5,
            leading=15.5, alignment=TA_JUSTIFY, firstLineIndent=2 * 9.5,
            spaceAfter=2.2 * mm, wordWrap="CJK",
        ),
        "abstract": ParagraphStyle(
            "ChineseAbstract", parent=base["BodyText"], fontName="CJK", fontSize=9.2,
            leading=15, alignment=TA_JUSTIFY, leftIndent=7 * mm, rightIndent=7 * mm,
            firstLineIndent=2 * 9.2, spaceAfter=2.2 * mm, wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "ChineseNote", parent=base["BodyText"], fontName="CJK", fontSize=8.5,
            leading=13.5, leftIndent=5 * mm, rightIndent=5 * mm, borderColor=colors.HexColor("#AAB7C4"),
            borderWidth=0.6, borderPadding=6, backColor=colors.HexColor("#F5F8FA"),
            spaceAfter=4 * mm, wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "ChineseEquation", parent=base["BodyText"], fontName="CJK", fontSize=9,
            leading=14, alignment=TA_CENTER, leftIndent=8 * mm, rightIndent=8 * mm,
            borderColor=colors.HexColor("#D9E2EA"), borderWidth=0.5, borderPadding=5,
            backColor=colors.HexColor("#FAFBFC"), spaceBefore=1.5 * mm, spaceAfter=3 * mm,
        ),
        "caption": ParagraphStyle(
            "ChineseCaption", parent=base["BodyText"], fontName="CJK", fontSize=8,
            leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#4A5560"),
            leftIndent=8 * mm, rightIndent=8 * mm, spaceBefore=1.5 * mm, spaceAfter=4 * mm,
            wordWrap="CJK",
        ),
        "ref": ParagraphStyle(
            "ChineseReference", parent=base["BodyText"], fontName="CJK", fontSize=6.4,
            leading=8.2, leftIndent=0, firstLineIndent=0, spaceAfter=0,
            wordWrap="CJK",
        ),
        "table_title": ParagraphStyle(
            "ChineseTableTitle", parent=base["BodyText"], fontName="CJK", fontSize=8.5,
            leading=12, alignment=TA_CENTER, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
    }


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                         topMargin=19 * mm, bottomMargin=17 * mm, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=self._page_decorations))

    @staticmethod
    def _page_decorations(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6DEE5"))
        canvas.setLineWidth(0.4)
        canvas.line(17 * mm, A4[1] - 13 * mm, A4[0] - 17 * mm, A4[1] - 13 * mm)
        canvas.setFont("CJK", 7.5)
        canvas.setFillColor(colors.HexColor("#5D6B78"))
        canvas.drawString(17 * mm, A4[1] - 10.5 * mm, "TextConsistency · 中文对照稿")
        canvas.drawRightString(A4[0] - 17 * mm, 9 * mm, f"第 {doc.page} 页")
        canvas.restoreState()


def make_table(rows: list[list[str]], styles) -> Table:
    column_count = len(rows[0])
    available = A4[0] - 34 * mm
    if column_count >= 12:
        first = 31 * mm
        widths = [first] + [(available - first) / (column_count - 1)] * (column_count - 1)
        font_size = 4.7
        leading = 6.2
    elif column_count >= 6:
        first = 35 * mm
        widths = [first] + [(available - first) / (column_count - 1)] * (column_count - 1)
        font_size = 6.4
        leading = 8
    else:
        first = 43 * mm
        widths = [first] + [(available - first) / (column_count - 1)] * (column_count - 1)
        font_size = 7.2
        leading = 9.2

    cell_style = ParagraphStyle(
        "TableCell", fontName="CJK", fontSize=font_size, leading=leading,
        alignment=TA_CENTER, wordWrap="CJK",
    )
    formatted = [[Paragraph(inline_markup(cell.strip()), cell_style) for cell in row] for row in rows]
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183153")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#6C879F")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#6C879F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def figure_flowables(key: str, caption: str, styles) -> list:
    path = FIGURES[key]
    with PILImage.open(path) as picture:
        width, height = picture.size
    max_width, max_height = A4[0] - 42 * mm, 92 * mm
    scale = min(max_width / width, max_height / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return [KeepTogether([image, Spacer(1, 1.5 * mm), Paragraph(inline_markup(caption), styles["caption"])])]


def markdown_story(source: Path, styles) -> list:
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph: list[str] = []
    in_equation = False
    equation: list[str] = []
    abstract_mode = False

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            selected = styles["abstract"] if abstract_mode else styles["body"]
            story.append(Paragraph(inline_markup(" ".join(part.strip() for part in paragraph)), selected))
            paragraph = []

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if stripped == "$$":
            flush_paragraph()
            if in_equation:
                content = "<br/>".join(html.escape(part) for part in equation)
                story.append(Paragraph(content, styles["equation"]))
                equation = []
                in_equation = False
            else:
                in_equation = True
            index += 1
            continue
        if in_equation:
            equation.append(stripped)
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Spacer(1, 12 * mm))
            story.append(Paragraph(inline_markup(stripped[2:]), styles["title"]))
            story.append(Paragraph("中文完整对照版 · 十一数据集实验稿", styles["caption"]))
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            title = stripped[3:]
            abstract_mode = title == "摘要"
            if title != "摘要":
                abstract_mode = False
            story.append(Paragraph(inline_markup(title), styles["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h2"]))
            index += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quote_lines.append(lines[index].lstrip()[1:].strip())
                index += 1
            story.append(Paragraph(inline_markup(" ".join(quote_lines)), styles["note"]))
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)]
            story.extend([make_table(rows, styles), Spacer(1, 3.5 * mm)])
            continue
        figure_match = re.match(r"\*\*(图[123])[^*]*\*\*(.*)", stripped)
        if figure_match:
            flush_paragraph()
            story.extend(figure_flowables(figure_match.group(1), stripped, styles))
            index += 1
            continue
        if re.match(r"\*\*表\d", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), styles["table_title"]))
            index += 1
            continue
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return story


def bibliography_story(path: Path, styles) -> list:
    entries = parse_bibtex(path)
    citations = []
    for number, entry in enumerate(entries, 1):
        authors = entry.get("author", "").replace(" and ", "; ")
        venue = entry.get("journal", entry.get("booktitle", ""))
        details = ", ".join(part for part in (venue, entry.get("volume", ""), entry.get("pages", ""), entry.get("year", "")) if part)
        citation = f"[{number}] {authors}. {entry.get('title', '')}. {details}."
        citations.append(inline_markup(citation))
    if len(entries) != 26:
        raise ValueError(f"参考文献数量应为 26，实际为 {len(entries)}")
    midpoint = (len(citations) + 1) // 2
    columns = ["<br/><br/>".join(citations[:midpoint]), "<br/><br/>".join(citations[midpoint:])]
    table = Table(
        [[Paragraph(columns[0], styles["ref"]), Paragraph(columns[1], styles["ref"])]],
        colWidths=[(A4[0] - 36 * mm) / 2] * 2,
        hAlign="CENTER",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEAFTER", (0, 0), (0, -1), 0.35, colors.HexColor("#D6DEE5")),
    ]))
    return [Paragraph("参考文献（26 项）", styles["h1"]), table]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    styles = make_styles()
    doc = PaperDocTemplate(
        str(args.output),
        title="TextConsistency 中文对照稿",
        author="TextConsistency authors",
        subject="十一数据集基类到新类识别实验中文对照版",
    )
    story = markdown_story(args.source, styles)
    story.extend(bibliography_story(args.bib, styles))
    doc.build(story)


if __name__ == "__main__":
    main()
