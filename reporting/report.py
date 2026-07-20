from __future__ import annotations

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))


def build_pdf_report(summary: dict) -> bytes:
    """한글을 지원하는 다중 섹션 PDF 분석 보고서를 생성합니다."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI 포장 지능 분석 보고서 V9.0",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "KTitle", parent=styles["Title"], fontName="HYSMyeongJo-Medium",
        fontSize=22, leading=30, alignment=TA_CENTER, spaceAfter=18,
    )
    heading = ParagraphStyle(
        "KHeading", parent=styles["Heading2"], fontName="HYSMyeongJo-Medium",
        fontSize=14, leading=20, spaceBefore=12, spaceAfter=8,
    )
    body = ParagraphStyle(
        "KBody", parent=styles["BodyText"], fontName="HYSMyeongJo-Medium",
        fontSize=9.5, leading=15,
    )
    small = ParagraphStyle(
        "KSmall", parent=body, fontSize=8, leading=12, textColor=colors.HexColor("#555555"),
    )

    story = [
        Spacer(1, 18 * mm),
        Paragraph("AI 포장 지능 분석 보고서", title),
        Paragraph("프로페셔널 한국어 에디션 V9.0", ParagraphStyle("Sub", parent=body, alignment=TA_CENTER, fontSize=12)),
        Spacer(1, 15 * mm),
        Paragraph("측정값 · 추정값 · 추천 근거 · 신뢰도 · 한계 통합 분석", ParagraphStyle("Center", parent=body, alignment=TA_CENTER)),
        PageBreak(),
    ]

    sections = summary.get("sections")
    if not sections:
        sections = {"분석 결과": summary}

    for section_name, values in sections.items():
        story.append(Paragraph(str(section_name), heading))
        if isinstance(values, dict):
            rows = [[Paragraph("항목", body), Paragraph("결과", body)]]
            for key, value in values.items():
                rows.append([Paragraph(str(key), body), Paragraph(str(value), body)])
            table = Table(rows, colWidths=[55 * mm, 110 * mm], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DFF2D8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAC7AE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "HYSMyeongJo-Medium"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(table)
        elif isinstance(values, (list, tuple)):
            for item in values:
                story.append(Paragraph(f"• {item}", body))
        else:
            story.append(Paragraph(str(values), body))
        story.append(Spacer(1, 5 * mm))

    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("결과 사용 원칙", heading),
        Paragraph(
            "본 결과는 이미지 기반 근사 분석과 내부 계산 로직을 사용한 의사결정 지원 자료입니다. 측정값과 추정값을 구분해 해석해야 합니다. "
            "공식 법적 적합성 인증, 구조 안전 시험, 환경성적표지 또는 전과정평가(LCA)를 대체하지 않습니다.",
            small,
        ),
    ])

    doc.build(story)
    return buffer.getvalue()
