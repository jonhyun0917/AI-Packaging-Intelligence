from __future__ import annotations

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
except Exception:
    pass


def build_analysis_certificate(data: dict) -> bytes:
    """전시·캠페인 공유용 AI 포장 분석 인증서를 생성합니다."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI 포장 분석 인증서 V8.0",
    )
    font = "HYSMyeongJo-Medium"
    title = ParagraphStyle("cert_title", fontName=font, fontSize=26, leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#0A4A31"))
    subtitle = ParagraphStyle("cert_sub", fontName=font, fontSize=11, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#4E665A"))
    body = ParagraphStyle("cert_body", fontName=font, fontSize=10, leading=16, alignment=TA_LEFT)
    center = ParagraphStyle("cert_center", parent=body, alignment=TA_CENTER)
    story = [
        Spacer(1, 7 * mm),
        Paragraph("AI Packaging Analysis Certificate", title),
        Paragraph("AI 포장 분석 인증서", ParagraphStyle("kr", parent=title, fontSize=18, leading=25)),
        Spacer(1, 5 * mm),
        Paragraph("Better Life For Us · Green Value YOUTH 2026", subtitle),
        Spacer(1, 10 * mm),
        Paragraph("본 인증서는 업로드된 포장 이미지를 기반으로 산출된 프로젝트 내부 분석 결과를 요약합니다.", center),
        Spacer(1, 8 * mm),
    ]
    rows = [
        ["분석 일시", data.get("analysis_date", "-")],
        ["포장 건강도", data.get("health_score", "-")],
        ["종합 등급", data.get("grade", "-")],
        ["면적 빈 공간", data.get("area_void", "-")],
        ["체적 빈 공간", data.get("volume_void", "-")],
        ["ESG 참고 점수", data.get("esg", "-")],
        ["측정 신뢰도", data.get("confidence", "-")],
        ["추천 박스", data.get("recommended_box", "-")],
        ["추천 규격", data.get("recommended_size", "-")],
        ["예상 탄소 절감", data.get("carbon_saving", "-")],
    ]
    table = Table([[Paragraph("분석 항목", body), Paragraph("결과", body)]] + [[Paragraph(str(a), body), Paragraph(str(b), body)] for a,b in rows], colWidths=[55*mm, 95*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DFF2D8")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#073B2A")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#AAC7AE")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("FONTNAME", (0,0), (-1,-1), font),
    ]))
    story.append(table)
    story.extend([
        Spacer(1, 10 * mm),
        Paragraph("AI 종합 의견", ParagraphStyle("head", parent=body, fontSize=13, textColor=colors.HexColor("#0A4A31"))),
        Spacer(1, 2 * mm),
        Paragraph(str(data.get("summary", "-")), body),
        Spacer(1, 14 * mm),
        Paragraph("데이터로 포장을 이해하고, 더 나은 선택으로 바꾸는 지속가능한 포장 솔루션", center),
        Spacer(1, 4 * mm),
        Paragraph("Better Life For Us", ParagraphStyle("sign", parent=center, fontSize=15, textColor=colors.HexColor("#1F7A3D"))),
        Spacer(1, 8 * mm),
        Paragraph("※ 본 인증서는 공식 시험성적서, 법적 적합성 인증 또는 환경성 인증을 대체하지 않습니다.", ParagraphStyle("note", parent=center, fontSize=8, textColor=colors.HexColor("#666666"))),
    ])
    doc.build(story)
    return buffer.getvalue()
