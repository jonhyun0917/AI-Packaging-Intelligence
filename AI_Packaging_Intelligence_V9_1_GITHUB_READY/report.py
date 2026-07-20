# report.py
# AI Packaging Intelligence System
# PDF Report Generator V2.0


from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.pagesizes import A4

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont



# ======================================
# Korean Font Setting
# ======================================


pdfmetrics.registerFont(
    UnicodeCIDFont(
        "HYSMyeongJo-Medium"
    )
)



# ======================================
# CREATE REPORT
# ======================================


def create_report(
        filename,
        result,
        detections,
        box_area,
        product_area,
        recommend,
        image_path=None
):


    """
    AI Packaging Report PDF 생성
    """



    doc = SimpleDocTemplate(

        filename,

        pagesize=A4

    )



    # ======================
    # Styles
    # ======================


    styles = getSampleStyleSheet()



    title_style = ParagraphStyle(

        "KoreanTitle",

        parent=styles["Title"],

        fontName="HYSMyeongJo-Medium",

        fontSize=18,

        leading=24

    )



    heading_style = ParagraphStyle(

        "KoreanHeading",

        parent=styles["Heading2"],

        fontName="HYSMyeongJo-Medium",

        fontSize=13,

        leading=18

    )



    body_style = ParagraphStyle(

        "KoreanBody",

        parent=styles["BodyText"],

        fontName="HYSMyeongJo-Medium",

        fontSize=10,

        leading=15

    )



    story = []



    # ======================================
    # TITLE
    # ======================================


    story.append(

        Paragraph(

            "AI Packaging Intelligence Report",

            title_style

        )

    )


    story.append(

        Spacer(1,20)

    )



    # ======================================
    # SUMMARY
    # ======================================


    story.append(

        Paragraph(

            "📊 Packaging Analysis Summary",

            heading_style

        )

    )



    summary = f"""

    PVI 점수 : {result['pvi']} 점<br/>

    최종 등급 : {result['grade']}<br/>

    Void Space : {result['void_percent']} %<br/>

    제품 점유율 : {result['product_ratio']} %<br/>

    Box Area : {box_area:,} px²<br/>

    Product Area : {product_area:,} px²<br/>

    """



    story.append(

        Paragraph(

            summary,

            body_style

        )

    )



    story.append(

        Spacer(1,20)

    )



    # ======================================
    # PVI SCORE
    # ======================================


    story.append(

        Paragraph(

            "📈 PVI Score Dashboard",

            heading_style

        )

    )



    for name,value in result["scores"].items():


        max_score = result["max_scores"][name]


        score_text = f"""

        {name} :

        {value}/{max_score}

        """



        story.append(

            Paragraph(

                score_text,

                body_style

            )

        )



    story.append(

        Spacer(1,20)

    )



    # ======================================
    # DETECTION RESULT
    # ======================================


    story.append(

        Paragraph(

            "📦 Detection Result",

            heading_style

        )

    )



    for obj in detections:


        detection_text = f"""

        제품명 : {obj['name']}<br/>

        Confidence :
        {obj['confidence']*100:.1f}%<br/>

        Area :
        {obj['area']:,} px²<br/>

        BBox :
        {obj['bbox']}

        """



        story.append(

            Paragraph(

                detection_text,

                body_style

            )

        )


        story.append(

            Spacer(1,10)

        )



    story.append(

        Spacer(1,20)

    )



    # ======================================
    # AI CONSULTANT
    # ======================================


    story.append(

        Paragraph(

            "🤖 AI Packaging Consultant",

            heading_style

        )

    )



    consultant = f"""

    우선순위 :

    {recommend['priority']}<br/><br/>


    추천 포장 :

    {recommend['box']}<br/><br/>


    포장재 절감 :

    {recommend['material']} %<br/>

    CO₂ 절감 :

    {recommend['carbon']} %<br/>

    비용 절감 :

    {recommend['saving']} %

    """



    story.append(

        Paragraph(

            consultant,

            body_style

        )

    )



    story.append(

        Spacer(1,20)

    )



    # ======================================
    # ANALYSIS
    # ======================================


    story.append(

        Paragraph(

            "AI 분석",

            heading_style

        )

    )


    for item in recommend["analysis"]:


        story.append(

            Paragraph(

                "✔ " + item,

                body_style

            )

        )



    story.append(

        Spacer(1,15)

    )



    # ======================================
    # IMPROVEMENT
    # ======================================


    story.append(

        Paragraph(

            "개선 방향",

            heading_style

        )

    )



    for item in recommend["improvement"]:


        story.append(

            Paragraph(

                "✅ " + item,

                body_style

            )

        )



    # ======================================
    # IMAGE
    # ======================================


    if image_path:


        story.append(

            Spacer(1,20)

        )


        story.append(

            Image(

                image_path,

                width=300,

                height=200

            )

        )



    # ======================================
    # BUILD PDF
    # ======================================


    doc.build(

        story

    )


    return filename
