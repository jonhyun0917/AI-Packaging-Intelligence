# pvi.py
# AI Packaging Intelligence System V4.2
# Packaging Value Index


def calculate_pvi(
        box_area,
        product_area
):


    """
    PVI V4.2

    Box Inner Space Based Analysis

    Total 100
    """



    # ======================
    # Basic
    # ======================

    if box_area <= 0:

        box_area = 1



    product_ratio = (

        product_area /

        box_area

    ) * 100



    void_percent = 100 - product_ratio



    # ======================
    # 1. Space Efficiency
    # 35
    # ======================


    if product_ratio >= 50:

        efficiency_score = 35


    elif product_ratio >= 30:

        efficiency_score = 30


    elif product_ratio >= 15:

        efficiency_score = 25


    elif product_ratio >= 8:

        efficiency_score = 18


    elif product_ratio >= 5:

        efficiency_score = 12


    else:

        efficiency_score = 8




    # ======================
    # 2. Void Management
    # 25
    # ======================


    if void_percent <= 30:

        void_score = 25


    elif void_percent <= 50:

        void_score = 20


    elif void_percent <= 70:

        void_score = 15


    elif void_percent <= 85:

        void_score = 10


    else:

        void_score = 5



    # ======================
    # 3. Product Protection
    # 25
    # ======================


    # 기본 점수

    protection_score = 18



    # 의류/소프트 제품 보정

    if product_ratio < 10:

        protection_score = 22



    if product_ratio < 5:

        protection_score = 24



    # ======================
    # 4. Sustainability
    # 15
    # ======================


    sustainability_score = 15



    # ======================
    # Total
    # ======================


    pvi = (

        efficiency_score

        +

        void_score

        +

        protection_score

        +

        sustainability_score

    )



    pvi = round(

        min(pvi,100),

        1

    )



    # ======================
    # Grade
    # ======================


    if pvi >= 90:

        grade="A+ 최적"


    elif pvi >= 80:

        grade="A 우수"


    elif pvi >= 70:

        grade="B 양호"


    elif pvi >= 50:

        grade="C 개선 필요"


    else:

        grade="D 과포장"



    scores={

        "공간 효율성":

        efficiency_score,


        "Void 관리":

        void_score,


        "제품 보호성":

        protection_score,


        "지속가능성":

        sustainability_score

    }



    max_scores={

        "공간 효율성":

        35,


        "Void 관리":

        25,


        "제품 보호성":

        25,


        "지속가능성":

        15

    }



    return {


        "pvi":

        pvi,


        "grade":

        grade,


        "void_percent":

        round(void_percent,2),


        "product_ratio":

        round(product_ratio,2),


        "scores":

        scores,


        "max_scores":

        max_scores,


        "confidence":

        90

    }
