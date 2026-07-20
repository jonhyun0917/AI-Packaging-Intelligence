# recommendation.py
# AI Packaging Intelligence System V4.0
# AI Packaging Consultant


def recommend_packaging(result):


    """
    PVI 기반 포장 개선 추천
    """



    pvi = result["pvi"]

    void = result["void_percent"]



    # =================================
    # 기본값
    # =================================


    recommendation = {


        "priority":
        "포장 공간 최적화",


        "esg":
        "B",


        "box":
        "제품 크기에 맞춘 높이 조절형 박스",


        "analysis":
        [],


        "improvement":
        [],


        "material":
        0,


        "carbon":
        0,


        "saving":
        0

    }



    # =================================
    # 과포장 분석
    # =================================


    if void >= 80:


        recommendation["priority"] = (

            "높은 Void Space 개선"

        )


        recommendation["esg"] = "C"



        recommendation["analysis"] = [

            f"제품 대비 빈 공간이 {void}%로 높습니다.",

            "현재 박스 크기가 제품 크기에 비해 과대합니다.",

            "제품 보호 목적은 유지하면서 박스 축소가 필요합니다."

        ]



        recommendation["improvement"] = [


            "높이 조절형 택배박스 적용",

            "제품 맞춤형 슬리브 또는 내부 고정 구조 적용",

            "불필요한 완충 공간 최소화"


        ]


        recommendation["box"] = (

            "현재 박스 대비 약 40~50% 축소 가능한 "

            "맞춤형 소형 박스"

        )


        recommendation["material"] = 35

        recommendation["carbon"] = 25

        recommendation["saving"] = 20



    elif void >= 50:


        recommendation["priority"] = (

            "박스 크기 개선"

        )


        recommendation["esg"]="B"


        recommendation["analysis"]=[

            "일부 빈 공간이 존재합니다.",

            "제품 보호와 공간 효율의 균형이 필요합니다."

        ]


        recommendation["improvement"]=[

            "내부 완충재 최적화",

            "박스 높이 감소"

        ]


        recommendation["material"]=20

        recommendation["carbon"]=15

        recommendation["saving"]=10



    else:


        recommendation["priority"] = (

            "현재 포장 효율 우수"

        )


        recommendation["esg"]="A"



        recommendation["analysis"]=[

            "제품 대비 박스 공간 효율이 우수합니다."

        ]


        recommendation["improvement"]=[

            "현재 포장 방식 유지"

        ]


        recommendation["material"]=5

        recommendation["carbon"]=5

        recommendation["saving"]=5



    return recommendation
