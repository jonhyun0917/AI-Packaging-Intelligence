"""
Volume PVI Engine

공간 효율 기반 과포장 지수
"""


def calculate_volume(
        width,
        height,
        depth
):

    return (
        width *
        height *
        depth
    )





def calculate_volume_pvi(
        box,
        product
):


    box_volume = calculate_volume(

        box["width"],
        box["height"],
        box["depth"]

    )


    product_volume = calculate_volume(

        product["width"],
        product["height"],
        product["depth"]

    )



    void_volume = (

        box_volume -
        product_volume

    )


    pvi = (

        void_volume /
        box_volume

    ) * 100




    if pvi < 20:

        grade="친환경 포장"


    elif pvi <40:

        grade="적정 포장"


    elif pvi <60:

        grade="개선 필요"


    else:

        grade="과포장"



    return {


        "box_volume":
            round(box_volume,2),


        "product_volume":
            round(product_volume,2),


        "void_volume":
            round(void_volume,2),


        "PVI":
            round(pvi,2),


        "grade":
            grade

    }
