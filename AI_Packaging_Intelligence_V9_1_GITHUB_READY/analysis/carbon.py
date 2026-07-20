# carbon.py

"""
Packaging Carbon Impact Calculator

과포장 공간 → 포장재 낭비량 → CO2 영향 추정
"""


# 종이 포장재 기준
PAPER_DENSITY = 0.7  
# g/cm³


CO2_FACTOR = 1.3
# kg CO2 / kg paper



def calculate_material_waste(
        wasted_area,
        thickness=0.05
):

    """
    면적 기반 포장재 낭비량 계산

    wasted_area:
        cm² 기준

    thickness:
        종이 두께(cm)

    """

    volume = (

        wasted_area *

        thickness

    )


    weight = (

        volume *

        PAPER_DENSITY

    )


    return round(
        weight,
        2
    )





def calculate_carbon(
        wasted_area
):


    material_g = calculate_material_waste(

        wasted_area

    )


    material_kg = (

        material_g /

        1000

    )



    carbon = (

        material_kg *

        CO2_FACTOR

    )



    return {


        "wasted_material_g":

        material_g,


        "carbon_kg":

        round(
            carbon,
            4
        ),


        "carbon_g":

        round(
            carbon*1000,
            2
        )

    }
