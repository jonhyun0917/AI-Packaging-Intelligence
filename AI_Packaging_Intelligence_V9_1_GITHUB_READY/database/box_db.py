"""
AI Packaging Intelligence System V5.0

Box Database

Author : Better Life For US
"""

from dataclasses import dataclass
from typing import List


# ==========================================================
# Box Class
# ==========================================================

@dataclass
class Box:

    company: str
    code: str

    length: int      # mm
    width: int
    height: int

    max_weight: float     # kg

    cost: int             # 원

    carbon: float         # kgCO2e

    recommended: List[str]

    @property
    def volume(self):

        return self.length * self.width * self.height

    @property
    def surface_area(self):

        return 2 * (
            self.length * self.width +
            self.length * self.height +
            self.width * self.height
        )


# ==========================================================
# Korea Standard Box Database
# ==========================================================

BOX_DATABASE = [

    Box(

        company="우체국",

        code="1호",

        length=220,

        width=190,

        height=90,

        max_weight=2,

        cost=700,

        carbon=0.18,

        recommended=[
            "화장품",
            "책",
            "문구"
        ]

    ),

    Box(

        company="우체국",

        code="2호",

        length=270,

        width=180,

        height=150,

        max_weight=5,

        cost=900,

        carbon=0.24,

        recommended=[
            "화장품",
            "생활용품",
            "의류"
        ]

    ),

    Box(

        company="우체국",

        code="2-1호",

        length=350,

        width=250,

        height=100,

        max_weight=7,

        cost=1100,

        carbon=0.32,

        recommended=[
            "신발",
            "생활용품"
        ]

    ),

    Box(

        company="우체국",

        code="3호",

        length=340,

        width=250,

        height=210,

        max_weight=10,

        cost=1300,

        carbon=0.45,

        recommended=[
            "신발",
            "화장품 세트",
            "소형가전"
        ]

    ),

    Box(

        company="우체국",

        code="4호",

        length=410,

        width=310,

        height=280,

        max_weight=15,

        cost=1700,

        carbon=0.62,

        recommended=[
            "주방용품",
            "전자제품",
            "식품"
        ]

    ),

    Box(

        company="우체국",

        code="5호",

        length=480,

        width=380,

        height=340,

        max_weight=20,

        cost=2200,

        carbon=0.84,

        recommended=[
            "대형생활용품",
            "대형가전",
            "묶음배송"
        ]

    )

]


# ==========================================================
# Search Function
# ==========================================================

def get_all_boxes():

    return BOX_DATABASE


def get_box(code):

    for box in BOX_DATABASE:

        if box.code == code:

            return box

    return None


# ==========================================================
# Print
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("AI Packaging Intelligence System")

    print("Standard Box Database")

    print("=" * 60)

    for box in BOX_DATABASE:

        print(

            f"{box.company} {box.code}"

        )

        print(

            f"Size : {box.length} × {box.width} × {box.height} mm"

        )

        print(

            f"Volume : {box.volume:,} mm³"

        )

        print(

            f"Surface : {box.surface_area:,} mm²"

        )

        print(

            f"Weight : {box.max_weight} kg"

        )

        print(

            f"Carbon : {box.carbon} kgCO₂"

        )

        print(

            f"Cost : {box.cost}원"

        )

        print("-" * 60)
        