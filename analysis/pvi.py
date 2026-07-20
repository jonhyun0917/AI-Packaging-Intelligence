from __future__ import annotations


def calculate_pvi(box_area: float, product_area: float) -> dict:
    """면적 기반 Packaging Value Index (100점)."""
    if box_area <= 0:
        box_area = 1.0

    product_ratio = max(0.0, min((product_area / box_area) * 100, 100.0))
    void_percent = 100.0 - product_ratio

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

    protection_score = 18
    if product_ratio < 10:
        protection_score = 22
    if product_ratio < 5:
        protection_score = 24

    sustainability_score = 15
    pvi = round(min(efficiency_score + void_score + protection_score + sustainability_score, 100), 1)

    if pvi >= 90:
        grade = "A+ 최적"
    elif pvi >= 80:
        grade = "A 우수"
    elif pvi >= 70:
        grade = "B 양호"
    elif pvi >= 50:
        grade = "C 개선 필요"
    else:
        grade = "D 과포장"

    return {
        "pvi": pvi,
        "grade": grade,
        "void_percent": round(void_percent, 2),
        "occupancy_percent": round(product_ratio, 2),
        "scores": {
            "공간 효율성": efficiency_score,
            "Void 관리": void_score,
            "제품 보호성": protection_score,
            "지속가능성": sustainability_score,
        },
        "max_scores": {
            "공간 효율성": 35,
            "Void 관리": 25,
            "제품 보호성": 25,
            "지속가능성": 15,
        },
    }
