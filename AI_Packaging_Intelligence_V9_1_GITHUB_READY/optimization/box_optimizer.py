from __future__ import annotations

from database.box_db import BOX_DATABASE


def rank_boxes(product_volume_mm3: float, target_utilization: float = 0.75, limit: int = 4) -> list[dict]:
    """제품 체적을 수용하는 표준 박스를 다기준 점수로 정렬합니다."""
    if product_volume_mm3 <= 0:
        return []

    candidates: list[dict] = []
    for box in BOX_DATABASE:
        if product_volume_mm3 > box.volume:
            continue

        utilization = product_volume_mm3 / box.volume
        void_percent = (1.0 - utilization) * 100.0
        fit_score = max(0.0, 100.0 - abs(target_utilization - utilization) * 140.0)
        cost_score = max(0.0, 100.0 - box.cost / 25.0)
        carbon_score = max(0.0, 100.0 - box.carbon * 90.0)
        total_score = fit_score * 0.60 + cost_score * 0.20 + carbon_score * 0.20

        candidates.append({
            "box": box,
            "utilization": utilization,
            "void_percent": void_percent,
            "fit_score": round(fit_score, 1),
            "cost_score": round(cost_score, 1),
            "carbon_score": round(carbon_score, 1),
            "total_score": round(total_score, 1),
        })

    candidates.sort(key=lambda item: item["total_score"], reverse=True)
    return candidates[: max(1, limit)]


def recommend_box(product_volume_mm3: float, target_utilization: float = 0.75):
    candidates = rank_boxes(product_volume_mm3, target_utilization=target_utilization, limit=1)
    return candidates[0] if candidates else None
