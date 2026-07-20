from __future__ import annotations


def build_consultant_report(
    *,
    area_void: float,
    volume_void: float,
    confidence: float,
    current_volume_mm3: float,
    recommendation: dict | None,
    esg_score: float,
    detected_products: int,
) -> dict:
    """수치 결과를 사람이 읽기 쉬운 진단·권고 문장으로 변환합니다."""
    primary_void = max(area_void, volume_void)

    if primary_void >= 80:
        status = "심각한 과대포장 가능성"
        severity = "높음"
        summary = "제품 대비 사용되지 않는 공간이 매우 커서 박스 축소를 가장 먼저 검토해야 합니다."
    elif primary_void >= 60:
        status = "과대포장 개선 필요"
        severity = "중간"
        summary = "제품 보호 수준을 유지하면서 더 작은 박스 또는 낮은 높이의 박스로 전환할 여지가 큽니다."
    elif primary_void >= 40:
        status = "부분 최적화 필요"
        severity = "보통"
        summary = "현재 포장은 사용 가능하지만 배치 방향과 박스 높이를 조정하면 효율을 더 높일 수 있습니다."
    else:
        status = "적정 포장"
        severity = "낮음"
        summary = "제품과 박스의 크기 적합도가 양호하며 보호성과 효율의 균형이 비교적 좋습니다."

    actions: list[str] = []
    if area_void >= 60:
        actions.append("제품 바닥면에 가까운 가로·세로 규격의 박스를 우선 검토하세요.")
    if volume_void >= 60:
        actions.append("박스 높이를 제품 높이와 완충 여유에 맞춰 낮추세요.")
    if detected_products > 1:
        actions.append("제품 간 간격을 최소화하고 회전 배치를 적용해 다품종 적재 효율을 높이세요.")
    actions.append("완충재는 제품 이동을 막는 최소 고정 구조 중심으로 설계하세요.")
    actions.append("반복 출고 품목은 PVI 결과를 누적해 표준 박스 종류를 단순화하세요.")

    savings = None
    if recommendation:
        box = recommendation["box"]
        recommended_volume = float(box.volume)
        volume_saving = max((current_volume_mm3 - recommended_volume) / current_volume_mm3 * 100.0, 0.0) if current_volume_mm3 > 0 else 0.0
        savings = {
            "box_name": f"{box.company} {box.code}",
            "size": f"{box.length} × {box.width} × {box.height} mm",
            "volume_saving_percent": round(volume_saving, 1),
            "expected_void_percent": round(recommendation["void_percent"], 1),
            "cost": box.cost,
            "carbon": box.carbon,
        }

    reliability = (
        "높음" if confidence >= 80 else
        "보통" if confidence >= 60 else
        "낮음"
    )

    return {
        "status": status,
        "severity": severity,
        "summary": summary,
        "actions": actions,
        "savings": savings,
        "reliability": reliability,
        "esg_comment": (
            "환경 효율이 우수합니다." if esg_score >= 80 else
            "환경 효율은 보통이며 박스 축소 효과가 기대됩니다." if esg_score >= 60 else
            "포장재와 탄소 영향을 줄이기 위한 규격 재설계가 필요합니다."
        ),
    }
