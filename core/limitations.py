from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def build_limitations(
    *,
    quality: Dict[str, Any],
    box: Dict[str, Any],
    detections: List[Dict[str, Any]],
    product_height_mm: float,
    dimensions_confirmed: bool,
) -> Dict[str, Any]:
    """분석 상태에 맞는 한계, 위험도, 재촬영 지침을 생성한다."""
    items: List[str] = []
    actions: List[str] = []
    signals: List[str] = []

    quality_score = float(quality.get("score", 0.0))
    box_conf = float(box.get("confidence", 0.0)) * 100
    product_conf = _mean(float(d.get("confidence", 0.0)) for d in detections) * 100
    fallback = box.get("detection_method") == "fallback"
    perspective = bool(box.get("perspective_corrected", False))

    signals.append(f"이미지 품질 {quality_score:.0f}/100")
    signals.append(f"박스 검출 {box_conf:.0f}%")
    signals.append(f"제품 검출 {product_conf:.0f}%" if detections else "제품 검출 실패")

    if quality_score < 60:
        items.append("이미지 품질이 낮아 박스 경계와 제품 영역의 측정 오차가 커질 수 있습니다.")
        actions.append("밝고 균일한 조명에서 카메라를 고정해 다시 촬영하세요.")
    elif quality_score < 85:
        items.append("조명·선명도·해상도 중 일부가 권장 수준보다 낮아 결과를 참고값으로 해석해야 합니다.")
        actions.append("그림자와 반사를 줄이고 박스 전체가 선명하게 보이도록 촬영하세요.")

    if fallback:
        items.append("박스 경계를 안정적으로 찾지 못해 이미지 가장자리를 이용한 자동 추정 영역을 사용했습니다.")
        actions.append("박스 네 모서리가 모두 보이도록 수직 상단에서 재촬영하세요.")
    elif box_conf < 55:
        items.append("박스 검출 신뢰도가 낮아 빈 공간 비율이 실제와 다를 수 있습니다.")
        actions.append("박스 테두리와 배경의 대비를 높여 다시 촬영하세요.")

    if not perspective:
        items.append("원근 보정이 적용되지 않아 촬영 각도에 따른 면적 왜곡이 남아 있을 수 있습니다.")
        actions.append("카메라 렌즈가 박스 바닥과 최대한 평행하도록 촬영하세요.")

    if not detections:
        items.append("제품 영역을 검출하지 못해 제품 점유율과 추천 결과를 신뢰하기 어렵습니다.")
        actions.append("제품이 겹치지 않게 배치하고 배경과 구분되도록 다시 촬영하세요.")
    elif product_conf < 55:
        items.append("제품 검출 신뢰도가 낮아 일부 제품 면적이 누락되거나 과대 계산될 수 있습니다.")
        actions.append("투명·검정·반사 제품은 무광 단색 배경과 확산 조명을 사용하세요.")

    if not dimensions_confirmed:
        items.append("입력된 박스 규격이 실측값인지 확인되지 않아 실제 길이·체적 결과는 추정값입니다.")
        actions.append("줄자나 캘리퍼스로 박스 내부 가로·세로·높이를 확인하세요.")

    if product_height_mm <= 0:
        items.append("제품 높이가 입력되지 않아 3차원 체적 빈 공간은 계산 근거가 부족합니다.")
        actions.append("제품의 평균 높이를 직접 측정해 입력하세요.")
    else:
        items.append("제품 체적은 평균 높이를 적용한 근사값으로, 곡면·빈 용기·복잡한 형상은 완전히 반영하지 못합니다.")

    items.extend([
        "사진 한 장만으로 제품의 재질 강도, 파손 민감도, 완충재 성능과 운송 충격을 평가할 수 없습니다.",
        "박스 추천은 등록된 규격 데이터베이스와 공간 효율을 기준으로 하며 실제 낙하·진동·압축 시험을 대체하지 않습니다.",
        "탄소·비용·ESG 관련 값은 설정된 가정에 따른 비교용 추정치이며 공식 LCA나 환경 인증 결과가 아닙니다.",
    ])

    risk_points = 0
    risk_points += 3 if fallback else 0
    risk_points += 2 if quality_score < 60 else (1 if quality_score < 85 else 0)
    risk_points += 2 if box_conf < 55 else 0
    risk_points += 3 if not detections else (2 if product_conf < 55 else 0)
    risk_points += 1 if not perspective else 0
    risk_points += 1 if not dimensions_confirmed else 0
    risk_points += 1 if product_height_mm <= 0 else 0

    if risk_points >= 5:
        level, label, decision = "red", "낮음 · 재촬영 권장", "현재 결과만으로 박스 변경을 결정하지 마세요. 재촬영과 실측 확인이 필요합니다."
    elif risk_points >= 2:
        level, label, decision = "yellow", "보통 · 참고용 분석", "결과를 참고값으로 사용하고 실제 치수와 제품 보호 조건을 함께 확인하세요."
    else:
        level, label, decision = "green", "높음 · 분석 가능", "이미지 조건은 양호하지만 최종 포장 변경 전 실제 포장시험을 진행하세요."

    # 중복 제거, 순서 유지
    items = list(dict.fromkeys(items))
    actions = list(dict.fromkeys(actions))
    return {
        "level": level,
        "label": label,
        "decision": decision,
        "limitations": items,
        "actions": actions,
        "signals": signals,
        "risk_points": risk_points,
        "fallback_used": fallback,
        "product_confidence": round(product_conf, 1),
        "box_confidence": round(box_conf, 1),
        "quality_score": round(quality_score, 1),
    }


def build_self_review(*, limitation_result: Dict[str, Any], detected_products: int, product_height_mm: float) -> List[str]:
    checks: List[str] = []
    checks.append("✓ 박스 경계 분석을 수행했습니다." if not limitation_result["fallback_used"] else "⚠ 박스 경계 대신 자동 추정 영역을 사용했습니다.")
    checks.append(f"✓ 제품 {detected_products}개를 인식했습니다." if detected_products else "⚠ 제품을 안정적으로 인식하지 못했습니다.")
    checks.append("✓ 제품 높이 입력값을 체적 계산에 반영했습니다." if product_height_mm > 0 else "⚠ 제품 높이가 없어 체적 결과가 제한됩니다.")
    checks.append("⚠ 운송 충격·파손 위험·완충재 성능은 평가하지 않았습니다.")
    checks.append("⚠ 탄소·비용 절감은 실제 실험값이 아닌 비교용 추정치입니다.")
    return checks
