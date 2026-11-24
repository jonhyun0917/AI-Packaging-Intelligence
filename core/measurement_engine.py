from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# 1. 평면 측정
# =========================================================

def calculate_planar_measurements(
    box_width_px: float,
    box_height_px: float,
    box_width_mm: float,
    box_length_mm: float,
    product_area_px: float,
) -> Dict[str, float]:

    if min(box_width_px, box_height_px, box_width_mm, box_length_mm) <= 0:
        raise ValueError("픽셀 및 실제 치수는 0보다 커야 합니다.")

    box_area_px = box_width_px * box_height_px
    box_area_mm2 = box_width_mm * box_length_mm

    product_area_px = max(
        0.0,
        min(float(product_area_px), box_area_px)
    )

    occupancy = product_area_px / box_area_px

    return {
        "box_area_px": round(box_area_px, 2),
        "box_area_mm2": round(box_area_mm2, 2),
        "product_area_px": round(product_area_px, 2),
        "product_area_mm2": round(occupancy * box_area_mm2, 2),
        "occupancy_percent": round(occupancy * 100, 2),
        "void_percent": round((1 - occupancy) * 100, 2),
        "mm_per_pixel_x": round(box_width_mm / box_width_px, 6),
        "mm_per_pixel_y": round(box_length_mm / box_height_px, 6),
    }


# =========================================================
# 2. 부피 측정
# =========================================================

def calculate_volume_measurements(
    box_width_mm: float,
    box_length_mm: float,
    box_height_mm: float,
    product_area_mm2: float,
    product_height_mm: float,
) -> Dict[str, float]:

    if min(box_width_mm, box_length_mm, box_height_mm) <= 0:
        raise ValueError("박스 치수는 0보다 커야 합니다.")

    box_volume = (
        box_width_mm
        * box_length_mm
        * box_height_mm
    )

    product_volume = (
        max(0.0, product_area_mm2)
        * max(0.0, product_height_mm)
    )

    occupancy = min(product_volume / box_volume, 1.0)

    return {
        "box_volume_mm3": round(box_volume, 2),
        "product_volume_mm3": round(product_volume, 2),
        "occupancy_percent": round(occupancy * 100, 2),
        "void_percent": round((1 - occupancy) * 100, 2),
        "overflow": product_volume > box_volume,
    }


# =========================================================
# 3. 측정 신뢰도 계산
# =========================================================

def estimate_measurement_confidence(
    box_confidence: float,
    segmentation_confidences: Optional[Iterable[float]],
    perspective_corrected: bool,
    dimensions_supplied: bool,
) -> float:

    vals = list(segmentation_confidences or [])

    seg = sum(vals) / len(vals) if vals else 0.0

    total = (
        max(0.0, min(box_confidence, 1.0)) * 0.30
        + max(0.0, min(seg, 1.0)) * 0.35
        + (1.0 if perspective_corrected else 0.35) * 0.20
        + (1.0 if dimensions_supplied else 0.20) * 0.15
    )

    return round(
        max(0.0, min(total, 1.0)) * 100,
        1
    )


# =========================================================
# 4. 카드 모서리 순서 정렬
# =========================================================

def order_card_points(points: np.ndarray) -> np.ndarray:
    """
    카드의 네 모서리를 다음 순서로 정렬한다.

    0: 좌측 상단
    1: 우측 상단
    2: 우측 하단
    3: 좌측 하단
    """

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    result = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    result[0] = pts[np.argmin(sums)]   # 좌측 상단
    result[2] = pts[np.argmax(sums)]   # 우측 하단
    result[1] = pts[np.argmin(diffs)]  # 우측 상단
    result[3] = pts[np.argmax(diffs)]  # 좌측 하단

    return result


# =========================================================
# 5. 카드 검출
# =========================================================

def detect_reference_card(
    img_bgr: np.ndarray,
    card_width_mm: float = 85.6,
    card_height_mm: float = 53.98,
) -> Optional[np.ndarray]:
    """
    이미지에서 신용카드 크기의 사각형을 찾는다.

    반환:
        카드 네 모서리 좌표 (4, 2)
        검출 실패 시 None
    """

    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return None

    if img_bgr.size == 0:
        return None

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return None

    if card_width_mm <= 0 or card_height_mm <= 0:
        return None

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8)
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = img_bgr.shape[:2]

    image_area = img_w * img_h

    min_area = image_area * 0.002
    max_area = image_area * 0.25

    target_ratio = (
        max(card_width_mm, card_height_mm)
        / min(card_width_mm, card_height_mm)
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        # 사각형 근사
        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # 사각형이 아니면 최소 외접 사각형 사용
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)

            rect_w, rect_h = rect[1]

            if min(rect_w, rect_h) < 10:
                continue

            corners = cv2.boxPoints(rect).astype(np.float32)

        corners = order_card_points(corners)

        # 네 변의 길이 계산
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])

        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])

        pixel_width = (top_width + bottom_width) / 2.0
        pixel_height = (left_height + right_height) / 2.0

        if min(pixel_width, pixel_height) < 10:
            continue

        ratio = (
            max(pixel_width, pixel_height)
            / min(pixel_width, pixel_height)
        )

        ratio_error = abs(ratio - target_ratio) / target_ratio

        # 촬영 각도를 고려해 비율 허용 범위 설정
        if ratio_error > 0.30:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]

        rect_area = rect_w * rect_h

        if rect_area <= 0:
            continue

        extent = area / rect_area

        if extent < 0.50:
            continue

        # 카드 후보 점수
        score = (
            ratio_error
            + abs(0.85 - extent) * 0.2
        )

        candidates.append(
            (score, area, corners)
        )

    if not candidates:
        return None

    # 점수가 낮은 후보 우선
    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][2]


# =========================================================
# 6. 카드 크기 기반 픽셀 환산
# =========================================================

def estimate_mm_per_px_from_card(
    card_px_width: float,
    card_px_height: float,
    card_width_mm: float = 85.6,
    card_height_mm: float = 53.98,
) -> float:

    if card_px_width <= 0 or card_px_height <= 0:
        raise ValueError("카드 픽셀 크기는 0보다 커야 합니다.")

    long_px = max(card_px_width, card_px_height)
    short_px = min(card_px_width, card_px_height)

    long_mm = max(card_width_mm, card_height_mm)
    short_mm = min(card_width_mm, card_height_mm)

    scale_long = long_mm / long_px
    scale_short = short_mm / short_px

    return round(
        float((scale_long + scale_short) / 2.0),
        6
    )


# =========================================================
# 7. 픽셀 크기를 mm / cm로 변환
# =========================================================

def px_size_to_mm_and_cm(
    px_w: float,
    px_h: float,
    mm_per_px: float,
) -> Dict[str, float]:

    if px_w <= 0 or px_h <= 0 or mm_per_px <= 0:
        raise ValueError("픽셀 크기와 mm_per_px는 0보다 커야 합니다.")

    width_mm = px_w * mm_per_px
    height_mm = px_h * mm_per_px

    return {
        "width_mm": round(float(width_mm), 1),
        "height_mm": round(float(height_mm), 1),
        "width_cm": round(float(width_mm / 10.0), 1),
        "height_cm": round(float(height_mm / 10.0), 1),
    }


# =========================================================
# 8. 박스 좌표를 네 모서리 형태로 변환
# =========================================================

def _bbox_to_corners(bbox) -> Optional[np.ndarray]:
    """
    입력 형식:
      - (x1, y1, x2, y2)
      - 네 모서리 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """

    if bbox is None:
        return None

    arr = np.asarray(bbox, dtype=np.float32)

    if arr.shape == (4, 2):
        return arr

    if arr.size == 4:
        x1, y1, x2, y2 = arr.reshape(4)

        return np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32
        )

    return None


# =========================================================
# 9. 원근 변환 후 사각형 실제 크기 계산
# =========================================================

def _measure_transformed_quad(
    corners: np.ndarray,
    homography: np.ndarray,
) -> Dict[str, float]:

    corners = np.asarray(
        corners,
        dtype=np.float32
    ).reshape(4, 2)

    transformed = cv2.perspectiveTransform(
        corners.reshape(1, 4, 2),
        homography
    )[0]

    top_width = np.linalg.norm(
        transformed[1] - transformed[0]
    )

    bottom_width = np.linalg.norm(
        transformed[2] - transformed[3]
    )

    left_height = np.linalg.norm(
        transformed[3] - transformed[0]
    )

    right_height = np.linalg.norm(
        transformed[2] - transformed[1]
    )

    width_mm = (top_width + bottom_width) / 2.0
    height_mm = (left_height + right_height) / 2.0

    return {
        "width_mm": round(float(width_mm), 1),
        "height_mm": round(float(height_mm), 1),
        "width_cm": round(float(width_mm / 10.0), 1),
        "height_cm": round(float(height_mm / 10.0), 1),
    }


# =========================================================
# 10. 기준 카드 기반 박스 치수 측정
# =========================================================

def measure_with_reference_card(
    img_bgr: np.ndarray,
    box_bbox_px: Optional[tuple] = None,
    product_bboxes_px: Optional[Iterable[tuple]] = None,
    card_width_mm: float = 85.6,
    card_height_mm: float = 53.98,
) -> Dict[str, object]:

    failure_result = {
        "reference_detected": False,
        "mm_per_px": None,
        "box_size": None,
        "products_size": [],
        "card_corners_px": None,
        "perspective_corrected": False,
    }

    if img_bgr is None or img_bgr.size == 0:
        return failure_result

    # 카드 네 모서리 검출
    card_corners = detect_reference_card(
        img_bgr,
        card_width_mm,
        card_height_mm
    )

    if card_corners is None:
        return failure_result

    # 카드의 가로·세로 픽셀 길이 계산
    top_width = np.linalg.norm(
        card_corners[1] - card_corners[0]
    )

    bottom_width = np.linalg.norm(
        card_corners[2] - card_corners[3]
    )

    left_height = np.linalg.norm(
        card_corners[3] - card_corners[0]
    )

    right_height = np.linalg.norm(
        card_corners[2] - card_corners[1]
    )

    pixel_width = (top_width + bottom_width) / 2.0
    pixel_height = (left_height + right_height) / 2.0

    # 카드가 세로 방향으로 놓인 경우도 고려
    if pixel_width >= pixel_height:
        physical_width_mm = card_width_mm
        physical_height_mm = card_height_mm
    else:
        physical_width_mm = card_height_mm
        physical_height_mm = card_width_mm

    mm_per_px = estimate_mm_per_px_from_card(
        pixel_width,
        pixel_height,
        physical_width_mm,
        physical_height_mm
    )

    # 카드 평면을 실제 mm 좌표계로 변환
    destination_points = np.array(
        [
            [0, 0],
            [physical_width_mm, 0],
            [physical_width_mm, physical_height_mm],
            [0, physical_height_mm],
        ],
        dtype=np.float32
    )

    homography = cv2.getPerspectiveTransform(
        card_corners.astype(np.float32),
        destination_points
    )

    box_size = None

    # 박스 외부 윗면 좌표가 전달된 경우에만 측정
    box_corners = _bbox_to_corners(box_bbox_px)

    if box_corners is not None:
        box_size = _measure_transformed_quad(
            box_corners,
            homography
        )

    # 제품 좌표가 전달된 경우 측정
    products_size = []

    for idx, bbox in enumerate(
        product_bboxes_px or [],
        start=1
    ):

        product_corners = _bbox_to_corners(bbox)

        if product_corners is None:
            continue

        dims = _measure_transformed_quad(
            product_corners,
            homography
        )

        products_size.append(
            {
                "index": idx,
                **dims,
            }
        )

    return {
        "reference_detected": True,
        "mm_per_px": mm_per_px,
        "box_size": box_size,
        "products_size": products_size,
        "card_corners_px": card_corners.tolist(),
        "perspective_corrected": True,
    }
