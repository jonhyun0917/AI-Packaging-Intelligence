from __future__ import annotations
from typing import Dict, Iterable, Optional


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
    product_area_px = max(0.0, min(float(product_area_px), box_area_px))
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


def calculate_volume_measurements(
    box_width_mm: float,
    box_length_mm: float,
    box_height_mm: float,
    product_area_mm2: float,
    product_height_mm: float,
) -> Dict[str, float]:
    if min(box_width_mm, box_length_mm, box_height_mm) <= 0:
        raise ValueError("박스 치수는 0보다 커야 합니다.")

    box_volume = box_width_mm * box_length_mm * box_height_mm
    product_volume = max(0.0, product_area_mm2) * max(0.0, product_height_mm)
    occupancy = min(product_volume / box_volume, 1.0)

    return {
        "box_volume_mm3": round(box_volume, 2),
        "product_volume_mm3": round(product_volume, 2),
        "occupancy_percent": round(occupancy * 100, 2),
        "void_percent": round((1 - occupancy) * 100, 2),
        "overflow": product_volume > box_volume,
    }


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
    return round(max(0.0, min(total, 1.0)) * 100, 1)


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
    return round(float((scale_long + scale_short) / 2.0), 6)


def px_size_to_mm_and_cm(px_w: float, px_h: float, mm_per_px: float) -> Dict[str, float]:
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


def measure_with_reference_card(
    img_bgr: np.ndarray,
    box_bbox_px: Optional[tuple] = None,
    product_bboxes_px: Optional[Iterable[tuple]] = None,
    card_width_mm: float = 85.6,
    card_height_mm: float = 53.98,
) -> Dict[str, object]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img_bgr.shape[:2]
    min_area = img_w * img_h * 0.002
    max_area = img_w * img_h * 0.25

    target_ratio = max(card_width_mm, card_height_mm) / min(card_width_mm, card_height_mm)
    best_match = None
    best_score = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        if w < 10 or h < 10:
            continue

        ratio = max(w, h) / min(w, h)
        ratio_err = abs(ratio - target_ratio) / target_ratio
        extent = area / (w * h) if (w * h) > 0 else 0

        if ratio_err <= 0.25 and extent >= 0.70:
            score = ratio_err + (1.0 - extent)
            if score < best_score:
                best_score = score
                best_match = (w, h)

    if best_match is None:
        return {"reference_detected": False, "mm_per_px": None, "box_size": None, "products_size": []}

    mm_per_px = estimate_mm_per_px_from_card(best_match[0], best_match[1], card_width_mm, card_height_mm)

    box_size = None
    if box_bbox_px is not None:
        x1, y1, x2, y2 = box_bbox_px
        box_size = px_size_to_mm_and_cm(abs(x2 - x1), abs(y2 - y1), mm_per_px)

    products_size = []
    for idx, bbox in enumerate(product_bboxes_px or [], start=1):
        px1, py1, px2, py2 = bbox
        dims = px_size_to_mm_and_cm(abs(px2 - px1), abs(py2 - py1), mm_per_px)
        products_size.append({"index": idx, **dims})

    return {
        "reference_detected": True,
        "mm_per_px": mm_per_px,
        "box_size": box_size,
        "products_size": products_size,
    }
