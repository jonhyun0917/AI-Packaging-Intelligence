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
