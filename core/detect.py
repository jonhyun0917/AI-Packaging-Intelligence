from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def _background_mask(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border = max(3, int(min(h, w) * 0.04))
    samples = np.concatenate(
        [
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    threshold = max(12.0, float(np.percentile(dist, 72)))
    return np.where(dist >= threshold, 255, 0).astype(np.uint8)


def _edge_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))

    k = max(3, int(min(crop.shape[:2]) * 0.012))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def detect_products(
    image: Any,
    box: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int, int]:
    rgb = np.array(image.convert("RGB") if hasattr(image, "convert") else image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB 이미지가 필요합니다.")

    original = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    result = original.copy()

    use_warped = bool(box and isinstance(box.get("warped_image"), np.ndarray))
    if use_warped:
        crop = box["warped_image"].copy()
        result = crop.copy()
        offset_x = offset_y = 0
    else:
        h, w = original.shape[:2]
        x1 = int(box.get("x1", 0)) if box else int(w * 0.05)
        y1 = int(box.get("y1", 0)) if box else int(h * 0.05)
        x2 = int(box.get("x2", w)) if box else int(w * 0.95)
        y2 = int(box.get("y2", h)) if box else int(h * 0.95)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = original[y1:y2, x1:x2].copy()
        offset_x, offset_y = x1, y1

    if crop.size == 0:
        raise ValueError("분석할 박스 내부 영역이 없습니다.")

    ch, cw = crop.shape[:2]
    crop_area = ch * cw

    mask = cv2.bitwise_or(_background_mask(crop), _edge_mask(crop))

    k = max(3, int(min(ch, cw) * 0.008))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    margin = max(2, int(min(ch, cw) * 0.015))
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(120, int(crop_area * 0.004))
    max_area = int(crop_area * 0.82)

    detections: List[Dict[str, Any]] = []
    union_mask = np.zeros((ch, cw), dtype=np.uint8)

    components = []
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area:
            continue

        rect_area = max(1, w * h)
        fill_ratio = area / rect_area
        aspect = w / max(1, h)
        if fill_ratio < 0.18 or aspect < 0.08 or aspect > 12:
            continue

        touches = sum([
            x <= margin,
            y <= margin,
            x + w >= cw - margin,
            y + h >= ch - margin,
        ])
        if touches >= 3 and area > crop_area * 0.20:
            continue

        area_ratio = area / crop_area
        confidence = np.clip(
            min(1.0, area_ratio / 0.12) * 0.35
            + min(1.0, fill_ratio) * 0.40
            + (1.0 - touches / 4.0) * 0.25,
            0.05,
            0.99,
        )
        components.append((area, label, x, y, w, h, float(confidence)))

    components.sort(reverse=True)

    for idx, (area, label, x, y, w, h, confidence) in enumerate(components, start=1):
        cmask = np.where(labels == label, 255, 0).astype(np.uint8)
        union_mask = cv2.bitwise_or(union_mask, cmask)

        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 180, 0), 2)
        cv2.putText(
            result,
            f"product {idx} {confidence:.2f}",
            (x, max(24, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "name": f"Detected Product {idx}",
                "confidence": round(confidence, 2),
                "area": area,
                "bbox": [x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y],
                "width_px": w,
                "height_px": h,
            }
        )

    product_area = int(cv2.countNonZero(union_mask))
    return detections, result, product_area, len(detections)
