from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _fallback_points(width: int, height: int) -> np.ndarray:
    mx, my = int(width * 0.08), int(height * 0.08)
    return np.array(
        [[mx, my], [width - mx, my], [width - mx, height - my], [mx, height - my]],
        dtype=np.float32,
    )


def _find_quadrilateral(image: np.ndarray) -> Tuple[np.ndarray | None, float]:
    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, med * 0.55)), int(min(255, med * 1.45)))

    k = max(3, int(min(h, w) * 0.012))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, np.ndarray, float]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if not (image_area * 0.12 <= area <= image_area * 0.98):
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = _order_points(approx.reshape(4, 2))
        poly_area = abs(float(cv2.contourArea(pts)))
        x, y, rw, rh = cv2.boundingRect(pts.astype(np.int32))
        rectangularity = poly_area / max(1.0, float(rw * rh))
        if rectangularity < 0.45:
            continue

        center = pts.mean(axis=0)
        image_center = np.array([w / 2, h / 2], dtype=np.float32)
        center_score = max(0.0, 1.0 - np.linalg.norm(center - image_center) / max(1.0, np.hypot(w, h)) * 2.0)
        area_ratio = poly_area / image_area

        score = area_ratio * 0.60 + rectangularity * 0.30 + center_score * 0.10
        confidence = float(np.clip(area_ratio * 0.45 + rectangularity * 0.45 + center_score * 0.10, 0, 1))
        candidates.append((score, pts, confidence))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _warp(image: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tl, tr, br, bl = points
    width = max(1, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    height = max(1, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def detect_box(image: Any) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    rgb = np.array(image.convert("RGB") if hasattr(image, "convert") else image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB 이미지가 필요합니다.")

    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    result = bgr.copy()
    h, w = bgr.shape[:2]

    points, confidence = _find_quadrilateral(bgr)
    auto_detected = points is not None
    if points is None:
        points = _fallback_points(w, h)
        confidence = 0.25

    points = _order_points(points)
    warped, matrix = _warp(bgr, points)
    area = int(round(abs(cv2.contourArea(points))))

    ipts = points.astype(np.int32)
    color = (0, 255, 0) if auto_detected else (0, 165, 255)
    cv2.polylines(result, [ipts], True, color, 4, cv2.LINE_AA)

    for label, pt in zip(("TL", "TR", "BR", "BL"), ipts):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(result, p, 8, (255, 0, 0), -1)
        cv2.putText(result, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(w, int(np.ceil(points[:, 0].max())))
    y2 = min(h, int(np.ceil(points[:, 1].max())))

    box = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "corners": points.tolist(),
        "confidence": round(float(confidence), 3),
        "auto_detected": auto_detected,
        "perspective_corrected": True,
        "warped_image": warped,
        "perspective_matrix": matrix,
        "warped_width": int(warped.shape[1]),
        "warped_height": int(warped.shape[0]),
        "area_px": area,
    }
    return result, area, box
