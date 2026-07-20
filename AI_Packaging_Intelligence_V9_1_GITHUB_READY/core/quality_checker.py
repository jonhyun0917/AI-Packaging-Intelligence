from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np


def check_image_quality(image: Any) -> Dict[str, float | str | bool]:
    rgb = np.array(image.convert("RGB") if hasattr(image, "convert") else image)
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape

    warnings = []
    if min(h, w) < 600:
        warnings.append("해상도가 낮습니다.")
    if brightness < 55:
        warnings.append("사진이 너무 어둡습니다.")
    elif brightness > 220:
        warnings.append("사진이 너무 밝습니다.")
    if sharpness < 70:
        warnings.append("사진이 흐릴 수 있습니다.")

    score = 100.0
    score -= 25 if min(h, w) < 600 else 0
    score -= 25 if brightness < 55 or brightness > 220 else 0
    score -= 30 if sharpness < 70 else 0

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "score": round(max(0.0, score), 1),
        "is_acceptable": len(warnings) == 0,
        "message": " / ".join(warnings) if warnings else "분석에 적합한 이미지입니다.",
    }
