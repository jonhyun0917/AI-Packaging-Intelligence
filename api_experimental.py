from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import base64

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageOps

from core.boxfit_experiment import analyze_image
from core.measurement_engine import measure_with_reference_card

app = FastAPI(title="BoxFit Experimental API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_bboxes_from_result(result: dict):
    if not isinstance(result, dict):
        return None, None

    box_bbox = result.get("box_bbox") or result.get("outer_box_bbox")
    product_bboxes = (
        result.get("product_bboxes")
        or result.get("products_bbox")
        or result.get("boxes")
    )

    if isinstance(product_bboxes, list) and product_bboxes and isinstance(product_bboxes[0], dict):
        extracted = []
        for item in product_bboxes:
            b = item.get("bbox") or item.get("xyxy")
            if b and len(b) == 4:
                extracted.append(tuple(b))
        product_bboxes = extracted or None

    return (tuple(box_bbox) if box_bbox and len(box_bbox) == 4 else None), product_bboxes


@app.get("/")
def root():
    return {
        "success": True,
        "message": "BoxFit Experimental API"
    }


@app.post("/analyze-experimental")
async def analyze_experimental(
    image: UploadFile = File(...)
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드할 수 있습니다."
        )

    raw = await image.read()

    try:
        pil_image = ImageOps.exif_transpose(
            Image.open(BytesIO(raw))
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="이미지를 읽을 수 없습니다."
        )

    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    with TemporaryDirectory() as directory:
        input_path = Path(directory) / "input.png"
        output_path = Path(directory) / "result.png"

        pil_image.save(input_path)

        try:
            result = analyze_image(
                input_path,
                output_path
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"실험 분석 실패: {type(e).__name__}"
            )

        box_bbox, product_bboxes = _extract_bboxes_from_result(result)
        try:
            size_measurement = measure_with_reference_card(
                img_bgr=img_bgr,
                box_bbox_px=box_bbox,
                product_bboxes_px=product_bboxes,
            )
        except Exception:
            size_measurement = {
                "reference_detected": False,
                "mm_per_px": None,
                "box_size": None,
                "products_size": [],
            }

        result_image = cv2.imread(str(output_path))

        if result_image is None:
            raise HTTPException(
                status_code=500,
                detail="분석 결과 이미지를 읽지 못했습니다."
            )

        success, encoded = cv2.imencode(
            ".png", result_image
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="결과 이미지 인코딩 실패"
            )

        image_base64 = base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

        return {
            "success": True,
            "analysis_mode": "photo",
            "product_count": result["product_count"],
            "occupancy_percent": result["occupancy_percent"],
            "void_percent": result["void_percent"],
            "confidence": result.get("confidence", 0.0),
            "confidence_label": "높음" if result.get("confidence", 0.0) >= 85 else ("보통" if result.get("confidence", 0.0) >= 65 else "낮음"),
            "quality_score": 0,
            "quality_message": "실험용 분석",
            "box_confidence": 0.0,
            "pvi_score": None,
            "status": "experimental",
            "message": "파란색 제품을 대상으로 한 실험 분석 결과입니다.",
            "result_image_base64": image_base64,
            "visualization_message": "초록색은 추정 바닥 경계, 분홍색은 검출된 제품 영역입니다.",
            "disclaimer": "사진 기반 면적 참고값이며 실제 부피 또는 법령상 포장공간비율이 아닙니다.",
            "size_measurement": size_measurement,
        }
