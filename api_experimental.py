
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import base64

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from core.boxfit_experiment import analyze_image


app = FastAPI(title="BoxFit Experimental API")


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
            "confidence": 0.0,
            "confidence_label": "미검증",
            "quality_score": 0,
            "quality_message": "실험용 분석",
            "box_confidence": 0.0,
            "pvi_score": None,
            "status": "experimental",
            "message": "파란색 제품을 대상으로 한 실험 분석 결과입니다.",
            "result_image_base64": image_base64,
            "visualization_message": "초록색은 추정 바닥 경계, 분홍색은 검출된 제품 영역입니다.",
            "disclaimer": "사진 기반 면적 참고값이며 실제 부피 또는 법령상 포장공간비율이 아닙니다."
        }
