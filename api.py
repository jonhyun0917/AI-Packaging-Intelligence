from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from analysis.pvi import calculate_pvi
from core.box_detector import detect_box
from core.detect import detect_products
from core.measurement_engine import (
    calculate_planar_measurements,
    estimate_measurement_confidence,
)
from core.quality_checker import check_image_quality


app = FastAPI(title="BoxFit AI Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "success": True,
        "message": "BoxFit AI Analysis API is running"
    }


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드할 수 있습니다."
        )

    raw = await image.read()

    try:
        pil_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="이미지를 읽을 수 없습니다."
        )

    try:
        # 1. 이미지 품질
        quality = check_image_quality(pil_image)

        # 2. 박스 검출 + 원근 보정
        _, _, box = detect_box(pil_image)

        # 3. 제품 영역 검출
        detections, _, product_area, product_count = detect_products(
            pil_image,
            box
        )

        # 4. 사진 기준 면적 분석
        box_w = max(float(box["warped_width"]), 1.0)
        box_h = max(float(box["warped_height"]), 1.0)

        planar = calculate_planar_measurements(
            box_w,
            box_h,
            box_w,
            box_h,
            product_area,
        )

        # 5. PVI 참고값
        pvi = calculate_pvi(
            planar["box_area_px"],
            planar["product_area_px"],
        )

        # 6. 분석 신뢰도
        confidence = estimate_measurement_confidence(
            float(box.get("confidence", 0)),
            [
                float(d.get("confidence", 0))
                for d in detections
            ],
            bool(box.get("perspective_corrected", False)),
            False,
        )

        occupancy = float(planar["occupancy_percent"])
        void = float(planar["void_percent"])

        if void >= 60:
            status = "improvement_review"
            message = (
                "현재 사진에서는 박스에 비해 제품이 차지하는 "
                "영역이 작은 편으로 분석됐어요. "
                "더 작은 박스를 비교해볼 여지가 있습니다."
            )
        elif void >= 40:
            status = "review"
            message = (
                "현재 사진에서는 일정 수준의 빈 공간이 확인됐어요. "
                "제품 보호 조건과 함께 박스 크기를 비교해보세요."
            )
        else:
            status = "efficient"
            message = (
                "현재 사진에서는 제품이 박스 내부 공간을 "
                "비교적 효율적으로 사용하고 있어요."
            )

        if confidence >= 80:
            confidence_label = "높음"
        elif confidence >= 60:
            confidence_label = "보통"
        else:
            confidence_label = "낮음"

        return {
            "success": True,
            "analysis_mode": "photo",

            "product_count": product_count,

            "occupancy_percent": round(occupancy, 1),
            "void_percent": round(void, 1),

            "confidence": round(confidence, 1),
            "confidence_label": confidence_label,

            "quality_score": quality["score"],
            "quality_message": quality["message"],

            "box_confidence":
                round(float(box["confidence"]) * 100, 1),

            "pvi_score": pvi["pvi"],

            "status": status,
            "message": message,

            "disclaimer":
                "사진 기반 값은 면적 비율에 따른 참고 분석이며, "
                "법령상 공식 포장공간비율 또는 포장검사 결과를 "
                "의미하지 않습니다."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {type(e).__name__}"
        )
