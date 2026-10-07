from io import BytesIO
import base64

import cv2
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


# --------------------------------------------------
# CORS 설정
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 서버 상태 확인
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "success": True,
        "message": "BoxFit AI Analysis API is running"
    }


# --------------------------------------------------
# 이미지 분석 API
# --------------------------------------------------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):

    # --------------------------------------------------
    # 1. 이미지 파일 여부 확인
    # --------------------------------------------------
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

        # --------------------------------------------------
        # 2. 이미지 품질 분석
        # --------------------------------------------------
        quality = check_image_quality(pil_image)


        # --------------------------------------------------
        # 3. 박스 검출 + 원근 보정
        # --------------------------------------------------
        _, _, box = detect_box(pil_image)


        # --------------------------------------------------
        # 4. 제품 영역 검출
        #
        # result_image:
        # core/detect.py에서 생성된
        # 제품 영역 + 예상 빈공간 시각화 이미지
        # --------------------------------------------------
        (
            detections,
            result_image,
            product_area,
            product_count,
        ) = detect_products(
            pil_image,
            box
        )


        # --------------------------------------------------
        # 5. 사진 기준 면적 분석
        # --------------------------------------------------
        box_w = max(
            float(box["warped_width"]),
            1.0
        )

        box_h = max(
            float(box["warped_height"]),
            1.0
        )

        planar = calculate_planar_measurements(
            box_w,
            box_h,
            box_w,
            box_h,
            product_area,
        )


        # --------------------------------------------------
        # 6. PVI 참고값
        # --------------------------------------------------
        pvi = calculate_pvi(
            planar["box_area_px"],
            planar["product_area_px"],
        )


        # --------------------------------------------------
        # 7. 분석 신뢰도
        # --------------------------------------------------
        confidence = estimate_measurement_confidence(
            float(
                box.get(
                    "confidence",
                    0
                )
            ),
            [
                float(
                    d.get(
                        "confidence",
                        0
                    )
                )
                for d in detections
            ],
            bool(
                box.get(
                    "perspective_corrected",
                    False
                )
            ),
            False,
        )


        # --------------------------------------------------
        # 8. 제품 점유율 / 예상 빈공간
        # --------------------------------------------------
        occupancy = float(
            planar["occupancy_percent"]
        )

        void = float(
            planar["void_percent"]
        )


        # --------------------------------------------------
        # 9. 분석 상태 메시지
        # --------------------------------------------------
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


        # --------------------------------------------------
        # 10. 신뢰도 단계
        # --------------------------------------------------
        if confidence >= 80:

            confidence_label = "높음"

        elif confidence >= 60:

            confidence_label = "보통"

        else:

            confidence_label = "낮음"


        # --------------------------------------------------
        # 11. 시각화 결과 이미지를 PNG로 변환
        # --------------------------------------------------
        success, encoded_image = cv2.imencode(
            ".png",
            result_image
        )

        if not success:
            raise ValueError(
                "분석 결과 이미지 인코딩에 실패했습니다."
            )


        # --------------------------------------------------
        # 12. PNG → Base64
        #
        # FlutterFlow에서 받을 값
        # --------------------------------------------------
        result_image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")


        # --------------------------------------------------
        # 13. API 응답
        # --------------------------------------------------
        return {

            "success": True,

            "analysis_mode": "photo",


            # --------------------------
            # 제품 분석
            # --------------------------
            "product_count":
                product_count,


            "occupancy_percent":
                round(
                    occupancy,
                    1
                ),

            "void_percent":
                round(
                    void,
                    1
                ),


            # --------------------------
            # 신뢰도
            # --------------------------
            "confidence":
                round(
                    confidence,
                    1
                ),

            "confidence_label":
                confidence_label,


            # --------------------------
            # 이미지 품질
            # --------------------------
            "quality_score":
                quality["score"],

            "quality_message":
                quality["message"],


            # --------------------------
            # 박스 검출 신뢰도
            # --------------------------
            "box_confidence":
                round(
                    float(
                        box["confidence"]
                    )
                    * 100,
                    1
                ),


            # --------------------------
            # PVI
            # --------------------------
            "pvi_score":
                pvi["pvi"],


            # --------------------------
            # 상태
            # --------------------------
            "status":
                status,

            "message":
                message,


            # --------------------------
            # 새로 추가된 핵심 값
            #
            # 제품 = 초록색
            # 예상 빈공간 = 주황색
            # --------------------------
            "result_image_base64":
                result_image_base64,


            # --------------------------
            # 안내문
            # --------------------------
            "visualization_message":
                (
                    "색상으로 표시된 영역은 사진을 기반으로 "
                    "추정한 제품 영역과 예상 빈공간입니다."
                ),


            "disclaimer":
                (
                    "사진 기반 값은 면적 비율에 따른 참고 분석이며, "
                    "법령상 공식 포장공간비율 또는 포장검사 결과를 "
                    "의미하지 않습니다."
                ),
        }


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "분석 중 오류가 발생했습니다: "
                f"{type(e).__name__}"
            )
        )
