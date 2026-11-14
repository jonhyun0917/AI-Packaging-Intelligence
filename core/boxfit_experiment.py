
from core.product_counter import count_products
from pathlib import Path
import cv2
import numpy as np

# ========================================
# 1. 테스트할 이미지
# ========================================

IMAGE_PATH = Path("test_images/test_B_top.png")
OUTPUT_PATH = Path("test_outputs/boxfit_integrated_v1.png")


def analyze_image(image_path, output_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"이미지를 읽을 수 없습니다: {image_path}"
        )

    h, w = image.shape[:2]

    # ====================================
    # 2. 파란색 제품 검출
    # ====================================

    hsv = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )

    product_mask = cv2.inRange(
        hsv,
        np.array([90, 70, 40]),
        np.array([135, 255, 255])
    )

    kernel = np.ones((5, 5), np.uint8)

    product_mask = cv2.morphologyEx(
        product_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    product_mask = cv2.morphologyEx(
        product_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ====================================
    # 3. 제품 영역 제외 후 경계 분석
    # ====================================

    lab = cv2.cvtColor(
        image, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    gx = cv2.Sobel(
        lab, cv2.CV_32F, 1, 0, ksize=3
    )

    gy = cv2.Sobel(
        lab, cv2.CV_32F, 0, 1, ksize=3
    )

    vertical_edges = np.linalg.norm(
        gx, axis=2
    )

    horizontal_edges = np.linalg.norm(
        gy, axis=2
    )

    exclusion = cv2.dilate(
        product_mask,
        np.ones((21, 21), np.uint8),
        iterations=1
    )

    vertical_edges[exclusion > 0] = 0
    horizontal_edges[exclusion > 0] = 0

    # ====================================
    # 4. 경계 후보 추출
    # ====================================

    x1 = int(w * 0.40)
    x2 = int(w * 0.60)

    y1 = int(h * 0.40)
    y2 = int(h * 0.60)

    x_profile = np.median(
        vertical_edges[y1:y2, :],
        axis=0
    )

    y_profile = np.median(
        horizontal_edges[:, x1:x2],
        axis=1
    )

    def smooth(profile):
        values = profile.astype(
            np.float32
        ).reshape(1, -1)

        return cv2.GaussianBlur(
            values, (21, 1), 0
        ).ravel()

    x_profile = smooth(x_profile)
    y_profile = smooth(y_profile)

    def strongest(profile, start, end):
        n = len(profile)

        a = int(n * start)
        b = int(n * end)

        region = profile[a:b]

        if region.size == 0:
            raise ValueError(
                "경계 탐색 범위가 비어 있습니다."
            )

        index = int(np.argmax(region))

        return a + index

    left = strongest(x_profile, 0.17, 0.38)
    right = strongest(x_profile, 0.62, 0.83)
    top = strongest(y_profile, 0.23, 0.43)
    bottom = strongest(y_profile, 0.57, 0.84)

    if left >= right or top >= bottom:
        raise ValueError(
            "바닥 경계 검출에 실패했습니다."
        )

    # ====================================
    # 5. 바닥 마스크
    # ====================================

    floor_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    floor_polygon = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ], dtype=np.int32)

    cv2.fillPoly(
        floor_mask,
        [floor_polygon],
        255
    )

    # ====================================
    # 6. 제품 면적 계산
    # ====================================

    inside_product_mask = cv2.bitwise_and(
        product_mask,
        floor_mask
    )


    # ====================================
    # 제품 개수 계산
    # ====================================

    count_mask = cv2.erode(
        inside_product_mask,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    count_contours, _ = cv2.findContours(
        count_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_product_area = max(
        100,
        int(cv2.countNonZero(floor_mask) * 0.01)
    )

    valid_contours = [
        contour
        for contour in count_contours
        if cv2.contourArea(contour) >= min_product_area
    ]

    product_count = count_products(image)

    print("검출된 제품 후보 개수:", product_count)


    product_area = int(
        cv2.countNonZero(
            inside_product_mask
        )
    )

    floor_area = int(
        cv2.countNonZero(
            floor_mask
        )
    )

    occupancy = (
        product_area / floor_area * 100
        if floor_area > 0
        else 0
    )

    void = 100 - occupancy

    # ====================================
    # 7. 결과 이미지 생성
    # ====================================

    result = image.copy()

    cv2.polylines(
        result,
        [floor_polygon],
        True,
        (0, 255, 0),
        4
    )

    contours, _ = cv2.findContours(
        inside_product_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        result,
        contours,
        -1,
        (255, 0, 255),
        3
    )

    cv2.putText(
        result,
        f"Occupancy: {occupancy:.1f}%",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Void: {void:.1f}%",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(
        str(output_path),
        result
    ):
        raise RuntimeError(
            "결과 이미지 저장 실패"
        )

    return {
        "product_count": product_count,
        "occupancy_percent": round(
            occupancy, 1
        ),
        "void_percent": round(
            void, 1
        ),
        "product_area_px": product_area,
        "floor_area_px": floor_area,
        "floor": {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        },
        "output": str(output_path),
    }


if __name__ == "__main__":
    result = analyze_image(
        IMAGE_PATH,
        OUTPUT_PATH
    )

    print("=== BoxFit 통합 분석 결과 ===")

    for key, value in result.items():
        print(f"{key}: {value}")
