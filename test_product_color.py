
from pathlib import Path
import cv2
import numpy as np

image_path = Path("test_images/generated.png")
output_path = Path("test_outputs/product_color_test.jpg")

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit("원본 사진을 찾지 못했습니다.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 진한 파란색 계열 검출 (실험용 범위)
lower_blue = np.array([90, 45, 35])
upper_blue = np.array([135, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)
mask = cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE, kernel
)

# 색상으로 검출한 영역의 작은 빈틈 연결
# 실험용 마스크이며 기존 API 계산에는 적용하지 않음

shape_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (17, 17),
)

expanded_mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    shape_kernel,
    iterations=1,
)

contours, _ = cv2.findContours(
    expanded_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

shape_result = image.copy()

if contours:
    largest = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        shape_result,
        [largest],
        -1,
        (0, 255, 0),
        4,
    )

shape_output = Path("test_outputs/product_shape_test.jpg")
cv2.imwrite(str(shape_output), shape_result)

print("제품 외곽선 이미지:", shape_output)


result = image.copy()
overlay = image.copy()
overlay[mask > 0] = (0, 255, 0)

result = cv2.addWeighted(
    overlay, 0.4, image, 0.6, 0
)

output_path.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output_path), result)

print("파란색 검출 픽셀:", cv2.countNonZero(mask))
print("결과 이미지:", output_path)

# 기존 AI 제품 검출과 HSV 색상 검출 비교
from PIL import Image
from core.box_detector import detect_box
from core.detect import detect_products

pil_image = Image.open(image_path).convert("RGB")

_, _, box = detect_box(pil_image)

detections, analysis_image, product_area, product_count = (
    detect_products(pil_image, box)
)

hsv_area = int(cv2.countNonZero(mask))

print("=== 제품 인식 방식 비교 ===")
print("기존 AI 제품 면적:", product_area)
print("HSV 파란색 면적:", hsv_area)
print("면적 차이:", product_area - hsv_area)
print("기존 검출 제품 개수:", product_count)
print("박스 자동 검출:", box["auto_detected"])
