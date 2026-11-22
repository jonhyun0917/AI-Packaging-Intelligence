
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")
if image is None:
    raise SystemExit("이미지를 찾지 못했습니다.")

h, w = image.shape[:2]
result = image.copy()

# 이전 자동 검출 실험에서 얻은 좌표
# 현재는 테스트값을 연결하는 단계
entrance = np.array([
    [11, 230],
    [1071, 230],
    [1071, 1232],
    [11, 1232]
], dtype=np.int32)

floor = np.array([
    [119, 376],
    [949, 376],
    [949, 1145],
    [119, 1145]
], dtype=np.int32)

# 1. 네 벽면 표시
wall_colors = [
    (80, 160, 220),
    (120, 200, 140),
    (180, 140, 220),
    (200, 150, 100)
]

overlay = result.copy()

for i in range(4):
    j = (i + 1) % 4
    polygon = np.array([
        entrance[i], entrance[j],
        floor[j], floor[i]
    ], dtype=np.int32)

    cv2.fillPoly(overlay, [polygon], wall_colors[i])

result = cv2.addWeighted(
    overlay, 0.30, result, 0.70, 0
)

# 2. 박스 입구와 바닥 경계
cv2.polylines(result, [entrance], True, (0, 0, 255), 4)
cv2.polylines(result, [floor], True, (0, 255, 0), 4)

# 3. 제품 색상 검출
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
product_mask = cv2.inRange(
    hsv,
    np.array([90, 45, 35]),
    np.array([135, 255, 255])
)

floor_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(floor_mask, [floor], 255)

product_mask = cv2.bitwise_and(
    product_mask, floor_mask
)

contours, _ = cv2.findContours(
    product_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(
        result, [largest], -1,
        (0, 255, 0), 4
    )

# 4. 사진상 면적 계산
floor_area = cv2.countNonZero(floor_mask)
product_area = cv2.countNonZero(product_mask)

occupancy = 100 * product_area / max(floor_area, 1)
void = 100 - occupancy

output = Path("test_outputs/integrated_box.jpg")
output.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output), result)

print("사진상 제품 점유율:", round(occupancy, 1), "%")
print("사진상 예상 빈 면적:", round(void, 1), "%")
print("결과 이미지:", output)
