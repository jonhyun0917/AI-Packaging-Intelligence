
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")

if image is None:
    raise SystemExit("이미지를 찾을 수 없습니다.")

result = image.copy()

# 앞선 실험에서 얻은 박스 바닥 경계 후보
# 아직 완전 자동 검출은 아님
left = 119
right = 949
top = 377
bottom = 1144

corners = np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom]
], dtype=np.int32)

# 바닥 영역 표시
overlay = result.copy()
cv2.fillPoly(
    overlay,
    [corners],
    (0, 255, 0)
)

result = cv2.addWeighted(
    overlay, 0.15,
    result, 0.85, 0
)

# 초록색 사각형
cv2.polylines(
    result,
    [corners],
    True,
    (0, 255, 0),
    4
)

# 네 모서리 표시
for point in corners:
    cv2.circle(
        result,
        tuple(point),
        9,
        (0, 0, 255),
        -1
    )

# 파란색 제품 영역 확인
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

product_mask = cv2.inRange(
    hsv,
    np.array([90, 45, 35]),
    np.array([135, 255, 255])
)

floor_mask = np.zeros(
    image.shape[:2],
    dtype=np.uint8
)

cv2.fillPoly(floor_mask, [corners], 255)

product_mask = cv2.bitwise_and(
    product_mask,
    floor_mask
)

floor_area = cv2.countNonZero(floor_mask)
product_area = cv2.countNonZero(product_mask)

occupancy = 100 * product_area / max(floor_area, 1)
void = 100 - occupancy

output = Path("test_outputs/floor_rectangle.jpg")
output.parent.mkdir(exist_ok=True)

cv2.imwrite(str(output), result)

print("박스 바닥 면적:", floor_area)
print("제품 색상 면적:", product_area)
print("사진상 제품 점유율:", round(occupancy, 1), "%")
print("사진상 예상 빈 면적:", round(void, 1), "%")
print("결과 이미지:", output)
