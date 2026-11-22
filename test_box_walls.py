
from pathlib import Path
import cv2
import numpy as np

image_path = Path("test_images/generated.png")
output_path = Path("test_outputs/box_walls_test.jpg")

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit("테스트 이미지를 찾지 못했습니다.")

h, w = image.shape[:2]
result = image.copy()

# generated.png를 위한 임시 기준 좌표 (비율)
# 박스 날개는 제외하고 입구의 안쪽 테두리 지정
outer = np.array([
    [0.01, 0.20],  # 왼쪽 위
    [0.99, 0.20],  # 오른쪽 위
    [0.99, 0.83],  # 오른쪽 아래
    [0.01, 0.83],  # 왼쪽 아래
], dtype=np.float32)

# 박스 바닥의 네 모서리
inner = np.array([
    [0.11, 0.28],
    [0.89, 0.28],
    [0.89, 0.76],
    [0.11, 0.76],
], dtype=np.float32)

def to_px(points):
    return np.rint(
        points * [w, h]
    ).astype(np.int32)

outer = to_px(outer)
inner = to_px(inner)

# 네 벽면: 뒷벽 / 오른쪽 / 앞벽 / 왼쪽
wall_names = ["back", "right", "front", "left"]
wall_colors = [
    (70, 170, 230),
    (80, 190, 160),
    (160, 140, 230),
    (180, 120, 110),
]

overlay = result.copy()

for i in range(4):
    j = (i + 1) % 4
    poly = np.array([
        outer[i], outer[j], inner[j], inner[i]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], wall_colors[i])

# 벽면 반투명 표시
result = cv2.addWeighted(overlay, 0.30, result, 0.70, 0)

# 박스 입구: 빨간색
cv2.polylines(
    result, [outer], True, (0, 0, 255), 4
)

# 바닥 경계: 초록색
cv2.polylines(
    result, [inner], True, (0, 255, 0), 4
)

# HSV로 파란색 제품 영역 표시
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(
    hsv,
    np.array([90, 45, 35]),
    np.array([135, 255, 255])
)

# 제품 검출을 바닥 영역으로 제한
floor_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(floor_mask, [inner], 255)
blue_mask = cv2.bitwise_and(blue_mask, floor_mask)

contours, _ = cv2.findContours(
    blue_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(
        result, [largest], -1, (255, 0, 0), 4
    )

floor_area = cv2.contourArea(inner)
product_area = cv2.countNonZero(blue_mask)
occupancy = 100 * product_area / max(floor_area, 1)
void = 100 - occupancy

output_path.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output_path), result)

print("박스 바닥 면적(px²):", round(floor_area))
print("제품 색상 면적(px):", product_area)
print("사진상 점유율(%):", round(occupancy, 1))
print("사진상 빈 공간(%):", round(void, 1))
print("결과 이미지:", output_path)
