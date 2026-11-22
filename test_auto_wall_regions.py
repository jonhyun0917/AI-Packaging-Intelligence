
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")
if image is None:
    raise SystemExit("이미지를 찾을 수 없습니다.")

h, w = image.shape[:2]
result = image.copy()

# 앞 단계에서 자동 검출한 바닥 좌표
# 현재는 결과값을 사용하며, 아직 자동 함수로 연결하지 않음
floor = np.array([
    [119, 376],
    [949, 376],
    [949, 1145],
    [119, 1145]
], dtype=np.int32)

# 박스 입구 후보 좌표 (실험용)
# 위쪽과 아래쪽 박스 날개는 제외
entrance = np.array([
    [12, 230],
    [1074, 230],
    [1074, 1235],
    [12, 1235]
], dtype=np.int32)

wall_names = [
    "Back Wall",
    "Right Wall",
    "Front Wall",
    "Left Wall"
]

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
        entrance[i],
        entrance[j],
        floor[j],
        floor[i]
    ], dtype=np.int32)

    cv2.fillPoly(
        overlay,
        [polygon],
        wall_colors[i]
    )

result = cv2.addWeighted(
    overlay, 0.30,
    result, 0.70,
    0
)

# 입구: 빨간색
cv2.polylines(
    result, [entrance], True,
    (0, 0, 255), 4
)

# 바닥: 초록색
cv2.polylines(
    result, [floor], True,
    (0, 255, 0), 4
)

output = Path("test_outputs/auto_wall_regions.jpg")
output.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output), result)

print("벽면 영역 수:", len(wall_names))
print("결과 이미지:", output)
