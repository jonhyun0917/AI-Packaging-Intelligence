
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")
if image is None:
    raise SystemExit("이미지를 읽을 수 없습니다.")

h, w = image.shape[:2]
result = image.copy()

# 파란색 제품의 상단과 하단을 피해
# 여러 높이에서 좌우 경계를 검출
sample_heights = [0.28, 0.35, 0.72, 0.76]

left_points = []
right_points = []

for ratio in sample_heights:
    y = int(h * ratio)

    strip = image[max(0, y-6):min(h, y+7)]
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    profile = np.median(lab.astype(np.float32), axis=0)

    diff = np.linalg.norm(
        np.diff(profile, axis=0),
        axis=1
    )

    # 좌우에서 강한 색상 변화 후보 찾기
    ranges = [
        (int(w*0.06), int(w*0.20)),
        (int(w*0.80), int(w*0.94)),
    ]

    points = []

    for start, end in ranges:
        x = start + int(np.argmax(diff[start:end]))
        points.append((x, y))

    left_points.append(points[0])
    right_points.append(points[1])

    cv2.circle(result, points[0], 8, (0, 255, 0), -1)
    cv2.circle(result, points[1], 8, (0, 0, 255), -1)

    print(f"y={y} | 왼쪽={points[0][0]} | 오른쪽={points[1][0]}")

# 각 측면의 검출 지점을 연결
cv2.polylines(
    result,
    [np.array(left_points, dtype=np.int32)],
    False, (0, 255, 0), 3
)

cv2.polylines(
    result,
    [np.array(right_points, dtype=np.int32)],
    False, (0, 0, 255), 3
)

output = Path("test_outputs/wall_profile_v2.jpg")
output.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output), result)

print("결과 이미지:", output)
