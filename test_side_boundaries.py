
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")
if image is None:
    raise SystemExit("이미지를 찾지 못했습니다.")

h, w = image.shape[:2]

# 파란색 제품을 피하기 위해 바닥의 아래쪽 가로 영역 분석
y = int(h * 0.74)
strip = image[max(0, y-8):min(h, y+9)]

lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
profile = np.median(lab.astype(np.float32), axis=0)

# 인접한 열 사이의 색상 변화량
difference = np.linalg.norm(
    np.diff(profile, axis=0),
    axis=1
)

# 노이즈 감소
difference = cv2.GaussianBlur(
    difference.reshape(1, -1),
    (1, 1), 0
).reshape(-1)

result = image.copy()

# 분석 위치: 노란색
cv2.line(
    result, (0, y), (w-1, y),
    (0, 255, 255), 2
)

# 왼쪽과 오른쪽에서 각각 강한 변화 후보 선택
left_range = (int(w*0.04), int(w*0.40))
right_range = (int(w*0.60), int(w*0.96))

for start, end in (left_range, right_range):
    candidates = difference[start:end]
    if candidates.size == 0:
        continue

    x = start + int(np.argmax(candidates))

    cv2.circle(
        result, (x, y),
        10, (0, 0, 255), -1
    )
    print("좌우 경계 후보 x:", x)

output = Path("test_outputs/side_boundaries.jpg")
output.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output), result)

print("결과:", output)
