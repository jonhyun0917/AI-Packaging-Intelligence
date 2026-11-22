
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/generated.png")
if image is None:
    raise SystemExit("테스트 이미지를 찾지 못했습니다.")

h, w = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 30, 100)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=55,
    minLineLength=int(min(h, w) * 0.12),
    maxLineGap=int(min(h, w) * 0.06),
)

top_lines = []
bottom_lines = []
left_lines = []
right_lines = []

if lines is None:
    raise SystemExit("직선 후보를 찾지 못했습니다.")

for x1, y1, x2, y2 in lines.reshape(-1, 4):
    dx = x2 - x1
    dy = y2 - y1
    angle = abs(np.degrees(np.arctan2(dy, dx))) % 180

    if angle < 12 or angle > 168:
        y = (y1 + y2) / 2
        length = abs(dx)

        if h * 0.12 < y < h * 0.23:
            top_lines.append((length, y))

        if h * 0.80 < y < h * 0.90:
            bottom_lines.append((length, y))

    elif 78 < angle < 102:
        x = (x1 + x2) / 2
        length = abs(dy)

        if x < w * 0.10:
            left_lines.append((length, x))

        if x > w * 0.90:
            right_lines.append((length, x))

groups = [
    top_lines,
    bottom_lines,
    left_lines,
    right_lines,
]

if any(not group for group in groups):
    raise SystemExit(
        "입구의 네 경계를 모두 찾지 못했습니다."
    )

# 각 위치에서 가장 긴 직선 후보 선택
top = round(max(top_lines)[1])
bottom = round(max(bottom_lines)[1])
left = round(max(left_lines)[1])
right = round(max(right_lines)[1])

if not (left < right and top < bottom):
    raise SystemExit("입구 좌표가 올바르지 않습니다.")

entrance = np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
], dtype=np.int32)

result = image.copy()

cv2.polylines(
    result, [entrance], True,
    (0, 0, 255), 4
)

for pt in entrance:
    cv2.circle(
        result, tuple(pt), 8,
        (255, 0, 0), -1
    )

output = Path("test_outputs/auto_entrance.jpg")
output.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output), result)

print("자동 입구 검출 좌표")
print("left:", left)
print("right:", right)
print("top:", top)
print("bottom:", bottom)
print("결과 이미지:", output)
