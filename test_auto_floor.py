
from pathlib import Path
import cv2
import numpy as np

src = Path("test_images/generated.png")
out = Path("test_outputs/auto_floor.jpg")

image = cv2.imread(str(src))
if image is None:
    raise SystemExit("원본 이미지를 찾지 못했습니다.")

h, w = image.shape[:2]

# 1. 여러 높이에서 좌우 바닥 경계 후보 수집
left_x = []
right_x = []

for ratio in [0.28, 0.35, 0.72, 0.76]:
    y = int(h * ratio)
    strip = image[y-6:y+7]
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    profile = np.median(lab.astype(np.float32), axis=0)
    diff = np.linalg.norm(np.diff(profile, axis=0), axis=1)

    a, b = int(w * 0.06), int(w * 0.20)
    c, d = int(w * 0.80), int(w * 0.94)

    left_x.append(a + int(np.argmax(diff[a:b])))
    right_x.append(c + int(np.argmax(diff[c:d])))

left = int(np.median(left_x))
right = int(np.median(right_x))

# 2. 수평 직선 후보 검출
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 30, 100)

lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180,
    threshold=55,
    minLineLength=int(w * 0.12),
    maxLineGap=int(w * 0.06)
)

top_candidates = []
bottom_candidates = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(np.degrees(
            np.arctan2(y2-y1, x2-x1)
        )) % 180

        if not (angle < 12 or angle > 168):
            continue

        y = (int(y1) + int(y2)) / 2
        length = abs(int(x2) - int(x1))

        # 현재 합성 이미지 구도에 맞춘 탐색 구간
        if h * 0.23 < y < h * 0.40:
            top_candidates.append((length, y))

        if h * 0.70 < y < h * 0.83:
            bottom_candidates.append((length, y))

if not top_candidates or not bottom_candidates:
    raise SystemExit(
        "상하 경계 자동 검출 실패. "
        "임시 사각형 대신 재검토가 필요합니다."
    )

top = round(max(top_candidates)[1])
bottom = round(max(bottom_candidates)[1])

if not (0 <= left < right < w and 0 <= top < bottom < h):
    raise SystemExit("검출된 사각형 좌표가 유효하지 않습니다.")

# 3. 자동으로 계산된 네 모서리
corners = np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom]
], dtype=np.int32)

result = image.copy()
cv2.polylines(
    result, [corners], True,
    (0, 255, 0), 4
)

for pt in corners:
    cv2.circle(
        result, tuple(pt), 8,
        (0, 0, 255), -1
    )

out.parent.mkdir(exist_ok=True)
if not cv2.imwrite(str(out), result):
    raise RuntimeError("결과 이미지 저장 실패")

print("자동 검출 좌표:")
print("left =", left)
print("right =", right)
print("top =", top)
print("bottom =", bottom)
print("결과:", out)
