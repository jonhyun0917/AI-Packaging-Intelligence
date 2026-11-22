
from pathlib import Path
import cv2
import numpy as np

src = Path("test_images/IMG_8383 2.JPG")
out = Path("test_outputs/real_edges.jpg")

image = cv2.imread(str(src))
if image is None:
    raise SystemExit("실제 사진을 읽을 수 없습니다.")

h, w = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 30, 100)

lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180,
    threshold=40,
    minLineLength=int(min(h, w) * 0.08),
    maxLineGap=int(min(h, w) * 0.05)
)

result = image.copy()
horizontal = []
vertical = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(np.degrees(
            np.arctan2(y2-y1, x2-x1)
        )) % 180

        if angle < 15 or angle > 165:
            horizontal.append((x1, y1, x2, y2))
            color = (0, 255, 255)
        elif 75 < angle < 105:
            vertical.append((x1, y1, x2, y2))
            color = (255, 0, 255)
        else:
            continue

        cv2.line(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color, 2
        )

out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), result)

print("이미지 크기:", w, "x", h)
print("수평 후보:", len(horizontal))
print("수직 후보:", len(vertical))
print("결과 이미지:", out)
