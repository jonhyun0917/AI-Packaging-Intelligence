
from pathlib import Path
import cv2
import numpy as np

IMAGE = Path("test_images/generated.png")
OUTPUT = Path("test_outputs/auto_walls_test.jpg")

image = cv2.imread(str(IMAGE))
if image is None:
    raise SystemExit("테스트 이미지를 찾을 수 없습니다.")

h, w = image.shape[:2]

# 1. 파란색 제품 영역 제외
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
product_mask = cv2.inRange(
    hsv,
    np.array([90, 45, 35]),
    np.array([135, 255, 255])
)

# 2. 박스 경계 검출을 위한 전처리
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 30, 100)

# 제품 주변의 불필요한 선 제외
kernel = np.ones((15, 15), np.uint8)
excluded = cv2.dilate(product_mask, kernel)
edges[excluded > 0] = 0

# 3. 수평/수직 직선 검출
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=55,
    minLineLength=int(min(h, w) * 0.12),
    maxLineGap=int(min(h, w) * 0.06)
)

horizontal = []
vertical = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(np.degrees(
            np.arctan2(y2 - y1, x2 - x1)
        )) % 180

        segment = (int(x1), int(y1), int(x2), int(y2))

        if angle < 12 or angle > 168:
            horizontal.append(segment)
        elif 78 < angle < 102:
            vertical.append(segment)

# 4. 결과 이미지 표시
result = image.copy()

for x1, y1, x2, y2 in horizontal:
    cv2.line(
        result, (x1, y1), (x2, y2),
        (0, 255, 255), 3
    )

for x1, y1, x2, y2 in vertical:
    cv2.line(
        result, (x1, y1), (x2, y2),
        (255, 0, 255), 3
    )

OUTPUT.parent.mkdir(exist_ok=True)

if not cv2.imwrite(str(OUTPUT), result):
    raise RuntimeError("이미지 저장에 실패했습니다.")

print("수평 직선 후보:", len(horizontal))
print("수직 직선 후보:", len(vertical))
print("결과 이미지:", OUTPUT)

# 박스 입구/바닥 경계 대표선 후보 추정
# 현재 generated.png의 정면 촬영 구도에 한정한 실험

def representative_lines(segments, axis, size, bins=30):
    if not segments:
        return []

    values = []
    for x1, y1, x2, y2 in segments:
        if axis == "horizontal":
            position = (y1 + y2) / 2
            length = abs(x2 - x1)
        else:
            position = (x1 + x2) / 2
            length = abs(y2 - y1)

        if length > 0:
            values.append((position, length))

    if not values:
        return []

    groups = {}

    for position, length in values:
        key = int(position / max(size / bins, 1))
        groups.setdefault(key, []).append((position, length))

    representatives = []

    for group in groups.values():
        positions = [p for p, _ in group]
        lengths = [length for _, length in group]
        weighted_position = np.average(
            positions, weights=lengths
        )
        total_length = sum(lengths)

        representatives.append(
            (float(weighted_position), float(total_length))
        )

    return sorted(
        representatives,
        key=lambda item: item[1],
        reverse=True
    )

horizontal_candidates = representative_lines(
    horizontal, "horizontal", h
)

vertical_candidates = representative_lines(
    vertical, "vertical", w
)

print("수평 대표선 상위 8개:")
for position, strength in horizontal_candidates[:8]:
    print(f"y={position:.1f}, 강도={strength:.0f}")

print("수직 대표선 상위 8개:")
for position, strength in vertical_candidates[:8]:
    print(f"x={position:.1f}, 강도={strength:.0f}")
