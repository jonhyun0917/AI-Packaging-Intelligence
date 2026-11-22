from pathlib import Path
import cv2
import numpy as np

# =========================================
# 입력 / 출력 경로
# =========================================
image_path = Path("test_images/test_D_top.png")
output_path = Path("test_outputs/integrated_box_auto_v2.jpg")

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit("테스트 이미지를 찾지 못했습니다.")

h, w = image.shape[:2]
result = image.copy()

# =========================================
# 1. 제품(파란 상자) 검출
# =========================================
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 파란색 범위
lower_blue = np.array([90, 80, 60])
upper_blue = np.array([130, 255, 255])

product_mask = cv2.inRange(hsv, lower_blue, upper_blue)

kernel = np.ones((5, 5), np.uint8)
product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_OPEN, kernel)
product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


if not contours:
    print("빈 박스 감지: 파란색 제품이 없습니다.")
    print("제품 면적을 0으로 계산합니다.")



# 검출된 파란색 영역 전체의 픽셀 수 계산
product_area = int(cv2.countNonZero(product_mask))

# 모든 제품 윤곽선 표시
cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    3
)


# =========================================
# 2. 입구(상자 윗면 안쪽 사각형) 자동 검출
#    - 기존 auto_entrance 실험 흐름을 단순화한 버전
# =========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 35, 110)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=80,
    minLineLength=120,
    maxLineGap=20
)

if lines is None:
    raise SystemExit("입구 검출용 직선을 찾지 못했습니다.")

left_lines = []
right_lines = []
top_lines = []
bottom_lines = []

for line in lines.reshape(-1, 4):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)

    # 수직선 후보
    if abs(dx) < 35 and abs(dy) > 80:
        x_mean = (x1 + x2) / 2
        y_mean = (y1 + y2) / 2

        # 중앙 바깥쪽만 사용
        if x_mean < w * 0.25:
            left_lines.append((length, x_mean))
        elif x_mean > w * 0.75:
            right_lines.append((length, x_mean))

    # 수평선 후보
    if abs(dy) < 35 and abs(dx) > 120:
        x_mean = (x1 + x2) / 2
        y_mean = (y1 + y2) / 2

        # 윗쪽/아랫쪽 경계 후보
        if y_mean < h * 0.35:
            top_lines.append((length, y_mean))
        elif y_mean > h * 0.75:
            bottom_lines.append((length, y_mean))

if not left_lines or not right_lines or not top_lines or not bottom_lines:
    raise SystemExit("입구 4개 경계를 모두 찾지 못했습니다.")

left = round(max(left_lines)[1])
right = round(max(right_lines)[1])
top = round(max(top_lines)[1])
bottom = round(max(bottom_lines)[1])

if left >= right or top >= bottom:
    raise SystemExit("입구 좌표가 올바르지 않습니다.")

entrance = np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom]
], dtype=np.int32)

# =========================================
# 박스 입구와 바닥 사이의 4개 벽면 표시
# =========================================
# =========================================
# 3. 바닥 사각형 자동 검출
#    - 입구보다 안쪽에 있는 바닥 영역을 찾는 방식
# =========================================
# 바닥은 중앙 근처의 갈색 큰 직사각형이라고 보고,
# 입구 사각형 안에서 안쪽 여백이 있는 영역으로 추정
margin_x = int((right - left) * 0.10)
margin_top = int((bottom - top) * 0.15)
margin_bottom = int((bottom - top) * 0.09)

floor_left = left + margin_x
floor_right = right - margin_x
floor_top = top + margin_top
floor_bottom = bottom - margin_bottom

if floor_left >= floor_right or floor_top >= floor_bottom:
    raise SystemExit("바닥 좌표가 올바르지 않습니다.")

floor_rect = np.array([
    [floor_left, floor_top],
    [floor_right, floor_top],
    [floor_right, floor_bottom],
    [floor_left, floor_bottom]
], dtype=np.int32)

wall_colors = [
    (80, 160, 220),    # 뒷벽
    (120, 200, 140),   # 오른쪽 벽
    (180, 140, 220),   # 앞벽
    (200, 150, 100),   # 왼쪽 벽
]

overlay = result.copy()

for i in range(4):
    j = (i + 1) % 4

    wall = np.array([
        entrance[i],
        entrance[j],
        floor_rect[j],
        floor_rect[i],
    ], dtype=np.int32)

    cv2.fillPoly(overlay, [wall], wall_colors[i])

result = cv2.addWeighted(
    overlay, 0.30,
    result, 0.70,
    0
)

# 입구 표시 (빨강)
cv2.polylines(result, [entrance], True, (0, 0, 255), 4)
for pt in entrance:
    cv2.circle(result, tuple(pt), 6, (255, 0, 0), -1)



# 바닥 표시 (초록)
cv2.polylines(result, [floor_rect], True, (0, 255, 0), 4)
for pt in floor_rect:
    cv2.circle(result, tuple(pt), 5, (0, 0, 255), -1)

# =========================================
# 4. 면적 계산
# =========================================
floor_area = (floor_right - floor_left) * (floor_bottom - floor_top)
occupancy_percent = round((product_area / floor_area) * 100, 1)
void_percent = round(100 - occupancy_percent, 1)

print("=== 통합 자동 검출 결과 ===")
print(f"입구 좌표: left={left}, right={right}, top={top}, bottom={bottom}")
print(f"바닥 좌표: left={floor_left}, right={floor_right}, top={floor_top}, bottom={floor_bottom}")
print(f"제품 면적(px²): {round(product_area)}")
print(f"바닥 면적(px²): {floor_area}")
print(f"사진상 제품 점유율(%): {occupancy_percent}")
print(f"사진상 빈 면적률(%): {void_percent}")

# =========================================
# 5. 텍스트 표시
# =========================================
cv2.putText(
    result,
    f"Occupancy: {occupancy_percent}%",
    (40, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    4,
    cv2.LINE_AA
)
cv2.putText(
    result,
    f"Occupancy: {occupancy_percent}%",
    (40, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    result,
    f"Void: {void_percent}%",
    (40, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    4,
    cv2.LINE_AA
)
cv2.putText(
    result,
    f"Void: {void_percent}%",
    (40, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# =========================================
# 6. 저장
# =========================================
output_path = Path("test_outputs/result_D_top.png")
cv2.imwrite(str(output_path), result)
print(f"결과 이미지: {output_path}")