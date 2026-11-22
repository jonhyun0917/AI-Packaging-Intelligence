
from pathlib import Path
import cv2
import numpy as np

image_path = Path("test_images/IMG_8383 2.JPG")
output_path = Path("test_outputs/box_filtered_lines.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise SystemExit("원본 이미지를 읽을 수 없습니다.")

h, w = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 35, 110)

# 박스 중앙의 제품·비닐 영역에서 발생한
# 불필요한 경계선을 줄이는 실험용 필터

filtered_edges = edges.copy()

x1 = int(w * 0.22)
x2 = int(w * 0.78)
y1 = int(h * 0.27)
y2 = int(h * 0.73)

filtered_edges[y1:y2, x1:x2] = 0

lines = cv2.HoughLinesP(
    filtered_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=55,
    minLineLength=int(min(h, w) * 0.13),
    maxLineGap=int(min(h, w) * 0.06),
)

result = image.copy()
count = 0

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3,
        )
        count += 1

output_path.parent.mkdir(exist_ok=True)

if not cv2.imwrite(str(output_path), result):
    raise RuntimeError("결과 이미지 저장 실패")

print("검출된 직선 수:", count)
print("결과 이미지:", output_path)

# 바깥쪽 직선 후보를 별도로 표시
border_result = image.copy()
border_count = 0

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        mid_x = (int(x1) + int(x2)) / 2
        mid_y = (int(y1) + int(y2)) / 2

        near_border = (
            mid_x < w * 0.28
            or mid_x > w * 0.72
            or mid_y < h * 0.25
            or mid_y > h * 0.75
        )

        if near_border:
            cv2.line(
                border_result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                3,
            )
            border_count += 1

border_output = Path("test_outputs/box_border_candidates.jpg")
cv2.imwrite(str(border_output), border_result)

print("박스 경계 후보 직선 수:", border_count)
print("경계 후보 이미지:", border_output)
