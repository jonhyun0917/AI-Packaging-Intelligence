
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/test_B_top.png")

if image is None:
    raise SystemExit("테스트 이미지를 찾지 못했습니다.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(
    hsv,
    np.array([90, 70, 40]),
    np.array([135, 255, 255])
)

# 각 가로줄에서 파란색 픽셀 개수 확인
row_counts = np.count_nonzero(mask, axis=1)

# 파란 제품이 존재하는 세로 범위
rows = np.where(row_counts > 20)[0]

if rows.size == 0:
    raise SystemExit("파란 제품을 검출하지 못했습니다.")

top = int(rows.min())
bottom = int(rows.max())
mid = (top + bottom) // 2

print("제품 위쪽:", top)
print("제품 아래쪽:", bottom)
print("제품 중앙:", mid)

# 중앙 부근에서 파란 픽셀이 가장 적은 줄
search_start = top + (bottom - top) // 3
search_end = top + (bottom - top) * 2 // 3

split_y = int(
    search_start
    + np.argmin(row_counts[search_start:search_end + 1])
)

print("가로 분리 후보:", split_y)
print("해당 줄의 파란 픽셀 수:", row_counts[split_y])

result = image.copy()

cv2.line(
    result,
    (0, split_y),
    (result.shape[1] - 1, split_y),
    (0, 0, 255),
    3
)

out = Path("test_outputs/product_row_split.png")
out.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(out), result)

print("결과 이미지:", out)

