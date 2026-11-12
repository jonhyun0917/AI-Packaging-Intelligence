
from pathlib import Path
import cv2
import numpy as np

image = cv2.imread("test_images/test_B_top.png")
if image is None:
    raise SystemExit("이미지를 읽지 못했습니다.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(
    hsv,
    np.array([90, 70, 40]),
    np.array([135, 255, 255])
)

# 작은 색상 잡음 제거
mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8)
)

# 중앙의 실제 빈 가로 틈 찾기
row_counts = np.count_nonzero(mask, axis=1)
rows = np.where(row_counts > 20)[0]

if rows.size == 0:
    raise SystemExit("제품 영역을 찾지 못했습니다.")

top, bottom = int(rows.min()), int(rows.max())
start = top + (bottom - top) // 3
end = top + (bottom - top) * 2 // 3

split_y = start + int(
    np.argmin(row_counts[start:end + 1])
)


# 실제 가로 틈인지 검증
count_mask = mask.copy()

product_rows = np.where(row_counts > 20)[0]
product_height = int(product_rows.max() - product_rows.min() + 1)

# 중앙 분리선 주변의 파란색 픽셀 수
window_start = max(0, split_y - 2)
window_end = min(len(row_counts), split_y + 3)

nearby_counts = row_counts[window_start:window_end]

# 위아래 충분한 영역에 제품이 존재하는지 확인
upper_pixels = int(np.count_nonzero(mask[top:split_y, :]))
lower_pixels = int(np.count_nonzero(mask[split_y + 1:bottom + 1, :]))

# 분리선 부근의 파란색 영역이 거의 없어야 실제 틈으로 판단
gap_is_clear = bool(
    nearby_counts.size > 0
    and np.max(nearby_counts) <= max(5, int(np.max(row_counts) * 0.03))
)

has_two_sides = (
    upper_pixels > 0.15 * np.count_nonzero(mask)
    and lower_pixels > 0.15 * np.count_nonzero(mask)
)

# 충분한 간격이 확인됐을 때만 분리
if gap_is_clear and has_two_sides and product_height > 30:
    count_mask[max(0, split_y - 3):split_y + 4, :] = 0
    print("가로 분리 적용")
else:
    print("가로 분리 생략")


# 연결된 영역 개수 계산
contours, _ = cv2.findContours(
    count_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

min_area = max(
    100,
    int(np.count_nonzero(mask) * 0.01)
)

valid = [
    c for c in contours
    if cv2.contourArea(c) >= min_area
]

result = image.copy()
cv2.drawContours(result, valid, -1, (0, 255, 0), 3)
cv2.line(
    result, (0, split_y),
    (image.shape[1] - 1, split_y),
    (0, 0, 255), 2
)

out = Path("test_outputs/product_split_v2.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), result)

print("가로 분리선:", split_y)
print("검출된 제품 개수:", len(valid))
print("결과 이미지:", out)
