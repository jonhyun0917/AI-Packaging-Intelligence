
from pathlib import Path
import cv2
import numpy as np

src = Path("test_images/test_D_top.png")
out = Path("test_outputs/floor_edges_v2.png")

image = cv2.imread(str(src))

if image is None:
    raise SystemExit("빈 박스 이미지를 찾지 못했습니다.")

h, w = image.shape[:2]

# 1. LAB 색상 공간으로 변환
lab = cv2.cvtColor(
    image, cv2.COLOR_BGR2LAB
).astype(np.float32)

# 2. 색상·밝기 변화량 계산
gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)

vertical_edges = np.linalg.norm(gx, axis=2)
horizontal_edges = np.linalg.norm(gy, axis=2)

# 제품이 없는 중앙 영역에서 경계 후보 수집
x1, x2 = int(w * 0.40), int(w * 0.60)
y1, y2 = int(h * 0.40), int(h * 0.60)

x_profile = np.median(
    vertical_edges[y1:y2, :], axis=0
)
y_profile = np.median(
    horizontal_edges[:, x1:x2], axis=1
)

# 노이즈 감소
def smooth(values):
    v = values.astype(np.float32).reshape(1, -1)
    return cv2.GaussianBlur(v, (21, 1), 0).ravel()

x_profile = smooth(x_profile)
y_profile = smooth(y_profile)

# 3. 각 방향의 경계 검색 범위
# 현재 정면 촬영 테스트 사진에 맞춘 범위
def strongest(profile, start_ratio, end_ratio):
    n = len(profile)
    start = int(n * start_ratio)
    end = int(n * end_ratio)

    region = profile[start:end]
    if region.size == 0:
        raise ValueError("검색 범위가 비어 있습니다.")

    index = int(np.argmax(region))
    return start + index, float(region[index])

left, ls = strongest(x_profile, 0.15, 0.40)
right, rs = strongest(x_profile, 0.60, 0.85)
top, ts = strongest(y_profile, 0.22, 0.43)
bottom, bs = strongest(y_profile, 0.57, 0.85)

if left >= right or top >= bottom:
    raise SystemExit("바닥 경계 후보가 올바르지 않습니다.")

# 4. 검출 결과 표시
result = image.copy()

corners = np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom]
], dtype=np.int32)

cv2.polylines(
    result, [corners], True,
    (0, 255, 0), 4
)

for pt in corners:
    cv2.circle(
        result, tuple(pt), 9,
        (0, 0, 255), -1
    )

out.parent.mkdir(exist_ok=True)

if not cv2.imwrite(str(out), result):
    raise RuntimeError("이미지 저장 실패")

print("=== 바닥 경계 후보 ===")
print("왼쪽:", left, "변화 강도:", round(ls, 1))
print("오른쪽:", right, "변화 강도:", round(rs, 1))
print("위쪽:", top, "변화 강도:", round(ts, 1))
print("아래쪽:", bottom, "변화 강도:", round(bs, 1))
print("결과:", out)
