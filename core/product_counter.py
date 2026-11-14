
import cv2
import numpy as np


def count_products(image):
    """
    파란색 제품 개수 추정.
    image: OpenCV BGR 이미지
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([90, 70, 40]),
        np.array([135, 255, 255])
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    row_counts = np.count_nonzero(mask, axis=1)
    rows = np.where(row_counts > 20)[0]

    # 빈 박스 처리
    if rows.size == 0:
        return 0

    top = int(rows.min())
    bottom = int(rows.max())

    start = top + (bottom - top) // 3
    end = top + (bottom - top) * 2 // 3

    split_y = start + int(
        np.argmin(row_counts[start:end + 1])
    )

    count_mask = mask.copy()

    nearby_counts = row_counts[
        max(0, split_y - 2):
        min(len(row_counts), split_y + 3)
    ]

    upper_pixels = int(
        np.count_nonzero(mask[top:split_y, :])
    )

    lower_pixels = int(
        np.count_nonzero(mask[split_y + 1:bottom + 1, :])
    )

    total_pixels = int(np.count_nonzero(mask))

    gap_is_clear = (
        nearby_counts.size > 0
        and np.max(nearby_counts)
        <= max(5, int(np.max(row_counts) * 0.03))
    )

    has_two_sides = (
        upper_pixels > total_pixels * 0.15
        and lower_pixels > total_pixels * 0.15
    )

    if (
        gap_is_clear
        and has_two_sides
        and bottom - top > 30
    ):
        count_mask[
            max(0, split_y - 3):split_y + 4, :
        ] = 0

    contours, _ = cv2.findContours(
        count_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = max(
        100,
        int(total_pixels * 0.01)
    )

    valid = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    return len(valid)
