from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _background_mask(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    border = max(3, int(min(h, w) * 0.04))

    samples = np.concatenate(
        [
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)

    threshold = max(
        12.0,
        float(np.percentile(dist, 72)),
    )

    return np.where(
        dist >= threshold,
        255,
        0,
    ).astype(np.uint8)


def _apply_color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:

    if cv2.countNonZero(mask) == 0:
        return image

    overlay = image.copy()
    overlay[mask > 0] = color

    return cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
    )


def detect_products(
    image: Any,
    box: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int, int]:

    # --------------------------------------------------
    # 1. 이미지 준비
    # --------------------------------------------------

    rgb = np.array(
        image.convert("RGB")
        if hasattr(image, "convert")
        else image
    )

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB 이미지가 필요합니다.")

    original = cv2.cvtColor(
        rgb[:, :, :3],
        cv2.COLOR_RGB2BGR,
    )


    # --------------------------------------------------
    # 2. 박스 내부 영역 확보
    # --------------------------------------------------

    use_warped = bool(
        box
        and isinstance(
            box.get("warped_image"),
            np.ndarray,
        )
    )

    if use_warped:

        crop = box["warped_image"].copy()

        offset_x = 0
        offset_y = 0

    else:

        h, w = original.shape[:2]

        x1 = (
            int(box.get("x1", 0))
            if box
            else int(w * 0.05)
        )

        y1 = (
            int(box.get("y1", 0))
            if box
            else int(h * 0.05)
        )

        x2 = (
            int(box.get("x2", w))
            if box
            else int(w * 0.95)
        )

        y2 = (
            int(box.get("y2", h))
            if box
            else int(h * 0.95)
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = original[
            y1:y2,
            x1:x2
        ].copy()

        offset_x = x1
        offset_y = y1


    if crop.size == 0:
        raise ValueError(
            "분석할 박스 내부 영역이 없습니다."
        )


    ch, cw = crop.shape[:2]
    crop_area = ch * cw


    # --------------------------------------------------
    # 3. 계산용 기본 마스크
    #
    # 박스 접힘선 오검출을 줄이기 위해
    # edge mask는 사용하지 않음
    # --------------------------------------------------

    mask = _background_mask(crop)


    k = max(
        3,
        int(min(ch, cw) * 0.008),
    )

    if k % 2 == 0:
        k += 1

    kernel = np.ones(
        (k, k),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )


    # --------------------------------------------------
    # 4. 박스 가장자리 제외
    # --------------------------------------------------

    margin = max(
        2,
        int(min(ch, cw) * 0.015),
    )

    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0


    # --------------------------------------------------
    # 5. 연결 영역 분석
    # --------------------------------------------------

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    min_area = max(
        120,
        int(crop_area * 0.004),
    )

    max_area = int(
        crop_area * 0.82
    )


    components = []


    for label in range(1, n):

        x, y, w, h, area = [
            int(v)
            for v in stats[label]
        ]

        if area < min_area:
            continue

        if area > max_area:
            continue


        rect_area = max(
            1,
            w * h,
        )

        fill_ratio = (
            area / rect_area
        )

        aspect = (
            w / max(1, h)
        )


        if fill_ratio < 0.18:
            continue

        if aspect < 0.08:
            continue

        if aspect > 12:
            continue


        touches = sum(
            [
                x <= margin,
                y <= margin,
                x + w >= cw - margin,
                y + h >= ch - margin,
            ]
        )


        # -------------------------------
        # 계산용 필터
        #
        # 지나치게 큰 벽면 영역만 제외
        # 작은 조각까지 여기서 삭제하지 않음
        # -------------------------------

        if (
            touches >= 3
            and area > crop_area * 0.20
        ):
            continue


        area_ratio = (
            area / crop_area
        )


        confidence = np.clip(
            min(
                1.0,
                area_ratio / 0.12
            ) * 0.35
            +
            min(
                1.0,
                fill_ratio
            ) * 0.40
            +
            (
                1.0
                - touches / 4.0
            ) * 0.25,
            0.05,
            0.99,
        )


        components.append(
            {
                "area": area,
                "label": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "touches": touches,
                "confidence":
                    float(confidence),
            }
        )


    components.sort(
        key=lambda item:
            item["area"],
        reverse=True,
    )


    # ==================================================
    # 6. 계산용 마스크
    #
    # 여기서 만들어진 면적만
    # occupancy / void 계산에 사용
    # ==================================================

    calculation_mask = np.zeros(
        (ch, cw),
        dtype=np.uint8,
    )


    for component in components:

        label = component["label"]

        cmask = np.where(
            labels == label,
            255,
            0,
        ).astype(np.uint8)

        calculation_mask = cv2.bitwise_or(
            calculation_mask,
            cmask,
        )


    # --------------------------------------------------
    # 중요:
    # 숫자 계산은 오직 calculation_mask 기준
    # --------------------------------------------------

    product_area = int(
        cv2.countNonZero(
            calculation_mask
        )
    )


    # ==================================================
    # 7. 시각화용 영역만 별도로 정리
    #
    # 여기서 무언가 삭제되어도
    # product_area에는 영향 없음
    # ==================================================

    visual_components = []

    if components:

        largest_area = (
            components[0]["area"]
        )


        for component in components:

            # 너무 작은 조각 제거
            if (
                component["area"]
                < largest_area * 0.25
            ):
                continue


            # 박스 테두리와 직접 연결된
            # 시각적 오검출 영역 제거
            if component["touches"] >= 1:
                continue


            visual_components.append(
                component
            )


    # 만약 필터가 너무 강해서
    # 아무것도 남지 않으면
    # 가장 큰 영역 하나는 보여준다.
    if (
        not visual_components
        and components
    ):
        visual_components = [
            components[0]
        ]


    visual_mask = np.zeros(
        (ch, cw),
        dtype=np.uint8,
    )


    detections: List[
        Dict[str, Any]
    ] = []

    visual_contours = []


    for idx, component in enumerate(
        visual_components,
        start=1,
    ):

        label = component["label"]

        cmask = np.where(
            labels == label,
            255,
            0,
        ).astype(np.uint8)


        visual_mask = cv2.bitwise_or(
            visual_mask,
            cmask,
        )


        contours, _ = cv2.findContours(
            cmask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )


        visual_contours.extend(
            contours
        )


        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]


        detections.append(
            {
                "name":
                    f"Detected Product {idx}",

                "confidence":
                    round(
                        component[
                            "confidence"
                        ],
                        2,
                    ),

                "area":
                    component["area"],

                "bbox":
                    [
                        x + offset_x,
                        y + offset_y,
                        x + w + offset_x,
                        y + h + offset_y,
                    ],

                "width_px":
                    w,

                "height_px":
                    h,
            }
        )


    # ==================================================
    # 8. 예상 빈공간 시각화
    # ==================================================

    inner_mask = np.ones(
        (ch, cw),
        dtype=np.uint8,
    ) * 255


    inner_mask[:margin, :] = 0
    inner_mask[-margin:, :] = 0
    inner_mask[:, :margin] = 0
    inner_mask[:, -margin:] = 0


    # 화면에 보여주는 빈공간
    # = 박스 내부 - 시각화용 제품 영역
    visual_void_mask = cv2.bitwise_and(
        inner_mask,
        cv2.bitwise_not(
            visual_mask
        ),
    )


    result_crop = crop.copy()


    # --------------------------------------------------
    # 9. 빈공간만 주황색 반투명 표시
    # --------------------------------------------------

    result_crop = _apply_color_overlay(
        result_crop,
        visual_void_mask,
        (255, 120, 60),
        alpha=0.30,
    )


    # --------------------------------------------------
    # 10. 제품은 면을 칠하지 않고
    # 초록색 윤곽선만 표시
    # --------------------------------------------------

    if visual_contours:

        cv2.drawContours(
            result_crop,
            visual_contours,
            -1,
            (36, 110, 75),
            3,
        )


    # --------------------------------------------------
    # 11. 원본 이미지와 합치기
    # --------------------------------------------------

    if use_warped:

        result = result_crop

    else:

        result = original.copy()

        result[
            y1:y2,
            x1:x2
        ] = result_crop


    # --------------------------------------------------
    # detection count 역시 화면에 표시된
    # 제품 후보 기준
    # --------------------------------------------------

    return (
        detections,
        result,
        product_area,
        len(detections),
    )
