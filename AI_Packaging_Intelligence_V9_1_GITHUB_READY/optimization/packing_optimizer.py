from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import ceil
from typing import Any, Iterable


@dataclass(frozen=True)
class ProductItem:
    """혼합 포장에 사용하는 단일 제품 규격."""

    name: str
    width: float
    length: float
    height: float
    quantity: int = 1
    rotatable: bool = True

    @property
    def volume_mm3(self) -> float:
        return self.width * self.length * self.height

    @property
    def total_volume_mm3(self) -> float:
        return self.volume_mm3 * self.quantity


@dataclass(frozen=True)
class BoxSize:
    """박스 내부 규격."""

    width: float
    length: float
    height: float

    @property
    def volume_mm3(self) -> float:
        return self.width * self.length * self.height


@dataclass
class Placement:
    """2D 상단 평면도에 표시할 제품 배치 결과."""

    name: str
    x: float
    y: float
    width: float
    length: float
    height: float
    layer: int
    rotated: bool = False


def _validate_positive(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name}은(는) 0보다 커야 합니다.")


def validate_box(box: BoxSize) -> None:
    _validate_positive(box.width, "박스 가로")
    _validate_positive(box.length, "박스 세로")
    _validate_positive(box.height, "박스 높이")


def validate_products(products: Iterable[ProductItem]) -> None:
    product_list = list(products)

    if not product_list:
        raise ValueError("제품을 한 개 이상 입력해야 합니다.")

    for item in product_list:
        _validate_positive(item.width, f"{item.name} 가로")
        _validate_positive(item.length, f"{item.name} 세로")
        _validate_positive(item.height, f"{item.name} 높이")

        if item.quantity < 1:
            raise ValueError(f"{item.name} 수량은 1개 이상이어야 합니다.")


def expand_products(products: Iterable[ProductItem]) -> list[ProductItem]:
    """수량을 개별 제품 목록으로 확장합니다."""

    expanded: list[ProductItem] = []

    for item in products:
        for index in range(item.quantity):
            expanded.append(
                ProductItem(
                    name=(
                        item.name
                        if item.quantity == 1
                        else f"{item.name} #{index + 1}"
                    ),
                    width=item.width,
                    length=item.length,
                    height=item.height,
                    quantity=1,
                    rotatable=item.rotatable,
                )
            )

    return expanded


def calculate_volume_metrics(
    box: BoxSize,
    products: Iterable[ProductItem],
) -> dict[str, float | bool]:
    """박스와 전체 제품의 체적, 점유율, 빈 공간 비율을 계산합니다."""

    validate_box(box)
    product_list = list(products)
    validate_products(product_list)

    box_volume = box.volume_mm3
    total_product_volume = sum(
        item.total_volume_mm3
        for item in product_list
    )

    void_volume = max(box_volume - total_product_volume, 0.0)
    utilization_percent = (
        total_product_volume / box_volume * 100
        if box_volume > 0
        else 0.0
    )
    void_percent = max(100.0 - utilization_percent, 0.0)

    return {
        "box_volume_mm3": round(box_volume, 2),
        "box_volume_cc": round(box_volume / 1000, 2),
        "product_volume_mm3": round(total_product_volume, 2),
        "product_volume_cc": round(total_product_volume / 1000, 2),
        "void_volume_mm3": round(void_volume, 2),
        "void_volume_cc": round(void_volume / 1000, 2),
        "utilization_percent": round(utilization_percent, 2),
        "void_percent": round(void_percent, 2),
        "volume_overflow": total_product_volume > box_volume,
    }


def legal_reference_status(
    box: BoxSize,
    void_percent: float,
) -> dict[str, str | float | bool]:
    """
    업로드된 HTML 로직과 동일한 참고 판정을 제공합니다.

    이 결과는 법률 자문이나 공식 적합성 인증이 아니라,
    앱 내부의 참고용 시뮬레이션 결과입니다.
    """

    three_side_sum_cm = (
        box.width + box.length + box.height
    ) / 10

    if three_side_sum_cm <= 50:
        return {
            "passed": True,
            "status": "PASS",
            "title": "참고 통과: 소형 박스 조건",
            "description": (
                f"박스 삼면의 합이 {three_side_sum_cm:.1f}cm입니다. "
                "현재 앱 기준에서는 소형 박스 참고 통과로 표시합니다."
            ),
            "three_side_sum_cm": round(three_side_sum_cm, 2),
        }

    if void_percent <= 50:
        return {
            "passed": True,
            "status": "PASS",
            "title": "참고 통과: 빈 공간 기준 이내",
            "description": (
                f"빈 공간 비율이 {void_percent:.1f}%로 "
                "앱의 참고 기준인 50% 이내입니다."
            ),
            "three_side_sum_cm": round(three_side_sum_cm, 2),
        }

    return {
        "passed": False,
        "status": "CHECK",
        "title": "과대포장 검토 필요",
        "description": (
            f"빈 공간 비율이 {void_percent:.1f}%입니다. "
            "더 작은 박스 또는 다른 배치를 검토하세요."
        ),
        "three_side_sum_cm": round(three_side_sum_cm, 2),
    }


def _orientation_options(
    item: ProductItem,
    tolerance: float,
) -> list[tuple[float, float, bool]]:
    """제품의 0도/90도 회전 바닥 규격을 반환합니다."""

    normal = (
        item.width + tolerance * 2,
        item.length + tolerance * 2,
        False,
    )

    if not item.rotatable or item.width == item.length:
        return [normal]

    rotated = (
        item.length + tolerance * 2,
        item.width + tolerance * 2,
        True,
    )

    return [normal, rotated]


def pack_products_2d(
    box: BoxSize,
    products: Iterable[ProductItem],
    tolerance: float = 5.0,
) -> dict[str, Any]:
    """
    Shelf 방식으로 제품을 박스 바닥면에 배치합니다.

    - 큰 제품부터 배치
    - 각 제품의 90도 회전 허용
    - 한 층의 최대 높이를 초과하면 새 층 생성
    - 완전한 3D Bin Packing은 아니지만 여러 제품의 실용적인
      2D/복층 근사 배치를 제공합니다.
    """

    validate_box(box)
    product_list = list(products)
    validate_products(product_list)

    if tolerance < 0:
        raise ValueError("안전 유격은 0 이상이어야 합니다.")

    expanded = expand_products(product_list)

    # 큰 바닥 면적, 큰 높이 순서로 우선 배치합니다.
    expanded.sort(
        key=lambda item: (
            item.width * item.length,
            item.height,
        ),
        reverse=True,
    )

    placements: list[Placement] = []
    unplaced: list[str] = []

    layer = 0
    layer_z = 0.0
    current_layer_height = 0.0
    cursor_x = 0.0
    cursor_y = 0.0
    row_depth = 0.0

    def start_new_layer() -> None:
        nonlocal layer, layer_z, current_layer_height
        nonlocal cursor_x, cursor_y, row_depth

        layer_z += current_layer_height
        layer += 1
        current_layer_height = 0.0
        cursor_x = 0.0
        cursor_y = 0.0
        row_depth = 0.0

    for item in expanded:
        placed = False

        # 너무 높은 제품은 어떤 층에도 배치할 수 없습니다.
        if item.height > box.height:
            unplaced.append(item.name)
            continue

        for _ in range(2):
            options = _orientation_options(item, tolerance)

            # 현재 행에서 들어갈 수 있는 방향을 우선 선택합니다.
            options.sort(
                key=lambda option: (
                    option[1],
                    option[0],
                )
            )

            for effective_w, effective_l, rotated in options:
                if (
                    cursor_x + effective_w <= box.width
                    and cursor_y + effective_l <= box.length
                    and layer_z + item.height <= box.height
                ):
                    placements.append(
                        Placement(
                            name=item.name,
                            x=cursor_x + tolerance,
                            y=cursor_y + tolerance,
                            width=effective_w - tolerance * 2,
                            length=effective_l - tolerance * 2,
                            height=item.height,
                            layer=layer,
                            rotated=rotated,
                        )
                    )

                    cursor_x += effective_w
                    row_depth = max(row_depth, effective_l)
                    current_layer_height = max(
                        current_layer_height,
                        item.height,
                    )
                    placed = True
                    break

            if placed:
                break

            # 새 행을 시작합니다.
            if cursor_x > 0:
                cursor_x = 0.0
                cursor_y += row_depth
                row_depth = 0.0
                continue

            # 현재 층에 더 이상 들어가지 않으면 새 층을 만듭니다.
            break

        if not placed:
            start_new_layer()

            options = _orientation_options(item, tolerance)
            options.sort(key=lambda option: option[0] * option[1])

            for effective_w, effective_l, rotated in options:
                if (
                    effective_w <= box.width
                    and effective_l <= box.length
                    and layer_z + item.height <= box.height
                ):
                    placements.append(
                        Placement(
                            name=item.name,
                            x=tolerance,
                            y=tolerance,
                            width=effective_w - tolerance * 2,
                            length=effective_l - tolerance * 2,
                            height=item.height,
                            layer=layer,
                            rotated=rotated,
                        )
                    )

                    cursor_x = effective_w
                    cursor_y = 0.0
                    row_depth = effective_l
                    current_layer_height = item.height
                    placed = True
                    break

        if not placed:
            unplaced.append(item.name)

    used_layers = (
        max((placement.layer for placement in placements), default=-1) + 1
    )

    used_height = 0.0

    for layer_index in range(used_layers):
        layer_height = max(
            (
                placement.height
                for placement in placements
                if placement.layer == layer_index
            ),
            default=0.0,
        )
        used_height += layer_height

    return {
        "fits": not unplaced,
        "placements": placements,
        "unplaced": unplaced,
        "layer_count": used_layers,
        "used_height": round(used_height, 2),
        "height_remaining": round(max(box.height - used_height, 0.0), 2),
    }


def _candidate_footprints(
    item: ProductItem,
    tolerance: float,
) -> list[tuple[float, float]]:
    return [
        (width, length)
        for width, length, _ in _orientation_options(item, tolerance)
    ]


def recommend_box_size(
    products: Iterable[ProductItem],
    tolerance: float = 5.0,
    standard_step: float = 10.0,
) -> dict[str, float | int]:
    """
    제품들을 한 층 또는 복층으로 담기 위한 추천 박스 규격을 계산합니다.

    여러 후보 폭을 만들고 Shelf 배치를 반복하여
    부피가 가장 작은 후보를 선택합니다.
    """

    product_list = list(products)
    validate_products(product_list)

    if tolerance < 0:
        raise ValueError("안전 유격은 0 이상이어야 합니다.")

    if standard_step <= 0:
        raise ValueError("규격 반올림 단위는 0보다 커야 합니다.")

    expanded = expand_products(product_list)

    min_width = max(
        min(width for width, _ in _candidate_footprints(item, tolerance))
        for item in expanded
    )

    total_width = sum(
        min(width for width, _ in _candidate_footprints(item, tolerance))
        for item in expanded
    )

    candidate_widths = {
        min_width,
        total_width,
    }

    for divisor in range(2, min(len(expanded) + 1, 8)):
        candidate_widths.add(max(min_width, total_width / divisor))

    best: dict[str, float | int] | None = None

    for candidate_width in sorted(candidate_widths):
        # 매우 긴 박스를 방지하기 위한 최대 길이 후보입니다.
        candidate_length = sum(
            max(length for _, length in _candidate_footprints(item, tolerance))
            for item in expanded
        )

        provisional_box = BoxSize(
            width=candidate_width,
            length=candidate_length,
            height=sum(item.height for item in expanded),
        )

        result = pack_products_2d(
            provisional_box,
            expanded,
            tolerance=tolerance,
        )

        if not result["fits"]:
            continue

        placements: list[Placement] = result["placements"]

        used_width = max(
            (
                placement.x + placement.width + tolerance
                for placement in placements
            ),
            default=min_width,
        )
        used_length = max(
            (
                placement.y + placement.length + tolerance
                for placement in placements
            ),
            default=min_width,
        )
        used_height = float(result["used_height"])

        rounded_width = ceil(used_width / standard_step) * standard_step
        rounded_length = ceil(used_length / standard_step) * standard_step
        rounded_height = ceil(
            (used_height + tolerance * 2) / standard_step
        ) * standard_step

        volume = rounded_width * rounded_length * rounded_height

        candidate = {
            "width": round(rounded_width, 2),
            "length": round(rounded_length, 2),
            "height": round(rounded_height, 2),
            "volume_mm3": round(volume, 2),
            "volume_cc": round(volume / 1000, 2),
            "layer_count": int(result["layer_count"]),
        }

        if best is None or candidate["volume_mm3"] < best["volume_mm3"]:
            best = candidate

    if best is None:
        # 최종 안전 대안: 모든 제품을 한 줄로 나열합니다.
        width = sum(
            min(width for width, _ in _candidate_footprints(item, tolerance))
            for item in expanded
        )
        length = max(
            min(length for _, length in _candidate_footprints(item, tolerance))
            for item in expanded
        )
        height = max(item.height for item in expanded) + tolerance * 2

        best = {
            "width": ceil(width / standard_step) * standard_step,
            "length": ceil(length / standard_step) * standard_step,
            "height": ceil(height / standard_step) * standard_step,
            "volume_mm3": 0.0,
            "volume_cc": 0.0,
            "layer_count": 1,
        }

        best["volume_mm3"] = (
            best["width"]
            * best["length"]
            * best["height"]
        )
        best["volume_cc"] = best["volume_mm3"] / 1000

    return best


def analyze_mixed_packaging(
    box: BoxSize,
    products: Iterable[ProductItem],
    tolerance: float = 5.0,
) -> dict[str, Any]:
    """혼합 포장 페이지에서 사용하는 통합 분석 함수입니다."""

    product_list = list(products)

    volume = calculate_volume_metrics(box, product_list)
    packing = pack_products_2d(
        box,
        product_list,
        tolerance=tolerance,
    )
    recommended_box = recommend_box_size(
        product_list,
        tolerance=tolerance,
    )
    legal = legal_reference_status(
        box,
        float(volume["void_percent"]),
    )

    current_volume = box.volume_mm3
    recommended_volume = float(recommended_box["volume_mm3"])

    saving_percent = (
        max(
            (current_volume - recommended_volume)
            / current_volume
            * 100,
            0.0,
        )
        if current_volume > 0
        else 0.0
    )

    recommended_void = calculate_volume_metrics(
        BoxSize(
            width=float(recommended_box["width"]),
            length=float(recommended_box["length"]),
            height=float(recommended_box["height"]),
        ),
        product_list,
    )

    return {
        "volume": volume,
        "packing": packing,
        "recommended_box": recommended_box,
        "recommended_void_percent": recommended_void["void_percent"],
        "box_saving_percent": round(saving_percent, 2),
        "legal_reference": legal,
    }
