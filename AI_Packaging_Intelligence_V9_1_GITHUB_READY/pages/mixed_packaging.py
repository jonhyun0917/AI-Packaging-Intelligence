from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimization.packing_optimizer import (
    BoxSize,
    Placement,
    ProductItem,
    analyze_mixed_packaging,
)


def _initialize_product_state() -> None:
    if "mixed_products" not in st.session_state:
        st.session_state.mixed_products = [
            {
                "name": "제품 A",
                "width": 90.0,
                "length": 90.0,
                "height": 90.0,
                "quantity": 1,
                "rotatable": True,
            },
            {
                "name": "제품 B",
                "width": 200.0,
                "length": 300.0,
                "height": 30.0,
                "quantity": 1,
                "rotatable": True,
            },
        ]


def _add_product() -> None:
    index = len(st.session_state.mixed_products) + 1

    st.session_state.mixed_products.append(
        {
            "name": f"제품 {index}",
            "width": 100.0,
            "length": 100.0,
            "height": 100.0,
            "quantity": 1,
            "rotatable": True,
        }
    )


def _remove_product(index: int) -> None:
    if len(st.session_state.mixed_products) <= 1:
        return

    st.session_state.mixed_products.pop(index)


def _render_top_view(
    box: BoxSize,
    placements: list[Placement],
    layer: int,
) -> go.Figure:
    figure = go.Figure()

    figure.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=box.width,
        y1=box.length,
        line={
            "width": 3,
            "dash": "dash",
        },
    )

    layer_placements = [
        placement
        for placement in placements
        if placement.layer == layer
    ]

    for index, placement in enumerate(layer_placements):
        figure.add_shape(
            type="rect",
            x0=placement.x,
            y0=placement.y,
            x1=placement.x + placement.width,
            y1=placement.y + placement.length,
            line={"width": 2},
            fillcolor=f"rgba({80 + index * 25 % 150}, 130, 200, 0.25)",
        )

        rotation_label = " ↻" if placement.rotated else ""

        figure.add_annotation(
            x=placement.x + placement.width / 2,
            y=placement.y + placement.length / 2,
            text=f"{placement.name}{rotation_label}",
            showarrow=False,
        )

    figure.update_xaxes(
        title="가로(mm)",
        range=[0, box.width],
        constrain="domain",
    )
    figure.update_yaxes(
        title="세로(mm)",
        range=[box.length, 0],
        scaleanchor="x",
        scaleratio=1,
    )
    figure.update_layout(
        title=f"혼합 적재 상단 평면도 — Layer {layer + 1}",
        height=520,
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        showlegend=False,
    )

    return figure


def _render_product_inputs() -> list[ProductItem]:
    products: list[ProductItem] = []

    st.subheader("2. 혼합 포장 대상 제품")

    add_col, count_col = st.columns([1, 3])

    with add_col:
        st.button(
            "➕ 제품 추가",
            on_click=_add_product,
            use_container_width=True,
        )

    with count_col:
        st.info(
            f"현재 제품 종류: {len(st.session_state.mixed_products)}개"
        )

    for index, product in enumerate(st.session_state.mixed_products):
        with st.expander(
            f"제품 {index + 1}: {product['name']}",
            expanded=True,
        ):
            header_col, remove_col = st.columns([5, 1])

            with header_col:
                name = st.text_input(
                    "제품명",
                    value=product["name"],
                    key=f"mixed_name_{index}",
                )

            with remove_col:
                st.write("")
                st.write("")
                remove_clicked = st.button(
                    "삭제",
                    key=f"remove_product_{index}",
                    disabled=len(st.session_state.mixed_products) <= 1,
                    use_container_width=True,
                )

            if remove_clicked:
                _remove_product(index)
                st.rerun()

            dimension_col1, dimension_col2, dimension_col3 = st.columns(3)

            with dimension_col1:
                width = st.number_input(
                    "가로(mm)",
                    min_value=1.0,
                    value=float(product["width"]),
                    step=1.0,
                    key=f"mixed_width_{index}",
                )

            with dimension_col2:
                length = st.number_input(
                    "세로(mm)",
                    min_value=1.0,
                    value=float(product["length"]),
                    step=1.0,
                    key=f"mixed_length_{index}",
                )

            with dimension_col3:
                height = st.number_input(
                    "높이(mm)",
                    min_value=1.0,
                    value=float(product["height"]),
                    step=1.0,
                    key=f"mixed_height_{index}",
                )

            option_col1, option_col2 = st.columns(2)

            with option_col1:
                quantity = st.number_input(
                    "수량",
                    min_value=1,
                    max_value=100,
                    value=int(product["quantity"]),
                    step=1,
                    key=f"mixed_quantity_{index}",
                )

            with option_col2:
                rotatable = st.checkbox(
                    "90도 회전 허용",
                    value=bool(product["rotatable"]),
                    key=f"mixed_rotatable_{index}",
                )

            st.session_state.mixed_products[index] = {
                "name": name,
                "width": width,
                "length": length,
                "height": height,
                "quantity": int(quantity),
                "rotatable": rotatable,
            }

            products.append(
                ProductItem(
                    name=name,
                    width=float(width),
                    length=float(length),
                    height=float(height),
                    quantity=int(quantity),
                    rotatable=rotatable,
                )
            )

    return products


def render_mixed_packaging_page() -> None:
    _initialize_product_state()

    st.title("📦 다품종 혼합 포장 최적화")
    st.markdown(
        """
서로 크기가 다른 여러 제품의 **실제 규격과 수량**을 입력하여
현재 박스의 공간 효율, 적재 가능성, 추천 박스 규격을 계산합니다.

> 배치 결과는 실무 검토를 돕는 근사 시뮬레이션입니다.
> 복잡한 형상, 눌림, 비닐 변형, 완충재 구조는 별도 검토가 필요합니다.
"""
    )

    st.divider()

    left, right = st.columns([5, 7])

    with left:
        st.subheader("1. 외함 박스 내부 규격")

        box_col1, box_col2, box_col3 = st.columns(3)

        with box_col1:
            box_width = st.number_input(
                "가로 W(mm)",
                min_value=1.0,
                value=330.0,
                step=1.0,
                key="mixed_box_width",
            )

        with box_col2:
            box_length = st.number_input(
                "세로 L(mm)",
                min_value=1.0,
                value=290.0,
                step=1.0,
                key="mixed_box_length",
            )

        with box_col3:
            box_height = st.number_input(
                "높이 H(mm)",
                min_value=1.0,
                value=160.0,
                step=1.0,
                key="mixed_box_height",
            )

        tolerance = st.number_input(
            "제품별 안전 유격(mm)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="mixed_tolerance",
            help="제품 주변에 확보할 최소 여유 공간입니다.",
        )

        products = _render_product_inputs()

        analyze = st.button(
            "🔍 혼합 포장 분석",
            type="primary",
            use_container_width=True,
        )

    with right:
        if not analyze and "mixed_analysis_result" not in st.session_state:
            st.info("왼쪽에서 규격을 입력한 뒤 혼합 포장 분석을 실행하세요.")

        if analyze:
            try:
                box = BoxSize(
                    width=float(box_width),
                    length=float(box_length),
                    height=float(box_height),
                )

                st.session_state.mixed_analysis_result = analyze_mixed_packaging(
                    box=box,
                    products=products,
                    tolerance=float(tolerance),
                )
                st.session_state.mixed_analysis_box = box
                st.session_state.mixed_analysis_products = products

            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"혼합 포장 분석 중 오류가 발생했습니다: {error}")
                st.exception(error)

        if "mixed_analysis_result" in st.session_state:
            result = st.session_state.mixed_analysis_result
            box = st.session_state.mixed_analysis_box
            analyzed_products = st.session_state.mixed_analysis_products

            volume = result["volume"]
            packing = result["packing"]
            recommended = result["recommended_box"]
            legal = result["legal_reference"]

            st.subheader("📊 분석 요약")

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric(
                    "박스 용적",
                    f"{volume['box_volume_cc']:,.1f} cc",
                )

            with metric2:
                st.metric(
                    "제품 총체적",
                    f"{volume['product_volume_cc']:,.1f} cc",
                )

            with metric3:
                st.metric(
                    "빈 공간 비율",
                    f"{volume['void_percent']:.1f}%",
                )

            with metric4:
                st.metric(
                    "공간 점유율",
                    f"{volume['utilization_percent']:.1f}%",
                )

            if volume["volume_overflow"]:
                st.error("제품 총체적이 현재 박스 용적을 초과합니다.")
            elif packing["fits"]:
                st.success(
                    f"물리적 근사 배치 가능: "
                    f"{packing['layer_count']}개 층, "
                    f"사용 높이 {packing['used_height']:.1f}mm"
                )
            else:
                unplaced_text = ", ".join(packing["unplaced"])
                st.warning(
                    "현재 근사 배치 방식으로 들어가지 않는 제품이 있습니다: "
                    f"{unplaced_text}"
                )

            if legal["passed"]:
                st.success(
                    f"참고 판정 — {legal['title']}\n\n"
                    f"{legal['description']}"
                )
            else:
                st.warning(
                    f"참고 판정 — {legal['title']}\n\n"
                    f"{legal['description']}"
                )

            st.caption(
                "※ 위 규제 판정은 업로드된 계산 로직을 옮긴 참고 기능이며, "
                "공식 법적 적합성 인증이 아닙니다."
            )

            st.divider()
            st.subheader("🧩 적재 평면도")

            if packing["placements"]:
                layer_count = max(int(packing["layer_count"]), 1)

                selected_layer = st.selectbox(
                    "확인할 적재 층",
                    options=list(range(layer_count)),
                    format_func=lambda value: f"Layer {value + 1}",
                    key="mixed_selected_layer",
                )

                figure = _render_top_view(
                    box=box,
                    placements=packing["placements"],
                    layer=int(selected_layer),
                )
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.warning("표시할 수 있는 배치 결과가 없습니다.")

            st.divider()
            st.subheader("✨ 추천 박스 규격")

            rec1, rec2, rec3, rec4 = st.columns(4)

            with rec1:
                st.metric(
                    "추천 가로",
                    f"{recommended['width']:.0f} mm",
                )

            with rec2:
                st.metric(
                    "추천 세로",
                    f"{recommended['length']:.0f} mm",
                )

            with rec3:
                st.metric(
                    "추천 높이",
                    f"{recommended['height']:.0f} mm",
                )

            with rec4:
                st.metric(
                    "예상 박스 절감",
                    f"{result['box_saving_percent']:.1f}%",
                )

            st.info(
                f"추천 박스 적용 시 예상 빈 공간 비율: "
                f"{result['recommended_void_percent']:.1f}%"
            )

            st.divider()
            st.subheader("📋 제품별 체적")

            rows = []

            for item in analyzed_products:
                rows.append(
                    {
                        "제품명": item.name,
                        "가로(mm)": item.width,
                        "세로(mm)": item.length,
                        "높이(mm)": item.height,
                        "수량": item.quantity,
                        "개별 체적(cc)": round(item.volume_mm3 / 1000, 2),
                        "합계 체적(cc)": round(
                            item.total_volume_mm3 / 1000,
                            2,
                        ),
                        "회전 허용": "예" if item.rotatable else "아니오",
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
            