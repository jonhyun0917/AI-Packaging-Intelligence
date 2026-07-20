from __future__ import annotations

import plotly.graph_objects as go


def _cuboid_edges(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float):
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    x, y, z = [], [], []
    for a, b in pairs:
        x += [vertices[a][0], vertices[b][0], None]
        y += [vertices[a][1], vertices[b][1], None]
        z += [vertices[a][2], vertices[b][2], None]
    return x, y, z


def build_digital_twin(
    box_width: float,
    box_length: float,
    box_height: float,
    product_width: float,
    product_length: float,
    product_height: float,
) -> go.Figure:
    """상단 면적 추정치를 직육면체로 단순화한 3D 디지털 트윈입니다."""
    fig = go.Figure()
    bx, by, bz = _cuboid_edges(0, 0, 0, box_width, box_length, box_height)
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", name="박스", line={"width": 5}))

    px0 = max((box_width - product_width) / 2, 0)
    py0 = max((box_length - product_length) / 2, 0)
    px1 = min(px0 + product_width, box_width)
    py1 = min(py0 + product_length, box_length)
    pz1 = min(product_height, box_height)
    px, py, pz = _cuboid_edges(px0, py0, 0, px1, py1, pz1)
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", name="제품 추정 영역", line={"width": 8}))

    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "가로(mm)",
            "yaxis_title": "세로(mm)",
            "zaxis_title": "높이(mm)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.6, "z": 1.2}},
        },
        title="3D Packaging Digital Twin (근사 시뮬레이션)",
    )
    return fig
