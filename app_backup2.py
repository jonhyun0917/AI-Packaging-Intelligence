from __future__ import annotations

import cv2
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pathlib import Path

from analysis.carbon import calculate_carbon
from analysis.consultant import build_consultant_report
from analysis.esg_score import calculate_esg_score
from analysis.pvi import calculate_pvi
from analysis.volume_pvi import calculate_volume_pvi
from core.box_detector import detect_box
from core.detect import detect_products
from core.digital_twin import build_digital_twin
from core.measurement_engine import calculate_planar_measurements, calculate_volume_measurements, estimate_measurement_confidence
from core.quality_checker import check_image_quality
from core.limitations import build_limitations, build_self_review
from optimization.box_optimizer import rank_boxes, recommend_box
from pages.mixed_packaging import render_mixed_packaging_page
from reporting.report import build_pdf_report
from reporting.certificate import build_analysis_certificate

st.set_page_config(
    page_title="AI Packaging Intelligence | Better Life For Us",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "greenvalue_logo_cropped.png"

st.markdown("""
<style>
:root {
    --ink:#18221D;
    --muted:#66736B;
    --forest:#214C39;
    --forest-2:#2F6B4F;
    --sage:#DCE9E0;
    --mint:#F1F6F3;
    --paper:#F7F9F8;
    --white:#FFFFFF;
    --line:#E3E9E5;
    --amber:#B9782E;
    --red:#B85245;
}

html, body, [class*="css"] {
    font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Noto Sans KR","Segoe UI",sans-serif;
    color:var(--ink);
}

.stApp {background:var(--paper);}
.block-container {padding-top:2.1rem;padding-bottom:4.2rem;max-width:1220px;}
#MainMenu, footer, header {visibility:hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background:var(--white);
    border-right:1px solid var(--line);
}
[data-testid="stSidebar"] .block-container {padding-top:1.3rem;}
[data-testid="stSidebar"] h3 {font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:#8A958E;margin:1.1rem 0 .5rem;}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding:.65rem .75rem;
    border-radius:10px;
    margin:.08rem 0;
    transition:background .15s ease;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {background:#F3F6F4;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#8A958E;}
.sidebar-brand {padding:.25rem .1rem .95rem;}
.sidebar-brand .name {font-size:1.08rem;font-weight:800;letter-spacing:-.02em;color:var(--forest);}
.sidebar-brand .copy {font-size:.78rem;line-height:1.55;color:#7C8981;margin-top:.28rem;}
.version-pill {display:inline-block;background:#F1F5F2;border:1px solid var(--line);color:#6F7D74;padding:5px 9px;border-radius:999px;font-size:.7rem;font-weight:700;}

/* Common */
h1, h2, h3 {letter-spacing:-.035em;}
.section-head {margin:2.25rem 0 1rem;}
.section-head .eyebrow {font-size:.72rem;letter-spacing:.1em;font-weight:800;color:var(--forest-2);text-transform:uppercase;margin-bottom:.36rem;}
.section-head h2 {font-size:1.42rem;margin:0;color:var(--ink);}
.section-head p {margin:.4rem 0 0;color:var(--muted);font-size:.94rem;line-height:1.68;}
.card {background:var(--white);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 6px 20px rgba(25,52,38,.035);}
.card-label {font-size:.7rem;letter-spacing:.09em;font-weight:800;color:#7C8C82;text-transform:uppercase;margin-bottom:9px;}
.card-title {font-size:1.03rem;font-weight:800;color:var(--ink);margin-bottom:7px;letter-spacing:-.025em;}
.card-copy {font-size:.88rem;line-height:1.7;color:var(--muted);}

/* Hero */
.hero {
    display:grid;
    grid-template-columns:minmax(0,1.25fr) minmax(330px,.75fr);
    gap:46px;
    align-items:center;
    padding:52px 54px;
    border:1px solid var(--line);
    border-radius:26px;
    background:linear-gradient(140deg,#FFFFFF 0%,#FFFFFF 62%,#F1F6F3 100%);
    box-shadow:0 16px 48px rgba(28,57,42,.06);
    overflow:hidden;
}
.kicker {font-size:.74rem;font-weight:850;letter-spacing:.13em;color:var(--forest-2);text-transform:uppercase;}
.hero h1 {margin:9px 0 16px;font-size:3rem;line-height:1.15;font-weight:850;color:var(--ink);letter-spacing:-.06em;}
.hero p {margin:0;max-width:700px;color:#59665E;font-size:1rem;line-height:1.82;}
.hero-note {margin-top:20px;display:flex;gap:8px;flex-wrap:wrap;}
.hero-chip {display:inline-flex;align-items:center;padding:7px 10px;background:#F6F8F7;border:1px solid var(--line);border-radius:999px;color:#5F6D64;font-size:.76rem;font-weight:650;}
.hero-visual {display:flex;flex-direction:column;gap:10px;position:relative;}
.flow-card {display:grid;grid-template-columns:36px 1fr auto;align-items:center;gap:12px;background:#FFF;border:1px solid #DCE6DF;border-radius:14px;padding:14px 15px;box-shadow:0 8px 24px rgba(31,74,53,.06);}
.flow-card:nth-child(2) {margin-left:24px;}
.flow-card:nth-child(3) {margin-left:48px;}
.flow-no {width:30px;height:30px;border-radius:9px;background:var(--mint);display:flex;align-items:center;justify-content:center;color:var(--forest);font-size:.7rem;font-weight:850;}
.flow-card b {font-size:.87rem;color:var(--ink);}
.flow-card small {display:block;font-size:.7rem;color:#89948E;margin-top:2px;}
.flow-tag {font-size:.66rem;font-weight:800;color:var(--forest-2);background:#EEF5F0;border-radius:999px;padding:5px 8px;}

/* Home blocks */
.problem-panel {display:grid;grid-template-columns:.9fr 1.1fr;gap:18px;background:var(--forest);color:white;border-radius:20px;padding:28px 30px;}
.problem-quote {font-size:1.55rem;line-height:1.45;font-weight:800;letter-spacing:-.04em;}
.problem-quote small {display:block;font-size:.74rem;font-weight:600;letter-spacing:0;opacity:.7;margin-bottom:10px;}
.problem-list {display:grid;gap:10px;}
.problem-item {background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.11);border-radius:12px;padding:12px 14px;font-size:.85rem;line-height:1.55;color:#ECF2EE;}
.problem-item b {color:white;}
.step-card {background:#FFF;border:1px solid var(--line);border-radius:18px;padding:22px;min-height:156px;}
.step-num {display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:9px;background:var(--forest);color:white;font-weight:800;font-size:.72rem;margin-bottom:12px;}
.use-card {background:#FFF;border:1px solid var(--line);border-radius:16px;padding:19px 20px;min-height:132px;}
.use-card .role {font-weight:800;color:var(--ink);font-size:.98rem;margin-bottom:7px;}
.use-card .pain {color:var(--muted);font-size:.85rem;line-height:1.65;}
.trust-strip {display:flex;justify-content:space-between;gap:18px;align-items:center;background:#EEF4F0;border:1px solid #DCE8E0;border-radius:16px;padding:18px 20px;margin-top:24px;}
.trust-strip b {color:var(--forest);font-size:.94rem;}
.trust-strip span {color:#6B786F;font-size:.82rem;line-height:1.55;}

/* Streamlit widgets */
[data-testid="stMetric"] {background:#FFF;border:1px solid var(--line);padding:18px 17px;border-radius:16px;box-shadow:0 5px 18px rgba(28,57,42,.03);}
[data-testid="stMetricLabel"] {font-weight:650;color:#718077;}
[data-testid="stMetricValue"] {font-weight:820;color:var(--forest);font-size:1.42rem;letter-spacing:-.035em;}
[data-testid="stFileUploader"] {background:#FFF;border:1.5px dashed #AABBB0;border-radius:17px;padding:12px;}
[data-testid="stFileUploader"] section {padding:1.6rem 1rem;}
.stButton>button,.stDownloadButton>button {border-radius:11px;min-height:44px;font-weight:760;border:1px solid var(--forest);background:var(--forest);color:white;box-shadow:none;transition:.15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:#183E2D;background:#183E2D;color:white;transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"] {gap:4px;background:#EEF2EF;padding:4px;border-radius:12px;overflow-x:auto;}
.stTabs [data-baseweb="tab"] {height:42px;border-radius:9px;font-weight:700;padding:0 14px;color:#66736B;}
.stTabs [aria-selected="true"] {background:#FFF !important;color:var(--forest) !important;box-shadow:0 2px 8px rgba(28,57,42,.05);}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:14px;overflow:hidden;}

/* Analysis */
.analysis-intro {background:#FFF;border:1px solid var(--line);border-radius:20px;padding:26px 28px;margin-bottom:20px;display:flex;justify-content:space-between;gap:24px;align-items:flex-end;}
.analysis-intro h1 {margin:5px 0 7px;font-size:2rem;color:var(--ink);}
.analysis-intro p {margin:0;color:var(--muted);line-height:1.7;font-size:.92rem;max-width:720px;}
.analysis-note {white-space:nowrap;font-size:.75rem;color:#728078;background:#F4F7F5;border:1px solid var(--line);border-radius:999px;padding:7px 10px;}
.info-panel {background:#FFF;border:1px solid var(--line);border-radius:17px;padding:22px 24px;margin:12px 0;}
.info-panel h3 {color:var(--ink);margin:0 0 10px;font-size:1.08rem;}
.info-panel p {line-height:1.78;margin:.35rem 0;color:#56645C;font-size:.92rem;}
.easy-card {background:#FFF;border:1px solid var(--line);border-radius:16px;padding:19px 20px;min-height:155px;}
.easy-card b {display:block;color:var(--ink);font-size:.98rem;margin-bottom:8px;}
.easy-card small {font-size:.86rem;line-height:1.68;color:var(--muted);}
.formula-box {background:var(--mint);border-left:3px solid var(--forest-2);border-radius:12px;padding:17px 19px;margin:14px 0;font-size:.94rem;line-height:1.85;}
.callout {background:#F6F3EA;border:1px solid #E7E0CD;border-radius:14px;padding:17px 19px;line-height:1.72;margin:14px 0;color:#5D594D;font-size:.9rem;}
.trace-card {background:#FFF;border:1px solid var(--line);border-radius:13px;padding:14px 16px;margin:7px 0;}
.trace-card b {color:var(--forest);}
.presentation-card {background:#FFF;border:1px solid var(--line);border-radius:16px;padding:20px;min-height:146px;}
.presentation-card .big {font-size:1.9rem;font-weight:850;color:var(--forest);letter-spacing:-.05em;}
.footer-brand {margin-top:3rem;padding:18px 20px;border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:18px;color:#7A877F;}
.footer-brand strong {color:var(--forest);font-size:.9rem;}
.footer-brand span {font-size:.76rem;text-align:right;line-height:1.55;}

@media (max-width: 900px) {
    .block-container {padding-top:1.3rem;}
    .hero {grid-template-columns:1fr;padding:32px 27px;gap:28px;}
    .hero h1 {font-size:2.25rem;}
    .flow-card:nth-child(2),.flow-card:nth-child(3) {margin-left:0;}
    .problem-panel {grid-template-columns:1fr;}
    .analysis-intro {display:block;}
    .analysis-note {display:inline-block;margin-top:12px;}
    .footer-brand {display:block;}
    .footer-brand span {display:block;text-align:left;margin-top:6px;}
}
</style>
""", unsafe_allow_html=True)

PAGES = ["시작", "포장 분석", "신뢰도·한계", "혼합 포장", "지표 안내", "프로젝트 소개"]
if "page" not in st.session_state:
    st.session_state["page"] = "시작"

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown("<div class='sidebar-brand'><div class='name'>AI Packaging Intelligence</div><div class='copy'>포장 상태를 분석하고 박스 후보를 비교하는 의사결정 지원 도구</div></div>", unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("메뉴", PAGES, key="page", label_visibility="collapsed")
    presentation_mode = st.toggle("발표용 간단 보기", value=False, help="핵심 지표만 간단하게 표시합니다.")
    st.divider()
    st.markdown("<span class='version-pill'>Better Life For Us · V9.3</span>", unsafe_allow_html=True)


def section_head(eyebrow: str, title: str, copy: str = "") -> None:
    desc = f"<p>{copy}</p>" if copy else ""
    st.markdown(f"<div class='section-head'><div class='eyebrow'>{eyebrow}</div><h2>{title}</h2>{desc}</div>", unsafe_allow_html=True)


def gauge(value: float, title: str, suffix: str = "") -> go.Figure:
    value = float(max(0, min(100, value)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": "#214C39"}},
        title={"text": title, "font": {"size": 14, "color": "#66736B"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#DCE5DF"},
            "bar": {"color": "#2F6B4F", "thickness": 0.22},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#F6E9E7"},
                {"range": [50, 70], "color": "#F7F0E2"},
                {"range": [70, 85], "color": "#EAF2EC"},
                {"range": [85, 100], "color": "#DDECE2"},
            ],
        },
    ))
    fig.update_layout(height=270, margin={"l": 20, "r": 20, "t": 48, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def input_image() -> Image.Image | None:
    section_head("INPUT", "포장 이미지 입력", "박스 네 모서리와 제품 전체가 보이도록 촬영하면 분석 품질이 좋아집니다.")
    source = st.radio("이미지 입력 방식", ["파일 업로드", "카메라 촬영"], horizontal=True, label_visibility="collapsed")
    if source == "파일 업로드":
        uploaded = st.file_uploader("이미지를 끌어다 놓거나 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("박스 전체가 보이도록 위에서 촬영하세요")
    return Image.open(uploaded).convert("RGB") if uploaded else None

if page == "시작":
    st.markdown("""<div class="hero">
        <div>
            <div class="kicker">AI PACKAGING DECISION SUPPORT</div>
            <h1>더 나은 박스 선택을,<br>더 빠르게.</h1>
            <p>사진과 실제 크기 정보를 바탕으로 현재 포장의 빈 공간을 확인하고, 등록된 규격 중 더 적합한 박스 후보를 비교합니다. 하나의 정답을 강요하기보다 판단에 필요한 근거와 한계를 함께 보여줍니다.</p>
            <div class="hero-note"><span class="hero-chip">이미지 분석</span><span class="hero-chip">빈 공간 계산</span><span class="hero-chip">박스 후보 비교</span><span class="hero-chip">신뢰도 공개</span></div>
        </div>
        <div class="hero-visual">
            <div class="flow-card"><div class="flow-no">01</div><div><b>이미지 입력</b><small>박스와 제품을 확인</small></div><div class="flow-tag">INPUT</div></div>
            <div class="flow-card"><div class="flow-no">02</div><div><b>포장 분석</b><small>Void · PVI · 신뢰도</small></div><div class="flow-tag">ANALYZE</div></div>
            <div class="flow-card"><div class="flow-no">03</div><div><b>후보 비교</b><small>규격과 개선 가능성</small></div><div class="flow-tag">COMPARE</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    cta1, cta2, _ = st.columns([1.15, 1.1, 3.2])
    with cta1:
        st.button("포장 분석 시작", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(page="포장 분석"))
    with cta2:
        st.caption("설치 없이 웹에서 바로 사용할 수 있습니다.")

    section_head("WHY", "현장의 문제는 ‘박스가 없어서’가 아니라 ‘선택 기준이 없어서’ 생깁니다.", "빠르게 판단해야 할수록 안전한 쪽, 즉 한 단계 큰 박스를 선택하기 쉬워집니다.")
    st.markdown("""<div class="problem-panel">
        <div class="problem-quote"><small>PACKAGING PAIN POINT</small>“이 물건은<br>몇 호 박스가 맞지?”</div>
        <div class="problem-list">
            <div class="problem-item"><b>반복되는 판단</b> · 상품마다 크기와 형태가 달라 매번 다시 고민합니다.</div>
            <div class="problem-item"><b>재포장 부담</b> · 작은 박스를 골랐다가 맞지 않으면 다시 포장해야 합니다.</div>
            <div class="problem-item"><b>경험 의존</b> · 숙련자의 감각을 신규 작업자나 소규모 판매자가 그대로 공유하기 어렵습니다.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    section_head("HOW IT WORKS", "3단계로 확인합니다.", "복잡한 수치보다 ‘현재 상태 → 비교 → 다음 행동’의 흐름을 먼저 보여줍니다.")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step-card"><div class="step-num">01</div><div class="card-title">사진과 실측값 입력</div><div class="card-copy">박스 내부 사진과 알고 있는 규격을 입력합니다. 실측 여부는 결과 신뢰도에 반영됩니다.</div></div>', unsafe_allow_html=True)
    s2.markdown('<div class="step-card"><div class="step-num">02</div><div class="card-title">현재 포장 분석</div><div class="card-copy">제품 점유율, 면적·체적 빈 공간, PVI와 촬영 품질을 함께 확인합니다.</div></div>', unsafe_allow_html=True)
    s3.markdown('<div class="step-card"><div class="step-num">03</div><div class="card-title">박스 후보 비교</div><div class="card-copy">등록된 규격 중 수용 가능한 후보를 비교하고 개선 가능성을 참고값으로 제시합니다.</div></div>', unsafe_allow_html=True)

    section_head("USE CASE", "이런 상황에서 활용할 수 있습니다.", "현재는 현장 검증을 위한 프로토타입이며 실제 업무 도입 전 추가 검증이 필요합니다.")
    u1, u2, u3 = st.columns(3)
    u1.markdown('<div class="use-card"><div class="role">우체국·택배 접수 현장</div><div class="pain">고객 물품에 맞는 박스 규격을 빠르게 비교할 때</div></div>', unsafe_allow_html=True)
    u2.markdown('<div class="use-card"><div class="role">소규모 온라인 판매자</div><div class="pain">다양한 상품을 반복 포장하며 박스 선택 시간을 줄이고 싶을 때</div></div>', unsafe_allow_html=True)
    u3.markdown('<div class="use-card"><div class="role">반품·풀필먼트 작업</div><div class="pain">재포장 과정의 판단 편차를 줄이고 표준화 가능성을 검토할 때</div></div>', unsafe_allow_html=True)

    st.markdown("""<div class="trust-strip"><div><b>결과는 ‘정답’이 아니라 의사결정 참고값입니다.</b><br><span>파손 위험, 재질 강도, 냉장·방수 조건과 실제 운송 충격은 별도 확인이 필요합니다.</span></div><span>Measured · Estimated · Limitations</span></div>""", unsafe_allow_html=True)

elif page == "지표 안내":
    st.title("포장 지표와 올바른 해석")
    st.markdown("""<div class="info-panel"><h3>PVI는 왜 필요한가요?</h3><p>“이 박스가 너무 큰 것 같다”는 느낌은 사람마다 다를 수 있습니다. PVI는 박스와 제품의 관계를 같은 기준으로 계산해, 여러 포장을 비교하고 개선 전·후의 변화를 확인하기 위한 프로젝트 내부 지표입니다.</p><p>이 프로그램에서는 혼동을 줄이기 위해 <b>두 종류의 값</b>을 함께 보여줍니다. 하나는 빈 공간 자체를 나타내는 <b>과포장 비율</b>이고, 다른 하나는 효율·빈 공간·보호성·지속가능성을 합산한 <b>포장가치점수</b>입니다.</p></div>""", unsafe_allow_html=True)

    st.subheader("1. 빈 공간 비율은 어떻게 계산하나요?")
    st.markdown("""<div class="formula-box"><b>제품 점유율(%)</b> = 제품이 차지한 면적 또는 체적 ÷ 박스 전체 면적 또는 체적 × 100<br><b>빈 공간 비율(%)</b> = 100 − 제품 점유율<br><br>예: 제품 점유율이 35%라면 빈 공간 비율은 65%입니다.</div>""", unsafe_allow_html=True)
    st.write("면적 기준은 사진에서 보이는 바닥 면적을 이용하고, 체적 기준은 입력한 박스 높이와 제품 높이를 함께 이용합니다. 제품이 여러 개이거나 형태가 복잡할수록 두 값이 다르게 나올 수 있습니다.")

    st.subheader("2. 빈 공간 비율을 어떻게 읽나요?")
    st.dataframe(pd.DataFrame([
        ["0~20%", "매우 낮음", "제품과 박스가 매우 밀착된 상태입니다. 보호 여유가 부족하지 않은지 함께 확인합니다."],
        ["20~40%", "낮음·적정 후보", "공간 사용이 비교적 효율적입니다. 제품 특성에 따라 적정 포장으로 볼 수 있습니다."],
        ["40~60%", "개선 검토", "빈 공간이 눈에 띄기 시작합니다. 더 작은 박스나 배치 변경을 검토합니다."],
        ["60~80%", "높음", "박스 축소, 완충 구조 개선, 묶음 배치 최적화가 필요할 가능성이 큽니다."],
        ["80% 초과", "매우 높음", "제품보다 박스가 지나치게 클 가능성이 높아 우선 개선 대상으로 봅니다."],
    ], columns=["빈 공간 비율", "해석", "쉽게 말하면"]), hide_index=True, use_container_width=True)
    st.caption("위 구간은 프로젝트 내 비교와 교육을 위한 운영 기준입니다. 제품의 파손 위험, 완충 필요성, 법적·산업별 기준에 따라 판단은 달라질 수 있습니다.")

    st.subheader("3. 포장가치점수(PVI Score)는 무엇인가요?")
    st.write("분석 화면의 100점 점수는 빈 공간만 보는 값이 아닙니다. 아래 네 항목을 합산하여 현재 포장이 공간을 효율적으로 사용하면서 제품도 보호하는지를 한눈에 보여줍니다.")
    st.dataframe(pd.DataFrame([
        ["공간 효율성", 35, "제품이 박스 바닥을 얼마나 효과적으로 사용하는지 평가합니다."],
        ["빈 공간 관리", 25, "불필요한 여유 공간이 적절히 관리되는지 평가합니다."],
        ["제품 보호성", 25, "너무 꽉 끼지 않고 이동·충격을 줄일 여유가 있는지 참고합니다."],
        ["지속가능성", 15, "박스와 완충재를 줄일 개선 가능성을 평가합니다."],
    ], columns=["평가 항목", "최대 점수", "무엇을 보는가"]), hide_index=True, use_container_width=True)

    st.dataframe(pd.DataFrame([
        ["90~100점", "A+ 최적", "효율과 보호의 균형이 매우 좋은 포장"],
        ["80~89점", "A 우수", "대체로 효율적이며 작은 보완만 필요한 포장"],
        ["70~79점", "B 양호", "사용 가능하지만 개선 여지가 있는 포장"],
        ["50~69점", "C 개선 필요", "빈 공간 또는 박스 규격을 우선 검토할 포장"],
        ["0~49점", "D 과포장", "박스 축소와 구조 개선이 필요한 가능성이 큰 포장"],
    ], columns=["점수", "등급", "의미"]), hide_index=True, use_container_width=True)

    st.subheader("4. 결과를 올바르게 해석하는 순서")
    st.markdown("""
1. **촬영 품질과 신뢰도**를 먼저 확인합니다.  
2. **면적 빈 공간**과 **체적 빈 공간**을 함께 봅니다.  
3. 제품이 깨지기 쉬운지, 완충재가 필요한지 등 **보호 조건**을 확인합니다.  
4. 추천 박스의 규격·비용·탄소 참고값을 비교합니다.  
5. 실제 샘플 포장과 낙하·압축 시험을 거쳐 최종 결정합니다.
""")
    st.warning("PVI는 공식 법정 시험, 환경성 인증 또는 제품 안전 인증이 아닙니다. 프로젝트 비교·교육·개선 의사결정을 위한 내부 지표입니다.")

elif page == "프로젝트 소개":
    st.title("Better Life For Us 프로젝트 소개")
    st.markdown("""<div class="analysis-intro"><div><div class="kicker">BETTER LIFE FOR US</div><h1>포장 선택에 기준을 더합니다.</h1><p>경험에 의존하던 박스 선택을 사진과 데이터로 확인하고, 더 적합한 후보를 비교할 수 있도록 만든 프로젝트입니다.</p></div><div class="analysis-note">GREEN VALUE YOUTH 2026</div></div>""", unsafe_allow_html=True)

    st.subheader("프로젝트 배경")
    st.write("온라인 소비와 택배 이용이 증가하면서 제품 크기와 맞지 않는 큰 박스와 내부 빈 공간이 일상적인 환경 문제로 나타나고 있습니다. 과도한 포장은 포장재와 완충재 사용을 늘릴 뿐 아니라 물류 적재 효율을 낮추고, 탄소배출과 폐기물 증가에도 영향을 줄 수 있습니다.")

    st.subheader("우리가 만들고 있는 해결 방법")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown('<div class="easy-card"><b>👁️ 1. 제품·박스 인식</b><small>사진에서 박스 경계와 제품 영역을 찾아 분석 가능한 데이터로 바꿉니다.</small></div>', unsafe_allow_html=True)
    p2.markdown('<div class="easy-card"><b>📐 2. 빈 공간 계산</b><small>박스와 제품의 면적·체적 관계를 계산해 포장 공간이 얼마나 비어 있는지 보여줍니다.</small></div>', unsafe_allow_html=True)
    p3.markdown('<div class="easy-card"><b>📊 3. PVI 분석</b><small>공간 효율, 빈 공간 관리, 보호성, 지속가능성을 점수와 등급으로 설명합니다.</small></div>', unsafe_allow_html=True)
    p4.markdown('<div class="easy-card"><b>🌱 4. 개선안 제안</b><small>더 적합한 박스 후보와 예상 재료·비용·탄소 절감 방향을 비교합니다.</small></div>', unsafe_allow_html=True)

    st.subheader("프로젝트가 추구하는 변화")
    st.markdown("""
- 소비자가 택배 속 빈 공간을 직관적으로 이해하도록 돕습니다.
- 기업과 소상공인이 포장 규격을 데이터로 비교하도록 지원합니다.
- 적정포장, 다회용·재사용 구조, 자원순환에 대한 사회적 관심을 높입니다.
- 분석 결과를 전시·캠페인·교육 콘텐츠와 연결해 친환경 소비 행동을 확산합니다.
""")

    st.subheader("프로젝트의 차별점")
    st.markdown("""<div class="info-panel"><p><b>① 감상이 아닌 데이터</b><br>“커 보인다”는 주관적 판단을 빈 공간과 점수로 시각화합니다.</p><p><b>② 진단에서 끝나지 않는 개선</b><br>문제 표시뿐 아니라 표준 박스 후보, 절감 가능성, 촬영 신뢰도와 한계까지 함께 제공합니다.</p><p><b>③ 기술과 시민 참여의 연결</b><br>AI 분석 결과를 캠페인, 교육, 다회용 박스 실험과 연결해 생활 속 자원순환 행동으로 확장합니다.</p></div>""", unsafe_allow_html=True)

    st.subheader("팀과 프로그램")
    st.markdown("""**Better Life For Us**는 LG생활건강 그린밸류 YOUTH 2026에서 자원순환과 소비습관을 주제로 활동하는 팀입니다. 본 플랫폼은 프로젝트의 분석 도구이자, 시민·청년·기업이 지속가능한 포장을 쉽게 이해하고 함께 개선 방향을 논의하기 위한 교육·소통 도구입니다.""")

    st.info("프로젝트 문구: 데이터로 포장을 이해하고, 더 나은 선택으로 바꾸는 지속가능한 포장 솔루션")

elif page == "신뢰도·한계":
    st.markdown("""<div class="analysis-intro"><div><div class="kicker">TRANSPARENT & RESPONSIBLE AI</div><h1>결과보다 먼저, 신뢰도를 확인합니다.</h1><p>측정값과 추정값을 구분하고 이번 분석이 놓칠 수 있는 조건과 사용 범위를 함께 안내합니다.</p></div><div class="analysis-note">MEASURED · ESTIMATED · LIMITATIONS</div></div>""", unsafe_allow_html=True)
    st.subheader("이 시스템이 할 수 있는 것")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="easy-card"><b>📐 이미지 기반 측정</b><small>박스와 제품의 보이는 영역을 분석해 공간 활용률과 빈 공간 비율을 계산합니다.</small></div>', unsafe_allow_html=True)
    c2.markdown('<div class="easy-card"><b>📦 후보 박스 비교</b><small>등록된 규격 중 제품 체적과 목표 활용률에 맞는 후보를 비교합니다.</small></div>', unsafe_allow_html=True)
    c3.markdown('<div class="easy-card"><b>🔎 근거 공개</b><small>계산식, 검출 신뢰도, 적용된 가정과 이번 분석의 제한사항을 보여줍니다.</small></div>', unsafe_allow_html=True)
    st.subheader("단독으로 판단하기 어려운 것")
    st.markdown("""
- 충격·진동·압축에 대한 실제 보호 성능
- 제품 재질, 파손 민감도, 누액·방수·냉장 조건
- 완충재의 실제 흡수 성능과 장거리 물류 환경
- 법적 포장 적합성, 환경 인증, 전과정평가(LCA)
- 등록되지 않은 박스 규격과 실제 구매·물류 단가
""")
    st.subheader("결과 표시 원칙")
    st.dataframe(pd.DataFrame([
        ["실측값", "사용자가 직접 확인해 입력한 박스 내부 규격", "결과의 기준값"],
        ["측정값", "이미지에서 계산한 제품·박스 영역과 빈 공간", "촬영 품질과 검출 신뢰도 함께 표시"],
        ["추정값", "평균 높이, 재료·탄소·비용 모델을 이용한 값", "가정과 오차 가능성 명시"],
        ["미검증", "실제 시험이나 충분한 검증 데이터가 없는 성능", "정확도 수치로 제시하지 않음"],
    ], columns=["구분", "의미", "표시 방식"]), hide_index=True, use_container_width=True)
    st.warning("사용 원칙: 중요한 포장 변경은 실제 치수 확인, 샘플 포장, 낙하·진동·압축 시험과 함께 검토해야 합니다.")
    st.info("프로젝트 원칙: 측정값은 보여주고, 추정값은 구분하며, 불확실한 결과에는 이유와 다음 행동을 함께 알려줍니다.")

elif page == "혼합 포장":
    render_mixed_packaging_page()

else:
    st.markdown("""<div class="analysis-intro"><div><div class="kicker">PACKAGING ANALYSIS</div><h1>현재 포장을 확인하고 후보를 비교합니다.</h1><p>사진과 실측값을 바탕으로 빈 공간을 계산하고, 분석 신뢰도와 등록된 박스 후보를 함께 살펴봅니다.</p></div><div class="analysis-note">STEP 1 · INPUT</div></div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("실측 입력")
        box_width_mm = st.number_input("박스 내부 가로(mm)", min_value=1.0, value=330.0)
        box_length_mm = st.number_input("박스 내부 세로(mm)", min_value=1.0, value=250.0)
        box_height_mm = st.number_input("박스 내부 높이(mm)", min_value=1.0, value=150.0)
        product_height_mm = st.number_input("제품 평균 높이(mm)", min_value=0.0, value=50.0)
        target_utilization = st.slider("추천 목표 활용률", 50, 90, 75) / 100
        dimensions_confirmed = st.checkbox("입력 규격은 직접 실측한 값입니다", value=False, help="체크하지 않으면 실제 길이·체적 결과를 추정값으로 표시합니다.")

    image = input_image()
    if image is None:
        st.info("이미지를 입력하면 자동 분석이 시작됩니다. 박스 네 모서리와 제품 전체가 보이는 사진을 권장합니다.")
        st.stop()

    quality = check_image_quality(image)
    progress = st.progress(0, text="1/6 이미지 품질을 확인하고 있습니다...")
    time.sleep(0.08)
    progress.progress(15, text="2/6 박스 경계를 찾고 원근을 보정하고 있습니다...")
    box_image, _, box = detect_box(image)
    time.sleep(0.08)
    progress.progress(35, text="3/6 제품 영역을 분리하고 있습니다...")
    detections, segmented_image, product_area, total_products = detect_products(image, box)
    time.sleep(0.08)
    progress.progress(55, text="4/6 면적·체적과 빈 공간을 계산하고 있습니다...")
    planar = calculate_planar_measurements(box["warped_width"], box["warped_height"], box_width_mm, box_length_mm, product_area)
    volume = calculate_volume_measurements(box_width_mm, box_length_mm, box_height_mm, planar["product_area_mm2"], product_height_mm)
    pvi = calculate_pvi(planar["box_area_px"], planar["product_area_px"])
    volume_pvi = calculate_volume_pvi(volume["occupancy_percent"])
    confidence = estimate_measurement_confidence(box["confidence"], [d["confidence"] for d in detections], box["perspective_corrected"], True)
    progress.progress(76, text="5/6 표준 박스를 비교하고 절감 효과를 계산하고 있습니다...")
    recommendation = recommend_box(volume["product_volume_mm3"], target_utilization)
    candidates = rank_boxes(volume["product_volume_mm3"], target_utilization, limit=4)
    time.sleep(0.08)
    progress.progress(100, text="6/6 분석이 완료되었습니다.")
    time.sleep(0.12)
    progress.empty()

    wasted_area_cm2 = max(planar["box_area_mm2"] - planar["product_area_mm2"], 0) / 100
    carbon = calculate_carbon(wasted_area_cm2)
    esg = calculate_esg_score(planar["void_percent"], carbon["carbon_g"])
    current_volume = box_width_mm * box_length_mm * box_height_mm
    consultant = build_consultant_report(
        area_void=planar["void_percent"], volume_void=volume["void_percent"], confidence=confidence,
        current_volume_mm3=current_volume, recommendation=recommendation, esg_score=esg["score"], detected_products=total_products,
    )
    limitation_result = build_limitations(
        quality=quality, box=box, detections=detections, product_height_mm=product_height_mm,
        dimensions_confirmed=dimensions_confirmed,
    )
    self_review = build_self_review(
        limitation_result=limitation_result, detected_products=total_products, product_height_mm=product_height_mm,
    )

    if recommendation:
        recommended_box = recommendation["box"]
        recommended_void = float(recommendation["void_percent"])
        volume_saving_percent = max((current_volume - recommended_box.volume) / current_volume * 100, 0.0) if current_volume else 0.0
        estimated_material_saving = min(max(volume_saving_percent * 0.78, 0.0), 65.0)
        estimated_carbon_saving = min(max(volume_saving_percent * 0.62, 0.0), 55.0)
        estimated_cost_saving = min(max(volume_saving_percent * 0.45, 0.0), 40.0)
        projected_health = min(96.0, pvi["pvi"] + max(planar["void_percent"] - recommended_void, 0) * 0.42)
    else:
        recommended_void = volume["void_percent"]
        volume_saving_percent = estimated_material_saving = estimated_carbon_saving = estimated_cost_saving = 0.0
        projected_health = pvi["pvi"]

    section_head("RESULT", "핵심 분석 요약", "가장 먼저 현재 포장 상태와 분석 신뢰도를 확인하세요.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("포장 효율 참고점수", f"{pvi['pvi']:.0f}/100")
    m2.metric("면적 Void", f"{planar['void_percent']:.1f}%")
    m3.metric("체적 Void", f"{volume['void_percent']:.1f}%")
    m4.metric("ESG", f"{esg['score']} / {esg['grade']}")
    m5.metric("신뢰도", f"{confidence:.1f}%")
    st.warning(f"**{consultant['status']} · 우선순위 {consultant['severity']}** — {consultant['summary']}")
    level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[limitation_result["level"]]
    st.info(f"{level_icon} **분석 품질: {limitation_result['label']}** — {limitation_result['decision']}")

    with st.expander("결과를 쉽게 설명해 주세요", expanded=True):
        st.markdown(f"""
**현재 사진에서 제품이 차지한 바닥 면적은 약 {planar['occupancy_percent']:.1f}%이고, 비어 있는 면적은 약 {planar['void_percent']:.1f}%입니다.**  
입력한 높이까지 반영한 체적 기준 빈 공간은 약 **{volume['void_percent']:.1f}%**로 추정됩니다.

- **포장 효율 참고점수 {pvi['pvi']:.0f}점**: 공간 효율, 빈 공간 관리, 제품 보호성, 지속가능성을 합산한 점수입니다.
- **ESG {esg['score']}점**: 현재 프로그램의 포장재·탄소 추정 로직을 이용한 비교용 참고 점수입니다.
- **측정 신뢰도 {confidence:.1f}%**: 박스 검출, 제품 분할, 원근 보정과 실측값 입력 여부를 종합한 값입니다.

점수가 낮거나 빈 공간이 높다면 더 작은 박스, 제품 배치 변경, 내부 고정 구조 또는 완충재 최소화를 검토합니다. 다만 깨지기 쉬운 제품은 보호 여유가 필요하므로 빈 공간 수치만으로 박스를 결정하면 안 됩니다.
""")

    if presentation_mode:
        st.markdown("<div class='section-title'>🎤 발표용 핵심 결과</div>", unsafe_allow_html=True)
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.markdown(f"<div class='presentation-card'><div>현재 포장 건강도</div><div class='big'>{pvi['pvi']:.0f}</div><small>{pvi['grade']}</small></div>", unsafe_allow_html=True)
        pc2.markdown(f"<div class='presentation-card'><div>빈 공간 비율</div><div class='big'>{volume['void_percent']:.1f}%</div><small>체적 기준</small></div>", unsafe_allow_html=True)
        pc3.markdown(f"<div class='presentation-card'><div>예상 개선 점수</div><div class='big'>{projected_health:.0f}</div><small>추천 박스 적용 시</small></div>", unsafe_allow_html=True)
        pc4.markdown(f"<div class='presentation-card'><div>예상 탄소 절감</div><div class='big'>{estimated_carbon_saving:.0f}%</div><small>비교용 추정치</small></div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(pvi["pvi"], "현재 포장 건강도"), use_container_width=True)
        st.success(f"핵심 메시지: {consultant['summary']}")
        if recommendation:
            rb = recommendation['box']
            st.info(f"추천 박스: {rb.company} {rb.code} · {rb.length}×{rb.width}×{rb.height} mm · 예상 Void {recommended_void:.1f}%")
    else:
        tabs = st.tabs(["요약", "분석 근거", "개선 비교", "이미지", "3D", "박스 후보", "AI 진단", "신뢰도"])

        with tabs[0]:
            g1, g2, g3 = st.columns(3)
            g1.plotly_chart(gauge(pvi["pvi"], "PVI"), use_container_width=True)
            g2.plotly_chart(gauge(100 - planar["void_percent"], "면적 효율", "%"), use_container_width=True)
            g3.plotly_chart(gauge(esg["score"], "ESG 점수"), use_container_width=True)
            score_df = pd.DataFrame([{"항목": k, "점수": v, "최대": pvi["max_scores"][k]} for k, v in pvi["scores"].items()])
            st.bar_chart(score_df.set_index("항목")[["점수", "최대"]])
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("박스 체적", f"{volume['box_volume_mm3']/1000:,.0f} cc")
            v2.metric("제품 추정 체적", f"{volume['product_volume_mm3']/1000:,.0f} cc")
            v3.metric("제품 점유율", f"{volume['occupancy_percent']:.1f}%")
            v4.metric("체적 PVI", f"{volume_pvi['volume_pvi']} / {volume_pvi['grade']}")

        with tabs[1]:
            st.subheader("AI 분석 근거와 계산 과정")
            trace_rows = [
                ("1", "이미지 품질 확인", f"품질 점수 {quality['score']:.0f}/100 · {quality['message']}"),
                ("2", "박스 경계 검출", f"검출 신뢰도 {box['confidence']*100:.1f}% · 원근 보정 {'적용' if box['perspective_corrected'] else '미적용'}"),
                ("3", "제품 영역 분리", f"제품 {total_products}개 · 평균 신뢰도 {(sum(d['confidence'] for d in detections)/len(detections)*100) if detections else 0:.1f}%"),
                ("4", "면적 환산", f"박스 {planar['box_area_mm2']/100:.1f}cm² · 제품 {planar['product_area_mm2']/100:.1f}cm²"),
                ("5", "체적 환산", f"박스 {volume['box_volume_mm3']/1000:.1f}cc · 제품 {volume['product_volume_mm3']/1000:.1f}cc"),
                ("6", "PVI·ESG 계산", f"포장 건강도 {pvi['pvi']:.0f}점 · ESG {esg['score']}점"),
                ("7", "추천 박스 탐색", f"후보 {len(candidates)}개 비교 · 목표 활용률 {target_utilization*100:.0f}%"),
            ]
            for no, title, detail in trace_rows:
                st.markdown(f"<div class='trace-card'><b>{no}. {title}</b><br><small>{detail}</small></div>", unsafe_allow_html=True)

            st.subheader("실제 계산값")
            st.markdown(f"""<div class='formula-box'>
<b>면적 제품 점유율</b> = {planar['product_area_px']:,.0f} ÷ {planar['box_area_px']:,.0f} × 100 = <b>{planar['occupancy_percent']:.1f}%</b><br>
<b>면적 빈 공간</b> = 100 − {planar['occupancy_percent']:.1f} = <b>{planar['void_percent']:.1f}%</b><br>
<b>체적 제품 점유율</b> = {volume['product_volume_mm3']/1000:,.1f}cc ÷ {volume['box_volume_mm3']/1000:,.1f}cc × 100 = <b>{volume['occupancy_percent']:.1f}%</b><br>
<b>체적 빈 공간</b> = 100 − {volume['occupancy_percent']:.1f} = <b>{volume['void_percent']:.1f}%</b>
</div>""", unsafe_allow_html=True)
            st.caption("포장 건강도는 위 빈 공간 값뿐 아니라 공간 효율성, 빈 공간 관리, 보호성, 지속가능성 점수를 합산합니다.")

        with tabs[2]:
            st.subheader("현재 포장과 추천 포장 비교")
            before_after = pd.DataFrame([
                ["체적 빈 공간", f"{volume['void_percent']:.1f}%", f"{recommended_void:.1f}%", f"-{max(volume['void_percent']-recommended_void,0):.1f}%p"],
                ["포장 건강도", f"{pvi['pvi']:.0f}점", f"{projected_health:.0f}점", f"+{max(projected_health-pvi['pvi'],0):.0f}점"],
                ["박스 체적", f"{current_volume/1000:,.0f}cc", f"{(recommendation['box'].volume/1000 if recommendation else current_volume/1000):,.0f}cc", f"-{volume_saving_percent:.1f}%"],
                ["포장재 사용", "100 기준", f"{100-estimated_material_saving:.0f} 기준", f"-{estimated_material_saving:.0f}%"],
                ["탄소 영향", "100 기준", f"{100-estimated_carbon_saving:.0f} 기준", f"-{estimated_carbon_saving:.0f}%"],
                ["예상 비용", "100 기준", f"{100-estimated_cost_saving:.0f} 기준", f"-{estimated_cost_saving:.0f}%"],
            ], columns=["비교 항목", "현재", "추천 적용", "예상 변화"])
            st.dataframe(before_after, hide_index=True, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("포장재 절감", f"{estimated_material_saving:.0f}%", help="박스 체적 축소율을 이용한 내부 비교용 추정치")
            c2.metric("탄소 절감", f"{estimated_carbon_saving:.0f}%", help="공식 LCA가 아닌 비교용 추정치")
            c3.metric("비용 절감", f"{estimated_cost_saving:.0f}%", help="실제 계약·운송 단가에 따라 달라질 수 있음")
            st.warning("절감률은 추천 박스 체적을 기준으로 계산한 시뮬레이션 값입니다. 실제 재질, 골지, 구매 단가, 운송 조건에 따라 달라집니다.")

        with tabs[3]:
            i1, i2, i3 = st.columns(3)
            i1.image(image, caption="원본", use_container_width=True)
            i2.image(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB), caption="박스 경계", use_container_width=True)
            i3.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="제품 분할", use_container_width=True)
            st.info(f"품질 {quality['score']:.0f}/100 · 밝기 {quality['brightness']} · 선명도 {quality['sharpness']} · {quality['message']}")
            if detections:
                st.dataframe(pd.DataFrame(detections), hide_index=True, use_container_width=True)
            else:
                st.error("제품 영역을 찾지 못했습니다. 균일한 조명과 단색 배경에서 다시 촬영하세요.")

        with tabs[4]:
            ratio = max(planar["occupancy_percent"] / 100, 0)
            aspect = box_width_mm / max(box_length_mm, 1)
            product_width = min(box_width_mm, np.sqrt(planar["product_area_mm2"] * aspect)) if planar["product_area_mm2"] > 0 else 1
            product_length = min(box_length_mm, planar["product_area_mm2"] / max(product_width, 1))
            st.plotly_chart(build_digital_twin(box_width_mm, box_length_mm, box_height_mm, product_width, product_length, product_height_mm), use_container_width=True)
            st.caption("제품 영역을 동일 면적의 직사각형으로 단순화한 근사 3D 모델입니다. 실제 형상 복원 결과가 아닙니다.")

        with tabs[5]:
            if candidates:
                rows = []
                for idx, item in enumerate(candidates, 1):
                    b = item["box"]
                    rows.append({"순위": idx, "후보": f"{b.company} {b.code}", "규격(mm)": f"{b.length}×{b.width}×{b.height}", "활용률(%)": round(item["utilization"]*100,1), "Void(%)": round(item["void_percent"],1), "비용(원)": b.cost, "탄소(kgCO₂e)": b.carbon, "종합점수": item["total_score"]})
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.bar_chart(df.set_index("후보")[["종합점수"]])
                best = candidates[0]["box"]
                st.success(f"최우선 추천: **{best.company} {best.code}** ({best.length} × {best.width} × {best.height} mm)")
            else:
                st.warning("현재 DB에서 제품 체적을 수용하는 표준 박스를 찾지 못했습니다. 맞춤 박스를 검토하세요.")

        with tabs[6]:
            st.subheader("AI 포장 종합 진단")
            st.write(consultant["summary"])
            for action in consultant["actions"]:
                st.write(f"• {action}")
            if consultant["savings"]:
                s = consultant["savings"]
                st.info(f"추천 {s['box_name']} · {s['size']} · 예상 Void {s['expected_void_percent']}% · 현재 박스 대비 체적 절감 {s['volume_saving_percent']}%")
            st.write(consultant["esg_comment"])
            st.markdown("#### 진단 구조")
            st.markdown(f"""
- **현재 문제:** 면적 빈 공간 {planar['void_percent']:.1f}%, 체적 빈 공간 {volume['void_percent']:.1f}%로 분석되었습니다.
- **주요 원인:** 제품 대비 박스 가로·세로 또는 높이가 크거나, 제품 배치가 중앙에 집중되어 있을 가능성이 있습니다.
- **환경 영향:** 불필요한 골판지·완충재 사용과 운송 적재 효율 저하 가능성이 있습니다.
- **개선 방법:** 추천 박스 검토, 높이 축소, 제품 회전 배치, 최소 고정 구조 적용 순으로 확인합니다.
- **예상 효과:** 포장재 {estimated_material_saving:.0f}%, 탄소 {estimated_carbon_saving:.0f}%, 비용 {estimated_cost_saving:.0f}% 절감 가능성을 시뮬레이션했습니다.
- **보호 주의:** 파손 위험 제품은 빈 공간을 무조건 줄이지 말고 낙하·진동 시험과 함께 검증해야 합니다.
""")
            with st.expander("ESG 참고 점수가 달라지는 이유", expanded=True):
                st.markdown(f"""
**현재 ESG 참고 점수는 {esg['score']}점({esg['grade']})입니다.**

- 박스가 작아지면 골판지 표면적과 재료 사용량을 줄일 수 있습니다.
- 동일 차량에 더 많은 상자를 적재할 수 있어 운송 효율이 높아질 수 있습니다.
- 완충재와 폐기물 발생량을 줄이는 방향으로 작용합니다.
- 표준 규격을 단순화하면 구매·보관·포장 공정 효율도 높일 수 있습니다.

다만 이 값은 공식 환경성적표지나 전과정평가(LCA)가 아닌 **프로젝트 내부 비교용 지표**입니다.
""")
            question = st.selectbox("분석 결과 질문", ["왜 이 박스를 추천했나요?", "탄소는 왜 줄어드나요?", "정확도를 높이려면 어떻게 하나요?", "제품 보호에는 문제가 없나요?"])
            answers = {
                "왜 이 박스를 추천했나요?": "제품 추정 체적을 수용하는 후보 중 목표 활용률, 박스 비용, 탄소값을 함께 비교해 종합점수가 가장 높은 후보를 선택했습니다.",
                "탄소는 왜 줄어드나요?": "박스가 작아지면 골판지 표면적과 완충재 사용량, 운송 적재 공간이 줄어드는 방향으로 작용합니다. 현재 값은 비교용 추정치입니다.",
                "정확도를 높이려면 어떻게 하나요?": "박스 내부 실측값을 입력하고 카메라를 수직 상단에 두며, 강한 그림자와 반사를 피하고 제품이 겹치지 않게 촬영하세요.",
                "제품 보호에는 문제가 없나요?": "본 추천은 체적과 효율 중심입니다. 파손 위험, 재질 강도, 완충 설계, 냉장·방수 요구는 별도 검증해야 합니다.",
            }
            st.info(answers[question])

        with tabs[7]:
            st.subheader("분석 신뢰도")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("이미지 품질", f"{limitation_result['quality_score']:.0f}/100")
            r2.metric("박스 검출", f"{limitation_result['box_confidence']:.1f}%")
            r3.metric("제품 검출 평균", f"{limitation_result['product_confidence']:.1f}%")
            r4.metric("종합 측정", f"{confidence:.1f}%")

            level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[limitation_result["level"]]
            if limitation_result["level"] == "red":
                st.error(f"{level_icon} 분석 품질: {limitation_result['label']} — {limitation_result['decision']}")
            elif limitation_result["level"] == "yellow":
                st.warning(f"{level_icon} 분석 품질: {limitation_result['label']} — {limitation_result['decision']}")
            else:
                st.success(f"{level_icon} 분석 품질: {limitation_result['label']} — {limitation_result['decision']}")

            st.markdown("#### 이번 분석에서 확인된 제한사항")
            for item in limitation_result["limitations"]:
                st.write(f"• {item}")

            if limitation_result["actions"]:
                st.markdown("#### 정확도를 높이는 다음 행동")
                for action in limitation_result["actions"]:
                    st.write(f"• {action}")

            st.markdown("#### AI Self Review")
            for check in self_review:
                st.write(check)
            st.caption("Self Review는 모델이 스스로 정확성을 증명하는 기능이 아니라, 입력 조건과 계산 범위를 다시 점검해 과도한 해석을 막기 위한 안전 장치입니다.")

            st.markdown("#### 적용 범위")
            st.markdown("""
- **사용 가능:** 포장 후보 비교, 빈 공간 개선 아이디어 탐색, 교육·캠페인·사전 검토
- **추가 확인 필요:** 실제 박스 규격 확정, 제품 보호 구조 설계, 비용·탄소 절감량 확정
- **대체 불가:** 법적 적합성 판단, 공인 시험, 낙하·진동·압축 시험, 공식 LCA·환경 인증
""")

    report_sections = {
        "핵심 분석 요약": {
            "포장 건강도 Score": f"{pvi['pvi']:.0f}/100",
            "등급": pvi["grade"], "진단": consultant["status"], "요약": consultant["summary"],
        },
        "공간 및 체적 분석": {
            "면적 점유율": f"{planar['occupancy_percent']:.1f}%", "면적 Void": f"{planar['void_percent']:.1f}%",
            "체적 점유율": f"{volume['occupancy_percent']:.1f}%", "체적 Void": f"{volume['void_percent']:.1f}%",
            "검출 제품 수": total_products,
        },
        "ESG 및 환경 영향": {
            "ESG 점수": f"{esg['score']} / {esg['grade']}", "추정 포장재 낭비": f"{carbon['wasted_material_g']:.2f} g",
            "추정 탄소 영향": f"{carbon['carbon_g']:.2f} gCO₂",
        },
        "추천 결과": ({
            "추천 박스": consultant["savings"]["box_name"], "추천 규격": consultant["savings"]["size"],
            "예상 Void": f"{consultant['savings']['expected_void_percent']}%", "체적 절감": f"{consultant['savings']['volume_saving_percent']}%",
        } if consultant["savings"] else {"결과": "적합한 표준 박스 후보 없음"}),
        "개선 권고": consultant["actions"],
        "신뢰도": {
            "이미지 품질": f"{limitation_result['quality_score']:.0f}/100",
            "박스 검출": f"{limitation_result['box_confidence']:.1f}%",
            "제품 검출 평균": f"{limitation_result['product_confidence']:.1f}%",
            "종합 측정 신뢰도": f"{confidence:.1f}%",
            "분석 품질 등급": limitation_result["label"],
            "해석": consultant["reliability"],
        },
        "이번 분석에서 확인된 제한사항": limitation_result["limitations"],
        "사용자가 추가로 확인해야 할 사항": limitation_result["actions"] or ["실제 치수와 제품 보호 조건을 확인하세요."],
        "AI 자기검토": self_review,
        "계산 및 적용 범위": {
            "측정값": "이미지에서 계산한 박스·제품 영역과 빈 공간",
            "추정값": "평균 높이와 내부 환경·비용 모델을 이용한 비교값",
            "미평가": "충격·진동·압축, 재질 강도, 법적 적합성, 공식 LCA",
        },
    }
    pdf_bytes = build_pdf_report({"sections": report_sections})
    st.divider()
    col_report, col_cert = st.columns(2)
    with col_report:
        st.download_button("투명성 보고서 다운로드", pdf_bytes, "AI_포장분석_투명성보고서_V9_3.pdf", "application/pdf", use_container_width=True)
    certificate_bytes = build_analysis_certificate({
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "health_score": f"{pvi['pvi']:.0f}/100",
        "grade": pvi["grade"],
        "area_void": f"{planar['void_percent']:.1f}%",
        "volume_void": f"{volume['void_percent']:.1f}%",
        "esg": f"{esg['score']} / {esg['grade']}",
        "confidence": f"{confidence:.1f}%",
        "recommended_box": consultant["savings"]["box_name"] if consultant["savings"] else "맞춤 박스 검토",
        "recommended_size": consultant["savings"]["size"] if consultant["savings"] else "표준 후보 없음",
        "carbon_saving": f"{estimated_carbon_saving:.0f}%",
        "summary": consultant["summary"],
    })
    with col_cert:
        st.download_button("분석 인증서 다운로드", certificate_bytes, "AI_포장분석_인증서_V9_3.pdf", "application/pdf", use_container_width=True)


st.markdown("""<div class="footer-brand"><strong>Better Life For Us</strong><span>AI Packaging Intelligence · 데이터로 포장을 이해하고, 더 나은 선택을 돕습니다.</span></div>""", unsafe_allow_html=True)