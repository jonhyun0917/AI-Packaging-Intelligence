from __future__ import annotations

import cv2
import time
from datetime import datetime
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
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
from database.box_db import BOX_DATABASE
from pages.mixed_packaging import render_mixed_packaging_page
from reporting.report import build_pdf_report
from reporting.certificate import build_analysis_certificate

st.set_page_config(
    page_title="AI Packaging Intelligence | Better Life For Us",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "greenvalue_logo_cropped.png"

st.markdown("""
<style>
:root {
    --ink:#162019;
    --ink-soft:#34463B;
    --muted:#66746B;
    --forest:#1F5A42;
    --forest-strong:#184936;
    --forest-soft:#EAF2ED;
    --paper:#F6F8F7;
    --white:#FFFFFF;
    --line:#DFE6E2;
    --line-strong:#CFD9D3;
    --amber:#9A6B26;
    --red:#A94B43;
}

html, body, [class*="css"] {
    font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Noto Sans KR","Segoe UI",sans-serif;
    color:var(--ink);
}
body {letter-spacing:-.01em;}
.stApp {background:var(--paper);}
.block-container {padding-top:1.65rem;padding-bottom:4rem;max-width:1180px;}
#MainMenu, footer, header {visibility:hidden;}
h1,h2,h3,h4 {letter-spacing:-.04em;color:var(--ink);}
p {word-break:keep-all;}

/* Sidebar */
[data-testid="stSidebar"] {background:#FBFCFB;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container {padding-top:1.15rem;padding-left:1rem;padding-right:1rem;}
[data-testid="stSidebar"] h3 {font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#8A958E;margin:1rem 0 .45rem;}
[data-testid="stSidebar"] [data-baseweb="radio"] label {padding:.58rem .68rem;border-radius:9px;margin:.04rem 0;transition:background .15s ease;color:#4F5F55;}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {background:#F0F4F1;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#89938D;}
.sidebar-brand {padding:.2rem 0 .75rem;}
.sidebar-brand .name {font-size:1.02rem;font-weight:800;letter-spacing:-.035em;color:var(--forest);}
.sidebar-brand .copy {font-size:.76rem;line-height:1.55;color:#7C8881;margin-top:.28rem;}
.version-pill {display:inline-block;background:#F0F4F1;border:1px solid var(--line);color:#6A776F;padding:5px 9px;border-radius:999px;font-size:.68rem;font-weight:750;}

/* Type hierarchy */
.kicker,.eyebrow {font-size:.69rem;letter-spacing:.12em;font-weight:850;color:var(--forest);text-transform:uppercase;}
.section-head {margin:2.7rem 0 1.05rem;}
.section-head h2 {font-size:clamp(1.28rem,2.1vw,1.55rem);margin:.25rem 0 0;font-weight:800;}
.section-head p {margin:.4rem 0 0;color:var(--muted);font-size:.91rem;line-height:1.72;max-width:760px;}
.card-title {font-size:1rem;font-weight:800;color:var(--ink);margin-bottom:7px;letter-spacing:-.03em;}
.card-copy {font-size:.86rem;line-height:1.68;color:var(--muted);}

/* Hero */
.hero {
    display:grid;
    grid-template-columns:minmax(0,1.08fr) minmax(330px,.92fr);
    gap:54px;
    align-items:center;
    padding:48px 50px;
    border:1px solid var(--line);
    border-radius:24px;
    background:#FFFFFF;
    box-shadow:0 18px 55px rgba(31,72,52,.055);
}
.hero h1 {margin:10px 0 15px;font-size:clamp(2.35rem,4vw,3.25rem);line-height:1.08;font-weight:860;letter-spacing:-.065em;}
.hero p {margin:0;max-width:680px;color:#59675E;font-size:.98rem;line-height:1.8;}
.hero-note {margin-top:18px;display:flex;gap:7px;flex-wrap:wrap;}
.hero-chip {display:inline-flex;align-items:center;padding:6px 9px;background:#F5F7F6;border:1px solid var(--line);border-radius:999px;color:#627067;font-size:.72rem;font-weight:680;}
.product-preview {border:1px solid var(--line-strong);border-radius:18px;background:#FBFCFB;padding:18px;box-shadow:0 10px 30px rgba(29,64,47,.04);}
.preview-top {display:flex;align-items:center;justify-content:space-between;gap:10px;padding-bottom:13px;border-bottom:1px solid var(--line);font-size:.76rem;color:#68756D;}
.preview-top b {font-size:.82rem;color:var(--ink);}
.preview-badge {font-size:.63rem;font-weight:800;letter-spacing:.07em;color:var(--forest);background:var(--forest-soft);padding:5px 7px;border-radius:999px;white-space:nowrap;}
.preview-grid {display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:14px 0;}
.preview-metric {background:#FFF;border:1px solid var(--line);border-radius:12px;padding:12px 13px;}
.preview-metric small {display:block;color:#8A958E;font-size:.65rem;margin-bottom:4px;}
.preview-metric strong {font-size:1.13rem;color:var(--forest);letter-spacing:-.04em;}
.preview-row {display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 2px;border-top:1px solid #E9EEEB;font-size:.76rem;color:#58675E;}
.preview-row b {font-size:.76rem;color:var(--ink);font-weight:700;}
.preview-state {font-size:.62rem;color:#728078;background:#F0F4F1;border-radius:999px;padding:4px 7px;font-weight:750;}
.preview-foot {font-size:.68rem;color:#8B958F;margin-top:9px;line-height:1.5;}

/* Home */
.problem-panel {display:grid;grid-template-columns:.78fr 1.22fr;gap:22px;background:#173E2E;color:white;border-radius:20px;padding:29px 31px;}
.problem-quote {font-size:clamp(1.3rem,2.4vw,1.72rem);line-height:1.42;font-weight:820;letter-spacing:-.045em;}
.problem-quote small {display:block;font-size:.66rem;font-weight:700;letter-spacing:.1em;opacity:.65;margin-bottom:10px;}
.problem-list {display:grid;gap:9px;}
.problem-item {background:rgba(255,255,255,.065);border:1px solid rgba(255,255,255,.1);border-radius:11px;padding:12px 14px;font-size:.82rem;line-height:1.6;color:#E7EFEA;}
.problem-item b {color:#FFF;}
.step-card,.use-card,.easy-card,.presentation-card {background:#FFF;border:1px solid var(--line);border-radius:15px;padding:20px;}
.step-card {min-height:148px;}
.step-num {display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:8px;background:var(--forest-soft);color:var(--forest);font-weight:850;font-size:.69rem;margin-bottom:12px;}
.use-card {min-height:120px;}
.use-card .role {font-weight:800;color:var(--ink);font-size:.95rem;margin-bottom:7px;}
.use-card .pain {color:var(--muted);font-size:.83rem;line-height:1.62;}
.trust-strip {display:flex;justify-content:space-between;gap:20px;align-items:center;background:#EEF4F0;border:1px solid #D8E4DC;border-radius:14px;padding:17px 19px;margin-top:23px;}
.trust-strip b {color:var(--forest);font-size:.9rem;}
.trust-strip span {color:#6B786F;font-size:.78rem;line-height:1.55;}

/* Streamlit widgets */
[data-testid="stMetric"] {background:#FFF;border:1px solid var(--line);padding:16px;border-radius:14px;box-shadow:none;}
[data-testid="stMetricLabel"] {font-weight:650;color:#738078;font-size:.78rem;}
[data-testid="stMetricValue"] {font-weight:820;color:var(--ink);font-size:1.38rem;letter-spacing:-.04em;}
[data-testid="stFileUploader"] {background:#FFF;border:1.5px dashed #A8B8AE;border-radius:15px;padding:10px;}
[data-testid="stFileUploader"] section {padding:1.45rem 1rem;}
.stButton>button,.stDownloadButton>button {border-radius:10px;min-height:43px;font-weight:760;border:1px solid var(--forest);background:var(--forest);color:white;box-shadow:none;transition:.15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:var(--forest-strong);background:var(--forest-strong);color:white;transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"] {gap:3px;background:#EEF2EF;padding:4px;border-radius:11px;overflow-x:auto;scrollbar-width:none;}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {display:none;}
.stTabs [data-baseweb="tab"] {height:40px;border-radius:8px;font-weight:720;padding:0 13px;color:#65736A;font-size:.83rem;white-space:nowrap;}
.stTabs [aria-selected="true"] {background:#FFF !important;color:var(--forest) !important;box-shadow:0 1px 6px rgba(28,57,42,.05);}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:var(--line)!important;}

/* Analysis */
.analysis-intro {background:#FFF;border:1px solid var(--line);border-radius:18px;padding:24px 26px;margin-bottom:18px;display:flex;justify-content:space-between;gap:24px;align-items:flex-end;}
.analysis-intro h1 {margin:5px 0 7px;font-size:clamp(1.65rem,2.8vw,2.05rem);}
.analysis-intro p {margin:0;color:var(--muted);line-height:1.7;font-size:.9rem;max-width:720px;}
.analysis-note {white-space:nowrap;font-size:.7rem;color:#6E7D74;background:#F3F6F4;border:1px solid var(--line);border-radius:999px;padding:6px 9px;}
.result-strip {display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.2rem 0 1rem;}
.result-card {background:#FFF;border:1px solid var(--line);border-radius:14px;padding:16px 17px;}
.result-card small {display:block;color:#7B8880;font-size:.7rem;margin-bottom:7px;}
.result-card strong {font-size:1.45rem;color:var(--ink);letter-spacing:-.045em;}
.result-card .sub {font-size:.68rem;color:#89948E;margin-top:4px;}
.status-line {display:flex;align-items:flex-start;gap:10px;background:#FFF;border:1px solid var(--line);border-radius:13px;padding:14px 16px;margin:10px 0;}
.status-dot {width:8px;height:8px;border-radius:50%;margin-top:6px;flex:none;}
.status-dot.green {background:#3E8C65}.status-dot.yellow{background:#C69537}.status-dot.red{background:#B85A50}
.status-line b {font-size:.88rem;color:var(--ink);}
.status-line span {display:block;font-size:.8rem;color:var(--muted);line-height:1.55;margin-top:2px;}
.info-panel {background:#FFF;border:1px solid var(--line);border-radius:15px;padding:20px 22px;margin:12px 0;}
.info-panel h3 {margin:0 0 9px;font-size:1.03rem;}
.info-panel p {line-height:1.75;margin:.35rem 0;color:#56645C;font-size:.89rem;}
.easy-card {min-height:148px;}
.easy-card b {display:block;color:var(--ink);font-size:.94rem;margin-bottom:8px;}
.easy-card small {font-size:.82rem;line-height:1.65;color:var(--muted);}
.formula-box {background:#F1F6F3;border-left:3px solid var(--forest);border-radius:10px;padding:16px 18px;margin:13px 0;font-size:.9rem;line-height:1.82;}
.trace-card {background:#FFF;border:1px solid var(--line);border-radius:11px;padding:13px 15px;margin:6px 0;}
.trace-card b {color:var(--forest);}
.presentation-card {min-height:138px;}
.presentation-card .big {font-size:1.75rem;font-weight:850;color:var(--forest);letter-spacing:-.05em;}
.report-strip {margin-top:1.8rem;background:#FFF;border:1px solid var(--line);border-radius:15px;padding:20px 22px;}
.report-strip h3 {margin:0 0 5px;font-size:1rem;}
.report-strip p {margin:0 0 14px;color:var(--muted);font-size:.82rem;line-height:1.6;}
.footer-brand {margin-top:3rem;padding:17px 2px;border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:18px;color:#7A877F;}
.footer-brand strong {color:var(--forest);font-size:.85rem;}
.footer-brand span {font-size:.72rem;text-align:right;line-height:1.5;}

@media (max-width: 900px) {
    .block-container {padding-top:1.15rem;padding-left:1rem;padding-right:1rem;}
    .hero {grid-template-columns:1fr;padding:31px 27px;gap:27px;}
    .problem-panel {grid-template-columns:1fr;}
    .analysis-intro {display:block;}
    .analysis-note {display:inline-block;margin-top:11px;}
    .result-strip {grid-template-columns:repeat(2,1fr);}
    .footer-brand {display:block;}
    .footer-brand span {display:block;text-align:left;margin-top:6px;}
}
@media (max-width: 640px) {
    .block-container {padding-top:.8rem;padding-left:.78rem;padding-right:.78rem;padding-bottom:3rem;}
    .hero {padding:24px 20px;border-radius:18px;gap:22px;}
    .hero h1 {font-size:2rem;}
    .hero p {font-size:.89rem;line-height:1.68;}
    .hero-chip {font-size:.68rem;}
    .product-preview {padding:14px;border-radius:14px;}
    .problem-panel {padding:22px 20px;border-radius:16px;}
    .step-card,.use-card,.easy-card {min-height:0;padding:17px;}
    .trust-strip {display:block;}
    .trust-strip>span {display:block;margin-top:8px;}
    .analysis-intro {padding:20px;border-radius:15px;}
    .result-strip {grid-template-columns:1fr 1fr;gap:8px;}
    .result-card {padding:14px;}
    .result-card strong {font-size:1.25rem;}
    [data-testid="stMetric"] {padding:13px;}
    [data-testid="stMetricValue"] {font-size:1.18rem;}
    .stTabs [data-baseweb="tab"] {font-size:.78rem;padding:0 10px;}
}


/* V9.7 · mobile-first app entry */
.app-brandbar {display:flex;align-items:center;justify-content:space-between;gap:14px;margin:.15rem 0 1.3rem;padding:0 2px;}
.app-brand {display:flex;align-items:center;gap:9px;color:var(--ink);font-size:.9rem;font-weight:830;letter-spacing:-.03em;}
.app-mark {display:inline-flex;align-items:center;justify-content:center;width:31px;height:31px;border-radius:10px;background:var(--forest);color:#fff;font-size:.7rem;font-weight:900;letter-spacing:.02em;box-shadow:0 5px 14px rgba(31,90,66,.15);}
.app-team {color:#819087;font-size:.72rem;font-weight:680;}
.home-hero-copy {padding:28px 0 24px;}
.home-hero-copy h1 {font-size:clamp(2.45rem,5vw,3.85rem);line-height:1.03;margin:10px 0 18px;letter-spacing:-.075em;max-width:620px;}
.home-hero-copy p {font-size:1rem;line-height:1.78;color:#55645B;max-width:610px;margin:0 0 18px;}
.home-points {display:flex;gap:8px;flex-wrap:wrap;margin:0 0 17px;}
.home-point {font-size:.72rem;color:#526158;background:#F1F5F2;border:1px solid #DDE6E0;padding:7px 10px;border-radius:999px;font-weight:700;}
.start-hint {margin-top:9px;color:#7A8880;font-size:.76rem;line-height:1.55;}
.quick-preview {background:#173E2E;border-radius:22px;padding:23px;color:white;box-shadow:0 22px 48px rgba(23,62,46,.13);}
.quick-preview .preview-label {font-size:.67rem;letter-spacing:.1em;font-weight:850;color:#A9C5B6;text-transform:uppercase;}
.quick-preview h3 {color:white;font-size:1.14rem;margin:8px 0 17px;letter-spacing:-.035em;}
.quick-preview-grid {display:grid;grid-template-columns:1fr 1fr;gap:9px;}
.quick-preview-card {border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);border-radius:13px;padding:14px;}
.quick-preview-card small {display:block;color:#B8CDC1;font-size:.65rem;margin-bottom:5px;}
.quick-preview-card strong {display:block;color:#FFF;font-size:1.28rem;letter-spacing:-.04em;}
.quick-preview-flow {margin-top:13px;border-top:1px solid rgba(255,255,255,.12);padding-top:12px;display:grid;gap:8px;}
.quick-preview-flow div {display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:.75rem;color:#E8F0EB;}
.quick-preview-flow span {font-size:.62rem;color:#AFC6B9;background:rgba(255,255,255,.07);border-radius:999px;padding:4px 7px;}
.mobile-mini-result {display:none;background:#EDF4EF;border:1px solid #D8E4DC;border-radius:14px;padding:14px 15px;margin-top:13px;color:#456052;font-size:.78rem;line-height:1.55;}
.mobile-mini-result b {color:var(--forest);}
.quick-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin-top:.3rem;}
.quick-step {background:#FFF;border:1px solid var(--line);border-radius:15px;padding:18px;min-height:116px;}
.quick-step .n {display:inline-flex;width:25px;height:25px;align-items:center;justify-content:center;border-radius:8px;background:var(--forest-soft);color:var(--forest);font-size:.67rem;font-weight:850;margin-bottom:11px;}
.quick-step b {display:block;font-size:.93rem;color:var(--ink);margin-bottom:6px;}
.quick-step span {display:block;font-size:.8rem;color:var(--muted);line-height:1.58;}
.value-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:11px;}
.value-item {background:#FFF;border:1px solid var(--line);border-radius:15px;padding:18px;}
.value-item b {display:block;font-size:.93rem;color:var(--ink);margin-bottom:6px;}
.value-item span {display:block;font-size:.8rem;color:var(--muted);line-height:1.58;}
.compact-trust {display:flex;align-items:center;justify-content:space-between;gap:16px;background:#F0F5F2;border:1px solid #DCE7E0;border-radius:14px;padding:15px 17px;margin-top:18px;}
.compact-trust b {font-size:.82rem;color:var(--forest);}
.compact-trust span {font-size:.74rem;color:#74827A;line-height:1.5;text-align:right;}
[data-testid="stExpander"] {border-color:var(--line)!important;border-radius:13px!important;background:#FFF;}
.analysis-stepbar {display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:4px 0 18px;}
.analysis-stepbar div {background:#EFF4F1;border:1px solid #DCE5DF;border-radius:11px;padding:10px 12px;color:#64736A;font-size:.72rem;font-weight:700;}
.analysis-stepbar div.active {background:var(--forest);border-color:var(--forest);color:white;}
.measure-grid-label {font-size:.77rem;color:#6B786F;margin:.15rem 0 .65rem;}
.result-hero {display:flex;align-items:center;justify-content:space-between;gap:20px;background:#173E2E;border-radius:16px;padding:18px 20px;margin:.2rem 0 .9rem;color:white;}
.result-hero small {display:block;font-size:.65rem;color:#AFC6B9;margin-bottom:4px;letter-spacing:.06em;text-transform:uppercase;font-weight:800;}
.result-hero strong {font-size:1.15rem;color:#FFF;letter-spacing:-.035em;}
.result-hero .candidate {text-align:right;}
.result-hero .candidate span {display:block;color:#C8DAD0;font-size:.72rem;margin-top:3px;}

@media (max-width: 900px) {
    .home-hero-copy {padding:10px 0 5px;}
    .quick-steps,.value-grid {grid-template-columns:1fr 1fr;}
}
@media (max-width: 640px) {
    .app-brandbar {margin:.05rem 0 .75rem;}
    .app-team {display:none;}
    .home-hero-copy {padding:5px 0 0;}
    .home-hero-copy h1 {font-size:2.3rem;line-height:1.06;margin:8px 0 13px;}
    .home-hero-copy p {font-size:.9rem;line-height:1.65;margin-bottom:14px;}
    .home-points {gap:6px;margin-bottom:13px;}
    .home-point {font-size:.67rem;padding:6px 8px;}
    .quick-preview {display:none;}
    .mobile-mini-result {display:block;}
    .quick-steps,.value-grid {grid-template-columns:1fr;gap:8px;}
    .quick-step,.value-item {padding:15px;min-height:0;}
    .section-head {margin:2rem 0 .8rem;}
    .section-head h2 {font-size:1.25rem;line-height:1.35;}
    .section-head p {font-size:.84rem;line-height:1.62;}
    .compact-trust {display:block;}
    .compact-trust span {display:block;text-align:left;margin-top:6px;}
    .analysis-stepbar {grid-template-columns:1fr 1fr 1fr;gap:5px;}
    .analysis-stepbar div {padding:8px 6px;font-size:.65rem;text-align:center;}
    .result-hero {display:block;padding:16px;}
    .result-hero .candidate {text-align:left;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,.12);}
    .footer-brand {margin-top:2rem;}
}


/* V9.7 · QR-first mobile web app */
.launch-wrap {max-width:860px;margin:0 auto;padding:5.2vh 0 1.2rem;text-align:center;}
.launch-logo {display:inline-flex;align-items:center;gap:9px;margin-bottom:1.35rem;color:#2A3A31;font-size:.82rem;font-weight:800;}
.launch-logo .mark {display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:11px;background:var(--forest);color:#fff;font-size:.72rem;font-weight:900;box-shadow:0 7px 20px rgba(31,90,66,.18);}
.launch-kicker {font-size:.7rem;letter-spacing:.12em;color:var(--forest);font-weight:850;margin-bottom:.8rem;}
.launch-wrap h1 {font-size:clamp(2.45rem,6vw,4.25rem);line-height:1.04;letter-spacing:-.075em;margin:0 auto 1rem;max-width:820px;font-weight:880;}
.launch-wrap .lead {max-width:690px;margin:0 auto;color:#5A695F;font-size:1rem;line-height:1.78;word-break:keep-all;}
.launch-meta {margin:.9rem 0 1.35rem;color:#7B8880;font-size:.76rem;font-weight:650;}
.launch-cta {max-width:520px;margin:0 auto;}
.launch-trust {max-width:660px;margin:1rem auto 0;padding:11px 14px;border-radius:12px;background:#EEF4F0;border:1px solid #DCE7E0;color:#617168;font-size:.76rem;line-height:1.55;}
.app-flow {max-width:760px;margin:2.5rem auto 0;display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center;text-align:left;}
.app-flow-step {display:flex;align-items:center;gap:10px;padding:13px 14px;background:#FFF;border:1px solid var(--line);border-radius:14px;}
.app-flow-n {display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:9px;background:var(--forest-soft);color:var(--forest);font-size:.67rem;font-weight:850;flex:none;}
.app-flow-step b {display:block;font-size:.84rem;color:var(--ink);margin-bottom:2px;}
.app-flow-step span {display:block;font-size:.69rem;color:#7A8880;line-height:1.4;}
.app-flow-arrow {color:#9EAAA3;font-size:1rem;text-align:center;}
.home-support {max-width:860px;margin:2.5rem auto 0;padding-top:2rem;border-top:1px solid var(--line);}
.home-support h3 {font-size:1.15rem;margin:0 0 .9rem;}
.home-support-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.home-support-item {padding:15px 16px;background:#FFF;border:1px solid var(--line);border-radius:14px;text-align:left;}
.home-support-item b {display:block;font-size:.86rem;color:var(--ink);margin-bottom:5px;}
.home-support-item span {display:block;font-size:.76rem;color:#6C7A71;line-height:1.55;}

/* Force primary CTAs to look active and app-like */
div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="baseButton-primary"],
div[data-testid="stFormSubmitButton"] > button {
    background:var(--forest)!important;
    border-color:var(--forest)!important;
    color:#FFF!important;
    min-height:52px!important;
    border-radius:13px!important;
    font-size:.95rem!important;
    font-weight:820!important;
    box-shadow:0 9px 24px rgba(31,90,66,.14)!important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background:var(--forest-strong)!important;
    border-color:var(--forest-strong)!important;
    color:#FFF!important;
}

.flow-shell {max-width:760px;margin:0 auto;}
.flow-top {display:flex;align-items:center;justify-content:space-between;gap:12px;margin:.35rem 0 1.1rem;}
.flow-brand {display:flex;align-items:center;gap:8px;color:#34463B;font-size:.82rem;font-weight:800;}
.flow-brand .dot {width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;background:var(--forest);color:white;border-radius:9px;font-size:.62rem;font-weight:900;}
.flow-count {font-size:.72rem;color:#6F7D74;background:#F1F5F2;border:1px solid #DDE6E0;border-radius:999px;padding:6px 9px;font-weight:760;}
.flow-title {margin:0 0 .45rem;font-size:clamp(1.7rem,4vw,2.25rem);line-height:1.18;letter-spacing:-.055em;}
.flow-copy {margin:0 0 1.15rem;color:#647269;font-size:.88rem;line-height:1.68;}
.flow-progress {height:6px;background:#E4EAE6;border-radius:999px;overflow:hidden;margin:0 0 1.65rem;}
.flow-progress span {display:block;height:100%;background:var(--forest);border-radius:999px;}
.flow-card {background:#FFF;border:1px solid var(--line);border-radius:18px;padding:22px;margin-bottom:12px;}
.flow-tip {background:#EFF5F1;border:1px solid #DDE8E1;border-radius:13px;padding:13px 14px;color:#5D6E64;font-size:.78rem;line-height:1.55;margin:.75rem 0 1rem;}
.flow-tip b {color:var(--forest);}
.photo-guide {display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0 0 1rem;}
.photo-guide div {background:#F5F7F6;border:1px solid var(--line);border-radius:11px;padding:10px;text-align:center;color:#6E7B73;font-size:.7rem;line-height:1.4;}
.result-app-head {display:grid;grid-template-columns:1.1fr .9fr;gap:12px;margin:.3rem 0 1rem;}
.result-app-main {background:#173E2E;color:#FFF;border-radius:18px;padding:20px 21px;}
.result-app-main small,.result-app-side small {display:block;font-size:.66rem;letter-spacing:.04em;color:#AFC6B9;margin-bottom:6px;font-weight:760;}
.result-app-main strong {display:block;font-size:1.45rem;letter-spacing:-.045em;color:#FFF;}
.result-app-main span {display:block;margin-top:7px;color:#C9D9D0;font-size:.76rem;line-height:1.55;}
.result-app-side {background:#FFF;border:1px solid var(--line);border-radius:18px;padding:20px 21px;}
.result-app-side small {color:#7B8880;}
.result-app-side strong {display:block;font-size:1.25rem;letter-spacing:-.04em;color:var(--forest);}
.result-app-side span {display:block;margin-top:6px;color:#7A8780;font-size:.74rem;line-height:1.5;}

@media (max-width:640px) {
    .launch-wrap {padding:2.5vh .1rem .5rem;text-align:left;}
    .launch-logo {margin-bottom:1.1rem;}
    .launch-kicker {margin-bottom:.55rem;}
    .launch-wrap h1 {font-size:2.55rem;line-height:1.04;margin-bottom:.8rem;}
    .launch-wrap .lead {font-size:.91rem;line-height:1.65;}
    .launch-meta {margin:.72rem 0 1rem;font-size:.72rem;}
    .launch-cta {max-width:none;}
    .launch-trust {font-size:.71rem;margin-top:.75rem;}
    .app-flow {grid-template-columns:1fr;margin-top:1.5rem;gap:7px;}
    .app-flow-arrow {display:none;}
    .app-flow-step {padding:11px 12px;}
    .home-support {margin-top:1.6rem;padding-top:1.4rem;}
    .home-support-grid {grid-template-columns:1fr;gap:7px;}
    .flow-shell {max-width:none;}
    .flow-top {margin-top:.05rem;}
    .flow-title {font-size:1.8rem;}
    .flow-card {padding:17px;border-radius:15px;}
    .photo-guide {grid-template-columns:1fr 1fr 1fr;gap:5px;}
    .photo-guide div {padding:8px 5px;font-size:.64rem;}
    .result-app-head {grid-template-columns:1fr;}
}


/* V9.8 · clean responsive product UI */
.stApp {background:#F8FAF9;}
.block-container {max-width:1080px;padding-top:1.2rem;}

.v98-home {max-width:820px;margin:0 auto;padding:8vh 0 1rem;text-align:center;}
.v98-brand {display:inline-flex;align-items:center;gap:9px;margin-bottom:1.15rem;color:#43544A;font-size:.8rem;font-weight:780;}
.v98-brand-mark {display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;border-radius:11px;background:var(--forest);color:#fff;font-size:.7rem;font-weight:900;box-shadow:0 8px 22px rgba(31,90,66,.16);}
.v98-home h1 {margin:0 auto .95rem;max-width:780px;font-size:clamp(2.55rem,5.7vw,4.2rem);line-height:1.04;letter-spacing:-.075em;font-weight:880;}
.v98-home .lead {max-width:650px;margin:0 auto;color:#5C6B62;font-size:1rem;line-height:1.72;word-break:keep-all;}
.v98-home .meta {display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin:1.05rem 0 0;}
.v98-home .meta span {font-size:.72rem;font-weight:700;color:#637169;background:#EFF4F1;border:1px solid #DDE6E0;border-radius:999px;padding:6px 9px;}
.v98-cta-wrap {max-width:420px;margin:.8rem auto 0;}
.v98-note {max-width:590px;margin:.8rem auto 0;color:#7A8780;font-size:.73rem;line-height:1.55;text-align:center;}
.v98-steps {max-width:760px;margin:2.2rem auto 0;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.v98-step {background:#FFF;border:1px solid var(--line);border-radius:14px;padding:14px 15px;text-align:left;}
.v98-step .num {display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:8px;background:var(--forest-soft);color:var(--forest);font-size:.65rem;font-weight:850;margin-bottom:9px;}
.v98-step b {display:block;font-size:.87rem;color:var(--ink);margin-bottom:4px;}
.v98-step span {display:block;color:#78857D;font-size:.72rem;line-height:1.48;}
.v98-detail {max-width:760px;margin:1rem auto 0;}

/* analysis form / mobile tap targets */
[data-testid="stNumberInput"] input {min-height:46px;border-radius:10px;}
[data-testid="stCheckbox"] label {padding:.15rem 0;}
[data-testid="stCameraInput"], [data-testid="stFileUploader"] {border-radius:16px;}
.flow-shell {max-width:720px;}
.flow-top {margin-top:.15rem;}
.flow-card {box-shadow:0 8px 28px rgba(28,62,45,.035);}
.flow-title {font-size:clamp(1.8rem,4vw,2.35rem);}

/* result hierarchy */
.v98-summary {background:#FFF;border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:.5rem 0 1rem;}
.v98-summary h3 {margin:0 0 7px;font-size:1rem;}
.v98-summary p {margin:0;color:#5D6B63;font-size:.86rem;line-height:1.7;}
.v98-confidence-grid {display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:.75rem 0;}
.v98-confidence-item {background:#F5F7F6;border:1px solid var(--line);border-radius:12px;padding:12px 13px;}
.v98-confidence-item small {display:block;color:#7A8780;font-size:.68rem;margin-bottom:4px;}
.v98-confidence-item strong {font-size:1rem;color:var(--ink);}
.stTabs [data-baseweb="tab-list"] {position:relative;}

@media (max-width:640px) {
    .block-container {padding-top:.55rem;padding-left:.85rem;padding-right:.85rem;padding-bottom:2.2rem;}
    .v98-home {padding:4.2vh 0 .4rem;text-align:left;}
    .v98-brand {margin-bottom:.95rem;}
    .v98-home h1 {font-size:2.45rem;line-height:1.03;margin-bottom:.8rem;}
    .v98-home .lead {font-size:.9rem;line-height:1.62;}
    .v98-home .meta {justify-content:flex-start;margin-top:.85rem;gap:6px;}
    .v98-home .meta span {font-size:.66rem;padding:5px 8px;}
    .v98-cta-wrap {max-width:none;margin-top:.75rem;}
    .v98-note {text-align:left;margin-top:.65rem;font-size:.69rem;}
    .v98-steps {grid-template-columns:1fr;margin-top:1.55rem;gap:7px;}
    .v98-step {display:grid;grid-template-columns:32px 1fr;padding:12px 13px;align-items:start;column-gap:8px;}
    .v98-step .num {grid-row:1 / span 2;margin:0;}
    .v98-step b {margin:1px 0 2px;}
    .v98-step span {grid-column:2;}
    .v98-detail {margin-top:.75rem;}
    .flow-title {font-size:1.72rem;line-height:1.16;}
    .flow-copy {font-size:.84rem;}
    .flow-card {padding:15px;border-radius:14px;}
    .result-strip {grid-template-columns:1fr 1fr;}
    .result-card {padding:12px;}
    .result-card small {font-size:.65rem;}
    .result-card strong {font-size:1.15rem;}
    .stTabs [data-baseweb="tab-list"] {gap:2px;}
    .stTabs [data-baseweb="tab"] {font-size:.74rem;padding:0 9px;}
    .footer-brand {display:none;}
}


/* V9.9 · mobile-app first / desktop service layout */
.v99-hero {
    max-width:1000px;
    margin:0 auto;
    padding:6.4vh 0 1rem;
    display:grid;
    grid-template-columns:minmax(0,1.05fr) minmax(320px,.95fr);
    gap:54px;
    align-items:center;
}
.v99-copy {min-width:0;}
.v99-brand {
    display:inline-flex;
    align-items:center;
    gap:9px;
    margin-bottom:1.35rem;
    color:#405047;
    font-size:.8rem;
    font-weight:800;
}
.v99-mark {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:34px;
    height:34px;
    border-radius:11px;
    background:var(--forest);
    color:#FFF;
    font-size:.69rem;
    font-weight:900;
    box-shadow:0 8px 22px rgba(31,90,66,.15);
}
.v99-copy h1 {
    margin:0 0 1rem;
    max-width:610px;
    font-size:clamp(2.75rem,5vw,4.2rem);
    line-height:1.03;
    letter-spacing:-.072em;
    font-weight:880;
}
.v99-lead {
    max-width:590px;
    color:#5B6A61;
    font-size:1rem;
    line-height:1.75;
    word-break:keep-all;
}
.v99-meta {
    margin-top:1rem;
    color:#718078;
    font-size:.76rem;
    font-weight:700;
}
.v99-preview {
    background:#173E2E;
    color:#FFF;
    border-radius:24px;
    padding:24px;
    box-shadow:0 24px 58px rgba(23,62,46,.13);
}
.v99-preview-top {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    margin-bottom:18px;
}
.v99-preview-title {font-size:.9rem;font-weight:800;color:#FFF;}
.v99-preview-badge {
    padding:5px 8px;
    border-radius:999px;
    background:rgba(255,255,255,.08);
    color:#BCD0C5;
    font-size:.62rem;
    font-weight:800;
    letter-spacing:.04em;
}
.v99-preview-photo {
    height:126px;
    border-radius:16px;
    border:1px dashed rgba(255,255,255,.2);
    background:linear-gradient(145deg,rgba(255,255,255,.08),rgba(255,255,255,.035));
    display:flex;
    align-items:center;
    justify-content:center;
    color:#BFD1C7;
    font-size:.77rem;
    margin-bottom:12px;
}
.v99-preview-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:8px;
}
.v99-preview-item {
    border:1px solid rgba(255,255,255,.1);
    background:rgba(255,255,255,.055);
    border-radius:13px;
    padding:12px;
}
.v99-preview-item small {
    display:block;
    color:#AEC5B9;
    font-size:.62rem;
    margin-bottom:5px;
}
.v99-preview-item strong {
    display:block;
    color:#FFF;
    font-size:.88rem;
}
.st-key-home_start_v99 {
    max-width:360px;
    margin:0 auto;
}
.st-key-home_start_v99 button {
    width:100%!important;
    min-height:54px!important;
}
.v99-mini-flow {
    max-width:620px;
    margin:1rem auto 0;
    display:flex;
    justify-content:center;
    align-items:center;
    gap:8px;
    color:#738078;
    font-size:.72rem;
    font-weight:700;
    white-space:nowrap;
}
.v99-mini-flow .arrow {color:#A5B0AA;font-weight:500;}
.v99-home-detail {
    max-width:760px;
    margin:2rem auto 0;
    padding-top:1.25rem;
    border-top:1px solid var(--line);
}
.v99-result-note {
    margin:.8rem 0 1rem;
    padding:12px 14px;
    border-radius:12px;
    background:#F0F5F2;
    border:1px solid #DDE7E1;
    color:#617168;
    font-size:.76rem;
    line-height:1.55;
}
.v99-result-note b {color:var(--forest);}

@media (max-width:900px) {
    .v99-hero {
        grid-template-columns:1fr;
        max-width:760px;
        gap:22px;
        padding:3.4vh 0 .8rem;
    }
    .v99-preview {display:none;}
}

@media (max-width:640px) {
    .block-container {
        padding-top:.48rem;
        padding-left:.9rem;
        padding-right:.9rem;
        padding-bottom:2rem;
    }
    .v99-hero {
        display:block;
        padding:1.8vh 0 .35rem;
    }
    .v99-brand {
        margin-bottom:1rem;
        font-size:.78rem;
    }
    .v99-mark {
        width:32px;
        height:32px;
        border-radius:10px;
    }
    .v99-copy h1 {
        font-size:2.12rem;
        line-height:1.08;
        margin-bottom:.72rem;
        max-width:420px;
    }
    .v99-lead {
        font-size:.91rem;
        line-height:1.62;
        max-width:440px;
    }
    .v99-meta {
        margin-top:.72rem;
        font-size:.72rem;
    }
    .st-key-home_start_v99 {
        max-width:none;
        margin-top:.72rem;
    }
    .st-key-home_start_v99 button {
        min-height:55px!important;
        border-radius:14px!important;
        font-size:.96rem!important;
    }
    .v99-mini-flow {
        margin-top:.85rem;
        max-width:none;
        justify-content:flex-start;
        gap:6px;
        font-size:.68rem;
        overflow-x:auto;
        scrollbar-width:none;
    }
    .v99-mini-flow::-webkit-scrollbar {display:none;}
    .v99-home-detail {
        margin-top:1.3rem;
        padding-top:1rem;
    }
    .flow-title {
        font-size:1.65rem;
        line-height:1.18;
    }
    .flow-copy {
        font-size:.84rem;
        line-height:1.58;
    }
    .photo-guide {
        grid-template-columns:1fr 1fr 1fr;
    }
    [data-testid="stNumberInput"] input {
        min-height:49px;
        font-size:1rem;
    }
}


/* V9.10 · visual box/product comparison */
.detect-compare-head {
    display:flex;
    justify-content:space-between;
    align-items:flex-end;
    gap:14px;
    margin:1.25rem 0 .65rem;
}
.detect-compare-head h4 {
    margin:0;
    font-size:1rem;
    letter-spacing:-.035em;
}
.detect-compare-head span {
    color:#77847C;
    font-size:.72rem;
}
.detect-compare-labels {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
    margin:0 0 6px;
}
.detect-compare-labels div {
    color:#66746B;
    font-size:.7rem;
    font-weight:760;
}
.detect-compare-labels div:last-child {text-align:right;}
.occupancy-card {
    margin:.75rem 0 1.15rem;
    padding:14px 15px;
    background:#FFF;
    border:1px solid var(--line);
    border-radius:14px;
}
.occupancy-top {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    margin-bottom:9px;
    font-size:.78rem;
}
.occupancy-top b {color:var(--ink);}
.occupancy-top span {color:#718078;}
.occupancy-bar {
    display:flex;
    width:100%;
    height:12px;
    overflow:hidden;
    border-radius:999px;
    background:#E7ECE9;
}
.occupancy-product {
    height:100%;
    background:var(--forest);
}
.occupancy-empty {
    height:100%;
    background:#DCE4DF;
}
.occupancy-legend {
    display:flex;
    justify-content:space-between;
    gap:12px;
    margin-top:8px;
    color:#6E7B73;
    font-size:.7rem;
}
.detect-explain {
    margin-top:.45rem;
    color:#68766E;
    font-size:.76rem;
    line-height:1.58;
}

@media (max-width:640px) {
    .detect-compare-head {
        display:block;
        margin-top:1rem;
    }
    .detect-compare-head span {
        display:block;
        margin-top:4px;
    }
    .occupancy-card {
        padding:13px;
    }
}


/* V9.12 · instant photo analysis */
.quick-mode-badge {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 9px;
    border-radius:999px;
    background:#EDF4EF;
    border:1px solid #D9E5DD;
    color:var(--forest);
    font-size:.69rem;
    font-weight:820;
    margin-bottom:.8rem;
}
.quick-mode-badge.precision {
    background:#173E2E;
    border-color:#173E2E;
    color:#FFF;
}
.quick-result-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:10px;
    margin:.4rem 0 1rem;
}
.quick-result-card {
    background:#FFF;
    border:1px solid var(--line);
    border-radius:16px;
    padding:17px 18px;
}
.quick-result-card small {
    display:block;
    color:#7B8880;
    font-size:.69rem;
    margin-bottom:7px;
}
.quick-result-card strong {
    display:block;
    color:var(--ink);
    font-size:1.45rem;
    letter-spacing:-.045em;
}
.quick-result-card span {
    display:block;
    color:#87938C;
    font-size:.68rem;
    margin-top:4px;
    line-height:1.45;
}
.capture-feedback {
    margin:.8rem 0 1rem;
    padding:13px 14px;
    border-radius:13px;
    border:1px solid #DCE7E0;
    background:#F0F5F2;
    color:#5D6E64;
    font-size:.78rem;
    line-height:1.58;
}
.capture-feedback.warn {
    background:#FAF5E9;
    border-color:#E9DDBE;
    color:#76603A;
}
.capture-feedback.bad {
    background:#FAEEEC;
    border-color:#E7D1CD;
    color:#7F514A;
}
.precision-upgrade {
    margin:1rem 0 1.25rem;
    padding:18px 19px;
    border-radius:16px;
    background:#173E2E;
    color:#FFF;
}
.precision-upgrade h4 {
    margin:0 0 5px;
    color:#FFF;
    font-size:1rem;
}
.precision-upgrade p {
    margin:0;
    color:#C9D9D0;
    font-size:.78rem;
    line-height:1.6;
}
.precision-list {
    display:flex;
    flex-wrap:wrap;
    gap:6px;
    margin-top:11px;
}
.precision-list span {
    padding:5px 8px;
    border-radius:999px;
    background:rgba(255,255,255,.08);
    color:#DCE8E1;
    font-size:.66rem;
    font-weight:720;
}
.result-visual-shell {
    margin:.65rem 0 1rem;
}
.result-action-row {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:9px;
    margin:.6rem 0 1rem;
}
.details-intro {
    margin-top:1rem;
    padding-top:1rem;
    border-top:1px solid var(--line);
}
@media (max-width:640px) {
    .quick-result-grid {
        grid-template-columns:repeat(3,1fr);
        gap:7px;
    }
    .quick-result-card {
        padding:13px 11px;
        border-radius:14px;
    }
    .quick-result-card small {
        font-size:.62rem;
    }
    .quick-result-card strong {
        font-size:1.18rem;
    }
    .quick-result-card span {
        font-size:.61rem;
    }
    .precision-upgrade {
        padding:16px;
    }
    .result-action-row {
        grid-template-columns:1fr;
    }
}


/* ==========================================================
   V9.14 FINAL POLISH
   Clean service UI · white / deep green / neutral gray
   ========================================================== */
:root {
    --ink:#17231C;
    --ink-soft:#35453C;
    --muted:#69766E;
    --forest:#245B45;
    --forest-strong:#194735;
    --forest-soft:#EEF5F1;
    --paper:#F7F9F8;
    --white:#FFFFFF;
    --line:#E2E8E4;
    --line-strong:#D4DDD7;
    --amber:#A36D1F;
    --amber-soft:#FBF5E9;
    --red:#A85249;
    --red-soft:#FBF0EE;
}

.stApp {background:var(--paper);}
.block-container {
    max-width:1160px;
    padding-top:1.35rem;
    padding-bottom:3.5rem;
}

/* softer surfaces */
.v99-preview,
.result-app-main,
.precision-upgrade {
    box-shadow:0 18px 44px rgba(25,71,53,.10);
}
.flow-card,
.quick-result-card,
.v98-summary,
.occupancy-card,
.result-app-side,
[data-testid="stMetric"] {
    box-shadow:0 8px 24px rgba(23,55,40,.035);
}

/* Button hierarchy: primary = green, secondary = white */
div[data-testid="stButton"] > button[kind="secondary"],
button[data-testid="baseButton-secondary"],
div[data-testid="stDownloadButton"] > button {
    background:#FFF!important;
    color:var(--forest)!important;
    border:1px solid #CFDDD5!important;
    box-shadow:none!important;
    min-height:46px!important;
    border-radius:12px!important;
    font-weight:760!important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background:#F4F8F6!important;
    color:var(--forest-strong)!important;
    border-color:#BFD2C7!important;
    transform:none!important;
}
div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="baseButton-primary"],
div[data-testid="stFormSubmitButton"] > button {
    background:var(--forest)!important;
    border-color:var(--forest)!important;
    color:#FFF!important;
    min-height:52px!important;
    border-radius:13px!important;
    font-size:.94rem!important;
    font-weight:820!important;
    box-shadow:0 8px 18px rgba(36,91,69,.14)!important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background:var(--forest-strong)!important;
    border-color:var(--forest-strong)!important;
    transform:translateY(-1px);
}

/* inputs */
[data-testid="stNumberInput"] input {
    min-height:48px;
    border-radius:11px!important;
    border-color:var(--line-strong)!important;
    background:#FFF!important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#83A795!important;
    box-shadow:0 0 0 3px rgba(36,91,69,.08)!important;
}
[data-testid="stFileUploader"] {
    background:#FFF!important;
    border:1.5px dashed #B8CBC0!important;
    border-radius:17px!important;
}
[data-testid="stFileUploader"] section {
    padding:1.25rem 1rem!important;
}
[data-testid="stCameraInput"] {
    border-radius:17px!important;
    overflow:hidden;
}

/* radio as a lightweight segmented control */
[data-testid="stRadio"] [role="radiogroup"] {
    gap:6px;
}
[data-testid="stRadio"] label {
    border-radius:10px;
}

/* HOME */
.v99-hero {
    max-width:1040px;
    padding:5.4vh 0 .9rem;
    gap:62px;
}
.v99-brand {
    color:#506057;
    font-size:.78rem;
    margin-bottom:1.2rem;
}
.v99-mark {
    background:var(--forest);
    box-shadow:none;
}
.v99-copy h1 {
    color:var(--ink);
    font-size:clamp(2.8rem,4.7vw,4rem);
    line-height:1.045;
    max-width:590px;
}
.v99-lead {
    color:#5C6A62;
    font-size:.98rem;
    line-height:1.72;
}
.v99-meta {
    color:#7B8880;
    font-size:.73rem;
    font-weight:680;
}
.v99-preview {
    background:#193F30;
    border-radius:22px;
    padding:22px;
}
.v99-preview-photo {
    height:118px;
    border-color:rgba(255,255,255,.16);
    background:rgba(255,255,255,.045);
}
.v99-preview-item {
    background:rgba(255,255,255,.055);
    border-color:rgba(255,255,255,.09);
}
.st-key-home_start_v99 {
    max-width:340px;
}
.v99-mini-flow {
    color:#77847C;
    font-size:.7rem;
    margin-top:.85rem;
}
.v99-home-detail {
    margin-top:1.65rem;
}

/* FLOW */
.flow-shell {max-width:700px;}
.flow-top {margin-bottom:.85rem;}
.flow-progress {
    height:5px;
    background:#E5EBE7;
    margin-bottom:1.45rem;
}
.flow-title {
    color:var(--ink);
    font-size:clamp(1.75rem,3.5vw,2.2rem);
}
.flow-copy {
    color:#66746B;
    max-width:640px;
}
.flow-card {
    border-color:var(--line);
    border-radius:17px;
    padding:20px;
}
.photo-guide div {
    background:#F2F6F4;
    border-color:#E1E8E4;
    color:#69776F;
}

/* RESULT HEADER */
.section-head {
    margin:2.15rem 0 .85rem;
}
.section-head .eyebrow {
    font-size:.65rem;
    color:#6E8879;
}
.section-head h2 {
    font-size:1.35rem;
}
.section-head p {
    font-size:.84rem;
}

/* top three metrics */
.quick-result-grid {
    gap:10px;
    margin:.35rem 0 .85rem;
}
.quick-result-card {
    position:relative;
    overflow:hidden;
    padding:18px 19px;
    border:1px solid var(--line);
    border-radius:16px;
    background:#FFF;
}
.quick-result-card::before {
    content:"";
    position:absolute;
    top:0;
    left:0;
    right:0;
    height:3px;
    background:#DCE6E0;
}
.quick-result-card.product::before {background:#88A998;}
.quick-result-card.confidence::before {background:#AAB7AF;}
.quick-result-card.void.good {
    background:#F6FAF8;
    border-color:#D9E7DF;
}
.quick-result-card.void.good::before {background:#4D8167;}
.quick-result-card.void.warn {
    background:var(--amber-soft);
    border-color:#E9DCC1;
}
.quick-result-card.void.warn::before {background:#B27B2A;}
.quick-result-card.void.alert {
    background:var(--red-soft);
    border-color:#E9D3CF;
}
.quick-result-card.void.alert::before {background:#B35F55;}
.quick-result-card small {
    color:#78857D;
    font-size:.68rem;
    margin-bottom:7px;
}
.quick-result-card strong {
    color:var(--ink);
    font-size:1.52rem;
}
.quick-result-card.void.warn strong {color:#845B1E;}
.quick-result-card.void.alert strong {color:#934A42;}
.quick-result-card span {
    color:#87928C;
    font-size:.67rem;
}

/* capture guidance */
.capture-feedback {
    margin:.65rem 0 .9rem;
    padding:12px 14px;
    border-radius:12px;
    box-shadow:none;
}

/* visual comparison */
.result-visual-shell {
    background:#FFF;
    border:1px solid var(--line);
    border-radius:17px;
    padding:17px;
    margin:.75rem 0 1rem;
    box-shadow:0 8px 24px rgba(23,55,40,.03);
}
.detect-compare-head {
    margin:.1rem 0 .65rem;
}
.detect-compare-head h4 {
    font-size:.96rem;
}
.detect-compare-labels {
    margin-bottom:7px;
}
.occupancy-card {
    margin:.7rem 0 0;
    background:#F8FAF9;
    box-shadow:none;
}
.occupancy-bar {
    height:10px;
}
.occupancy-product {background:var(--forest);}
.occupancy-empty {background:#DCE4DF;}

/* precision upgrade */
.precision-upgrade {
    background:#193F30;
    border-radius:17px;
    padding:18px 19px;
    margin:.9rem 0 .65rem;
    box-shadow:none;
}
.precision-upgrade h4 {font-size:.98rem;}
.precision-upgrade p {font-size:.76rem;}
.precision-list {margin-top:9px;}
.precision-list span {
    padding:4px 7px;
    font-size:.63rem;
}

/* result status */
.status-line {
    border-color:var(--line);
    box-shadow:none;
    padding:13px 14px;
}
.v99-result-note {
    background:#F5F7F6;
    border-color:#E2E7E4;
    color:#738078;
    font-size:.72rem;
    padding:10px 12px;
}

/* detailed tabs */
.details-intro {
    margin-top:1.25rem;
}
.stTabs [data-baseweb="tab-list"] {
    background:#EEF2F0!important;
    border-radius:12px!important;
    padding:4px!important;
}
.stTabs [data-baseweb="tab"] {
    height:39px!important;
    border-radius:9px!important;
    font-size:.8rem!important;
    color:#66736B!important;
}
.stTabs [aria-selected="true"] {
    background:#FFF!important;
    color:var(--forest)!important;
    box-shadow:0 1px 4px rgba(29,57,43,.05)!important;
}

/* tables / expanders */
[data-testid="stDataFrame"] {
    border-color:var(--line)!important;
    border-radius:13px!important;
}
[data-testid="stExpander"] {
    border:1px solid var(--line)!important;
    border-radius:13px!important;
    background:#FFF!important;
}

/* sidebar */
[data-testid="stSidebar"] {
    background:#FBFCFB;
    border-right:1px solid var(--line);
}
.version-pill {
    background:#F1F5F3;
    border-color:#E0E7E3;
}

/* footer */
.footer-brand {
    border-top-color:var(--line);
    color:#849089;
}

/* DESKTOP: keep content balanced */
@media (min-width:901px) {
    .quick-result-grid {
        grid-template-columns:repeat(3,minmax(0,1fr));
    }
    .result-visual-shell {
        padding:19px;
    }
}

/* MOBILE FINAL */
@media (max-width:640px) {
    .block-container {
        padding-top:.42rem;
        padding-left:.82rem;
        padding-right:.82rem;
        padding-bottom:1.9rem;
    }

    .v99-hero {
        padding:1.25vh 0 .2rem;
    }
    .v99-brand {
        margin-bottom:.82rem;
        font-size:.74rem;
    }
    .v99-mark {
        width:30px;
        height:30px;
        border-radius:9px;
    }
    .v99-copy h1 {
        font-size:2.02rem;
        line-height:1.09;
        margin-bottom:.65rem;
        letter-spacing:-.06em;
    }
    .v99-lead {
        font-size:.88rem;
        line-height:1.58;
    }
    .v99-meta {
        font-size:.68rem;
        margin-top:.6rem;
    }
    .st-key-home_start_v99 {
        margin-top:.58rem;
    }
    .st-key-home_start_v99 button {
        min-height:53px!important;
        border-radius:13px!important;
    }
    .v99-mini-flow {
        justify-content:flex-start;
        margin-top:.72rem;
        font-size:.64rem;
        gap:5px;
    }
    .v99-home-detail {
        margin-top:1.05rem;
        padding-top:.85rem;
    }

    .flow-top {
        margin-bottom:.7rem;
    }
    .flow-progress {
        margin-bottom:1.1rem;
    }
    .flow-title {
        font-size:1.58rem;
        line-height:1.18;
        margin-bottom:.4rem;
    }
    .flow-copy {
        font-size:.81rem;
        line-height:1.55;
        margin-bottom:.9rem;
    }
    .photo-guide {
        gap:5px;
        margin-bottom:.75rem;
    }
    .photo-guide div {
        padding:8px 5px;
        font-size:.61rem;
        line-height:1.35;
    }
    .flow-card {
        padding:14px;
        border-radius:14px;
    }
    [data-testid="stNumberInput"] input {
        min-height:47px;
        font-size:.96rem;
    }

    .section-head {
        margin:1.55rem 0 .65rem;
    }
    .section-head h2 {
        font-size:1.2rem;
    }
    .section-head p {
        font-size:.79rem;
        line-height:1.55;
    }

    /* 2 + 1 layout for readability */
    .quick-result-grid {
        grid-template-columns:1fr 1fr;
        gap:7px;
        margin-bottom:.7rem;
    }
    .quick-result-card {
        padding:13px 12px;
        border-radius:13px;
    }
    .quick-result-card.confidence {
        grid-column:1 / -1;
        display:grid;
        grid-template-columns:1fr auto;
        align-items:end;
        column-gap:10px;
    }
    .quick-result-card.confidence small,
    .quick-result-card.confidence span {
        grid-column:1;
    }
    .quick-result-card.confidence strong {
        grid-column:2;
        grid-row:1 / span 2;
        align-self:center;
    }
    .quick-result-card strong {
        font-size:1.2rem;
    }
    .quick-result-card small {
        font-size:.62rem;
    }
    .quick-result-card span {
        font-size:.61rem;
        line-height:1.38;
    }

    .capture-feedback {
        margin:.55rem 0 .7rem;
        padding:10px 11px;
        font-size:.72rem;
        line-height:1.5;
    }

    .result-visual-shell {
        padding:12px;
        border-radius:14px;
        margin:.62rem 0 .8rem;
    }
    .detect-compare-head {
        display:block;
        margin-bottom:.55rem;
    }
    .detect-compare-head h4 {
        font-size:.9rem;
    }
    .detect-compare-head span {
        margin-top:3px;
        font-size:.65rem;
    }
    .detect-compare-labels div {
        font-size:.62rem;
    }
    .occupancy-card {
        padding:11px;
        border-radius:12px;
    }
    .occupancy-top {
        font-size:.72rem;
    }
    .occupancy-legend,
    .detect-explain {
        font-size:.64rem;
    }

    .precision-upgrade {
        padding:14px;
        border-radius:14px;
        margin:.72rem 0 .55rem;
    }
    .precision-upgrade h4 {
        font-size:.91rem;
    }
    .precision-upgrade p {
        font-size:.7rem;
        line-height:1.5;
    }
    .precision-list span {
        font-size:.59rem;
    }

    .status-line {
        padding:11px 12px;
        margin:7px 0;
    }
    .status-line b {font-size:.8rem;}
    .status-line span {font-size:.72rem;}
    .v99-result-note {
        font-size:.66rem;
        line-height:1.48;
        padding:9px 10px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size:.7rem!important;
        padding:0 8px!important;
        height:37px!important;
    }

    /* make tap targets comfortable */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height:48px!important;
    }

    .footer-brand {display:none;}
}

</style>
""", unsafe_allow_html=True)

PAGES = ["홈", "분석", "혼합 포장", "지표", "신뢰도", "프로젝트"]
if "page" not in st.session_state:
    st.session_state["page"] = "홈"


def go_to_analysis() -> None:
    """Navigate safely before the sidebar radio is recreated."""
    st.session_state["analysis_stage"] = 1
    st.session_state["measurement_mode"] = "auto"
    st.session_state.pop("analysis_image_bytes", None)
    st.session_state["page"] = "분석"


def go_to_home() -> None:
    """Navigate safely before the sidebar radio is recreated."""
    st.session_state["analysis_stage"] = 1
    st.session_state["measurement_mode"] = "auto"
    st.session_state.pop("analysis_image_bytes", None)
    st.session_state["page"] = "홈"


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=138)
    st.markdown("<div class='sidebar-brand'><div class='name'>AI Packaging Intelligence</div><div class='copy'>사진으로 포장 상태를 먼저 확인하고, 필요하면 실제 치수를 입력해 박스 후보까지 비교할 수 있어요.</div></div>", unsafe_allow_html=True)
    st.markdown("### 메뉴")
    page = st.radio("메뉴", PAGES, key="page", label_visibility="collapsed")
    presentation_mode = st.toggle("발표 모드", value=False, help="분석 화면에서 핵심 지표만 간단히 보여줍니다.") if page == "분석" else False
    st.divider()
    st.caption("Prototype · Decision Support")
    st.markdown("<span class='version-pill'>Better Life For Us · Prototype</span>", unsafe_allow_html=True)


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



def build_detection_comparison(
    original: Image.Image,
    detected_bgr: np.ndarray,
    max_panel_width: int = 720,
) -> Image.Image:
    """Create a single responsive image comparing the original and AI-detected box interior."""
    original_rgb = np.array(original.convert("RGB"))
    detected_rgb = cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)

    def fit_panel(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = arr.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, target_w, 3), 247, dtype=np.uint8)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    panel_w = min(max_panel_width, 720)
    panel_h = int(panel_w * 0.72)
    left = fit_panel(original_rgb, panel_w, panel_h)
    right = fit_panel(detected_rgb, panel_w, panel_h)

    gap = 18
    canvas = np.full((panel_h, panel_w * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[:, :panel_w] = left
    canvas[:, panel_w + gap:] = right
    canvas[:, panel_w:panel_w + gap] = 238

    return Image.fromarray(canvas)



def infer_box_shape_candidates(box: dict, limit: int = 3) -> list[dict]:
    """
    사진에서 검출한 박스의 가로·세로 비율과 DB 규격의 바닥면 비율만 비교합니다.
    절대 크기(mm)를 알아내는 기능이 아니며, '형상이 비슷한 후보'를 보여주기 위한 참고 기능입니다.
    """
    w = float(box.get("warped_width", 0) or 0)
    h = float(box.get("warped_height", 0) or 0)
    if min(w, h) <= 0:
        return []

    detected_ratio = max(w, h) / min(w, h)
    rows = []
    for b in BOX_DATABASE:
        box_ratio = max(b.length, b.width) / max(min(b.length, b.width), 1)
        diff = abs(detected_ratio - box_ratio)
        similarity = max(0.0, 100.0 - diff / max(detected_ratio, 0.001) * 100.0)
        rows.append({
            "box": b,
            "detected_ratio": detected_ratio,
            "box_ratio": box_ratio,
            "similarity": round(similarity, 1),
        })
    rows.sort(key=lambda x: x["similarity"], reverse=True)
    return rows[:max(1, limit)]


def input_image() -> Image.Image | None:
    section_head("STEP 2", "포장 사진을 올려주세요.", "박스 네 모서리와 제품 전체가 보이게 찍으면 더 안정적으로 분석할 수 있습니다.")
    source = st.radio("이미지 입력 방식", ["사진 선택", "카메라로 촬영"], horizontal=True, label_visibility="collapsed")
    if source == "사진 선택":
        uploaded = st.file_uploader("사진을 선택하거나 이곳에 끌어다 놓으세요", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("박스 전체가 보이도록 촬영해주세요")
    return Image.open(uploaded).convert("RGB") if uploaded else None

if page == "홈":
    st.markdown("""<div class="v99-hero">
        <div class="v99-copy">
            <div class="v99-brand"><span class="v99-mark">AI</span><span>AI 포장 분석 · Better Life For Us</span></div>
            <h1>이 포장,<br>잘 맞는 걸까요?</h1>
            <div class="v99-lead">포장 사진 한 장만 올리면 빈 공간을 먼저 확인할 수 있어요. 박스 크기를 알고 있다면 치수를 더해 더 자세히 분석할 수도 있습니다.</div>
            <div class="v99-meta">약 1분 · 설치 없이 바로 사용 · 모바일·PC 지원</div>
        </div>
        <div class="v99-preview">
            <div class="v99-preview-top"><span class="v99-preview-title">이런 결과를 확인할 수 있어요</span><span class="v99-preview-badge">예시 화면</span></div>
            <div class="v99-preview-photo">포장 사진을 올리면 바로 분석해요</div>
            <div class="v99-preview-grid">
                <div class="v99-preview-item"><small>빈 공간</small><strong>자동 분석</strong></div>
                <div class="v99-preview-item"><small>박스 후보</small><strong>규격 비교</strong></div>
                <div class="v99-preview-item"><small>신뢰도</small><strong>함께 표시</strong></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.button(
        "사진으로 분석 시작하기",
        key="home_start_v99",
        type="primary",
        use_container_width=True,
        on_click=go_to_analysis,
    )

    st.markdown("""<div class="v99-mini-flow">
        <span>01 사진 올리기</span><span class="arrow">→</span>
        <span>02 바로 결과 확인</span><span class="arrow">→</span>
        <span>03 필요하면 자세히 분석</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="v99-home-detail">', unsafe_allow_html=True)
    with st.expander("이 서비스는 무엇을 하나요?"):
        st.markdown("""
사진에서 박스와 제품이 차지하는 공간을 분석하고, 입력한 박스 크기를 바탕으로 **현재 빈 공간과 비교 가능한 박스 후보**를 보여줍니다.

결과는 박스를 대신 결정하는 정답이 아니라, 포장 선택을 비교할 때 참고할 수 있는 정보입니다.
""")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "지표":
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
        ["60~80%", "높음", "박스 축소, 완충 구조 개선, 묶음 배치 개선이 필요할 가능성이 큽니다."],
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
        ["90~100점", "A+ 매우 우수", "효율과 보호의 균형이 매우 좋은 포장"],
        ["80~89점", "A 우수", "대체로 효율적이며 작은 보완만 필요한 포장"],
        ["70~79점", "B 양호", "사용 가능하지만 개선 여지가 있는 포장"],
        ["50~69점", "C 개선 필요", "빈 공간 또는 박스 규격을 우선 검토할 포장"],
        ["0~49점", "D 개선 우선", "박스 축소와 구조 개선을 우선 검토할 포장"],
    ], columns=["점수", "등급", "의미"]), hide_index=True, use_container_width=True)

    st.subheader("4. 결과를 올바르게 해석하는 순서")
    st.markdown("""
1. **촬영 품질과 신뢰도**를 먼저 확인합니다.  
2. **면적 빈 공간**과 **체적 기준 빈 공간**을 함께 봅니다.  
3. 제품이 깨지기 쉬운지, 완충재가 필요한지 등 **보호 조건**을 확인합니다.  
4. 추천 박스의 규격·비용·탄소 참고값을 비교합니다.  
5. 실제 샘플 포장과 낙하·압축 시험을 거쳐 최종 결정합니다.
""")
    st.warning("PVI는 공식 법정 시험, 환경성 인증 또는 제품 안전 인증이 아닙니다. 프로젝트 비교·교육·개선 의사결정을 위한 내부 지표입니다.")

elif page == "프로젝트":
    st.title("Better Life For Us 프로젝트 소개")
    st.markdown("""<div class="analysis-intro"><div><div class="kicker">BETTER LIFE FOR US</div><h1>포장 선택에 기준을 더합니다.</h1><p>경험에 의존하던 박스 선택을 사진과 데이터로 확인하고, 더 적합한 후보를 비교할 수 있도록 만든 프로젝트입니다.</p></div><div class="analysis-note">GREEN VALUE YOUTH 2026</div></div>""", unsafe_allow_html=True)

    st.subheader("프로젝트 배경")
    st.write("온라인 소비와 택배 이용이 증가하면서 제품 크기와 맞지 않는 큰 박스와 내부 빈 공간이 일상적인 환경 문제로 나타나고 있습니다. 과도한 포장은 포장재와 완충재 사용을 늘릴 뿐 아니라 물류 적재 효율을 낮추고, 탄소배출과 폐기물 증가에도 영향을 줄 수 있습니다.")

    st.subheader("우리가 만들고 있는 해결 방법")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown('<div class="easy-card"><b>01 · 제품·박스 인식</b><small>사진에서 박스 경계와 제품 영역을 찾아 분석 가능한 데이터로 바꿉니다.</small></div>', unsafe_allow_html=True)
    p2.markdown('<div class="easy-card"><b>02 · 빈 공간 계산</b><small>박스와 제품의 면적·체적 관계를 계산해 포장 공간이 얼마나 비어 있는지 보여줍니다.</small></div>', unsafe_allow_html=True)
    p3.markdown('<div class="easy-card"><b>03 · PVI 분석</b><small>공간 효율, 빈 공간 관리, 보호성, 지속가능성을 점수와 등급으로 설명합니다.</small></div>', unsafe_allow_html=True)
    p4.markdown('<div class="easy-card"><b>04 · 개선안 제안</b><small>더 적합한 박스 후보와 예상 재료·비용·탄소 절감 방향을 비교합니다.</small></div>', unsafe_allow_html=True)

    st.subheader("프로젝트가 추구하는 변화")
    st.markdown("""
- 소비자가 택배 속 빈 공간을 직관적으로 이해하도록 돕습니다.
- 우체국·소규모 판매자·물류 작업자가 포장 규격을 데이터로 비교하도록 지원합니다.
- 적정포장, 다회용·재사용 구조, 자원순환에 대한 사회적 관심을 높입니다.
- 분석 결과를 전시·캠페인·교육 콘텐츠와 연결해 친환경 소비 행동을 확산합니다.
""")

    st.subheader("프로젝트의 차별점")
    st.markdown("""<div class="info-panel"><p><b>① 감상이 아닌 데이터</b><br>“커 보인다”는 주관적 판단을 빈 공간과 점수로 시각화합니다.</p><p><b>② 진단에서 끝나지 않는 개선</b><br>문제 표시뿐 아니라 표준 박스 후보, 절감 가능성, 촬영 신뢰도와 한계까지 함께 제공합니다.</p><p><b>③ 기술과 시민 참여의 연결</b><br>AI 분석 결과를 캠페인, 교육, 다회용 박스 실험과 연결해 생활 속 자원순환 행동으로 확장합니다.</p></div>""", unsafe_allow_html=True)

    st.subheader("팀과 프로그램")
    st.markdown("""**Better Life For Us**는 LG생활건강 그린밸류 YOUTH 2026에서 자원순환과 소비습관을 주제로 활동하는 팀입니다. 본 플랫폼은 프로젝트의 분석 도구이자, 시민·청년·현장 작업자가 지속가능한 포장을 쉽게 이해하고 함께 개선 방향을 논의하기 위한 교육·소통 도구입니다.""")

    st.info("프로젝트 문구: 데이터로 포장을 이해하고, 더 나은 선택으로 바꾸는 지속가능한 포장 솔루션")

elif page == "신뢰도":
    st.markdown("""<div class="analysis-intro"><div><div class="kicker">TRANSPARENT & RESPONSIBLE AI</div><h1>결과보다 먼저, 신뢰도를 확인합니다.</h1><p>측정값과 추정값을 구분하고 이번 분석이 놓칠 수 있는 조건과 사용 범위를 함께 안내합니다.</p></div><div class="analysis-note">MEASURED · ESTIMATED · LIMITATIONS</div></div>""", unsafe_allow_html=True)
    st.subheader("이 시스템이 할 수 있는 것")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="easy-card"><b>이미지 기반 측정</b><small>박스와 제품의 보이는 영역을 분석해 공간 활용률과 빈 공간 비율을 계산합니다.</small></div>', unsafe_allow_html=True)
    c2.markdown('<div class="easy-card"><b>후보 박스 비교</b><small>등록된 규격 중 제품 체적과 목표 활용률에 맞는 후보를 비교합니다.</small></div>', unsafe_allow_html=True)
    c3.markdown('<div class="easy-card"><b>근거 공개</b><small>계산식, 검출 신뢰도, 적용된 가정과 이번 분석의 제한사항을 보여줍니다.</small></div>', unsafe_allow_html=True)
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
    if "analysis_stage" not in st.session_state:
        st.session_state["analysis_stage"] = 1
    stage = int(st.session_state.get("analysis_stage", 1))

    st.markdown(f"""<div class="flow-shell">
        <div class="flow-top"><div class="flow-brand"><span class="dot">AI</span><span>포장 분석</span></div><div class="flow-count">{stage} / 3</div></div>
        <div class="flow-progress"><span style="width:{stage/3*100:.0f}%"></span></div>
    </div>""", unsafe_allow_html=True)

    if stage == 1:
        st.markdown("""<div class="flow-shell"><h1 class="flow-title">포장 사진을 올려주세요.</h1><p class="flow-copy">박스와 제품이 한 화면에 보이도록 찍어주세요. 사진을 올리면 바로 빈 공간을 분석합니다.</p>
        <div class="photo-guide"><div>박스 전체가 보이도록</div><div>가능하면 위에서 촬영</div><div>강한 그림자는 피하기</div></div></div>""", unsafe_allow_html=True)

        source = st.radio(
            "사진 입력 방식",
            ["카메라로 촬영", "사진에서 선택"],
            horizontal=True,
            label_visibility="collapsed",
            key="flow_photo_source",
        )
        if source == "카메라로 촬영":
            uploaded = st.camera_input("포장 사진 촬영", key="flow_camera")
        else:
            uploaded = st.file_uploader("사진 선택", type=["jpg", "jpeg", "png"], key="flow_uploader")

        if uploaded is not None:
            # 사진이 들어오면 별도 선택 과정 없이 자동 분석으로 바로 이동합니다.
            current_bytes = uploaded.getvalue()
            if st.session_state.get("analysis_image_bytes") != current_bytes:
                st.session_state["analysis_image_bytes"] = current_bytes
                st.session_state["measurement_mode"] = "auto"
                st.session_state["flow_dimensions_confirmed"] = False
                st.session_state["analysis_stage"] = 3
                st.rerun()
        else:
            st.markdown("<div class='flow-tip'><b>사진 한 장이면 시작할 수 있어요.</b> 박스 크기를 몰라도 사진에서 제품이 차지하는 면적과 남는 공간의 비율을 먼저 확인할 수 있습니다.</div>", unsafe_allow_html=True)

        st.button(
            "홈으로 돌아가기",
            use_container_width=True,
            on_click=go_to_home,
        )
        st.stop()

    if stage == 2:
        st.markdown("""<div class="flow-shell">
            <div class="quick-mode-badge precision">자세히 분석</div>
            <h1 class="flow-title">박스 크기를 알고 있다면 입력해주세요.</h1>
            <p class="flow-copy">사진은 다시 올리지 않아도 됩니다. 박스 안쪽 치수와 제품 높이를 입력하면 체적 기준 빈 공간과 비교해볼 박스 후보까지 확인할 수 있어요.</p>
        </div>""", unsafe_allow_html=True)

        with st.form("dimension_form"):
            st.markdown("<div class='flow-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                box_width_mm = st.number_input("가로 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_width_mm", 330.0)), step=1.0)
                box_length_mm = st.number_input("세로 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_length_mm", 250.0)), step=1.0)
            with c2:
                box_height_mm = st.number_input("높이 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_height_mm", 150.0)), step=1.0)
                product_height_mm = st.number_input("제품 높이 · 대략 (mm)", min_value=0.0, value=float(st.session_state.get("flow_product_height_mm", 50.0)), step=1.0)

            dimensions_confirmed = st.checkbox(
                "박스 안쪽 치수를 직접 확인했어요.",
                value=bool(st.session_state.get("flow_dimensions_confirmed", True)),
            )
            st.markdown("<div class='flow-tip'><b>Tip</b> · 박스는 바깥쪽이 아니라 제품이 실제로 들어가는 안쪽을 기준으로 재주세요. 제품 높이는 대략 입력해도 됩니다.</div>", unsafe_allow_html=True)
            next_step = st.form_submit_button("자세히 분석 결과 보기", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if next_step:
            st.session_state["measurement_mode"] = "manual"
            st.session_state["flow_box_width_mm"] = box_width_mm
            st.session_state["flow_box_length_mm"] = box_length_mm
            st.session_state["flow_box_height_mm"] = box_height_mm
            st.session_state["flow_product_height_mm"] = product_height_mm
            st.session_state["flow_dimensions_confirmed"] = dimensions_confirmed
            st.session_state["analysis_stage"] = 3
            st.rerun()

        if st.button("사진 분석 결과로 돌아가기", use_container_width=True):
            st.session_state["measurement_mode"] = "auto"
            st.session_state["analysis_stage"] = 3
            st.rerun()
        st.stop()

    if "analysis_image_bytes" not in st.session_state:
        st.session_state["analysis_stage"] = 1
        st.rerun()

    measurement_mode = st.session_state.get("measurement_mode", "auto")
    has_exact_dimensions = measurement_mode == "manual"
    dimensions_confirmed = bool(st.session_state.get("flow_dimensions_confirmed", False)) if has_exact_dimensions else False
    target_utilization = float(st.session_state.get("flow_target_utilization", 0.75))
    image = Image.open(BytesIO(st.session_state["analysis_image_bytes"])).convert("RGB")

    if has_exact_dimensions:
        box_width_mm = float(st.session_state.get("flow_box_width_mm", 330.0))
        box_length_mm = float(st.session_state.get("flow_box_length_mm", 250.0))
        box_height_mm = float(st.session_state.get("flow_box_height_mm", 150.0))
        product_height_mm = float(st.session_state.get("flow_product_height_mm", 50.0))
    else:
        # 자동 모드에서의 값은 실제 mm가 아니라, 후속 로직 호환을 위한 내부 정규화 값입니다.
        # 사용자에게 물리 치수/체적으로 표시하지 않습니다.
        box_width_mm = box_length_mm = box_height_mm = product_height_mm = 0.0

    st.markdown("""<div class="flow-shell"><h1 class="flow-title">분석이 완료됐어요.</h1><p class="flow-copy">사진에서 확인한 핵심 결과부터 보여드릴게요.</p></div>""", unsafe_allow_html=True)
    if has_exact_dimensions:
        st.markdown('<div class="quick-mode-badge precision">자세히 분석 · 실제 치수 반영</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="quick-mode-badge">사진 분석 · 사진 기준</div>', unsafe_allow_html=True)

    top_a, top_b = st.columns([1, 1])
    with top_a:
        if st.button("다른 사진 분석하기", use_container_width=True):
            st.session_state["analysis_stage"] = 1
            st.session_state.pop("analysis_image_bytes", None)
            st.session_state["measurement_mode"] = "auto"
            st.rerun()
    with top_b:
        with st.expander("분석 방식 보기"):
            if has_exact_dimensions:
                st.write(f"박스 내부: {box_width_mm:.0f} × {box_length_mm:.0f} × {box_height_mm:.0f} mm")
                st.write(f"제품 높이: 약 {product_height_mm:.0f} mm")
                st.caption("실제 치수를 반영한 결과입니다.")
            else:
                st.write("사진만으로 분석했습니다.")
                st.caption("사진 기준 빈 공간은 확인할 수 있지만, 실제 크기와 체적은 치수 입력 전에는 계산하지 않습니다.")

    quality = check_image_quality(image)
    progress = st.progress(0, text="1/6 이미지 품질을 확인하고 있습니다...")
    time.sleep(0.08)
    progress.progress(15, text="2/6 박스 경계를 찾고 원근을 보정하고 있습니다...")
    box_image, _, box = detect_box(image)
    time.sleep(0.08)
    progress.progress(35, text="3/6 제품 영역을 분리하고 있습니다...")
    detections, segmented_image, product_area, total_products = detect_products(image, box)
    time.sleep(0.08)

    progress.progress(55, text="4/6 제품과 박스의 공간 비율을 계산하고 있습니다...")

    if has_exact_dimensions:
        planar = calculate_planar_measurements(
            box["warped_width"], box["warped_height"],
            box_width_mm, box_length_mm,
            product_area,
        )
        volume = calculate_volume_measurements(
            box_width_mm, box_length_mm, box_height_mm,
            planar["product_area_mm2"], product_height_mm,
        )
        volume_pvi = calculate_volume_pvi(volume["occupancy_percent"])
    else:
        # 상대 비율은 실제 mm 크기와 무관하므로 사진의 픽셀 영역만으로 계산할 수 있습니다.
        auto_w = max(float(box["warped_width"]), 1.0)
        auto_h = max(float(box["warped_height"]), 1.0)
        planar = calculate_planar_measurements(
            auto_w, auto_h,
            auto_w, auto_h,
            product_area,
        )
        volume = None
        volume_pvi = None

    pvi = calculate_pvi(planar["box_area_px"], planar["product_area_px"])
    confidence = estimate_measurement_confidence(
        box["confidence"],
        [d["confidence"] for d in detections],
        box["perspective_corrected"],
        has_exact_dimensions,
    )

    progress.progress(76, text="5/6 분석 결과와 비교 정보를 정리하고 있습니다...")

    shape_candidates = infer_box_shape_candidates(box, limit=3)
    if has_exact_dimensions and volume is not None:
        recommendation = recommend_box(volume["product_volume_mm3"], target_utilization)
        candidates = rank_boxes(volume["product_volume_mm3"], target_utilization, limit=4)
        wasted_area_cm2 = max(planar["box_area_mm2"] - planar["product_area_mm2"], 0) / 100
        carbon = calculate_carbon(wasted_area_cm2)
        current_volume = box_width_mm * box_length_mm * box_height_mm
        volume_void_for_consultant = volume["void_percent"]
    else:
        recommendation = None
        candidates = []
        carbon = {"wasted_material_g": 0.0, "carbon_kg": 0.0, "carbon_g": 0.0}
        current_volume = 0.0
        # 자동 모드는 2D 면적 빈 공간만 진단에 사용합니다.
        volume_void_for_consultant = planar["void_percent"]

    esg = calculate_esg_score(planar["void_percent"], carbon["carbon_g"])
    consultant = build_consultant_report(
        area_void=planar["void_percent"],
        volume_void=volume_void_for_consultant,
        confidence=confidence,
        current_volume_mm3=current_volume,
        recommendation=recommendation,
        esg_score=esg["score"],
        detected_products=total_products,
    )
    if not has_exact_dimensions:
        consultant["summary"] = (
            f"사진에서 보이는 바닥 면적 기준으로 빈 공간은 약 {planar['void_percent']:.1f}%입니다. "
            "실제 박스 높이와 제품 높이는 사진 한 장만으로 확정할 수 없어 체적 분석과 규격 추천은 보류했습니다."
        )
        consultant["actions"] = [
            "현재 면적 빈 공간을 기준으로 과도한 여유가 있는지 먼저 확인하세요.",
            "정확한 박스 안쪽 치수를 알고 있다면 입력해 체적 분석과 표준 박스 후보를 확인하세요.",
            "박스 전체와 제품이 잘 보이도록 위에서 촬영하면 면적 분석의 안정성이 높아집니다.",
        ]

    limitation_result = build_limitations(
        quality=quality,
        box=box,
        detections=detections,
        product_height_mm=(product_height_mm if has_exact_dimensions else 0.0),
        dimensions_confirmed=dimensions_confirmed,
    )
    self_review = build_self_review(
        limitation_result=limitation_result,
        detected_products=total_products,
        product_height_mm=(product_height_mm if has_exact_dimensions else 0.0),
    )

    if recommendation and volume is not None:
        recommended_box = recommendation["box"]
        recommended_void = float(recommendation["void_percent"])
        volume_saving_percent = max((current_volume - recommended_box.volume) / current_volume * 100, 0.0) if current_volume else 0.0
        estimated_material_saving = min(max(volume_saving_percent * 0.78, 0.0), 65.0)
        estimated_carbon_saving = min(max(volume_saving_percent * 0.62, 0.0), 55.0)
        estimated_cost_saving = min(max(volume_saving_percent * 0.45, 0.0), 40.0)
        projected_health = min(96.0, pvi["pvi"] + max(planar["void_percent"] - recommended_void, 0) * 0.42)
    else:
        recommended_void = planar["void_percent"]
        volume_saving_percent = estimated_material_saving = estimated_carbon_saving = estimated_cost_saving = 0.0
        projected_health = pvi["pvi"]

    progress.progress(100, text="6/6 분석이 완료되었습니다.")
    time.sleep(0.12)
    progress.empty()

    # 사진 품질을 숫자만 보여주지 않고 다음 행동으로 안내합니다.
    quality_score = float(quality.get("score", 0))
    box_conf_pct = float(box.get("confidence", 0)) * 100
    if quality_score < 45 or box_conf_pct < 45:
        capture_feedback_class = "capture-feedback bad"
        capture_feedback_text = "박스 경계가 선명하게 잡히지 않았어요. 네 모서리가 모두 보이도록 조금 더 위에서 다시 찍어보세요."
    elif quality_score < 65 or box_conf_pct < 65:
        capture_feedback_class = "capture-feedback warn"
        capture_feedback_text = "분석은 가능하지만, 사진을 조금만 다르게 찍으면 더 안정적인 결과를 얻을 수 있어요. 가능하면 박스 위에서 촬영하고 강한 그림자나 반사는 피해주세요."
    else:
        capture_feedback_class = "capture-feedback"
        capture_feedback_text = "박스와 제품이 잘 보이는 사진이에요. 이 사진을 기준으로 분석했습니다."

    section_head("결과", "한눈에 보기", "사진에서 확인한 핵심 결과입니다. 실제 치수를 입력하면 체적과 박스 후보까지 더 자세히 비교할 수 있어요.")

    if has_exact_dimensions and recommendation:
        result_candidate = f"{recommended_box.company} {recommended_box.code}"
        result_candidate_sub = f"{recommended_box.length}×{recommended_box.width}×{recommended_box.height} mm · 예상 빈 공간 {recommended_void:.1f}%"
    elif has_exact_dimensions:
        result_candidate = "표준 박스 후보 없음"
        result_candidate_sub = "현재 DB에 맞는 후보가 없어 다른 규격을 검토해야 합니다."
    else:
        result_candidate = "자세히 분석에서 확인"
        result_candidate_sub = "실제 치수 입력 시 표준 박스 후보를 비교합니다."

    main_result_sub = (
        f"체적 기준 빈 공간 {volume['void_percent']:.1f}% · 실제 치수 반영"
        if has_exact_dimensions and volume is not None
        else f"면적 기준 빈 공간 {planar['void_percent']:.1f}% · 사진 기준 분석"
    )

    display_void = float(volume["void_percent"] if has_exact_dimensions and volume is not None else planar["void_percent"])
    if display_void < 40:
        void_tone = "good"
    elif display_void < 60:
        void_tone = "warn"
    else:
        void_tone = "alert"

    st.markdown(f"""<div class="quick-result-grid">
        <div class="quick-result-card product"><small>제품 점유율</small><strong>{planar['occupancy_percent']:.1f}%</strong><span>사진에서 보이는 박스 바닥 기준</span></div>
        <div class="quick-result-card void {void_tone}"><small>빈 공간</small><strong>{display_void:.1f}%</strong><span>{'체적 기준' if has_exact_dimensions and volume is not None else '면적 기준'}</span></div>
        <div class="quick-result-card confidence"><small>분석 신뢰도</small><strong>{confidence:.1f}%</strong><span>사진 상태와 인식 결과 반영</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="{capture_feedback_class}">{capture_feedback_text}</div>', unsafe_allow_html=True)

    if has_exact_dimensions:
        st.markdown(f"""<div class="result-app-head">
            <div class="result-app-main"><small>자세히 분석 결과</small><strong>{consultant['status']}</strong><span>{main_result_sub}</span></div>
            <div class="result-app-side"><small>먼저 비교해볼 박스</small><strong>{result_candidate}</strong><span>{result_candidate_sub}</span></div>
        </div>""", unsafe_allow_html=True)

    level_class = limitation_result["level"]
    st.markdown(f"""<div class="status-line"><span class="status-dot {level_class}"></span><div><b>{consultant['status']}</b><span>{consultant['summary']}</span></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="v99-result-note"><b>참고해 주세요.</b> 사진에서 계산한 값과 직접 입력한 값을 구분해 보여드리며, 실제 포장에 적용하기 전 확인이 필요한 내용도 함께 안내합니다.</div>""", unsafe_allow_html=True)

    st.markdown('<div class="result-visual-shell">', unsafe_allow_html=True)
    st.markdown(f"""<div class="detect-compare-head">
        <h4>사진을 이렇게 인식했어요.</h4>
        <span>제품 {total_products}개 인식 · 박스 안쪽을 기준으로 비교</span>
    </div>
    <div class="detect-compare-labels"><div>원본 사진</div><div>AI 인식 결과</div></div>""", unsafe_allow_html=True)

    comparison_image = build_detection_comparison(image, segmented_image)
    st.image(comparison_image, use_container_width=True)

    occupancy_pct = max(0.0, min(100.0, float(planar["occupancy_percent"])))
    void_pct = max(0.0, min(100.0, float(planar["void_percent"])))
    st.markdown(f"""<div class="occupancy-card">
        <div class="occupancy-top"><b>박스 안에서 제품이 차지하는 비율</b><span>{occupancy_pct:.1f}% 점유</span></div>
        <div class="occupancy-bar">
            <div class="occupancy-product" style="width:{occupancy_pct:.1f}%"></div>
            <div class="occupancy-empty" style="width:{void_pct:.1f}%"></div>
        </div>
        <div class="occupancy-legend"><span>제품 영역 {occupancy_pct:.1f}%</span><span>빈 공간 {void_pct:.1f}%</span></div>
        <div class="detect-explain">오른쪽은 사진을 보정한 뒤 제품으로 인식한 영역을 표시한 화면입니다. 사진에서 보이는 바닥 면적을 기준으로 계산하므로 실제 3차원 부피와는 차이가 있을 수 있어요.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    if not has_exact_dimensions:
        st.markdown("""<div class="precision-upgrade">
            <h4>박스 크기를 알고 있나요?</h4>
            <p>박스 안쪽 가로·세로·높이를 입력하면 같은 사진으로 더 자세히 분석할 수 있어요.</p>
            <div class="precision-list"><span>체적 기준 빈 공간</span><span>박스 후보 비교</span><span>비용·탄소 참고값</span><span>3D 참고 모델</span></div>
        </div>""", unsafe_allow_html=True)
        if st.button("박스 크기 입력하고 더 자세히 보기", type="primary", use_container_width=True, key="precision_upgrade_v912"):
            st.session_state["analysis_stage"] = 2
            st.rerun()

    if presentation_mode:
        st.markdown("### 발표용 핵심 결과")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.markdown(f"<div class='presentation-card'><div class='card-copy'>현재 포장 효율</div><div class='big'>{pvi['pvi']:.0f}</div><small>{pvi['grade']}</small></div>", unsafe_allow_html=True)
        if has_exact_dimensions and volume is not None:
            pc2.markdown(f"<div class='presentation-card'><div class='card-copy'>체적 기준 빈 공간</div><div class='big'>{volume['void_percent']:.1f}%</div><small>정확한 치수 반영</small></div>", unsafe_allow_html=True)
        else:
            pc2.markdown(f"<div class='presentation-card'><div class='card-copy'>면적 빈 공간</div><div class='big'>{planar['void_percent']:.1f}%</div><small>사진 기준 분석</small></div>", unsafe_allow_html=True)
        pc3.markdown(f"<div class='presentation-card'><div class='card-copy'>예상 개선 점수</div><div class='big'>{projected_health:.0f}</div><small>추천 후보 적용 가정</small></div>", unsafe_allow_html=True)
        pc4.markdown(f"<div class='presentation-card'><div class='card-copy'>예상 탄소 변화</div><div class='big'>-{estimated_carbon_saving:.0f}%</div><small>비교용 추정치</small></div>", unsafe_allow_html=True)
        st.success(consultant["summary"])
        if recommendation:
            rb = recommendation['box']
            st.info(f"추천 후보: **{rb.company} {rb.code}** · {rb.length}×{rb.width}×{rb.height} mm · 예상 빈 공간 {recommended_void:.1f}%")
    else:
        st.markdown('<div class="details-intro"></div>', unsafe_allow_html=True)
        tabs = st.tabs(["결과 자세히", "박스 비교", "근거 · 신뢰도"])

        with tabs[0]:
            if has_exact_dimensions and volume is not None:
                summary_copy = f"사진에서 제품이 차지한 바닥 면적은 <b>{planar['occupancy_percent']:.1f}%</b>, 면적 기준 빈 공간은 <b>{planar['void_percent']:.1f}%</b>입니다. 입력한 치수와 높이를 반영한 체적 기준 빈 공간은 약 <b>{volume['void_percent']:.1f}%</b>입니다."
            else:
                summary_copy = f"사진에서 제품이 차지한 바닥 면적은 <b>{planar['occupancy_percent']:.1f}%</b>, 면적 기준 빈 공간은 <b>{planar['void_percent']:.1f}%</b>입니다. 이 값은 실제 박스 크기를 몰라도 사진 속 영역 비율만으로 계산할 수 있습니다."
            st.markdown(f"""<div class='v98-summary'><h3>결과를 이렇게 볼 수 있어요</h3><p>{summary_copy}</p></div>""", unsafe_allow_html=True)



            v1, v2, v3, v4 = st.columns(4)
            if has_exact_dimensions and volume is not None:
                v1.metric("박스 체적", f"{volume['box_volume_mm3']/1000:,.0f} cc")
                v2.metric("제품 추정 체적", f"{volume['product_volume_mm3']/1000:,.0f} cc")
                v3.metric("체적 점유율", f"{volume['occupancy_percent']:.1f}%")
                v4.metric("ESG 참고점수", f"{esg['score']} / {esg['grade']}")
            else:
                detected_ratio = max(float(box['warped_width']), float(box['warped_height'])) / max(min(float(box['warped_width']), float(box['warped_height'])), 1.0)
                v1.metric("제품 인식", f"{total_products}개")
                v2.metric("면적 점유율", f"{planar['occupancy_percent']:.1f}%")
                v3.metric("박스 형상비", f"{detected_ratio:.2f}:1")
                v4.metric("분석 방식", "사진 자동")

            st.markdown("#### 결과 해석")
            if has_exact_dimensions and volume is not None:
                st.markdown(f"""
- **포장 효율 참고점수 {pvi['pvi']:.0f}점**: 공간 효율, 빈 공간 관리, 보호성, 지속가능성을 합산한 프로젝트 내부 지표입니다.
- **체적 기준 빈 공간 {volume['void_percent']:.1f}%**: 입력한 박스 높이와 제품 높이를 반영한 계산값입니다.
- **측정 신뢰도 {confidence:.1f}%**: 박스 검출, 제품 분할, 원근 보정, 실측 입력 여부를 종합한 참고값입니다.

**깨지기 쉬운 제품은 보호 여유가 필요하므로 빈 공간 수치만으로 최종 박스를 결정하지 않습니다.**
""")
            else:
                st.markdown(f"""
- **면적 빈 공간 {planar['void_percent']:.1f}%**: 사진 속 박스 바닥과 제품 영역의 상대 비율로 자동 계산했습니다.
- **포장 효율 참고점수 {pvi['pvi']:.0f}점**: 현재 사진에서 확인할 수 있는 공간 효율을 중심으로 보는 프로젝트 내부 참고값입니다.
- **측정 신뢰도 {confidence:.1f}%**: 사진 품질, 박스 검출, 제품 인식, 원근 보정 상태를 종합한 참고값입니다.
- **체적·실제 박스 규격·탄소량은 아직 계산하지 않았습니다.** 사진 한 장만으로 절대 크기와 높이를 확정할 수 없기 때문입니다.

정확한 박스 안쪽 치수를 알고 있다면 **박스 비교 탭에서 바로 입력해 같은 사진으로 다시 계산**할 수 있습니다.
""")

            with st.expander("점수 구성 보기"):
                score_df = pd.DataFrame([{"항목": k, "점수": v, "최대": pvi["max_scores"][k]} for k, v in pvi["scores"].items()])
                st.bar_chart(score_df.set_index("항목")[["점수", "최대"]])

        with tabs[1]:
            if not has_exact_dimensions:
                st.markdown("#### 사진만으로 알 수 있는 박스 정보")
                st.info("사진 한 장만으로는 실제 박스의 절대 크기(mm)를 확정할 수 없습니다. 대신 박스의 가로·세로 **형상비**와 제품이 차지하는 **면적 비율**은 자동으로 계산할 수 있습니다.")

                if shape_candidates:
                    st.markdown("##### 모양 비율이 비슷한 규격")
                    shape_rows = []
                    for idx, item in enumerate(shape_candidates, 1):
                        b = item["box"]
                        shape_rows.append({
                            "순위": idx,
                            "규격 후보": f"{b.company} {b.code}",
                            "실제 규격(mm)": f"{b.length}×{b.width}×{b.height}",
                            "바닥면 비율": f"{item['box_ratio']:.2f}:1",
                            "형상 유사도": f"{item['similarity']:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(shape_rows), hide_index=True, use_container_width=True)
                    st.caption("위 규격은 사진에서 보이는 가로·세로 비율만 비교한 참고 후보입니다. 실제 박스 크기를 알아낸 결과는 아닙니다.")

                st.markdown("#### 박스 크기를 알고 있다면")
                st.write("사진은 다시 올릴 필요 없이, 치수만 입력하면 체적 기준 빈 공간과 표준 박스 후보를 바로 다시 계산할 수 있어요.")
                with st.form("upgrade_exact_dimensions"):
                    c1, c2 = st.columns(2)
                    with c1:
                        exact_w = st.number_input("박스 가로 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_width_mm", 330.0)), step=1.0, key="upgrade_w")
                        exact_l = st.number_input("박스 세로 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_length_mm", 250.0)), step=1.0, key="upgrade_l")
                    with c2:
                        exact_h = st.number_input("박스 높이 (mm)", min_value=1.0, value=float(st.session_state.get("flow_box_height_mm", 150.0)), step=1.0, key="upgrade_h")
                        exact_ph = st.number_input("제품 높이 · 대략 (mm)", min_value=0.0, value=float(st.session_state.get("flow_product_height_mm", 50.0)), step=1.0, key="upgrade_ph")
                    exact_confirmed = st.checkbox("박스 안쪽 치수를 직접 확인했어요", value=True, key="upgrade_confirmed")
                    upgrade = st.form_submit_button("입력한 치수로 다시 분석", use_container_width=True)

                if upgrade:
                    st.session_state["measurement_mode"] = "manual"
                    st.session_state["flow_box_width_mm"] = exact_w
                    st.session_state["flow_box_length_mm"] = exact_l
                    st.session_state["flow_box_height_mm"] = exact_h
                    st.session_state["flow_product_height_mm"] = exact_ph
                    st.session_state["flow_dimensions_confirmed"] = exact_confirmed
                    st.session_state["analysis_stage"] = 3
                    st.rerun()

            else:
                st.markdown("#### 먼저 비교해볼 박스")
                if candidates:
                    rows = []
                    for idx, item in enumerate(candidates, 1):
                        b = item["box"]
                        rows.append({
                            "순위": idx,
                            "후보": f"{b.company} {b.code}",
                            "규격(mm)": f"{b.length}×{b.width}×{b.height}",
                            "활용률(%)": round(item["utilization"]*100,1),
                            "빈 공간(%)": round(item["void_percent"],1),
                            "비용(원)": b.cost,
                            "탄소(kgCO₂e)": b.carbon,
                            "종합점수": item["total_score"],
                        })
                    df = pd.DataFrame(rows)
                    best = candidates[0]["box"]
                    st.success(f"먼저 비교해볼 후보: **{best.company} {best.code}** · {best.length} × {best.width} × {best.height} mm")
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.warning("현재 등록된 규격에서는 제품이 들어갈 만한 박스를 찾지 못했어요. 다른 규격이나 맞춤 포장을 함께 검토해보세요.")

                st.markdown("#### 현재 포장과 비교해보기")
                before_after = pd.DataFrame([
                    ["체적 기준 빈 공간", f"{volume['void_percent']:.1f}%", f"{recommended_void:.1f}%", f"-{max(volume['void_percent']-recommended_void,0):.1f}%p"],
                    ["포장 효율 참고점수", f"{pvi['pvi']:.0f}점", f"{projected_health:.0f}점", f"+{max(projected_health-pvi['pvi'],0):.0f}점"],
                    ["박스 체적", f"{current_volume/1000:,.0f}cc", f"{(recommendation['box'].volume/1000 if recommendation else current_volume/1000):,.0f}cc", f"-{volume_saving_percent:.1f}%"],
                    ["포장재 사용", "100 기준", f"{100-estimated_material_saving:.0f} 기준", f"-{estimated_material_saving:.0f}%"],
                    ["탄소 영향", "100 기준", f"{100-estimated_carbon_saving:.0f} 기준", f"-{estimated_carbon_saving:.0f}%"],
                    ["예상 비용", "100 기준", f"{100-estimated_cost_saving:.0f} 기준", f"-{estimated_cost_saving:.0f}%"],
                ], columns=["비교 항목", "현재", "후보 적용", "예상 변화"])
                st.dataframe(before_after, hide_index=True, use_container_width=True)
                st.caption("변화율은 후보 박스의 체적을 기준으로 계산한 참고값입니다. 실제 재질, 구매 단가, 운송 조건에 따라 달라질 수 있어요.")

                st.markdown("#### 함께 확인하면 좋은 점")
                st.write(consultant["summary"])
                for action in consultant["actions"]:
                    st.write(f"• {action}")

        with tabs[2]:
            st.markdown("#### 이번 결과는 어느 정도 참고할 수 있을까요?")
            st.markdown(f"""<div class='v98-confidence-grid'>
                <div class='v98-confidence-item'><small>이미지 품질</small><strong>{limitation_result['quality_score']:.0f} / 100</strong></div>
                <div class='v98-confidence-item'><small>종합 측정 신뢰도</small><strong>{confidence:.1f}%</strong></div>
                <div class='v98-confidence-item'><small>박스 검출</small><strong>{limitation_result['box_confidence']:.1f}%</strong></div>
                <div class='v98-confidence-item'><small>제품 검출 평균</small><strong>{limitation_result['product_confidence']:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

            if limitation_result["level"] == "red":
                st.error(f"분석 품질: {limitation_result['label']} — {limitation_result['decision']}")
            elif limitation_result["level"] == "yellow":
                st.warning(f"분석 품질: {limitation_result['label']} — {limitation_result['decision']}")
            else:
                st.success(f"분석 품질: {limitation_result['label']} — {limitation_result['decision']}")

            with st.expander("계산 과정 보기"):
                trace_rows = [
                    ("01", "이미지 품질 확인", f"품질 점수 {quality['score']:.0f}/100 · {quality['message']}"),
                    ("02", "박스 경계 검출", f"검출 신뢰도 {box['confidence']*100:.1f}% · 원근 보정 {'적용' if box['perspective_corrected'] else '미적용'}"),
                    ("03", "제품 영역 분리", f"제품 {total_products}개 · 평균 신뢰도 {(sum(d['confidence'] for d in detections)/len(detections)*100) if detections else 0:.1f}%"),
                    ("04", "공간 비율 계산", (f"면적 빈 공간 {planar['void_percent']:.1f}% · 체적 기준 빈 공간 {volume['void_percent']:.1f}%" if has_exact_dimensions and volume is not None else f"면적 빈 공간 {planar['void_percent']:.1f}% · 체적은 미계산")),
                    ("05", "점수·후보 비교", (f"포장 효율 {pvi['pvi']:.0f}점 · 후보 {len(candidates)}개 비교" if has_exact_dimensions else f"포장 효율 {pvi['pvi']:.0f}점 · 형상 후보 {len(shape_candidates)}개 참고")),
                ]
                for no, title, detail in trace_rows:
                    st.markdown(f"<div class='trace-card'><b>{no} · {title}</b><br><small>{detail}</small></div>", unsafe_allow_html=True)
                if has_exact_dimensions and volume is not None:
                    st.markdown(f"""<div class='formula-box'>
<b>면적 제품 점유율</b> = {planar['product_area_px']:,.0f} ÷ {planar['box_area_px']:,.0f} × 100 = <b>{planar['occupancy_percent']:.1f}%</b><br>
<b>면적 빈 공간</b> = 100 − {planar['occupancy_percent']:.1f} = <b>{planar['void_percent']:.1f}%</b><br>
<b>체적 제품 점유율</b> = {volume['product_volume_mm3']/1000:,.1f}cc ÷ {volume['box_volume_mm3']/1000:,.1f}cc × 100 = <b>{volume['occupancy_percent']:.1f}%</b><br>
<b>체적 기준 빈 공간</b> = 100 − {volume['occupancy_percent']:.1f} = <b>{volume['void_percent']:.1f}%</b>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='formula-box'>
<b>면적 제품 점유율</b> = 제품 픽셀 면적 ÷ 박스 내부 픽셀 면적 × 100 = <b>{planar['occupancy_percent']:.1f}%</b><br>
<b>면적 빈 공간</b> = 100 − {planar['occupancy_percent']:.1f} = <b>{planar['void_percent']:.1f}%</b><br><br>
사진 속 <b>비율</b>만 사용하므로 실제 mm 크기를 몰라도 계산할 수 있습니다. 절대 치수와 높이가 필요한 체적 계산은 수행하지 않습니다.
</div>""", unsafe_allow_html=True)

            with st.expander("박스 경계·세부 인식 이미지 보기"):
                i1, i2, i3 = st.columns(3)
                i1.image(image, caption="원본", use_container_width=True)
                i2.image(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB), caption="박스 경계", use_container_width=True)
                i3.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="제품 영역", use_container_width=True)
                st.caption(f"촬영 품질 {quality['score']:.0f}/100 · 밝기 {quality['brightness']} · 선명도 {quality['sharpness']} · {quality['message']}")

            with st.expander("3D 참고 모델 보기"):
                if has_exact_dimensions:
                    aspect = box_width_mm / max(box_length_mm, 1)
                    product_width = min(box_width_mm, np.sqrt(planar["product_area_mm2"] * aspect)) if planar["product_area_mm2"] > 0 else 1
                    product_length = min(box_length_mm, planar["product_area_mm2"] / max(product_width, 1))
                    st.plotly_chart(build_digital_twin(box_width_mm, box_length_mm, box_height_mm, product_width, product_length, product_height_mm), use_container_width=True)
                    st.caption("제품 영역을 단순한 직사각형 형태로 표현한 참고 모델이며, 실제 제품 모양을 그대로 복원한 것은 아닙니다.")
                else:
                    st.info("3D 참고 모델은 박스 가로·세로·높이가 있어야 의미 있는 크기로 만들 수 있습니다. 정확한 치수를 입력하면 이 화면이 활성화됩니다.")

            st.markdown("#### 이번 분석에서 확인하기 어려운 점")
            for item in limitation_result["limitations"]:
                st.write(f"• {item}")
            if limitation_result["actions"]:
                st.markdown("#### 더 안정적인 결과를 얻으려면")
                for action in limitation_result["actions"]:
                    st.write(f"• {action}")

            with st.expander("AI Self Review"):
                for check in self_review:
                    st.write(check)
                st.caption("Self Review는 모델의 정확성을 증명하는 기능이 아니라 입력 조건과 계산 범위를 다시 확인해 과도한 해석을 줄이기 위한 안전 장치입니다.")

    if has_exact_dimensions and volume is not None:
        report_sections = {
            "핵심 분석 요약": {
                "포장 효율 참고점수": f"{pvi['pvi']:.0f}/100",
                "등급": pvi["grade"], "진단": consultant["status"], "요약": consultant["summary"],
            },
            "공간 및 체적 분석": {
                "면적 점유율": f"{planar['occupancy_percent']:.1f}%",
                "면적 빈 공간": f"{planar['void_percent']:.1f}%",
                "체적 점유율": f"{volume['occupancy_percent']:.1f}%",
                "체적 빈 공간": f"{volume['void_percent']:.1f}%",
                "검출 제품 수": total_products,
            },
            "ESG 및 환경 영향": {
                "ESG 참고점수": f"{esg['score']} / {esg['grade']}",
                "추정 포장재 낭비": f"{carbon['wasted_material_g']:.2f} g",
                "추정 탄소 영향": f"{carbon['carbon_g']:.2f} gCO₂",
            },
            "추천 결과": ({
                "추천 박스": consultant["savings"]["box_name"],
                "추천 규격": consultant["savings"]["size"],
                "예상 빈 공간": f"{consultant['savings']['expected_void_percent']}%",
                "체적 절감": f"{consultant['savings']['volume_saving_percent']}%",
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
                "입력값": "사용자가 제공한 박스 내부 치수와 제품 높이",
                "추정값": "환경·비용 모델을 이용한 비교값",
                "미평가": "충격·진동·압축, 재질 강도, 법적 적합성, 공식 LCA",
            },
        }
    else:
        report_sections = {
            "핵심 분석 요약": {
                "분석 방식": "사진만으로 분석했습니다.",
                "포장 효율 참고점수": f"{pvi['pvi']:.0f}/100",
                "진단": consultant["status"],
                "요약": consultant["summary"],
            },
            "사진 기반 공간 분석": {
                "면적 점유율": f"{planar['occupancy_percent']:.1f}%",
                "면적 빈 공간": f"{planar['void_percent']:.1f}%",
                "검출 제품 수": total_products,
            },
            "미계산 항목": {
                "실제 박스 규격(mm)": "사진 한 장만으로 확정 불가",
                "체적 기준 빈 공간": "정확한 박스 높이·제품 높이 입력 시 계산",
                "박스 후보 비교": "정확한 실제 치수 입력 시 제공",
                "포장재·탄소 추정": "물리 크기 정보가 없어 계산하지 않음",
            },
            "신뢰도": {
                "이미지 품질": f"{limitation_result['quality_score']:.0f}/100",
                "박스 검출": f"{limitation_result['box_confidence']:.1f}%",
                "제품 검출 평균": f"{limitation_result['product_confidence']:.1f}%",
                "종합 측정 신뢰도": f"{confidence:.1f}%",
            },
            "개선 권고": consultant["actions"],
            "이번 분석에서 확인된 제한사항": limitation_result["limitations"],
            "AI 자기검토": self_review,
        }
    pdf_bytes = build_pdf_report({"sections": report_sections})
    st.divider()
    col_report, col_cert = st.columns(2)
    with col_report:
        st.download_button("투명성 보고서 다운로드", pdf_bytes, "AI_포장분석_투명성보고서_FINAL.pdf", "application/pdf", use_container_width=True)
    certificate_bytes = build_analysis_certificate({
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "health_score": f"{pvi['pvi']:.0f}/100",
        "grade": pvi["grade"],
        "area_void": f"{planar['void_percent']:.1f}%",
        "volume_void": (f"{volume['void_percent']:.1f}%" if has_exact_dimensions and volume is not None else "미계산"),
        "esg": (f"{esg['score']} / {esg['grade']}" if has_exact_dimensions else "사진 기반 참고"),
        "confidence": f"{confidence:.1f}%",
        "recommended_box": consultant["savings"]["box_name"] if consultant["savings"] else ("정확한 치수 입력 필요" if not has_exact_dimensions else "맞춤 박스 검토"),
        "recommended_size": consultant["savings"]["size"] if consultant["savings"] else ("미계산" if not has_exact_dimensions else "표준 후보 없음"),
        "carbon_saving": (f"{estimated_carbon_saving:.0f}%" if has_exact_dimensions else "미계산"),
        "summary": consultant["summary"],
    })
    with col_cert:
        st.download_button("분석 결과 요약서 다운로드", certificate_bytes, "AI_포장분석_결과요약서_FINAL.pdf", "application/pdf", use_container_width=True)


st.markdown("""<div class="footer-brand"><strong>Better Life For Us</strong><span>AI Packaging Intelligence · 사진으로 포장 상태를 확인하고, 필요하면 실제 치수를 더해 박스 선택을 비교할 수 있어요.</span></div>""", unsafe_allow_html=True)