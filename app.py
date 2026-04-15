"""
DriveValue — used-car price estimator for the Indian market.
Built on top of a CarDekho-trained ExtraTrees / RandomForest pipeline.
"""

from html import escape
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.data_loader import load_market_data
import joblib

MODEL_PATH = Path("models/car_price_model_v3.pkl")

SPEC_COLS = [
    "mileage", "engine_cc", "max_power", "max_torque",
    "length", "width", "height", "seats",
]

st.set_page_config(
    page_title="DriveValue · Car Price Estimator",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── small helpers ──

def inr(v):
    return f"₹ {int(round(v)):,}"

def inr_short(v):
    if v >= 1e7:  return f"₹ {v/1e7:.2f} Cr"
    if v >= 1e5:  return f"₹ {v/1e5:.2f} L"
    return inr(v)

def num(v):
    return f"{int(round(float(v))):,}"

def safe_idx(opts, pref):
    return opts.index(pref) if pref in opts else 0

def mode_of(s, fb=None):
    c = s.dropna()
    if c.empty: return fb
    m = c.mode()
    return str(m.iloc[0]) if not m.empty else str(c.iloc[0])

def numeric_mode(s, fb):
    c = pd.to_numeric(s, errors="coerce").dropna()
    if c.empty: return float(fb)
    m = c.mode()
    return float(m.iloc[0]) if not m.empty else float(c.median())

# ── cached loaders ──

@st.cache_data(show_spinner=False)
def get_data():
    # reload data augmented
    return load_market_data()

@st.cache_resource(show_spinner=False)
def get_bundle():
    # reload model bundle v4
    return joblib.load(MODEL_PATH)

# ── vehicle profile ──

def vehicle_profile(df, brand, model_name, variant_name):
    sub = df[(df["brand"] == brand) & (df["model"] == model_name) & (df["car_name"] == variant_name)].copy()
    if sub.empty:
        sub = df[(df["brand"] == brand) & (df["model"] == model_name)].copy()
    if sub.empty:
        sub = df[df["brand"] == brand].copy()

    p = {
        "brand": brand, "model": model_name, "car_name": variant_name,
        "n": len(sub),
        "med_price": float(sub["price"].median()),
        "p25": float(sub["price"].quantile(0.25)),
        "p75": float(sub["price"].quantile(0.75)),
    }
    for col, fb in [
        ("fuel_type","petrol"), ("transmission_type","manual"),
        ("seller_type","dealer"), ("owner_type","first"),
        ("body_type","hatchback cars"), ("drive_type","fwd"),
    ]:
        p[f"def_{col}"] = mode_of(sub[col], fb) or fb
    p["def_year"] = float(sub["model_year"].median())
    for col in SPEC_COLS:
        p[col] = numeric_mode(sub[col], float(sub[col].median()))
    return sub, p

def market_band(sub, year, fuel, trans, seller):
    c = sub[sub["fuel_type"] == fuel].copy()
    if len(c) >= 20: c = c[c["transmission_type"] == trans]
    if len(c) >= 20: c = c[c["seller_type"] == seller]
    w = c[np.abs(c["model_year"] - year) <= 1]
    if len(w) >= 10:
        c = w
    elif len(c) >= 20:
        w2 = c[np.abs(c["model_year"] - year) <= 2]
        if len(w2) >= 10: c = w2
    if c.empty: c = sub
    return float(c["price"].median()), float(c["price"].quantile(.25)), float(c["price"].quantile(.75))


# ═══════════════════════════════════════════════════
#  CSS — warm dark theme with color and visual depth
# ═══════════════════════════════════════════════════

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0E0E12; 
        --surf: linear-gradient(160deg, rgba(30, 30, 36, 0.92), rgba(18, 18, 22, 0.96)); 
        --surf-hov: linear-gradient(160deg, rgba(36, 36, 42, 0.95), rgba(22, 22, 26, 0.98));
        --bdr: rgba(255, 255, 255, 0.08); 
        --bdr2: rgba(255, 255, 255, 0.16);
        --txt: #ffffff; 
        --sub: #A3A3A8; 
        --dim: #6B6B72;
        --accent: #cda434; 
        --accent-hov: #e0b439;
        --glow: 0 8px 32px rgba(205, 164, 52, 0.08);
        --r: 16px; 
        --rs: 12px;
        --bd: 'Inter', sans-serif;
    }

    /* ── keyframes ── */
    @keyframes spinRing { 0% { transform: rotateY(0deg) rotateX(20deg); } 100% { transform: rotateY(360deg) rotateX(20deg); } }
    @keyframes pulseGlow { 0%, 100% { box-shadow: 0 0 15px rgba(205,164,52,0.1); } 50% { box-shadow: 0 0 30px rgba(205,164,52,0.3); } }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes road { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    @keyframes carBob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
    @keyframes wheelSpin { to { transform: rotate(360deg); } }

    html, body, .stApp { background: var(--bg)!important; color: var(--txt); font-family: var(--bd); }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent!important; }
    #MainMenu, footer, [data-testid="stDecoration"] { display: none!important; }
    .block-container { max-width: 1220px; padding: 0 2rem 4rem; }

    /* ── nav ── */
    .dv-nav { display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 0; border-bottom: 1px solid var(--bdr); }
    .dv-logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.35rem; letter-spacing: -0.03em; color: var(--txt); text-decoration: none; }
    .dv-logo-badge { width: 34px; height: 34px; background: var(--surf); border: 1px solid var(--accent); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: var(--glow); }
    .dv-logo-badge svg { width: 22px; height: 22px; stroke: var(--accent); }
    .dv-logo .lo-dim { color: var(--dim); }
    .dv-tag { font-size: 0.75rem; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--bdr); color: var(--sub); background: var(--surf); font-weight: 500; }

    /* ── hero ── */
    .dv-hero { padding: 4.5rem 0 3rem; max-width: 680px; animation: fadeUp 0.6s ease both; }
    .dv-hero .ey { display: inline-block; font-size: 0.75rem; font-weight: 600; color: var(--accent); letter-spacing: 0.06em; text-transform: uppercase; padding: 6px 12px; border-radius: 6px; background: rgba(205, 164, 52, 0.1); border: 1px solid rgba(205,164,52,0.2); margin-bottom: 1.5rem; }
    .dv-hero h1 { font-size: clamp(2.4rem, 5vw, 3.8rem); font-weight: 700; line-height: 1.1; letter-spacing: -0.04em; color: var(--txt); margin: 0 0 1.2rem; }
    .dv-hero h1 em { font-style: normal; color: var(--sub); }
    .dv-hero .hsub { color: var(--sub); font-size: 1.05rem; line-height: 1.6; font-weight: 400; }
    .dv-hero .hsub strong { color: var(--txt); font-weight: 500; }

    /* ── section label ── */
    .sec-lbl { font-size: 0.85rem; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px; }
    .sec-lbl::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(205,164,52,0.3), transparent); }

    /* ── 3D glass cards ── */
    .gc { background: var(--surf); border: 1px solid var(--bdr); border-radius: var(--r); padding: 1.5rem; margin-bottom: 16px; transition: all 0.3s ease; box-shadow: 0 8px 24px rgba(0,0,0,0.3); position: relative; overflow: hidden; }
    .gc::before { content: ''; position: absolute; top:0; left:10%; right:10%; height:1px; background: linear-gradient(90deg, transparent, rgba(205,164,52,0.15), transparent); }
    .gc:hover { border-color: var(--bdr2); box-shadow: var(--glow); transform: translateY(-2px); }
    .gc .lbl { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); margin-bottom: 0.6rem; }
    .gc h3 { font-size: 1.3rem; font-weight: 600; color: var(--txt); margin: 0 0 0.5rem; letter-spacing: -0.02em; }
    .gc .desc { color: var(--sub); font-size: 0.85rem; line-height: 1.6; margin-bottom: 1.2rem; }

    /* kv grid */
    .kvg { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .kv { background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.03); border-radius: var(--rs); padding: 12px 14px; }
    .kv span { display: block; font-size: 0.75rem; color: var(--dim); margin-bottom: 4px; }
    .kv strong { font-size: 0.95rem; font-weight: 600; color: var(--txt); }
    .kv strong.w { color: var(--accent); }

    /* spec list */
    .spl { display: flex; flex-direction: column; gap: 8px; }
    .spl .sr { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.03); border-radius: var(--rs); transition: background 0.2s; }
    .spl .sr:hover { background: rgba(255,255,255,0.03); }
    .spl .sr span { font-size: 0.85rem; color: var(--sub); }
    .spl .sr strong { font-size: 0.85rem; font-weight: 500; color: var(--txt); }

    /* ── result box (modal output) ── */
    .res-wrap { background: var(--surf); border: 1px solid rgba(205,164,52,0.2); border-radius: var(--r); padding: 2.5rem; margin: 1rem 0; box-shadow: var(--glow); position: relative; overflow: hidden; animation: fadeUp 0.5s ease both; }
    .res-grid { display: block; text-align: center; }
    .res-price .lbl { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); margin-bottom: 0.5rem; }
    .big-p { font-size: clamp(2.4rem, 4.5vw, 3.6rem); font-weight: 700; letter-spacing: -0.04em; color: var(--txt); margin: 0 0 1.2rem; filter: drop-shadow(0 0 10px rgba(205,164,52,0.1)); }
    
    .dtag { display: inline-block; font-size: 0.8rem; font-weight: 500; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--bdr); margin-bottom: 1.5rem; }
    .dtag.up { background: rgba(255, 255, 255, 0.05); color: var(--sub); }
    .dtag.dn { background: rgba(205, 164, 52, 0.1); color: var(--accent); border-color: rgba(205, 164, 52, 0.3); }
    
    .res-dets { display: flex; flex-direction: column; gap: 10px; margin-top: 1rem; }
    .rd { background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.04); border-radius: var(--rs); padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; }
    .rd span { font-size: 0.8rem; color: var(--sub); }
    .rd strong { font-size: 0.85rem; font-weight: 500; color: var(--txt); text-align: right; }

    /* ── about steps ── */
    .step-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 0.5rem; }
    .step { background: var(--surf); border: 1px solid var(--bdr); border-radius: var(--r); padding: 1.6rem; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .step:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
    .step .snum { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem; background: rgba(205,164,52,0.1); border: 1px solid rgba(205,164,52,0.2); color: var(--accent); }
    .step h4 { font-size: 1rem; font-weight: 600; color: var(--txt); margin: 0 0 0.6rem; }
    .step p { color: var(--sub); font-size: 0.85rem; line-height: 1.6; margin: 0; }

    /* ── 3d modal loader ── */
    .car-loader{ display:flex; flex-direction:column; align-items:center; padding:2rem 0; animation:fadeUp .4s ease both;}
    .car-scene{ position:relative; width:200px; height:65px; overflow:hidden; }
    .car-body-w{ position:absolute; left:50%; top:10px; transform:translateX(-50%); z-index:2; animation:carBob .6s ease-in-out infinite; }
    .cb{ width:64px; height:20px; background:linear-gradient(135deg,var(--accent),var(--accent-hov)); border-radius:4px 4px 2px 2px; position:relative; box-shadow: var(--glow); }
    .cb .roof{ position:absolute; top:-13px; left:10px; width:34px; height:13px; background:var(--accent); border-radius:7px 7px 0 0; opacity:.9; }
    .cb .win{ position:absolute; top:-11px; left:14px; width:26px; height:9px; background:rgba(0,0,0,.3); border-radius:5px 5px 0 0; }
    .whl{ width:13px; height:13px; border-radius:50%; background:#111; border:2px solid #555;
           position:absolute; bottom:-6px; animation:wheelSpin .4s linear infinite; }
    .whl::after{ content:''; position:absolute; top:3px; left:3px; width:3px; height:3px; background:var(--accent); border-radius:50%; }
    .whl-l{ left:7px; } .whl-r{ right:7px; }
    .road-ln{ position:absolute; bottom:7px; left:0; width:400px; display:flex; gap:20px; animation:road .8s linear infinite; }
    .road-ln span{ display:block; width:18px; height:2px; background:var(--accent); opacity:.3; border-radius:1px; }
    .car-loader p{ color:var(--accent); font-size:.85rem; margin-top:1.5rem; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; }

    /* ── streamlit overrides ── */
    .stSelectbox label, .stNumberInput label, .stRadio label { color: var(--txt)!important; font-weight: 500; font-size: 0.85rem; }
    div[data-baseweb="select"] > div { background: var(--bg)!important; border: 1px solid rgba(255,255,255,0.1)!important; border-radius: var(--rs)!important; color: var(--txt)!important; transition: all 0.2s!important; }
    div[data-baseweb="select"] > div:hover { border-color: rgba(205,164,52,0.3)!important; }
    div[data-baseweb="select"] > div:focus-within { border-color: var(--accent)!important; box-shadow: 0 0 0 1px var(--accent)!important; }
    
    .stNumberInput input { background: var(--bg)!important; border: 1px solid rgba(255,255,255,0.1)!important; border-radius: var(--rs)!important; color: var(--txt)!important; }
    .stNumberInput input:focus { border-color: var(--accent)!important; box-shadow: 0 0 0 1px var(--accent)!important; }
    
    div[role="radiogroup"] > label { border: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2); border-radius: var(--rs); padding: 0.7rem 0.85rem; margin-bottom: 0.35rem; transition: background 0.2s; }
    div[role="radiogroup"] > label:hover { background: rgba(205,164,52,0.05); border-color: rgba(205,164,52,0.2); }
    
    .stFormSubmitButton > button { width: 100%; min-height: 2.9rem; border-radius: var(--rs); border: none; background: linear-gradient(135deg, #e0b439, #cda434)!important; color: #0E0E12!important; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; transition: all 0.25s; cursor: pointer; box-shadow: 0 4px 15px rgba(205,164,52,0.2); }
    .stFormSubmitButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(205,164,52,0.4); filter: brightness(1.1); }

    .stButton > button[kind="primary"] { background: var(--surf)!important; color: var(--accent)!important; border-radius: var(--rs)!important; font-weight: 500; border: 1px solid rgba(205,164,52,0.3)!important; min-height: 2.8rem; transition: all 0.2s;}
    .stButton > button[kind="primary"]:hover { background: rgba(205,164,52,0.1)!important; border-color: var(--accent)!important; box-shadow: var(--glow); }

    .stDataFrame { border: 1px solid rgba(255,255,255,0.1)!important; border-radius: var(--r)!important; overflow: hidden; background: var(--bg); }

    /* footer */
    .dv-ft { text-align: center; padding: 3rem 0 1rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 2rem; color: var(--dim); font-size: 0.8rem; }
    .dv-ft strong { color: var(--txt); font-weight: 500; }

    /* responsive */
    @media(max-width: 900px) {
        .kvg, .res-grid, .step-grid { grid-template-columns: 1fr; }
        .dv-hero h1 { font-size: 2.2rem; }
        .rd { flex-direction: column; align-items: flex-start; gap: 4px; }
    }
    </style>""", unsafe_allow_html=True)





# ═══════════════════════════════════════════════════
#  Modal Popup Estimation
# ═══════════════════════════════════════════════════

@st.dialog("Prediction Engine")
def estimation_popup(price, med, lo, hi, pctl, pkey):
    # Create an empty placeholder container
    container = st.empty()
    
    # 1. Render the Car Animation
    container.markdown("""
    <div class="car-loader">
        <div class="car-scene">
            <div class="car-body-w">
                <div class="cb"><div class="roof"></div><div class="win"></div></div>
                <div class="whl whl-l"></div><div class="whl whl-r"></div>
            </div>
            <div class="road-ln"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
        </div>
        <p>Crunching the numbers…</p>
    </div>""", unsafe_allow_html=True)

    import time
    time.sleep(2.5)

    # 2. Render the actual Price Estimate Result inside the modal
    delta = price - med
    sign = "+" if delta >= 0 else "−"
    cls = "up" if delta >= 0 else "dn"

    container.markdown(f"""
    <div class="res-wrap">
        <div class="res-grid">
            <div class="res-price">
                <p class="lbl">Estimated price</p>
                <h2 class="big-p">{inr(price)}</h2>
                <span class="dtag {cls}">{sign} {inr_short(abs(delta))} vs market median</span>
            </div>
            <div class="res-dets">
                <div class="rd"><span>Comparable band</span><strong>{inr_short(lo)} – {inr_short(hi)}</strong></div>
                <div class="rd"><span>Market median</span><strong>{inr_short(med)}</strong></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  Main application
# ═══════════════════════════════════════════════════

def main():
    inject_css()

    df = get_data()
    bun = get_bundle()
    model = bun["model"]
    met = bun["candidate_metrics"][bun["selected_model_name"]]

    # ── Navbar ──
    rows_count = int(bun["training_rows"])
    st.markdown(f"""
    <div class="dv-nav">
        <div class="dv-logo">
            <div class="dv-logo-badge">
                <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M24 2 L44 14 L44 34 L24 46 L4 34 L4 14 Z" fill="rgba(255,255,255,0.05)" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
                    <path d="M12 30 L15 30 L17 26 L21 23 L27 23 L31 26 L33 30 L36 30" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M21 23 L22 20 L26 20 L27 23" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="16" cy="31" r="2.2" stroke="currentColor" stroke-width="1.5"/>
                    <circle cx="32" cy="31" r="2.2" stroke="currentColor" stroke-width="1.5"/>
                </svg>
            </div>
            <span class="lo-dim">DRIVE</span><span class="lo-hi">VALUE</span>
        </div>
        <span class="dv-tag">{rows_count:,} listings analysed</span>
    </div>""", unsafe_allow_html=True)

    # ── Hero ──
    st.markdown(f"""
    <div class="dv-hero">
        <span class="ey">ML-Powered Valuation</span>
        <h1>Know what your car<br>is <em>really worth</em></h1>
        <p class="hsub">
            Trained on <strong>{rows_count:,}</strong> verified Indian used-car listings from
            CarDekho — covering <strong>{df["brand"].nunique()}</strong> brands and
            <strong>{df["model"].nunique()}</strong> models. Pick your car, tweak a few
            parameters, and get a data-backed price estimate in seconds.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── KPI logging backend ──
    mape = met.get("mape", (met["mae"] / float(df["price"].median())) * 100)
    # the frontend display of MAE/MAPE has been removed out of consumer-view
    
    # ── How it works — step cards (not raw code!) ──
    st.markdown('<div class="sec-lbl">How it works — our data pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-grid">
        <div class="step">
            <div class="snum c1">1</div>
            <h4>Data Collection</h4>
            <p>We load a merged CarDekho listing export from Kaggle — each row is a real used-car
               listing with 25+ attributes including brand, variant, specifications, seller info,
               and the actual transaction price.</p>
        </div>
        <div class="step">
            <div class="snum c1">2</div>
            <h4>Text Cleaning</h4>
            <p>All categorical fields (brand, model, fuel, body type, seller, etc.) are normalised —
               stripped of whitespace, collapsed to single spaces, and lowercased so that different
               spellings of the same value get grouped correctly.</p>
        </div>
        <div class="step">
            <div class="snum c1">3</div>
            <h4>Numeric Parsing</h4>
            <p>Engine displacement, max power, max torque arrive as mixed-format strings like
               "1498 cc" or "113.4bhp". We extract the first number using a regex pattern and
               cross-reference between related columns when one is missing.</p>
        </div>
        <div class="step">
            <div class="snum c1">4</div>
            <h4>Null Removal & Dedup</h4>
            <p>Cells with "nan", "none", or blank values are converted to proper nulls. Any row
               missing even one of the 19 required columns is dropped. Exact duplicate rows are
               removed — leaving us with 31,702 clean records.</p>
        </div>
        <div class="step">
            <div class="snum c1">5</div>
            <h4>Outlier Filtering</h4>
            <p>Extreme values are capped: price and km at the 99.5th percentile. Hard bounds enforce
               mileage 5–50 kmpl, engine 500–7000 cc, power &amp; torque ≥ 20,
               seats 2–14, and realistic body dimensions to keep only valid listings.</p>
        </div>
        <div class="step">
            <div class="snum c1">6</div>
            <h4>Predictive Modeling</h4>
            <p>We use Gradient Boosting to predict the <b>Full Listing Value</b> based on 
               historical trends and market inflation (6.0% annually). No discounts are applied.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:3.5rem'></div>", unsafe_allow_html=True)



    # ── Two-column: form + vehicle info ──
    brands = sorted(df["brand"].unique().tolist())
    cur_brand = st.session_state.get("brand_sel", brands[0])

    col_form, _, col_info = st.columns([1.1, 0.04, 0.86], gap="small")

    with col_form:
        st.markdown('<div class="sec-lbl">Configure your vehicle</div>', unsafe_allow_html=True)

        cur_brand = st.selectbox("Brand", brands, index=safe_idx(brands, cur_brand), key="brand_sel")
        models = sorted(df.loc[df["brand"]==cur_brand, "model"].unique().tolist())
        pref_model = st.session_state.get("model_sel")
        cur_model = st.selectbox("Model", models, index=safe_idx(models, pref_model), key="model_sel")
        
        variants = sorted(df.loc[(df["brand"]==cur_brand) & (df["model"]==cur_model), "car_name"].unique().tolist())
        pref_variant = st.session_state.get("variant_sel")
        cur_variant = st.selectbox("Variant", variants, index=safe_idx(variants, pref_variant), key="variant_sel")

        sub, prof = vehicle_profile(df, cur_brand, cur_model, cur_variant)
        pkey = f"{cur_brand}|{cur_model}|{cur_variant}"
        fuels = sorted(sub["fuel_type"].unique().tolist())
        trans_opts = sorted(sub["transmission_type"].unique().tolist())
        sellers = ["dealer", "individual"]

        with st.form("est_form"):
            c1, c2 = st.columns(2)
            with c1:
                reg_year = st.number_input("Registration year", 1990, 2024,
                    int(round(prof["def_year"])), step=1)
            with c2:
                km = st.number_input("Kilometers driven", 0, 500000, 0, step=1000)

            fuel = st.selectbox("Fuel type", fuels, index=safe_idx(fuels, str(prof["def_fuel_type"])))

            r1, r2 = st.columns(2)
            with r1:
                ttype = st.radio("Transmission", trans_opts, index=safe_idx(trans_opts, str(prof["def_transmission_type"])))
            with r2:
                stype = st.radio("Seller type", sellers, index=safe_idx(sellers, str(prof["def_seller_type"])))

            owner_opts = ["first", "second", "third", "fourth & above owner", "test drive car"]
            otype = st.selectbox("Owner type", owner_opts, format_func=lambda x: x.title(), index=safe_idx(owner_opts, str(prof["def_owner_type"])))

            go = st.form_submit_button("Estimate price", width="stretch")

        if go:
            car_age = max(2024 - reg_year, 1)
            km_val = float(km)
            row = pd.DataFrame([{
                "brand": cur_brand, "model": cur_model, "car_name": cur_variant,
                "seller_type": stype, "fuel_type": fuel,
                "transmission_type": ttype,
                "owner_type": otype,
                "body_type": str(prof["def_body_type"]),
                "drive_type": str(prof["def_drive_type"]),
                "model_year": float(reg_year),
                "car_age": float(car_age),
                "km_driven": km_val,
                "km_per_year": km_val / car_age,
                "mileage": float(prof["mileage"]),
                "engine_cc": float(prof["engine_cc"]),
                "max_power": float(prof["max_power"]),
                "max_torque": float(prof["max_torque"]),
                "seats": float(prof["seats"]),
                "length": float(prof["length"]),
                "width": float(prof["width"]),
                "height": float(prof["height"]),
            }])
            price = max(float(np.expm1(model.predict(row)[0])), 0.0)
            med, lo, hi = market_band(sub, reg_year, fuel, ttype, stype)
            pctl = int(round(float((sub["price"] <= price).mean()) * 100))

            estimation_popup(price, med, lo, hi, pctl, pkey)

    with col_info:
        # vehicle card
        st.markdown(f"""
        <div class="gc">
            <p class="lbl">Selected vehicle</p>
            <h3>{escape(str(prof["car_name"]).title())}</h3>
            <p class="desc">Technical specs and hidden fields are auto-filled from the model cluster.</p>
            <div class="kvg">
                <div class="kv"><span>Listings</span><strong>{num(prof["n"])}</strong></div>
                <div class="kv"><span>Market price</span><strong class="w">{inr_short(prof["med_price"])}</strong></div>
                <div class="kv"><span>Owner</span><strong>{escape(str(prof["def_owner_type"]).title())}</strong></div>
                <div class="kv"><span>Drivetrain</span><strong>{escape(str(prof["def_drive_type"]).upper())}</strong></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # specs card
        spec_rows = [
            ("Mileage", f"{float(prof['mileage']):.1f} kmpl"),
            ("Engine", f"{num(prof['engine_cc'])} cc"),
            ("Power", f"{float(prof['max_power']):.1f} bhp"),
            ("Torque", f"{float(prof['max_torque']):.1f} Nm"),
            ("L × W × H", f"{num(prof['length'])} × {num(prof['width'])} × {num(prof['height'])} mm"),
            ("Seats", num(prof["seats"])),
        ]
        spec_html = "".join(
            f'<div class="sr"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>'
            for k, v in spec_rows
        )
        st.markdown(f"""
        <div class="gc">
            <p class="lbl">Locked specifications</p>
            <div class="spl">{spec_html}</div>
        </div>""", unsafe_allow_html=True)

    # ── Result section moved entirely to Modal ──

    # ── Market intelligence ──
    st.markdown("<div style='height:3.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Comparable listings</div>', unsafe_allow_html=True)

    preview = sub[
        ["car_name","model_year","km_driven","fuel_type","transmission_type","seller_type","price"]
    ].head(8).copy()
    preview["price"] = preview["price"].map(inr)
    preview["km_driven"] = preview["km_driven"].astype(int).map(lambda v: f"{v:,} km")
    preview["car_name"] = preview["car_name"].astype(str).str.title()
    preview.columns = ["Car","Year","Km","Fuel","Trans.","Seller","Price"]
    st.dataframe(preview, width="stretch", hide_index=True)

    # ── Footer ──
    st.markdown(f"""
    <div class="dv-ft">
        <p><strong>drivevalue</strong> · {rows_count:,} listings · {df["brand"].nunique()} brands
        · {df["model"].nunique()} models · data from CarDekho</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
