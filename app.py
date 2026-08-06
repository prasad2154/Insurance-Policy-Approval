"""
Insurance Policy Approval Predictor
====================================
A modern, professional Streamlit web application that predicts
whether an insurance policy will be approved or rejected based on
applicant attributes. Uses a pre-trained scikit-learn model loaded
from Insurance.pkl.

Author : AI Course – G_38
Stack  : Streamlit · Scikit-Learn · Plotly · Pandas · NumPy
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime
import os

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Approval Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS – Glassmorphism, Animations & Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ──────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root Variables ───────────────────────── */
:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --glass-bg: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-cyan: #00d4ff;
    --accent-purple: #7b2ff7;
    --accent-pink: #ff2d95;
    --accent-green: #00e676;
    --accent-red: #ff1744;
    --text-primary: #f0f0f5;
    --text-secondary: #9e9eb8;
    --gradient-1: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff2d95 100%);
    --gradient-2: linear-gradient(135deg, #7b2ff7 0%, #ff2d95 100%);
    --gradient-3: linear-gradient(135deg, #00d4ff 0%, #00e676 100%);
}

/* ── Global Styles ────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0a14 0%, #0f0f1a 30%, #1a1a2e 60%, #16213e 100%) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #131328 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-cyan) !important;
}

/* ── Glass Card ───────────────────────────── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.12);
}

/* ── Hero Section ─────────────────────────── */
.hero-section {
    background: linear-gradient(135deg,
        rgba(123, 47, 247, 0.15) 0%,
        rgba(0, 212, 255, 0.10) 50%,
        rgba(255, 45, 149, 0.10) 100%);
    backdrop-filter: blur(30px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 28px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeInDown 0.8s ease;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%,
        rgba(0, 212, 255, 0.06) 0%,
        transparent 50%),
        radial-gradient(circle at 70% 50%,
        rgba(123, 47, 247, 0.06) 0%,
        transparent 50%);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}

@keyframes rotate { to { transform: rotate(360deg); } }

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    position: relative;
}

.hero-sub {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 620px;
    margin: 0 auto;
    position: relative;
}

/* ── Status Badge ─────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 1rem;
    position: relative;
}

.status-online {
    background: rgba(0, 230, 118, 0.12);
    color: #00e676;
    border: 1px solid rgba(0, 230, 118, 0.25);
}

.status-offline {
    background: rgba(255, 23, 68, 0.12);
    color: #ff1744;
    border: 1px solid rgba(255, 23, 68, 0.25);
}

/* ── Pulse animation for status dot ───────── */
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
}

.pulse-green {
    background: #00e676;
    box-shadow: 0 0 6px #00e676;
}

.pulse-red {
    background: #ff1744;
    box-shadow: 0 0 6px #ff1744;
}

@keyframes pulse {
    0%   { opacity: 1; transform: scale(1); }
    50%  { opacity: 0.5; transform: scale(1.4); }
    100% { opacity: 1; transform: scale(1); }
}

/* ── Result Cards ─────────────────────────── */
.result-approved {
    background: linear-gradient(135deg,
        rgba(0, 230, 118, 0.08) 0%,
        rgba(0, 212, 255, 0.06) 100%);
    border: 1px solid rgba(0, 230, 118, 0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: fadeInUp 0.6s ease;
}

.result-rejected {
    background: linear-gradient(135deg,
        rgba(255, 23, 68, 0.08) 0%,
        rgba(255, 45, 149, 0.06) 100%);
    border: 1px solid rgba(255, 23, 68, 0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: fadeInUp 0.6s ease;
}

.result-icon {
    font-size: 4rem;
    margin-bottom: 0.5rem;
}

.result-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.approved-text { color: #00e676; }
.rejected-text { color: #ff1744; }

/* ── Metric Cards ─────────────────────────── */
.metric-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section Headings ─────────────────────── */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title .icon {
    font-size: 1.3rem;
}

/* ── Insight Cards ────────────────────────── */
.insight-item {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: transform 0.2s ease, border-color 0.3s ease;
}

.insight-item:hover {
    transform: translateX(6px);
    border-color: rgba(0, 212, 255, 0.3);
}

.insight-icon {
    font-size: 1.4rem;
    flex-shrink: 0;
}

.insight-text {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Footer ───────────────────────────────── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    margin-top: 3rem;
    border-top: 1px solid var(--glass-border);
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.footer a {
    color: var(--accent-cyan);
    text-decoration: none;
}

/* ── Animations ───────────────────────────── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── Streamlit Widget Overrides ───────────── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

.stNumberInput input,
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0f0f5 !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 0.5rem;
}

/* Gradient Button */
.stButton > button {
    background: var(--gradient-2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(123, 47, 247, 0.35) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #00e676 100%) !important;
    color: #0f0f1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(123, 47, 247, 0.15) !important;
    color: var(--accent-cyan) !important;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--glass-bg) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model Loading (Cached)
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Insurance.pkl")


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from Insurance.pkl with error handling."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        st.error(f"⚠️ Model file not found at `{MODEL_PATH}`")
        return None, False
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, False


model, model_loaded = load_model()

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "approved_count" not in st.session_state:
    st.session_state.approved_count = 0

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def get_risk_level(probability: float) -> tuple:
    """Return (label, color) based on approval probability."""
    if probability >= 0.75:
        return "Low Risk", "#00e676"
    elif probability >= 0.50:
        return "Moderate Risk", "#ffab00"
    elif probability >= 0.30:
        return "High Risk", "#ff6d00"
    else:
        return "Very High Risk", "#ff1744"


def create_gauge_chart(probability: float, title: str = "Approval Probability") -> go.Figure:
    """Create a sleek gauge chart for approval probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#f0f0f5", "family": "Inter"}},
        title={"text": title, "font": {"size": 16, "color": "#9e9eb8", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2a2a4a",
                     "tickfont": {"color": "#9e9eb8"}},
            "bar": {"color": "#7b2ff7", "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(255,23,68,0.15)"},
                {"range": [30, 50], "color": "rgba(255,109,0,0.12)"},
                {"range": [50, 75], "color": "rgba(255,171,0,0.10)"},
                {"range": [75, 100], "color": "rgba(0,230,118,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#00d4ff", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        font={"family": "Inter"},
    )
    return fig


def create_risk_meter(probability: float) -> go.Figure:
    """Create a horizontal risk meter bar chart."""
    risk_level, risk_color = get_risk_level(probability)
    risk_score = (1 - probability) * 100  # invert: higher = riskier

    fig = go.Figure(go.Bar(
        x=[risk_score],
        y=["Risk"],
        orientation="h",
        marker=dict(
            color=risk_color,
            line=dict(width=0),
            cornerradius=8,
        ),
        text=f"{risk_level} ({risk_score:.0f}%)",
        textposition="inside",
        textfont=dict(color="white", size=14, family="Inter"),
    ))
    fig.add_shape(
        type="rect", x0=0, x1=100, y0=-0.4, y1=0.4,
        fillcolor="rgba(255,255,255,0.03)", line=dict(width=0),
        layer="below",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False,
                    showticklabels=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=100,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def create_feature_chart(input_dict: dict) -> go.Figure:
    """Create a radar-style bar chart showing normalised feature contributions."""
    # Normalise features to 0-1 for visualisation
    norm_map = {
        "Age": (input_dict["Age"], 18, 90),
        "BMI": (input_dict["BMI"], 10, 50),
        "Annual Income": (input_dict["AnnualIncome"], 10000, 200000),
        "Prev Claims": (input_dict["PreviousClaims"], 0, 10),
        "Credit Score": (input_dict["CreditScore"], 300, 900),
        "Gender": (input_dict["Gender_Male"], 0, 1),
        "Smoker": (input_dict["Smoker_Yes"], 0, 1),
        "Diabetic": (input_dict["Diabetic_Yes"], 0, 1),
    }

    labels, values, colors = [], [], []
    palette = ["#00d4ff", "#7b2ff7", "#ff2d95", "#00e676",
               "#ffab00", "#ff6d00", "#64ffda", "#e040fb", "#40c4ff"]

    for i, (label, (val, lo, hi)) in enumerate(norm_map.items()):
        norm = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
        labels.append(label)
        values.append(round(norm * 100, 1))
        colors.append(palette[i % len(palette)])

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{v:.0f}%" for v in values],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False,
                    title="Relative Scale (%)",
                    title_font=dict(color="#9e9eb8", size=12),
                    tickfont=dict(color="#9e9eb8")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#f0f0f5", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=110, r=30, t=20, b=40),
    )
    return fig


def create_history_chart(history: list) -> go.Figure:
    """Create a line+scatter chart of prediction history."""
    if not history:
        return None
    df = pd.DataFrame(history)
    colors = ["#00e676" if p == "Approved" else "#ff1744" for p in df["Prediction"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["Probability"],
        mode="lines+markers",
        line=dict(color="#7b2ff7", width=2, shape="spline"),
        marker=dict(size=10, color=colors, line=dict(width=2, color="#1a1a2e")),
        text=df["Prediction"],
        hovertemplate="<b>Prediction %{x}</b><br>Probability: %{y:.1%}<br>Result: %{text}<extra></extra>",
    ))
    # 50 % decision line
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,171,0,0.5)",
                  annotation_text="50% Threshold", annotation_font_color="#ffab00")
    fig.update_layout(
        xaxis=dict(title="Prediction #", tickfont=dict(color="#9e9eb8"),
                    title_font=dict(color="#9e9eb8"), showgrid=False),
        yaxis=dict(title="Approval Probability", range=[0, 1],
                   tickformat=".0%", tickfont=dict(color="#9e9eb8"),
                   title_font=dict(color="#9e9eb8"), showgrid=True,
                   gridcolor="rgba(255,255,255,0.04)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=60, r=20, t=20, b=50),
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Insurance Predictor")
    st.markdown("---")

    # Model status
    if model_loaded:
        st.markdown('<div class="status-badge status-online">'
                    '<span class="pulse-dot pulse-green"></span> Model Loaded</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-offline">'
                    '<span class="pulse-dot pulse-red"></span> Model Not Found</div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # Feature descriptions
    with st.expander("📋 Feature Descriptions", expanded=False):
        st.markdown("""
        | Feature | Description |
        |:--------|:-----------|
        | **Age** | Applicant's age in years |
        | **Gender** | Male or Female |
        | **BMI** | Body Mass Index |
        | **Smoker** | Smoking status |
        | **Diabetic** | Diabetes status |
        | **Annual Income** | Yearly income (₹) |

        | **Previous Claims** | Past insurance claims |
        | **Credit Score** | Financial credit score |
        """)

    st.markdown("---")

    # Prediction statistics
    st.markdown("### 📊 Session Statistics")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Total", st.session_state.prediction_count)
    col_s2.metric("Approved", st.session_state.approved_count)

    if st.session_state.prediction_count > 0:
        approval_rate = st.session_state.approved_count / st.session_state.prediction_count * 100
        st.progress(int(approval_rate), text=f"Approval Rate: {approval_rate:.1f}%")
    else:
        st.progress(0, text="Approval Rate: N/A")

    st.markdown("---")

    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.prediction_count = 0
        st.session_state.approved_count = 0
        st.rerun()

    st.markdown(
        "<br><p style='text-align:center;color:#555;font-size:0.75rem;'>"
        "v1.0.0 • Built with ❤️</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Hero Section
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div style="font-size:3.5rem; margin-bottom:0.5rem;">🛡️</div>
    <div class="hero-title">Insurance Policy Approval Predictor</div>
    <p class="hero-sub">
        Leverage machine learning to instantly predict whether an insurance
        policy application will be <strong>approved</strong> or
        <strong>rejected</strong> based on applicant profile data.
    </p>
    {} 
</div>
""".format(
    '<div class="status-badge status-online" style="margin-top:1.2rem;">'
    '<span class="pulse-dot pulse-green"></span> Model Online – Ready for Predictions</div>'
    if model_loaded else
    '<div class="status-badge status-offline" style="margin-top:1.2rem;">'
    '<span class="pulse-dot pulse-red"></span> Model Offline – Check Insurance.pkl</div>'
), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────
st.markdown('<div class="section-title"><span class="icon">📝</span> Applicant Details</div>',
            unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1,
                              help="Applicant's age in years")
        gender = st.selectbox("👤 Gender", ["Male", "Female"],
                              help="Applicant's gender")
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0,
                              step=0.1, format="%.1f",
                              help="Body Mass Index")

    with col2:
        smoker = st.selectbox("🚬 Smoker", ["No", "Yes"],
                              help="Does the applicant smoke?")
        diabetic = st.selectbox("💉 Diabetic", ["No", "Yes"],
                                help="Is the applicant diabetic?")
        annual_income = st.number_input("💰 Annual Income (₹)", min_value=10000,
                                        max_value=500000, value=50000, step=5000,
                                        help="Yearly income in Rupees")

    with col3:
        previous_claims = st.number_input("📑 Previous Claims", min_value=0, max_value=20,
                                           value=0, step=1,
                                           help="Number of prior insurance claims")
        credit_score = st.slider("📈 Credit Score", min_value=300, max_value=900,
                                  value=650, step=10,
                                  help="Financial credit score (300-900)")

vehicle_age = 3  # Fixed default since vehicle age input is removed from the form

# ──────────────────────────────────────────────
# Encode categorical inputs
# ──────────────────────────────────────────────
# Encode categoricals to match one-hot encoded column names from training
gender_enc = 1 if gender == "Male" else 0
smoker_enc = 1 if smoker == "Yes" else 0
diabetic_enc = 1 if diabetic == "Yes" else 0

input_dict = {
    "Age": age,
    "BMI": bmi,
    "AnnualIncome": annual_income,
    "VehicleAge": vehicle_age,
    "PreviousClaims": previous_claims,
    "CreditScore": credit_score,
    "Gender_Male": gender_enc,
    "Smoker_Yes": smoker_enc,
    "Diabetic_Yes": diabetic_enc,
}

# Feature columns must match training order exactly:
# X = ['Age', 'BMI', 'AnnualIncome', 'VehicleAge', 'PreviousClaims',
#      'CreditScore', 'Gender_Male', 'Smoker_Yes', 'Diabetic_Yes']
# y = ['PolicyApproved']
FEATURE_COLUMNS = ['Age', 'BMI', 'AnnualIncome', 'VehicleAge', 'PreviousClaims',
                   'CreditScore', 'Gender_Male', 'Smoker_Yes', 'Diabetic_Yes']
input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)

# ──────────────────────────────────────────────
# Model Input Preview
# ──────────────────────────────────────────────
with st.expander("🔍 Model Input Preview", expanded=False):
    st.dataframe(input_df, use_container_width=True)

# ──────────────────────────────────────────────
# Predict Button
# ──────────────────────────────────────────────
st.markdown("")
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
with predict_col2:
    predict_btn = st.button("🚀 Predict Approval", use_container_width=True)

# ──────────────────────────────────────────────
# Prediction Logic & Results
# ──────────────────────────────────────────────
if predict_btn:
    if not model_loaded:
        st.error("❌ Cannot predict – model is not loaded. Please ensure `Insurance.pkl` exists.")
    else:
        with st.spinner("🔮 Analysing application…"):
            try:
                prediction = model.predict(input_df)[0]

                # Attempt to get probability
                has_proba = hasattr(model, "predict_proba")
                if has_proba:
                    proba = model.predict_proba(input_df)[0]
                    prob_approved = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    prob_approved = 1.0 if prediction == 1 else 0.0

                is_approved = int(prediction) == 1
                risk_level, risk_color = get_risk_level(prob_approved)
                confidence = max(prob_approved, 1 - prob_approved)

                # Update session state
                st.session_state.prediction_count += 1
                if is_approved:
                    st.session_state.approved_count += 1

                # Save to history
                st.session_state.history.append({
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age,
                    "Income": annual_income,
                    "Credit Score": credit_score,
                    "Prediction": "Approved" if is_approved else "Rejected",
                    "Probability": round(prob_approved, 4),
                })

                # ── Result Banner ─────────────────
                st.markdown("")
                if is_approved:
                    st.markdown("""
                    <div class="result-approved">
                        <div class="result-icon">🟢</div>
                        <div class="result-title approved-text">Policy Approved ✅</div>
                        <p style="color:#9e9eb8;">
                            Congratulations! The model predicts this application will be <strong>approved</strong>.
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-rejected">
                        <div class="result-icon">🔴</div>
                        <div class="result-title rejected-text">Policy Rejected ❌</div>
                        <p style="color:#9e9eb8;">
                            Unfortunately, the model predicts this application will be <strong>rejected</strong>.
                        </p>
                    </div>""", unsafe_allow_html=True)

                # ── Metric Cards ──────────────────
                st.markdown("")
                m1, m2, m3, m4 = st.columns(4, gap="medium")

                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{"✅ Approved" if is_approved else "❌ Rejected"}</div>
                        <div class="metric-label">Approval Status</div>
                    </div>""", unsafe_allow_html=True)

                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prob_approved:.1%}</div>
                        <div class="metric-label">Approval Probability</div>
                    </div>""", unsafe_allow_html=True)

                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="-webkit-text-fill-color:{risk_color};color:{risk_color};">
                            {risk_level}
                        </div>
                        <div class="metric-label">Risk Level</div>
                    </div>""", unsafe_allow_html=True)

                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence:.1%}</div>
                        <div class="metric-label">Confidence Score</div>
                    </div>""", unsafe_allow_html=True)

                # ── Charts ────────────────────────
                st.markdown("")
                st.markdown('<div class="section-title"><span class="icon">📊</span> Visual Analytics</div>',
                            unsafe_allow_html=True)

                chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
                    ["🎯 Approval Gauge", "⚠️ Risk Meter",
                     "📊 Feature Contributions", "📈 Prediction History"])

                with chart_tab1:
                    st.plotly_chart(create_gauge_chart(prob_approved), use_container_width=True)

                with chart_tab2:
                    st.plotly_chart(create_risk_meter(prob_approved), use_container_width=True)
                    st.markdown(
                        f"<p style='text-align:center;color:{risk_color};font-weight:600;'>"
                        f"Risk Assessment: {risk_level}</p>",
                        unsafe_allow_html=True)

                with chart_tab3:
                    st.plotly_chart(create_feature_chart(input_dict), use_container_width=True)

                with chart_tab4:
                    hist_fig = create_history_chart(st.session_state.history)
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        st.info("No history yet. Make predictions to see the trend.")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

# ──────────────────────────────────────────────
# Prediction History Section
# ──────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-title"><span class="icon">🕒</span> Prediction History</div>',
                unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    # CSV download
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="insurance_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# Insights Section
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title"><span class="icon">💡</span> Approval Insights</div>',
            unsafe_allow_html=True)

insights = [
    ("📈", "A <strong>higher Credit Score</strong> (700+) significantly increases the probability of policy approval."),
    ("💰", "A <strong>higher Annual Income</strong> demonstrates financial stability and improves approval chances."),
    ("📑", "<strong>Previous Claims</strong> negatively impact approval – fewer claims lead to better outcomes."),
    ("🚬", "<strong>Smoking</strong> is a major risk factor that can reduce the likelihood of approval."),
    ("💉", "Being <strong>Diabetic</strong> increases the perceived health risk and may reduce approval probability."),

    ("⚖️", "A <strong>healthy BMI</strong> (18.5 – 24.9) is associated with lower health risk and better chances."),
    ("👤", "Both <strong>Age</strong> and <strong>Gender</strong> may influence risk assessment models."),
]

ins_col1, ins_col2 = st.columns(2, gap="medium")

for idx, (icon, text) in enumerate(insights):
    target = ins_col1 if idx % 2 == 0 else ins_col2
    target.markdown(
        f'<div class="insight-item">'
        f'<span class="insight-icon">{icon}</span>'
        f'<span class="insight-text">{text}</span></div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🛡️ <strong>Insurance Policy Approval Predictor</strong><br>
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>,
    <a href="https://scikit-learn.org" target="_blank">Scikit-Learn</a> &amp;
    <a href="https://plotly.com/python/" target="_blank">Plotly</a><br><br>
    <span style="font-size:0.78rem;">© 2026 • All Rights Reserved</span>
</div>
""", unsafe_allow_html=True)
