import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from black_scholes import (
    black_scholes_price,
    bs_greeks,
    numerical_greeks,
    crr_binomial_tree,
    implied_volatility,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricing & Greeks Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom Theme CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(160deg, #0a0e17 0%, #0f1523 40%, #111b2e 100%);
}

/* ── Sidebar ───────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111927 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] {
    color: #e2e8f0 !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    color: #38bdf8 !important;
}

section[data-testid="stSidebar"] .stRadio label span {
    color: #cbd5e1 !important;
}

/* ── Header ────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
}

.main-header h1 {
    color: #f1f5f9;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
}

.main-header p {
    color: #64748b;
    font-size: 0.92rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section Headers ───────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 1.8rem 0 1rem 0;
}

.section-header .badge {
    background: linear-gradient(135deg, #1e3a5f, #1e40af);
    color: #93c5fd;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.section-header h3 {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.01em;
}

/* ── Metric Cards ──────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.6) 100%);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.08);
}

.metric-card .label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.55rem;
}

.metric-card .value {
    color: #f1f5f9;
    font-size: 1.65rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}

.metric-card .delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}

.delta-positive { color: #4ade80; }
.delta-negative { color: #f87171; }
.delta-neutral  { color: #94a3b8; }

/* ── Accent Borders for Metric Cards ───── */
.metric-card.accent-blue::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #38bdf8, #0ea5e9);
}
.metric-card.accent-purple::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #a78bfa, #7c3aed);
}
.metric-card.accent-emerald::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #34d399, #10b981);
}
.metric-card.accent-amber::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #fbbf24, #f59e0b);
}
.metric-card.accent-rose::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #fb7185, #e11d48);
}

/* ── Greek Chip Cards ──────────────────── */
.greek-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.5) 100%);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.greek-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.08);
}

.greek-card .symbol {
    font-size: 1.3rem;
    margin-bottom: 0.3rem;
}

.greek-card .greek-name {
    color: #94a3b8;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}

.greek-card .greek-value {
    color: #e2e8f0;
    font-size: 1.35rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cross-Check Table ─────────────────── */
.xcheck-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.1);
    margin: 0.5rem 0;
}

.xcheck-table th {
    background: rgba(15, 23, 42, 0.9);
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.85rem 1.2rem;
    text-align: left;
    border-bottom: 1px solid rgba(56, 189, 248, 0.08);
}

.xcheck-table td {
    background: rgba(15, 23, 42, 0.5);
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    padding: 0.75rem 1.2rem;
    border-bottom: 1px solid rgba(56, 189, 248, 0.05);
}

.xcheck-table tr:last-child td {
    border-bottom: none;
}

.xcheck-table .match { color: #4ade80; }
.xcheck-table .mismatch { color: #fbbf24; }

/* ── IV Solver Section ─────────────────── */
.iv-result {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.iv-result .iv-label {
    color: #6ee7b7;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.iv-result .iv-value {
    color: #f1f5f9;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.iv-error {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.1) 0%, rgba(244, 63, 94, 0.05) 100%);
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #fda4af;
    font-size: 0.9rem;
}

/* ── Expander ──────────────────────────── */
.stExpander {
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.5) !important;
}

.stExpander summary {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ── Buttons ───────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35) !important;
}

/* ── Input Fields ──────────────────────── */
.stNumberInput input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Footer ────────────────────────────── */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(56, 189, 248, 0.06);
    margin-top: 3rem;
}

/* ── Hide default Streamlit ────────────── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Plotly theme constants
# ─────────────────────────────────────────────
PLOTLY_BG = "rgba(10, 14, 23, 0)"
PLOTLY_PAPER = "rgba(15, 23, 42, 0.6)"
PLOTLY_GRID = "rgba(56, 189, 248, 0.06)"
PLOTLY_TEXT = "#94a3b8"
PLOTLY_TITLE = "#e2e8f0"

PALETTE = {
    "blue": "#38bdf8",
    "indigo": "#818cf8",
    "emerald": "#34d399",
    "amber": "#fbbf24",
    "rose": "#fb7185",
    "purple": "#c084fc",
    "cyan": "#22d3ee",
}


def plotly_layout_defaults(**overrides):
    """Shared Plotly layout for consistent dark theme."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG,
        font=dict(family="Inter, sans-serif", color=PLOTLY_TEXT, size=12),
        title_font=dict(color=PLOTLY_TITLE, size=15, family="Inter, sans-serif"),
        margin=dict(l=55, r=30, t=50, b=45),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#94a3b8"),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="JetBrains Mono, monospace",
            font_color="#e2e8f0",
            bordercolor="rgba(56,189,248,0.2)",
        ),
    )
    base.update(overrides)
    return base


# ─────────────────────────────────────────────
# Sidebar – Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Option Parameters")
    st.markdown("---")

    spot = st.number_input(
        "Spot Price (S)", min_value=1.0, value=100.0, step=1.0, format="%.2f"
    )
    strike = st.number_input(
        "Strike Price (K)", min_value=1.0, value=100.0, step=1.0, format="%.2f"
    )
    st.markdown("### ⏱️ Time Horizon")
    time_mode = st.radio(
        "Time Unit",
        ["Days (Swing/Monthly)", "Hours (0DTE / Day Trade)", "Years (Quant Mode)"],
        horizontal=False,
    )

    if time_mode == "Days (Swing/Monthly)":
        days_input = st.number_input(
            "Days to Expiry (DTE)",
            min_value=0.1,
            max_value=730.0,
            value=41.0,
            step=1.0,
        )
        tenor = max(days_input / 365.0, 1e-5)
        st.caption(f"Equivalent T = {tenor:.4f} years")

    elif time_mode == "Hours (0DTE / Day Trade)":
        hours_input = st.slider(
            "Hours Remaining Today",
            min_value=0.25,
            max_value=6.5,
            value=2.0,
            step=0.25,
        )
        # 252 trading days × 6.5 trading hours/day = 1638 trading hours/year
        tenor = max(hours_input / (252.0 * 6.5), 1e-5)
        st.caption(f"Intraday T = {tenor:.6f} years ({hours_input} hrs left)")

    else:
        tenor = st.slider(
            "Time to Expiry — T (years)",
            min_value=0.001,
            max_value=2.0,
            value=0.11,
            step=0.01,
        )

    st.markdown("---")
    rate = st.slider(
        "Risk-Free Rate — r",
        min_value=0.0,
        max_value=0.15,
        value=0.05,
        step=0.005,
        format="%.3f",
    )
    vol = st.slider(
        "Volatility — σ",
        min_value=0.01,
        max_value=1.5,
        value=0.20,
        step=0.01,
        format="%.2f",
    )

    st.markdown("---")
    option_type = st.radio(
        "Option Type", options=["Call", "Put"], horizontal=True
    ).lower()

    # Moneyness indicator
    moneyness_ratio = spot / strike
    if option_type == "call":
        moneyness = "ITM" if spot > strike else ("ATM" if spot == strike else "OTM")
    else:
        moneyness = "ITM" if spot < strike else ("ATM" if spot == strike else "OTM")

    moneyness_colors = {"ITM": "#4ade80", "ATM": "#fbbf24", "OTM": "#f87171"}
    st.markdown(
        f"""
    <div style="
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(56,189,248,0.1);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        text-align: center;
    ">
        <span style="color:#64748b; font-size:0.72rem; font-weight:600;
            text-transform:uppercase; letter-spacing:0.06em;">Moneyness</span><br>
        <span style="color:{moneyness_colors[moneyness]}; font-size:1.1rem;
            font-weight:700; font-family:'JetBrains Mono',monospace;">
            {moneyness}
        </span>
        <span style="color:#64748b; font-size:0.8rem;"> ({moneyness_ratio:.2f})</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────
bs_px = black_scholes_price(spot, strike, tenor, rate, vol, option_type)
greeks = bs_greeks(spot, strike, tenor, rate, vol, option_type)
num_gk = numerical_greeks(spot, strike, tenor, rate, vol, option_type)
crr_px = crr_binomial_tree(spot, strike, tenor, rate, vol, N=200, option_type=option_type)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>📊 Options Pricing & Greeks Toolkit</h1>
    <p>Black-Scholes analytical engine · CRR binomial tree · Interactive Greek surfaces · IV solver</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 1. Pricing Summary
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="section-header">
    <span class="badge">Pricing</span>
    <h3>Pricing Summary</h3>
</div>
""",
    unsafe_allow_html=True,
)

price_diff = crr_px - bs_px
diff_sign = "+" if price_diff >= 0 else ""
diff_class = "delta-positive" if abs(price_diff) < 0.005 else "delta-neutral"

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
    <div class="metric-card accent-blue">
        <div class="label">Black-Scholes Price</div>
        <div class="value">${bs_px:.4f}</div>
        <div class="delta delta-neutral">Analytical closed-form</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card accent-purple">
        <div class="label">CRR Binomial Tree (N=200)</div>
        <div class="value">${crr_px:.4f}</div>
        <div class="delta {diff_class}">Δ {diff_sign}{price_diff:.4f} vs BS</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    type_color = PALETTE["cyan"] if option_type == "call" else PALETTE["rose"]
    st.markdown(
        f"""
    <div class="metric-card accent-emerald">
        <div class="label">Option Type</div>
        <div class="value" style="color: {type_color};">{option_type.upper()}</div>
        <div class="delta delta-neutral">S={spot:.0f} K={strike:.0f} T={tenor:.2f}y</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# 2. Greeks Dashboard
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="section-header">
    <span class="badge">Risk</span>
    <h3>Greeks Dashboard</h3>
</div>
""",
    unsafe_allow_html=True,
)

greek_info = [
    ("Δ", "Delta", greeks["delta"], PALETTE["blue"]),
    ("Γ", "Gamma", greeks["gamma"], PALETTE["emerald"]),
    ("Θ", "Theta /yr", greeks["theta"], PALETTE["amber"]),
    ("𝒱", "Vega", greeks["vega"], PALETTE["purple"]),
    ("ρ", "Rho", greeks["rho"], PALETTE["rose"]),
]

gcols = st.columns(5)
for col, (symbol, name, val, color) in zip(gcols, greek_info):
    sign = "+" if val >= 0 else ""
    with col:
        st.markdown(
            f"""
        <div class="greek-card">
            <div class="symbol" style="color: {color};">{symbol}</div>
            <div class="greek-name">{name}</div>
            <div class="greek-value">{sign}{val:.4f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Cross-Check Table
# ─────────────────────────────────────────────
with st.expander("📋 Analytical vs. Numerical Greeks Cross-Check"):
    cross_data = [
        ("Delta", greeks["delta"], num_gk["num_delta"]),
        ("Gamma", greeks["gamma"], num_gk["num_gamma"]),
        ("Vega", greeks["vega"], num_gk["num_vega"]),
    ]

    rows_html = ""
    for name, analytical, numerical in cross_data:
        diff = abs(analytical - numerical)
        css_class = "match" if diff < 0.005 else "mismatch"
        rows_html += f"""
        <tr>
            <td style="color:#cbd5e1; font-weight:500;">{name}</td>
            <td>{analytical:.6f}</td>
            <td>{numerical:.6f}</td>
            <td class="{css_class}">{"✓ match" if diff < 0.005 else f"Δ {diff:.6f}"}</td>
        </tr>
        """

    st.markdown(
        f"""
    <table class="xcheck-table">
        <thead>
            <tr>
                <th>Greek</th>
                <th>Analytical (BS)</th>
                <th>Numerical (FD)</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# 3. Interactive Sensitivity Curve
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="section-header">
    <span class="badge">Analytics</span>
    <h3>Dynamic Sensitivity Curve</h3>
</div>
""",
    unsafe_allow_html=True,
)

spot_range = [spot * (0.6 + i * 0.005) for i in range(161)]
prices = [black_scholes_price(s, strike, tenor, rate, vol, option_type) for s in spot_range]
deltas = [bs_greeks(s, strike, tenor, rate, vol, option_type)["delta"] for s in spot_range]

fig_sens = make_subplots(specs=[[{"secondary_y": True}]])

fig_sens.add_trace(
    go.Scatter(
        x=spot_range,
        y=prices,
        name="Option Price ($)",
        line=dict(color=PALETTE["blue"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.06)",
        hovertemplate="S = $%{x:.2f}<br>Price = $%{y:.4f}<extra></extra>",
    ),
    secondary_y=False,
)

fig_sens.add_trace(
    go.Scatter(
        x=spot_range,
        y=deltas,
        name="Delta",
        line=dict(color=PALETTE["emerald"], width=2, dash="dot"),
        hovertemplate="S = $%{x:.2f}<br>Δ = %{y:.4f}<extra></extra>",
    ),
    secondary_y=True,
)

# Strike line
fig_sens.add_vline(
    x=strike,
    line_dash="dash",
    line_color="rgba(148, 163, 184, 0.4)",
    annotation_text=f"K = ${strike:.0f}",
    annotation_font_color="#94a3b8",
    annotation_font_size=11,
)

# Current spot marker
fig_sens.add_trace(
    go.Scatter(
        x=[spot],
        y=[bs_px],
        mode="markers",
        name="Current",
        marker=dict(color=PALETTE["amber"], size=10, symbol="diamond",
                    line=dict(color="#0a0e17", width=2)),
        hovertemplate="Current Spot = $%{x:.2f}<br>Price = $%{y:.4f}<extra></extra>",
    ),
    secondary_y=False,
)

fig_sens.update_layout(
    **plotly_layout_defaults(
        title="Price & Delta vs Spot Price",
        height=420,
        yaxis_title="Option Price ($)",
        yaxis2_title="Delta",
    )
)
fig_sens.update_yaxes(title_font_color=PALETTE["blue"], secondary_y=False)
fig_sens.update_yaxes(title_font_color=PALETTE["emerald"], secondary_y=True)
fig_sens.update_xaxes(title_text="Spot Price ($)")

st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────────
# 4. Greeks Sensitivity Surface (multi-panel)
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="section-header">
    <span class="badge">Surface</span>
    <h3>Greeks Sensitivity Surfaces</h3>
</div>
""",
    unsafe_allow_html=True,
)

spots_surface = [60.0 + i * 0.5 for i in range(161)]
tenors_surface = [0.1, 0.5, 1.0]
tenor_labels = {0.1: "T = 0.1y", 0.5: "T = 0.5y", 1.0: "T = 1.0y"}
tenor_colors = {0.1: PALETTE["rose"], 0.5: PALETTE["amber"], 1.0: PALETTE["blue"]}

fig_greeks = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Call Delta vs Spot", "Gamma vs Spot (ATM Clustering)",
                    "Vega vs Spot", "Theta Decay vs Time to Expiry"),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

for T_val in tenors_surface:
    delta_vals = [bs_greeks(S, strike, T_val, rate, vol, "call")["delta"] for S in spots_surface]
    gamma_vals = [bs_greeks(S, strike, T_val, rate, vol, "call")["gamma"] for S in spots_surface]
    vega_vals = [bs_greeks(S, strike, T_val, rate, vol, "call")["vega"] for S in spots_surface]

    common = dict(line=dict(color=tenor_colors[T_val], width=2))

    fig_greeks.add_trace(
        go.Scatter(x=spots_surface, y=delta_vals, name=tenor_labels[T_val],
                   legendgroup=tenor_labels[T_val], showlegend=True, **common),
        row=1, col=1,
    )
    fig_greeks.add_trace(
        go.Scatter(x=spots_surface, y=gamma_vals, name=tenor_labels[T_val],
                   legendgroup=tenor_labels[T_val], showlegend=False, **common),
        row=1, col=2,
    )
    fig_greeks.add_trace(
        go.Scatter(x=spots_surface, y=vega_vals, name=tenor_labels[T_val],
                   legendgroup=tenor_labels[T_val], showlegend=False, **common),
        row=2, col=1,
    )

# Theta decay subplot
tenor_steps = [0.01 + i * 0.01 for i in range(100)]
moneyness_spots = [90.0, 100.0, 110.0]
mn_labels = {90.0: "OTM (S=90)", 100.0: "ATM (S=100)", 110.0: "ITM (S=110)"}
mn_colors = {90.0: PALETTE["rose"], 100.0: PALETTE["amber"], 110.0: PALETTE["emerald"]}

for S_val in moneyness_spots:
    theta_vals = [bs_greeks(S_val, strike, t, rate, vol, "call")["theta"] for t in tenor_steps]
    fig_greeks.add_trace(
        go.Scatter(x=tenor_steps, y=theta_vals, name=mn_labels[S_val],
                   legendgroup=mn_labels[S_val],
                   line=dict(color=mn_colors[S_val], width=2)),
        row=2, col=2,
    )

# Strike reference lines on spot-based subplots
for row, col in [(1, 1), (1, 2), (2, 1)]:
    fig_greeks.add_vline(
        x=strike, row=row, col=col,
        line_dash="dash", line_color="rgba(148,163,184,0.25)",
    )

fig_greeks.update_layout(
    **plotly_layout_defaults(
        height=620,
        title=None,
    )
)

# Style all subplot axes
for i in range(1, 5):
    fig_greeks.update_xaxes(gridcolor=PLOTLY_GRID, row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
    fig_greeks.update_yaxes(gridcolor=PLOTLY_GRID, row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

# Annotations styling
for annotation in fig_greeks.layout.annotations:
    annotation.font = dict(size=12, color=PLOTLY_TITLE, family="Inter, sans-serif")

st.plotly_chart(fig_greeks, use_container_width=True)

# ─────────────────────────────────────────────
# 5. Implied Volatility Solver
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="section-header">
    <span class="badge">Tool</span>
    <h3>Implied Volatility Solver</h3>
</div>
""",
    unsafe_allow_html=True,
)

iv_col1, iv_col2 = st.columns([2, 1])

with iv_col1:
    mkt_price_input = st.number_input(
        "Observed Option Price ($)",
        min_value=0.0001,
        value=max(float(round(bs_px, 4)), 0.0001),
        step=0.01,
        format="%.4f",
    )

with iv_col2:
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    solve_btn = st.button("🔍  Solve IV", use_container_width=True)

if solve_btn:
    try:
        solved_iv = implied_volatility(mkt_price_input, spot, strike, tenor, rate, option_type)
        diff_iv = solved_iv - vol
        diff_sign_iv = "+" if diff_iv >= 0 else ""

        st.markdown(
            f"""
        <div class="iv-result">
            <div>
                <div class="iv-label">Implied Volatility</div>
                <div class="iv-value">{solved_iv * 100:.4f}%</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <span style="color:#64748b; font-size:0.78rem;">vs input σ</span><br>
                <span style="color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:0.9rem;">
                    {diff_sign_iv}{diff_iv * 100:.4f}%
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown(
            f'<div class="iv-error">⚠️ {e}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    Black-Scholes analytical engine · CRR binomial tree (N=200) · Newton-Raphson IV solver · Finite-difference Greeks
</div>
""",
    unsafe_allow_html=True,
)