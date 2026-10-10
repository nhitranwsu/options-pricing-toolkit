import math

import numpy as np
import streamlit as st

import ui
from analytics import (Contract, desk_greeks, greek_curves, max_loss, option_chain, parity_gap, payoff_curves,
                       pnl_explain, price_vs_vol, scenario_grid, spot_range, tenor_from, expected_move)
from black_scholes import (black_scholes_price, bs_greeks, crr_binomial_tree, implied_volatility,
                           numerical_greeks)

st.set_page_config(page_title="Options pricer", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.css(), unsafe_allow_html=True)

VIEWS = ["Pricer", "Scenarios", "Chain", "Model"]
UNITS = ["Calendar days", "Trading hours (0DTE)", "Years"]

DEFAULTS = dict(otype="Call", S=100.0, K=100.0, unit=UNITS[0], days=41.0, hours=2.0, years=0.11,
                vol=20.0, rate=5.0, div=0.0, side="Long", n=1, mult=100)
REFERENCE = dict(otype="Put", S=230.71, K=230.0, unit=UNITS[0], days=1.0, vol=36.0, rate=4.5, div=0.0)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def on_reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def on_reference():
    on_reset()
    for k, v in REFERENCE.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    b1, b2 = st.columns(2)
    b1.button("Reset", on_click=on_reset, width="stretch")
    b2.button("Test case", on_click=on_reference, width="stretch",
              help="Loads the reference put from test_options.py: S 230.71, K 230, one day, 36% vol, 4.5% rate.")

    st.markdown('<div class="tm pt sp">Contract</div>', unsafe_allow_html=True)
    st.radio("Type", ["Call", "Put"], horizontal=True, key="otype", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.number_input("Spot S", min_value=0.01, step=1.0, format="%.2f", key="S")
    c2.number_input("Strike K", min_value=0.01, step=1.0, format="%.2f", key="K")

    st.markdown('<div class="tm pt sp">Expiry</div>', unsafe_allow_html=True)
    st.selectbox("Unit", UNITS, key="unit", label_visibility="collapsed")
    unit = st.session_state.unit
    if unit == UNITS[0]:
        st.number_input("Days to expiry", min_value=0.1, max_value=730.0, step=1.0, key="days")
        expiry_value = st.session_state.days
    elif unit == UNITS[1]:
        st.slider("Trading hours left today", 0.25, 6.5, step=0.25, key="hours")
        expiry_value = st.session_state.hours
    else:
        st.slider("Years to expiry", 0.001, 2.0, step=0.01, key="years")
        expiry_value = st.session_state.years

    st.markdown('<div class="tm pt sp">Market</div>', unsafe_allow_html=True)
    st.number_input("Volatility σ, %", min_value=0.1, max_value=500.0, step=0.5, format="%.2f", key="vol")
    st.slider("Risk free rate r, %", 0.0, 15.0, step=0.05, key="rate")
    st.slider("Dividend yield q, %", 0.0, 10.0, step=0.05, key="div")

    st.markdown('<div class="tm pt sp">Position</div>', unsafe_allow_html=True)
    st.radio("Side", ["Long", "Short"], horizontal=True, key="side", label_visibility="collapsed")
    c3, c4 = st.columns(2)
    c3.number_input("Contracts", min_value=1, max_value=10000, step=1, key="n")
    c4.number_input("Multiplier", min_value=1, max_value=1000, step=1, key="mult")

s = st.session_state
T, dpy = tenor_from(unit, expiry_value)
c = Contract(S=float(s.S), K=float(s.K), T=T, r=s.rate / 100.0, sigma=s.vol / 100.0, q=s.div / 100.0,
             option_type=s.otype.lower(), qty=int(s.n) * (1 if s.side == "Long" else -1), mult=int(s.mult), dpy=dpy)
d = desk_greeks(c)

# ----------------------------------------------------------------------------------------------- header
c_brand, c_view = st.columns([4.4, 2.4], vertical_alignment="center")
c_brand.markdown(ui.brand_html(), unsafe_allow_html=True)
view = c_view.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
st.markdown(ui.meta_html(c), unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False}


def tree_prices(american):
    try:
        return crr_binomial_tree(c.S, c.K, c.T, c.r, c.sigma, N=200, option_type=c.option_type, q=c.q,
                                 american=american, digits=None)
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------- views
def pricer():
    st.markdown(ui.kpi_html(c, d), unsafe_allow_html=True)
    spots = spot_range(c)
    fracs = (1.0, 0.5, 0.25)
    left, right = st.columns([2.4, 1], gap="medium")
    with left:
        st.markdown(ui.payoff_legend(), unsafe_allow_html=True)
        st.plotly_chart(ui.payoff_figure(c, d, spots, payoff_curves(c, spots)), theme=None, width="stretch",
                        config=PLOT_CFG)
        st.markdown(ui.greeks_legend(c, fracs), unsafe_allow_html=True)
        st.plotly_chart(ui.greeks_figure(c, spots, greek_curves(c, spots, fracs), fracs), theme=None,
                        width="stretch", config=PLOT_CFG)
    with right:
        st.markdown('<div class="tm pt">Position</div>', unsafe_allow_html=True)
        st.markdown(ui.position_html(c, d, max_loss(c, d["price"])), unsafe_allow_html=True)
        st.markdown('<div class="tm pt sp">Model check</div>', unsafe_allow_html=True)
        st.markdown(ui.model_check_html(d["price"], tree_prices(False), tree_prices(True),
                                        parity_gap(c)), unsafe_allow_html=True)
        st.markdown('<div class="tm note">Black-Scholes prices a European option. The American tree is shown for '
                    'comparison: a premium above the European value means early exercise has value, which is common '
                    'for puts and for calls ahead of dividends.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tm note">Greeks are per option. Position figures multiply by contracts and '
                    'multiplier. Theta is per calendar day on the 365 day basis, or per trading day on the intraday '
                    'basis. Breakeven ignores carry.</div>', unsafe_allow_html=True)


def scenarios():
    _, mvp = expected_move(c)
    span = float(min(40.0, max(5.0, 5.0 * math.ceil(3.0 * mvp / 5.0))))
    max_days = float(max(c.T * c.dpy, 0.05))
    step_days = 0.25 if max_days > 2 else 0.01

    st.markdown('<div class="tm pt">Scenario</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    spot_mv = k1.slider("Spot move, %", -span, span, -(round(span / 5.0 * 2.0) / 2.0), 0.1 if span < 10 else 0.5)
    vol_mv = k2.slider("Vol move, points", -15.0, 15.0, 0.0, 0.5)
    days = k3.slider("Time forward, days", 0.0, max_days, min(1.0, max_days), step_days)

    spot_shocks = np.linspace(-span, span, 41)
    vol_shocks = np.arange(-15.0, 15.01, 1.0)
    vol_shocks = vol_shocks[c.sigma * 100.0 + vol_shocks >= 0.5]
    grid = scenario_grid(c, spot_shocks, vol_shocks, days)
    ex = pnl_explain(c, spot_mv, vol_mv, days)

    left, right = st.columns([2.4, 1], gap="medium")
    with left:
        st.plotly_chart(ui.scenario_heatmap(c, grid, spot_shocks, vol_shocks, sel=(spot_mv, vol_mv)), theme=None,
                        width="stretch", config=PLOT_CFG)
    with right:
        st.markdown('<div class="tm pt">P&amp;L explain</div>', unsafe_allow_html=True)
        st.plotly_chart(ui.explain_figure(ex), theme=None, width="stretch", config=PLOT_CFG)
        st.markdown(ui.explain_table_html(ex), unsafe_allow_html=True)
        st.markdown(f'<div class="tm note">Spot {ex["S1"]:,.2f} ({ui._n(spot_mv, ".1f")}%), vol {ex["sigma1"] * 100:.1f}%, '
                    f'{days:g} {"day" if days == 1 else "days"} forward. Delta, gamma, vega and theta are the Taylor terms from today\'s Greeks. '
                    f'The rest is higher order: large moves, vanna, volga and the curvature of time decay.</div>',
                    unsafe_allow_html=True)


def chain():
    st.markdown('<div class="tm pt">Option chain at the input vol, with an illustrative smile</div>',
                unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    skew = k1.slider("Skew, vol points per +10% in strike", -8.0, 8.0, -2.0, 0.5)
    curv = k2.slider("Curvature, vol points", 0.0, 6.0, 0.5, 0.5)
    n = k3.select_slider("Strikes", options=[7, 9, 11, 13, 15], value=11)
    ch = option_chain(c, n, skew, curv)
    st.markdown(ui.chain_html(c, ch), unsafe_allow_html=True)
    st.markdown('<div class="tm note">The smile is a simple skew and curvature around the input vol, not market '
                'data. Every row is priced with Black-Scholes at that strike\'s vol. Shaded cells are in the money, '
                'the outlined row is the strike nearest spot.</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(ui.smile_figure(c, ch), theme=None, width="stretch", config=PLOT_CFG)
    with right:
        st.markdown(f'<div class="tm pt">Implied vol solver for the {c.option_type} at strike {c.K:,.2f}</div>',
                    unsafe_allow_html=True)
        target = st.number_input("Observed option price", min_value=0.0001, value=max(round(d["price"], 4), 0.0001),
                                 step=0.01, format="%.4f")
        iv = None
        try:
            iv = implied_volatility(target, c.S, c.K, c.T, c.r, c.option_type, q=c.q)
            st.markdown(ui.iv_result_html(iv, c.sigma), unsafe_allow_html=True)
        except ValueError as e:
            st.markdown(ui.iv_result_html(0, c.sigma, ok=False, msg=str(e)), unsafe_allow_html=True)
        st.markdown('<div class="tm note">The default price is the model price rounded to four decimals, so it '
                    'solves back to about the input vol. Type a market price to get its implied vol.</div>',
                    unsafe_allow_html=True)
        vols = np.linspace(0.02, max(1.0, (iv or c.sigma) * 1.6), 200)
        st.plotly_chart(ui.iv_figure(c, vols, price_vs_vol(c, vols), target, iv), theme=None, width="stretch",
                        config=PLOT_CFG)


@st.cache_data(show_spinner=False)
def convergence(S, K, T, r, sigma, q, opt):
    bs = black_scholes_price(S, K, T, r, sigma, opt, q)
    ns, errs = [], []
    for n in (10, 20, 40, 80, 160, 320, 640, 1280):
        try:
            errs.append(crr_binomial_tree(S, K, T, r, sigma, N=n, option_type=opt, q=q, digits=None) - bs)
            ns.append(n)
        except ValueError:
            continue
    return ns, errs


def model():
    st.markdown('<div class="tm pt">Pricing rule</div>', unsafe_allow_html=True)
    st.latex(r"d_{1} = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^{2})T}{\sigma\sqrt{T}}, \qquad d_{2} = d_{1} - \sigma\sqrt{T}")
    st.latex(r"C = S e^{-qT} N(d_{1}) - K e^{-rT} N(d_{2}), \qquad P = K e^{-rT} N(-d_{2}) - S e^{-qT} N(-d_{1})")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}")
    st.markdown('<div class="tm note">The tree uses the same inputs and converges to the Black-Scholes value as the '
                'number of steps grows. With early exercise switched on it prices the American option.</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        st.markdown('<div class="tm pt sp">Tree convergence</div>', unsafe_allow_html=True)
        ns, errs = convergence(c.S, c.K, c.T, c.r, c.sigma, c.q, c.option_type)
        if len(ns) >= 3:
            st.plotly_chart(ui.convergence_figure(ns, errs), theme=None, width="stretch", config=PLOT_CFG)
            k = float(np.median(np.abs(errs) * np.array(ns)))
            st.markdown(f'<div class="tm note">The error shrinks roughly like {k:.2f} / N. The saw tooth is normal for '
                        f'a CRR tree when the strike does not sit on a node: the envelope still falls like 1 / N, but '
                        f'a single N can land above or below it.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tm note">The tree is unstable for these inputs at small N.</div>',
                        unsafe_allow_html=True)
    with right:
        st.markdown('<div class="tm pt sp">Analytical against finite difference Greeks</div>', unsafe_allow_html=True)
        an = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.option_type, q=c.q, digits=None)
        nu = numerical_greeks(c.S, c.K, c.T, c.r, c.sigma, c.option_type, q=c.q)
        rows = [("Delta", an["delta"], nu["num_delta"], 0.01), ("Gamma", an["gamma"], nu["num_gamma"], 0.01),
                ("Theta, per yr", an["theta"], nu["num_theta"], 0.05), ("Vega, per 1.00", an["vega"], nu["num_vega"], 0.01),
                ("Rho, per 1.00", an["rho"], nu["num_rho"], 0.01)]
        st.markdown(ui.xcheck_html(rows), unsafe_allow_html=True)
        st.markdown('<div class="tm pt sp">Put-call parity</div>', unsafe_allow_html=True)
        st.markdown(ui.kv_html([("C − P − (S·e^(−qT) − K·e^(−rT))", f'{parity_gap(c):.1e}')]), unsafe_allow_html=True)
        st.markdown('<div class="tm note">The reference tests in test_options.py also pin the put price, Greeks and '
                    'implied vol to known values and check that prices outside the no arbitrage bounds are '
                    'rejected. The Test case button in the sidebar loads that same put.</div>', unsafe_allow_html=True)

    st.markdown('<div class="tm pt sp">What the model assumes, and what a desk will push on</div>',
                unsafe_allow_html=True)
    st.markdown(ui.assumptions_html(), unsafe_allow_html=True)


{"Pricer": pricer, "Scenarios": scenarios, "Chain": chain, "Model": model}[view]()
