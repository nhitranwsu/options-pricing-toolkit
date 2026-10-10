"""Desk-style calculations on top of black_scholes.py. Pure functions, no Streamlit and no plotting.

Units used on the desk and shown in the app:
    delta   per 1.00 of spot (shares of underlying per option)
    gamma   per 1.00 of spot, and as change in delta for a 1% spot move
    vega    per 1 volatility point (1.00% of vol)
    theta   per day
    rho     per 1% move in the rate
"""
import math
from dataclasses import dataclass

import numpy as np

from black_scholes import bs_greeks, bs_np, black_scholes_price, norm_cdf


@dataclass(frozen=True)
class Contract:
    S: float
    K: float
    T: float            # years, on the time basis chosen in the sidebar
    r: float
    sigma: float
    q: float = 0.0
    option_type: str = "call"
    qty: int = 1        # contracts, negative = short
    mult: int = 100     # shares per contract
    dpy: float = 365.0  # days per year on the time basis (252 for the intraday trading hours basis)

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def pos(self) -> float:
        """Position size in shares of underlying."""
        return float(self.qty * self.mult)

    def with_(self, **kw):
        d = {**self.__dict__, **kw}
        return Contract(**d)


def moneyness(c: Contract):
    """Returns (label, percent in the money). Negative percent means out of the money."""
    pct = (c.S / c.K - 1.0) * 100.0 if c.is_call else (c.K / c.S - 1.0) * 100.0
    if abs(math.log(c.S / c.K)) < 0.0025:
        return "ATM", pct
    return ("ITM" if pct > 0 else "OTM"), pct


def expected_move(c: Contract):
    """One standard deviation move in spot over the life of the option, in price units and percent."""
    mv = c.S * c.sigma * math.sqrt(c.T)
    return mv, 100.0 * mv / c.S


def prob_itm(c: Contract) -> float:
    """Risk neutral probability of finishing in the money, N(d2) for a call and N(-d2) for a put."""
    d2 = (math.log(c.S / c.K) + (c.r - c.q - 0.5 * c.sigma**2) * c.T) / (c.sigma * math.sqrt(c.T))
    return norm_cdf(d2) if c.is_call else norm_cdf(-d2)


def desk_greeks(c: Contract) -> dict:
    """Price, Greeks in desk units, and the same numbers scaled to the position."""
    px = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q)
    g = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.option_type, q=c.q, digits=None)
    intrinsic = max(0.0, c.S - c.K) if c.is_call else max(0.0, c.K - c.S)
    pos = c.pos
    d = {
        "price": px, "intrinsic": intrinsic, "time_value": px - intrinsic,
        "delta": g["delta"], "gamma": g["gamma"],
        "gamma_1pct": g["gamma"] * c.S * 0.01,          # change in delta for a +1% spot move, per option
        "vega_pt": g["vega"] / 100.0,
        "theta_day": g["theta"] / c.dpy,
        "rho_pct": g["rho"] / 100.0,
        "theta_year": g["theta"], "vega": g["vega"], "rho": g["rho"],
        "prob_itm": prob_itm(c),
        "breakeven": c.K + px if c.is_call else c.K - px,
    }
    d["breakeven_pct"] = 100.0 * (d["breakeven"] / c.S - 1.0)
    d["pos"] = {
        "shares": pos,
        "value": px * pos,
        "delta_sh": d["delta"] * pos,
        "delta_usd": d["delta"] * pos * c.S,
        "gamma_1pct_sh": d["gamma_1pct"] * pos,
        "theta_day": d["theta_day"] * pos,
        "vega_pt": d["vega_pt"] * pos,
        "rho_pct": d["rho_pct"] * pos,
    }
    return d


def max_loss(c: Contract, price: float):
    """Largest possible loss of the position from today's price, or None when it is unlimited."""
    if c.qty > 0:
        return price * c.pos
    if c.is_call:
        return None
    return (c.K - price) * abs(c.pos)


def tenor_from(unit: str, value: float):
    """Convert the sidebar expiry input to (T in years, days per year on that time basis)."""
    if unit == "Calendar days":
        return max(value / 365.0, 1e-5), 365.0
    if unit == "Trading hours (0DTE)":
        # 252 trading days of 6.5 hours, so one trading day is 1/252 of a year
        return max(value / (252.0 * 6.5), 1e-5), 252.0
    return max(value, 1e-5), 365.0


def spot_range(c: Contract, lo=0.65, hi=1.35, n=241):
    base = min(c.S, c.K), max(c.S, c.K)
    return np.linspace(lo * base[0], hi * base[1], n)


# ---------------------------------------------------------------------------------------------- payoff and curves
def payoff_curves(c: Contract, spots, fractions=(0.0, 0.5, 1.0)):
    """Position P&L against spot for a few points in time. fraction = share of the remaining life that is used up."""
    px0 = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q)
    out = {}
    for f in fractions:
        T = c.T * (1.0 - f)
        v = bs_np(spots, c.K, T, c.r, c.sigma, c.is_call, c.q)["price"]
        out[f] = (v - px0) * c.pos
    return out


def breakevens(spots, pnl):
    """Spot levels where a P&L curve crosses zero, by linear interpolation."""
    s = np.sign(pnl)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return [float(spots[i] - pnl[i] * (spots[i + 1] - spots[i]) / (pnl[i + 1] - pnl[i])) for i in idx]


def greek_curves(c: Contract, spots, tenor_fracs=(1.0, 0.5, 0.25)):
    """Per option Greeks against spot for the current expiry and shorter ones, in desk units."""
    out = {}
    for f in tenor_fracs:
        g = bs_np(spots, c.K, c.T * f, c.r, c.sigma, c.is_call, c.q)
        out[f] = {"delta": g["delta"], "gamma": g["gamma"], "vega_pt": g["vega"] / 100.0,
                  "theta_day": g["theta"] / c.dpy}
    return out


# ---------------------------------------------------------------------------------------------- scenarios
def scenario_grid(c: Contract, spot_shocks_pct, vol_shocks_pts, days: float):
    """Full revaluation P&L of the position on a grid of spot moves (columns) and vol moves (rows).

    Time is rolled forward by `days`. Rates and dividend yield are held fixed.
    Returns a matrix of shape (len(vol_shocks), len(spot_shocks)).
    """
    px0 = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q)
    S = c.S * (1.0 + np.asarray(spot_shocks_pct, dtype=float) / 100.0)[None, :]
    sig = np.maximum(c.sigma + np.asarray(vol_shocks_pts, dtype=float) / 100.0, 0.005)[:, None]
    T = max(c.T - days / c.dpy, 0.0)
    v = bs_np(S, c.K, T, c.r, sig, c.is_call, c.q)["price"]
    return (v - px0) * c.pos


def pnl_explain(c: Contract, dS_pct: float, dvol_pts: float, days: float) -> dict:
    """Splits the revaluation P&L of one scenario into delta, gamma, vega and theta, plus what is left over."""
    g = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.option_type, q=c.q, digits=None)
    px0 = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q)
    dS = c.S * dS_pct / 100.0
    sig1 = max(c.sigma + dvol_pts / 100.0, 0.005)
    T1 = max(c.T - days / c.dpy, 0.0)
    px1 = black_scholes_price(c.S + dS, c.K, T1, c.r, sig1, c.option_type, c.q)
    pos = c.pos
    parts = {
        "Delta": g["delta"] * dS * pos,
        "Gamma": 0.5 * g["gamma"] * dS**2 * pos,
        "Vega": g["vega"] * (sig1 - c.sigma) * pos,
        "Theta": g["theta"] / c.dpy * min(days, c.T * c.dpy) * pos,
    }
    full = (px1 - px0) * pos
    explained = sum(parts.values())
    return {"parts": parts, "explained": explained, "full": full, "residual": full - explained,
            "S1": c.S + dS, "sigma1": sig1, "px1": px1}


# ---------------------------------------------------------------------------------------------- option chain
def nice_step(S: float) -> float:
    for lim, step in ((20, 0.5), (50, 1.0), (100, 2.5), (250, 5.0), (500, 10.0), (1000, 25.0)):
        if S < lim:
            return step
    return 50.0


def smile_vol(c: Contract, strikes, skew_pts: float, curv_pts: float):
    """Illustrative smile around the input vol: skew_pts vol points per +10% in strike, plus curvature."""
    x = (np.asarray(strikes, dtype=float) / c.S - 1.0) / 0.10
    return np.maximum(c.sigma + (skew_pts * x + curv_pts * x**2) / 100.0, 0.01)


def option_chain(c: Contract, n_strikes: int, skew_pts: float, curv_pts: float):
    step = nice_step(c.S)
    centre = round(c.S / step) * step
    half = n_strikes // 2
    strikes = centre + step * np.arange(-half, half + 1)
    strikes = strikes[strikes > 0]
    iv = smile_vol(c, strikes, skew_pts, curv_pts)
    call = bs_np(c.S, strikes, c.T, c.r, iv, True, c.q)
    put = bs_np(c.S, strikes, c.T, c.r, iv, False, c.q)
    return {"K": strikes, "iv": iv, "call": call, "put": put, "step": step}


# ---------------------------------------------------------------------------------------------- validation
def price_vs_vol(c: Contract, vols):
    return bs_np(c.S, c.K, c.T, c.r, vols, c.is_call, c.q)["price"]


def parity_gap(c: Contract) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}). Zero up to rounding if the pricer is right."""
    cp = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, "call", c.q)
    pp = black_scholes_price(c.S, c.K, c.T, c.r, c.sigma, "put", c.q)
    return cp - pp - (c.S * math.exp(-c.q * c.T) - c.K * math.exp(-c.r * c.T))
