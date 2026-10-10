import math

import numpy as np
import pytest

from analytics import (Contract, breakevens, desk_greeks, option_chain, parity_gap, payoff_curves, pnl_explain,
                       scenario_grid, tenor_from)
from black_scholes import black_scholes_price

C0 = Contract(S=100.0, K=100.0, T=41 / 365, r=0.05, sigma=0.20, q=0.0, option_type="call", qty=3, mult=100)


def test_zero_scenario_has_zero_pnl():
    g = scenario_grid(C0, [0.0], [0.0], 0.0)
    assert g[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_scenario_grid_matches_scalar_revaluation():
    g = scenario_grid(C0, [-5.0, 8.0], [-3.0, 6.0], 10.0)
    px0 = black_scholes_price(100, 100, C0.T, 0.05, 0.20, "call")
    for i, dv in enumerate([-3.0, 6.0]):
        for j, ds in enumerate([-5.0, 8.0]):
            px1 = black_scholes_price(100 * (1 + ds / 100), 100, C0.T - 10 / 365, 0.05, 0.20 + dv / 100, "call")
            assert g[i, j] == pytest.approx((px1 - px0) * 300, abs=1e-8)


def test_pnl_explain_small_shock_is_almost_fully_explained():
    ex = pnl_explain(C0, 0.5, 0.5, 1.0)
    assert abs(ex["residual"]) < 0.05 * abs(ex["full"])
    assert ex["explained"] + ex["residual"] == pytest.approx(ex["full"])


def test_pnl_explain_gamma_is_positive_for_long_option_and_negative_for_short():
    long_gamma = pnl_explain(C0, 3.0, 0.0, 0.0)["parts"]["Gamma"]
    short_gamma = pnl_explain(C0.with_(qty=-3), 3.0, 0.0, 0.0)["parts"]["Gamma"]
    assert long_gamma > 0 > short_gamma


def test_short_position_flips_every_sign():
    a, b = desk_greeks(C0)["pos"], desk_greeks(C0.with_(qty=-3))["pos"]
    for k in ("value", "delta_sh", "theta_day", "vega_pt", "rho_pct"):
        assert a[k] == pytest.approx(-b[k])


def test_desk_units():
    d = desk_greeks(C0)
    assert d["theta_day"] == pytest.approx(d["theta_year"] / 365.0)
    assert d["vega_pt"] == pytest.approx(d["vega"] / 100.0)
    assert d["time_value"] == pytest.approx(d["price"])          # at the money, no intrinsic value
    assert 0.0 < d["prob_itm"] < 1.0


def test_intraday_time_basis_uses_trading_days():
    T, dpy = tenor_from("Trading hours (0DTE)", 2.0)
    assert dpy == 252.0 and T == pytest.approx(2.0 / (252 * 6.5))


def test_breakeven_of_long_call_at_expiry_is_strike_plus_premium():
    spots = np.linspace(60, 140, 8001)
    pnl = payoff_curves(C0, spots, fractions=(1.0,))[1.0]
    be = breakevens(spots, pnl)
    assert len(be) == 1
    assert be[0] == pytest.approx(desk_greeks(C0)["breakeven"], abs=0.02)


def test_chain_is_symmetric_around_the_centre_with_flat_vol_and_obeys_parity():
    ch = option_chain(C0.with_(S=100.0), 11, 0.0, 0.0)
    assert len(ch["K"]) == 11 and np.allclose(ch["iv"], 0.20)
    lhs = ch["call"]["price"] - ch["put"]["price"]
    rhs = 100.0 - ch["K"] * math.exp(-0.05 * C0.T)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_parity_gap_is_zero():
    assert abs(parity_gap(C0.with_(q=0.02))) < 1e-10
