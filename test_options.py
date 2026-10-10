import pytest
from black_scholes import (
    black_scholes_price,
    bs_greeks,
    numerical_greeks,
    implied_volatility,
)

# Reference parameters
S_REF = 230.71
K_REF = 230.0
T_REF = 1.0 / 365.0
R_REF = 0.045
SIGMA_REF = 0.36
TOLERANCE = 1e-3


def test_put_price_reference():
    """Verify Black-Scholes put price matches reference value 1.3873 within 1e-3 tolerance."""
    price = black_scholes_price(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, option_type="put")
    assert pytest.approx(1.3873, abs=TOLERANCE) == price


def test_put_greeks_reference():
    """Verify analytical Greeks match reference values within 1e-3 tolerance."""
    greeks = bs_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, option_type="put")
    assert pytest.approx(-0.4288, abs=TOLERANCE) == greeks["delta"]
    assert pytest.approx(0.0903, abs=TOLERANCE) == greeks["gamma"]
    assert pytest.approx(4.7406, abs=TOLERANCE) == greeks["vega"]  # per 1.00 vol
    assert pytest.approx(-0.2748, abs=TOLERANCE) == greeks["rho"]
    assert pytest.approx(-306.94, abs=TOLERANCE) == round(greeks["theta"], 2)


def test_implied_volatility_reference():
    """Verify IV solver recovers ~0.3574 for an observed put price of 1.375."""
    iv = implied_volatility(1.375, S_REF, K_REF, T_REF, R_REF, option_type="put")
    assert pytest.approx(0.3574, abs=TOLERANCE) == iv


def test_put_bounds_check_rejection():
    """Verify that a put priced at 300 with K=230 is strictly rejected by the bounds check."""
    with pytest.raises(ValueError, match="violates no-arbitrage boundary conditions"):
        implied_volatility(300.0, S_REF, K_REF, T_REF, R_REF, option_type="put")


def test_call_bounds_check_rejection():
    """Verify that a call priced above spot is strictly rejected by the bounds check."""
    with pytest.raises(ValueError, match="violates no-arbitrage boundary conditions"):
        implied_volatility(250.0, S_REF, K_REF, T_REF, R_REF, option_type="call")


def test_numerical_greeks_consistency():
    """Verify all 5 numerical Greeks (Delta, Gamma, Theta, Vega, Rho) match analytical Greeks."""
    num_gk = numerical_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, option_type="put")
    an_gk = bs_greeks(S_REF, K_REF, T_REF, R_REF, SIGMA_REF, option_type="put")
    assert pytest.approx(an_gk["delta"], abs=0.01) == num_gk["num_delta"]
    assert pytest.approx(an_gk["gamma"], abs=0.01) == num_gk["num_gamma"]
    assert pytest.approx(an_gk["theta"], abs=0.05) == num_gk["num_theta"]
    assert pytest.approx(an_gk["vega"], abs=0.01) == num_gk["num_vega"]
    assert pytest.approx(an_gk["rho"], abs=0.01) == num_gk["num_rho"]


# ---------------------------------------------------------------- dividend yield, vectorised grids, American tree
import math

import numpy as np

from black_scholes import bs_np, crr_binomial_tree

Q_REF = 0.02


def test_put_call_parity_with_dividend_yield():
    """C - P = S*exp(-qT) - K*exp(-rT) for any inputs."""
    S, K, T, r, sig = 105.0, 100.0, 0.75, 0.04, 0.27
    c = black_scholes_price(S, K, T, r, sig, "call", q=Q_REF)
    p = black_scholes_price(S, K, T, r, sig, "put", q=Q_REF)
    assert c - p == pytest.approx(S * math.exp(-Q_REF * T) - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_vectorised_matches_scalar(opt):
    """bs_np must agree with the scalar functions on price and all five Greeks, with and without dividends."""
    spots = np.array([70.0, 95.0, 100.0, 118.0])
    for q in (0.0, Q_REF):
        out = bs_np(spots, 100.0, 0.5, 0.03, 0.31, opt == "call", q)
        for i, s in enumerate(spots):
            assert out["price"][i] == pytest.approx(black_scholes_price(s, 100.0, 0.5, 0.03, 0.31, opt, q), abs=1e-12)
            g = bs_greeks(s, 100.0, 0.5, 0.03, 0.31, opt, q=q, digits=None)
            for k in ("delta", "gamma", "vega", "theta", "rho"):
                assert out[k][i] == pytest.approx(g[k], abs=1e-10)


def test_vectorised_expiry_is_intrinsic():
    out = bs_np(np.array([90.0, 110.0]), 100.0, 0.0, 0.03, 0.3, True)
    assert list(out["price"]) == [0.0, 10.0]
    assert list(out["delta"]) == [0.0, 0.0]


def test_numerical_greeks_consistency_with_dividend_yield():
    num = numerical_greeks(S_REF, K_REF, 0.25, R_REF, SIGMA_REF, option_type="call", q=Q_REF)
    an = bs_greeks(S_REF, K_REF, 0.25, R_REF, SIGMA_REF, option_type="call", q=Q_REF)
    assert pytest.approx(an["delta"], abs=0.01) == num["num_delta"]
    assert pytest.approx(an["gamma"], abs=0.01) == num["num_gamma"]
    assert pytest.approx(an["theta"], abs=0.05) == num["num_theta"]
    assert pytest.approx(an["vega"], abs=0.01) == num["num_vega"]
    assert pytest.approx(an["rho"], abs=0.01) == num["num_rho"]


def test_crr_european_converges_to_black_scholes():
    """CRR is first order accurate: the error shrinks like 1/N, so error * N stays roughly constant."""
    bs = black_scholes_price(100.0, 100.0, 0.5, 0.05, 0.25, "put", q=Q_REF)
    errs = {}
    for n in (100, 400, 1600):
        tree = crr_binomial_tree(100.0, 100.0, 0.5, 0.05, 0.25, N=n, option_type="put", q=Q_REF, digits=None)
        errs[n] = abs(tree - bs)
    assert errs[1600] < errs[400] < errs[100]
    assert errs[1600] * 1600 == pytest.approx(errs[100] * 100, rel=0.05)
    assert errs[1600] < 2e-3


def test_american_put_is_worth_at_least_european():
    eu = crr_binomial_tree(100.0, 110.0, 1.0, 0.08, 0.2, N=300, option_type="put", digits=None)
    am = crr_binomial_tree(100.0, 110.0, 1.0, 0.08, 0.2, N=300, option_type="put", american=True, digits=None)
    assert am > eu + 0.05           # a deep in the money put with a high rate has a real early exercise premium
    assert am >= 110.0 - 100.0      # never below intrinsic value


def test_american_call_without_dividends_equals_european():
    eu = crr_binomial_tree(100.0, 95.0, 1.0, 0.05, 0.3, N=300, option_type="call", digits=None)
    am = crr_binomial_tree(100.0, 95.0, 1.0, 0.05, 0.3, N=300, option_type="call", american=True, digits=None)
    assert am == pytest.approx(eu, abs=1e-9)


def test_implied_volatility_round_trip_with_dividend_yield():
    px = black_scholes_price(100.0, 105.0, 0.4, 0.04, 0.33, "call", q=Q_REF)
    assert implied_volatility(px, 100.0, 105.0, 0.4, 0.04, "call", q=Q_REF) == pytest.approx(0.33, abs=1e-4)
