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
