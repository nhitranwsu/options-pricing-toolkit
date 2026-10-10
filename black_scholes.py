import math


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Calculates European option price via analytical Black-Scholes formula."""
    if T <= 0.0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type.lower() == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """Calculates first and second-order analytical Greeks."""
    if T <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    is_call = option_type.lower() == "call"

    delta = norm_cdf(d1) if is_call else (norm_cdf(d1) - 1.0)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)

    if is_call:
        theta = -(S * sigma * pdf_d1) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    else:
        theta = -(S * sigma * pdf_d1) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)

    return {
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 4),
        "theta": round(float(theta), 4),
        "vega": round(float(vega), 4),
        "rho": round(float(rho), 4),
    }


def numerical_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    eps_s: float = 0.01,
    eps_vol: float = 0.0001,
    eps_t: float = 1e-6,
    eps_r: float = 0.0001,
) -> dict:
    """Calculates Greeks via finite-difference numerical approximations."""
    base_px = black_scholes_price(S, K, T, r, sigma, option_type)

    px_up = black_scholes_price(S + eps_s, K, T, r, sigma, option_type)
    px_down = black_scholes_price(S - eps_s, K, T, r, sigma, option_type)
    num_delta = (px_up - px_down) / (2.0 * eps_s)
    num_gamma = (px_up - 2.0 * base_px + px_down) / (eps_s**2)

    px_vol_up = black_scholes_price(S, K, T, r, sigma + eps_vol, option_type)
    px_vol_down = black_scholes_price(S, K, T, r, sigma - eps_vol, option_type)
    num_vega = (px_vol_up - px_vol_down) / (2.0 * eps_vol)

    px_t_up = black_scholes_price(S, K, T + eps_t, r, sigma, option_type)
    px_t_down = black_scholes_price(S, K, max(1e-8, T - eps_t), r, sigma, option_type)
    num_theta = -(px_t_up - px_t_down) / (2.0 * eps_t)

    px_r_up = black_scholes_price(S, K, T, r + eps_r, sigma, option_type)
    px_r_down = black_scholes_price(S, K, T, max(0.0, r - eps_r), sigma, option_type)
    num_rho = (px_r_up - px_r_down) / (2.0 * eps_r)

    return {
        "num_delta": round(float(num_delta), 4),
        "num_gamma": round(float(num_gamma), 4),
        "num_theta": round(float(num_theta), 4),
        "num_vega": round(float(num_vega), 4),
        "num_rho": round(float(num_rho), 4),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Computes Implied Volatility using Newton-Raphson with bisection fallback and no-arbitrage boundary checks."""
    disc_strike = K * math.exp(-r * T)
    is_call = option_type.lower() == "call"

    if is_call:
        lower = max(0.0, S - disc_strike)
        upper = float(S)
    elif option_type.lower() == "put":
        lower = max(0.0, disc_strike - S)
        upper = float(disc_strike)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    if not (lower < market_price < upper):
        raise ValueError(
            f"Price {market_price:.4f} violates no-arbitrage boundary conditions ({lower:.4f}, {upper:.4f})"
        )

    # Initial volatility estimate
    sigma = math.sqrt(2.0 * math.pi / T) * (market_price / S)
    sigma = max(0.05, min(sigma, 3.0))

    # Newton-Raphson iteration
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return round(sigma, 6)

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pdf_d1 = norm_pdf(d1)
        vega = S * pdf_d1 * math.sqrt(T)

        if vega < 1e-8:
            break

        sigma -= diff / vega
        if sigma < 0.001 or sigma > 5.0:
            break

    # Fallback to bisection on [0.001, 5.0]
    low, high = 0.001, 5.0
    p_low = black_scholes_price(S, K, T, r, low, option_type)
    p_high = black_scholes_price(S, K, T, r, high, option_type)

    if market_price < p_low or market_price > p_high:
        raise ValueError(
            f"Implied volatility has no root within sane range [0.001, 5.0] (pricing bounds: [{p_low:.4f}, {p_high:.4f}])"
        )

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        p = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(p - market_price) < tol or (high - low) < 1e-7:
            return round(mid, 6)
        if p > market_price:
            high = mid
        else:
            low = mid

    return round((low + high) / 2.0, 6)


def crr_binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "call",
) -> float:
    """Prices European option via Cox-Ross-Rubinstein (CRR) Binomial Tree."""
    if T <= 0.0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    discount = math.exp(-r * dt)

    # Terminal payoffs at step N
    is_call = option_type.lower() == "call"
    values = []
    for j in range(N + 1):
        s_terminal = S * (u ** (N - j)) * (d**j)
        payoff = max(0.0, s_terminal - K) if is_call else max(0.0, K - s_terminal)
        values.append(payoff)

    # Backward induction to step 0
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            values[j] = discount * (p * values[j] + (1.0 - p) * values[j + 1])

    return round(float(values[0]), 4)


if __name__ == "__main__":
    spot = 100.0
    strike = 100.0
    tenor = 1.0
    rate = 0.05
    vol = 0.20

    # 1. Black-Scholes Pricing
    bs_call = black_scholes_price(spot, strike, tenor, rate, vol, "call")
    bs_put = black_scholes_price(spot, strike, tenor, rate, vol, "put")

    # 2. Greeks
    analytical = bs_greeks(spot, strike, tenor, rate, vol, "call")
    numerical = numerical_greeks(spot, strike, tenor, rate, vol, "call")

    # 3. Binomial Tree Pricing
    crr_call = crr_binomial_tree(spot, strike, tenor, rate, vol, N=250, option_type="call")
    crr_put = crr_binomial_tree(spot, strike, tenor, rate, vol, N=250, option_type="put")

    # 4. Implied Volatility Inversion
    inverted_iv = implied_volatility(bs_call, spot, strike, tenor, rate, "call")

    print(f"Call Price (Black-Scholes): {bs_call:.4f} | Binomial CRR: {crr_call:.4f}")
    print(f"Put Price  (Black-Scholes): {bs_put:.4f}  | Binomial CRR: {crr_put:.4f}")

    print("\n--- GREEKS CROSS-CHECK ---")
    print(f"Delta -> Analytical: {analytical['delta']} | Numerical: {numerical['num_delta']}")
    print(f"Gamma -> Analytical: {analytical['gamma']} | Numerical: {numerical['num_gamma']}")
    print(f"Vega  -> Analytical: {analytical['vega']}  | Numerical: {numerical['num_vega']}")

    print("\n--- IMPLIED VOLATILITY INVERSION ---")
    print(f"Observed Market Price: {bs_call:.4f}")
    print(f"Inverted Implied Volatility: {inverted_iv * 100:.2f}% (Expected: {vol * 100:.2f}%)")