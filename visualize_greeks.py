import matplotlib.pyplot as plt
from black_scholes import bs_greeks, black_scholes_price


def plot_greeks_sensitivity():
    # Base parameters
    K = 100.0
    r = 0.05
    sigma = 0.20

    # 1. Delta & Gamma vs Spot for varying Time to Expiry (T)
    spots = [60.0 + i * 0.5 for i in range(161)]  # Spot from 60 to 140
    tenors = [0.1, 0.5, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Options Greeks Sensitivity Surfaces", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Subplot 1: Delta vs Spot
    ax_delta = axes[0, 0]
    for T in tenors:
        deltas = [bs_greeks(S, K, T, r, sigma, "call")["delta"] for S in spots]
        ax_delta.plot(spots, deltas, label=f"T = {T}y")
    ax_delta.axvline(K, color="gray", linestyle="--", alpha=0.6, label="Strike K=100")
    ax_delta.set_title("Call Delta vs Spot Price")
    ax_delta.set_xlabel("Spot Price (S)")
    ax_delta.set_ylabel("Delta")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend()

    # Subplot 2: Gamma vs Spot (Peak near ATM)
    ax_gamma = axes[0, 1]
    for T in tenors:
        gammas = [bs_greeks(S, K, T, r, sigma, "call")["gamma"] for S in spots]
        ax_gamma.plot(spots, gammas, label=f"T = {T}y")
    ax_gamma.axvline(K, color="gray", linestyle="--", alpha=0.6, label="Strike K=100")
    ax_gamma.set_title("Gamma vs Spot Price (Clustering around ATM)")
    ax_gamma.set_xlabel("Spot Price (S)")
    ax_gamma.set_ylabel("Gamma")
    ax_gamma.grid(True, alpha=0.3)
    ax_gamma.legend()

    # Subplot 3: Vega vs Spot
    ax_vega = axes[1, 0]
    for T in tenors:
        vegas = [bs_greeks(S, K, T, r, sigma, "call")["vega"] for S in spots]
        ax_vega.plot(spots, vegas, label=f"T = {T}y")
    ax_vega.axvline(K, color="gray", linestyle="--", alpha=0.6, label="Strike K=100")
    ax_vega.set_title("Vega vs Spot Price")
    ax_vega.set_xlabel("Spot Price (S)")
    ax_vega.set_ylabel("Vega")
    ax_vega.grid(True, alpha=0.3)
    ax_vega.legend()

    # Subplot 4: Theta Decay vs Time to Expiry (T -> 0)
    ax_theta = axes[1, 1]
    tenor_steps = [0.01 + i * 0.01 for i in range(100)]  # 0.01y to 1.0y
    moneyness_spots = [90.0, 100.0, 110.0]  # OTM, ATM, ITM
    for S in moneyness_spots:
        label = "ATM (S=100)" if S == 100.0 else ("OTM (S=90)" if S < 100.0 else "ITM (S=110)")
        thetas = [bs_greeks(S, K, t, r, sigma, "call")["theta"] for t in tenor_steps]
        ax_theta.plot(tenor_steps, thetas, label=label)
    ax_theta.set_title("Theta Decay (Time Decay Acceleration)")
    ax_theta.set_xlabel("Time to Expiry T (years)")
    ax_theta.set_ylabel("Theta (Annualized)")
    ax_theta.grid(True, alpha=0.3)
    ax_theta.legend()

    plt.tight_layout()
    plt.savefig("greeks_sensitivities.png", dpi=300)
    print("Sensitivity charts successfully saved to 'greeks_sensitivities.png'!")
    plt.show()


if __name__ == "__main__":
    plot_greeks_sensitivity()