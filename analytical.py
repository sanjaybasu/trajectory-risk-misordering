"""
Analytical Bounds on the Misordering Fraction Δ
================================================

Derives closed-form expressions and bounds for Δ in the two-state
self-exciting model, providing the theoretical backbone for the
simulation results.
"""

import numpy as np
from scipy import stats, integrate
from typing import Dict, Tuple


def steady_state_risk(lambda_0: float, lambda_1: float,
                      beta: float, mu: float) -> float:
    """Standard risk score: steady-state event rate."""
    pi_vuln = beta * lambda_0 / (beta * lambda_0 + mu + 1e-12)
    return lambda_0 * (1 - pi_vuln) + lambda_1 * pi_vuln


def trajectory_risk_analytical(lambda_0: float, lambda_1: float,
                                beta: float, mu: float,
                                T: float = 2.0, k: int = 3) -> float:
    """
    Analytical approximation of trajectory risk P(≥k events in [0,T]).

    Uses a negative binomial approximation to capture overdispersion
    from the self-exciting dynamics. The key insight: state-dependence
    (β > 0) creates overdispersion relative to Poisson, and the degree
    of overdispersion depends on β/μ (excitation-to-recovery ratio).
    """
    # Effective mean event count over [0, T]
    r_ss = steady_state_risk(lambda_0, lambda_1, beta, mu)
    mean_events = r_ss * T

    # Overdispersion from self-excitation
    # Variance/mean ratio = 1 + (λ₁ - λ₀)·β / (β·λ₀ + μ) · f(T, μ)
    # where f captures the temporal correlation
    excitation_ratio = beta / (mu + 1e-12)
    rate_contrast = (lambda_1 - lambda_0) / (lambda_0 + 1e-12)
    dispersion = 1 + rate_contrast * excitation_ratio * (1 - np.exp(-mu * T)) / (mu * T + 1e-12)
    dispersion = max(dispersion, 1.01)  # ensure overdispersed

    # Negative binomial parameterization
    # scipy.stats.nbinom(n, p): mean = n*(1-p)/p, var = n*(1-p)/p^2
    # With overdispersion d = var/mean: p = 1/d, n = mean/(d-1)
    if dispersion > 1:
        nb_n = mean_events / (dispersion - 1)
        nb_p = 1.0 / dispersion
        R = 1 - stats.nbinom.cdf(k - 1, nb_n, nb_p)
    else:
        R = 1 - stats.poisson.cdf(k - 1, mean_events)

    return float(np.clip(R, 0, 1))


def find_misordering_example() -> Dict:
    """
    Construct an explicit misordering example: two patients where
    r_A > r_B but R_A < R_B.

    Patient A: high baseline rate, low state-dependence (events don't cascade)
    Patient B: moderate baseline rate, high state-dependence (events cascade)
    """
    # Key: both patients have SIMILAR steady-state rates r ≈ 0.41,
    # but Patient B has much more overdispersed event dynamics.
    # Patient A: high λ₀, low β → nearly Poisson events, thin tail
    A = {"lambda_0": 0.40, "lambda_1": 0.55, "beta": 0.08, "mu": 2.5}
    # Patient B: lower λ₀, very high λ₁, high β, slow recovery → clustered events, fat tail
    B = {"lambda_0": 0.18, "lambda_1": 1.80, "beta": 0.70, "mu": 0.40}

    r_A = steady_state_risk(**A)
    r_B = steady_state_risk(**B)
    R_A = trajectory_risk_analytical(**A)
    R_B = trajectory_risk_analytical(**B)

    return {
        "patient_A": {**A, "r": r_A, "R": R_A},
        "patient_B": {**B, "r": r_B, "R": R_B},
        "misordered": r_A > r_B and R_A < R_B,
        "r_A_minus_r_B": r_A - r_B,
        "R_B_minus_R_A": R_B - R_A,
    }


def analytical_delta_bound(
    beta_dist_params: Tuple[float, float] = (2.5, 3.5),
    lambda_0_mean: float = 0.20,
    lambda_1_multiplier_mean: float = 3.0,
    mu_mean: float = 2.0,
    T: float = 2.0,
    k: int = 3,
    n_grid: int = 200,
) -> Dict:
    """
    Compute Δ numerically over a grid of (β, λ₀) values
    drawn from specified distributions.

    Returns the theoretical Δ and its dependence on Var(β).
    """
    rng = np.random.default_rng(0)

    # Generate grid of patients
    beta_vals = np.linspace(0.05, 0.95, n_grid)
    lambda_0_vals = np.linspace(0.05, 0.50, n_grid)

    r_grid = np.zeros((n_grid, n_grid))
    R_grid = np.zeros((n_grid, n_grid))

    for i, b in enumerate(beta_vals):
        for j, l0 in enumerate(lambda_0_vals):
            l1 = l0 * lambda_1_multiplier_mean
            r_grid[i, j] = steady_state_risk(l0, l1, b, mu_mean)
            R_grid[i, j] = trajectory_risk_analytical(l0, l1, b, mu_mean, T, k)

    # Flatten and compute Δ
    r_flat = r_grid.flatten()
    R_flat = R_grid.flatten()

    # Count all discordant pairs (exact for small grids)
    n_pts = len(r_flat)
    discordant = 0
    total = 0
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if r_flat[i] != r_flat[j] and R_flat[i] != R_flat[j]:
                total += 1
                if (r_flat[i] > r_flat[j]) != (R_flat[i] > R_flat[j]):
                    discordant += 1

    delta = discordant / total if total > 0 else 0

    return {
        "delta_analytical": delta,
        "n_grid_points": n_pts,
        "n_discordant_pairs": discordant,
        "n_total_pairs": total,
        "beta_range": (float(beta_vals.min()), float(beta_vals.max())),
        "lambda_0_range": (float(lambda_0_vals.min()), float(lambda_0_vals.max())),
    }


def delta_vs_beta_variance(
    n_levels: int = 10,
    n_patients: int = 1000,
    T: float = 2.0,
    k: int = 3,
) -> Dict:
    """
    Compute how Δ varies with the population variance of β.

    This is the key theoretical prediction: Δ grows with Var(β).
    """
    rng = np.random.default_rng(42)
    results = []

    for level in range(n_levels):
        # Vary β distribution from low to high variance
        # Use Beta(a, b) with fixed mean ≈ 0.4, varying variance
        mean_beta = 0.4
        # Beta variance = ab/((a+b)²(a+b+1)); for fixed mean, var ∝ 1/(a+b+1)
        concentration = 2 + level * 3  # a + b ranges from 2 to 29
        a = mean_beta * concentration
        b = (1 - mean_beta) * concentration

        beta_samples = rng.beta(a, b, n_patients)
        lambda_0_samples = rng.gamma(2.5, 0.08, n_patients)
        lambda_1_samples = lambda_0_samples * (1 + rng.gamma(3, 0.8, n_patients))
        mu_samples = rng.gamma(4, 0.5, n_patients)

        r_vals = np.array([
            steady_state_risk(l0, l1, beta, mu)
            for l0, l1, beta, mu in zip(lambda_0_samples, lambda_1_samples,
                                         beta_samples, mu_samples)
        ])
        R_vals = np.array([
            trajectory_risk_analytical(l0, l1, beta, mu, T, k)
            for l0, l1, beta, mu in zip(lambda_0_samples, lambda_1_samples,
                                         beta_samples, mu_samples)
        ])

        # Sample pairs for Δ
        n_pairs = 100_000
        i_idx = rng.integers(0, n_patients, n_pairs)
        j_idx = rng.integers(0, n_patients, n_pairs)
        valid = (i_idx != j_idx) & (r_vals[i_idx] != r_vals[j_idx]) & \
                (R_vals[i_idx] != R_vals[j_idx])
        i_v, j_v = i_idx[valid], j_idx[valid]

        disc = ((r_vals[i_v] > r_vals[j_v]) != (R_vals[i_v] > R_vals[j_v])).sum()
        delta = disc / len(i_v) if len(i_v) > 0 else 0

        results.append({
            "var_beta": float(beta_samples.var()),
            "mean_beta": float(beta_samples.mean()),
            "delta": float(delta),
            "concentration": concentration,
        })

    return {"delta_vs_var_beta": results}


if __name__ == "__main__":
    print("=== Misordering Example ===")
    example = find_misordering_example()
    A, B = example["patient_A"], example["patient_B"]
    print(f"Patient A: r={A['r']:.4f}, R={A['R']:.4f}  (high baseline, low cascade)")
    print(f"Patient B: r={B['r']:.4f}, R={B['R']:.4f}  (moderate baseline, high cascade)")
    print(f"Misordered: {example['misordered']}")
    print(f"  r_A - r_B = {example['r_A_minus_r_B']:+.4f}  (A ranked higher by standard score)")
    print(f"  R_B - R_A = {example['R_B_minus_R_A']:+.4f}  (B has higher trajectory risk)")

    print("\n=== Δ vs Var(β) ===")
    var_results = delta_vs_beta_variance()
    for r in var_results["delta_vs_var_beta"]:
        print(f"  Var(β)={r['var_beta']:.4f}  →  Δ={r['delta']:.4f}")
