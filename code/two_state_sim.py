"""
Two-state self-exciting microsimulation.

Each patient is governed by parameters (lambda_0, lambda_1, beta, mu).
Standard risk score: closed-form steady-state event rate.
Trajectory risk: Monte Carlo simulation, P(>= k events in T years).
Misordering fraction Delta: discordance probability between the two rankings.

This file is intentionally minimal so that BO, CMA-ES, agent search, and
random search can all call a single evaluate(params, fidelity) function.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class Population:
    n: int
    lambda_0: np.ndarray
    lambda_1: np.ndarray
    beta: np.ndarray
    mu: np.ndarray


def build_population(params: Dict, n: int, seed: int) -> Population:
    rng = np.random.default_rng(seed)
    l0_shape = params.get("lambda_0_shape", 3.0)
    l0_scale = params.get("lambda_0_scale", 0.20)
    lambda_0 = rng.gamma(l0_shape, l0_scale, n)
    l1_mult_shape = params.get("lambda_1_mult_shape", 2.0)
    l1_mult_scale = params.get("lambda_1_mult_scale", 0.8)
    lambda_1 = lambda_0 * (1 + rng.gamma(l1_mult_shape, l1_mult_scale, n))
    beta_a = params.get("beta_a", 3.0)
    beta_b = params.get("beta_b", 7.0)
    beta = rng.beta(beta_a, beta_b, n)
    mu_shape = params.get("mu_shape", 4.0)
    mu_scale = params.get("mu_scale", 1.0)
    mu = rng.gamma(mu_shape, mu_scale, n)
    # Floor mu to keep simulation numerically stable but allow agents to
    # explore near-absorbing regimes (mu >= 0.005/year ~ 200-year recovery).
    mu = np.clip(mu, 0.005, None)
    return Population(n, lambda_0, lambda_1, beta, mu)


def standard_score(pop: Population) -> np.ndarray:
    """Closed-form steady-state event rate (the standard risk score)."""
    pi = pop.beta * pop.lambda_0 / (pop.beta * pop.lambda_0 + pop.mu)
    return pop.lambda_0 * (1 - pi) + pop.lambda_1 * pi


def simulate_trajectory_risk(
    pop: Population, T: float, k: int, n_sims: int, seed: int, dt: float = 1 / 365
) -> np.ndarray:
    """Monte Carlo P(>= k events in [0, T])."""
    rng = np.random.default_rng(seed)
    n = pop.n
    n_steps = int(round(T / dt))
    catastrophic = np.zeros(n)
    for _ in range(n_sims):
        state = np.zeros(n, dtype=np.int8)
        events = np.zeros(n, dtype=np.int32)
        for _t in range(n_steps):
            rate = np.where(state == 0, pop.lambda_0, pop.lambda_1)
            ev = rng.random(n) < rate * dt
            events += ev
            becomes_vuln = ev & (state == 0) & (rng.random(n) < pop.beta)
            state[becomes_vuln] = 1
            recovers = (state == 1) & (rng.random(n) < pop.mu * dt)
            state[recovers] = 0
        catastrophic += events >= k
    return catastrophic / n_sims


def misordering_fraction(
    r: np.ndarray, R: np.ndarray, n_pairs: int = 500_000, seed: int = 0
) -> float:
    """Pairwise discordance Delta = P(r_i > r_j and R_i < R_j) + reverse."""
    rng = np.random.default_rng(seed)
    n = len(r)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    valid = (i != j) & (r[i] != r[j]) & (R[i] != R[j])
    i, j = i[valid], j[valid]
    discordant = ((r[i] > r[j]) & (R[i] < R[j])) | ((r[i] < r[j]) & (R[i] > R[j]))
    return float(discordant.mean()) if len(i) > 0 else 0.0


# ------------------------------------------------------------------
# Unified evaluator: search methods (random, BO, CMA-ES, agents)
# all call this with a parameter dict and fidelity tuple.
# ------------------------------------------------------------------

CALIBRATED_PARAMS: Dict[str, float] = {
    "lambda_0_shape": 3.0,
    "lambda_0_scale": 0.20,
    "lambda_1_mult_shape": 2.0,
    "lambda_1_mult_scale": 0.8,
    "beta_a": 3.0,
    "beta_b": 7.0,
    "mu_shape": 4.0,
    "mu_scale": 1.0,
}

# Parameter space bounds shared by all search methods.
PARAM_BOUNDS = {
    "lambda_0_shape": (1.0, 8.0),
    "lambda_0_scale": (0.05, 0.35),
    "lambda_1_mult_shape": (0.5, 5.0),
    "lambda_1_mult_scale": (0.3, 3.0),
    "beta_a": (0.5, 5.0),
    "beta_b": (0.5, 8.0),
    "mu_shape": (1.0, 6.0),
    "mu_scale": (0.05, 2.0),
}


def evaluate(
    params: Dict,
    n: int = 5_000,
    n_sims: int = 500,
    T: float = 2.0,
    k: int = 3,
    seed: int = 42,
) -> Dict:
    pop = build_population(params, n=n, seed=seed)
    r = standard_score(pop)
    R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed + 1)
    delta = misordering_fraction(r, R, seed=seed + 2)
    return {
        "delta": delta,
        "pop_mean_beta": float(pop.beta.mean()),
        "pop_var_beta": float(pop.beta.var()),
        "pop_mean_lambda_0": float(pop.lambda_0.mean()),
        "pop_var_lambda_0": float(pop.lambda_0.var()),
        "pop_mean_mu": float(pop.mu.mean()),
        "pop_corr_r_R": float(np.corrcoef(r, R)[0, 1]),
    }


def calibrated_baseline(n: int = 5_000, n_sims: int = 500, seed: int = 42) -> Dict:
    return evaluate(CALIBRATED_PARAMS, n=n, n_sims=n_sims, seed=seed)
