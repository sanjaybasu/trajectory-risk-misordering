"""
The Misordering Problem in Clinical Risk Prediction
====================================================

Core microsimulation and misordering metric (Δ).

Given a population of patients with heterogeneous state-dependent health dynamics,
we compute two risk quantities for each patient:

  r_i = E[events in Δt | current state]  (standard risk score — ensemble average)
  R_i = P(trajectory ∈ catastrophic set)  (trajectory risk — path-level tail probability)

The misordering fraction Δ = P(r_i > r_j AND R_i < R_j) measures how often
standard risk scores give the WRONG rank ordering for trajectory-level decisions.

Theorem: For state-dependent dynamics (β > 0), Δ > 0 and grows with Var(β).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy import stats


# ---------------------------------------------------------------------------
# Population model
# ---------------------------------------------------------------------------

@dataclass
class Population:
    """Patient population with heterogeneous self-exciting dynamics."""
    n: int
    lambda_0: np.ndarray   # baseline event rate (stable state), events/year
    lambda_1: np.ndarray   # elevated event rate (vulnerable state), events/year
    beta: np.ndarray       # state-dependence: P(stable → vulnerable | event)
    mu: np.ndarray         # recovery rate (vulnerable → stable), per year
    group: np.ndarray      # demographic group label (for equity analysis)
    group_names: dict       # group label → name


def generate_population(
    n: int = 10_000,
    seed: int = 42,
    equity_structure: bool = True,
) -> Population:
    """
    Generate a heterogeneous Medicaid-like population.

    If equity_structure=True, creates demographic groups with systematically
    different β distributions (reflecting differential recovery buffers).
    """
    rng = np.random.default_rng(seed)

    if equity_structure:
        # Four demographic groups with different state-dependence distributions
        # Group 0: Advantaged (lower β — more recovery buffers)
        # Group 1-3: Disadvantaged subgroups (higher β — fewer buffers)
        group_props = [0.45, 0.25, 0.20, 0.10]
        group_names = {0: "Advantaged", 1: "Disadvantaged-A",
                       2: "Disadvantaged-B", 3: "Disadvantaged-C"}
        group_beta_params = [
            (2.0, 5.0),   # mean β ≈ 0.29 — low state-dependence
            (3.0, 3.5),   # mean β ≈ 0.46 — moderate
            (3.5, 3.0),   # mean β ≈ 0.54 — high
            (4.0, 2.5),   # mean β ≈ 0.62 — very high
        ]
        group_mu_scale = [1.0, 0.75, 0.65, 0.55]  # recovery rate scaling

        # Assign groups
        group = rng.choice(len(group_props), size=n, p=group_props)

        # Generate β per group
        beta = np.zeros(n)
        for g, (a, b) in enumerate(group_beta_params):
            mask = group == g
            beta[mask] = rng.beta(a, b, mask.sum())

        # Recovery rate: lower for high-β groups
        mu_base = rng.gamma(4, 0.5, n)  # base recovery ~ 2/year
        mu = np.zeros(n)
        for g, scale in enumerate(group_mu_scale):
            mask = group == g
            mu[mask] = mu_base[mask] * scale
    else:
        group = np.zeros(n, dtype=int)
        group_names = {0: "All"}
        beta = rng.beta(2.5, 3.5, n)
        mu = rng.gamma(4, 0.5, n)

    # Baseline event rate: similar across groups (the point is that r looks similar
    # but trajectory risk R diverges due to different β)
    lambda_0 = rng.gamma(2.5, 0.08, n)  # mean ~ 0.20 events/year
    lambda_1 = lambda_0 * (1 + rng.gamma(3, 0.8, n))  # elevated: 1.4-6x baseline

    return Population(n, lambda_0, lambda_1, beta, mu, group, group_names)


# ---------------------------------------------------------------------------
# Standard risk score: r
# ---------------------------------------------------------------------------

def compute_standard_risk(pop: Population) -> np.ndarray:
    """
    Compute the standard (ensemble-average) risk score.

    r_i = steady-state event rate under the Markov chain.
    At steady state: π_vuln = β·λ₀ / (β·λ₀ + μ)
    r = λ₀·(1 - π_vuln) + λ₁·π_vuln
    """
    pi_vuln = pop.beta * pop.lambda_0 / (pop.beta * pop.lambda_0 + pop.mu + 1e-12)
    r = pop.lambda_0 * (1 - pi_vuln) + pop.lambda_1 * pi_vuln
    return r


# ---------------------------------------------------------------------------
# Trajectory risk: R (via Monte Carlo microsimulation)
# ---------------------------------------------------------------------------

def simulate_trajectory_risk(
    pop: Population,
    T: float = 2.0,
    k: int = 3,
    n_sims: int = 1000,
    dt: float = 1/365,   # daily time steps
    seed: int = 42,
) -> np.ndarray:
    """
    Compute trajectory risk R_i = P(≥k events over [0, T]) via Monte Carlo.

    Simulates full state-dependent trajectories: each adverse event can
    transition the patient to a vulnerable state where future events are
    more likely (self-exciting / Hawkes-like dynamics).
    """
    rng = np.random.default_rng(seed)
    n = pop.n
    n_steps = int(T / dt)
    catastrophic = np.zeros(n)

    for sim in range(n_sims):
        state = np.zeros(n, dtype=np.int8)  # 0=stable, 1=vulnerable
        events = np.zeros(n, dtype=np.int32)

        for t in range(n_steps):
            # Event rate depends on current state
            rate = np.where(state == 0, pop.lambda_0, pop.lambda_1)

            # Event occurs with probability rate * dt
            event_occurs = rng.random(n) < rate * dt

            events += event_occurs

            # State transitions: stable → vulnerable with probability β on event
            becomes_vuln = event_occurs & (state == 0) & (rng.random(n) < pop.beta)
            state[becomes_vuln] = 1

            # Recovery: vulnerable → stable with probability μ * dt
            recovers = (state == 1) & (rng.random(n) < pop.mu * dt)
            state[recovers] = 0

        catastrophic += (events >= k)

    R = catastrophic / n_sims
    return R


# ---------------------------------------------------------------------------
# The misordering metric: Δ
# ---------------------------------------------------------------------------

def compute_delta(
    r: np.ndarray,
    R: np.ndarray,
    n_pairs: int = 500_000,
    seed: int = 42,
) -> Dict:
    """
    Compute the misordering fraction Δ.

    Δ = P(r_i > r_j AND R_i < R_j)  over random pairs (i, j).

    Δ = 0 means standard risk score perfectly preserves trajectory risk ranking.
    Δ > 0 means standard risk score gives wrong ordering for that fraction of pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(r)

    i_idx = rng.integers(0, n, n_pairs)
    j_idx = rng.integers(0, n, n_pairs)

    # Remove self-pairs and ties
    valid = (i_idx != j_idx) & (r[i_idx] != r[j_idx]) & (R[i_idx] != R[j_idx])
    i_idx, j_idx = i_idx[valid], j_idx[valid]

    r_i_gt = r[i_idx] > r[j_idx]
    R_i_lt = R[i_idx] < R[j_idx]

    discordant = r_i_gt & R_i_lt       # r says i > j, but R says i < j
    reverse_disc = (~r_i_gt) & (~R_i_lt)  # r says j > i, but R says j < i
    total_discordant = discordant.sum() + reverse_disc.sum()
    n_valid = len(i_idx)

    delta = total_discordant / n_valid if n_valid > 0 else 0.0
    tau = 1 - 2 * delta  # Kendall's τ

    return {
        "delta": float(delta),
        "kendall_tau": float(tau),
        "n_discordant": int(total_discordant),
        "n_valid_pairs": int(n_valid),
    }


def compute_delta_by_group(
    r: np.ndarray,
    R: np.ndarray,
    pop: Population,
    n_pairs_per_group: int = 200_000,
    seed: int = 42,
) -> Dict:
    """Compute Δ stratified by demographic group."""
    results = {}
    rng = np.random.default_rng(seed)

    for g, name in pop.group_names.items():
        mask = pop.group == g
        if mask.sum() < 50:
            continue
        idx = np.where(mask)[0]
        r_g, R_g = r[idx], R[idx]
        result = compute_delta(r_g, R_g, n_pairs=n_pairs_per_group, seed=seed + g)
        result["group"] = name
        result["n_patients"] = int(mask.sum())
        result["mean_beta"] = float(pop.beta[mask].mean())
        result["mean_r"] = float(r_g.mean())
        result["mean_R"] = float(R_g.mean())
        results[name] = result

    return results


# ---------------------------------------------------------------------------
# Missed catastrophes: the headline number
# ---------------------------------------------------------------------------

def compute_missed_catastrophes(
    r: np.ndarray,
    R: np.ndarray,
    r_threshold_pctl: float = 50.0,
    R_threshold_pctl: float = 90.0,
) -> Dict:
    """
    Among patients with truly high trajectory risk (R ≥ 90th percentile),
    what fraction are ranked below median by the standard risk score?

    This is the 'headline number': the fraction of catastrophic trajectories
    that are invisible to standard risk-score-based care management.
    """
    r_cutoff = np.percentile(r, r_threshold_pctl)
    R_cutoff = np.percentile(R, R_threshold_pctl)

    truly_high_risk = R >= R_cutoff
    ranked_below_median = r < r_cutoff
    missed = truly_high_risk & ranked_below_median

    n_truly_high = truly_high_risk.sum()
    n_missed = missed.sum()
    frac_missed = n_missed / n_truly_high if n_truly_high > 0 else 0.0

    return {
        "frac_missed": float(frac_missed),
        "n_truly_high_risk": int(n_truly_high),
        "n_missed_by_standard_score": int(n_missed),
        "r_cutoff": float(r_cutoff),
        "R_cutoff": float(R_cutoff),
    }


# ---------------------------------------------------------------------------
# Quick baseline computation
# ---------------------------------------------------------------------------

def run_baseline(n: int = 5000, n_sims: int = 500, T: float = 2.0, k: int = 3,
                 seed: int = 42, verbose: bool = True) -> Dict:
    """Run the full pipeline and return all results."""
    if verbose:
        print(f"Generating population (n={n})...")
    pop = generate_population(n=n, seed=seed)

    if verbose:
        print("Computing standard risk scores...")
    r = compute_standard_risk(pop)

    if verbose:
        print(f"Simulating trajectories (n_sims={n_sims}, T={T}, k={k})...")
    R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed)

    if verbose:
        print("Computing misordering Δ...")
    delta_overall = compute_delta(r, R)
    delta_by_group = compute_delta_by_group(r, R, pop)
    missed = compute_missed_catastrophes(r, R)

    results = {
        "overall": delta_overall,
        "by_group": delta_by_group,
        "missed_catastrophes": missed,
        "population_stats": {
            "n": n,
            "mean_beta": float(pop.beta.mean()),
            "std_beta": float(pop.beta.std()),
            "mean_r": float(r.mean()),
            "mean_R": float(R.mean()),
            "corr_r_R": float(np.corrcoef(r, R)[0, 1]),
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: The Misordering Problem")
        print(f"{'='*60}")
        print(f"Population: n={n}, mean(β)={pop.beta.mean():.3f}, std(β)={pop.beta.std():.3f}")
        print(f"Correlation r vs R: {np.corrcoef(r, R)[0,1]:.4f}")
        print(f"\nOverall Δ = {delta_overall['delta']:.4f}")
        print(f"  (Kendall τ = {delta_overall['kendall_tau']:.4f})")
        print(f"\nMissed catastrophes: {missed['frac_missed']:.1%} of high-trajectory-risk")
        print(f"  patients ranked below median by standard score")
        print(f"\nΔ by demographic group:")
        for name, res in delta_by_group.items():
            print(f"  {name:20s}: Δ={res['delta']:.4f}, mean(β)={res['mean_beta']:.3f}, "
                  f"mean(R)={res['mean_R']:.3f}")

    return results


if __name__ == "__main__":
    results = run_baseline(n=5000, n_sims=500, T=2.0, k=3)
