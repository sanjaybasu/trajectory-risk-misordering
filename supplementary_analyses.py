#!/usr/bin/env python3
"""
Supplementary Analyses for Peer Review Revision
=================================================

Addresses methodological issues identified in simulated peer review:

1. Expanded random search (200 configs instead of 15) — Reviewer 1
2. Multi-seed stability analysis replacing broken bootstrap coverage — Reviewers 1, 4, 5
   a. Population variability: Δ across 20 different population seeds
   b. MC variability: Δ across 20 different simulation seeds (same population)
3. Random search uses same N/n_sims as primary — Reviewer 4

Outputs: results/supplementary_analyses.json
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (
    Population, compute_standard_risk, simulate_trajectory_risk,
    compute_delta,
)
from revised_analysis import (
    generate_calibrated_population,
    compute_trajectory_aware_score,
)


def run_expanded_random_search(
    n_configs: int = 200,
    n: int = 5000,
    n_sims: int = 500,
    seed: int = 42,
) -> dict:
    """
    Expanded random search comparison (200 configs, same N/n_sims as primary).

    Addresses Reviewer 1 (underpowered comparison) and Reviewer 4
    (random search used smaller N/n_sims than primary).
    """
    print("=" * 70)
    print(f"EXPANDED RANDOM SEARCH: {n_configs} configs (N={n}, n_sims={n_sims})")
    print("=" * 70)

    rng = np.random.default_rng(seed + 999)

    results = []
    for i in range(n_configs):
        lambda_0_shape = rng.uniform(1.5, 8.0)
        lambda_0_scale = rng.uniform(0.05, 0.4)
        lambda_1_mult_shape = rng.uniform(1.0, 6.0)
        lambda_1_mult_scale = rng.uniform(0.3, 2.0)
        beta_a = rng.uniform(0.5, 5.0)
        beta_b = rng.uniform(0.5, 8.0)
        mu_shape = rng.uniform(1.0, 6.0)
        mu_scale = rng.uniform(0.1, 2.0)
        k = int(rng.choice([1, 2, 3, 5]))
        T = float(rng.choice([0.5, 1.0, 2.0, 5.0]))

        pop_rng = np.random.default_rng(seed + i)
        lambda_0 = pop_rng.gamma(lambda_0_shape, lambda_0_scale, n)
        lambda_1 = lambda_0 * (1 + pop_rng.gamma(lambda_1_mult_shape, lambda_1_mult_scale, n))
        beta = pop_rng.beta(beta_a, beta_b, n)
        mu = pop_rng.gamma(mu_shape, mu_scale, n)
        group = pop_rng.choice(4, size=n, p=[0.45, 0.25, 0.20, 0.10])
        group_names = {0: "A", 1: "B", 2: "C", 3: "D"}

        pop = Population(n, lambda_0, lambda_1, beta, mu, group, group_names)
        r = compute_standard_risk(pop)
        R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed + i)
        delta = compute_delta(r, R)

        results.append({
            "config_idx": i,
            "delta": delta["delta"],
            "params": {
                "lambda_0_shape": float(lambda_0_shape),
                "lambda_0_scale": float(lambda_0_scale),
                "beta_a": float(beta_a),
                "beta_b": float(beta_b),
                "mu_shape": float(mu_shape),
                "mu_scale": float(mu_scale),
                "k": int(k),
                "T": float(T),
            },
        })
        if (i + 1) % 20 == 0:
            best_so_far = max(r["delta"] for r in results)
            print(f"  {i+1}/{n_configs} configs done. Best Δ so far: {best_so_far:.4f}")

    deltas = [r["delta"] for r in results]
    best_random = max(results, key=lambda x: x["delta"])

    # How many exceed agent thresholds?
    n_exceed_0_1 = sum(1 for d in deltas if d > 0.10)
    n_exceed_0_2 = sum(1 for d in deltas if d > 0.20)
    n_exceed_0_3 = sum(1 for d in deltas if d > 0.30)

    print(f"\n  Best random: Δ={best_random['delta']:.4f}")
    print(f"  Configs with Δ>0.10: {n_exceed_0_1}/{n_configs}")
    print(f"  Configs with Δ>0.20: {n_exceed_0_2}/{n_configs}")
    print(f"  Configs with Δ>0.30: {n_exceed_0_3}/{n_configs}")

    return {
        "random_search_results": results,
        "best_delta": best_random["delta"],
        "best_config": best_random,
        "n_configs": n_configs,
        "n_patients": n,
        "n_sims": n_sims,
        "summary": {
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "sd_delta": float(np.std(deltas)),
            "p25_delta": float(np.percentile(deltas, 25)),
            "p75_delta": float(np.percentile(deltas, 75)),
            "max_delta": float(np.max(deltas)),
            "n_exceed_0_10": n_exceed_0_1,
            "n_exceed_0_20": n_exceed_0_2,
            "n_exceed_0_30": n_exceed_0_3,
        },
        "agent_comparison": {
            "agent_r1_best": 0.222,
            "agent_r2_best": 0.461,
            "agent_validated_worst": 0.312,
            "random_best": best_random["delta"],
        },
    }


def run_multi_seed_stability(
    n_seeds: int = 20,
    n: int = 5000,
    n_sims: int = 500,
    T: float = 2.0,
    k: int = 3,
    base_seed: int = 42,
) -> dict:
    """
    Multi-seed stability analysis replacing broken bootstrap CI coverage.

    Separately quantifies two sources of variability in Δ:
    1. Population variability: different draws from the parameter distributions
    2. MC variability: same population, different simulation RNG seeds

    Addresses Reviewers 1, 4, 5.
    """
    print("\n" + "=" * 70)
    print("MULTI-SEED STABILITY ANALYSIS")
    print("=" * 70)

    # --- Source 1: Population variability ---
    print(f"\n  Population variability ({n_seeds} seeds)...")
    pop_deltas = []
    pop_deltas_traj = []
    for i in range(n_seeds):
        seed = base_seed + i * 1000
        pop, _ = generate_calibrated_population(n=n, seed=seed, config="primary")
        r = compute_standard_risk(pop)
        R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed)
        delta_std = compute_delta(r, R)["delta"]

        R_a = compute_trajectory_aware_score(pop, T=T, k=k)
        delta_traj = compute_delta(R_a, R)["delta"]

        pop_deltas.append(delta_std)
        pop_deltas_traj.append(delta_traj)
        print(f"    Seed {i+1}/{n_seeds}: Δ_std={delta_std:.4f}, Δ_traj={delta_traj:.4f}")

    # --- Source 2: MC variability (same population, different sim seeds) ---
    print(f"\n  MC variability ({n_seeds} seeds, fixed population)...")
    pop_fixed, _ = generate_calibrated_population(n=n, seed=base_seed, config="primary")
    r_fixed = compute_standard_risk(pop_fixed)
    R_a_fixed = compute_trajectory_aware_score(pop_fixed, T=T, k=k)

    mc_deltas = []
    mc_deltas_traj = []
    for i in range(n_seeds):
        mc_seed = base_seed + 50000 + i * 100
        R = simulate_trajectory_risk(pop_fixed, T=T, k=k, n_sims=n_sims, seed=mc_seed)
        delta_std = compute_delta(r_fixed, R)["delta"]
        delta_traj = compute_delta(R_a_fixed, R)["delta"]
        mc_deltas.append(delta_std)
        mc_deltas_traj.append(delta_traj)
        print(f"    MC seed {i+1}/{n_seeds}: Δ_std={delta_std:.4f}, Δ_traj={delta_traj:.4f}")

    # --- Source 3: High-precision estimate (same pop, very large n_sims) ---
    print(f"\n  High-precision estimate (n_sims=5000)...")
    R_precise = simulate_trajectory_risk(pop_fixed, T=T, k=k, n_sims=5000, seed=base_seed)
    delta_precise = compute_delta(r_fixed, R_precise)["delta"]
    delta_precise_traj = compute_delta(R_a_fixed, R_precise)["delta"]
    print(f"    Δ_std(n_sims=5000)={delta_precise:.4f}, Δ_traj={delta_precise_traj:.4f}")

    pop_deltas = np.array(pop_deltas)
    pop_deltas_traj = np.array(pop_deltas_traj)
    mc_deltas = np.array(mc_deltas)
    mc_deltas_traj = np.array(mc_deltas_traj)

    return {
        "population_variability": {
            "standard": {
                "deltas": pop_deltas.tolist(),
                "mean": float(pop_deltas.mean()),
                "sd": float(pop_deltas.std()),
                "iqr": [float(np.percentile(pop_deltas, 25)),
                        float(np.percentile(pop_deltas, 75))],
                "range": [float(pop_deltas.min()), float(pop_deltas.max())],
            },
            "trajectory_aware": {
                "deltas": pop_deltas_traj.tolist(),
                "mean": float(pop_deltas_traj.mean()),
                "sd": float(pop_deltas_traj.std()),
                "iqr": [float(np.percentile(pop_deltas_traj, 25)),
                        float(np.percentile(pop_deltas_traj, 75))],
                "range": [float(pop_deltas_traj.min()), float(pop_deltas_traj.max())],
            },
        },
        "mc_variability": {
            "standard": {
                "deltas": mc_deltas.tolist(),
                "mean": float(mc_deltas.mean()),
                "sd": float(mc_deltas.std()),
                "iqr": [float(np.percentile(mc_deltas, 25)),
                        float(np.percentile(mc_deltas, 75))],
                "range": [float(mc_deltas.min()), float(mc_deltas.max())],
            },
            "trajectory_aware": {
                "deltas": mc_deltas_traj.tolist(),
                "mean": float(mc_deltas_traj.mean()),
                "sd": float(mc_deltas_traj.std()),
                "iqr": [float(np.percentile(mc_deltas_traj, 25)),
                        float(np.percentile(mc_deltas_traj, 75))],
                "range": [float(mc_deltas_traj.min()), float(mc_deltas_traj.max())],
            },
        },
        "high_precision": {
            "n_sims": 5000,
            "delta_standard": float(delta_precise),
            "delta_trajectory_aware": float(delta_precise_traj),
        },
        "n_seeds": n_seeds,
        "n_patients": n,
        "n_sims": n_sims,
        "T": T,
        "k": k,
    }


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    # 1. Expanded random search
    random_search = run_expanded_random_search(
        n_configs=200, n=5000, n_sims=500, seed=42,
    )

    # 2. Multi-seed stability
    stability = run_multi_seed_stability(
        n_seeds=20, n=5000, n_sims=500, T=2.0, k=3, base_seed=42,
    )

    # Save results
    results = {
        "expanded_random_search": random_search,
        "multi_seed_stability": stability,
    }
    with open("results/supplementary_analyses.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Results saved to results/supplementary_analyses.json")
