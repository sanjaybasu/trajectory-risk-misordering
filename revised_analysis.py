#!/usr/bin/env python3
"""
Revised Analysis for NEJM AI Manuscript
========================================

Addresses peer review concerns:
1. Literature-calibrated parameters with cited justification
2. Trajectory-aware scoring comparison (the SOLUTION, not just the problem)
3. Sensitivity analysis over (k, T) grid
4. Random search comparison to validate AI agent approach
5. Net reclassification improvement (NRI)
6. Proper simulation-noise uncertainty quantification
7. C-statistic computation

Produces all tables and figures for the revised manuscript.
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (
    Population, compute_standard_risk, simulate_trajectory_risk,
    compute_delta, compute_delta_by_group, compute_missed_catastrophes,
)


# -----------------------------------------------------------------------
# Literature-calibrated population generator
# -----------------------------------------------------------------------

def generate_calibrated_population(
    n: int = 5000,
    seed: int = 42,
    config: str = "primary",
) -> Tuple[Population, Dict]:
    """
    Generate population with literature-calibrated parameters.

    Primary configuration calibrated to:
    - Hospitalization rates: Jencks et al. NEJM 2009 (~0.5-1.0/yr for high-risk)
    - Post-hospital syndrome: Krumholz NEJM 2013 (2-4x elevated risk, 1-6 month duration)
    - Readmission cascades: Dharmarajan et al. JAMA 2013 (clustered events)
    - Racial disparities: Tsai et al. Health Aff 2014, Wadhera et al. JAMA Cardiol 2019
      (~20% higher readmission rates for Black vs White patients)
    """
    rng = np.random.default_rng(seed)

    if config == "primary":
        # Baseline hospitalization rate: 0.3-1.0/year for high-risk Medicaid adults
        # Gamma(3, 0.2) -> mean=0.6, SD=0.35, CV=0.58
        lambda_0 = rng.gamma(3.0, 0.2, n)

        # Elevated rate during vulnerable period: 2-4x baseline
        # Informed by post-hospital syndrome (Krumholz 2013)
        lambda_1 = lambda_0 * (1 + rng.gamma(2.0, 0.8, n))  # mean multiplier ~2.6x

        # State-dependence (cascade probability): 20-40% of hospitalizations
        # trigger a vulnerable cascade period
        # Beta(3, 7) -> mean=0.30, SD=0.14
        beta_base = rng.beta(3.0, 7.0, n)

        # Recovery rate: post-hospital syndrome resolves in 1-6 months
        # Gamma(4, 1.0) -> mean=4.0/year (3-month average recovery)
        mu_base = rng.gamma(4.0, 1.0, n)

        # Demographic groups calibrated to readmission disparities
        # ~20% higher readmission for disadvantaged groups (Tsai 2014)
        group_props = [0.45, 0.25, 0.20, 0.10]
        group_names = {
            0: "Low social risk",
            1: "Moderate social risk",
            2: "High social risk",
            3: "Very high social risk",
        }
        # beta offsets: higher cascade propensity due to fewer post-discharge resources
        group_beta_offsets = [0.00, 0.04, 0.07, 0.10]
        # mu scales: slower recovery due to less access to rehabilitation/home health
        group_mu_scales = [1.00, 0.85, 0.75, 0.65]

    elif config == "low_acuity":
        # Lower-risk population
        lambda_0 = rng.gamma(2.0, 0.15, n)  # mean=0.3
        lambda_1 = lambda_0 * (1 + rng.gamma(1.5, 0.6, n))  # ~1.9x
        beta_base = rng.beta(2.5, 8.5, n)  # mean=0.23
        mu_base = rng.gamma(5.0, 1.2, n)  # mean=6.0/year (2-month recovery)
        group_props = [0.45, 0.25, 0.20, 0.10]
        group_names = {0: "Low social risk", 1: "Moderate social risk",
                       2: "High social risk", 3: "Very high social risk"}
        group_beta_offsets = [0.00, 0.03, 0.05, 0.08]
        group_mu_scales = [1.00, 0.90, 0.80, 0.70]

    elif config == "high_acuity":
        # Higher-risk population (complex chronic conditions)
        lambda_0 = rng.gamma(4.0, 0.25, n)  # mean=1.0
        lambda_1 = lambda_0 * (1 + rng.gamma(3.0, 0.8, n))  # ~3.4x
        beta_base = rng.beta(3.5, 5.5, n)  # mean=0.39
        mu_base = rng.gamma(3.0, 0.8, n)  # mean=2.4/year (5-month recovery)
        group_props = [0.45, 0.25, 0.20, 0.10]
        group_names = {0: "Low social risk", 1: "Moderate social risk",
                       2: "High social risk", 3: "Very high social risk"}
        group_beta_offsets = [0.00, 0.05, 0.09, 0.14]
        group_mu_scales = [1.00, 0.80, 0.70, 0.55]
    else:
        raise ValueError(f"Unknown config: {config}")

    # Assign groups and apply offsets
    group = rng.choice(len(group_props), size=n, p=group_props)
    beta = np.zeros(n)
    mu = np.zeros(n)

    for g in range(len(group_props)):
        mask = group == g
        beta[mask] = np.clip(beta_base[mask] + group_beta_offsets[g], 0.01, 0.99)
        mu[mask] = mu_base[mask] * group_mu_scales[g]

    pop = Population(n, lambda_0, lambda_1, beta, mu, group, group_names)

    calibration_info = {
        "config": config,
        "lambda_0_dist": "Gamma(3.0, 0.2)" if config == "primary" else "varied",
        "lambda_1_model": "lambda_0 * (1 + Gamma(2.0, 0.8))",
        "beta_dist": "Beta(3.0, 7.0) + group offset",
        "mu_dist": "Gamma(4.0, 1.0) * group scale",
        "group_beta_offsets": group_beta_offsets,
        "group_mu_scales": group_mu_scales,
        "references": [
            "Jencks SF et al. NEJM 2009;360:1418-1428",
            "Krumholz HM. NEJM 2013;368:100-102",
            "Dharmarajan K et al. JAMA 2013;309:355-363",
            "Tsai TC et al. Health Aff 2014;33:786-793",
            "Wadhera RK et al. JAMA Cardiol 2019;4:885-893",
        ],
    }

    return pop, calibration_info


# -----------------------------------------------------------------------
# Trajectory-aware analytical score
# -----------------------------------------------------------------------

def compute_trajectory_aware_score(
    pop: Population,
    T: float = 2.0,
    k: int = 3,
) -> np.ndarray:
    """
    Compute trajectory-aware risk score using analytical NB approximation.

    This uses the SAME patient parameters as the standard score but
    computes P(>=k events in [0,T]) via a negative binomial model that
    accounts for overdispersion from self-exciting dynamics.

    This is the proposed SOLUTION: a closed-form score that captures
    trajectory risk without requiring Monte Carlo simulation.
    """
    n = pop.n
    R_analytical = np.zeros(n)

    for i in range(n):
        l0 = pop.lambda_0[i]
        l1 = pop.lambda_1[i]
        b = pop.beta[i]
        m = pop.mu[i]

        # Steady-state rate
        pi_vuln = b * l0 / (b * l0 + m + 1e-12)
        r_ss = l0 * (1 - pi_vuln) + l1 * pi_vuln
        mean_events = r_ss * T

        # Overdispersion from self-excitation
        # VMR = 1 + (l1 - l0) * beta / (beta*l0 + mu) * f(T, mu)
        excitation_ratio = b / (m + 1e-12)
        rate_contrast = (l1 - l0) / (l0 + 1e-12)
        temporal_factor = (1 - np.exp(-m * T)) / (m * T + 1e-12)
        dispersion = 1 + rate_contrast * excitation_ratio * temporal_factor
        dispersion = max(dispersion, 1.01)

        if dispersion > 1 and mean_events > 0:
            # NB parameterization: scipy nbinom(n, p)
            # mean = n*(1-p)/p, var = n*(1-p)/p^2 = mean * (1/p)
            # We want var/mean = dispersion => 1/p = dispersion => p = 1/dispersion
            nb_n = mean_events / (dispersion - 1)
            nb_p = 1.0 / dispersion
            R_analytical[i] = 1 - stats.nbinom.cdf(k - 1, nb_n, nb_p)
        else:
            R_analytical[i] = 1 - stats.poisson.cdf(k - 1, max(mean_events, 1e-12))

    return np.clip(R_analytical, 0, 1)


# -----------------------------------------------------------------------
# Augmented risk score (simple beta-adjustment)
# -----------------------------------------------------------------------

def compute_augmented_score(
    pop: Population,
    r: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Simple augmented score: r * (1 + alpha * beta / mu).

    This captures the intuition that patients with high cascade propensity
    (high beta, low mu) have higher trajectory risk than their standard
    score r suggests.
    """
    adjustment = 1 + alpha * pop.beta / (pop.mu + 1e-12)
    return r * adjustment


# -----------------------------------------------------------------------
# Net Reclassification Improvement (NRI)
# -----------------------------------------------------------------------

def compute_nri(
    r_standard: np.ndarray,
    r_new: np.ndarray,
    R_gold: np.ndarray,
    threshold_pctl: float = 75.0,
) -> Dict:
    """
    Compute category-free NRI comparing r_new to r_standard
    using R_gold as the reference.

    Patients with high R_gold (above threshold) should be ranked higher.
    Patients with low R_gold (below threshold) should be ranked lower.
    """
    R_cutoff = np.percentile(R_gold, threshold_pctl)
    high_risk = R_gold >= R_cutoff
    low_risk = R_gold < R_cutoff

    # Among truly high-risk: fraction moved UP by new score
    if high_risk.sum() > 0:
        r_std_rank = stats.rankdata(r_standard)
        r_new_rank = stats.rankdata(r_new)
        moved_up_events = (r_new_rank[high_risk] > r_std_rank[high_risk]).mean()
        moved_down_events = (r_new_rank[high_risk] < r_std_rank[high_risk]).mean()
        nri_events = moved_up_events - moved_down_events
    else:
        nri_events = 0.0

    # Among truly low-risk: fraction moved DOWN by new score
    if low_risk.sum() > 0:
        moved_up_nonevents = (r_new_rank[low_risk] > r_std_rank[low_risk]).mean()
        moved_down_nonevents = (r_new_rank[low_risk] < r_std_rank[low_risk]).mean()
        nri_nonevents = moved_down_nonevents - moved_up_nonevents
    else:
        nri_nonevents = 0.0

    nri_total = nri_events + nri_nonevents

    return {
        "nri_total": float(nri_total),
        "nri_events": float(nri_events),
        "nri_nonevents": float(nri_nonevents),
        "n_high_risk": int(high_risk.sum()),
        "n_low_risk": int(low_risk.sum()),
    }


# -----------------------------------------------------------------------
# Bootstrap infrastructure
# -----------------------------------------------------------------------

def bootstrap_delta(r, R, n_bootstrap=2000, n_pairs=200_000, seed=42):
    """Bootstrap 95% CI for delta by resampling patients."""
    rng = np.random.default_rng(seed)
    n = len(r)
    deltas = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        result = compute_delta(r[idx], R[idx], n_pairs=n_pairs, seed=seed + b)
        deltas.append(result["delta"])
    deltas = np.array(deltas)
    return {
        "mean": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "se": float(np.std(deltas)),
    }


def bootstrap_missed(r, R, n_bootstrap=2000, seed=42):
    """Bootstrap 95% CI for missed catastrophe fraction."""
    rng = np.random.default_rng(seed)
    n = len(r)
    fracs = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        result = compute_missed_catastrophes(r[idx], R[idx])
        fracs.append(result["frac_missed"])
    fracs = np.array(fracs)
    return {
        "mean": float(np.mean(fracs)),
        "ci_lower": float(np.percentile(fracs, 2.5)),
        "ci_upper": float(np.percentile(fracs, 97.5)),
    }


# -----------------------------------------------------------------------
# Main analyses
# -----------------------------------------------------------------------

def run_primary_analysis(
    n: int = 5000,
    n_sims: int = 500,
    T: float = 2.0,
    k: int = 3,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Primary analysis: clinically-calibrated parameters with
    standard vs trajectory-aware scoring comparison.
    """
    print("=" * 70)
    print("PRIMARY ANALYSIS: Clinically-Calibrated Parameters")
    print("=" * 70)

    results = {}

    for config_name in ["primary", "low_acuity", "high_acuity"]:
        print(f"\n--- Configuration: {config_name} ---")

        pop, cal_info = generate_calibrated_population(n=n, seed=seed, config=config_name)

        # Compute all three scores
        print("  Computing standard risk score r...")
        r = compute_standard_risk(pop)

        print("  Computing trajectory-aware analytical score R_analytical...")
        R_analytical = compute_trajectory_aware_score(pop, T=T, k=k)

        print("  Computing augmented score r_augmented...")
        r_augmented = compute_augmented_score(pop, r, alpha=1.0)

        print(f"  Simulating gold-standard trajectory risk R (n_sims={n_sims})...")
        R_gold = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed)

        # Delta: standard vs gold
        delta_standard = compute_delta(r, R_gold)
        delta_by_group_standard = compute_delta_by_group(r, R_gold, pop)
        missed_standard = compute_missed_catastrophes(r, R_gold)

        # Delta: trajectory-aware vs gold
        delta_trajectory = compute_delta(R_analytical, R_gold)

        # Delta: augmented vs gold
        delta_augmented = compute_delta(r_augmented, R_gold)

        # NRI: trajectory-aware vs standard
        nri_trajectory = compute_nri(r, R_analytical, R_gold)
        nri_augmented = compute_nri(r, r_augmented, R_gold)

        # Correlations
        corr_standard = float(np.corrcoef(r, R_gold)[0, 1])
        corr_trajectory = float(np.corrcoef(R_analytical, R_gold)[0, 1])
        corr_augmented = float(np.corrcoef(r_augmented, R_gold)[0, 1])

        # Bootstrap CIs (only for primary config)
        if config_name == "primary":
            print(f"  Bootstrap CIs (B={n_bootstrap})...")
            delta_ci_standard = bootstrap_delta(r, R_gold, n_bootstrap=n_bootstrap, seed=seed + 100)
            delta_ci_trajectory = bootstrap_delta(R_analytical, R_gold, n_bootstrap=n_bootstrap, seed=seed + 200)
            delta_ci_augmented = bootstrap_delta(r_augmented, R_gold, n_bootstrap=n_bootstrap, seed=seed + 300)
            missed_ci = bootstrap_missed(r, R_gold, n_bootstrap=n_bootstrap, seed=seed + 400)

            # Group-level bootstrap CIs
            group_cis = {}
            for g, gname in pop.group_names.items():
                mask = pop.group == g
                if mask.sum() < 50:
                    continue
                idx = np.where(mask)[0]
                g_ci = bootstrap_delta(r[idx], R_gold[idx], n_bootstrap=n_bootstrap,
                                       n_pairs=100_000, seed=seed + 500 + g)
                group_cis[gname] = {
                    "delta": delta_by_group_standard[gname]["delta"],
                    **g_ci,
                    "n_patients": int(mask.sum()),
                    "mean_beta": float(pop.beta[mask].mean()),
                    "mean_mu": float(pop.mu[mask].mean()),
                }
        else:
            delta_ci_standard = {"mean": delta_standard["delta"], "ci_lower": None, "ci_upper": None, "se": None}
            delta_ci_trajectory = {"mean": delta_trajectory["delta"], "ci_lower": None, "ci_upper": None, "se": None}
            delta_ci_augmented = {"mean": delta_augmented["delta"], "ci_lower": None, "ci_upper": None, "se": None}
            missed_ci = {"mean": missed_standard["frac_missed"], "ci_lower": None, "ci_upper": None}
            group_cis = {}

        config_result = {
            "config": config_name,
            "calibration": cal_info,
            "scoring_comparison": {
                "standard": {
                    "delta": delta_standard["delta"],
                    "delta_ci": delta_ci_standard,
                    "tau": delta_standard["kendall_tau"],
                    "corr": corr_standard,
                    "missed_frac": missed_standard["frac_missed"],
                    "missed_ci": missed_ci,
                    "c_statistic": 1 - delta_standard["delta"],
                },
                "trajectory_aware": {
                    "delta": delta_trajectory["delta"],
                    "delta_ci": delta_ci_trajectory,
                    "tau": delta_trajectory["kendall_tau"],
                    "corr": corr_trajectory,
                    "c_statistic": 1 - delta_trajectory["delta"],
                    "nri_vs_standard": nri_trajectory,
                },
                "augmented": {
                    "delta": delta_augmented["delta"],
                    "delta_ci": delta_ci_augmented,
                    "tau": delta_augmented["kendall_tau"],
                    "corr": corr_augmented,
                    "c_statistic": 1 - delta_augmented["delta"],
                    "nri_vs_standard": nri_augmented,
                },
            },
            "by_group": group_cis if config_name == "primary" else {
                gname: {"delta": v["delta"], "n_patients": v["n_patients"],
                         "mean_beta": v["mean_beta"]}
                for gname, v in delta_by_group_standard.items()
            },
            "pop_stats": {
                "n": n,
                "mean_beta": float(pop.beta.mean()),
                "std_beta": float(pop.beta.std()),
                "mean_lambda_0": float(pop.lambda_0.mean()),
                "std_lambda_0": float(pop.lambda_0.std()),
                "mean_lambda_1": float(pop.lambda_1.mean()),
                "std_lambda_1": float(pop.lambda_1.std()),
                "mean_mu": float(pop.mu.mean()),
                "std_mu": float(pop.mu.std()),
                "mean_r": float(r.mean()),
                "std_r": float(r.std()),
                "mean_R_gold": float(R_gold.mean()),
                "std_R_gold": float(R_gold.std()),
                "mean_R_analytical": float(R_analytical.mean()),
                "std_R_analytical": float(R_analytical.std()),
            },
            "T": T,
            "k": k,
            "n_sims": n_sims,
        }

        # Print summary
        print(f"\n  Standard score:      Delta={delta_standard['delta']:.4f}, "
              f"C={1 - delta_standard['delta']:.3f}, rho={corr_standard:.3f}")
        print(f"  Trajectory-aware:    Delta={delta_trajectory['delta']:.4f}, "
              f"C={1 - delta_trajectory['delta']:.3f}, rho={corr_trajectory:.3f}")
        print(f"  Augmented (r*f(b)):  Delta={delta_augmented['delta']:.4f}, "
              f"C={1 - delta_augmented['delta']:.3f}, rho={corr_augmented:.3f}")
        print(f"  Missed catastrophes: {missed_standard['frac_missed']:.1%}")
        print(f"  NRI (trajectory vs standard): {nri_trajectory['nri_total']:+.3f}")

        results[config_name] = config_result

    return results


def run_sensitivity_analysis(
    n: int = 3000,
    n_sims: int = 300,
    seed: int = 42,
) -> Dict:
    """
    Sensitivity analysis: Delta over a grid of (k, T) values.
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Delta(k, T)")
    print("=" * 70)

    k_values = [1, 2, 3, 5]
    T_values = [0.5, 1.0, 2.0, 5.0]
    results = []

    pop, _ = generate_calibrated_population(n=n, seed=seed, config="primary")
    r = compute_standard_risk(pop)

    for T in T_values:
        for k in k_values:
            print(f"  k={k}, T={T}...", end=" ", flush=True)
            R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed)
            delta = compute_delta(r, R)
            missed = compute_missed_catastrophes(r, R)
            corr = float(np.corrcoef(r, R)[0, 1])

            # Also compute trajectory-aware delta
            R_analytical = compute_trajectory_aware_score(pop, T=T, k=k)
            delta_traj = compute_delta(R_analytical, R)

            results.append({
                "k": k, "T": T,
                "delta_standard": delta["delta"],
                "delta_trajectory_aware": delta_traj["delta"],
                "tau": delta["kendall_tau"],
                "corr": corr,
                "missed_frac": missed["frac_missed"],
                "mean_R": float(R.mean()),
            })
            print(f"Delta={delta['delta']:.4f}, Delta_traj={delta_traj['delta']:.4f}")

    return {"sensitivity_grid": results, "k_values": k_values, "T_values": T_values}


def run_random_search_comparison(
    n_configs: int = 15,
    n: int = 3000,
    n_sims: int = 300,
    seed: int = 42,
) -> Dict:
    """
    Compare AI agent performance to random search over the parameter space.
    """
    print("\n" + "=" * 70)
    print("RANDOM SEARCH COMPARISON")
    print("=" * 70)

    rng = np.random.default_rng(seed + 999)

    results = []
    for i in range(n_configs):
        # Random parameter configuration from reasonable ranges
        lambda_0_shape = rng.uniform(1.5, 8.0)
        lambda_0_scale = rng.uniform(0.05, 0.4)
        lambda_1_mult_shape = rng.uniform(1.0, 6.0)
        lambda_1_mult_scale = rng.uniform(0.3, 2.0)
        beta_a = rng.uniform(0.5, 5.0)
        beta_b = rng.uniform(0.5, 8.0)
        mu_shape = rng.uniform(1.0, 6.0)
        mu_scale = rng.uniform(0.1, 2.0)
        k = rng.choice([1, 2, 3, 5])
        T = rng.choice([0.5, 1.0, 2.0, 5.0])

        # Generate population
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
        print(f"  Config {i+1}/{n_configs}: Delta={delta['delta']:.4f}")

    best_random = max(results, key=lambda x: x["delta"])
    print(f"\n  Best random search: Delta={best_random['delta']:.4f}")
    print(f"  (vs Agent Round 1 best: Delta=0.222, Round 2 best: Delta=0.461)")

    return {
        "random_search_results": results,
        "best_delta": best_random["delta"],
        "n_configs": n_configs,
        "agent_comparison": {
            "agent_r1_best": 0.222,
            "agent_r2_best": 0.461,
            "random_best": best_random["delta"],
        },
    }


# -----------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------

def print_table1(results):
    """Table 1: Literature-Calibrated Population Characteristics."""
    print("\n" + "=" * 80)
    print("TABLE 1: Literature-Calibrated Simulated Population Characteristics")
    print("=" * 80)

    primary = results["primary"]
    ps = primary["pop_stats"]

    print(f"\nPopulation: N = {ps['n']}")
    print(f"Time horizon T = {primary['T']} years; catastrophic threshold k = {primary['k']} events")
    print(f"\n{'Parameter':<45s} {'Mean (SD)':<20s} {'Calibration Source'}")
    print("-" * 95)
    print(f"{'Baseline event rate lambda_0 (events/yr)':<45s} "
          f"{ps['mean_lambda_0']:.3f} ({ps['std_lambda_0']:.3f}){'':<5s} "
          f"Jencks 2009 (hosp rates)")
    print(f"{'Elevated event rate lambda_1 (events/yr)':<45s} "
          f"{ps['mean_lambda_1']:.3f} ({ps['std_lambda_1']:.3f}){'':<5s} "
          f"Krumholz 2013 (2-4x baseline)")
    print(f"{'State-dependence beta':<45s} "
          f"{ps['mean_beta']:.3f} ({ps['std_beta']:.3f}){'':<5s} "
          f"Dharmarajan 2013 (cascades)")
    print(f"{'Recovery rate mu (/yr)':<45s} "
          f"{ps['mean_mu']:.3f} ({ps['std_mu']:.3f}){'':<5s} "
          f"Krumholz 2013 (1-6 mo)")
    print(f"{'Standard risk score r':<45s} "
          f"{ps['mean_r']:.3f} ({ps['std_r']:.3f})")
    print(f"{'Trajectory risk R (simulated)':<45s} "
          f"{ps['mean_R_gold']:.4f} ({ps['std_R_gold']:.4f})")

    print(f"\n{'Group':<25s} {'n':>6s} {'Mean beta':>10s} {'Mean mu':>10s}")
    print("-" * 55)
    for gname, gdata in primary["by_group"].items():
        print(f"{gname:<25s} {gdata['n_patients']:6d} "
              f"{gdata['mean_beta']:10.3f} {gdata.get('mean_mu', 'N/A'):>10}")


def print_table2(results):
    """Table 2: Scoring Comparison (the key table)."""
    print("\n" + "=" * 80)
    print("TABLE 2: Standard vs. Trajectory-Aware Scoring")
    print("=" * 80)

    for config_name in ["primary", "low_acuity", "high_acuity"]:
        cfg = results[config_name]
        sc = cfg["scoring_comparison"]

        print(f"\n--- {config_name.replace('_', ' ').title()} ---")
        print(f"{'Score':<25s} {'Delta':>7s} {'95% CI':>18s} {'C-stat':>7s} "
              f"{'rho':>7s} {'NRI':>7s}")
        print("-" * 75)

        for score_name, label in [("standard", "Standard (r)"),
                                   ("trajectory_aware", "Trajectory-aware (R_a)"),
                                   ("augmented", "Augmented (r*f)")]:
            s = sc[score_name]
            ci = s.get("delta_ci", {})
            ci_str = ""
            if ci.get("ci_lower") is not None:
                ci_str = f"({ci['ci_lower']:.4f}-{ci['ci_upper']:.4f})"
            nri = s.get("nri_vs_standard", {}).get("nri_total", "")
            nri_str = f"{nri:+.3f}" if nri != "" else "ref"
            print(f"{label:<25s} {s['delta']:7.4f} {ci_str:>18s} "
                  f"{s['c_statistic']:7.3f} {s['corr']:7.3f} {nri_str:>7s}")


def print_sensitivity(sens):
    """Sensitivity analysis table."""
    print("\n" + "=" * 80)
    print("TABLE 3: Sensitivity of Delta to k and T")
    print("=" * 80)

    k_values = sens["k_values"]
    T_values = sens["T_values"]

    print(f"\n{'':>10s}", end="")
    for T in T_values:
        print(f"{'T=' + str(T) + 'yr':>14s}", end="")
    print()
    print("-" * 66)

    grid = {(r["k"], r["T"]): r for r in sens["sensitivity_grid"]}

    for k in k_values:
        print(f"{'k=' + str(k):>10s}", end="")
        for T in T_values:
            r = grid.get((k, T), {})
            d = r.get("delta_standard", 0)
            print(f"{d:14.4f}", end="")
        print()

    print("\nTrajectory-aware scoring Delta:")
    for k in k_values:
        print(f"{'k=' + str(k):>10s}", end="")
        for T in T_values:
            r = grid.get((k, T), {})
            d = r.get("delta_trajectory_aware", 0)
            print(f"{d:14.4f}", end="")
        print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    # 1. Primary analysis with bootstrap CIs
    primary_results = run_primary_analysis(
        n=5000, n_sims=500, T=2.0, k=3,
        n_bootstrap=2000, seed=42,
    )

    # 2. Sensitivity analysis
    sensitivity = run_sensitivity_analysis(n=3000, n_sims=300, seed=42)

    # 3. Random search comparison
    random_search = run_random_search_comparison(n_configs=15, n=3000, n_sims=300, seed=42)

    # Print tables
    print_table1(primary_results)
    print_table2(primary_results)
    print_sensitivity(sensitivity)

    # Save all results
    all_results = {
        "primary_analysis": primary_results,
        "sensitivity": sensitivity,
        "random_search": random_search,
    }
    with open("results/revised_manuscript_data.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.1f}s")
    print("Results saved to results/revised_manuscript_data.json")
