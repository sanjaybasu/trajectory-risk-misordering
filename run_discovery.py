#!/usr/bin/env python3
"""
Run the full Trajectory Risk Misordering Discovery.

Phase 1: Establish baseline Δ with default parameters
Phase 2: Analytical bounds and structure
Phase 3: Multi-agent discovery competition
"""

import sys
import json
import time
import os

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import run_baseline
from analytical import (
    find_misordering_example,
    delta_vs_beta_variance,
    analytical_delta_bound,
)


def phase1_baseline():
    """Phase 1: Establish baseline Δ."""
    print("=" * 70)
    print("PHASE 1: BASELINE MISORDERING COMPUTATION")
    print("=" * 70)

    # Quick baseline with moderate parameters
    results = run_baseline(n=5000, n_sims=500, T=2.0, k=3, seed=42)

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/phase1_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def phase2_analytical():
    """Phase 2: Analytical bounds and structure."""
    print("\n" + "=" * 70)
    print("PHASE 2: ANALYTICAL STRUCTURE")
    print("=" * 70)

    # Explicit misordering example
    print("\n--- Explicit Misordering Example ---")
    example = find_misordering_example()
    A, B = example["patient_A"], example["patient_B"]
    print(f"Patient A (high baseline, low cascade):  r={A['r']:.4f}, R={A['R']:.4f}")
    print(f"Patient B (mod baseline, high cascade):  r={B['r']:.4f}, R={B['R']:.4f}")
    print(f"Standard score ranks A higher (r_A > r_B): {A['r'] > B['r']}")
    print(f"But B has higher trajectory risk (R_B > R_A): {B['R'] > A['R']}")
    print(f"MISORDERED: {example['misordered']}")

    # Δ vs Var(β)
    print("\n--- Δ as a Function of Var(β) ---")
    var_results = delta_vs_beta_variance(n_levels=8, n_patients=2000)
    for r in var_results["delta_vs_var_beta"]:
        bar = "█" * int(r["delta"] * 200)
        print(f"  Var(β)={r['var_beta']:.4f}  Δ={r['delta']:.4f}  {bar}")

    # Save
    with open("results/phase2_analytical.json", "w") as f:
        json.dump({"example": example, "delta_vs_var_beta": var_results}, f,
                  indent=2, default=str)

    return var_results


def phase3_discovery():
    """Phase 3: Multi-agent discovery."""
    print("\n" + "=" * 70)
    print("PHASE 3: MULTI-AGENT DISCOVERY COMPETITION")
    print("=" * 70)

    from discovery_platform import run_discovery
    results = run_discovery(n_rounds=3, n_patients=3000, n_sims=300)
    return results


if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: Baseline
    baseline = phase1_baseline()

    # Phase 2: Analytical
    analytical = phase2_analytical()

    # Phase 3: Discovery (requires Anthropic API key)
    try:
        discovery = phase3_discovery()
    except Exception as e:
        print(f"\nPhase 3 skipped: {e}")
        print("Set ANTHROPIC_API_KEY to run the multi-agent discovery.")
        discovery = None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*70}")
