"""
Pilot: run random, BO, CMA-ES on BOTH models at low fidelity.
Agent search is skipped in pilot (API cost); it runs in full only.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

import two_state_sim as ts
import nyha_sim as nyha
import search_methods as sm


PILOT_BUDGET = 30  # evaluations per method per model


def run_two_state_pilot():
    def eval_fn(params):
        return ts.evaluate(params, n=2000, n_sims=150, T=2.0, k=3, seed=42)

    base = eval_fn(ts.CALIBRATED_PARAMS)
    print(f"  calibrated baseline delta = {base['delta']:.4f}")

    out = {"calibrated_baseline": base, "methods": {}}
    for name, fn in [
        ("random", lambda: sm.random_search(PILOT_BUDGET, eval_fn, ts.PARAM_BOUNDS, seed=1)),
        ("bo", lambda: sm.bo_search(PILOT_BUDGET, eval_fn, ts.PARAM_BOUNDS,
                                     n_initial=8, seed=1)),
        ("cma", lambda: sm.cma_search(PILOT_BUDGET, eval_fn, ts.PARAM_BOUNDS, seed=1)),
    ]:
        t0 = time.time()
        hist = fn()
        best = max(h["delta"] for h in hist)
        print(f"  {name:6s}: best delta = {best:.4f}  ({time.time()-t0:.0f}s)")
        out["methods"][name] = {
            "history": hist,
            "best": best,
            "wall_seconds": time.time() - t0,
        }
    return out


def run_nyha_pilot():
    def eval_fn(params):
        return nyha.hf_evaluate(params, n=2000, n_sims=100, T=3.0, k=2, seed=42)

    base = eval_fn({})
    print(f"  calibrated baseline delta = {base['delta']:.4f}")

    out = {"calibrated_baseline": base, "methods": {}}
    for name, fn in [
        ("random", lambda: sm.random_search(PILOT_BUDGET, eval_fn, nyha.HF_PARAM_BOUNDS, seed=1)),
        ("bo", lambda: sm.bo_search(PILOT_BUDGET, eval_fn, nyha.HF_PARAM_BOUNDS,
                                     n_initial=10, seed=1)),
        ("cma", lambda: sm.cma_search(PILOT_BUDGET, eval_fn, nyha.HF_PARAM_BOUNDS, seed=1)),
    ]:
        t0 = time.time()
        hist = fn()
        best = max(h["delta"] for h in hist)
        print(f"  {name:6s}: best delta = {best:.4f}  ({time.time()-t0:.0f}s)")
        out["methods"][name] = {
            "history": hist,
            "best": best,
            "wall_seconds": time.time() - t0,
        }
    return out


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TWO-STATE PILOT ===")
    ts_results = run_two_state_pilot()
    with open(out_dir / "pilot_two_state.json", "w") as f:
        json.dump(ts_results, f, indent=2, default=float)

    print("\n=== NYHA PILOT ===")
    nyha_results = run_nyha_pilot()
    with open(out_dir / "pilot_nyha.json", "w") as f:
        json.dump(nyha_results, f, indent=2, default=float)

    print("\nPilot complete. Summary:")
    print(f"  two-state:")
    print(f"    calibrated baseline: {ts_results['calibrated_baseline']['delta']:.4f}")
    for name, d in ts_results["methods"].items():
        print(f"    {name:6s}: best = {d['best']:.4f}  ({d['wall_seconds']:.0f}s)")
    print(f"  nyha:")
    print(f"    calibrated baseline: {nyha_results['calibrated_baseline']['delta']:.4f}")
    for name, d in nyha_results["methods"].items():
        print(f"    {name:6s}: best = {d['best']:.4f}  ({d['wall_seconds']:.0f}s)")
