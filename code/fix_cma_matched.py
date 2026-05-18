"""Re-run only the CMA-ES matched-budget (15 evals) cells, with the popsize fix.

Updates the per-model results json in place. Idempotent.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import two_state_sim as ts
import nyha_sim as nyha
import search_methods as sm


def patch_model(model: str, eval_fn, bounds, seeds=(1, 2, 3)):
    path = Path(__file__).resolve().parents[1] / "results" / f"results_{model}.json"
    if not path.exists():
        print(f"  {model}: results file missing, skipping")
        return
    record = json.load(open(path))
    for seed in seeds:
        key = f"cma_matched_seed{seed}"
        t0 = time.time()
        try:
            hist = sm.cma_search(15, eval_fn, bounds, seed=seed, popsize=5)
            best = max(h["delta"] for h in hist)
        except Exception as exc:
            print(f"  {model} {key}: ERROR {exc}")
            continue
        record["methods"][key] = {
            "history": hist,
            "best": best,
            "wall_seconds": time.time() - t0,
        }
        print(f"  {model} {key}: best={best:.4f} ({time.time()-t0:.0f}s)")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=float)


def two_state_eval(p):
    return ts.evaluate(p, n=3000, n_sims=300, T=2.0, k=3, seed=42)


def nyha_eval(p):
    return nyha.hf_evaluate(p, n=5000, n_sims=200, T=3.0, k=2, seed=42)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["two_state", "nyha", "both"])
    args = parser.parse_args()
    if args.model in ("two_state", "both"):
        print("Patching two_state CMA matched...")
        patch_model("two_state", two_state_eval, ts.PARAM_BOUNDS)
    if args.model in ("nyha", "both"):
        print("Patching nyha CMA matched...")
        patch_model("nyha", nyha_eval, nyha.HF_PARAM_BOUNDS)
