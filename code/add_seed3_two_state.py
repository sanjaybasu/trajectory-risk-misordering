"""Add seed-3 runs to two-state random and BO (matched + extended)."""
from __future__ import annotations
import json
import time
from pathlib import Path

import two_state_sim as ts
import search_methods as sm


def main():
    path = Path(__file__).resolve().parents[1] / "results" / "results_two_state.json"
    record = json.load(open(path))

    def eval_fn(p):
        return ts.evaluate(p, n=3000, n_sims=300, T=2.0, k=3, seed=42)

    seed = 3
    plan = [
        ("random_matched_seed3", lambda: sm.random_search(15, eval_fn, ts.PARAM_BOUNDS, seed=seed)),
        ("random_extended_seed3", lambda: sm.random_search(60, eval_fn, ts.PARAM_BOUNDS, seed=seed)),
        ("bo_matched_seed3", lambda: sm.bo_search(15, eval_fn, ts.PARAM_BOUNDS, n_initial=8, seed=seed)),
        ("bo_extended_seed3", lambda: sm.bo_search(60, eval_fn, ts.PARAM_BOUNDS, n_initial=10, seed=seed)),
        ("cma_extended_seed3", lambda: sm.cma_search(60, eval_fn, ts.PARAM_BOUNDS, seed=seed)),
    ]
    for name, fn in plan:
        if name in record["methods"]:
            print(f"  {name}: already present, skipping")
            continue
        t0 = time.time()
        hist = fn()
        best = max(h["delta"] for h in hist)
        elapsed = time.time() - t0
        record["methods"][name] = {
            "history": hist, "best": best, "wall_seconds": elapsed,
        }
        print(f"  {name}: best={best:.4f} ({elapsed:.0f}s)")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=float)


if __name__ == "__main__":
    main()
