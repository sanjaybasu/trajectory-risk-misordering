"""
Four search methods that all share the same evaluator and budget:
  - random_search: uniform sampling
  - bo_search:     scikit-optimize gp_minimize (GP-EI)
  - cma_search:    CMA-ES (pycma)
  - agent_search:  LLM agents (defined in agent_search.py)

All methods maximise misordering fraction (Delta) over the given bounds.
The parameter order is taken from the keys of the bounds dict, so the same
search code applies to the two-state and NYHA models.

The evaluator returns a dict with at least key 'delta' (float).
"""
from __future__ import annotations
import json
import time
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def _names(bounds: Dict[str, Tuple[float, float]]) -> Tuple[str, ...]:
    return tuple(bounds.keys())


def vec_to_params(x: np.ndarray, names: Tuple[str, ...]) -> Dict[str, float]:
    return {n: float(x[i]) for i, n in enumerate(names)}


def lower_upper(bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    names = _names(bounds)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    return lo, hi, names


# ------------------------------------------------------------------
# Random search
# ------------------------------------------------------------------

def random_search(
    n_calls: int,
    evaluator: Callable[[Dict], Dict],
    bounds: Dict[str, Tuple[float, float]],
    seed: int = 0,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    lo, hi, names = lower_upper(bounds)
    history: List[Dict] = []
    for i in range(n_calls):
        x = lo + rng.random(len(lo)) * (hi - lo)
        params = vec_to_params(x, names)
        t0 = time.time()
        result = evaluator(params)
        history.append({
            "iteration": i,
            "params": params,
            "delta": result["delta"],
            "pop_stats": {k: v for k, v in result.items() if k != "delta"},
            "wall_seconds": time.time() - t0,
        })
    return history


# ------------------------------------------------------------------
# Bayesian optimisation (skopt gp_minimize)
# ------------------------------------------------------------------

def bo_search(
    n_calls: int,
    evaluator: Callable[[Dict], Dict],
    bounds: Dict[str, Tuple[float, float]],
    n_initial: int = 8,
    seed: int = 0,
) -> List[Dict]:
    from skopt import gp_minimize
    from skopt.space import Real

    lo, hi, names = lower_upper(bounds)
    space = [Real(low=lo[i], high=hi[i], name=names[i]) for i in range(len(lo))]
    history: List[Dict] = []

    def neg_delta(x):
        params = vec_to_params(np.asarray(x), names)
        t0 = time.time()
        result = evaluator(params)
        history.append({
            "iteration": len(history),
            "params": params,
            "delta": result["delta"],
            "pop_stats": {k: v for k, v in result.items() if k != "delta"},
            "wall_seconds": time.time() - t0,
        })
        return -result["delta"]  # gp_minimize minimises

    gp_minimize(
        func=neg_delta,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=seed,
        acq_func="EI",
    )
    return history


# ------------------------------------------------------------------
# CMA-ES
# ------------------------------------------------------------------

def cma_search(
    n_calls: int,
    evaluator: Callable[[Dict], Dict],
    bounds: Dict[str, Tuple[float, float]],
    seed: int = 0,
    sigma0: float = 0.25,
    popsize: int | None = None,
) -> List[Dict]:
    import cma

    lo, hi, names = lower_upper(bounds)
    centre = (lo + hi) / 2
    scale = (hi - lo) / 2

    # Pick a population size that divides the budget cleanly so the final
    # generation isn't truncated mid-tell (pycma errors if tell() receives
    # fewer solutions than it asked for).
    if popsize is None:
        for candidate in (5, 6, 4, 7, 8, 10, 12):
            if n_calls % candidate == 0 and candidate >= 4:
                popsize = candidate
                break
        else:
            popsize = max(4, n_calls // 3)

    # Operate in normalised [-1, 1] space and let CMA-ES handle bounds.
    def neg_delta(z):
        x = np.clip(centre + np.asarray(z) * scale, lo, hi)
        params = vec_to_params(x, names)
        t0 = time.time()
        result = evaluator(params)
        history.append({
            "iteration": len(history),
            "params": params,
            "delta": result["delta"],
            "pop_stats": {k: v for k, v in result.items() if k != "delta"},
            "wall_seconds": time.time() - t0,
        })
        return -result["delta"]

    history: List[Dict] = []
    es = cma.CMAEvolutionStrategy(
        np.zeros(len(lo)),
        sigma0,
        {
            "bounds": [[-1.0] * len(lo), [1.0] * len(lo)],
            "seed": seed if seed > 0 else 1,
            "maxfevals": n_calls,
            "verbose": -9,
            "popsize": popsize,
        },
    )
    while len(history) + popsize <= n_calls and not es.stop():
        solutions = es.ask()
        fitnesses = [neg_delta(z) for z in solutions]
        es.tell(solutions, fitnesses)
    return history


# ------------------------------------------------------------------
# Common driver
# ------------------------------------------------------------------

def best_trace(history: List[Dict]) -> List[float]:
    """Cumulative-best Delta after each evaluation."""
    best = -np.inf
    out = []
    for h in history:
        best = max(best, h["delta"])
        out.append(best)
    return out


def save_history(history: List[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=float)


def load_history(path: str | Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
