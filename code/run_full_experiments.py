"""
Full experimental grid for the method comparison.

Two models:
  - two-state self-exciting (analytically tractable benchmark)
  - NYHA I-IV + death multi-state HF (analytically intractable)

Four search methods:
  - random       (uniform sampling)
  - bo           (skopt gp_minimize, GP-EI)
  - cma          (CMA-ES)
  - agent        (Claude Opus 4.7, 5 personas x R rounds)

Two budgets per non-agent method:
  - matched budget (15 evals)  -> apples-to-apples with agent budget
  - extended budget (60 evals) -> see whether BO/CMA scale further

All agent + non-agent evaluations call the SAME evaluate function so
results are paired.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

import two_state_sim as ts
import nyha_sim as nyha
import search_methods as sm
import agent_search as ag


AGENT_ROUNDS = 3
AGENTS_PER_ROUND = 5
AGENT_BUDGET = AGENT_ROUNDS * AGENTS_PER_ROUND  # 15
MATCHED_BUDGET = AGENT_BUDGET
EXTENDED_BUDGET = 60


def _eval_factory_two_state(n: int, n_sims: int):
    def fn(params: Dict) -> Dict:
        return ts.evaluate(params, n=n, n_sims=n_sims, T=2.0, k=3, seed=42)
    return fn


def _eval_factory_nyha(n: int, n_sims: int):
    def fn(params: Dict) -> Dict:
        return nyha.hf_evaluate(params, n=n, n_sims=n_sims, T=3.0, k=2, seed=42)
    return fn


TWO_STATE_PROBLEM = (
    "We have a two-state self-exciting microsimulation. Each simulated patient "
    "is parameterised by a baseline event rate lambda_0, an elevated rate "
    "lambda_1 (during a vulnerable state), a cascade propensity beta (probability "
    "of moving from stable to vulnerable after each event), and a recovery rate "
    "mu (rate of returning to stable). Population-level parameters define the "
    "distributions from which patient-level (lambda_0, lambda_1, beta, mu) are "
    "drawn. The standard risk score is the closed-form steady-state event rate. "
    "The trajectory risk is the Monte Carlo probability of >=3 events within 2 "
    "years. The misordering fraction Delta is the probability of pairwise "
    "discordance between standard-score and trajectory-risk rankings. "
    "We are searching for population-level parameter configurations that "
    "maximise Delta -- i.e., regimes in which the standard risk score systematically "
    "misorders patients relative to their trajectory risk."
)

TWO_STATE_SCHEMA = (
    "lambda_0_shape (gamma shape), lambda_0_scale (gamma scale), "
    "lambda_1_mult_shape (gamma shape on lambda_0 multiplier), "
    "lambda_1_mult_scale (gamma scale on the multiplier), "
    "beta_a (Beta-distribution alpha), beta_b (Beta-distribution beta), "
    "mu_shape (gamma shape), mu_scale (gamma scale)."
)

TWO_STATE_BOUNDS_DESC = "\n".join(
    f"  - {k}: [{lo}, {hi}]" for k, (lo, hi) in ts.PARAM_BOUNDS.items()
)


NYHA_PROBLEM = (
    "We have a NYHA I-IV + death multi-state heart-failure microsimulation. "
    "Each simulated patient has age, eGFR, ejection fraction, and diabetes "
    "covariates, plus an initial NYHA class. Patients transition between NYHA "
    "states (progression, regression), die from competing-risk death, and "
    "experience HF hospitalisations whose rate depends on the current NYHA "
    "class and covariates AND is excited above baseline for several months "
    "after each hospitalisation (self-exciting). The standard risk score is "
    "the Cox-PH expected hospitalisation rate evaluated at each patient's "
    "INITIAL NYHA class with covariate multipliers -- the standard practice in "
    "risk prediction from baseline data. The trajectory risk is the Monte "
    "Carlo probability of >=2 HF hospitalisations within 3 years. The "
    "misordering fraction Delta is the pairwise discordance probability. "
    "We are searching for parameter configurations that maximise Delta -- "
    "regimes in which the standard score systematically misorders patients "
    "relative to trajectory risk."
)

NYHA_SCHEMA = (
    "Annual transition / hazard rates and case-mix moments. All values are "
    "non-negative; reg_* may be zero. hosp_amp_after_event is the multiplier "
    "on the hospitalisation hazard immediately after a hospitalisation, "
    "decaying linearly to 1 over hosp_amp_decay_months months."
)

NYHA_BOUNDS_DESC = "\n".join(
    f"  - {k}: [{lo}, {hi}]" for k, (lo, hi) in nyha.HF_PARAM_BOUNDS.items()
)


def _safe(method_name: str, fn, *args, **kwargs):
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:
        return {
            "history": [],
            "best": float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.time() - t0,
        }
    best = max((h["delta"] for h in out), default=float("nan"))
    return {
        "history": out,
        "best": best,
        "wall_seconds": time.time() - t0,
    }


def run_model(
    name: str,
    eval_fn: Callable[[Dict], Dict],
    bounds: Dict,
    default_params: Dict,
    problem: str,
    schema: str,
    bounds_desc: str,
    matched_budget: int,
    extended_budget: int,
    seeds: List[int],
    skip_agent: bool,
    out_dir: Path,
):
    print(f"\n=== MODEL: {name} ===")
    baseline = eval_fn(default_params)
    print(f"  calibrated baseline delta = {baseline['delta']:.4f}")
    record = {"name": name, "calibrated_baseline": baseline, "methods": {}}

    for seed in seeds:
        print(f"  seed = {seed}")
        record["methods"][f"random_matched_seed{seed}"] = _safe(
            "random_matched",
            sm.random_search, matched_budget, eval_fn, bounds, seed=seed,
        )
        print(f"    random matched ({matched_budget}): "
              f"{record['methods'][f'random_matched_seed{seed}']['best']:.4f}")
        record["methods"][f"random_extended_seed{seed}"] = _safe(
            "random_extended",
            sm.random_search, extended_budget, eval_fn, bounds, seed=seed,
        )
        print(f"    random extended ({extended_budget}): "
              f"{record['methods'][f'random_extended_seed{seed}']['best']:.4f}")
        record["methods"][f"bo_matched_seed{seed}"] = _safe(
            "bo_matched",
            sm.bo_search, matched_budget, eval_fn, bounds,
            n_initial=min(8, matched_budget // 2), seed=seed,
        )
        print(f"    BO matched: {record['methods'][f'bo_matched_seed{seed}']['best']:.4f}")
        record["methods"][f"bo_extended_seed{seed}"] = _safe(
            "bo_extended",
            sm.bo_search, extended_budget, eval_fn, bounds,
            n_initial=10, seed=seed,
        )
        print(f"    BO extended: {record['methods'][f'bo_extended_seed{seed}']['best']:.4f}")
        record["methods"][f"cma_matched_seed{seed}"] = _safe(
            "cma_matched",
            sm.cma_search, matched_budget, eval_fn, bounds, seed=seed,
        )
        print(f"    CMA matched: {record['methods'][f'cma_matched_seed{seed}']['best']:.4f}")
        record["methods"][f"cma_extended_seed{seed}"] = _safe(
            "cma_extended",
            sm.cma_search, extended_budget, eval_fn, bounds, seed=seed,
        )
        print(f"    CMA extended: {record['methods'][f'cma_extended_seed{seed}']['best']:.4f}")

        # Save incrementally so we can resume.
        with open(out_dir / f"results_{name}.json", "w") as f:
            json.dump(record, f, indent=2, default=float)

    if not skip_agent:
        print("  agent search (3 rounds x 5 personas):")
        record["methods"]["agent"] = _safe(
            "agent",
            ag.agent_search,
            AGENT_ROUNDS, AGENTS_PER_ROUND, eval_fn, bounds, default_params,
            problem, schema, bounds_desc,
        )
        print(f"    agent best: {record['methods']['agent']['best']:.4f}")
        with open(out_dir / f"results_{name}.json", "w") as f:
            json.dump(record, f, indent=2, default=float)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-state", action="store_true")
    parser.add_argument("--nyha", action="store_true")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--n-sims-ts", type=int, default=500)
    parser.add_argument("--n-sims-nyha", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if not (args.two_state or args.nyha):
        args.two_state = args.nyha = True

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.two_state:
        ts_eval = _eval_factory_two_state(args.n, args.n_sims_ts)
        run_model(
            name="two_state",
            eval_fn=ts_eval,
            bounds=ts.PARAM_BOUNDS,
            default_params=ts.CALIBRATED_PARAMS,
            problem=TWO_STATE_PROBLEM,
            schema=TWO_STATE_SCHEMA,
            bounds_desc=TWO_STATE_BOUNDS_DESC,
            matched_budget=MATCHED_BUDGET,
            extended_budget=EXTENDED_BUDGET,
            seeds=args.seeds,
            skip_agent=args.skip_agent,
            out_dir=out_dir,
        )

    if args.nyha:
        nyha_eval = _eval_factory_nyha(args.n, args.n_sims_nyha)
        run_model(
            name="nyha",
            eval_fn=nyha_eval,
            bounds=nyha.HF_PARAM_BOUNDS,
            default_params={},   # empty dict -> use HF_CALIBRATED_PARAMS defaults
            problem=NYHA_PROBLEM,
            schema=NYHA_SCHEMA,
            bounds_desc=NYHA_BOUNDS_DESC,
            matched_budget=MATCHED_BUDGET,
            extended_budget=EXTENDED_BUDGET,
            seeds=args.seeds,
            skip_agent=args.skip_agent,
            out_dir=out_dir,
        )

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
