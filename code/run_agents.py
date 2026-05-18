"""Run LLM-agent search on a single model. Saves to results/agent_<model>.json."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import two_state_sim as ts
import nyha_sim as nyha
import agent_search as ag


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
    "maximise Delta -- regimes in which the standard risk score systematically "
    "misorders patients relative to their trajectory risk."
)
TWO_STATE_SCHEMA = (
    "lambda_0_shape (gamma shape), lambda_0_scale (gamma scale), "
    "lambda_1_mult_shape (gamma shape on lambda_0 multiplier), "
    "lambda_1_mult_scale (gamma scale on the multiplier), "
    "beta_a (Beta-distribution alpha), beta_b (Beta-distribution beta), "
    "mu_shape (gamma shape for recovery rate mu), mu_scale (gamma scale)."
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
    "on the hospitalisation hazard immediately after a hospitalisation; "
    "hosp_amp_decay_months is the number of months over which the multiplier "
    "decays back to baseline."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["two_state", "nyha"])
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--agents-per-round", type=int, default=5)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--n-sims-ts", type=int, default=300)
    parser.add_argument("--n-sims-nyha", type=int, default=200)
    args = parser.parse_args()

    if args.model == "two_state":
        def eval_fn(p):
            return ts.evaluate(p, n=args.n, n_sims=args.n_sims_ts, T=2.0, k=3, seed=42)
        bounds = ts.PARAM_BOUNDS
        default = ts.CALIBRATED_PARAMS
        problem = TWO_STATE_PROBLEM
        schema = TWO_STATE_SCHEMA
    else:
        def eval_fn(p):
            return nyha.hf_evaluate(p, n=5000, n_sims=args.n_sims_nyha, T=3.0, k=2, seed=42)
        bounds = nyha.HF_PARAM_BOUNDS
        default = {k: nyha.HF_CALIBRATED_PARAMS[k] for k in bounds.keys()
                   if k in nyha.HF_CALIBRATED_PARAMS}
        problem = NYHA_PROBLEM
        schema = NYHA_SCHEMA

    bounds_desc = "\n".join(f"  - {k}: [{lo}, {hi}]" for k, (lo, hi) in bounds.items())

    print(f"=== Agent search: {args.model} ===")
    t0 = time.time()
    hist = ag.agent_search(
        n_rounds=args.n_rounds,
        n_agents_per_round=args.agents_per_round,
        evaluator=eval_fn,
        bounds=bounds,
        default_params=default if default else {},
        problem_description=problem,
        param_schema=schema,
        bounds_description=bounds_desc,
    )
    best = max(h["delta"] for h in hist)
    elapsed = time.time() - t0
    print(f"best delta = {best:.4f}  ({elapsed:.0f}s, {len(hist)} proposals)")

    out_path = Path(__file__).resolve().parents[1] / "results" / f"agent_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"history": hist, "best": best}, f, indent=2, default=float)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
