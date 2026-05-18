# trajectory-risk-misordering

Reproducible code for benchmarking LLM agents against Bayesian optimisation, evolutionary search, and uniform random search at structural sensitivity analysis of clinical microsimulations. The decision-relevant output is the misordering fraction Δ — the pairwise discordance probability between a standard risk score and Monte-Carlo trajectory risk.

This repository contains the simulators, the four search methods behind a unified evaluator interface, and the analysis scripts that produce the head-to-head comparison.

## Contents

```
code/
  two_state_sim.py            Two-state self-exciting microsimulation (closed-form
                              steady-state score; Monte-Carlo trajectory risk).
  nyha_sim.py                 NYHA I–IV + Death multi-state heart-failure
                              microsimulation. Cox-PH covariate effects on age,
                              eGFR, ejection fraction, diabetes; competing-risk
                              death; Hawkes-like post-hospitalisation self-
                              excitation. No closed-form expected event rate.
  search_methods.py           Random / BO (scikit-optimize gp_minimize, GP-EI) /
                              CMA-ES (pycma). All accept a common
                              (evaluator, bounds, n_calls) signature.
  agent_search.py             Five-persona Claude Opus 4.7 agent search behind
                              the same evaluator API. Personas: combinatorial
                              extremiser, survival statistician, ergodicity
                              physicist, stochastic-process theorist,
                              algorithmic-fairness researcher. Full system
                              prompts inline.
  run_pilot.py                Low-fidelity smoke test of all four methods.
  run_full_experiments.py     Production driver: 4 methods × 2 budgets
                              (15 matched / 60 extended) × 3 seeds × 2 models.
                              Incremental JSON saves.
  run_agents.py               Standalone agent-search runner (uses
                              ANTHROPIC_API_KEY environment variable).
  fix_cma_matched.py          Re-runs CMA-ES at the matched 15-eval budget with
                              an explicit popsize so the budget divides cleanly.
  add_seed3_two_state.py      Adds an extra seed for the two-state model.
  analyze_results.py          Summary table, per-method best-Δ figure, per-method
                              cumulative-best-trace figure, mechanism Lasso.
  mechanism_figure.py         Standardised Lasso coefficient bar plot.
  supplementary_figures.py    Per-seed traces and Δ distribution boxplots.
```

## Reproducing the comparison

```bash
make env                  # creates a Python venv with all dependencies
export ANTHROPIC_API_KEY=...
make full-two-state
make full-nyha
make agents
make analyze              # generates summary CSV + figures from saved JSON
```

`make all` chains every step.

## What the simulators define

**Two-state self-exciting.** Each simulated patient is characterised by four parameters: a baseline event rate λ₀, an elevated event rate λ₁ during a vulnerable state, a cascade propensity β (probability that an event in the stable state triggers a transition to the vulnerable state), and a recovery rate µ. The standard risk score is the closed-form steady-state event rate; the trajectory risk is the Monte-Carlo probability of ≥k events within T years. We include this model as an analytically tractable benchmark.

**NYHA I-IV + Death multi-state HF.** Each simulated patient has age, eGFR, ejection fraction, and diabetes covariates and an initial NYHA class. In monthly cycles the patient progresses or regresses one NYHA class with state-dependent transition hazards, dies with NYHA- and covariate-dependent hazard, and experiences HF hospitalisations whose rate depends on the current NYHA class and covariates and is amplified above baseline for several months after each hospitalisation. The standard risk score is the Cox-PH expected baseline hospitalisation rate evaluated at the patient's *initial* NYHA class with covariate multipliers — the form a risk-prediction model trained on baseline data would produce. The trajectory risk is the Monte-Carlo probability of ≥k HF hospitalisations within T years. There is no closed-form expected hospitalisation rate.

## Search methods (shared evaluator)

All four methods optimise Δ over the same continuous, bounded parameter space (`PARAM_BOUNDS` in each simulator). For the classical methods, three random seeds are run at each budget. For the LLM agents, five Claude Opus 4.7 personas propose configurations across three rounds (15 configurations total). Every proposal — regardless of method — is evaluated by running the same full microsimulation, so the comparison is paired.

## Mechanism recovery

`analyze_results.py` pools every (parameter vector, Δ) pair from every method on each model and fits Lasso with 5-fold cross-validated penalty on z-standardised parameters. The non-zero coefficients identify which parameter manipulations move Δ in which direction across the entire explored space. The agents' free-text rationales are coded against these coefficients.

## Notes

- All data are simulated. No patient data were used.
- Random seeds are fixed throughout. The agent search uses one seed per model (agent stochasticity is internal to the Anthropic API).
- The exact LLM model is Claude Opus 4.7 (Anthropic, model ID `claude-opus-4-7`). Full system prompts are in `code/agent_search.py:PERSONAS`.
- Results JSON files and figures are not included in this repository; they are generated by running the pipeline above.

## Dependencies

- Python 3.14
- numpy, scipy
- scikit-optimize (BO), pycma (CMA-ES)
- scikit-learn (Lasso)
- matplotlib
- anthropic

See `requirements.txt` and `Makefile:env` for exact versions.

## License

MIT.
