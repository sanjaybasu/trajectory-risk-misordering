# Using Adversarial AI Agents to Stress-Test Health Decision Models: Application to Clinical Risk Scoring


## Overview

Clinical risk scores rank patients by expected event rates. Two patients with the same expected rate can have different probabilities of accumulating multiple adverse events if one is prone to cascading crises (self-exciting dynamics). The misordering fraction (Delta) quantifies how often standard scores produce incorrect pairwise rankings relative to trajectory-level catastrophic risk.

This repository implements an adversarial artificial intelligence (AI) agent exploration framework that systematically searches for population configurations where standard risk scores fail. Multiple AI agents compete to discover parameter regimes that maximize the misordering fraction, then the discovered configurations are validated through independent Monte Carlo simulation. The case study applies this framework to clinical risk scoring using a Hawkes process model of patient event trajectories.

All data are simulated. No patient data were used.

## Repository Structure

```
core.py                    # Population model, standard and trajectory risk scoring,
                           #   misordering fraction computation
analytical.py              # Analytical bounds using negative binomial approximation
revised_analysis.py        # Primary analysis pipeline: calibrated populations,
                           #   bootstrap confidence intervals, sensitivity grid,
                           #   random search comparison
supplementary_analyses.py  # Random search benchmarking and stability analyses
discovery_platform.py      # Multi-agent AI scientist competition platform
run_discovery.py           # Full discovery pipeline: baseline, analytical bounds,
                           #   agent-driven exploration
figures.py                 # Manuscript figure generation (Figures 1-2, eFigure 1)
results/
  revised_manuscript_data.json   # Primary analysis results reported in the manuscript
  supplementary_analyses.json    # Random search and stability analysis results
  round1.json                    # AI agent Round 1 competition results
  figure1_discovery.pdf/.png     # Figure 1: Discovery progression
  figure2_mechanism.pdf/.png     # Figure 2: Mechanism scatter and boxplot
  efigure1_nb_fit.pdf/.png       # eFigure 1: Negative binomial fit
```

## Reproducing Results

### Requirements

Python 3.11 or later. Install dependencies:

```
pip install -r requirements.txt
```

The `anthropic` package is required only for the AI agent discovery (`discovery_platform.py` and `run_discovery.py`). All other analyses run without it.

### Quick Start

Verify that the included results match manuscript values:

```
make check
```

### Step-by-Step

1. Primary analysis (`make analysis` or `python revised_analysis.py`): Generates three literature-calibrated populations (primary, low-acuity, high-acuity), computes standard and trajectory-aware scores, runs bootstrap confidence intervals (B = 2,000), sensitivity analysis over (k, T) grid, and random search comparison. Outputs `results/revised_manuscript_data.json`. Runtime is approximately 10 minutes on Apple Silicon.

2. Supplementary analyses (`make supplementary` or `python supplementary_analyses.py`): Runs random search benchmarking (200 configurations) and population and Monte Carlo stability analyses. Outputs `results/supplementary_analyses.json`.

3. Figures (`make figures` or `python figures.py`): Reads analysis results and generates Figure 1 (discovery progression), Figure 2 (mechanism scatter and boxplot), and eFigure 1 (negative binomial fit) as Portable Document Format (PDF) and Portable Network Graphics (PNG) files.

4. AI agent discovery (`make discovery` or `python run_discovery.py`): Runs baseline computation, analytical bounds, and the multi-agent competition. Requires an `ANTHROPIC_API_KEY` environment variable. Runtime is approximately 45 minutes including API latency. The agent competition is not required to reproduce the manuscript's numerical results; pre-computed agent results are included in the JSON files.

5. Full pipeline (`make all`): Runs analysis, supplementary, discovery, and figures in sequence.

## Key Parameters

All random seeds are fixed at 42. Primary analysis uses N = 5,000 patients, 500 Monte Carlo trajectories per patient, T = 2.0 years, k = 3 events, and B = 2,000 bootstrap resamples.

| Parameter | Distribution | Mean | Calibration Source |
|---|---|---|---|
| Baseline event rate (lambda_0) | Gamma(3.0, 0.2) | 0.60 per year | Jencks et al., New England Journal of Medicine, 2009 |
| Post-hospital rate elevation (lambda_1 / lambda_0) | 1 + Gamma(2.0, 0.8) | 2.6x | Krumholz, New England Journal of Medicine, 2013 |
| Cascade propensity (beta) | Beta(3.0, 7.0) | 0.30 | Dharmarajan et al., Journal of the American Medical Association, 2013 |
| Recovery rate (mu) | Gamma(4.0, 1.0) | 4.0 per year | Krumholz, New England Journal of Medicine, 2013 |

## Key Results

| Metric | Value |
|---|---|
| Standard score concordance statistic (C-statistic) | 0.965 |
| Misordering fraction (Delta) | 0.035 (95% confidence interval, 0.033-0.035) |
| Brier score | 0.142 |
| Validated worst-case Delta (agent-discovered) | 0.312 |
| Random search best Delta (200 configurations) | 0.270 |
| Population variability (standard deviation of Delta) | 0.0004 |
| Monte Carlo variability (standard deviation of Delta) | 0.0006 |

## License

This code is provided for research reproducibility. See LICENSE for terms. Contact the corresponding author for reuse inquiries.

## Contact

Sanjay Basu, MD, PhD
University of California San Francisco / Waymark
sanjay.basu@waymarkcare.com
