# The Misordering Problem in Clinical Risk Prediction

## Overview

Standard risk scores rank patients by their expected event rate. This study shows that two patients with the same expected rate can have different probabilities of accumulating multiple adverse events if one is prone to cascading crises (self-exciting dynamics). The misordering fraction quantifies how often standard scores produce incorrect pairwise rankings relative to trajectory-level catastrophic risk.

This repository contains the microsimulation code, analysis pipeline, and AI-agent discovery platform described in the manuscript. All data are simulated; no patient data were used.

## Repository Structure

```
core.py                  # Population model, standard/trajectory risk, misordering metric
analytical.py            # Analytical bounds (NB approximation, misordering examples)
revised_analysis.py      # Primary analysis pipeline (calibrated populations, bootstrap CIs,
                         #   sensitivity grid, random search comparison)
discovery_platform.py    # Multi-agent AI scientist competition
run_discovery.py         # Full pipeline: baseline -> analytical -> agent discovery
figures.py               # Manuscript figure generation
results/
  revised_manuscript_data.json   # All numerical results reported in the manuscript
  round1.json                    # AI agent Round 1 results
```

## Reproducing Results

### Requirements

Python 3.11+. Install dependencies:

```
pip install -r requirements.txt
```

The `anthropic` package is required only for the AI-agent discovery (`discovery_platform.py`). All other analyses run without it.

### Quick Start

Verify that the included results match manuscript values:

```
make check
```

Reproduce the primary analysis from scratch (~10 min):

```
make analysis
```

Generate figures:

```
make figures
```

Run the full pipeline including AI-agent discovery (~45 min; requires `ANTHROPIC_API_KEY`):

```
make discovery
```

### Step-by-Step

1. **Primary analysis** (`python revised_analysis.py`): Generates three literature-calibrated populations (primary, low-acuity, high-acuity), computes standard and trajectory-aware scores, runs bootstrap CIs (B=2,000), sensitivity analysis over (k, T) grid, and random search comparison. Outputs `results/revised_manuscript_data.json`.

2. **Figures** (`python figures.py`): Reads analysis results and generates Figure 1 (discovery progression) and Figure 2 (mechanism scatter/boxplot) as PDF and PNG.

3. **AI-agent discovery** (`python run_discovery.py`): Runs baseline computation, analytical bounds, and the multi-agent competition (requires Anthropic API access). The agent competition is not required to reproduce the manuscript's numerical results; pre-computed agent results are in the JSON files.

## Key Parameters

All random seeds are fixed at 42. Primary analysis: N=5,000 patients, 500 Monte Carlo trajectories per patient, T=2.0 years, k=3 events, B=2,000 bootstrap resamples.

| Parameter | Distribution | Mean | Calibration Source |
|---|---|---|---|
| Baseline event rate | Gamma(3.0, 0.2) | 0.60/yr | Jencks et al. NEJM 2009 |
| Post-hospital elevation | 1 + Gamma(2.0, 0.8) | 2.6x | Krumholz NEJM 2013 |
| Cascade propensity | Beta(3.0, 7.0) | 0.30 | Dharmarajan et al. JAMA 2013 |
| Recovery rate | Gamma(4.0, 1.0) | 4.0/yr | Krumholz NEJM 2013 |

## Key Results

| Metric | Value |
|---|---|
| Standard score C-statistic | 0.965 |
| Trajectory-aware score C-statistic | 0.965 |
| Trajectory-aware NRI vs standard | +0.73 |
| High-acuity population Delta | 0.061 |
| Validated worst-case Delta | 0.312 (95% CI, 0.303-0.319) |

## License

This code is provided for research reproducibility. Contact the corresponding author for reuse inquiries.

