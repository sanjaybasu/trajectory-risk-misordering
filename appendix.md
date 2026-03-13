# Supplementary Appendix

## Using Adversarial AI Agents to Stress-Test Health Decision Models: Application to Clinical Risk Scoring

---

## eAppendix A. Mathematical Framework

### A.1. Two-State Self-Exciting Process

Each patient $$i$$ is characterized by parameters $$(\lambda_{0i}, \lambda_{1i}, \beta_i, \mu_i)$$ governing a continuous-time Markov-modulated Poisson process.

State space. $$S_i(t) \in \{0, 1\}$$ (stable, vulnerable).

Event process. Conditional on state $$S_i(t) = s$$, events occur as a Poisson process with rate:

$$\lambda_i(t) = \lambda_{0i} \cdot \mathbb{1}[S_i(t) = 0] + \lambda_{1i} \cdot \mathbb{1}[S_i(t) = 1]$$

State transitions.
- Stable $$\to$$ Vulnerable: probability $$\beta_i$$ immediately following each event in the stable state.
- Vulnerable $$\to$$ Stable: rate $$\mu_i$$ per unit time (exponential recovery).

### A.2. Standard Risk Score

At stationarity, the vulnerable-state occupancy probability is:

$$\pi_i = \frac{\beta_i \lambda_{0i}}{\beta_i \lambda_{0i} + \mu_i}$$

The steady-state event rate is:

$$r_i = \lambda_{0i}(1 - \pi_i) + \lambda_{1i} \pi_i$$

This is the quantity estimated by conventional risk prediction algorithms.<sup>1,3</sup> In our simulation, $$r_i$$ is computed from the true generative parameters, providing an oracle (best-case) assessment of standard scoring.

### A.3. Trajectory-Aware Score (Negative Binomial Approximation)

The trajectory risk is defined as $$R_i = P\left(\sum_{t=0}^{T} N_i(t) \geq k\right)$$. We approximate the event count distribution as negative binomial with:

Mean: $$\mu_N = r_i \cdot T$$

Overdispersion (variance-to-mean ratio):

$$d_i = 1 + \frac{(\lambda_{1i} - \lambda_{0i}) \cdot \beta_i}{\beta_i \lambda_{0i} + \mu_i} \cdot \frac{1 - e^{-\mu_i T}}{\mu_i T}$$

The temporal factor $$(1 - e^{-\mu_i T})/(\mu_i T)$$ captures the decay of temporal correlation: for short horizons (small $$\mu_i T$$), the factor approaches 1 (maximum overdispersion); for long horizons (large $$\mu_i T$$), the factor approaches 0 (overdispersion attenuates as the law of large numbers applies).<sup>9</sup>

Negative binomial parameters: $$n = \mu_N/(d_i - 1)$$, $$p = 1/d_i$$, where scipy.stats.nbinom parameterizes mean = $$n(1-p)/p$$ and var = $$n(1-p)/p^2$$.

Trajectory-aware score: $$R_{a,i} = 1 - F_{\text{NB}}(k-1; n, p)$$, where $$F_{\text{NB}}$$ is the negative binomial cumulative distribution function (CDF).

### A.4. Why the Approximation Resolves Misordering at the Tail

The standard score $$r$$ is a mean; the trajectory-aware score $$R_a$$ is a tail probability. Two patients with equal $$r$$ but different $$d$$ (overdispersion) will have different $$R_a$$: the patient with higher $$d$$ has a fatter tail and thus higher $$R_a$$. This is the mechanism by which $$R_a$$ can correctly rank patients that $$r$$ miorders.

The approximation is most accurate when (1) the time horizon is long relative to the recovery time ($$\mu T \gg 1$$), so the process mixes well, and (2) the event count is not rare, so the negative binomial approximation has sufficient support. For short horizons ($$T < 1/\mu$$), the Markov assumption underlying the overdispersion formula is less accurate. This explains why the trajectory-aware score does not consistently reduce $$\Delta$$ relative to the standard score for small $$T$$ in Table 4 of the main text.

### A.5. Misordering Fraction

$$\Delta = P\left[(r_i > r_j \text{ and } R_i < R_j) \text{ or } (r_i < r_j \text{ and } R_i > R_j)\right]$$

This is the discordance probability estimated over random pairs.<sup>15,16</sup> Key properties:
- $$\Delta = 0$$: the score produces the same pairwise ranking as trajectory risk (perfect concordance).
- $$\Delta = 0.5$$: the score ranks as poorly as random ordering (no concordance).
- C-statistic $$= 1 - \Delta$$.
- Kendall's $$\tau = 1 - 2\Delta$$.

### A.6. Proper Scoring Rules

We evaluate scoring performance using two strictly proper scoring rules.<sup>21,22</sup>

Brier score. For binary classification of high trajectory risk ($$R_i \geq$$ 75th percentile of the simulated population):

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(D_i - p_i)^2$$

where $$D_i = \mathbb{1}[R_i \geq R_{(0.75)}]$$ and $$p_i$$ is the predicted score rescaled to [0, 1] via min-max normalization. Lower values indicate better calibrated binary predictions. The Brier score admits a decomposition into reliability (calibration), resolution (discrimination), and uncertainty components (Murphy 1973). The 75th percentile threshold is computed from the same simulated population used for evaluation; this is standard for simulation studies where no separate test set exists.

Continuous ranked probability score (CRPS). For deterministic (point) forecasts, the CRPS reduces to the mean absolute error:

$$\text{CRPS} = \frac{1}{N}\sum_{i=1}^{N}|p_i - R_i^*|$$

where $$p_i$$ and $$R_i^*$$ are min-max normalized to [0, 1]. The CRPS is strictly proper: for distributional forecasts, it equals the integral of the squared difference between the predicted CDF and the empirical step function. For the deterministic point forecasts used here, it reduces to normalized MAE.<sup>21</sup>

Both metrics are strictly proper: they are uniquely minimized when predicted probabilities equal true probabilities.<sup>21</sup> Unlike the net reclassification improvement (NRI), which is not a proper scoring rule and can show positive "improvement" for uninformative or miscalibrated models,<sup>22,23</sup> the Brier score and CRPS cannot be inflated by poor calibration.

### A.7. MLE Negative Binomial Score

For the correctly specified model comparison, we fit a negative binomial distribution to each patient's simulated event counts via the method of moments. Given $$M$$ simulated trajectories for patient $$i$$, we observe event counts $$\{c_{i1}, \ldots, c_{iM}\}$$ and compute:

$$\hat{\mu}_i = \bar{c}_i, \quad \hat{\sigma}^2_i = \text{Var}(c_{i1}, \ldots, c_{iM})$$

If $$\hat{\sigma}^2_i > \hat{\mu}_i$$ (overdispersed), we parameterize NB($$\hat{n}_i, \hat{p}_i$$) with:

$$\hat{n}_i = \frac{\hat{\mu}_i^2}{\hat{\sigma}^2_i - \hat{\mu}_i}, \quad \hat{p}_i = \frac{\hat{n}_i}{\hat{n}_i + \hat{\mu}_i}$$

and compute $$\hat{R}_{i,\text{MLE}} = 1 - F_{\text{NB}}(k-1; \hat{n}_i, \hat{p}_i)$$. If equi- or underdispersed, we use a Poisson fallback.

---

## eAppendix B. Microsimulation Algorithm

For each patient $$i$$ and Monte Carlo replicate $$m = 1, \ldots, M$$:

```
Initialize: S_i(0) = 0 (stable), event_count = 0
For t = 0, dt, 2dt, ..., T-dt    (dt = 1/365 year):
    rate = lambda_0i if S_i(t) = 0, else lambda_1i
    If Uniform(0,1) < rate * dt:          # event occurs
        event_count += 1
        If S_i(t) = 0 and Uniform(0,1) < beta_i:
            S_i(t+dt) = 1                 # become vulnerable
    If S_i(t) = 1 and Uniform(0,1) < mu_i * dt:
        S_i(t+dt) = 0                     # recover
R_i = (1/M) * sum(event_count >= k for each replicate m)
```

The time step dt = 1/365 year ensures the Poisson thinning approximation ($$\lambda \cdot dt \ll 1$$) holds for typical rates ($$\lambda < 10$$/year). For the primary analysis and sensitivity grid, $$\lambda_{1,\max} \approx 3.4$$/year, giving $$\lambda \cdot dt \approx 0.009$$. For agent-discovered extreme regimes, we verified that $$\lambda \cdot dt < 0.05$$ for all configurations; regimes with higher rates would require a smaller dt.<sup>14</sup> Both the primary and sensitivity analyses use M = 500 trajectories per patient.

### B.1. Misordering Fraction Estimation

$$\Delta$$ is estimated from 500,000 random pairs, excluding self-pairs and ties:

$$\hat{\Delta} = \frac{\#\text{discordant pairs}}{\#\text{non-tied pairs}}$$

### B.2. Bootstrap Confidence Intervals

For $$b = 1, \ldots, 2{,}000$$: resample $$n$$ patients with replacement; recompute $$\hat{\Delta}_b$$ using pre-computed $$(r, R)$$ values for resampled patients. The 95% confidence interval (CI) is $$[\hat{\Delta}_{(0.025)}, \hat{\Delta}_{(0.975)}]$$ (percentile method). These CIs quantify patient-sampling uncertainty conditional on the specific Monte Carlo realization of trajectory risk. Monte Carlo variability is characterized separately via the multi-seed stability analysis (eTable 3).

### B.3. Multi-Seed Stability Analysis

To characterize the total variability in $$\Delta$$, we separately quantify two sources:

1. Population variability: We regenerate the N = 5,000 patient population from the same parameter distributions using 20 independent random seeds, running the full simulation for each. This captures variability due to which patients are drawn from the underlying distribution.

2. Monte Carlo (MC) variability: For the primary population (seed 42), we rerun the trajectory simulation with 20 independent simulation seeds, keeping the population fixed. This captures variability due to stochastic trajectory realizations.

We also compute a high-precision estimate using 5,000 trajectories per patient (10× the primary analysis) to assess convergence.

---

## eAppendix C. AI Agent Details

### C.1. Agent Design

Five agents were implemented as calls to Claude Opus 4 (Anthropic, model ID claude-opus-4-6) with persona-specific system prompts.<sup>10</sup> Each agent received a description of the parameter space, the misordering fraction metric, and results from prior rounds; each returned a proposed parameter configuration with a written rationale. Agent outputs are stochastic: the same prompts with a different model version or random seed may produce different configurations. We report the exact model version and full prompts in the code repository to support reproducibility.

| Agent | Perspective | Methodological Focus |
|---|---|---|
| Combinatorics | Extremal constructions | Counting arguments, explicit examples that maximize $$\Delta$$ |
| Survival analysis | Hazard rates | Overdispersion, recurrent event processes |
| Ergodicity economics | Time-ensemble divergence | Multiplicative dynamics, non-ergodic processes<sup>17</sup> |
| Self-exciting process | Branching ratios | Criticality, spectral properties of excitation kernels<sup>5</sup> |
| Algorithmic fairness | Equity implications | Who is harmed by misordering, group-level disparities<sup>18</sup> |

### C.2. Key Findings

Round 1 -- Tight $$\lambda_0$$ maximizes $$\Delta$$. The combinatorics agent identified that making baseline rates homogeneous (high shape parameter for the Gamma distribution of $$\lambda_0$$, giving low coefficient of variation) maximizes misordering ($$\Delta$$ = 0.222). When all patients have similar baseline rates, the standard score $$r$$ loses discriminative power, and differences in state-dependence $$\beta$$ dominate trajectory risk but are not captured by $$r$$. We verified this computationally by restricting the primary population to patients with $$\lambda_0$$ within the interquartile range, which increased $$\Delta$$ from 0.035 to 0.082.

Round 1 -- High Var($$\lambda_0$$) decreases $$\Delta$$ (negative result). The survival analysis agent increased baseline rate variance, which decreased $$\Delta$$ to 0.044 rather than increasing it. High Var($$\lambda_0$$) makes $$r$$ more informative because baseline rate heterogeneity dominates the ranking.

Round 2 -- Near-absorbing vulnerable state. The ergodicity agent combined tight $$\lambda_0$$ with $$\mu \approx 0.01$$ (near-zero recovery), creating effectively permanent vulnerability: once a patient enters the vulnerable state, recovery requires an average of 100 years. This produced $$\Delta$$ = 0.461 (Round 2 evaluation) and $$\Delta$$ = 0.312 on re-evaluation with the full simulation protocol (N = 5,000; M = 500). The result is mathematically informative but clinically implausible.

Diminishing returns. Round 3 was used to validate and synthesize Round 1-2 findings rather than to generate novel configurations. The Round 3 validated worst-case ($$\Delta$$ = 0.312) combined the mechanisms from Rounds 1-2 but did not exceed the Round 2 best. This suggests diminishing returns after 2-3 rounds for this particular microsimulation, though the optimal number of rounds may vary for other models.

### C.3. Comparison to Random Search

Two hundred random parameter configurations drawn from the same parameter space, evaluated at the same simulation fidelity (N = 5,000; M = 500), achieved a maximum $$\Delta$$ = 0.270 (eTable 1), compared to the agents' Round 1 best of 0.222 and validated worst-case of 0.312. The agents' advantage was interpretability rather than search efficiency: agents identified the mechanistic principles — tight $$\lambda_0$$, bimodal $$\beta$$, low $$\mu$$ — that explain why misordering occurs. Post-hoc analysis of the random search results (e.g., regressing $$\Delta$$ on parameter values) could identify important parameter associations, but would not generate the structured causal reasoning or connections to established theory (e.g., ergodicity economics, branching processes) that agents provided.

---

## eAppendix D. Literature Calibration Details

| Parameter | Distribution | Mean | Source and Justification |
|---|---|---|---|
| $$\lambda_0$$ | Gamma(3.0, 0.2) | 0.6/yr | Hospitalization rates of 0.5–1.0/yr among high-risk adults<sup>8</sup> |
| $$\lambda_1/\lambda_0$$ | 1 + Gamma(2.0, 0.8) | 2.6× | Post-hospital syndrome: 2–4× elevated readmission risk<sup>7</sup> |
| $$\beta$$ | Beta(3.0, 7.0) | 0.30 | ~20–40% of hospitalizations initiate cascade trajectories<sup>7,11</sup> |
| $$\mu$$ | Gamma(4.0, 1.0) | 4.0/yr | Recovery from post-hospital vulnerability in 1–6 months<sup>7</sup> |
| Group $$\beta$$ offsets | [0, +0.04, +0.07, +0.10] | — | Differential readmission rates across socioeconomic groups<sup>12,13</sup> |
| Group $$\mu$$ scales | [1.0, 0.85, 0.75, 0.65] | — | Reduced access to post-discharge services in disadvantaged groups<sup>12,13</sup> |

Low-acuity population: $$\lambda_0$$ ~ Gamma(2.0, 0.15) [mean 0.3/yr], $$\beta$$ ~ Beta(2.5, 8.5) [mean 0.23], $$\mu$$ ~ Gamma(5.0, 1.2) [mean 6.0/yr]. Attenuated group offsets (β: 0, +0.03, +0.05, +0.08; μ scales: 1.0, 0.9, 0.8, 0.7).

High-acuity population: $$\lambda_0$$ ~ Gamma(4.0, 0.25) [mean 1.0/yr], $$\beta$$ ~ Beta(3.5, 5.5) [mean 0.39], $$\mu$$ ~ Gamma(3.0, 0.8) [mean 2.4/yr]. Amplified group offsets (β: 0, +0.05, +0.09, +0.14; μ scales: 1.0, 0.8, 0.7, 0.55).

---

## eAppendix E. Sensitivity Analysis Details

The sensitivity grid evaluates $$\Delta$$ for the standard score and trajectory-aware score across all combinations of $$k \in \{1, 2, 3, 5\}$$ and $$T \in \{0.5, 1.0, 2.0, 5.0\}$$ years using the primary clinically calibrated population (N = 5,000; M = 500 trajectories per patient).

Three patterns emerge:

1. Short T and moderate k produce the highest $$\Delta$$. At T = 0.5 years and k = 3, $$\Delta$$ = 0.098. Short horizons amplify the overdispersion effect because there is less time for recovery dynamics to average out.

2. Long T and high k produce the lowest $$\Delta$$. At T = 5.0 years and k = 5, $$\Delta$$ = 0.026. Over long horizons, the law of large numbers dampens overdispersion, and the standard score's ranking converges toward the trajectory risk ranking.

3. The trajectory-aware score is most beneficial for long T and low k. The negative binomial approximation is most accurate when $$\mu T \gg 1$$ (the process has time to mix). For short horizons (T $$\leq$$ 1.0 years) and moderate-to-high thresholds ($$k \geq 2$$), the trajectory-aware score produces higher $$\Delta$$ than the standard score (e.g., 0.111 vs 0.098 at k = 3, T = 0.5 years), reflecting the limited accuracy of the approximation for unmixed processes.

---

## eAppendix F. International Society for Pharmacoeconomics and Outcomes Research and Society for Medical Decision Making (ISPOR-SMDM) Modeling Good Research Practices Checklist<sup>19</sup>

| Item | Description | Section |
|---|---|---|
| 1 | Statement of decision problem | Introduction |
| 2 | Statement of scope and perspective | Methods (Ethical Oversight) |
| 3 | Rationale for model structure | Methods (Model Structure) |
| 4 | Structural assumptions | Methods (Model Structure); eAppendix A |
| 5 | Strategies/comparators | Methods (Risk Scores Compared) |
| 6 | Model type | Methods (microsimulation, two-state self-exciting) |
| 7 | Time horizon | Methods (T = 2.0 yr primary; sensitivity: 0.5–5.0 yr) |
| 8 | Disease states/pathways | eAppendix A.1 |
| 9 | Cycle length | eAppendix B (dt = 1/365 yr) |
| 10 | Data identification | Methods (Literature-Calibrated Parameters) |
| 11 | Data modeling | eAppendix D |
| 12 | Baseline data | Table 1; eAppendix D |
| 13 | Treatment effects | N/A (scoring comparison, not treatment) |
| 14 | Quality of life | N/A |
| 15 | Costs | N/A |
| 16 | Incorporation of uncertainty | Methods (Statistical Analysis); bootstrap CIs; eTable 3 |
| 17 | Methodological uncertainty | eAppendix E (sensitivity analysis); eTable 3 (multi-seed stability) |
| 18 | Heterogeneity | Results (Social Risk Stratification); eTable 2 |
| 19 | Validation | eAppendix C.3 (random search comparison) |

---

## eAppendix G. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis plus Artificial Intelligence (TRIPOD+AI) Reporting Checklist<sup>20</sup>

| TRIPOD+AI Item | Section |
|---|---|
| 1. Study identified as developing a prediction model | Title, Abstract |
| 2. Background and rationale | Introduction |
| 3. Objectives | Introduction |
| 4a. Study design | Methods |
| 4b. Setting | Methods (simulated population) |
| 5. Participants | Methods (Literature-Calibrated Parameters) |
| 6. Outcome | Methods (Trajectory Risk definition) |
| 7. Predictors | Methods (Standard Risk Score, Trajectory-Aware Score) |
| 8. Sample size | Methods (N = 5,000; B = 2,000 bootstrap; 20-seed stability) |
| 9. Missing data | N/A (simulated, no missingness) |
| 10a. Statistical analysis | Methods (Statistical Analysis) |
| 10b. AI/ML model specification | Methods (AI-Agent Adversarial Exploration Framework); eAppendix C |
| 10c. Model performance measures | Methods (Misordering Fraction, Brier score, CRPS) |
| 11. Risk groups | Methods (four social risk strata) |
| 12. Development vs. validation | N/A (simulation study) |
| 13. Participant flow | N/A (simulated) |
| 14. Model specification | eAppendix A, B |
| 15. Model performance | Results (Tables 2, 3, 4); eTable 1 |
| 16. Interpretation | Discussion |
| 17. Limitations | Discussion (paragraph 5) |
| 18. Implications | Discussion (final paragraph) |
| 19. Supplementary information | This appendix |
| 20. Funding | N/A |

---

## eAppendix H. Software and Reproducibility

All simulations were performed in Python 3.11 with NumPy 1.26 and SciPy 1.12.<sup>14</sup> AI agent interactions used Claude Opus 4 (Anthropic, model ID claude-opus-4-6) via the Claude Code CLI.<sup>10</sup> Agent outputs are stochastic and may vary across runs; the exact model version and full system prompts are provided in the code repository. Random seeds were fixed at 42 for all primary analyses.

Computation times (Apple Silicon):
- Primary analysis with bootstrap CIs (N = 5,000; M = 500; B = 2,000): ~200 seconds
- Sensitivity analysis (16 configurations; N = 5,000; M = 500): ~320 seconds
- Expanded random search (200 configurations; N = 5,000; M = 500): ~16,200 seconds
- Multi-seed stability analysis (41 runs; N = 5,000; M = 500/5,000): ~800 seconds
- AI agent exploration (3 rounds, 5 agents): ~45 minutes (dominated by API latency)

Code is available at https://github.com/sanjaybasu/trajectory-risk-misordering.

---

## eTable 1. Expanded Random Search Comparison (200 Configurations)

Summary of 200 random parameter configurations drawn from the same parameter space as the AI agents and evaluated at the same simulation fidelity (N = 5,000; 500 trajectories per patient).

| Statistic | Value |
|---|---|
| Number of configurations | 200 |
| Mean $$\Delta$$ | 0.091 |
| Median $$\Delta$$ | 0.086 |
| Standard deviation (SD) | 0.042 |
| Interquartile range (IQR) | 0.060 to 0.112 |
| Maximum $$\Delta$$ | 0.270 |
| Configurations with $$\Delta$$ > 0.10 | 67 of 200 |
| Configurations with $$\Delta$$ > 0.20 | 3 of 200 |
| Configurations with $$\Delta$$ > 0.30 | 0 of 200 |

For comparison: agents' Round 1 best = 0.222; validated worst-case = 0.312. The parameter space was: $$\lambda_0$$ shape $$\in$$ [1, 8], $$\lambda_0$$ scale $$\in$$ [0.05, 0.35], $$\beta$$ a $$\in$$ [0.5, 5], $$\beta$$ b $$\in$$ [0.5, 8], $$\mu$$ shape $$\in$$ [1, 6], $$\mu$$ scale $$\in$$ [0.1, 2], $$k \in$$ {1, 2, 3, 5}, $$T \in$$ {0.5, 1.0, 2.0, 5.0}.

---

## eTable 2. Misordering Fraction by Social Risk Stratum (Primary Analysis)

| Social Risk Stratum | n | Mean $$\beta$$ | Mean $$\mu$$ (/yr) | $$\Delta$$ | 95% CI |
|---|---|---|---|---|---|
| Low | 2,268 | 0.30 | 3.91 | 0.035 | 0.033 to 0.037 |
| Moderate | 1,238 | 0.35 | 3.39 | 0.033 | 0.030 to 0.035 |
| High | 990 | 0.38 | 2.92 | 0.033 | 0.031 to 0.036 |
| Very high | 504 | 0.39 | 2.64 | 0.035 | 0.031 to 0.038 |

$$\Delta$$: misordering fraction for the standard risk score $$r$$ relative to gold-standard trajectory risk $$R$$ within each social risk stratum. Social risk strata were defined by additive offsets to the state-dependence parameter $$\beta$$ and multiplicative scales to the recovery rate $$\mu$$ (eAppendix D). 95% CIs from patient-resampling bootstrap (B = 2,000). Despite mean $$\beta$$ increasing from 0.30 to 0.39 and mean $$\mu$$ decreasing from 3.91 to 2.64/yr across strata, $$\Delta$$ remains consistent at 0.033–0.035, indicating that the degree of misordering by the standard score does not differ across social risk groups under the primary analysis parameters.

---

## eTable 3. Multi-Seed Stability Analysis

| Source of Variability | Score | Mean $$\Delta$$ | SD | IQR | Range |
|---|---|---|---|---|---|
| Population draws (20 seeds) | Standard | 0.035 | 0.0004 | 0.035 to 0.035 | 0.034 to 0.035 |
| | Trajectory-aware | 0.035 | 0.0004 | 0.035 to 0.035 | 0.034 to 0.035 |
| MC seeds (20 seeds, fixed pop) | Standard | 0.035 | 0.0006 | 0.035 to 0.036 | 0.034 to 0.036 |
| | Trajectory-aware | 0.035 | 0.0006 | 0.035 to 0.036 | 0.034 to 0.036 |
| High-precision (5,000 sims) | Standard | 0.019 | — | — | — |
| | Trajectory-aware | 0.018 | — | — | — |

Population variability: N = 5,000 patients drawn from the same parameter distributions with 20 independent seeds; M = 500 trajectories. MC variability: same primary population (seed 42), 20 independent simulation seeds; M = 500 trajectories. High-precision: same primary population, M = 5,000 trajectories. Both sources of variability are small (SD $$\leq$$ 0.001), confirming that $$\Delta$$ is stable across population draws and simulation seeds. The high-precision estimate ($$\Delta$$ = 0.019) is lower than the M = 500 estimate ($$\Delta$$ = 0.035) because Monte Carlo noise in the gold-standard trajectory risk $$R_i$$ introduces noise in pairwise rankings, inflating apparent misordering; with M = 5,000, $$R_i$$ is more precisely estimated. This means that the primary analysis provides a conservative (upward-biased) estimate of the true misordering fraction, strengthening the conclusion that standard scoring is robust under calibrated parameters.

---

## eFigure 1. Negative Binomial Fit Quality

Nine representative patients spanning the ($$\beta$$, $$\lambda_0$$) parameter space. Blue bars: empirical event count distribution from 500 Monte Carlo trajectories. Red circles/line: negative binomial predicted distribution using the analytical overdispersion formula (eAppendix A.3). Each panel reports the patient's $$\beta$$, $$\lambda_0$$, and $$\chi^2$$ goodness-of-fit p-value. The NB approximation provides adequate fit ($$\chi^2$$ p > 0.05) for the majority of patients, with the poorest fit occurring for patients with high $$\beta$$ and low $$\mu$$ (high overdispersion), consistent with the limitations described in eAppendix A.4.

---

## References (Supplementary)

Reference numbers correspond to the main text reference list.

1. Kansagara D, et al. JAMA. 2011;306(15):1688-1698.
3. Steyerberg EW. Clinical Prediction Models. 2nd ed. Springer; 2019.
5. Hawkes AG. Biometrika. 1971;58(1):83-90.
7. Krumholz HM. N Engl J Med. 2013;368(2):100-102.
8. Jencks SF, et al. N Engl J Med. 2009;360(14):1418-1428.
9. Hilbe JM. Negative Binomial Regression. 2nd ed. Cambridge University Press; 2011.
10. Lu C, et al. arXiv:2408.06292. 2024.
11. Dharmarajan K, et al. JAMA. 2013;309(4):355-363.
12. Joynt KE, Orav EJ, Jha AK. JAMA. 2011;305(7):675-681.
13. Wadhera RK, et al. BMJ. 2019;366:l4563.
14. Krijkamp EM, et al. Med Decis Making. 2018;38(3):400-422.
15. Harrell FE Jr, et al. Stat Med. 1996;15(4):361-387.
16. Kendall MG. Rank Correlation Methods. 4th ed. Charles Griffin; 1970.
17. Peters O. Nat Phys. 2019;15(12):1216-1221.
18. Obermeyer Z, et al. Science. 2019;366(6464):447-453.
19. Caro JJ, et al. Med Decis Making. 2012;32(5):667-677.
20. Collins GS, et al. BMJ. 2024;385:e078378.
21. Gneiting T, Raftery AE. J Am Stat Assoc. 2007;102(477):359-378.
22. Hilden J, Gerds TA. Stat Med. 2014;33(19):3405-3414.
23. Pepe MS, et al. Stat Biosci. 2015;7(2):282-295.
