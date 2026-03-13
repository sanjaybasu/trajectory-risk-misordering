# TITLE PAGE (separate file for double-blind review)

**Title:** Using Adversarial AI Agents to Stress-Test Health Decision Models: Application to Clinical Risk Scoring

**Authors:**

Sanjay Basu, MD, PhD<sup>1,2</sup>; Maya B. Mathur, PhD<sup>3</sup>

**Affiliations:**

1. University of California San Francisco, San Francisco, CA, USA
2. Waymark, San Francisco, CA, USA
3. Stanford University, Stanford, CA, USA

**Corresponding Author:**

Sanjay Basu, MD, PhD
University of California San Francisco
Email: sanjay.basu@waymarkcare.com

**Word Count:** 3,500

**Tables and Figures:** 4 tables, 2 figures (main text); 3 eTables, 1 eFigure (appendix)

**Keywords:** microsimulation, sensitivity analysis, AI agents, risk prediction, self-exciting process, proper scoring rules

---

# ANONYMIZED MAIN DOCUMENT

## Abstract

**Background.** Microsimulations underpin health policy and clinical decisions, yet their behavior across the parameter space is typically explored through mechanical sensitivity analyses — one-at-a-time sweeps and probabilistic sensitivity analysis (PSA) — that identify which parameters matter but not why or when model assumptions break. We propose adversarial AI-agent exploration, in which multiple large language model (LLM) agents with distinct methodological perspectives compete to find parameter regimes that stress-test a model's core assumptions.

**Methods.** Five AI agents (Claude Opus 4; combinatorics, survival analysis, ergodicity economics, self-exciting process theory, and algorithmic fairness perspectives) competed across three iterative rounds to maximize the misordering fraction ($$\Delta$$) of a hospitalization cascade microsimulation — the probability that a standard risk score incorrectly ranks a pair of patients relative to their trajectory risk. Each proposed configuration was evaluated by running the full simulation (5,000 patients; 500 Monte Carlo trajectories each; verifiable computational reward). We compared agent-identified regimes against 200 random configurations from the same parameter space evaluated at the same simulation fidelity.

**Results.** Under literature-calibrated parameters, the standard risk score was robust ($$\Delta$$ = 0.035; C-statistic = 0.965; stable across 20 independent population draws, standard deviation [SD] 0.0004). AI agents identified three mechanistic principles that degrade scoring performance: (1) homogeneous baseline rates remove the standard score's discriminative signal ($$\Delta$$ = 0.222); (2) near-zero recovery rates create absorbing vulnerable states ($$\Delta$$ = 0.312); and (3) high baseline rate variance paradoxically strengthens standard scoring ($$\Delta$$ = 0.044). The best of 200 random configurations achieved $$\Delta$$ = 0.270 but produced no mechanistic interpretation. Extreme misordering required clinically implausible parameters, bounding the practical scope of the problem.

**Limitations.** Five agents across three rounds is a demonstration, not exhaustive search. The standard risk score uses true generative parameters (an oracle score), providing a best-case assessment.

**Conclusions.** Adversarial AI-agent exploration generates mechanistic understanding of microsimulation boundary conditions that traditional sensitivity analysis and random search do not. Applied to clinical risk scoring, the agents identified specific, interpretable conditions under which standard scores fail — and confirmed that these conditions are clinically implausible under realistic parameterizations.

---

## Highlights

- We propose adversarial AI-agent exploration — multiple LLM agents with distinct methodological perspectives competing to stress-test microsimulation assumptions — as a complement to probabilistic sensitivity analysis.
- Applied to a hospitalization cascade model, agents identified three mechanistic principles governing risk score failure: homogeneous baselines, absorbing vulnerable states, and a baseline variance paradox (high variance strengthens, rather than weakens, standard scoring).
- With 13 times more budget (200 vs 15 configurations), random search matched agents on optimization but produced no generalizable mechanistic insight; agents bounded the problem by showing extreme failure requires clinically implausible parameters.

---

## Introduction

Microsimulation models are central to health decision making. They inform cost-effectiveness analyses, resource allocation decisions, and clinical guideline development across disease areas from cancer screening to infectious disease control.<sup>14,19</sup> The validity of these models depends on understanding how their outputs change across the parameter space — yet the tools available for this task have not kept pace with the complexity of the models themselves.

Current approaches to parameter space exploration are mechanical. Probabilistic sensitivity analysis (PSA) draws parameters from prior distributions and propagates uncertainty to outputs; tornado diagrams identify the most influential parameters one at a time.<sup>24</sup> These methods answer "how sensitive is the output to parameter X?" but not "under what conditions does the model's core assumption break, and why?" For nonlinear models with parameter interactions — where the effect of one parameter depends on the value of another — mechanical exploration can miss critical regions of the parameter space entirely. Expected value of partial perfect information (EVPPI) analysis prioritizes parameters for future data collection but, like PSA, does not characterize the structure of model failure.<sup>25</sup>

We propose adversarial AI-agent exploration as a complement to these established methods. In this framework, multiple large language model (LLM) agents, each given a distinct methodological perspective, compete to find parameter configurations that stress-test a model's assumptions. Each proposed configuration is evaluated by running the full simulation — a verifiable computational reward that prevents confabulation. The agents' value lies not in optimization efficiency but in generating testable hypotheses about why certain parameter regimes produce unusual model behavior, which are then verified computationally.

We demonstrate this framework on a clinically relevant question: how robust are standard clinical risk scores to the dynamics of hospitalization cascades? Care management programs use risk scores to rank patients by their expected event rate and allocate resources accordingly.<sup>1,2,4</sup> These scores capture the mean of each patient's event rate distribution but not the tail — the probability of accumulating multiple events in a short period, which depends on cascade dynamics documented in post-hospital syndrome.<sup>7,8</sup> We formalize this as the misordering fraction ($$\Delta$$): the probability that a standard risk score incorrectly ranks a pair of patients relative to their trajectory risk.<sup>15,16</sup> We use AI agents to identify when, why, and how much standard scoring fails — and to bound the conditions under which it remains reliable.

## Methods

### AI-Agent Adversarial Exploration Framework

Five agents were implemented as calls to Claude Opus 4 (Anthropic, model ID claude-opus-4-6) with persona-specific system prompts, each encoding a distinct methodological perspective (Table 2).<sup>10</sup> In each round, every agent received: (a) a description of the microsimulation model and parameter space, (b) the misordering fraction metric ($$\Delta$$), and (c) results from all prior rounds, including other agents' configurations and $$\Delta$$ values. Each agent proposed a parameter configuration with a written mechanistic rationale. Three rounds were conducted; agents showed diminishing returns after Round 2, with Round 3 used to validate and refine earlier findings rather than produce novel configurations.

The verifiable reward was critical: each proposed configuration was evaluated by running the full microsimulation (N = 5,000 patients; 500 Monte Carlo trajectories per patient), producing a $$\Delta$$ value that was independent of the agent's own assessment. This grounds the exploration in computation rather than in the agent's reasoning alone, since LLM-generated rationales could otherwise be post-hoc confabulations.<sup>10</sup>

To benchmark agent performance, we drew 200 random parameter configurations from the same parameter space ($$\lambda_0$$ shape $$\in$$ [1, 8], $$\lambda_0$$ scale $$\in$$ [0.05, 0.35], $$\beta$$ a $$\in$$ [0.5, 5], $$\beta$$ b $$\in$$ [0.5, 8], $$\mu$$ shape $$\in$$ [1, 6], $$\mu$$ scale $$\in$$ [0.1, 2], $$k \in$$ {1, 2, 3, 5}, $$T \in$$ {0.5, 1.0, 2.0, 5.0} years) and evaluated each by the same simulation protocol (N = 5,000; 500 trajectories). This gave the random search 13 times the budget of the agents' 15 total configurations.

### Ethical Oversight

This study used simulated data generated from a microsimulation model with parameters calibrated to published literature. No patient data were used. No institutional review board approval was required.

### Case Study: Hospitalization Cascade Microsimulation

We modeled patient health dynamics as a two-state self-exciting process.<sup>5,6</sup> Each simulated patient $$i$$ ($$i = 1, \ldots, N$$) was characterized by four patient-specific parameters: a baseline event rate $$\lambda_{0i}$$ (events per year in the stable state), an elevated event rate $$\lambda_{1i}$$ (events per year in the vulnerable state), a state-dependence parameter $$\beta_i$$ (the probability that an adverse event triggers a transition to the vulnerable state), and a recovery rate $$\mu_i$$ (the rate of return to stability, per year). This captures the self-exciting dynamics documented in hospitalization cascades: an adverse event can trigger a vulnerable period in which subsequent events become more likely.<sup>7,11</sup>

We generated N = 5,000 patients with parameters calibrated to published data on hospitalization dynamics among high-risk adults (Table 1).<sup>7,8,11</sup> Baseline hospitalization rates ($$\lambda_0$$) were drawn from a Gamma(3.0, 0.2) distribution (mean 0.60 events/year, SD 0.34). Elevated rates during vulnerable periods ($$\lambda_1$$) were set at approximately 2 to 4 times baseline. The state-dependence parameter ($$\beta$$) was drawn from a Beta(3.0, 7.0) distribution (mean 0.30, SD 0.14). Recovery rates ($$\mu$$) were drawn from a Gamma(4.0, 1.0) distribution (mean 4.0/year). To evaluate robustness, we generated low-acuity and high-acuity populations (eAppendix D). Four social risk strata were generated with differential state-dependence and recovery parameters.<sup>12,13</sup>

### Risk Scores Compared

We computed four risk scores and compared each against a gold-standard measure of trajectory risk. Importantly, all scores use the true generative parameters ($$\lambda_0$$, $$\lambda_1$$, $$\beta$$, $$\mu$$) rather than estimates from observed data. This provides a best-case assessment: any misordering found under oracle conditions would be at least as severe with estimated parameters in practice.

The standard risk score ($$r_i$$) was computed as the steady-state event rate: $$r_i = \lambda_{0i}(1 - \pi_i) + \lambda_{1i} \pi_i$$, where $$\pi_i$$ is the stationary vulnerable-state occupancy (eAppendix A).<sup>1,3</sup>

The trajectory-aware score ($$\hat{R}_{ai}$$) was computed as the tail probability $$P(\geq k \text{ events in } [0, T])$$ using a negative binomial approximation that captures overdispersion from self-exciting dynamics (eAppendix A).<sup>9</sup>

The maximum likelihood estimation (MLE) negative binomial score was obtained by fitting a negative binomial distribution via method of moments to each patient's simulated event counts, then computing $$P(\geq k)$$ from the fitted distribution (eAppendix A.7).

The augmented score was computed as $$r_i \times (1 + \beta_i / \mu_i)$$, a heuristic scaling. We included it to test whether naive parameter incorporation degrades performance.

Gold-standard trajectory risk ($$R_i$$) was $$P(\geq k \text{ events in } [0, T])$$ via Monte Carlo microsimulation (500 trajectories per patient; daily time steps; eAppendix B).<sup>14</sup> Primary analysis: $$T$$ = 2.0 years, $$k$$ = 3.

### Evaluation Metrics

The misordering fraction ($$\Delta$$) is the probability of an incorrect pairwise ranking: $$\Delta = P(\text{score}_i > \text{score}_j \text{ and } R_i < R_j) + P(\text{score}_i < \text{score}_j \text{ and } R_i > R_j)$$ over random patient pairs, excluding ties. The C-statistic equals 1 $$-$$ $$\Delta$$; Kendall's $$\tau$$ equals 1 $$-$$ 2$$\Delta$$.<sup>15,16</sup>

We evaluated calibration using two strictly proper scoring rules.<sup>21</sup> The Brier score was computed for binary classification of high trajectory risk ($$R_i \geq$$ 75th percentile of the simulated population). The continuous ranked probability score (CRPS) was computed as the mean absolute error between the normalized predicted score and normalized gold-standard trajectory risk; for the deterministic (point) forecasts evaluated here, CRPS reduces to the normalized mean absolute error (MAE).<sup>21</sup> Both are strictly proper — uniquely minimized at true probabilities — unlike the net reclassification improvement.<sup>22,23</sup> We report Pearson ($$\rho$$) and Spearman ($$\rho_s$$) correlations with gold-standard trajectory risk.

### Sensitivity Analysis and Statistical Analysis

We evaluated $$\Delta$$ across a grid of catastrophic thresholds ($$k \in$$ {1, 2, 3, 5}) and time horizons ($$T \in$$ {0.5, 1.0, 2.0, 5.0 years}) using the primary population (N = 5,000; 500 trajectories). Bootstrap 95% confidence intervals (CIs; B = 2,000) were computed for the primary analysis by resampling patients with replacement; these CIs quantify patient-sampling uncertainty conditional on the specific Monte Carlo realization.

To characterize total variability in $$\Delta$$, we conducted a multi-seed stability analysis. We recomputed $$\Delta$$ across 20 independent population draws (different random seeds for patient-parameter generation) and across 20 independent Monte Carlo simulation seeds (same population, different trajectory realizations). This separates two sources of variability: population sampling and Monte Carlo noise (eTable 3).

All scores were computed on the same N = 5,000 patients, so differences in $$\Delta$$ between scoring methods reflect paired comparisons within the same patient sample. Patient-level parameters were drawn iid from specified distributions. All analyses used Python 3.11, NumPy 1.26, SciPy 1.12, with primary random seed 42. This study follows International Society for Pharmacoeconomics and Outcomes Research and Society for Medical Decision Making (ISPOR-SMDM) Modeling Good Research Practices<sup>19</sup> and Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis plus Artificial Intelligence (TRIPOD+AI).<sup>20</sup> Code: https://github.com/sanjaybasu/trajectory-risk-misordering.

## Results

### Case Study Baseline: Risk Score Performance Under Calibrated Parameters

The primary simulated population (N = 5,000) had a mean baseline event rate of 0.60 events/year (SD 0.34), mean $$\beta$$ = 0.34 (SD 0.15), mean $$\mu$$ = 3.46/year (SD 1.78), and mean gold-standard trajectory risk $$R$$ = 0.17 (SD 0.17; Table 1).

Under literature-calibrated parameters, the standard risk score was robust: $$\Delta$$ = 0.035 (95% CI, 0.033 to 0.035), C-statistic = 0.965, Spearman $$\rho_s$$ = 0.99. Across 20 independent population draws, $$\Delta$$ for the standard score had mean 0.035 (SD 0.0004; interquartile range [IQR] 0.035 to 0.035), confirming stability. Monte Carlo variability was comparable: across 20 simulation seeds for the same population, SD = 0.0006 (eTable 3). No patients in the top decile of trajectory risk were ranked below the population median (Table 3). The trajectory-aware score achieved identical concordance ($$\Delta$$ = 0.035; C = 0.965; $$\rho_s$$ = 0.99) but was better calibrated: Brier score 0.093 vs 0.142; CRPS 0.018 vs 0.057. The MLE negative binomial score outperformed both on concordance ($$\Delta$$ = 0.018; C = 0.982) and CRPS (0.007), demonstrating that trajectory risk can be recovered from observable event count data. The augmented score performed worst ($$\Delta$$ = 0.052; Brier 0.151).

Misordering increased with population acuity: $$\Delta$$ = 0.050 (low acuity), 0.035 (primary), 0.061 (high acuity). The trajectory-aware score's calibration advantage was largest in the high-acuity population (CRPS 0.068 vs 0.299). $$\Delta$$ varied with clinical context: highest for short horizons and moderate thresholds ($$\Delta$$ = 0.098 at T = 0.5 years, $$k$$ = 3), lowest for long horizons at the primary threshold ($$\Delta$$ = 0.030 at T = 5.0 years, $$k$$ = 3; Table 4). These baseline findings established the case study's parameters: a model where standard scoring usually works but whose failure modes were not yet characterized.

### AI-Agent Findings

Across three rounds, agents identified three mechanistic principles governing when standard risk scores fail (Table 2; Figure 1).

**Finding 1: Homogeneous baselines remove discriminative signal.** The combinatorics agent identified that making baseline event rates nearly identical across patients (high shape parameter for the $$\lambda_0$$ distribution, producing a tight distribution with low coefficient of variation), while simultaneously allowing wide variation in cascade propensity ($$\beta$$), maximally decouples the standard score from trajectory risk ($$\Delta$$ = 0.222). The proposed mechanism: when all patients have similar $$\lambda_0$$, the standard score $$r$$ compresses into a narrow range, losing the ability to discriminate between patients. Trajectory risk still varies widely because it depends on $$\beta$$, which the standard score underweights. We verified this computationally: restricting the population to patients within the middle 50% of $$\lambda_0$$ increased $$\Delta$$ from 0.035 to 0.082, confirming the mechanism.

**Finding 2: Absorbing vulnerable states create permanent divergence.** The ergodicity economics agent, drawing on Peters' framework of time-ensemble divergence,<sup>17</sup> set recovery rates near zero ($$\mu \approx 0.01$$, implying a 100-year average recovery time). This creates an effectively absorbing vulnerable state: once a patient enters the cascade, recovery is negligible. Under these conditions, $$\Delta$$ = 0.461 (Round 2); re-evaluated with the full protocol (N = 5,000; 500 trajectories), $$\Delta$$ = 0.312, with 14.0% of patients in the top trajectory-risk decile ranked below the population median by the standard score.

**Finding 3 (negative result): High baseline variance strengthens standard scoring.** The survival analysis agent proposed that increasing variance in baseline rates would amplify misordering by creating more diverse event patterns. Instead, $$\Delta$$ decreased to 0.044 — lower than the calibrated baseline. The explanation: high Var($$\lambda_0$$) makes the standard score $$r$$ more informative because baseline rate heterogeneity dominates the ranking, leaving less room for state-dependence to produce discordance. This counter-intuitive negative result would be difficult to identify through mechanical parameter sweeps, which test parameters independently.

Additional agent contributions included: the self-exciting process agent achieving $$\Delta$$ = 0.083 through saturated $$\beta$$ and high excitation rates; the algorithmic fairness agent achieving $$\Delta$$ = 0.152 through extreme group offsets, with the proposed mechanism being that differential cascade propensity across groups amplifies misordering when between-group $$\beta$$ variance exceeds within-group $$\lambda_0$$ variance.

### Agent Versus Random Search

The best of 200 random configurations achieved $$\Delta$$ = 0.270; 3 of 200 configurations exceeded $$\Delta$$ = 0.20 (eTable 1). The agents produced 15 total configurations (5 agents $$\times$$ 3 rounds), with a Round 1 best of 0.222 and a validated worst-case of 0.312. Even with 13 times the search budget and the same simulation fidelity (N = 5,000; 500 trajectories), random search did not exceed the agents' validated worst-case.

The agents' advantage was not optimization efficiency — random search, and methods like Bayesian optimization, can find high-$$\Delta$$ regions. The advantage was that agents produced testable mechanistic hypotheses alongside each configuration: that tight $$\lambda_0$$ distributions remove discriminative signal, that near-zero $$\mu$$ creates absorbing states connecting to ergodicity economics,<sup>17</sup> and that increasing baseline variance strengthens scoring. Post-hoc regression analysis of the 200 random configurations could in principle recover similar parameter-outcome relationships, but would not generate the structured causal reasoning or connect findings to established theory (e.g., ergodicity economics, branching processes). The agents delivered interpretation in real time, integrated with the search itself.

The agents also bounded the problem: extreme misordering ($$\Delta$$ > 0.30) required recovery rates implying vulnerability durations exceeding decades, clinically implausible for most patient populations. This bounding — establishing not just where the model breaks but that breaking it requires unrealistic parameters — is a distinct contribution.

### Social Risk Stratification

$$\Delta$$ was consistent across social risk strata under calibrated parameters, ranging from 0.033 for moderate social risk to 0.035 for low social risk (eTable 2), indicating that misordering by the standard score does not differ meaningfully across social risk groups.

## Discussion

Adversarial AI-agent exploration generates a qualitatively different kind of understanding from traditional sensitivity analysis. PSA quantifies output uncertainty; tornado diagrams rank parameter importance; EVPPI prioritizes data collection — all essential tools for model-based decision making.<sup>24,25</sup> None of these methods ask: "under what structural conditions does the model's core assumption break, and what is the mechanism?" The agents answered this question for a hospitalization cascade microsimulation, producing three interpretable principles — homogeneous baselines, absorbing states, and the baseline variance paradox — and bounded the conditions under which standard risk scoring remains reliable.

The agents' advantage was not search efficiency. Random search with 200 configurations (13 times the agent budget) achieved $$\Delta$$ = 0.270, below the agents' validated worst-case of 0.312. But random search produced configurations with no structural understanding. The agents produced testable hypotheses that were verified computationally: restricting baseline rate heterogeneity increased $$\Delta$$, confirming the homogeneity mechanism; setting $$\mu \to 0$$ produced absorbing states, confirming the ergodicity mechanism. We acknowledge that post-hoc statistical analysis of random search outputs (e.g., regressing $$\Delta$$ on parameter values) could also identify important parameter associations — but would not generate the causal framing or theoretical connections that the agents provided.

The case study findings are reassuring for clinical practice. Standard risk scores correctly rank patients by trajectory risk 96.5% of the time (C-statistic = 0.965) under literature-calibrated parameters — and this finding was stable across 20 independent population draws (SD 0.0004). Both standard and trajectory-aware scores achieve equivalently high rank correlations (Spearman $$\rho_s$$ = 0.99); the difference lies in calibration, not discrimination, as revealed by strictly proper scoring rules (Brier 0.093 vs 0.142; CRPS 0.018 vs 0.057; Figure 2).<sup>21</sup> The MLE negative binomial score (C = 0.982; CRPS = 0.007) demonstrates that fitting event count distributions to longitudinal data — feasible with administrative claims — can substantially improve both concordance and calibration. The augmented score's poor performance ($$\Delta$$ = 0.052; Brier 0.151) confirms that naive heuristic adjustments degrade rather than improve scoring.

The framework is generalizable. Any microsimulation with a computable output metric can serve as the verifiable reward. The agent perspectives can be adapted to the domain: for a cost-effectiveness model, agents might take the perspectives of a health economist, a clinician, a patient advocate, and a payer. The key design principle is that agent-proposed configurations must be evaluated by running the actual simulation, ensuring that the exploration is grounded in computation rather than LLM reasoning alone.

This study has several limitations. Five agents across three rounds is a proof of concept, not exhaustive exploration; additional rounds showed diminishing returns but a formal stopping rule was not specified. LLM agents are stochastic; the same prompts with a different model version or random seed may produce different configurations. We report the exact model version (Claude Opus 4, claude-opus-4-6) and full prompts (eAppendix C) to support reproducibility. The agents did not outperform random search on raw optimization, and for purely finding maxima, Bayesian optimization or evolutionary algorithms would be more efficient. The value proposition is interpretability, which we have not formally benchmarked against structured post-hoc analysis of random search results. The case study uses a simplified two-state model; real hospitalization dynamics involve multiple comorbidities and social determinants. Parameters were calibrated to published aggregates, not fitted to individual-level data, and the standard score uses true generative parameters — an oracle condition more favorable to standard scoring than any real-world implementation. The Brier score threshold (75th percentile) was defined from the same population used for evaluation, standard for simulation studies but worth noting. The CRPS for these deterministic point forecasts reduces to normalized mean absolute error.

For the microsimulation community, adversarial AI-agent exploration offers a practical complement to PSA, tornado diagrams, and EVPPI — not a replacement but an additional tool for generating mechanistic understanding of model behavior. For clinical risk scoring, the agents bounded a specific concern: standard scores have a blind spot for cascade dynamics, but the blind spot is small under realistic conditions and resolvable through trajectory-aware calibration correction or negative binomial fitting to longitudinal event counts.

---

## References

1. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA. 2011;306(15):1688-1698. DOI: 10.1001/jama.2011.1515
2. Bates DW, Saria S, Ohno-Machado L, Shah A, Escobar G. Big data in health care: using analytics to identify and manage high-risk and high-cost patients. Health Aff (Millwood). 2014;33(7):1123-1131. DOI: 10.1377/hlthaff.2014.0041
3. Steyerberg EW. Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating. 2nd ed. Cham: Springer; 2019. DOI: 10.1007/978-3-030-16399-0
4. McIlvennan CK, Eapen ZJ, Allen LA. Hospital readmissions reduction program. Circulation. 2015;131(20):1796-1803. DOI: 10.1161/CIRCULATIONAHA.114.010270
5. Hawkes AG. Spectra of some self-exciting and mutually exciting point processes. Biometrika. 1971;58(1):83-90. DOI: 10.1093/biomet/58.1.83
6. Cook RJ, Lawless JF. The Statistical Analysis of Recurrent Events. New York: Springer; 2007.
7. Krumholz HM. Post-hospital syndrome — an acquired, transient condition of generalized risk. N Engl J Med. 2013;368(2):100-102. DOI: 10.1056/NEJMp1212324
8. Jencks SF, Williams MV, Coleman EA. Rehospitalizations among patients in the Medicare fee-for-service program. N Engl J Med. 2009;360(14):1418-1428. DOI: 10.1056/NEJMsa0803563
9. Hilbe JM. Negative Binomial Regression. 2nd ed. Cambridge: Cambridge University Press; 2011.
10. Lu C, Lu C, Lange RT, et al. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv preprint arXiv:2408.06292. 2024.
11. Dharmarajan K, Hsieh AF, Lin Z, et al. Diagnoses and timing of 30-day readmissions after hospitalization for heart failure, acute myocardial infarction, or pneumonia. JAMA. 2013;309(4):355-363. DOI: 10.1001/jama.2012.216476
12. Joynt KE, Orav EJ, Jha AK. Thirty-day readmission rates for Medicare beneficiaries by race and site of care. JAMA. 2011;305(7):675-681. DOI: 10.1001/jama.2011.123
13. Wadhera RK, Joynt Maddox KE, Kazi DS, Shen C, Yeh RW. Hospital revisits within 30 days after discharge for medical conditions targeted by the Hospital Readmissions Reduction Program in the United States: national retrospective analysis. BMJ. 2019;366:l4563. DOI: 10.1136/bmj.l4563
14. Krijkamp EM, Alarid-Escudero F, Enns EA, Jalal HJ, Hunink MGM, Pechlivanoglou P. Microsimulation modeling for health decision sciences using R: a tutorial. Med Decis Making. 2018;38(3):400-422. DOI: 10.1177/0272989X18754513
15. Harrell FE Jr, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat Med. 1996;15(4):361-387.
16. Kendall MG. Rank Correlation Methods. 4th ed. London: Charles Griffin; 1970.
17. Peters O. The ergodicity problem in economics. Nat Phys. 2019;15(12):1216-1221. DOI: 10.1038/s41567-019-0732-0
18. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453. DOI: 10.1126/science.aax2342
19. Caro JJ, Briggs AH, Siebert U, Kuntz KM; ISPOR-SMDM Modeling Good Research Practices Task Force. Modeling good research practices — overview: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-1. Med Decis Making. 2012;32(5):667-677. DOI: 10.1177/0272989X12454577
20. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378
21. Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. J Am Stat Assoc. 2007;102(477):359-378. DOI: 10.1198/016214506000001437
22. Hilden J, Gerds TA. A note on the evaluation of novel biomarkers: do not rely on integrated discrimination improvement and net reclassification index. Stat Med. 2014;33(19):3405-3414. DOI: 10.1002/sim.6167
23. Pepe MS, Fan J, Feng Z, Gerds T, Hilden J. The net reclassification index (NRI): a misleading measure of prediction improvement even with independent test data sets. Stat Biosci. 2015;7(2):282-295. DOI: 10.1007/s12561-014-9118-0
24. Briggs AH, Weinstein MC, Fenwick EAL, Karnon J, Sculpher MJ, Paltiel AD. Model parameter estimation and uncertainty analysis: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force Working Group-6. Med Decis Making. 2012;32(5):722-732. DOI: 10.1177/0272989X12458348
25. Claxton K, Sculpher M, McCabe C, et al. Probabilistic sensitivity analysis for NICE technology assessment: not an optional extra. Health Econ. 2005;14(4):339-347. DOI: 10.1002/hec.985

---

## Tables and Figures

### Table 1. Literature-Calibrated Simulated Population Characteristics

| Parameter | Distribution | Primary | Low Acuity | High Acuity | Calibration Source |
|---|---|---|---|---|---|
| N | — | 5,000 | 5,000 | 5,000 | — |
| Baseline event rate $$\lambda_0$$ (events/yr) | Gamma | 0.60 (0.34) | 0.30 (0.21) | 1.00 (0.50) | Jencks 2009<sup>8</sup> |
| Elevated event rate $$\lambda_1$$ (events/yr) | $$\lambda_0 \times$$ (1 + Gamma) | 1.58 (1.22) | 0.58 (0.50) | 3.42 (2.31) | Krumholz 2013<sup>7</sup> |
| State-dependence $$\beta$$ | Beta + group offset | 0.34 (0.15) | 0.25 (0.13) | 0.43 (0.16) | Dharmarajan 2013<sup>11</sup> |
| Recovery rate $$\mu$$ (/yr) | Gamma $$\times$$ group scale | 3.46 (1.78) | 5.38 (2.48) | 1.99 (1.21) | Krumholz 2013<sup>7</sup> |
| Standard risk score $$r$$ (events/yr) | — | 0.69 (0.45) | 0.31 (0.22) | 1.61 (1.19) | — |
| Trajectory risk $$R$$ | — | 0.17 (0.17) | 0.04 (0.07) | 0.42 (0.24) | — |
| **Social Risk Stratum (Primary)** | **n** | **Mean $$\beta$$** | **Mean $$\mu$$ (/yr)** | | |
| Low social risk | 2,268 | 0.30 | 3.91 | | |
| Moderate social risk | 1,238 | 0.35 | 3.39 | | |
| High social risk | 990 | 0.38 | 2.92 | | |
| Very high social risk | 504 | 0.39 | 2.64 | | |

Values shown as mean (SD). Primary analysis: T = 2.0 years, $$k$$ = 3 events, 500 Monte Carlo trajectories per patient. The standard risk score $$r$$ is the steady-state event rate computed from true generative parameters (oracle condition). The trajectory risk $$R$$ is P($$\geq$$3 events in 2 years) from Monte Carlo simulation.

### Table 2. AI-Agent Adversarial Exploration: Findings Across Three Rounds

| Agent Perspective | Round | $$\Delta$$ | $$\tau$$ | Strategy and Mechanistic Rationale |
|---|---|---|---|---|
| (Clinically calibrated baseline) | — | 0.035 | 0.931 | Literature-calibrated parameters; standard scoring is robust |
| **Combinatorics** | 1 | **0.222** | 0.557 | Tight $$\lambda_0$$ distribution (low CV), bimodal $$\beta$$. Rationale: homogeneous baselines remove $$r$$'s discriminative signal; trajectory risk varies via $$\beta$$ alone |
| **Survival analysis** | 1 | 0.044 | 0.912 | High Var($$\lambda_0$$). Negative result: high baseline variance strengthens $$r$$ because baseline heterogeneity dominates ranking |
| Ergodicity economics | 1 | 0.119 | 0.762 | Low $$\mu$$, high $$\beta$$ variance. Rationale: slow recovery amplifies cascade dynamics |
| Self-exciting process | 1 | 0.083 | 0.834 | Saturated $$\beta$$, high $$\lambda_1/\lambda_0$$. Rationale: near-critical branching ratio |
| Algorithmic fairness | 1 | 0.152 | 0.696 | Extreme group offsets. Rationale: between-group $$\beta$$ variance exceeds within-group $$\lambda_0$$ variance |
| **Ergodicity economics** | 2 | **0.461** | 0.077 | $$\mu \approx 0.01$$ (absorbing state). Rationale: permanent vulnerability creates non-ergodic dynamics<sup>17</sup> |
| **Validated worst-case** | 3 | **0.312** | 0.376 | Peters-combinatorics synthesis (N = 5,000; 500 sims). Bound: requires $$\mu$$ implying >10-year recovery |
| Random search best (of 200) | — | 0.270 | 0.461 | No mechanistic rationale provided |

$$\Delta$$: misordering fraction (fraction of patient pairs incorrectly ranked by the standard score relative to trajectory risk). $$\tau$$: Kendall's rank correlation ($$\tau = 1 - 2\Delta$$). The combinatorics and ergodicity agents identified the two principal mechanisms of scoring failure; the survival agent provided a counter-intuitive negative result. The random search best is the highest $$\Delta$$ from 200 configurations drawn from the same parameter space and evaluated at the same simulation fidelity (N = 5,000; 500 trajectories); it produced no mechanistic rationale. The validated worst-case ($$\Delta$$ = 0.312) required clinically implausible parameters (recovery time >10 years), bounding the practical scope of the problem.

### Table 3. Risk Score Performance Across Population Acuity Levels

| Population | Score | $$\Delta$$ (95% CI) | C-statistic | $$\rho$$ | $$\rho_s$$ | Brier | CRPS |
|---|---|---|---|---|---|---|---|
| **Primary** | Standard ($$r$$) | 0.035 (0.033 to 0.035) | 0.965 | 0.97 | 0.99 | 0.142 | 0.057 |
| | Trajectory-aware ($$\hat{R}_a$$) | 0.035 (0.033 to 0.036) | 0.965 | 0.99 | 0.99 | 0.093 | 0.018 |
| | MLE NB | 0.018 | 0.982 | 1.00 | 1.00 | 0.099 | 0.007 |
| | Augmented ($$r \times f$$) | 0.052 (0.050 to 0.054) | 0.948 | 0.93 | 0.98 | 0.151 | 0.065 |
| **Low acuity** | Standard ($$r$$) | 0.050 | 0.950 | 0.94 | 0.97 | 0.137 | 0.091 |
| | Trajectory-aware ($$\hat{R}_a$$) | 0.049 | 0.951 | 0.99 | 0.97 | 0.181 | 0.012 |
| | Augmented ($$r \times f$$) | 0.051 | 0.949 | 0.94 | 0.97 | 0.141 | 0.083 |
| **High acuity** | Standard ($$r$$) | 0.061 | 0.939 | 0.86 | 0.98 | 0.155 | 0.299 |
| | Trajectory-aware ($$\hat{R}_a$$) | 0.049 | 0.951 | 0.98 | 0.99 | 0.136 | 0.068 |
| | Augmented ($$r \times f$$) | 0.114 | 0.886 | 0.67 | 0.92 | 0.201 | 0.365 |

$$\Delta$$: misordering fraction. C-statistic = 1 $$-$$ $$\Delta$$. $$\rho$$: Pearson correlation with gold-standard trajectory risk. $$\rho_s$$: Spearman rank correlation. Brier: Brier score for binary high-risk classification ($$\geq$$75th percentile of the simulated population); lower is better.<sup>21</sup> CRPS: normalized mean absolute error (equivalent to CRPS for deterministic forecasts); lower is better.<sup>21</sup> MLE NB: negative binomial fitted to simulated event counts. 95% CIs from patient-resampling bootstrap (B = 2,000) for the primary analysis; secondary analyses report point estimates (N = 5,000; 500 trajectories). All scores are computed on the same patients (paired comparison). Standard and trajectory-aware scores achieve equivalent discrimination ($$\Delta$$, $$\rho_s$$) but differ in calibration (Brier, CRPS). The MLE score demonstrates that trajectory risk can be recovered from observable event counts.

### Table 4. Sensitivity of $$\Delta$$ to Catastrophic Threshold ($$k$$) and Time Horizon ($$T$$)

| | T = 0.5 yr | T = 1.0 yr | T = 2.0 yr | T = 5.0 yr |
|---|---|---|---|---|
| **Standard risk score $$\Delta$$** | | | | |
| $$k$$ = 1 | 0.065 | 0.057 | 0.052 | 0.052 |
| $$k$$ = 2 | 0.065 | 0.045 | 0.036 | 0.039 |
| $$k$$ = 3 | 0.098 | 0.060 | 0.035 | 0.030 |
| $$k$$ = 5 | 0.087 | 0.089 | 0.058 | 0.026 |
| **Trajectory-aware score $$\Delta$$** | | | | |
| $$k$$ = 1 | 0.053 | 0.045 | 0.043 | 0.048 |
| $$k$$ = 2 | 0.077 | 0.047 | 0.033 | 0.036 |
| $$k$$ = 3 | 0.111 | 0.063 | 0.035 | 0.028 |
| $$k$$ = 5 | 0.099 | 0.087 | 0.054 | 0.025 |

Misordering fraction $$\Delta$$ across 16 combinations of catastrophic threshold $$k$$ and time horizon $$T$$, using the primary population (N = 5,000; 500 trajectories). The trajectory-aware score reduces $$\Delta$$ in 10 of 16 combinations, with the largest reductions at long horizons and low thresholds. At short horizons (T $$\leq$$ 1.0 years) and moderate thresholds ($$k \geq$$ 2), the trajectory-aware score produces higher $$\Delta$$, reflecting limited accuracy of the negative binomial approximation when the process has insufficient time to mix.

### Figure 1. AI-Agent Discovery Progression

Misordering fraction ($$\Delta$$) across clinically calibrated and AI-agent-identified parameter regimes. Blue bars: clinically calibrated populations. Orange bars: agent findings exceeding the calibrated range. Gray bar: agent strategy that decreased $$\Delta$$ (survival analysis agent; negative result). Red bars: configurations requiring clinically implausible parameters (recovery rate $$\mu \approx$$ 0.01/year). Dashed line at $$\Delta$$ = 0.5: random ranking. The validated worst-case ($$\Delta$$ = 0.312) required parameters implying >10-year average recovery time, bounding the practical scope of the problem.

### Figure 2. Mechanism of Misordering: Standard Score Versus Trajectory Risk

(A) Standard risk score $$r_i$$ versus gold-standard trajectory risk $$R_i$$ (N = 5,000), colored by state-dependence $$\beta_i$$. Dashed line: linear regression (Pearson $$\rho$$ = 0.97). Patients with high $$\beta_i$$ (red) fall above the line. (B) Residuals from regression in (A), stratified by $$\beta_i$$ quartile. The highest $$\beta_i$$ quartile has the largest positive residuals, confirming cascade propensity as the driver of discrepancy between standard and trajectory risk.
