"""
NYHA I-IV + Death multi-state heart-failure microsimulation.

This is the analytically intractable model used to address Reviewer 1's
concern that the two-state self-exciting model has closed-form steady-state
quantities. The standard risk score is the conditional expected
hospitalisation rate given current state and covariates, derived from a Cox
proportional-hazards model fitted on simulated data. The trajectory risk is
the probability of accumulating >= k HF hospitalisations within T years.
There is no closed-form steady-state event rate.

Patient state: NYHA class in {1, 2, 3, 4, D} (D = death, absorbing).
Covariates: age (continuous), eGFR (continuous), ejection_fraction (continuous),
diabetes (binary). All time-invariant for simplicity (clinically reasonable
over 2-5 year horizons).

Per-cycle transitions (cycle = 1 month = 1/12 yr):
  P(progress NYHA k -> k+1) = 1 - exp(-h_prog(k, X) * dt)
  P(regress NYHA k -> k-1) = 1 - exp(-h_reg(k) * dt)
  P(death from NYHA k)     = 1 - exp(-h_death(k, X) * dt)
  P(HF hospitalisation | NYHA k, X) per cycle = 1 - exp(-h_hosp(k, X) * dt)

A hospitalisation does not change NYHA state (state changes are independent
Cox processes). After a hospitalisation, h_hosp is multiplied by an
event-amplification factor amp(k) for the next decay_months months,
introducing self-excitation analogous to the two-state model.

This produces a microsimulation with NO closed-form expected event rate:
the standard score must be a Cox-PH approximation derived from observed
event histories, which differs systematically from the true rate.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict


DEATH = 4  # state index for death (absorbing)
N_NYHA = 4  # NYHA I..IV are states 0..3
DT = 1 / 12.0  # one-month cycle
SEED_OFFSET = 0


# ------------------------------------------------------------------
# Patient generation
# ------------------------------------------------------------------

@dataclass
class HFPopulation:
    n: int
    age: np.ndarray
    egfr: np.ndarray
    ef: np.ndarray
    diabetes: np.ndarray
    init_nyha: np.ndarray  # initial state in {0,1,2,3}


def build_hf_population(params: Dict, n: int, seed: int) -> HFPopulation:
    """
    Calibrated to typical HF clinic populations (e.g. CHARM, PARADIGM-HF).
    All distribution parameters are exposed so that search methods can
    explore alternative case-mix scenarios.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(params.get("age_mean", 68.0), params.get("age_sd", 11.0), n)
    age = np.clip(age, 30.0, 95.0)
    egfr = rng.normal(params.get("egfr_mean", 60.0), params.get("egfr_sd", 22.0), n)
    egfr = np.clip(egfr, 10.0, 120.0)
    ef = rng.normal(params.get("ef_mean", 35.0), params.get("ef_sd", 14.0), n)
    ef = np.clip(ef, 5.0, 70.0)
    diabetes = rng.random(n) < params.get("diabetes_prev", 0.40)

    # Initial NYHA distribution: pi_init = softmax of log-odds vector
    init_logits = np.array(
        [
            params.get("init_log_p_nyha1", 0.4),
            params.get("init_log_p_nyha2", 0.9),
            params.get("init_log_p_nyha3", 0.4),
            params.get("init_log_p_nyha4", -0.6),
        ]
    )
    init_logits = init_logits - init_logits.max()
    p = np.exp(init_logits) / np.exp(init_logits).sum()
    init_nyha = rng.choice(N_NYHA, size=n, p=p)
    return HFPopulation(n, age, egfr, ef, diabetes.astype(np.int8), init_nyha.astype(np.int8))


# ------------------------------------------------------------------
# Hazard functions (continuous-time rates, per year)
# ------------------------------------------------------------------

def _hazards(
    pop: HFPopulation, nyha: np.ndarray, params: Dict
) -> Dict[str, np.ndarray]:
    """
    Returns four arrays of length pop.n containing per-patient annual hazards:
      h_prog, h_reg, h_death, h_hosp_base (pre-excitation).

    Calibrated central estimates (params override defaults):
      Baseline NYHA progression rate per state:
        I->II: 0.12/yr, II->III: 0.18/yr, III->IV: 0.22/yr, IV: 0 (absorbing
        to death only)
      Regression: II->I: 0.10/yr, III->II: 0.08/yr, IV->III: 0.05/yr
      Death by NYHA: I 0.04, II 0.08, III 0.16, IV 0.30 (/yr)
      Hospitalisation by NYHA: I 0.20, II 0.45, III 0.95, IV 1.6 (/yr)

    Covariate multipliers (log-linear):
      Death:  age (per 10 yr) +0.45, eGFR (per 10) -0.20, EF (per 10) -0.15,
              diabetes +0.30
      Hosp:   age (per 10) +0.10, eGFR (per 10) -0.10, EF (per 10) -0.10,
              diabetes +0.25
    """
    n = pop.n
    h_prog = np.zeros(n)
    h_reg = np.zeros(n)
    base_prog = np.array([
        params.get("prog_I_II", 0.12),
        params.get("prog_II_III", 0.18),
        params.get("prog_III_IV", 0.22),
        0.0,
    ])
    base_reg = np.array([
        0.0,
        params.get("reg_II_I", 0.10),
        params.get("reg_III_II", 0.08),
        params.get("reg_IV_III", 0.05),
    ])
    for k in range(N_NYHA):
        mask = nyha == k
        h_prog[mask] = base_prog[k]
        h_reg[mask] = base_reg[k]

    base_death = np.array([
        params.get("death_I", 0.04),
        params.get("death_II", 0.08),
        params.get("death_III", 0.16),
        params.get("death_IV", 0.30),
    ])
    base_hosp = np.array([
        params.get("hosp_I", 0.20),
        params.get("hosp_II", 0.45),
        params.get("hosp_III", 0.95),
        params.get("hosp_IV", 1.60),
    ])
    age10 = (pop.age - 65.0) / 10.0
    egfr10 = (pop.egfr - 60.0) / 10.0
    ef10 = (pop.ef - 35.0) / 10.0

    death_mult = np.exp(
        params.get("death_beta_age", 0.45) * age10
        + params.get("death_beta_egfr", -0.20) * egfr10
        + params.get("death_beta_ef", -0.15) * ef10
        + params.get("death_beta_diabetes", 0.30) * pop.diabetes
    )
    hosp_mult = np.exp(
        params.get("hosp_beta_age", 0.10) * age10
        + params.get("hosp_beta_egfr", -0.10) * egfr10
        + params.get("hosp_beta_ef", -0.10) * ef10
        + params.get("hosp_beta_diabetes", 0.25) * pop.diabetes
    )

    h_death = base_death[nyha] * death_mult
    h_hosp = base_hosp[nyha] * hosp_mult

    return {"h_prog": h_prog, "h_reg": h_reg, "h_death": h_death, "h_hosp": h_hosp}


# ------------------------------------------------------------------
# Standard score: Cox-PH approximation of expected hospitalisation rate
# given current state. We approximate it (without state dynamics or
# excitation) as the linear-predictor expected rate at baseline NYHA, which
# is what a typical risk-prediction model derived from administrative data
# would produce.
# ------------------------------------------------------------------

def standard_hf_score(pop: HFPopulation, params: Dict) -> np.ndarray:
    """
    Expected baseline hospitalisation rate at the patient's INITIAL NYHA
    class, with covariate multipliers, but ignoring state transitions and
    excitation. This mirrors what a Cox PH model trained on a single
    baseline measurement would produce — the standard practice that a
    trajectory-aware analyst would compare against.
    """
    nyha = pop.init_nyha
    h = _hazards(pop, nyha, params)
    return h["h_hosp"]


# ------------------------------------------------------------------
# Trajectory simulation
# ------------------------------------------------------------------

def simulate_hf_trajectory(
    pop: HFPopulation,
    params: Dict,
    T: float,
    k: int,
    n_sims: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Run n_sims independent Monte Carlo trajectories per patient over T years
    in monthly cycles. Returns:
      R: P(>= k HF hospitalisations within T | survival)
      survival: P(alive at T)
      mean_events: expected hospitalisations within T (over survivors)
    """
    rng = np.random.default_rng(seed)
    n = pop.n
    n_steps = int(round(T / DT))
    excit_factor = params.get("hosp_amp_after_event", 2.0)
    excit_decay_months = max(1.0, params.get("hosp_amp_decay_months", 6.0))

    catastrophic = np.zeros(n)
    survival = np.zeros(n)
    total_events = np.zeros(n)

    base_haz_template = {  # cached helper to avoid recompute
        "age10": (pop.age - 65.0) / 10.0,
        "egfr10": (pop.egfr - 60.0) / 10.0,
        "ef10": (pop.ef - 35.0) / 10.0,
        "diab": pop.diabetes.astype(float),
    }

    death_mult_per_patient = np.exp(
        params.get("death_beta_age", 0.45) * base_haz_template["age10"]
        + params.get("death_beta_egfr", -0.20) * base_haz_template["egfr10"]
        + params.get("death_beta_ef", -0.15) * base_haz_template["ef10"]
        + params.get("death_beta_diabetes", 0.30) * base_haz_template["diab"]
    )
    hosp_mult_per_patient = np.exp(
        params.get("hosp_beta_age", 0.10) * base_haz_template["age10"]
        + params.get("hosp_beta_egfr", -0.10) * base_haz_template["egfr10"]
        + params.get("hosp_beta_ef", -0.10) * base_haz_template["ef10"]
        + params.get("hosp_beta_diabetes", 0.25) * base_haz_template["diab"]
    )

    base_prog = np.array([
        params.get("prog_I_II", 0.12),
        params.get("prog_II_III", 0.18),
        params.get("prog_III_IV", 0.22),
        0.0,
    ])
    base_reg = np.array([
        0.0,
        params.get("reg_II_I", 0.10),
        params.get("reg_III_II", 0.08),
        params.get("reg_IV_III", 0.05),
    ])
    base_death = np.array([
        params.get("death_I", 0.04),
        params.get("death_II", 0.08),
        params.get("death_III", 0.16),
        params.get("death_IV", 0.30),
    ])
    base_hosp = np.array([
        params.get("hosp_I", 0.20),
        params.get("hosp_II", 0.45),
        params.get("hosp_III", 0.95),
        params.get("hosp_IV", 1.60),
    ])

    for _ in range(n_sims):
        nyha = pop.init_nyha.copy()
        alive = np.ones(n, dtype=bool)
        events = np.zeros(n, dtype=np.int32)
        amp = np.zeros(n)  # current excitation multiplier above 1
        amp_remaining = np.zeros(n)  # months remaining of excitation
        for _t in range(n_steps):
            alive_idx = np.where(alive)[0]
            if alive_idx.size == 0:
                break
            nyha_a = nyha[alive_idx]

            # Hospitalisation
            h_hosp = (base_hosp[nyha_a] * hosp_mult_per_patient[alive_idx]) * (
                1.0 + amp[alive_idx]
            )
            p_hosp = 1.0 - np.exp(-h_hosp * DT)
            hosp_event = rng.random(alive_idx.size) < p_hosp
            events[alive_idx[hosp_event]] += 1
            # Trigger / refresh excitation on event
            idx_event = alive_idx[hosp_event]
            amp[idx_event] = excit_factor - 1.0  # multiplier above 1
            amp_remaining[idx_event] = excit_decay_months
            # Decay excitation
            amp_remaining[alive_idx] -= 1
            decayed = (amp_remaining < 0) & alive
            amp[decayed] = 0.0
            amp_remaining[decayed] = 0.0

            # Death
            h_death = base_death[nyha_a] * death_mult_per_patient[alive_idx]
            p_death = 1.0 - np.exp(-h_death * DT)
            dies = rng.random(alive_idx.size) < p_death
            died_idx = alive_idx[dies]
            alive[died_idx] = False

            # Progression / regression (survivors only)
            still_alive_mask = ~dies
            sa_idx = alive_idx[still_alive_mask]
            if sa_idx.size:
                nyha_sa = nyha[sa_idx]
                p_prog = 1.0 - np.exp(-base_prog[nyha_sa] * DT)
                p_reg = 1.0 - np.exp(-base_reg[nyha_sa] * DT)
                u1 = rng.random(sa_idx.size)
                u2 = rng.random(sa_idx.size)
                prog_event = (u1 < p_prog) & (nyha_sa < 3)
                reg_event = (u2 < p_reg) & (nyha_sa > 0) & (~prog_event)
                nyha[sa_idx[prog_event]] = nyha_sa[prog_event] + 1
                nyha[sa_idx[reg_event]] = nyha_sa[reg_event] - 1

        catastrophic += events >= k
        survival += alive
        total_events += events

    return {
        "R": catastrophic / n_sims,
        "survival": survival / n_sims,
        "mean_events": total_events / n_sims,
    }


# ------------------------------------------------------------------
# Misordering metric (shared with two-state)
# ------------------------------------------------------------------

def hf_misordering_fraction(
    r: np.ndarray, R: np.ndarray, n_pairs: int = 500_000, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    n = len(r)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    valid = (i != j) & (r[i] != r[j]) & (R[i] != R[j])
    i, j = i[valid], j[valid]
    discordant = ((r[i] > r[j]) & (R[i] < R[j])) | ((r[i] < r[j]) & (R[i] > R[j]))
    return float(discordant.mean()) if len(i) > 0 else 0.0


# ------------------------------------------------------------------
# Calibrated parameters and search bounds
# ------------------------------------------------------------------

HF_CALIBRATED_PARAMS: Dict[str, float] = {
    # Case mix
    "age_mean": 68.0, "age_sd": 11.0,
    "egfr_mean": 60.0, "egfr_sd": 22.0,
    "ef_mean": 35.0, "ef_sd": 14.0,
    "diabetes_prev": 0.40,
    "init_log_p_nyha1": 0.4, "init_log_p_nyha2": 0.9,
    "init_log_p_nyha3": 0.4, "init_log_p_nyha4": -0.6,
    # Transition rates (per year)
    "prog_I_II": 0.12, "prog_II_III": 0.18, "prog_III_IV": 0.22,
    "reg_II_I": 0.10, "reg_III_II": 0.08, "reg_IV_III": 0.05,
    "death_I": 0.04, "death_II": 0.08, "death_III": 0.16, "death_IV": 0.30,
    "hosp_I": 0.20, "hosp_II": 0.45, "hosp_III": 0.95, "hosp_IV": 1.60,
    # Covariate effects
    "death_beta_age": 0.45, "death_beta_egfr": -0.20,
    "death_beta_ef": -0.15, "death_beta_diabetes": 0.30,
    "hosp_beta_age": 0.10, "hosp_beta_egfr": -0.10,
    "hosp_beta_ef": -0.10, "hosp_beta_diabetes": 0.25,
    # Self-excitation
    "hosp_amp_after_event": 2.0,
    "hosp_amp_decay_months": 6.0,
}

# Subset of parameters the search methods are allowed to vary.
HF_PARAM_BOUNDS = {
    "prog_I_II": (0.02, 0.50),
    "prog_II_III": (0.02, 0.60),
    "prog_III_IV": (0.02, 0.80),
    "reg_II_I": (0.0, 0.40),
    "reg_III_II": (0.0, 0.30),
    "reg_IV_III": (0.0, 0.20),
    "death_III": (0.04, 0.50),
    "death_IV": (0.10, 1.00),
    "hosp_II": (0.10, 1.20),
    "hosp_III": (0.20, 2.50),
    "hosp_IV": (0.50, 4.00),
    "hosp_amp_after_event": (1.0, 8.0),
    "hosp_amp_decay_months": (1.0, 24.0),
    "diabetes_prev": (0.10, 0.80),
    "ef_mean": (15.0, 55.0),
    "egfr_mean": (25.0, 90.0),
}


def hf_evaluate(
    params_partial: Dict,
    n: int = 5_000,
    n_sims: int = 200,
    T: float = 3.0,
    k: int = 2,
    seed: int = 42,
) -> Dict:
    """
    Evaluate a partial parameter override against the calibrated defaults.
    Returns delta plus population-level diagnostics.
    """
    params = dict(HF_CALIBRATED_PARAMS)
    params.update(params_partial)
    pop = build_hf_population(params, n=n, seed=seed)
    r = standard_hf_score(pop, params)
    sim = simulate_hf_trajectory(pop, params, T=T, k=k, n_sims=n_sims, seed=seed + 1)
    R = sim["R"]
    delta = hf_misordering_fraction(r, R, seed=seed + 2)
    return {
        "delta": delta,
        "pop_mean_age": float(pop.age.mean()),
        "pop_mean_ef": float(pop.ef.mean()),
        "pop_mean_egfr": float(pop.egfr.mean()),
        "pop_mean_R": float(R.mean()),
        "pop_mean_r": float(r.mean()),
        "pop_survival_T": float(sim["survival"].mean()),
        "pop_mean_events": float(sim["mean_events"].mean()),
        "pop_corr_r_R": float(np.corrcoef(r, R)[0, 1]),
    }


def hf_calibrated_baseline(n: int = 5_000, n_sims: int = 200, T: float = 3.0, k: int = 2,
                           seed: int = 42) -> Dict:
    return hf_evaluate({}, n=n, n_sims=n_sims, T=T, k=k, seed=seed)
