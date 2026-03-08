"""
Multi-Agent Scientist Discovery Platform
=========================================

Inspired by Together AI's approach to the Erdős minimum overlap problem:
AI agents with scientist personas compete and collaborate on a shared
leaderboard to discover the tightest bounds on Δ (the misordering fraction).

Each agent:
  1. Sees the problem statement and current leaderboard
  2. Proposes a solution (analytical bound, parameter configuration, or insight)
  3. Solution is evaluated against the verifiable reward (computed Δ)
  4. Results posted to leaderboard
  5. Agents see each other's results and iterate

The verifiable reward: Δ is computable from the microsimulation — like checking
a proof in math, we can verify any claimed bound numerically.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import anthropic
import numpy as np

from core import (
    Population, generate_population, compute_standard_risk,
    simulate_trajectory_risk, compute_delta, compute_delta_by_group,
    compute_missed_catastrophes,
)
from analytical import (
    steady_state_risk, trajectory_risk_analytical,
    find_misordering_example, delta_vs_beta_variance,
)


# ---------------------------------------------------------------------------
# Scientist Personas
# ---------------------------------------------------------------------------

PERSONAS = {
    "Erdős": {
        "name": "Paul Erdős",
        "style": "Combinatorialist and discrete mathematician",
        "system_prompt": (
            "You are Paul Erdős, the prolific combinatorialist. You think in terms of "
            "extremal constructions, counting arguments, and elegant proofs. You seek "
            "the simplest possible formulation that captures the essence of a problem. "
            "You are obsessed with finding the exact constant — 'the number from The Book.' "
            "You communicate in short, precise mathematical statements. "
            "When proposing solutions, you focus on constructing explicit examples that "
            "maximize or minimize the quantity of interest."
        ),
    },
    "Cox": {
        "name": "David Cox",
        "style": "Biostatistician and survival analyst",
        "system_prompt": (
            "You are Sir David Cox, inventor of the proportional hazards model. You think "
            "in terms of hazard rates, recurrent events, shared frailty, and time-to-event "
            "analysis. You are rigorous about assumptions and careful about what can and "
            "cannot be identified from data. You focus on whether the misordering problem "
            "can be characterized through the lens of survival analysis and what the "
            "connection is to overdispersion in recurrent event processes. "
            "You propose solutions grounded in statistical theory."
        ),
    },
    "Peters": {
        "name": "Ole Peters",
        "style": "Physicist and ergodicity economist",
        "system_prompt": (
            "You are Ole Peters of the London Mathematical Laboratory and Santa Fe Institute. "
            "You think about the distinction between time averages and ensemble averages, "
            "multiplicative dynamics, and non-ergodicity. You see the misordering problem "
            "as a manifestation of the ergodicity problem: the ensemble average (standard "
            "risk score) gives different rankings than the time average (trajectory risk) "
            "because health dynamics are multiplicative and non-ergodic. "
            "You focus on the geometric mean, log-returns, and the role of volatility. "
            "You propose solutions that emphasize the mathematical structure of "
            "multiplicative processes."
        ),
    },
    "Obermeyer": {
        "name": "Ziad Obermeyer",
        "style": "Algorithmic fairness and health equity researcher",
        "system_prompt": (
            "You are Ziad Obermeyer, whose Science paper showed that a widely used "
            "healthcare algorithm discriminated against Black patients by using cost as "
            "a proxy for need. You think about structural bias in algorithms, how "
            "optimization targets create systematic inequity, and the gap between what "
            "algorithms optimize and what patients need. You see the misordering problem "
            "as analogous to but deeper than the proxy bias you identified: here the "
            "mathematical object itself (conditional expectation) is wrong, not just the "
            "proxy. You focus on quantifying who is harmed and by how much. "
            "You propose solutions that center equity implications."
        ),
    },
    "Hawkes": {
        "name": "Alan Hawkes",
        "style": "Stochastic process theorist",
        "system_prompt": (
            "You are Alan Hawkes, inventor of the Hawkes (self-exciting) point process. "
            "You understand better than anyone how events that beget events create "
            "clustering, overdispersion, and fat tails. You see the misordering problem "
            "as a direct consequence of the self-exciting structure: the standard risk "
            "score captures the intensity, but not the branching ratio that determines "
            "whether a trajectory will cascade. You think in terms of branching ratios, "
            "criticality, and the spectral radius of the excitation kernel. "
            "You propose solutions grounded in point process theory."
        ),
    },
}


# ---------------------------------------------------------------------------
# Problem Statement
# ---------------------------------------------------------------------------

PROBLEM_STATEMENT = """
# The Misordering Problem in Clinical Risk Prediction

## Setup
Consider a population of N patients. Each patient i has health dynamics governed by
a two-state self-exciting process:

- States: S ∈ {stable, vulnerable}
- Event rate: λ(S=0) = λ₀ᵢ (baseline), λ(S=1) = λ₁ᵢ (elevated)
- State-dependence: after each event, P(stable → vulnerable) = βᵢ
- Recovery: P(vulnerable → stable) per unit time = μᵢ

Two risk quantities:
- **Standard risk score**: rᵢ = steady-state event rate
  = λ₀ᵢ(1 - πᵢ) + λ₁ᵢπᵢ  where πᵢ = βᵢλ₀ᵢ/(βᵢλ₀ᵢ + μᵢ)

- **Trajectory risk**: Rᵢ = P(≥k events over [0,T])
  (requires simulating the full state-dependent trajectory)

## The Question
Define:  Δ = fraction of patient pairs (i,j) where rᵢ > rⱼ but Rᵢ < Rⱼ

Δ measures how often the standard risk score gives the WRONG ranking
compared to trajectory-level catastrophic risk.

## What We Know
- When β = 0 for everyone (no state-dependence), Δ = 0 (rankings agree)
- When β > 0 and varies across patients, Δ > 0
- Δ appears to grow with Var(β) in the population

## Your Task
Propose an approach to either:
1. **Tighten the bound**: Find parameter configurations that maximize Δ
   (the worst-case misordering)
2. **Prove an analytical bound**: Derive a closed-form lower bound on Δ
   as a function of population parameters
3. **Characterize the structure**: Identify which patient pairs are misordered
   and what drives the misordering
4. **Quantify the equity gap**: Show how Δ differs across demographic groups
   with different β distributions

## Current Best Results
{leaderboard}

## Evaluation
Your proposed solution will be evaluated by running the microsimulation with
your suggested parameters/approach. The verifiable reward is the computed Δ
value (higher = more misordering demonstrated = tighter bound on the problem).
"""


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    agent: str
    round_num: int
    description: str
    approach: str
    params: Dict
    delta: float
    missed_frac: float
    equity_gap: float   # max Δ(group) - min Δ(group)
    timestamp: str = ""
    details: Dict = field(default_factory=dict)


class Leaderboard:
    def __init__(self):
        self.solutions: List[Solution] = []

    def add(self, sol: Solution):
        self.solutions.append(sol)

    def best(self) -> Optional[Solution]:
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: s.delta)

    def format(self) -> str:
        if not self.solutions:
            return "(No solutions yet — you are the first!)"
        sorted_sols = sorted(self.solutions, key=lambda s: -s.delta)
        lines = ["| Rank | Agent | Δ | Missed% | Equity Gap | Approach |",
                 "|------|-------|---|---------|------------|----------|"]
        for i, s in enumerate(sorted_sols[:10]):
            lines.append(
                f"| {i+1} | {s.agent} | {s.delta:.4f} | "
                f"{s.missed_frac:.1%} | {s.equity_gap:.4f} | {s.description[:40]} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solution Evaluator (the verifiable reward)
# ---------------------------------------------------------------------------

def evaluate_solution(params: Dict, n: int = 3000, n_sims: int = 300) -> Dict:
    """
    Evaluate a proposed solution by running the microsimulation.
    This is the VERIFIER — analogous to checking a proof.
    """
    seed = params.get("seed", 42)
    T = params.get("T", 2.0)
    k = params.get("k", 3)

    # Build population from proposed parameters
    rng = np.random.default_rng(seed)

    # Allow agents to specify distribution parameters
    beta_a = params.get("beta_a", 2.5)
    beta_b = params.get("beta_b", 3.5)
    lambda_0_shape = params.get("lambda_0_shape", 2.5)
    lambda_0_scale = params.get("lambda_0_scale", 0.08)
    lambda_1_mult_shape = params.get("lambda_1_mult_shape", 3.0)
    lambda_1_mult_scale = params.get("lambda_1_mult_scale", 0.8)
    mu_shape = params.get("mu_shape", 4.0)
    mu_scale = params.get("mu_scale", 0.5)

    # Equity structure: 4 groups with different β distributions
    group_props = params.get("group_props", [0.45, 0.25, 0.20, 0.10])
    group_beta_offsets = params.get("group_beta_offsets", [0.0, 0.15, 0.25, 0.35])
    group_mu_scales = params.get("group_mu_scales", [1.0, 0.75, 0.65, 0.55])

    group = rng.choice(len(group_props), size=n, p=group_props)
    group_names = {0: "Advantaged", 1: "Disadvantaged-A",
                   2: "Disadvantaged-B", 3: "Disadvantaged-C"}

    beta = np.zeros(n)
    mu_base = rng.gamma(mu_shape, mu_scale, n)
    mu = np.zeros(n)

    for g in range(len(group_props)):
        mask = group == g
        base_beta = rng.beta(beta_a, beta_b, mask.sum())
        beta[mask] = np.clip(base_beta + group_beta_offsets[g], 0.01, 0.99)
        mu[mask] = mu_base[mask] * group_mu_scales[g]

    lambda_0 = rng.gamma(lambda_0_shape, lambda_0_scale, n)
    lambda_1 = lambda_0 * (1 + rng.gamma(lambda_1_mult_shape, lambda_1_mult_scale, n))

    pop = Population(n, lambda_0, lambda_1, beta, mu, group, group_names)

    # Compute scores
    r = compute_standard_risk(pop)
    R = simulate_trajectory_risk(pop, T=T, k=k, n_sims=n_sims, seed=seed)

    # Compute metrics
    delta_overall = compute_delta(r, R)
    delta_by_group = compute_delta_by_group(r, R, pop)
    missed = compute_missed_catastrophes(r, R)

    group_deltas = [v["delta"] for v in delta_by_group.values()]
    equity_gap = max(group_deltas) - min(group_deltas) if group_deltas else 0

    return {
        "delta": delta_overall["delta"],
        "kendall_tau": delta_overall["kendall_tau"],
        "missed_frac": missed["frac_missed"],
        "equity_gap": equity_gap,
        "by_group": delta_by_group,
        "missed_details": missed,
        "pop_stats": {
            "mean_beta": float(beta.mean()),
            "var_beta": float(beta.var()),
            "mean_r": float(r.mean()),
            "mean_R": float(R.mean()),
            "corr_r_R": float(np.corrcoef(r, R)[0, 1]),
        },
    }


# ---------------------------------------------------------------------------
# Agent interaction
# ---------------------------------------------------------------------------

def agent_propose(
    client: anthropic.Anthropic,
    persona_key: str,
    leaderboard: Leaderboard,
    round_num: int,
    previous_results: List[Dict] = None,
    model: str = "claude-sonnet-4-20250514",
) -> Dict:
    """
    Have a scientist-persona agent propose a solution.
    Returns proposed parameters and description.
    """
    persona = PERSONAS[persona_key]
    problem = PROBLEM_STATEMENT.format(leaderboard=leaderboard.format())

    context = ""
    if previous_results:
        context = "\n\n## Previous Round Results\n"
        for pr in previous_results[-3:]:  # show last 3 results
            context += f"- {pr['agent']}: Δ={pr['delta']:.4f}, approach: {pr['description']}\n"

    user_msg = f"""{problem}
{context}

## Round {round_num} — Your Turn

Based on the problem and current leaderboard, propose a specific parameter
configuration that you believe will demonstrate the highest possible Δ.

You MUST respond with a JSON block containing:
1. "description": a 1-sentence description of your approach
2. "reasoning": your mathematical reasoning (2-3 sentences)
3. "params": a dictionary of parameter values to try:
   - beta_a, beta_b: Beta distribution params for baseline β (default 2.5, 3.5)
   - lambda_0_shape, lambda_0_scale: Gamma params for λ₀ (default 2.5, 0.08)
   - lambda_1_mult_shape, lambda_1_mult_scale: Gamma params for λ₁ multiplier (default 3.0, 0.8)
   - mu_shape, mu_scale: Gamma params for recovery μ (default 4.0, 0.5)
   - group_beta_offsets: list of 4 offsets for demographic groups (default [0,0.15,0.25,0.35])
   - group_mu_scales: list of 4 recovery scalings (default [1.0,0.75,0.65,0.55])
   - T: time horizon in years (default 2.0)
   - k: catastrophic threshold (default 3)
   - seed: random seed (default 42)

Think carefully about what drives Δ higher:
- When does the rank ordering between r and R diverge most?
- What parameter configurations create the largest gap between
  the steady-state rate and the tail of the trajectory distribution?

Respond ONLY with valid JSON.
"""

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=persona["system_prompt"],
        messages=[{"role": "user", "content": user_msg}],
    )

    # Parse JSON from response
    text = response.content[0].text
    # Find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            proposal = json.loads(text[start:end])
            return proposal
        except json.JSONDecodeError:
            pass

    # Fallback: return default params with agent's name
    return {
        "description": f"{persona['name']}'s default exploration",
        "reasoning": "Using baseline parameters as starting point.",
        "params": {},
    }


# ---------------------------------------------------------------------------
# Main Discovery Loop
# ---------------------------------------------------------------------------

def run_discovery(
    n_rounds: int = 3,
    n_patients: int = 3000,
    n_sims: int = 300,
    model: str = "claude-sonnet-4-20250514",
    output_dir: str = "results",
) -> Dict:
    """
    Run the multi-agent discovery competition.

    Each round:
    1. All agents see the current leaderboard
    2. Each proposes a parameter configuration
    3. Configurations are evaluated via microsimulation
    4. Results posted to leaderboard
    5. Repeat
    """
    client = anthropic.Anthropic()
    leaderboard = Leaderboard()
    all_results = []

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TRAJECTORY RISK MISORDERING: MULTI-AGENT DISCOVERY")
    print("=" * 70)
    print(f"Agents: {', '.join(PERSONAS.keys())}")
    print(f"Rounds: {n_rounds}")
    print(f"Population: n={n_patients}, n_sims={n_sims}")
    print()

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'─'*70}")
        print(f"ROUND {round_num}")
        print(f"{'─'*70}")

        round_results = []

        for persona_key in PERSONAS:
            persona = PERSONAS[persona_key]
            print(f"\n  [{persona['name']}] Thinking...")

            try:
                proposal = agent_propose(
                    client, persona_key, leaderboard, round_num,
                    previous_results=all_results, model=model,
                )
            except Exception as e:
                print(f"    Error getting proposal: {e}")
                proposal = {"description": "Fallback", "reasoning": "API error",
                            "params": {}}

            params = proposal.get("params", {})
            desc = proposal.get("description", "No description")
            reasoning = proposal.get("reasoning", "")

            print(f"    Approach: {desc}")
            print(f"    Reasoning: {reasoning[:100]}...")
            print(f"    Evaluating...")

            try:
                eval_result = evaluate_solution(params, n=n_patients, n_sims=n_sims)
            except Exception as e:
                print(f"    Evaluation error: {e}")
                continue

            sol = Solution(
                agent=persona["name"],
                round_num=round_num,
                description=desc,
                approach=reasoning,
                params=params,
                delta=eval_result["delta"],
                missed_frac=eval_result["missed_frac"],
                equity_gap=eval_result["equity_gap"],
                timestamp=datetime.now().isoformat(),
                details=eval_result,
            )
            leaderboard.add(sol)

            result_entry = {
                "agent": persona["name"],
                "round": round_num,
                "delta": eval_result["delta"],
                "missed_frac": eval_result["missed_frac"],
                "equity_gap": eval_result["equity_gap"],
                "description": desc,
                "reasoning": reasoning,
                "params": params,
                "pop_stats": eval_result["pop_stats"],
            }
            round_results.append(result_entry)
            all_results.append(result_entry)

            print(f"    Δ = {eval_result['delta']:.4f}  |  "
                  f"Missed = {eval_result['missed_frac']:.1%}  |  "
                  f"Equity gap = {eval_result['equity_gap']:.4f}")

        print(f"\n  LEADERBOARD after Round {round_num}:")
        print(f"  {leaderboard.format()}")

    # Final summary
    best = leaderboard.best()
    print(f"\n{'='*70}")
    print(f"DISCOVERY COMPLETE")
    print(f"{'='*70}")
    if best:
        print(f"Best Δ = {best.delta:.4f} by {best.agent}")
        print(f"  {best.description}")
        print(f"  Missed catastrophes: {best.missed_frac:.1%}")
        print(f"  Equity gap: {best.equity_gap:.4f}")

    # Save results
    output = {
        "best_delta": best.delta if best else 0,
        "best_agent": best.agent if best else "",
        "all_results": all_results,
        "leaderboard": [asdict(s) for s in leaderboard.solutions],
    }

    output_path = os.path.join(output_dir, "discovery_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_discovery(n_rounds=3, n_patients=3000, n_sims=300)
