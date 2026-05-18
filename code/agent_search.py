"""
LLM-agent search wrapped behind the same evaluator API as random/BO/CMA-ES.

Five Claude Opus 4.7 personas propose JSON parameter configurations. Each
configuration is evaluated by the verifiable simulator. After every round,
agents see the cumulative leaderboard and prior rationales.

This replaces the original five-persona prompts with simpler, model-agnostic
phrasing so the search can be applied to multiple underlying simulators
(two-state self-exciting; NYHA multi-state HF).
"""
from __future__ import annotations
import json
import os
import re
import time
from typing import Callable, Dict, List, Tuple

import anthropic

ANTHROPIC_MODEL = "claude-opus-4-7"

PERSONAS = {
    "combinatorial_extremiser": (
        "You are a combinatorial extremiser. You search for parameter "
        "configurations that maximise a target metric by reasoning about "
        "monotone relationships, boundary cases, and counting arguments. "
        "Prefer simple, explainable proposals that test a single mechanism."
    ),
    "survival_statistician": (
        "You are a survival analyst with deep expertise in recurrent-event "
        "processes, frailty models, and overdispersion. Reason in terms of "
        "hazard ratios, censoring, and the difference between ensemble and "
        "subject-specific quantities."
    ),
    "ergodicity_physicist": (
        "You are a statistical physicist working on ergodicity and "
        "time-ensemble divergence. Reason about absorbing states, "
        "multiplicative dynamics, and when long-run averages fail to "
        "represent individual trajectories."
    ),
    "stochastic_process_theorist": (
        "You are a stochastic-process theorist who studies self-exciting "
        "processes and branching ratios. Reason about criticality, "
        "near-critical regimes, and the spectral radius of excitation kernels."
    ),
    "algorithmic_fairness_researcher": (
        "You are an algorithmic-fairness researcher. Reason about subgroup "
        "heterogeneity, when between-group variation dominates within-group "
        "variation, and when a single score fails differently for different "
        "subpopulations."
    ),
}


def _build_user_message(
    problem_description: str,
    param_schema: str,
    bounds_description: str,
    history: List[Dict],
    persona_name: str,
    round_num: int,
) -> str:
    leaderboard = "(no proposals yet)"
    if history:
        sorted_h = sorted(history, key=lambda h: -h["delta"])[:10]
        rows = [
            f"  {i+1}. delta={h['delta']:.4f}  agent={h['agent']}  "
            f"rationale={h.get('rationale','')[:80]}"
            for i, h in enumerate(sorted_h)
        ]
        leaderboard = "\n".join(rows)

    return (
        f"# Problem\n{problem_description}\n\n"
        f"# Parameter schema (continuous, bounded)\n{param_schema}\n\n"
        f"# Bounds\n{bounds_description}\n\n"
        f"# Current top configurations (cumulative across agents and rounds)\n"
        f"{leaderboard}\n\n"
        f"# Your role this round (round {round_num})\n"
        f"You are the {persona_name}. Propose ONE new parameter configuration "
        f"you predict will maximise the misordering fraction Delta. Reason "
        f"about the mechanism in 2-4 sentences, then output a JSON object "
        f"with exactly two top-level keys: 'rationale' (string) and 'params' "
        f"(dict matching the schema). Use only the parameter names in the "
        f"schema. Output ONLY the JSON object; no prose around it."
    )


def _extract_json(text: str) -> Dict:
    """Robust JSON extraction from a model response."""
    # Try outermost {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object found in response")
    return json.loads(m.group(0))


def _clip_to_bounds(
    params: Dict, bounds: Dict[str, Tuple[float, float]], default: Dict
) -> Dict:
    """Default to the midpoint of each bound if not supplied or invalid."""
    clipped = {}
    for name, (lo, hi) in bounds.items():
        midpoint = (lo + hi) / 2.0
        v = params.get(name, default.get(name, midpoint))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = default.get(name, midpoint)
        clipped[name] = max(lo, min(hi, v))
    return clipped


def agent_search(
    n_rounds: int,
    n_agents_per_round: int,
    evaluator: Callable[[Dict], Dict],
    bounds: Dict[str, Tuple[float, float]],
    default_params: Dict,
    problem_description: str,
    param_schema: str,
    bounds_description: str,
    model: str = ANTHROPIC_MODEL,
    seed: int = 0,
) -> List[Dict]:
    client = anthropic.Anthropic()
    persona_keys = list(PERSONAS.keys())[:n_agents_per_round]

    history: List[Dict] = []
    for round_num in range(1, n_rounds + 1):
        for persona_name in persona_keys:
            user_msg = _build_user_message(
                problem_description=problem_description,
                param_schema=param_schema,
                bounds_description=bounds_description,
                history=history,
                persona_name=persona_name,
                round_num=round_num,
            )
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=1200,
                    system=PERSONAS[persona_name],
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = resp.content[0].text
                parsed = _extract_json(text)
                rationale = parsed.get("rationale", "")
                proposed = parsed.get("params", {})
            except Exception as exc:
                rationale = f"(API/parse error: {exc})"
                proposed = {}

            params = _clip_to_bounds(proposed, bounds, default_params)
            t_eval = time.time()
            result = evaluator(params)
            history.append({
                "iteration": len(history),
                "round": round_num,
                "agent": persona_name,
                "params": params,
                "rationale": rationale,
                "delta": result["delta"],
                "pop_stats": {k: v for k, v in result.items() if k != "delta"},
                "wall_seconds_api": t_eval - t0,
                "wall_seconds_eval": time.time() - t_eval,
            })
    return history
