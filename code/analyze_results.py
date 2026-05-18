"""
Analyse full-experiment results:
  - Per-method best-Δ at matched (15) and extended (60) budgets.
  - Cumulative-best trace for each method (mean ± SD across seeds).
  - Post-hoc sparse regression: which parameter moves Δ?
  - Agent-rationale vs. post-hoc-regression concordance.

Produces:
  results/summary_table.csv
  results/mechanism_table.csv
  figures/figure1_best_delta.pdf|png
  figures/figure2_best_trace.pdf|png
  figures/figure3_mechanism.pdf|png
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def _load_method_results(model_name: str) -> Dict:
    path = RESULTS / f"results_{model_name}.json"
    with open(path) as f:
        return json.load(f)


def _load_agent(model_name: str) -> Dict | None:
    path = RESULTS / f"agent_{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def cumulative_best(history: List[Dict]) -> np.ndarray:
    deltas = np.array([h["delta"] for h in history], dtype=float)
    return np.maximum.accumulate(deltas)


def summarise_method(
    method_prefix: str,
    record: Dict,
    seeds: List[int],
) -> Tuple[float, float, np.ndarray]:
    bests = []
    traces = []
    for seed in seeds:
        key = f"{method_prefix}_seed{seed}"
        if key not in record["methods"]:
            continue
        m = record["methods"][key]
        if "history" not in m or not m["history"]:
            continue
        trace = cumulative_best(m["history"])
        traces.append(trace)
        bests.append(trace[-1])
    if not bests:
        return float("nan"), float("nan"), np.array([])
    bests_arr = np.array(bests)
    return float(bests_arr.mean()), float(bests_arr.std(ddof=0)), np.stack(traces)


def build_summary_table(models=("two_state", "nyha"), seeds=(1, 2, 3)) -> List[Dict]:
    rows: List[Dict] = []
    seeds = list(seeds)
    for model in models:
        path = RESULTS / f"results_{model}.json"
        if not path.exists():
            continue
        record = _load_method_results(model)
        baseline = record["calibrated_baseline"]["delta"]
        rows.append({
            "model": model,
            "method": "calibrated_baseline",
            "budget": 0,
            "best_delta_mean": baseline,
            "best_delta_sd": 0.0,
            "n_runs": 1,
        })
        for method_prefix, label, budget in [
            ("random_matched", "random_matched", 15),
            ("bo_matched", "bo_matched", 15),
            ("cma_matched", "cma_matched", 15),
            ("random_extended", "random_extended", 60),
            ("bo_extended", "bo_extended", 60),
            ("cma_extended", "cma_extended", 60),
        ]:
            m, s, _ = summarise_method(method_prefix, record, seeds)
            rows.append({
                "model": model,
                "method": label,
                "budget": budget,
                "best_delta_mean": m,
                "best_delta_sd": s,
                "n_runs": sum(1 for seed in seeds
                              if f"{method_prefix}_seed{seed}" in record["methods"]),
            })
        # Agent
        ag = _load_agent(model)
        if ag is not None and ag.get("history"):
            best = max(h["delta"] for h in ag["history"])
            rows.append({
                "model": model, "method": "agent", "budget": 15,
                "best_delta_mean": best, "best_delta_sd": 0.0, "n_runs": 1,
            })
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def best_trace_figure(models=("two_state", "nyha"), seeds=(1, 2, 3)) -> None:
    import matplotlib.pyplot as plt

    available = [m for m in models if (RESULTS / f"results_{m}.json").exists()]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, available):
        record = _load_method_results(model)
        ag = _load_agent(model)
        baseline = record["calibrated_baseline"]["delta"]

        method_traces: Dict[str, np.ndarray] = {}
        for prefix, label in [
            ("random_extended", "Random (60)"),
            ("bo_extended", "BO (60)"),
            ("cma_extended", "CMA-ES (60)"),
        ]:
            _, _, traces = summarise_method(prefix, record, list(seeds))
            if traces.size == 0:
                continue
            method_traces[label] = traces

        for label, traces in method_traces.items():
            mean = traces.mean(axis=0)
            sd = traces.std(axis=0, ddof=0)
            xs = np.arange(1, len(mean) + 1)
            ax.plot(xs, mean, label=label, lw=1.6)
            ax.fill_between(xs, mean - sd, mean + sd, alpha=0.15)

        if ag is not None and ag.get("history"):
            ag_trace = cumulative_best(ag["history"])
            ax.plot(
                np.arange(1, len(ag_trace) + 1), ag_trace,
                label="LLM agents (15)", lw=2.0, ls="--", color="black",
            )

        ax.axhline(baseline, color="gray", lw=1.0, ls=":", label="Calibrated baseline")
        ax.set_title("Two-state self-exciting" if model == "two_state"
                     else "NYHA I-IV + Death HF")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best Δ so far")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "figure2_best_trace.pdf")
    plt.savefig(FIGS / "figure2_best_trace.png", dpi=200)
    plt.close()


def best_delta_bar(models=("two_state", "nyha"), seeds=(1, 2, 3)) -> None:
    import matplotlib.pyplot as plt

    methods = ["random_matched", "bo_matched", "cma_matched", "agent",
               "random_extended", "bo_extended", "cma_extended"]
    method_labels = ["Random\n(15)", "BO\n(15)", "CMA-ES\n(15)", "LLM\nagents (15)",
                     "Random\n(60)", "BO\n(60)", "CMA-ES\n(60)"]

    available = [m for m in models if (RESULTS / f"results_{m}.json").exists()]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, available):
        record = _load_method_results(model)
        ag = _load_agent(model)
        baseline = record["calibrated_baseline"]["delta"]

        means: List[float] = []
        sds: List[float] = []
        for prefix in methods:
            if prefix == "agent":
                if ag is None or not ag.get("history"):
                    means.append(float("nan")); sds.append(0.0)
                else:
                    means.append(max(h["delta"] for h in ag["history"]))
                    sds.append(0.0)
            else:
                m, s, _ = summarise_method(prefix, record, list(seeds))
                means.append(m); sds.append(s)

        xs = np.arange(len(methods))
        ax.bar(xs, means, yerr=sds, capsize=4,
               color=["#888888", "#3366cc", "#cc6633", "#000000",
                      "#888888", "#3366cc", "#cc6633"])
        ax.axhline(baseline, color="gray", lw=1.0, ls=":", label="Calibrated")
        ax.set_xticks(xs)
        ax.set_xticklabels(method_labels, rotation=0, fontsize=8)
        ax.set_ylabel("Best Δ achieved")
        ax.set_title("Two-state" if model == "two_state" else "NYHA HF")
        ax.set_ylim(0, max(0.6, max([m for m in means if not np.isnan(m)] + [baseline]) * 1.1))
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "figure1_best_delta.pdf")
    plt.savefig(FIGS / "figure1_best_delta.png", dpi=200)
    plt.close()


def pool_evaluations(model: str, seeds=(1, 2, 3)) -> Tuple[List[Dict], List[str]]:
    """Pool every (params, delta) pair from every method on a single model."""
    path = RESULTS / f"results_{model}.json"
    if not path.exists():
        return [], []
    record = _load_method_results(model)
    ag = _load_agent(model)
    seeds = list(seeds)
    rows: List[Dict] = []
    for key, m in record["methods"].items():
        for h in m.get("history", []):
            rows.append({"params": h["params"], "delta": h["delta"], "method": key})
    if ag is not None:
        for h in ag.get("history", []):
            rows.append({"params": h["params"], "delta": h["delta"], "method": "agent"})
    if not rows:
        return rows, []
    param_names = sorted({k for r in rows for k in r["params"].keys()})
    return rows, param_names


def mechanism_regression(model: str, seeds=(1, 2, 3)) -> Dict:
    """Fit a Lasso of Δ on standardised parameters across all evaluations."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    rows, names = pool_evaluations(model, seeds)
    if not rows:
        return {}
    X = np.array([[r["params"].get(n, np.nan) for n in names] for r in rows])
    y = np.array([r["delta"] for r in rows])
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    if X.shape[0] < 10:
        return {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=0, max_iter=20000).fit(Xs, y)
    coefs = sorted(
        zip(names, lasso.coef_.tolist()),
        key=lambda kv: -abs(kv[1]),
    )
    return {
        "model": model,
        "n_observations": int(X.shape[0]),
        "alpha_chosen": float(lasso.alpha_),
        "r2": float(lasso.score(Xs, y)),
        "coefficients_sorted_by_abs": coefs,
        "intercept": float(lasso.intercept_),
    }


def main():
    rows = build_summary_table()
    write_csv(rows, RESULTS / "summary_table.csv")
    best_delta_bar()
    best_trace_figure()

    mech = {}
    for model in ("two_state", "nyha"):
        mech[model] = mechanism_regression(model)
    with open(RESULTS / "mechanism_regression.json", "w") as f:
        json.dump(mech, f, indent=2, default=float)

    # Plain text summary for quick eyeballing
    print("\n=== SUMMARY ===")
    for r in rows:
        print(
            f"  {r['model']:>9s} | {r['method']:>22s} | budget={r['budget']:3d} | "
            f"mean Δ = {r['best_delta_mean']:.4f} ± {r['best_delta_sd']:.4f} "
            f"(n={r['n_runs']})"
        )
    print("\nMechanism regression (top coefficients):")
    for model, info in mech.items():
        if not info:
            continue
        print(f"  {model}: R² = {info['r2']:.3f}, top 5 |coef|:")
        for n, c in info["coefficients_sorted_by_abs"][:5]:
            print(f"    {n:>30s}  coef = {c:+.4f}")


if __name__ == "__main__":
    main()
