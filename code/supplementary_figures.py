"""eFigure 1 (per-seed traces) and eFigure 2 (distribution boxplots)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def _load(model: str) -> Dict:
    return json.load(open(RESULTS / f"results_{model}.json"))


def _load_agent(model: str) -> Dict | None:
    p = RESULTS / f"agent_{model}.json"
    return json.load(open(p)) if p.exists() else None


def per_seed_traces() -> None:
    available = [m for m in ("two_state", "nyha") if (RESULTS / f"results_{m}.json").exists()]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, available):
        rec = _load(model)
        ag = _load_agent(model)
        baseline = rec["calibrated_baseline"]["delta"]
        method_color = {"random": "tab:blue", "bo": "tab:orange", "cma": "tab:green"}
        for method, color in method_color.items():
            for key, m in rec["methods"].items():
                if not key.startswith(f"{method}_extended_"):
                    continue
                hist = m.get("history", [])
                if not hist:
                    continue
                deltas = np.maximum.accumulate([h["delta"] for h in hist])
                ax.plot(
                    np.arange(1, len(deltas) + 1), deltas,
                    color=color, lw=1.0, alpha=0.6,
                    label=method.upper() if key.endswith("_seed1") else None,
                )
        if ag is not None and ag.get("history"):
            deltas = np.maximum.accumulate([h["delta"] for h in ag["history"]])
            ax.plot(
                np.arange(1, len(deltas) + 1), deltas,
                color="black", ls="--", lw=2.0, label="LLM agents",
            )
        ax.axhline(baseline, color="gray", lw=0.8, ls=":", label="Calibrated")
        ax.set_title("Two-state" if model == "two_state" else "NYHA HF")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best Δ so far")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "efigure1_per_seed_traces.pdf")
    plt.savefig(FIGS / "efigure1_per_seed_traces.png", dpi=200)
    plt.close()


def distribution_boxplots() -> None:
    available = [m for m in ("two_state", "nyha") if (RESULTS / f"results_{m}.json").exists()]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, available):
        rec = _load(model)
        data: List[np.ndarray] = []
        labels: List[str] = []
        for method, label in [("random", "Random"), ("bo", "BO"), ("cma", "CMA-ES")]:
            pooled = []
            for key, m in rec["methods"].items():
                if key.startswith(f"{method}_extended_"):
                    pooled.extend(h["delta"] for h in m.get("history", []))
            if pooled:
                data.append(np.array(pooled))
                labels.append(label)
        if data:
            ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True)
        ax.set_ylabel("Δ across 60 evaluations (pooled across seeds)")
        ax.set_title("Two-state" if model == "two_state" else "NYHA HF")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "efigure2_delta_distribution.pdf")
    plt.savefig(FIGS / "efigure2_delta_distribution.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    per_seed_traces()
    print("saved efigure1_per_seed_traces.{pdf,png}")
    distribution_boxplots()
    print("saved efigure2_delta_distribution.{pdf,png}")
