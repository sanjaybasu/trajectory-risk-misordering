"""Figure 3: Lasso coefficient bars for both models."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_results import mechanism_regression


def plot_mechanism(top_n: int = 10) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in [
        (axes[0], "two_state", "Two-state self-exciting"),
        (axes[1], "nyha", "NYHA I-IV + Death HF"),
    ]:
        info = mechanism_regression(model)
        if not info:
            ax.text(0.5, 0.5, "Pending", ha="center", va="center")
            ax.set_title(title)
            continue
        coefs = info["coefficients_sorted_by_abs"][:top_n]
        names = [c[0] for c in coefs][::-1]
        values = [c[1] for c in coefs][::-1]
        colors = ["#2266aa" if v > 0 else "#aa3322" for v in values]
        ys = np.arange(len(names))
        ax.barh(ys, values, color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Standardised Lasso coefficient")
        ax.set_title(f"{title}\nR² = {info['r2']:.3f}  (n = {info['n_observations']})")
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "figure3_mechanism.pdf")
    plt.savefig(out / "figure3_mechanism.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_mechanism()
    print("saved figure3_mechanism.{pdf,png}")
