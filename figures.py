#!/usr/bin/env python3
"""
Generate Figures for Medical Decision Making Manuscript.

Figure 1: AI agent discovery progression with clinically calibrated baseline
Figure 2: r vs R scatter plot colored by beta (mechanism visualization)
eFigure 1: NB fit quality (predicted vs simulated event count distributions)
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import compute_standard_risk, simulate_trajectory_risk
from revised_analysis import generate_calibrated_population, assess_nb_fit_quality


def figure1():
    """Discovery progression with clinically calibrated baseline.

    Bar chart: Delta across clinically calibrated and agent-discovered regimes.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = [
        'Clinically\ncalibrated',
        'High\nacuity',
        'R1:\nCombinatorics',
        'R1: Survival\nanalysis',
        'R1:\nErgodicity',
        'R2: Near-\nabsorbing',
        'Validated\nworst-case',
    ]
    deltas = [0.035, 0.061, 0.222, 0.044, 0.119, 0.461, 0.312]

    # Grayscale-compatible palette: calibrated=medium gray, agents=dark,
    # negative result=light, implausible=hatched
    colors = ['#5B9BD5', '#5B9BD5', '#ED7D31', '#A5A5A5', '#ED7D31',
              '#FF4444', '#C00000']

    bars = ax.bar(range(len(labels)), deltas, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Misordering Fraction ($\\Delta$)', fontsize=10)

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.15, lw=0.6)

    ax.set_ylim(0, 0.56)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/figure1_discovery.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure1_discovery.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/figure1_discovery.pdf/.png")


def figure2():
    """Scatter plot of r vs R colored by beta.

    Panel A: r vs R colored by state-dependence beta.
    Panel B: Residuals from r-R regression by beta quartile.
    """
    pop, _ = generate_calibrated_population(n=2000, seed=42, config="primary")
    r = compute_standard_risk(pop)
    R = simulate_trajectory_risk(pop, T=2.0, k=3, n_sims=300, seed=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: r vs R colored by beta
    scatter = ax1.scatter(r, R, c=pop.beta, cmap='RdYlBu_r', s=6, alpha=0.5,
                          edgecolors='none', vmin=0.1, vmax=0.7)
    cbar = plt.colorbar(scatter, ax=ax1, label='State-dependence ($\\beta$)',
                        shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    # Regression line
    coeffs = np.polyfit(r, R, 1)
    x_line = np.linspace(r.min(), r.max(), 100)
    ax1.plot(x_line, np.polyval(coeffs, x_line), 'k--', alpha=0.3, lw=0.8)

    ax1.set_xlabel('Standard Risk Score ($r$)', fontsize=10)
    ax1.set_ylabel('Trajectory Risk ($R$)', fontsize=10)
    ax1.set_title('A', fontsize=11, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Residuals by beta quartile
    R_predicted = np.polyval(coeffs, r)
    residuals = R - R_predicted
    beta_quartiles = np.digitize(pop.beta,
                                 np.percentile(pop.beta, [25, 50, 75]))
    quartile_labels = ['Q1\n(low $\\beta$)', 'Q2', 'Q3',
                       'Q4\n(high $\\beta$)']

    bp_data = [residuals[beta_quartiles == q] for q in range(4)]
    bp = ax2.boxplot(bp_data, tick_labels=quartile_labels, patch_artist=True,
                     showfliers=False, widths=0.45,
                     medianprops=dict(color='black', linewidth=1))

    colors_bp = ['#4472C4', '#8FAADC', '#F4B183', '#C00000']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, lw=0.5)
    ax2.set_ylabel('Residual ($R$ $-$ predicted $R$ from $r$)', fontsize=10)
    ax2.set_xlabel('$\\beta$ Quartile', fontsize=10)
    ax2.set_title('B', fontsize=11, fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/figure2_mechanism.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure2_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/figure2_mechanism.pdf/.png")


def efigure1_nb_fit():
    """eFigure 1: NB fit quality — predicted vs simulated distributions.

    3x3 grid of patients spanning (beta, lambda_0) percentiles.
    Each panel shows empirical (bars) vs NB-predicted (line) PMF.
    """
    pop, _ = generate_calibrated_population(n=5000, seed=42, config="primary")
    nb_fit = assess_nb_fit_quality(pop, T=2.0, k=3, n_sims=500, seed=42)

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    axes = axes.flatten()

    for i, patient in enumerate(nb_fit["nb_fit_patients"][:9]):
        ax = axes[i]
        bins = np.array(patient["bins"])
        emp_pmf = np.array(patient["empirical_pmf"])
        nb_pmf = np.array(patient["nb_pmf"])

        ax.bar(bins, emp_pmf, width=0.8, alpha=0.5, color='#4472C4',
               edgecolor='#4472C4', linewidth=0.5, label='Simulated')
        ax.plot(bins, nb_pmf, 'o-', color='#C00000', markersize=3,
                linewidth=1, label='NB predicted')

        ax.set_title(
            f"$\\beta$={patient['beta']:.2f}, "
            f"$\\lambda_0$={patient['lambda_0']:.2f}\n"
            f"$\\chi^2$ p={patient['chi2_p']:.2f}",
            fontsize=8)
        ax.set_xlim(-0.5, min(bins.max() + 0.5, 15))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

        if i == 0:
            ax.legend(fontsize=7, frameon=False)
        if i >= 6:
            ax.set_xlabel('Event count', fontsize=8)
        if i % 3 == 0:
            ax.set_ylabel('Probability', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/efigure1_nb_fit.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/efigure1_nb_fit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/efigure1_nb_fit.pdf/.png")


if __name__ == "__main__":
    figure1()
    figure2()
    efigure1_nb_fit()
    print("\nAll figures generated.")
