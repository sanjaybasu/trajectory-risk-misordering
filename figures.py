#!/usr/bin/env python3
"""
Generate Figures for Medical Decision Making Manuscript.

Figure 1: AI agent discovery progression with clinically calibrated baseline
Figure 2: r vs R scatter plot colored by beta (mechanism visualization)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import compute_standard_risk, simulate_trajectory_risk
from revised_analysis import generate_calibrated_population


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

    # Bracket: clinically calibrated range
    ax.annotate('', xy=(0, 0.075), xytext=(1, 0.075),
                arrowprops=dict(arrowstyle='|-|', color='#5B9BD5', lw=1.2))
    ax.text(0.5, 0.085, 'Clinically\ncalibrated', ha='center', fontsize=7,
            color='#5B9BD5')

    # Bracket: AI exploration
    ax.annotate('', xy=(2, 0.49), xytext=(6, 0.49),
                arrowprops=dict(arrowstyle='|-|', color='#C00000', lw=1.2))
    ax.text(4, 0.50, 'AI-agent exploration', ha='center', fontsize=7,
            color='#C00000')

    # Annotation for negative result
    ax.annotate('High Var($\\lambda_0$)\nstrengthened\nstandard score',
                xy=(3, 0.044), xytext=(3.6, 0.17),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7),
                fontsize=6.5, ha='center', color='#555555', style='italic')

    # Annotation for implausible
    ax.annotate('$\\mu \\approx 0.01$/yr\n(clinically\nimplausible)',
                xy=(5, 0.461), xytext=(4.2, 0.39),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7),
                fontsize=6.5, ha='center', color='#555555', style='italic')

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.15, lw=0.6)
    ax.text(6.5, 0.49, 'Random\nranking', fontsize=6.5, color='black',
            alpha=0.3, va='top')

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

    rho = np.corrcoef(r, R)[0, 1]
    ax1.set_xlabel('Standard Risk Score ($r$)', fontsize=10)
    ax1.set_ylabel('Trajectory Risk ($R$)', fontsize=10)
    ax1.set_title('A', fontsize=11, fontweight='bold', loc='left')
    ax1.text(0.05, 0.95, f'$\\rho$ = {rho:.2f}',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                       edgecolor='gray', linewidth=0.5))

    ax1.annotate('High-$\\beta$ patients:\ntrajectory risk exceeds\nstandard score prediction',
                 xy=(0.4, 0.45), xytext=(1.0, 0.55),
                 arrowprops=dict(arrowstyle='->', color='#C00000', lw=0.8),
                 fontsize=7, color='#C00000', ha='center')

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
    ax2.text(0.05, 0.95,
             'Patients with high cascade\npropensity ($\\beta$) have higher\ntrajectory risk than $r$ predicts',
             transform=ax2.transAxes, fontsize=7, va='top', color='#C00000',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                       edgecolor='gray', linewidth=0.5))

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/figure2_mechanism.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure2_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/figure2_mechanism.pdf/.png")


if __name__ == "__main__":
    figure1()
    figure2()
    print("\nAll figures generated.")
