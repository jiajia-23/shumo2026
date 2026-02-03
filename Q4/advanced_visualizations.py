"""
Advanced Visualizations for Q4 Analysis
========================================
Additional visualization functions for the Dynamic Weight system analysis.

This module provides:
1. Fairness-Entertainment heatmap
2. System comparison visualizations
3. Aggregate analysis across all seasons
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Import configuration from main file
import sys
sys.path.append(os.path.dirname(__file__))

# Visualization Configuration
COLORS = {
    'judge': '#2E86AB',
    'fan': '#F18F01',
    'dynamic': '#06A77D',
    'old_system': '#C73E1D',
    'new_system': '#2E86AB',
    'save_event': '#D4AF37',
    'neutral': '#6C757D',
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11


def visualize_fairness_entertainment_heatmap(fairness_df, entertainment_df, output_dir):
    """
    Visualization: Fairness-Entertainment Heatmap

    Shows the trade-off between fairness and entertainment across seasons.
    Each cell represents a season, colored by a combined metric.
    """
    print("\n[Advanced Viz 1] Generating Fairness-Entertainment Heatmap...")

    # Merge fairness and entertainment metrics
    combined = pd.merge(fairness_df, entertainment_df, on='season')

    # Normalize metrics to 0-1 scale
    combined['fairness_norm'] = (combined['skill_rank_correlation'] -
                                 combined['skill_rank_correlation'].min()) / \
                                (combined['skill_rank_correlation'].max() -
                                 combined['skill_rank_correlation'].min())

    combined['entertainment_norm'] = (combined['avg_disagreement'] -
                                     combined['avg_disagreement'].min()) / \
                                    (combined['avg_disagreement'].max() -
                                     combined['avg_disagreement'].min())

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create scatter plot
    scatter = ax.scatter(combined['fairness_norm'], combined['entertainment_norm'],
                        s=300, c=combined['season'], cmap='viridis',
                        alpha=0.7, edgecolors='white', linewidths=2)

    # Add season labels
    for _, row in combined.iterrows():
        ax.annotate(f"S{int(row['season'])}",
                   xy=(row['fairness_norm'], row['entertainment_norm']),
                   fontsize=9, ha='center', va='center', fontweight='bold')

    # Add quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # Add quadrant labels
    ax.text(0.75, 0.75, 'High Fairness\nHigh Entertainment',
           fontsize=11, ha='center', va='center', alpha=0.6, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.25, 0.75, 'Low Fairness\nHigh Entertainment',
           fontsize=11, ha='center', va='center', alpha=0.6, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.text(0.75, 0.25, 'High Fairness\nLow Entertainment',
           fontsize=11, ha='center', va='center', alpha=0.6, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(0.25, 0.25, 'Low Fairness\nLow Entertainment',
           fontsize=11, ha='center', va='center', alpha=0.6, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    # Formatting
    ax.set_xlabel('Fairness (Skill-Rank Correlation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Entertainment (Disagreement Level)', fontsize=14, fontweight='bold')
    ax.set_title('Fairness-Entertainment Trade-off Analysis\n' +
                'Dynamic Weight System Performance Across Seasons',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Season Number', fontsize=12, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'fairness_entertainment_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")
    plt.close()


def visualize_system_comparison(fairness_df, output_dir):
    """
    Visualization: System Comparison

    Compares the new dynamic weight system with the old system
    across multiple metrics.
    """
    print("\n[Advanced Viz 2] Generating System Comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dynamic Weight System vs. Traditional System\nComprehensive Comparison',
                fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: Fairness distribution
    ax1 = axes[0, 0]
    ax1.hist(fairness_df['skill_rank_correlation'], bins=15,
            color=COLORS['new_system'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axvline(fairness_df['skill_rank_correlation'].mean(),
               color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {fairness_df["skill_rank_correlation"].mean():.3f}')
    ax1.set_xlabel('Skill-Rank Correlation', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Fairness Distribution', fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Regret reduction
    ax2 = axes[0, 1]
    seasons = fairness_df['season'].values
    regrets = fairness_df['regret_count'].values
    ax2.bar(seasons, regrets, color=COLORS['old_system'], alpha=0.7,
           edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Season', fontweight='bold')
    ax2.set_ylabel('Regret Count', fontweight='bold')
    ax2.set_title('Regret Cases by Season\n(High Skill, Early Exit)',
                 fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Fairness over time
    ax3 = axes[1, 0]
    ax3.plot(seasons, fairness_df['skill_rank_correlation'],
            'o-', linewidth=2.5, markersize=8, color=COLORS['dynamic'], alpha=0.9)
    ax3.axhline(y=fairness_df['skill_rank_correlation'].mean(),
               color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Season', fontweight='bold')
    ax3.set_ylabel('Skill-Rank Correlation', fontweight='bold')
    ax3.set_title('Fairness Trend Over Seasons', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    SYSTEM PERFORMANCE SUMMARY
    {'='*40}

    Fairness Metrics:
    • Average Skill-Rank Correlation: {fairness_df['skill_rank_correlation'].mean():.3f}
    • Std Dev: {fairness_df['skill_rank_correlation'].std():.3f}
    • Min: {fairness_df['skill_rank_correlation'].min():.3f}
    • Max: {fairness_df['skill_rank_correlation'].max():.3f}

    Regret Analysis:
    • Total Regret Cases: {fairness_df['regret_count'].sum():.0f}
    • Average per Season: {fairness_df['regret_count'].mean():.2f}
    • Seasons with Zero Regret: {len(fairness_df[fairness_df['regret_count'] == 0]):.0f}

    System Characteristics:
    • Total Seasons Analyzed: {len(fairness_df):.0f}
    • Dynamic Weight Adjustment: ✓ Active
    • Judges' Save Mechanism: ✓ Active
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'system_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")
    plt.close()


def visualize_weight_distribution(weight_df, output_dir):
    """
    Visualization: Weight Distribution Analysis

    Shows the distribution of judge/fan weights across all seasons.
    """
    print("\n[Advanced Viz 3] Generating Weight Distribution Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dynamic Weight Distribution Analysis\nAcross All Seasons',
                fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: Judge weight distribution
    ax1 = axes[0, 0]
    ax1.hist(weight_df['w_judge'], bins=30, color=COLORS['judge'],
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axvline(weight_df['w_judge'].mean(), color='red', linestyle='--',
               linewidth=2.5, label=f'Mean: {weight_df["w_judge"].mean():.3f}')
    ax1.set_xlabel('Judge Weight', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Judge Weight Distribution', fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Fan weight distribution
    ax2 = axes[0, 1]
    ax2.hist(weight_df['w_fan'], bins=30, color=COLORS['fan'],
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(weight_df['w_fan'].mean(), color='red', linestyle='--',
               linewidth=2.5, label=f'Mean: {weight_df["w_fan"].mean():.3f}')
    ax2.set_xlabel('Fan Weight', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Fan Weight Distribution', fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Disagreement distribution
    ax3 = axes[1, 0]
    ax3.hist(weight_df['disagreement'], bins=30, color=COLORS['dynamic'],
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axvline(weight_df['disagreement'].mean(), color='red', linestyle='--',
               linewidth=2.5, label=f'Mean: {weight_df["disagreement"].mean():.3f}')
    ax3.set_xlabel('Disagreement Index', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Judge-Fan Disagreement Distribution', fontweight='bold', pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Correlation distribution
    ax4 = axes[1, 1]
    ax4.hist(weight_df['correlation'], bins=30, color=COLORS['neutral'],
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.axvline(weight_df['correlation'].mean(), color='red', linestyle='--',
               linewidth=2.5, label=f'Mean: {weight_df["correlation"].mean():.3f}')
    ax4.set_xlabel('Spearman Correlation', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Judge-Fan Correlation Distribution', fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'weight_distribution_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("This module provides advanced visualization functions for Q4 analysis.")
    print("Import and use these functions from the main analysis script.")
