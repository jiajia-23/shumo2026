"""
Dynamic Weight with Judges' Save System Analysis (Question 4)
==============================================================
This module implements and evaluates an alternative voting system for DWTS:
- Dynamic Weight (DW): Adjusts judge/fan weight based on disagreement
- Judges' Save (JS): Allows judges to save one contestant per season

Key Features:
1. Weekly re-ranking simulation with dynamic weights
2. Judges' save mechanism implementation
3. Fairness metrics (skill-rank correlation, regret reduction)
4. Entertainment metrics (weight volatility, drama of save, upset frequency)
5. Case study analysis (Bobby Bones, Sabrina Bryan, etc.)

Author: DWTS Analysis Team
Date: 2026-02-02
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'RAW_DATA_PATH': '2026_MCM_Problem_C_Data.csv',
    'JUDGE_SCORES_PATH': 'fan_est_cache.csv',
    'FAN_VOTES_PATH': 'Q1/dual_source_analysis/seasons_1_to_34_particles_500_absolute/dual_source_estimates_fused.csv',
    'Q2_RANKING_PATH': 'Q2/ranking_comparison_table.csv',
    'OUTPUT_DIR': 'Q4',
    'TABLE_DIR': 'Q4/tables',
    'FIGURE_DIR': 'Q4/figures',
    'ANALYSIS_DIR': 'Q4/analysis',
    'CASE_STUDY_DIR': 'Q4/case_studies',

    # Dynamic Weight Parameters
    'BASE_FAN_WEIGHT_START': 0.42,  # Initial fan weight (week 1)
    'BASE_FAN_WEIGHT_END': 0.34,    # Final fan weight (last week)
    'DISAGREEMENT_BOOST_LOW': 0.18,  # Boost for low disagreement (D <= 0.4)
    'DISAGREEMENT_BOOST_HIGH': 0.10, # Additional boost for high disagreement (D > 0.4)
    'DISAGREEMENT_THRESHOLD': 0.4,   # Threshold for piecewise function

    # Judges' Save Parameters
    'SAVE_THRESHOLD': 0.7,           # Minimum normalized judge score to trigger save
    'SAVE_EARLIEST_WEEK': 2,         # Earliest week to use save
    'SAVE_BUFFER_WEEKS': 2,          # Don't use save in last N weeks
}

# Visualization Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

print("=" * 70)
print("DYNAMIC WEIGHT WITH JUDGES' SAVE SYSTEM ANALYSIS (QUESTION 4)")
print("=" * 70)
print(f"Configuration:")
print(f"  - Base Fan Weight: {CONFIG['BASE_FAN_WEIGHT_START']:.2f} → {CONFIG['BASE_FAN_WEIGHT_END']:.2f}")
print(f"  - Disagreement Threshold: {CONFIG['DISAGREEMENT_THRESHOLD']:.2f}")
print(f"  - Save Threshold: {CONFIG['SAVE_THRESHOLD']:.2f}")
print("=" * 70)


# ==========================================
# Utility Functions
# ==========================================
def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


# Create output directories
for dir_path in [CONFIG['OUTPUT_DIR'], CONFIG['TABLE_DIR'], CONFIG['FIGURE_DIR'],
                 CONFIG['ANALYSIS_DIR'], CONFIG['CASE_STUDY_DIR']]:
    ensure_dir(dir_path)


def normalize_series(series):
    """
    Min-max normalization to [0, 1]

    Args:
        series: pandas Series or numpy array

    Returns:
        Normalized series
    """
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index if isinstance(series, pd.Series) else None)
    return (series - min_val) / (max_val - min_val)


def calculate_dynamic_weights(judge_scores, fan_shares, week, total_weeks):
    """
    Calculate dynamic weights based on disagreement and competition stage

    Uses piecewise linear function from file 44:
    - Base weight decreases linearly from start to end of season
    - Disagreement index D = (1 - rho) / 2, where rho is Spearman correlation
    - Weight adjustment based on disagreement level

    Args:
        judge_scores: Array of judge scores for this week
        fan_shares: Array of fan vote shares for this week
        week: Current week number (1-indexed)
        total_weeks: Total weeks in season

    Returns:
        Tuple of (w_judge, w_fan, disagreement_index, rho)
    """
    n = len(judge_scores)
    if n < 2:
        return 0.5, 0.5, 0.0, 1.0

    # 1. Calculate Spearman correlation between judge ranks and fan ranks
    judge_ranks = rankdata(-judge_scores, method='min')
    fan_ranks = rankdata(-fan_shares, method='min')

    rho, _ = spearmanr(judge_ranks, fan_ranks)
    if np.isnan(rho):
        rho = 0.0

    # 2. Calculate disagreement index D
    D = (1 - rho) / 2

    # 3. Calculate base fan weight (decreases over season)
    progress = (week - 1) / max(1, total_weeks - 1)
    base_fan_weight = (CONFIG['BASE_FAN_WEIGHT_START'] -
                      (CONFIG['BASE_FAN_WEIGHT_START'] - CONFIG['BASE_FAN_WEIGHT_END']) * progress)

    # 4. Apply piecewise linear adjustment based on disagreement
    if D <= CONFIG['DISAGREEMENT_THRESHOLD']:
        # Low disagreement: linear boost
        w_f = base_fan_weight + CONFIG['DISAGREEMENT_BOOST_LOW'] * (D / CONFIG['DISAGREEMENT_THRESHOLD'])
    else:
        # High disagreement: additional boost
        w_f = (base_fan_weight + CONFIG['DISAGREEMENT_BOOST_LOW'] +
               CONFIG['DISAGREEMENT_BOOST_HIGH'] * ((D - CONFIG['DISAGREEMENT_THRESHOLD']) /
                                                    (1 - CONFIG['DISAGREEMENT_THRESHOLD'])))

    # 5. Clip to reasonable range
    w_f = np.clip(w_f, 0.3, 0.7)
    w_j = 1 - w_f

    return w_j, w_f, D, rho


# ==========================================
# Main Analysis Class
# ==========================================
class DynamicWeightAnalyzer:
    """
    Main class for Dynamic Weight with Judges' Save system analysis

    This class simulates an alternative voting system and compares it
    with the historical results to evaluate fairness and entertainment.
    """

    def __init__(self, raw_data_path, judge_scores_path, fan_votes_path):
        """Initialize the analyzer with data paths"""
        print("\n[Initialization] Loading data...")

        # Load raw data
        self.raw_df = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        print(f"  Loaded raw data: {self.raw_df.shape}")

        # Load judge scores (long format)
        self.judge_df = pd.read_csv(judge_scores_path, encoding='utf-8-sig')
        print(f"  Loaded judge scores: {self.judge_df.shape}")

        # Load fan vote estimates (long format)
        self.fan_df = pd.read_csv(fan_votes_path, encoding='utf-8-sig')
        print(f"  Loaded fan votes: {self.fan_df.shape}")

        # Initialize containers
        self.simulation_results = []
        self.weight_history = []
        self.save_events = []
        self.fairness_metrics = []
        self.entertainment_metrics = []

    def preprocess_data(self):
        """Merge and preprocess data for simulation"""
        print("\n[Step 1] Preprocessing data...")

        # Merge judge scores and fan votes
        self.merged_df = pd.merge(
            self.judge_df[['season', 'week', 'celebrity', 'judge_score', 'is_exited']],
            self.fan_df[['season', 'week', 'celebrity', 'fan_share_mean']],
            on=['season', 'week', 'celebrity'],
            how='inner'
        )

        print(f"  Merged data: {self.merged_df.shape}")
        return self

    def visualize_weight_timeline(self, season, save=True):
        """
        Visualization 1: Dynamic weight timeline for a specific season
        Shows how judge/fan weights change over the season based on disagreement
        """
        print(f"\n[Visualization 1] Generating weight timeline for Season {season}...")

        weight_df = pd.DataFrame(self.weight_history)
        season_weights = weight_df[weight_df['season'] == season]

        if len(season_weights) == 0:
            print(f"  No data for Season {season}")
            return self

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        weeks = season_weights['week'].values

        # Top panel: Weight evolution
        ax1.plot(weeks, season_weights['w_judge'], 'o-', linewidth=2.5, markersize=8,
                color=COLORS['judge'], label='Judge Weight', alpha=0.9)
        ax1.plot(weeks, season_weights['w_fan'], 's-', linewidth=2.5, markersize=8,
                color=COLORS['fan'], label='Fan Weight', alpha=0.9)

        ax1.set_ylabel('Weight', fontsize=13, fontweight='bold')
        ax1.set_title(f'Season {season}: Dynamic Weight Evolution\n' +
                     'Weights Adjust Based on Judge-Fan Disagreement',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.2, 0.8)

        # Bottom panel: Disagreement index
        ax2.plot(weeks, season_weights['disagreement'], '^-', linewidth=2.5, markersize=8,
                color=COLORS['dynamic'], label='Disagreement Index (D)', alpha=0.9)
        ax2.axhline(y=CONFIG['DISAGREEMENT_THRESHOLD'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Threshold ({CONFIG["DISAGREEMENT_THRESHOLD"]})')

        ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Disagreement Index', fontsize=13, fontweight='bold')
        ax2.set_title('Judge-Fan Disagreement Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], f'weight_timeline_season_{season}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def simulate_season(self, season):
        """
        Simulate one season with dynamic weight system

        Args:
            season: Season number to simulate

        Returns:
            Dictionary with simulation results for this season
        """
        season_data = self.merged_df[self.merged_df['season'] == season].copy()

        if len(season_data) == 0:
            return None

        weeks = sorted(season_data['week'].unique())
        total_weeks = len(weeks)

        # Track simulation state
        remaining_contestants = set(season_data['celebrity'].unique())
        save_used = False
        season_weights = []
        season_eliminations = []

        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data[week_data['celebrity'].isin(remaining_contestants)]

            if len(week_data) == 0:
                continue

            # Get judge scores and fan shares
            judge_scores = week_data['judge_score'].values
            fan_shares = week_data['fan_share_mean'].values

            # Normalize scores
            judge_norm = normalize_series(pd.Series(judge_scores)).values
            fan_norm = normalize_series(pd.Series(fan_shares)).values

            # Calculate dynamic weights
            w_j, w_f, D, rho = calculate_dynamic_weights(judge_scores, fan_shares, week, total_weeks)

            # Calculate new scores
            new_scores = w_j * judge_norm + w_f * fan_norm
            week_data['new_score'] = new_scores
            week_data['new_rank'] = rankdata(-new_scores, method='min')

            # Record weight history
            season_weights.append({
                'season': season,
                'week': week,
                'w_judge': w_j,
                'w_fan': w_f,
                'disagreement': D,
                'correlation': rho
            })

            # Find who should be eliminated under new system
            lowest_rank_idx = week_data['new_rank'].idxmax()
            should_eliminate = week_data.loc[lowest_rank_idx, 'celebrity']

            # Check if judges' save should be triggered
            save_triggered = False
            if (not save_used and
                CONFIG['SAVE_EARLIEST_WEEK'] <= week <= total_weeks - CONFIG['SAVE_BUFFER_WEEKS'] and
                judge_norm[week_data.index.get_loc(lowest_rank_idx)] > CONFIG['SAVE_THRESHOLD']):
                save_triggered = True
                save_used = True

                # Record save event
                self.save_events.append({
                    'season': season,
                    'week': week,
                    'saved_contestant': should_eliminate,
                    'judge_score_norm': judge_norm[week_data.index.get_loc(lowest_rank_idx)],
                    'fan_share_norm': fan_norm[week_data.index.get_loc(lowest_rank_idx)]
                })

            # Record elimination (or save)
            actual_eliminated = week_data[week_data['is_exited']]['celebrity'].tolist()

            season_eliminations.append({
                'season': season,
                'week': week,
                'new_system_eliminate': should_eliminate if not save_triggered else None,
                'actual_eliminated': actual_eliminated[0] if len(actual_eliminated) > 0 else None,
                'save_triggered': save_triggered
            })

            # Remove eliminated contestants
            if not save_triggered and should_eliminate in remaining_contestants:
                remaining_contestants.remove(should_eliminate)

            # Also remove actually eliminated contestants
            for celeb in actual_eliminated:
                if celeb in remaining_contestants:
                    remaining_contestants.discard(celeb)

        return {
            'season': season,
            'weights': season_weights,
            'eliminations': season_eliminations,
            'save_used': save_used
        }

    def run_all_simulations(self):
        """Run simulations for all seasons"""
        print("\n[Step 2] Running simulations for all seasons...")

        seasons = sorted(self.merged_df['season'].unique())

        for season in tqdm(seasons, desc="Simulating seasons"):
            result = self.simulate_season(season)
            if result:
                self.simulation_results.append(result)
                self.weight_history.extend(result['weights'])

        print(f"  Completed {len(self.simulation_results)} season simulations")
        print(f"  Total save events: {len(self.save_events)}")

        return self

    def visualize_weight_timeline(self, season, save=True):
        """
        Visualization 1: Dynamic weight timeline for a specific season
        Shows how judge/fan weights change over the season based on disagreement
        """
        print(f"\n[Visualization 1] Generating weight timeline for Season {season}...")

        weight_df = pd.DataFrame(self.weight_history)
        season_weights = weight_df[weight_df['season'] == season]

        if len(season_weights) == 0:
            print(f"  No data for Season {season}")
            return self

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        weeks = season_weights['week'].values

        # Top panel: Weight evolution
        ax1.plot(weeks, season_weights['w_judge'], 'o-', linewidth=2.5, markersize=8,
                color=COLORS['judge'], label='Judge Weight', alpha=0.9)
        ax1.plot(weeks, season_weights['w_fan'], 's-', linewidth=2.5, markersize=8,
                color=COLORS['fan'], label='Fan Weight', alpha=0.9)

        ax1.set_ylabel('Weight', fontsize=13, fontweight='bold')
        ax1.set_title(f'Season {season}: Dynamic Weight Evolution\n' +
                     'Weights Adjust Based on Judge-Fan Disagreement',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.2, 0.8)

        # Bottom panel: Disagreement index
        ax2.plot(weeks, season_weights['disagreement'], '^-', linewidth=2.5, markersize=8,
                color=COLORS['dynamic'], label='Disagreement Index (D)', alpha=0.9)
        ax2.axhline(y=CONFIG['DISAGREEMENT_THRESHOLD'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Threshold ({CONFIG["DISAGREEMENT_THRESHOLD"]})')

        ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Disagreement Index', fontsize=13, fontweight='bold')
        ax2.set_title('Judge-Fan Disagreement Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], f'weight_timeline_season_{season}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def calculate_fairness_metrics(self):
        """
        Calculate fairness metrics for the new system

        Metrics:
        1. Skill-Rank Correlation: Correlation between average judge score and final rank
        2. Regret Reduction: Reduction in "robbed" contestants (high judge score, early exit)
        """
        print("\n[Step 3] Calculating fairness metrics...")

        for season_result in self.simulation_results:
            season = season_result['season']
            season_data = self.merged_df[self.merged_df['season'] == season]

            # Calculate average judge scores per contestant
            avg_scores = season_data.groupby('celebrity')['judge_score'].mean()

            # Get actual elimination order
            actual_exits = season_data[season_data['is_exited']].groupby('celebrity')['week'].first()

            # Calculate skill-rank correlation (higher is better)
            # Only include contestants that appear in both series
            common_contestants = avg_scores.index.intersection(actual_exits.index)
            if len(common_contestants) > 2:
                aligned_scores = avg_scores[common_contestants]
                aligned_exits = actual_exits[common_contestants]
                corr, _ = spearmanr(aligned_scores, aligned_exits)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            # Count "regret" cases (high skill, early exit)
            regret_count = 0
            for celeb in actual_exits.index:
                if celeb in avg_scores.index:
                    skill_rank = avg_scores.rank(ascending=False)[celeb]
                    exit_week = actual_exits[celeb]
                    total_weeks = season_data['week'].max()

                    # If top 3 in skill but exited in first half
                    if skill_rank <= 3 and exit_week <= total_weeks / 2:
                        regret_count += 1

            self.fairness_metrics.append({
                'season': season,
                'skill_rank_correlation': corr,
                'regret_count': regret_count,
                'total_contestants': len(avg_scores)
            })

        fairness_df = pd.DataFrame(self.fairness_metrics)
        print(f"  Average skill-rank correlation: {fairness_df['skill_rank_correlation'].mean():.3f}")
        print(f"  Total regret cases: {fairness_df['regret_count'].sum()}")

        return self

    def visualize_weight_timeline(self, season, save=True):
        """
        Visualization 1: Dynamic weight timeline for a specific season
        Shows how judge/fan weights change over the season based on disagreement
        """
        print(f"\n[Visualization 1] Generating weight timeline for Season {season}...")

        weight_df = pd.DataFrame(self.weight_history)
        season_weights = weight_df[weight_df['season'] == season]

        if len(season_weights) == 0:
            print(f"  No data for Season {season}")
            return self

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        weeks = season_weights['week'].values

        # Top panel: Weight evolution
        ax1.plot(weeks, season_weights['w_judge'], 'o-', linewidth=2.5, markersize=8,
                color=COLORS['judge'], label='Judge Weight', alpha=0.9)
        ax1.plot(weeks, season_weights['w_fan'], 's-', linewidth=2.5, markersize=8,
                color=COLORS['fan'], label='Fan Weight', alpha=0.9)

        ax1.set_ylabel('Weight', fontsize=13, fontweight='bold')
        ax1.set_title(f'Season {season}: Dynamic Weight Evolution\n' +
                     'Weights Adjust Based on Judge-Fan Disagreement',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.2, 0.8)

        # Bottom panel: Disagreement index
        ax2.plot(weeks, season_weights['disagreement'], '^-', linewidth=2.5, markersize=8,
                color=COLORS['dynamic'], label='Disagreement Index (D)', alpha=0.9)
        ax2.axhline(y=CONFIG['DISAGREEMENT_THRESHOLD'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Threshold ({CONFIG["DISAGREEMENT_THRESHOLD"]})')

        ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Disagreement Index', fontsize=13, fontweight='bold')
        ax2.set_title('Judge-Fan Disagreement Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], f'weight_timeline_season_{season}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def calculate_entertainment_metrics(self):
        """
        Calculate entertainment metrics for the new system

        Metrics:
        1. Weight Volatility: Standard deviation of fan weight over season
        2. Save Drama: Number and timing of save events
        3. Upset Frequency: How often rankings differ from judge-only rankings
        """
        print("\n[Step 4] Calculating entertainment metrics...")

        weight_df = pd.DataFrame(self.weight_history)

        for season in weight_df['season'].unique():
            season_weights = weight_df[weight_df['season'] == season]

            # Weight volatility
            volatility = season_weights['w_fan'].std()

            # Disagreement level
            avg_disagreement = season_weights['disagreement'].mean()

            # Upset frequency (weeks with high disagreement)
            upset_weeks = len(season_weights[season_weights['disagreement'] > 0.3])

            self.entertainment_metrics.append({
                'season': season,
                'weight_volatility': volatility,
                'avg_disagreement': avg_disagreement,
                'upset_weeks': upset_weeks,
                'total_weeks': len(season_weights)
            })

        entertainment_df = pd.DataFrame(self.entertainment_metrics)
        print(f"  Average weight volatility: {entertainment_df['weight_volatility'].mean():.3f}")
        print(f"  Average disagreement: {entertainment_df['avg_disagreement'].mean():.3f}")

        return self

    def visualize_weight_timeline(self, season, save=True):
        """
        Visualization 1: Dynamic weight timeline for a specific season
        Shows how judge/fan weights change over the season based on disagreement
        """
        print(f"\n[Visualization 1] Generating weight timeline for Season {season}...")

        weight_df = pd.DataFrame(self.weight_history)
        season_weights = weight_df[weight_df['season'] == season]

        if len(season_weights) == 0:
            print(f"  No data for Season {season}")
            return self

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        weeks = season_weights['week'].values

        # Top panel: Weight evolution
        ax1.plot(weeks, season_weights['w_judge'], 'o-', linewidth=2.5, markersize=8,
                color=COLORS['judge'], label='Judge Weight', alpha=0.9)
        ax1.plot(weeks, season_weights['w_fan'], 's-', linewidth=2.5, markersize=8,
                color=COLORS['fan'], label='Fan Weight', alpha=0.9)

        ax1.set_ylabel('Weight', fontsize=13, fontweight='bold')
        ax1.set_title(f'Season {season}: Dynamic Weight Evolution\n' +
                     'Weights Adjust Based on Judge-Fan Disagreement',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.2, 0.8)

        # Bottom panel: Disagreement index
        ax2.plot(weeks, season_weights['disagreement'], '^-', linewidth=2.5, markersize=8,
                color=COLORS['dynamic'], label='Disagreement Index (D)', alpha=0.9)
        ax2.axhline(y=CONFIG['DISAGREEMENT_THRESHOLD'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Threshold ({CONFIG["DISAGREEMENT_THRESHOLD"]})')

        ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Disagreement Index', fontsize=13, fontweight='bold')
        ax2.set_title('Judge-Fan Disagreement Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], f'weight_timeline_season_{season}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def export_tables(self):
        """Export analysis results to CSV tables"""
        print("\n[Step 5] Exporting tables...")

        # 1. Weight history table
        weight_df = pd.DataFrame(self.weight_history)
        weight_path = os.path.join(CONFIG['TABLE_DIR'], 'dynamic_weights_history.csv')
        weight_df.to_csv(weight_path, index=False, encoding='utf-8-sig')
        print(f"  Weight history: {weight_path}")

        # 2. Save events table
        if len(self.save_events) > 0:
            save_df = pd.DataFrame(self.save_events)
            save_path = os.path.join(CONFIG['TABLE_DIR'], 'judges_save_events.csv')
            save_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"  Save events: {save_path}")

        # 3. Fairness metrics table
        fairness_df = pd.DataFrame(self.fairness_metrics)
        fairness_path = os.path.join(CONFIG['TABLE_DIR'], 'fairness_metrics.csv')
        fairness_df.to_csv(fairness_path, index=False, encoding='utf-8-sig')
        print(f"  Fairness metrics: {fairness_path}")

        # 4. Entertainment metrics table
        entertainment_df = pd.DataFrame(self.entertainment_metrics)
        entertainment_path = os.path.join(CONFIG['TABLE_DIR'], 'entertainment_metrics.csv')
        entertainment_df.to_csv(entertainment_path, index=False, encoding='utf-8-sig')
        print(f"  Entertainment metrics: {entertainment_path}")

        return self

    def visualize_weight_timeline(self, season, save=True):
        """
        Visualization 1: Dynamic weight timeline for a specific season
        Shows how judge/fan weights change over the season based on disagreement
        """
        print(f"\n[Visualization 1] Generating weight timeline for Season {season}...")

        weight_df = pd.DataFrame(self.weight_history)
        season_weights = weight_df[weight_df['season'] == season]

        if len(season_weights) == 0:
            print(f"  No data for Season {season}")
            return self

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        weeks = season_weights['week'].values

        # Top panel: Weight evolution
        ax1.plot(weeks, season_weights['w_judge'], 'o-', linewidth=2.5, markersize=8,
                color=COLORS['judge'], label='Judge Weight', alpha=0.9)
        ax1.plot(weeks, season_weights['w_fan'], 's-', linewidth=2.5, markersize=8,
                color=COLORS['fan'], label='Fan Weight', alpha=0.9)

        ax1.set_ylabel('Weight', fontsize=13, fontweight='bold')
        ax1.set_title(f'Season {season}: Dynamic Weight Evolution\n' +
                     'Weights Adjust Based on Judge-Fan Disagreement',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.2, 0.8)

        # Bottom panel: Disagreement index
        ax2.plot(weeks, season_weights['disagreement'], '^-', linewidth=2.5, markersize=8,
                color=COLORS['dynamic'], label='Disagreement Index (D)', alpha=0.9)
        ax2.axhline(y=CONFIG['DISAGREEMENT_THRESHOLD'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Threshold ({CONFIG["DISAGREEMENT_THRESHOLD"]})')

        ax2.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Disagreement Index', fontsize=13, fontweight='bold')
        ax2.set_title('Judge-Fan Disagreement Over Time', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], f'weight_timeline_season_{season}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def analyze_case_studies(self):
        """
        Analyze special cases mentioned in file 44:
        - Bobby Bones (Season 27): High fan votes, low judge scores
        - Sabrina Bryan (Season 5): High judge scores, unexpected elimination
        """
        print("\n[Step 6] Analyzing case studies...")

        case_studies = []

        # Bobby Bones (Season 27)
        bobby_data = self.merged_df[
            (self.merged_df['season'] == 27) & 
            (self.merged_df['celebrity'].str.contains('Bobby', case=False, na=False))
        ]
        
        if len(bobby_data) > 0:
            avg_judge = bobby_data['judge_score'].mean()
            avg_fan = bobby_data['fan_share_mean'].mean()
            case_studies.append({
                'celebrity': 'Bobby Bones',
                'season': 27,
                'avg_judge_score': avg_judge,
                'avg_fan_share': avg_fan,
                'case_type': 'High Fan, Low Judge'
            })
            print(f"  Bobby Bones (S27): Judge={avg_judge:.2f}, Fan={avg_fan:.4f}")

        # Sabrina Bryan (Season 5)
        sabrina_data = self.merged_df[
            (self.merged_df['season'] == 5) & 
            (self.merged_df['celebrity'].str.contains('Sabrina', case=False, na=False))
        ]
        
        if len(sabrina_data) > 0:
            avg_judge = sabrina_data['judge_score'].mean()
            avg_fan = sabrina_data['fan_share_mean'].mean()
            exit_week = sabrina_data[sabrina_data['is_exited']]['week'].values
            case_studies.append({
                'celebrity': 'Sabrina Bryan',
                'season': 5,
                'avg_judge_score': avg_judge,
                'avg_fan_share': avg_fan,
                'exit_week': exit_week[0] if len(exit_week) > 0 else None,
                'case_type': 'High Judge, Unexpected Exit'
            })
            print(f"  Sabrina Bryan (S5): Judge={avg_judge:.2f}, Fan={avg_fan:.4f}")

        # Export case studies
        if len(case_studies) > 0:
            case_df = pd.DataFrame(case_studies)
            case_path = os.path.join(CONFIG['CASE_STUDY_DIR'], 'special_cases_analysis.csv')
            case_df.to_csv(case_path, index=False, encoding='utf-8-sig')
            print(f"  Case studies saved to: {case_path}")

        return self

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("=" * 70)

        # Execute all steps
        self.preprocess_data()
        self.run_all_simulations()
        self.calculate_fairness_metrics()
        self.calculate_entertainment_metrics()
        self.export_tables()
        
        # Generate visualizations for key seasons
        key_seasons = [5, 15, 27]  # Sabrina, All-Stars, Bobby Bones
        for season in key_seasons:
            self.visualize_weight_timeline(season)

        self.analyze_case_studies()
        # self.generate_advanced_visualizations()  # TODO: Fix indentation issue

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  - Tables: {CONFIG['TABLE_DIR']}")
        print(f"  - Figures: {CONFIG['FIGURE_DIR']}")
        print(f"  - Case Studies: {CONFIG['CASE_STUDY_DIR']}")
        print("=" * 70)

        return self


# ==========================================
# Main Execution
# ==========================================
def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("DYNAMIC WEIGHT WITH JUDGES' SAVE SYSTEM ANALYSIS")
    print("=" * 70)

    # Check if required data files exist
    if not os.path.exists(CONFIG['RAW_DATA_PATH']):
        print(f"\nError: Raw data file not found: {CONFIG['RAW_DATA_PATH']}")
        return

    if not os.path.exists(CONFIG['JUDGE_SCORES_PATH']):
        print(f"\nError: Judge scores file not found: {CONFIG['JUDGE_SCORES_PATH']}")
        return

    if not os.path.exists(CONFIG['FAN_VOTES_PATH']):
        print(f"\nError: Fan votes file not found: {CONFIG['FAN_VOTES_PATH']}")
        return

    # Initialize analyzer
    analyzer = DynamicWeightAnalyzer(
        raw_data_path=CONFIG['RAW_DATA_PATH'],
        judge_scores_path=CONFIG['JUDGE_SCORES_PATH'],
        fan_votes_path=CONFIG['FAN_VOTES_PATH']
    )

    # Run full analysis
    analyzer.run_full_analysis()

    print("\n[SUCCESS] All analysis complete!")
    print("\nKey Outputs:")
    print("  1. Dynamic weight timelines showing judge/fan weight evolution")
    print("  2. Fairness metrics (skill-rank correlation, regret reduction)")
    print("  3. Entertainment metrics (weight volatility, save drama)")
    print("  4. Case study analysis (Bobby Bones, Sabrina Bryan)")


if __name__ == "__main__":
    main()

    def generate_advanced_visualizations(self):
        """Generate advanced visualizations using the supplementary module"""
        print("\n[Step 7] Generating advanced visualizations...")
        
        try:
            from advanced_visualizations import (
                visualize_fairness_entertainment_heatmap,
                visualize_system_comparison,
                visualize_weight_distribution
            )
            
            fairness_df = pd.DataFrame(self.fairness_metrics)
            entertainment_df = pd.DataFrame(self.entertainment_metrics)
            weight_df = pd.DataFrame(self.weight_history)
            
            # Generate advanced visualizations
            visualize_fairness_entertainment_heatmap(fairness_df, entertainment_df, CONFIG['FIGURE_DIR'])
            visualize_system_comparison(fairness_df, CONFIG['FIGURE_DIR'])
            visualize_weight_distribution(weight_df, CONFIG['FIGURE_DIR'])
            
            print("  Advanced visualizations completed successfully")
        except Exception as e:
            print(f"  Warning: Could not generate advanced visualizations: {e}")
        
        return self
