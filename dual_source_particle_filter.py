"""
Dual-Source Particle Filter for Fan Vote Estimation
====================================================
This module implements a particle filter to estimate fan voting patterns
by decomposing votes into two components:
1. Base Support (铁粉基数): Loyal fan base that votes regardless of performance
2. Casual Support (表现转化): Performance-driven votes from casual viewers

Mathematical Framework:
- State Space Model with hidden variables (base_share, alpha)
- Particle Filter for Bayesian inference under elimination constraints
- Scoring system adaptation across different seasons (Rank vs Percentage)
"""

import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 0. Configuration and Utilities
# ==========================================
CONFIG = {
    'N_PARTICLES': 1000,      # Number of particles (higher = more accurate but slower)
    'SYSTEM_CHANGE_S3': 3,    # Season when percentage system was introduced
    'SYSTEM_CHANGE_S28': 28,  # Season when judge save (Bottom 2) was introduced
    'JUDGE_SAVE_ERA_RETURN_TO_RANK': True,  # Whether S28+ returns to rank system
    'BASE_VOTE_NOISE': 0.05,  # Week-to-week volatility in base support
    'ALPHA_NOISE': 0.01,      # Week-to-week volatility in performance sensitivity
    'MIN_SURVIVAL_RATE': 0.01, # Minimum particle survival rate before reset
}

# Output directories
OUTPUT_DIR = 'Q1/dual_source_analysis'

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_run_directories(seasons_list, n_particles):
    """
    Create organized directory structure for this run

    Args:
        seasons_list: List of seasons being analyzed
        n_particles: Number of particles used

    Returns:
        Dictionary with paths for different types of outputs
    """
    # Create run identifier
    if len(seasons_list) == 1:
        season_str = f"season_{seasons_list[0]}"
    elif len(seasons_list) <= 3:
        season_str = f"seasons_{'_'.join(map(str, seasons_list))}"
    else:
        season_str = f"seasons_{min(seasons_list)}_to_{max(seasons_list)}"

    run_name = f"{season_str}_particles_{n_particles}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)

    # Create subdirectories
    paths = {
        'base': run_dir,
        'prediction': os.path.join(run_dir, 'prediction'),
        'fan_support': os.path.join(run_dir, 'fan_support'),
        'analysis': os.path.join(run_dir, 'analysis')
    }

    for path in paths.values():
        ensure_dir(path)

    return paths

# Create base output directory
ensure_dir(OUTPUT_DIR)

# ==========================================
# Visualization Configuration
# ==========================================
# Set matplotlib style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D4AF37',      # Gold
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'base': '#4A90E2',         # Light Blue for base support
    'casual': '#F39C12',       # Orange for casual support
}

# Font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16

print("=" * 60)
print("DUAL-SOURCE PARTICLE FILTER SYSTEM")
print("=" * 60)
print(f"Configuration:")
print(f"  - Particles: {CONFIG['N_PARTICLES']}")
print(f"  - Base Vote Noise: {CONFIG['BASE_VOTE_NOISE']}")
print(f"  - Output Directory: {OUTPUT_DIR}")
print("=" * 60)

# ==========================================
# 1. Scoring System Logic (Strategy Pattern)
# ==========================================
class ScoringSystem:
    """
    Handles different scoring systems across seasons:
    - S1-2: Rank System (sum of ranks, lower is better)
    - S3-27: Percentage System (sum of percentages, higher is better)
    - S28+: Rank System with Judge Save (Bottom 2 rule)
    """

    @staticmethod
    def calculate_total_rank(judge_scores, fan_shares, season):
        """
        Calculate total scores based on season's scoring system

        Args:
            judge_scores: Array of judge scores
            fan_shares: Array of fan vote shares (already normalized)
            season: Season number

        Returns:
            total_scores: Combined scores
            score_type: "Higher is Better" or "Lower is Better"
        """
        n = len(judge_scores)

        # 1. Process judge scores - normalize to shares
        j_sum = np.sum(judge_scores)
        j_share = judge_scores / j_sum if j_sum > 0 else np.ones(n) / n

        # 2. Fan shares are already normalized
        f_share = fan_shares

        # 3. Combine based on season's system
        if 3 <= season < 28:
            # === Percentage System (S3-27) ===
            # Total = Judge% + Fan%
            # Higher score is better
            total_score = j_share + f_share
            return total_score, "Higher is Better"
        else:
            # === Rank System (S1-2, S28+) ===
            # Convert shares to ranks (higher share -> better rank -> lower number)
            j_rank = rankdata(-j_share, method='min')
            f_rank = rankdata(-f_share, method='min')

            # Total = Sum of Ranks (lower is better)
            total_score = j_rank + f_rank
            return total_score, "Lower is Better"

    @staticmethod
    def check_elimination_constraint(total_scores, score_type,
                                     eliminated_indices, safe_indices, season):
        """
        Check if particle's scores are consistent with elimination results

        Args:
            total_scores: Array of total scores
            score_type: "Higher is Better" or "Lower is Better"
            eliminated_indices: Indices of eliminated contestants
            safe_indices: Indices of safe contestants
            season: Season number

        Returns:
            Boolean: True if consistent with elimination rules
        """
        if len(eliminated_indices) == 0:
            return True  # No elimination week, all particles valid

        # Extract scores
        elim_scores = total_scores[eliminated_indices]
        safe_scores = total_scores[safe_indices] if len(safe_indices) > 0 else np.array([])

        if len(safe_scores) == 0:
            return True  # Finals week, special handling

        # === Constraint Logic ===
        if season >= 28:
            # === S28+: Judge Save (Bottom 2) ===
            # Rule: The eliminated person must be in Bottom 2
            # But one person from Bottom 2 can be saved by judges
            # So: At most 1 safe person can have worse score than eliminated person

            if score_type == "Lower is Better":
                # For ranks: higher number = worse
                # Count how many safe contestants have worse (higher) scores than eliminated
                worst_elim_score = np.max(elim_scores)
                count_worse_survivors = np.sum(safe_scores > worst_elim_score)
            else:
                # For percentages: lower number = worse
                best_elim_score = np.max(elim_scores)
                count_worse_survivors = np.sum(safe_scores < best_elim_score)

            # Allow at most 1 survivor to be worse (the judge-saved contestant)
            return count_worse_survivors <= 1
        else:
            # === Regular Elimination (S1-27) ===
            # Rule: All eliminated must be worse than all safe
            # (Best eliminated must be worse than worst safe)

            if score_type == "Higher is Better":
                # Percentage system: Min(Safe) > Max(Elim)
                return np.min(safe_scores) > np.max(elim_scores)
            else:
                # Rank system: Max(Safe) < Min(Elim)
                # (lower rank number is better)
                return np.max(safe_scores) < np.min(elim_scores)

# ==========================================
# 2. Particle Filter Model (State Space Model)
# ==========================================
class ContestantState:
    """
    Represents the hidden state of a single contestant

    Hidden Variables:
    - base_share: Loyal fan base (铁粉基数)
    - alpha: Performance sensitivity coefficient (表现转化率)
    """

    def __init__(self, name, judge_corr=0.5):
        """
        Initialize contestant state with prior knowledge

        Args:
            name: Contestant name
            judge_corr: Judge-fan correlation (from Q1 analysis)
                       Lower correlation suggests higher base support
        """
        self.name = name

        # Prior: Low correlation -> High base support
        # This encodes the insight from Q1 that contestants with low
        # judge-fan correlation have strong loyal fan bases
        base_prior = 0.15 if judge_corr < 0.3 else 0.05

        # B_t: Base share (initialized with Beta distribution + prior)
        # Beta(2, 20) gives a right-skewed distribution (most people have low base)
        self.base_share = np.random.beta(2, 20) + base_prior
        self.base_share = min(0.5, self.base_share)  # Cap at 50%

        # alpha: Performance sensitivity (how much judge scores affect votes)
        self.alpha = np.random.uniform(0.1, 0.5)

    def predict(self):
        """
        State transition: Random walk with bounded support

        Models week-to-week changes in hidden variables:
        - Base support drifts slowly (loyal fans are stable)
        - Alpha drifts very slowly (personality trait)
        """
        # B_t = B_{t-1} + Noise
        noise = np.random.normal(0, CONFIG['BASE_VOTE_NOISE'])
        self.base_share = np.clip(self.base_share + noise, 0.001, 0.5)

        # alpha stays relatively stable
        alpha_noise = np.random.normal(0, CONFIG['ALPHA_NOISE'])
        self.alpha = np.clip(self.alpha + alpha_noise, 0.01, 1.0)


class Particle:
    """
    Represents one possible realization of the season (a world line)

    Each particle maintains states for all contestants and tracks
    whether its predictions are consistent with observed eliminations
    """

    def __init__(self, contestants_data):
        """
        Initialize particle with contestant states

        Args:
            contestants_data: Dict {name: judge_corr, ...}
        """
        self.states = {name: ContestantState(name, corr)
                      for name, corr in contestants_data.items()}
        self.history = []  # Track predictions over time
        self.weight = 1.0

    def step(self, week_df, season):
        """
        Advance particle by one week and check consistency

        Args:
            week_df: DataFrame with current week's data
            season: Season number

        Returns:
            Boolean: True if particle is consistent with observations
        """
        # 1. Extract active contestants this week
        active_names = week_df['celebrity'].values
        active_judge_scores = week_df['judge_score'].values

        # 2. State prediction (Predict step)
        for name in active_names:
            if name in self.states:
                self.states[name].predict()
            else:
                # Handle new contestant (rare edge case)
                self.states[name] = ContestantState(name)

        # 3. Calculate fan votes (Observation function)
        # Dual-Source Model: Fan_Share = Normalize(Base + Alpha * Norm(JudgeScore))

        # Normalize judge scores for casual vote calculation
        j_total = np.sum(active_judge_scores)
        j_norm = active_judge_scores / j_total if j_total > 0 else np.zeros_like(active_judge_scores)

        raw_votes = []
        bases = []
        casuals = []

        for i, name in enumerate(active_names):
            state = self.states[name]
            # Dual-source equation
            casual_vote = state.alpha * j_norm[i]
            total_vote = state.base_share + casual_vote

            raw_votes.append(total_vote)
            bases.append(state.base_share)
            casuals.append(casual_vote)

        # Normalize to get shares
        raw_votes = np.array(raw_votes)
        fan_shares = raw_votes / np.sum(raw_votes)

        # 4. Check consistency with elimination (Update step)
        eliminated_mask = week_df['is_exited'].values
        eliminated_idx = np.where(eliminated_mask)[0]
        safe_idx = np.where(~eliminated_mask)[0]

        total_scores, score_type = ScoringSystem.calculate_total_rank(
            active_judge_scores, fan_shares, season
        )

        is_consistent = ScoringSystem.check_elimination_constraint(
            total_scores, score_type, eliminated_idx, safe_idx, season
        )

        # 5. Record snapshot if consistent
        if is_consistent:
            snapshot = []
            for i, name in enumerate(active_names):
                snapshot.append({
                    'celebrity': name,
                    'week': week_df['week'].iloc[0],
                    'fan_share': fan_shares[i],
                    'base_share_est': bases[i],
                    'casual_share_est': casuals[i]
                })
            self.history.append(snapshot)
            return True
        else:
            return False


# ==========================================
# 3. Core Controller (Dual-Source Estimator)
# ==========================================
class DualSourceEstimator:
    """
    Main controller for dual-source particle filter estimation

    Manages data preprocessing, particle filter execution,
    and result aggregation across all seasons
    """

    def __init__(self, data_path):
        """
        Initialize estimator with data

        Args:
            data_path: Path to preprocessed data CSV
        """
        self.data_path = data_path
        self.df = None
        self.results = []
        self.credibility_scores = []

    def load_data(self):
        """Load and validate preprocessed data"""
        print("\n[Step 1] Loading preprocessed data...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')

        # Validate required columns
        required_cols = ['season', 'week', 'celebrity', 'judge_score', 'is_exited']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"  Loaded {len(self.df)} records")
        print(f"  Seasons: {sorted(self.df['season'].unique())}")
        print(f"  Contestants: {self.df['celebrity'].nunique()}")
        return self

    def run_season_estimation(self, season_df, contestant_correlations=None):
        """
        Run particle filter for a single season

        Args:
            season_df: DataFrame for one season
            contestant_correlations: Dict {name: judge_corr} from Q1 analysis

        Returns:
            None (results stored in self.results)
        """
        season = season_df['season'].iloc[0]
        print(f"\n[Season {season}] Starting particle filter...")

        # 1. Initialize particle swarm
        contestants = season_df['celebrity'].unique()

        # Use correlations from Q1 if available, otherwise default to 0.5
        if contestant_correlations is None:
            contestant_meta = {name: 0.5 for name in contestants}
        else:
            contestant_meta = {name: contestant_correlations.get(name, 0.5)
                             for name in contestants}

        particles = [Particle(contestant_meta) for _ in range(CONFIG['N_PARTICLES'])]

        weeks = sorted(season_df['week'].unique())

        for week in tqdm(weeks, desc=f"  S{season}", leave=False):
            week_df = season_df[season_df['week'] == week]

            # --- Step A: Particle propagation and validation ---
            valid_particles = []
            for p in particles:
                p_next = copy.deepcopy(p)
                if p_next.step(week_df, season):
                    valid_particles.append(p_next)

            # --- Step B: Credibility assessment ---
            survival_rate = len(valid_particles) / len(particles) if particles else 0
            self.credibility_scores.append({
                'season': season,
                'week': week,
                'survival_rate': survival_rate,
                'n_valid': len(valid_particles),
                'n_total': len(particles)
            })

            # --- Step C: Handle particle depletion ---
            if survival_rate < CONFIG['MIN_SURVIVAL_RATE']:
                print(f"    Warning: S{season} W{week} - Low survival rate ({survival_rate:.2%})")
                # Emergency reset: reinitialize with noise
                if len(valid_particles) > 0:
                    valid_particles = copy.deepcopy(particles)
                else:
                    # Complete reset
                    particles = [Particle(contestant_meta) for _ in range(CONFIG['N_PARTICLES'])]
                    continue

            # --- Step D: Result aggregation (State estimation) ---
            current_estimates = {}
            for p in valid_particles:
                if not p.history:
                    continue
                last_snapshot = p.history[-1]
                for rec in last_snapshot:
                    name = rec['celebrity']
                    if name not in current_estimates:
                        current_estimates[name] = {'fan': [], 'base': [], 'casual': []}
                    current_estimates[name]['fan'].append(rec['fan_share'])
                    current_estimates[name]['base'].append(rec['base_share_est'])
                    current_estimates[name]['casual'].append(rec['casual_share_est'])

            # Save mean and confidence intervals
            for name, values in current_estimates.items():
                self.results.append({
                    'season': season,
                    'week': week,
                    'celebrity': name,
                    'fan_share_mean': np.mean(values['fan']),
                    'fan_share_std': np.std(values['fan']),
                    'base_component': np.mean(values['base']),
                    'casual_component': np.mean(values['casual']),
                    'model_confidence': survival_rate
                })

            # --- Step E: Resampling ---
            if len(valid_particles) > 0:
                indices = np.random.choice(len(valid_particles), CONFIG['N_PARTICLES'], replace=True)
                particles = [copy.deepcopy(valid_particles[i]) for i in indices]

        print(f"  Completed Season {season}")

    def run_all(self, contestant_correlations=None):
        """
        Run particle filter for all seasons

        Args:
            contestant_correlations: Optional dict from Q1 analysis

        Returns:
            DataFrame with all results
        """
        print("\n[Step 2] Running particle filter for all seasons...")
        seasons = sorted(self.df['season'].unique())

        for season in seasons:
            season_df = self.df[self.df['season'] == season]
            self.run_season_estimation(season_df, contestant_correlations)

        print("\n[Step 3] Aggregating results...")
        results_df = pd.DataFrame(self.results)
        credibility_df = pd.DataFrame(self.credibility_scores)

        print(f"  Total estimates: {len(results_df)}")
        print(f"  Average model confidence: {results_df['model_confidence'].mean():.2%}")

        return results_df, credibility_df

    def save_results(self, results_df, credibility_df):
        """Save results to CSV files"""
        print("\n[Step 4] Saving results...")

        results_path = os.path.join(OUTPUT_DIR, 'dual_source_estimates.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"  Saved estimates to: {results_path}")

        credibility_path = os.path.join(OUTPUT_DIR, 'model_credibility.csv')
        credibility_df.to_csv(credibility_path, index=False, encoding='utf-8-sig')
        print(f"  Saved credibility scores to: {credibility_path}")

        return results_path, credibility_path


# ==========================================
# 4. Visualization and Evaluation
# ==========================================
class ModelEvaluator:
    """
    Visualization and evaluation tools for model results
    """

    def __init__(self, paths):
        """
        Initialize evaluator with output paths

        Args:
            paths: Dictionary with keys 'prediction', 'fan_support', 'analysis'
        """
        self.paths = paths

    def plot_confidence_heatmap(self, credibility_df, save=True):
        """
        Plot model credibility heatmap across seasons and weeks

        Args:
            credibility_df: DataFrame with credibility scores
            save: Whether to save the plot
        """
        print("\n[Visualization 1] Plotting model credibility heatmap...")

        # Pivot data for heatmap
        pivot = credibility_df.pivot_table(
            index='season',
            columns='week',
            values='survival_rate',
            aggfunc='mean'
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot heatmap
        sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1,
                   annot=False, fmt='.2f',
                   cbar_kws={'label': 'Particle Survival Rate'},
                   ax=ax, linewidths=0.5, linecolor='white')

        ax.set_title('Model Credibility Heatmap\n(Particle Survival Rate by Season and Week)',
                    fontweight='bold', pad=20)
        ax.set_xlabel('Week', fontweight='bold')
        ax.set_ylabel('Season', fontweight='bold')

        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['analysis'], 'model_credibility_heatmap.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_latent_dynamics(self, results_df, celebrity_name, save=True):
        """
        Plot dual-source decomposition for a specific contestant

        Args:
            results_df: DataFrame with estimation results
            celebrity_name: Name of contestant to plot
            save: Whether to save the plot
        """
        print(f"\n[Visualization 2] Plotting latent dynamics for {celebrity_name}...")

        # Filter data
        sub = results_df[results_df['celebrity'] == celebrity_name].sort_values('week')

        if len(sub) == 0:
            print(f"  Warning: No data found for {celebrity_name}")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Stackplot for base and casual components
        ax.stackplot(sub['week'],
                    sub['base_component'],
                    sub['casual_component'],
                    labels=['Base Support (Loyal Fans)', 'Casual Support (Performance-driven)'],
                    colors=[COLORS['base'], COLORS['casual']],
                    alpha=0.7)

        # Total line
        total = sub['base_component'] + sub['casual_component']
        ax.plot(sub['week'], total, 'k--', linewidth=2, label='Total Raw Strength', alpha=0.8)

        ax.set_title(f'Dual-Source Dynamics: {celebrity_name}\n(Base vs Casual Support Over Time)',
                    fontweight='bold', pad=20)
        ax.set_xlabel('Week', fontweight='bold')
        ax.set_ylabel('Vote Share Component', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            safe_name = celebrity_name.replace(' ', '_').replace('/', '_')
            out_path = os.path.join(self.paths['fan_support'], f'latent_dynamics_{safe_name}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()


    def plot_top_contestants_comparison(self, results_df, top_n=10, save=True):
        """
        Compare base vs casual support for top contestants

        Args:
            results_df: DataFrame with estimation results
            top_n: Number of top contestants to show
            save: Whether to save the plot
        """
        print(f"\n[Visualization 3] Plotting top {top_n} contestants comparison...")

        # Calculate average components per contestant
        contestant_avg = results_df.groupby('celebrity').agg({
            'base_component': 'mean',
            'casual_component': 'mean',
            'fan_share_mean': 'mean'
        }).reset_index()

        # Get top contestants by total fan share
        top_contestants = contestant_avg.nlargest(top_n, 'fan_share_mean')

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Horizontal bar chart
        y_pos = np.arange(len(top_contestants))

        ax.barh(y_pos, top_contestants['base_component'],
               label='Base Support', color=COLORS['base'], alpha=0.8)
        ax.barh(y_pos, top_contestants['casual_component'],
               left=top_contestants['base_component'],
               label='Casual Support', color=COLORS['casual'], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_contestants['celebrity'])
        ax.invert_yaxis()
        ax.set_xlabel('Average Vote Share Component', fontweight='bold')
        ax.set_title(f'Top {top_n} Contestants: Base vs Casual Support',
                    fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['fan_support'], f'top_{top_n}_contestants_comparison.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()



    def plot_confidence_over_time(self, credibility_df, save=True):
        """
        Plot average model confidence over weeks
        """
        print("\n[Visualization 4] Plotting confidence over time...")

        # Calculate average survival rate per week across all seasons
        week_avg = credibility_df.groupby('week').agg({
            'survival_rate': ['mean', 'std']
        }).reset_index()
        week_avg.columns = ['week', 'mean', 'std']

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot with confidence band
        ax.plot(week_avg['week'], week_avg['mean'],
               color=COLORS['primary'], linewidth=2, marker='o', markersize=6)
        ax.fill_between(week_avg['week'],
                       week_avg['mean'] - week_avg['std'],
                       week_avg['mean'] + week_avg['std'],
                       color=COLORS['primary'], alpha=0.2)

        ax.set_xlabel('Week', fontweight='bold')
        ax.set_ylabel('Average Particle Survival Rate', fontweight='bold')
        ax.set_title('Model Confidence Over Competition Weeks\n(Average Across All Seasons)',
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['analysis'], 'confidence_over_time.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_fan_support_trend_season(self, results_df, season=15, save=True):
        """
        Plot fan support trend for a specific season (similar to fan_percent_estimation style)

        Args:
            results_df: DataFrame with estimation results
            season: Season number to plot
            save: Whether to save the plot
        """
        print(f"\n[Visualization 6] Plotting fan support trend for Season {season}...")

        # Filter data for the specified season
        season_data = results_df[results_df['season'] == season]

        if len(season_data) == 0:
            print(f"  Warning: No data found for Season {season}")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get all contestants in this season
        celebrities = season_data['celebrity'].unique()

        # Plot each contestant's trend
        for celebrity in celebrities:
            celeb_data = season_data[season_data['celebrity'] == celebrity].sort_values('week')

            weeks = celeb_data['week'].values
            fan_mean = celeb_data['fan_share_mean'].values
            fan_std = celeb_data['fan_share_std'].values

            # Plot mean line
            ax.plot(weeks, fan_mean, marker='o', label=celebrity, linewidth=2, markersize=6)

            # Plot confidence interval (mean ± 1 std)
            ax.fill_between(weeks,
                           fan_mean - fan_std,
                           fan_mean + fan_std,
                           alpha=0.2)

        ax.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fan Support Share', fontsize=12, fontweight='bold')
        ax.set_title(f'Season {season} - Fan Support Trend with Confidence Intervals\n(Dual-Source Particle Filter Estimates)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['fan_support'], f'fan_support_trend_season{season}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_prediction_interval_comparison(self, old_results_path, new_results_df, season=15, top_n=3, save=True):
        """
        Compare prediction intervals by overlaying old and new confidence intervals

        Args:
            old_results_path: Path to fan_est_cache.csv (old method results)
            new_results_df: DataFrame with new dual-source estimates
            season: Season to compare
            top_n: Number of top contestants to show (default 3 for clarity)
            save: Whether to save the plot
        """
        print(f"\n[Visualization 7] Comparing prediction intervals for Season {season}...")

        # Load old results
        if not os.path.exists(old_results_path):
            print(f"  Warning: Old results file not found: {old_results_path}")
            return

        old_results = pd.read_csv(old_results_path, encoding='utf-8-sig')

        # Filter both datasets for the specified season
        old_season = old_results[old_results['season'] == season]
        new_season = new_results_df[new_results_df['season'] == season]

        if len(old_season) == 0 or len(new_season) == 0:
            print(f"  Warning: Insufficient data for Season {season}")
            return

        # Get top contestants by average fan support (from new method)
        top_contestants = new_season.groupby('celebrity')['fan_share_mean'].mean().nlargest(top_n).index.tolist()

        # Create subplots for each contestant
        fig, axes = plt.subplots(top_n, 1, figsize=(14, 5*top_n))
        if top_n == 1:
            axes = [axes]

        for idx, celebrity in enumerate(top_contestants):
            ax = axes[idx]

            # Get data for this contestant
            old_celeb = old_season[old_season['celebrity'] == celebrity].sort_values('week')
            new_celeb = new_season[new_season['celebrity'] == celebrity].sort_values('week')

            if len(old_celeb) == 0 or len(new_celeb) == 0:
                continue

            # Old method data
            old_weeks = old_celeb['week'].values
            old_mean = old_celeb['fan_percent_mean'].values
            old_std = old_celeb['fan_percent_std'].values

            # New method data
            new_weeks = new_celeb['week'].values
            new_mean = new_celeb['fan_share_mean'].values
            new_std = new_celeb['fan_share_std'].values

            # Plot old method (Monte Carlo) - in red/pink
            ax.plot(old_weeks, old_mean, 'o-', color=COLORS['danger'],
                   linewidth=2, markersize=6, label='Monte Carlo (Old)', alpha=0.7)
            ax.fill_between(old_weeks,
                           old_mean - old_std,
                           old_mean + old_std,
                           color=COLORS['danger'], alpha=0.2,
                           label='Old CI (±1σ)')

            # Plot new method (Particle Filter) - in blue
            ax.plot(new_weeks, new_mean, 's-', color=COLORS['primary'],
                   linewidth=2, markersize=6, label='Particle Filter (New)', alpha=0.7)
            ax.fill_between(new_weeks,
                           new_mean - new_std,
                           new_mean + new_std,
                           color=COLORS['primary'], alpha=0.2,
                           label='New CI (±1σ)')

            # Calculate average interval width improvement
            avg_old_width = np.mean(old_std) * 2
            avg_new_width = np.mean(new_std) * 2
            improvement = ((avg_old_width - avg_new_width) / avg_old_width) * 100

            ax.set_xlabel('Week', fontweight='bold')
            ax.set_ylabel('Fan Support Share', fontweight='bold')
            ax.set_title(f'{celebrity}\nInterval Width: Old={avg_old_width:.4f}, New={avg_new_width:.4f} (Improvement: {improvement:.1f}%)',
                        fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Season {season}: Confidence Interval Comparison (Old vs New Method)\nRed=Monte Carlo, Blue=Particle Filter',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['prediction'], f'interval_overlay_season{season}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_single_contestant_interval_comparison(self, old_results_path, new_results_df,
                                                   celebrity_name, season=None, save=True):
        """
        Compare prediction intervals for a single contestant (more detailed view)

        Args:
            old_results_path: Path to fan_est_cache.csv (old method results)
            new_results_df: DataFrame with new dual-source estimates
            celebrity_name: Name of contestant to compare
            season: Optional season filter (if None, shows all seasons)
            save: Whether to save the plot
        """
        print(f"\n[Visualization 8] Comparing intervals for {celebrity_name}...")

        # Load old results
        if not os.path.exists(old_results_path):
            print(f"  Warning: Old results file not found: {old_results_path}")
            return

        old_results = pd.read_csv(old_results_path, encoding='utf-8-sig')

        # Filter for the specific contestant
        old_celeb = old_results[old_results['celebrity'] == celebrity_name]
        new_celeb = new_results_df[new_results_df['celebrity'] == celebrity_name]

        if season is not None:
            old_celeb = old_celeb[old_celeb['season'] == season]
            new_celeb = new_celeb[new_celeb['season'] == season]
            title_suffix = f' (Season {season})'
        else:
            title_suffix = ' (All Seasons)'

        if len(old_celeb) == 0 or len(new_celeb) == 0:
            print(f"  Warning: No data found for {celebrity_name}")
            return

        old_celeb = old_celeb.sort_values('week')
        new_celeb = new_celeb.sort_values('week')

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Old method data
        old_weeks = old_celeb['week'].values
        old_mean = old_celeb['fan_percent_mean'].values
        old_std = old_celeb['fan_percent_std'].values

        # New method data
        new_weeks = new_celeb['week'].values
        new_mean = new_celeb['fan_share_mean'].values
        new_std = new_celeb['fan_share_std'].values

        # Plot old method (Monte Carlo) - in red/pink
        ax.plot(old_weeks, old_mean, 'o-', color=COLORS['danger'],
               linewidth=3, markersize=8, label='Monte Carlo (Old)', alpha=0.8, zorder=3)
        ax.fill_between(old_weeks,
                       old_mean - old_std,
                       old_mean + old_std,
                       color=COLORS['danger'], alpha=0.25,
                       label='Old Confidence Interval (±1σ)', zorder=1)

        # Plot new method (Particle Filter) - in blue
        ax.plot(new_weeks, new_mean, 's-', color=COLORS['primary'],
               linewidth=3, markersize=8, label='Particle Filter (New)', alpha=0.8, zorder=3)
        ax.fill_between(new_weeks,
                       new_mean - new_std,
                       new_mean + new_std,
                       color=COLORS['primary'], alpha=0.25,
                       label='New Confidence Interval (±1σ)', zorder=2)

        # Calculate statistics
        avg_old_width = np.mean(old_std) * 2
        avg_new_width = np.mean(new_std) * 2
        improvement = ((avg_old_width - avg_new_width) / avg_old_width) * 100

        ax.set_xlabel('Week', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fan Support Share', fontsize=13, fontweight='bold')
        ax.set_title(f'Confidence Interval Comparison: {celebrity_name}{title_suffix}\n' +
                    f'Average Interval Width - Old: {avg_old_width:.4f}, New: {avg_new_width:.4f} ' +
                    f'(Improvement: {improvement:.1f}%)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            safe_name = celebrity_name.replace(' ', '_').replace('/', '_')
            season_str = f'_season{season}' if season else '_all'
            out_path = os.path.join(self.paths['prediction'], f'interval_comparison_{safe_name}{season_str}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
            print(f"  Improvement: {improvement:.1f}%")
        else:
            plt.show()

        plt.close()


# ==========================================
# 5. Main Execution
# ==========================================
def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("STARTING DUAL-SOURCE PARTICLE FILTER ANALYSIS")
    print("=" * 60)

    # Initialize estimator with preprocessed data
    # Using the data from fan_percent_estimation.py
    data_path = 'fan_est_cache.csv'
    
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found: {data_path}")
        print("Please run fan_percent_estimation.py first to generate the cache file.")
        return

    estimator = DualSourceEstimator(data_path)
    estimator.load_data()

    # Create organized directory structure for this run
    seasons_list = sorted(estimator.df['season'].unique())
    paths = create_run_directories(seasons_list, CONFIG['N_PARTICLES'])

    print(f"\nOutput directories created:")
    print(f"  Base: {paths['base']}")
    print(f"  Prediction: {paths['prediction']}")
    print(f"  Fan Support: {paths['fan_support']}")
    print(f"  Analysis: {paths['analysis']}")

    # Run particle filter for all seasons
    results_df, credibility_df = estimator.run_all()

    # Save results to base directory
    results_path = os.path.join(paths['base'], 'dual_source_estimates.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[Results] Saved estimates to: {results_path}")

    credibility_path = os.path.join(paths['base'], 'model_credibility.csv')
    credibility_df.to_csv(credibility_path, index=False, encoding='utf-8-sig')
    print(f"[Results] Saved credibility scores to: {credibility_path}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    evaluator = ModelEvaluator(paths)

    # 1. Model credibility heatmap
    evaluator.plot_confidence_heatmap(credibility_df)

    # 2. Confidence over time
    evaluator.plot_confidence_over_time(credibility_df)

    # 3. Top contestants comparison
    evaluator.plot_top_contestants_comparison(results_df, top_n=15)

    # 4. Sample latent dynamics for top 5 contestants
    print("\n[Visualization 5] Plotting latent dynamics for top contestants...")
    top_5 = results_df.groupby('celebrity')['fan_share_mean'].mean().nlargest(5).index
    for celebrity in top_5:
        evaluator.plot_latent_dynamics(results_df, celebrity)

    # 5. Fan support trend for Season 15 (similar to fan_percent_estimation style)
    evaluator.plot_fan_support_trend_season(results_df, season=15)

    # 6. Prediction interval comparison - Top 3 contestants overlay
    evaluator.plot_prediction_interval_comparison(data_path, results_df, season=15, top_n=3)

    # 7. Single contestant detailed comparison (top contestant from season 15)
    season_15 = results_df[results_df['season'] == 15]
    if len(season_15) > 0:
        top_contestant = season_15.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df,
                                                            top_contestant, season=15)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
