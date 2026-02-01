"""
Factor Attribution Analysis for DWTS (Question 3)
==================================================
This module implements a dual-model attribution analysis to understand
what factors drive success in DWTS from two perspectives:
1. Judge Preference (Technical Merit)
2. Fan Preference (Popularity)

Key Features:
- Linear Mixed Effects Models (LMM) with partner as random effect
- Four factor analysis: Partner, Industry, Age, Region
- Forest plot for effect comparison
- Partner efficiency analysis
- Case study mining (robbed vs overrated contestants)

Author: DWTS Analysis Team
Date: 2026-02-02
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy import stats
import math

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'RAW_DATA_PATH': '2026_MCM_Problem_C_Data.csv',
    'JUDGE_SCORES_PATH': 'fan_est_cache.csv',  # Has judge scores in long format
    'FAN_VOTES_PATH': 'Q1/dual_source_analysis/seasons_1_to_34_particles_500_absolute/dual_source_estimates_fused.csv',
    'OUTPUT_DIR': 'Q3/results',
    'FIGURE_DIR': 'Q3/figures',
    'MIN_APPEARANCES': 3,  # Minimum weeks for inclusion
}

# Visualization Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'judge': '#2E86AB',      # Blue for judges
    'fan': '#F18F01',        # Orange for fans
    'positive': '#06A77D',   # Green for positive effects
    'negative': '#C73E1D',   # Red for negative effects
    'neutral': '#6C757D',    # Gray for neutral
    'highlight': '#D4AF37',  # Gold for highlights
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
print("DWTS FACTOR ATTRIBUTION ANALYSIS (QUESTION 3)")
print("=" * 70)
print(f"Configuration:")
print(f"  - Raw Data: {CONFIG['RAW_DATA_PATH']}")
print(f"  - Judge Scores: {CONFIG['JUDGE_SCORES_PATH']}")
print(f"  - Fan Votes: {CONFIG['FAN_VOTES_PATH']}")
print(f"  - Output Directory: {CONFIG['OUTPUT_DIR']}")
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
ensure_dir(CONFIG['OUTPUT_DIR'])
ensure_dir(CONFIG['FIGURE_DIR'])


# ==========================================
# Feature Engineering Functions
# ==========================================
def map_industry_to_group(industry):
    """
    Map detailed industry to 5 major groups

    Groups:
    - Athlete: Sports professionals
    - Actor: Film/TV actors
    - Musician: Singers, rappers, musicians
    - RealityStar: Reality TV personalities
    - Other: Hosts, politicians, etc.
    """
    if pd.isna(industry):
        return 'Other'

    ind = str(industry).lower()

    # Athlete keywords
    if any(x in ind for x in ['athlete', 'nfl', 'nba', 'mlb', 'nhl', 'olympian',
                               'gymnast', 'figure skater', 'boxer', 'wrestler',
                               'football', 'basketball', 'baseball', 'hockey']):
        return 'Athlete'

    # Actor keywords
    if any(x in ind for x in ['actor', 'actress', 'movie', 'film', 'television']):
        return 'Actor'

    # Musician keywords
    if any(x in ind for x in ['singer', 'rapper', 'musician', 'pop star',
                               'country singer', 'rock', 'band']):
        return 'Musician'

    # Reality Star keywords
    if any(x in ind for x in ['reality', 'bachelor', 'bachelorette', 'housewife',
                               'kardashian', 'survivor']):
        return 'RealityStar'

    return 'Other'


def map_state_to_region(state):
    """
    Map US states to 4 major census regions + International

    Regions:
    - Northeast: New England + Mid-Atlantic
    - Midwest: Great Lakes + Great Plains
    - South: Southeast + Southwest
    - West: Mountain + Pacific
    - International: Non-US
    """
    if pd.isna(state):
        return 'Unknown'

    state = str(state).strip()

    # Northeast
    northeast = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island',
                 'Vermont', 'New Jersey', 'New York', 'Pennsylvania']

    # Midwest
    midwest = ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin',
               'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska',
               'North Dakota', 'South Dakota']

    # South
    south = ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina',
             'South Carolina', 'Virginia', 'West Virginia', 'Alabama',
             'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana',
             'Oklahoma', 'Texas']

    # West
    west = ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico',
            'Utah', 'Wyoming', 'Alaska', 'California', 'Hawaii', 'Oregon',
            'Washington']

    if state in northeast:
        return 'Northeast'
    elif state in midwest:
        return 'Midwest'
    elif state in south:
        return 'South'
    elif state in west:
        return 'West'
    else:
        return 'International'


# ==========================================
# Main Analysis Class
# ==========================================
class DWTS_Factor_Analyzer:
    """
    Main class for factor attribution analysis

    This class implements a dual-model approach to understand what factors
    drive success in DWTS from both judge and fan perspectives.
    """

    def __init__(self, raw_data_path, judge_scores_path, fan_votes_path):
        """
        Initialize the analyzer

        Args:
            raw_data_path: Path to raw DWTS data (demographics)
            judge_scores_path: Path to judge scores cache (long format)
            fan_votes_path: Path to Q1 fan vote estimates (long format)
        """
        print("\n[Initialization] Loading data...")

        # Load raw data (demographics - wide format)
        self.raw_df = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        print(f"  Loaded raw data: {self.raw_df.shape}")

        # Load judge scores (long format with week-by-week data)
        self.judge_df = pd.read_csv(judge_scores_path, encoding='utf-8-sig')
        print(f"  Loaded judge scores: {self.judge_df.shape}")

        # Load fan vote estimates from Q1 (long format)
        self.fan_df = pd.read_csv(fan_votes_path, encoding='utf-8-sig')
        print(f"  Loaded fan votes: {self.fan_df.shape}")

        # Initialize containers
        self.merged_df = None
        self.model_judge = None
        self.model_fan = None
        self.effects_df = None
        self.partner_df = None
        self.case_studies = None

    def preprocess_and_merge(self):
        """
        Data fusion and feature engineering

        Steps:
        1. Merge judge scores and fan votes (both in long format)
        2. Aggregate to get per-contestant season averages
        3. Merge with demographics from raw data
        4. Apply feature engineering (industry grouping, region mapping)
        5. Standardize target variables
        """
        print("\n[Step 1] Preprocessing and merging data...")

        # --- A. Merge judge scores and fan votes ---
        # Both are in long format (one row per contestant per week)
        combined_long = pd.merge(
            self.judge_df[['season', 'week', 'celebrity', 'judge_score']],
            self.fan_df[['season', 'week', 'celebrity', 'fan_share_mean', 'fan_votes_mean']],
            on=['season', 'week', 'celebrity'],
            how='inner'
        )

        print(f"  Combined long format data: {combined_long.shape}")

        # --- B. Aggregate to season-level statistics ---
        season_stats = combined_long.groupby(['season', 'celebrity']).agg({
            'judge_score': 'mean',         # Average judge score
            'fan_share_mean': 'mean',      # Average fan share
            'fan_votes_mean': 'mean',      # Average absolute votes
            'week': 'count'                # Number of weeks survived
        }).reset_index()

        season_stats.columns = ['season', 'celebrity', 'judge_score_mean',
                               'fan_share_mean', 'fan_votes_mean', 'weeks_survived']

        print(f"  Aggregated season statistics: {season_stats.shape}")

        # --- C. Get demographics from raw data (one row per contestant) ---
        demographics = self.raw_df[['season', 'celebrity_name', 'ballroom_partner',
                                    'celebrity_industry', 'celebrity_age_during_season',
                                    'celebrity_homestate']].copy()

        # --- D. Merge season stats with demographics ---
        self.merged_df = pd.merge(
            season_stats,
            demographics,
            left_on=['season', 'celebrity'],
            right_on=['season', 'celebrity_name'],
            how='inner'
        )

        # Drop duplicate column
        self.merged_df = self.merged_df.drop(columns=['celebrity_name'])

        print(f"  Merged with demographics: {self.merged_df.shape}")

        # --- E. Feature Engineering ---
        print("  Applying feature engineering...")

        # 1. Industry grouping
        self.merged_df['Industry_Group'] = self.merged_df['celebrity_industry'].apply(
            map_industry_to_group)

        # 2. Region mapping
        self.merged_df['Region_Group'] = self.merged_df['celebrity_homestate'].apply(
            map_state_to_region)

        # 3. Age squared (for non-linear effects)
        self.merged_df['Age_Squared'] = self.merged_df['celebrity_age_during_season'] ** 2

        # --- F. Standardize target variables (Z-Score) ---
        scaler = StandardScaler()

        # Judge preference: standardized judge score
        self.merged_df['Y_Judge'] = scaler.fit_transform(
            self.merged_df[['judge_score_mean']])

        # Fan preference: standardized fan share
        self.merged_df['Y_Fan'] = scaler.fit_transform(
            self.merged_df[['fan_share_mean']])

        # --- G. Filter out contestants with too few appearances ---
        min_weeks = CONFIG['MIN_APPEARANCES']
        before_filter = len(self.merged_df)
        self.merged_df = self.merged_df[self.merged_df['weeks_survived'] >= min_weeks]
        after_filter = len(self.merged_df)

        print(f"  Filtered out {before_filter - after_filter} contestants with < {min_weeks} weeks")
        print(f"  Final sample size: {after_filter} contestants")

        # --- H. Summary statistics ---
        print("\n  Data Summary:")
        print(f"    Industry distribution:")
        print(self.merged_df['Industry_Group'].value_counts().to_string())
        print(f"\n    Region distribution:")
        print(self.merged_df['Region_Group'].value_counts().to_string())
        print(f"\n    Age range: {self.merged_df['celebrity_age_during_season'].min():.0f} - "
              f"{self.merged_df['celebrity_age_during_season'].max():.0f}")

        return self

    def build_mixed_effects_models(self):
        """
        Build Linear Mixed Effects Models (LMM)

        Fixed Effects: Age, Age^2, Industry, Region
        Random Effects: Pro Partner (accounts for partner skill variation)

        Two parallel models:
        1. Judge Model: Y_Judge ~ Fixed + (1|Partner)
        2. Fan Model: Y_Fan ~ Fixed + (1|Partner)
        """
        print("\n[Step 2] Building Mixed Effects Models...")

        # Formula: Y ~ Age + Age^2 + Industry + Region + (1|Partner)
        formula = ("~ celebrity_age_during_season + Age_Squared + "
                   "C(Industry_Group) + C(Region_Group)")

        try:
            # 1. Judge Preference Model
            print("  Building Judge Preference Model...")
            self.model_judge = smf.mixedlm(
                "Y_Judge" + formula,
                self.merged_df,
                groups=self.merged_df["ballroom_partner"]
            ).fit(reml=True)

            # 2. Fan Preference Model
            print("  Building Fan Preference Model...")
            self.model_fan = smf.mixedlm(
                "Y_Fan" + formula,
                self.merged_df,
                groups=self.merged_df["ballroom_partner"]
            ).fit(reml=True)

            print("  ✓ Models converged successfully")

            # Print model summaries
            print("\n  Judge Model Summary:")
            print(f"    Log-Likelihood: {self.model_judge.llf:.2f}")
            print(f"    AIC: {self.model_judge.aic:.2f}")

            print("\n  Fan Model Summary:")
            print(f"    Log-Likelihood: {self.model_fan.llf:.2f}")
            print(f"    AIC: {self.model_fan.aic:.2f}")

        except Exception as e:
            print(f"  ✗ Model fitting failed: {e}")
            raise

        return self

    def extract_and_compare_effects(self):
        """
        Extract fixed effects from both models and prepare for comparison

        Returns:
            DataFrame with coefficients and confidence intervals for both models
        """
        print("\n[Step 3] Extracting and comparing effects...")

        # Extract fixed effects (drop intercept for cleaner visualization)
        params_j = self.model_judge.params.drop('Intercept', errors='ignore')
        conf_j = self.model_judge.conf_int().drop('Intercept', errors='ignore')

        params_f = self.model_fan.params.drop('Intercept', errors='ignore')
        conf_f = self.model_fan.conf_int().drop('Intercept', errors='ignore')

        # Build comparison dataframe
        effects = []
        for idx in params_j.index:
            # Clean variable names for display
            clean_name = (idx.replace("C(Industry_Group)[T.", "")
                            .replace("C(Region_Group)[T.", "")
                            .replace("]", "")
                            .replace("celebrity_age_during_season", "Age")
                            .replace("Age_Squared", "Age²"))

            # Judge effects
            effects.append({
                'Factor': clean_name,
                'Model': 'Judge Preference',
                'Coefficient': params_j[idx],
                'CI_Lower': conf_j.loc[idx, 0],
                'CI_Upper': conf_j.loc[idx, 1],
                'Significant': not (conf_j.loc[idx, 0] <= 0 <= conf_j.loc[idx, 1])
            })

            # Fan effects
            effects.append({
                'Factor': clean_name,
                'Model': 'Fan Preference',
                'Coefficient': params_f[idx],
                'CI_Lower': conf_f.loc[idx, 0],
                'CI_Upper': conf_f.loc[idx, 1],
                'Significant': not (conf_f.loc[idx, 0] <= 0 <= conf_f.loc[idx, 1])
            })

        self.effects_df = pd.DataFrame(effects)

        # Calculate effect size differences
        judge_effects = self.effects_df[self.effects_df['Model'] == 'Judge Preference'].set_index('Factor')
        fan_effects = self.effects_df[self.effects_df['Model'] == 'Fan Preference'].set_index('Factor')

        print(f"  Extracted {len(params_j)} fixed effects from each model")
        print(f"\n  Largest Judge-Fan Gaps:")

        gaps = (judge_effects['Coefficient'] - fan_effects['Coefficient']).abs().sort_values(ascending=False)
        for factor, gap in gaps.head(5).items():
            print(f"    {factor}: {gap:.3f}")

        return self

    def analyze_pro_partners(self):
        """
        Analyze pro partner effects (Random Effects)

        Extracts BLUPs (Best Linear Unbiased Predictors) to quantify
        each partner's contribution to judge scores and fan votes.

        Returns:
            DataFrame with partner effects for both models
        """
        print("\n[Step 4] Analyzing pro partner effects...")

        # Extract random effects (BLUPs)
        re_judge = self.model_judge.random_effects
        re_fan = self.model_fan.random_effects

        # Build partner dataframe
        partners = []
        for name in re_judge.keys():
            partners.append({
                'Partner': name,
                'Judge_Effect': re_judge[name].values[0],  # Effect on judge scores
                'Fan_Effect': re_fan[name].values[0],      # Effect on fan votes
            })

        self.partner_df = pd.DataFrame(partners)

        # Calculate combined effect (Euclidean distance from origin)
        self.partner_df['Combined_Effect'] = np.sqrt(
            self.partner_df['Judge_Effect']**2 + self.partner_df['Fan_Effect']**2
        )

        # Identify partner types
        self.partner_df['Type'] = 'Balanced'
        self.partner_df.loc[
            (self.partner_df['Judge_Effect'] > 0.2) & (self.partner_df['Fan_Effect'] < 0.1),
            'Type'] = 'Technical Specialist'
        self.partner_df.loc[
            (self.partner_df['Fan_Effect'] > 0.2) & (self.partner_df['Judge_Effect'] < 0.1),
            'Type'] = 'Crowd Pleaser'
        self.partner_df.loc[
            (self.partner_df['Judge_Effect'] > 0.2) & (self.partner_df['Fan_Effect'] > 0.2),
            'Type'] = 'Elite (Both)'

        # Sort by combined effect
        self.partner_df = self.partner_df.sort_values('Combined_Effect', ascending=False)

        print(f"  Analyzed {len(self.partner_df)} pro partners")
        print(f"\n  Top 5 Partners (Combined Effect):")
        print(self.partner_df.head(5)[['Partner', 'Judge_Effect', 'Fan_Effect', 'Type']].to_string(index=False))

        return self

    def mine_case_studies(self):
        """
        Mine typical case studies based on model residuals

        Identifies three types of contestants:
        1. Robbed (High judge score, low fan votes)
        2. Overrated (Low judge score, high fan votes)
        3. Perfect Package (High both, low residuals)
        """
        print("\n[Step 5] Mining case studies...")

        # Calculate ranks
        self.merged_df['Rank_Judge'] = self.merged_df['Y_Judge'].rank(ascending=False)
        self.merged_df['Rank_Fan'] = self.merged_df['Y_Fan'].rank(ascending=False)
        self.merged_df['Rank_Gap'] = self.merged_df['Rank_Judge'] - self.merged_df['Rank_Fan']

        # Calculate model predictions and residuals
        self.merged_df['Pred_Judge'] = self.model_judge.fittedvalues
        self.merged_df['Pred_Fan'] = self.model_fan.fittedvalues
        self.merged_df['Residual_Judge'] = self.merged_df['Y_Judge'] - self.merged_df['Pred_Judge']
        self.merged_df['Residual_Fan'] = self.merged_df['Y_Fan'] - self.merged_df['Pred_Fan']

        # Identify cases
        # 1. Robbed: Large positive rank gap (judge rank >> fan rank)
        robbed = self.merged_df.nlargest(3, 'Rank_Gap')

        # 2. Overrated: Large negative rank gap (fan rank >> judge rank)
        overrated = self.merged_df.nsmallest(3, 'Rank_Gap')

        # 3. Perfect Package: High both, small residuals
        self.merged_df['Total_Residual'] = (self.merged_df['Residual_Judge'].abs() +
                                            self.merged_df['Residual_Fan'].abs())
        high_performers = self.merged_df[
            (self.merged_df['Y_Judge'] > 1) & (self.merged_df['Y_Fan'] > 1)
        ]
        perfect = high_performers.nsmallest(3, 'Total_Residual')

        self.case_studies = {
            'robbed': robbed,
            'overrated': overrated,
            'perfect': perfect
        }

        print(f"\n  Case Studies Identified:")
        print(f"\n  Robbed (High Technical, Low Popularity):")
        for _, row in robbed.iterrows():
            print(f"    {row['celebrity']} (S{row['season']:.0f}): "
                  f"Judge Rank={row['Rank_Judge']:.0f}, Fan Rank={row['Rank_Fan']:.0f}, "
                  f"Gap={row['Rank_Gap']:.0f}")

        print(f"\n  Overrated (Low Technical, High Popularity):")
        for _, row in overrated.iterrows():
            print(f"    {row['celebrity']} (S{row['season']:.0f}): "
                  f"Judge Rank={row['Rank_Judge']:.0f}, Fan Rank={row['Rank_Fan']:.0f}, "
                  f"Gap={row['Rank_Gap']:.0f}")

        print(f"\n  Perfect Package (High Both, Model-Consistent):")
        for _, row in perfect.iterrows():
            print(f"    {row['celebrity']} (S{row['season']:.0f}): "
                  f"Judge={row['Y_Judge']:.2f}, Fan={row['Y_Fan']:.2f}")

        return self

    def visualize_forest_plot(self, save=True):
        """
        Visualization 1: Forest Plot - Core Attribution Analysis

        Shows fixed effects from both models side-by-side with confidence intervals.
        This is the key visualization for comparing judge vs fan preferences.
        """
        print("\n[Visualization 1] Generating Forest Plot...")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Prepare data for plotting
        plot_df = self.effects_df.copy()

        # Sort by absolute coefficient magnitude for better visualization
        factor_order = (plot_df.groupby('Factor')['Coefficient']
                       .apply(lambda x: x.abs().max())
                       .sort_values(ascending=True).index)

        plot_df['Factor'] = pd.Categorical(plot_df['Factor'], categories=factor_order, ordered=True)
        plot_df = plot_df.sort_values('Factor')

        # Create positions for each factor
        factors = plot_df['Factor'].unique()
        y_positions = np.arange(len(factors))

        # Plot for each model
        for i, (model, color) in enumerate([('Judge Preference', COLORS['judge']),
                                            ('Fan Preference', COLORS['fan'])]):
            model_data = plot_df[plot_df['Model'] == model]

            # Offset positions slightly for side-by-side display
            offset = 0.15 if i == 0 else -0.15
            y_pos = y_positions + offset

            # Plot confidence intervals
            for j, (_, row) in enumerate(model_data.iterrows()):
                ax.plot([row['CI_Lower'], row['CI_Upper']], [y_pos[j], y_pos[j]],
                       color=color, linewidth=2, alpha=0.6, zorder=1)

            # Plot point estimates
            ax.scatter(model_data['Coefficient'], y_pos,
                      color=color, s=100, alpha=0.9, label=model,
                      edgecolors='white', linewidths=1.5, zorder=2)

            # Mark significant effects with stars
            sig_data = model_data[model_data['Significant']]
            if len(sig_data) > 0:
                sig_y = [y_pos[j] for j, (_, row) in enumerate(model_data.iterrows())
                        if row['Significant']]
                ax.scatter(sig_data['Coefficient'], sig_y,
                          marker='*', s=200, color='gold', edgecolors='black',
                          linewidths=0.5, zorder=3)

        # Add reference line at 0
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(factors)
        ax.set_xlabel('Standardized Effect Size (β Coefficient)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Factor', fontsize=13, fontweight='bold')
        ax.set_title('Attribution Analysis: What Drives Success in DWTS?\n' +
                    'Judge Preference vs. Fan Preference (Forest Plot)',
                    fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='x')

        # Add note about significance
        ax.text(0.02, 0.98, '★ = Statistically significant (95% CI excludes 0)',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], 'forest_plot_attribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def visualize_partner_scatter(self, save=True):
        """
        Visualization 2: Partner Efficiency Scatter Plot

        Shows each pro partner's effect on judge scores vs fan votes.
        Identifies different partner types (technical specialists, crowd pleasers, elite).
        """
        print("\n[Visualization 2] Generating Partner Scatter Plot...")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Color by partner type
        type_colors = {
            'Elite (Both)': COLORS['highlight'],
            'Technical Specialist': COLORS['judge'],
            'Crowd Pleaser': COLORS['fan'],
            'Balanced': COLORS['neutral']
        }

        # Plot each partner type
        for ptype, color in type_colors.items():
            subset = self.partner_df[self.partner_df['Type'] == ptype]
            if len(subset) > 0:
                ax.scatter(subset['Judge_Effect'], subset['Fan_Effect'],
                          s=200, alpha=0.7, color=color, label=ptype,
                          edgecolors='white', linewidths=1.5)

        # Add reference lines at 0
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Annotate top partners
        top_partners = self.partner_df.head(10)
        for _, row in top_partners.iterrows():
            ax.annotate(row['Partner'],
                       xy=(row['Judge_Effect'], row['Fan_Effect']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        # Formatting
        ax.set_xlabel('Judge Effect (Technical Skill)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fan Effect (Popularity)', fontsize=13, fontweight='bold')
        ax.set_title('Pro Partner Analysis: Technical Skill vs. Crowd Appeal\n' +
                    'Random Effects from Mixed Models',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Add quadrant labels
        ax.text(0.95, 0.95, 'Elite\n(High Both)', transform=ax.transAxes,
               fontsize=10, ha='right', va='top', alpha=0.5, fontweight='bold')
        ax.text(0.95, 0.05, 'Technical\nSpecialist', transform=ax.transAxes,
               fontsize=10, ha='right', va='bottom', alpha=0.5, fontweight='bold')
        ax.text(0.05, 0.95, 'Crowd\nPleaser', transform=ax.transAxes,
               fontsize=10, ha='left', va='top', alpha=0.5, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], 'partner_scatter_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def visualize_case_study_radar(self, save=True):
        """
        Visualization 3: Radar Chart for Typical Cases

        Shows multi-dimensional profiles of typical contestants:
        - Robbed (high technical, low popularity)
        - Overrated (low technical, high popularity)
        - Perfect Package (high both)
        """
        print("\n[Visualization 3] Generating Case Study Radar Charts...")

        # Select one representative from each category
        robbed_case = self.case_studies['robbed'].iloc[0]
        overrated_case = self.case_studies['overrated'].iloc[0]
        perfect_case = self.case_studies['perfect'].iloc[0]

        cases = [
            ('Robbed', robbed_case, COLORS['judge']),
            ('Overrated', overrated_case, COLORS['fan']),
            ('Perfect', perfect_case, COLORS['highlight'])
        ]

        # Define dimensions for radar chart
        categories = ['Technical\nScore', 'Fan\nVotes', 'Weeks\nSurvived',
                     'Age\nAdvantage', 'Industry\nBonus']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

        for idx, (case_type, case, color) in enumerate(cases):
            ax = axes[idx]

            # Normalize values to 0-1 scale for radar chart
            values = [
                (case['Y_Judge'] - self.merged_df['Y_Judge'].min()) /
                (self.merged_df['Y_Judge'].max() - self.merged_df['Y_Judge'].min()),

                (case['Y_Fan'] - self.merged_df['Y_Fan'].min()) /
                (self.merged_df['Y_Fan'].max() - self.merged_df['Y_Fan'].min()),

                (case['weeks_survived'] - self.merged_df['weeks_survived'].min()) /
                (self.merged_df['weeks_survived'].max() - self.merged_df['weeks_survived'].min()),

                1 - abs(case['celebrity_age_during_season'] - 30) / 30,  # Age advantage (peak at 30)

                0.6 if case['Industry_Group'] in ['Athlete', 'Musician'] else 0.4  # Industry bonus
            ]

            # Close the plot
            values += values[:1]

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2.5, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9, fontweight='bold')

            # Set y-axis limits
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=8)

            # Add title with contestant name
            ax.set_title(f"{case_type}\n{case['celebrity']} (S{case['season']:.0f})",
                        fontsize=12, fontweight='bold', pad=20)

            ax.grid(True, alpha=0.3)

        plt.suptitle('Case Study Profiles: Multi-Dimensional Analysis',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            save_path = os.path.join(CONFIG['FIGURE_DIR'], 'case_study_radar_charts.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return self

    def export_results(self):
        """
        Export all analysis results to CSV files
        """
        print("\n[Step 6] Exporting results...")

        # 1. Fixed effects comparison
        effects_path = os.path.join(CONFIG['OUTPUT_DIR'], 'fixed_effects_comparison.csv')
        self.effects_df.to_csv(effects_path, index=False, encoding='utf-8-sig')
        print(f"  Fixed effects: {effects_path}")

        # 2. Partner analysis
        partner_path = os.path.join(CONFIG['OUTPUT_DIR'], 'partner_effects_analysis.csv')
        self.partner_df.to_csv(partner_path, index=False, encoding='utf-8-sig')
        print(f"  Partner effects: {partner_path}")

        # 3. Case studies
        for case_type, case_df in self.case_studies.items():
            case_path = os.path.join(CONFIG['OUTPUT_DIR'], f'case_study_{case_type}.csv')
            case_df.to_csv(case_path, index=False, encoding='utf-8-sig')
            print(f"  Case study ({case_type}): {case_path}")

        # 4. Full merged data with predictions
        full_path = os.path.join(CONFIG['OUTPUT_DIR'], 'full_analysis_data.csv')
        self.merged_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"  Full data: {full_path}")

        return self

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("\n" + "=" * 70)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("=" * 70)

        # Execute all steps in sequence
        self.preprocess_and_merge()
        self.build_mixed_effects_models()
        self.extract_and_compare_effects()
        self.analyze_pro_partners()
        self.mine_case_studies()

        # Generate all visualizations
        self.visualize_forest_plot()
        self.visualize_partner_scatter()
        self.visualize_case_study_radar()

        # Export results
        self.export_results()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nResults saved to: {CONFIG['OUTPUT_DIR']}")
        print(f"Figures saved to: {CONFIG['FIGURE_DIR']}")
        print("=" * 70)

        return self


# ==========================================
# Main Execution
# ==========================================
def main():
    """
    Main execution function
    """
    print("\n" + "=" * 70)
    print("DWTS FACTOR ATTRIBUTION ANALYSIS - QUESTION 3")
    print("=" * 70)

    # Check if required data files exist
    if not os.path.exists(CONFIG['RAW_DATA_PATH']):
        print(f"\nError: Raw data file not found: {CONFIG['RAW_DATA_PATH']}")
        return

    if not os.path.exists(CONFIG['JUDGE_SCORES_PATH']):
        print(f"\nError: Judge scores file not found: {CONFIG['JUDGE_SCORES_PATH']}")
        print("Please ensure fan_est_cache.csv exists in the root directory.")
        return

    if not os.path.exists(CONFIG['FAN_VOTES_PATH']):
        print(f"\nError: Fan votes file not found: {CONFIG['FAN_VOTES_PATH']}")
        print("Please run Q1 analysis first to generate fan vote estimates.")
        return

    # Initialize analyzer
    analyzer = DWTS_Factor_Analyzer(
        raw_data_path=CONFIG['RAW_DATA_PATH'],
        judge_scores_path=CONFIG['JUDGE_SCORES_PATH'],
        fan_votes_path=CONFIG['FAN_VOTES_PATH']
    )

    # Run full analysis
    analyzer.run_full_analysis()

    print("\n✓ All analysis complete!")
    print("\nKey Outputs:")
    print("  1. Forest Plot: Shows which factors matter to judges vs fans")
    print("  2. Partner Scatter: Identifies elite partners and their specialties")
    print("  3. Radar Charts: Profiles of typical contestants (robbed, overrated, perfect)")
    print("\nNext Steps:")
    print("  - Review visualizations in Q3/figures/")
    print("  - Examine detailed results in Q3/results/")
    print("  - Use insights for paper discussion section")


if __name__ == "__main__":
    main()
