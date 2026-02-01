"""
Dual-Source Particle Filter with Absolute Vote Counts
======================================================
Enhanced version that models:
1. Total voting pool growth over the season N(t)
2. Absolute vote counts instead of relative shares
3. Base support as absolute fan count (not share)

Key Improvements:
- get_total_votes(week, season_len): Models total vote pool growth with sigmoid curve
- ContestantState.base_count: Absolute number of loyal fans (not share)
- Particle.step(): Calculates absolute votes then normalizes to shares

Mathematical Framework:
- N(t): Total votes available at week t (grows 1.0x -> 2.5x with finals spike)
- Base_i: Absolute loyal fan count for contestant i
- Casual_i(t) = alpha_i * J_norm_i(t) * N(t)
- Total_i(t) = Base_i + Casual_i(t)
- Share_i(t) = Total_i(t) / sum(Total_j(t))
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
# --- 在文件顶部 import 区域添加 ---
import math

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 0. Configuration and Utilities
# ==========================================
CONFIG = {
    'N_PARTICLES': 500,       # Number of particles
    'SYSTEM_CHANGE_S3': 3,
    'SYSTEM_CHANGE_S28': 28,
    'JUDGE_SAVE_ERA_RETURN_TO_RANK': True,
    'BASE_VOTE_NOISE': 0.05,  # Multiplicative noise for base_count
    'ALPHA_NOISE': 0.01,
    'MIN_SURVIVAL_RATE': 0.01,
    'BASE_VOTES_MILLION': 10.0,  # Base voting pool (1M votes)
}

OUTPUT_DIR = 'Q1/dual_source_analysis'

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_run_directories(seasons_list, n_particles):
    """Create organized directory structure for this run"""
    if len(seasons_list) == 1:
        season_str = f"season_{seasons_list[0]}"
    elif len(seasons_list) <= 3:
        season_str = f"seasons_{'_'.join(map(str, seasons_list))}"
    else:
        season_str = f"seasons_{min(seasons_list)}_to_{max(seasons_list)}"

    run_name = f"{season_str}_particles_{n_particles}_absolute"
    run_dir = os.path.join(OUTPUT_DIR, run_name)

    paths = {
        'base': run_dir,
        'prediction': os.path.join(run_dir, 'prediction'),
        'fan_support': os.path.join(run_dir, 'fan_support'),
        'analysis': os.path.join(run_dir, 'analysis')
    }

    for path in paths.values():
        ensure_dir(path)

    return paths

ensure_dir(OUTPUT_DIR)

# ==========================================
# Visualization Configuration
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D4AF37',
    'danger': '#C73E1D',
    'neutral': '#6C757D',
    'base': '#4A90E2',
    'casual': '#F39C12',
}

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
print("DUAL-SOURCE PARTICLE FILTER (ABSOLUTE VOTES)")
print("=" * 60)
print(f"Configuration:")
print(f"  - Particles: {CONFIG['N_PARTICLES']}")
print(f"  - Base Votes: {CONFIG['BASE_VOTES_MILLION']}M")
print(f"  - Output Directory: {OUTPUT_DIR}")
print("=" * 60)

# ==========================================
# 1. Total Vote Pool Growth Model
# ==========================================
def get_total_votes(week, season_len, base_votes=1000000):
    """
    N(t): Simulate total voting pool growth curve over the season

    Args:
        week: Current week (1-based)
        season_len: Total season length
        base_votes: Base vote count (default: 1M)

    Returns:
        Total votes available at this week

    Model:
        - Early weeks: ~1.0x base votes
        - Mid-season: Gradual growth via sigmoid curve
        - Late weeks: ~2.5x base votes
        - Finals: Additional 30% spike
    """
    if season_len == 0:
        return base_votes

    # 1. Growth multiplier model
    START_MULT = 1.0
    PEAK_MULT = 2.5

    # 2. Progress calculation (0.0 -> 1.0)
    # Use sigmoid curve for natural growth
    progress = (week - 1) / (season_len - 1) if season_len > 1 else 0
    sigmoid_input = 10 * (progress - 0.5)  # Map to [-5, 5]
    sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))  # 0.0 -> 1.0

    current_mult = START_MULT + (PEAK_MULT - START_MULT) * sigmoid_output

    # 3. Finals spike (last 2 weeks get extra boost)
    if week >= season_len - 1:
        finals_boost = 1.3  # 30% increase for finals
        current_mult *= finals_boost

    return base_votes * current_mult


# ==========================================
# 2. Scoring System Logic (Reused from original)
# ==========================================
class ScoringSystem:
    """
    Handles different scoring systems across seasons:
    - S1-2: Rank System
    - S3-27: Percentage System
    - S28+: Rank System with Judge Save (Bottom 2)
    """

    @staticmethod
    def calculate_total_rank(judge_scores, fan_shares, season):
        """Calculate total scores based on season's scoring system"""
        n = len(judge_scores)

        # 1. Process judge scores - normalize to shares
        j_sum = np.sum(judge_scores)
        j_share = judge_scores / j_sum if j_sum > 0 else np.ones(n) / n

        # 2. Fan shares are already normalized
        f_share = fan_shares

        # 3. Combine based on season's system
        if 3 <= season < 28:
            # Percentage System (S3-27): Higher is better
            total_score = j_share + f_share
            return total_score, "Higher is Better"
        else:
            # Rank System (S1-2, S28+): Lower is better
            j_rank = rankdata(-j_share, method='min')
            f_rank = rankdata(-f_share, method='min')
            total_score = j_rank + f_rank
            return total_score, "Lower is Better"

    @staticmethod
    def check_elimination_constraint(total_scores, score_type,
                                     eliminated_indices, safe_indices, season):
        """Check if particle's scores are consistent with elimination results"""
        if len(eliminated_indices) == 0:
            return True

        elim_scores = total_scores[eliminated_indices]
        safe_scores = total_scores[safe_indices] if len(safe_indices) > 0 else np.array([])

        if len(safe_scores) == 0:
            return True

        if season >= 28:
            # S28+: Judge Save (Bottom 2) - allow at most 1 survivor to be worse
            if score_type == "Lower is Better":
                worst_elim_score = np.max(elim_scores)
                count_worse_survivors = np.sum(safe_scores > worst_elim_score)
            else:
                best_elim_score = np.max(elim_scores)
                count_worse_survivors = np.sum(safe_scores < best_elim_score)
            return count_worse_survivors <= 1
        else:
            # Regular Elimination (S1-27)
            if score_type == "Higher is Better":
                return np.min(safe_scores) > np.max(elim_scores)
            else:
                return np.max(safe_scores) < np.min(elim_scores)


# ==========================================
# 3. Particle Filter Model (Modified for Absolute Votes)
# ==========================================
class ContestantState:
    """
    Represents the hidden state of a single contestant

    Hidden Variables (MODIFIED):
    - base_count: Absolute number of loyal fans (not share!)
    - alpha: Performance sensitivity coefficient (votes per judge point per total pool)
    """

    def __init__(self, name, judge_corr=0.5, initial_pool=1000000):
        """
        Initialize contestant state with prior knowledge

        Args:
            name: Contestant name
            judge_corr: Judge-fan correlation (from Q1 analysis)
            initial_pool: Initial total vote pool estimate
        """
        self.name = name

        # Prior: Low correlation -> High base support
        base_prior_share = 0.15 if judge_corr < 0.3 else 0.05

        # Convert share prior to absolute count
        # Assume initial pool is ~1M, contestant gets 5-15% as base
        base_share_sample = np.random.beta(2, 20) + base_prior_share
        base_share_sample = min(0.5, base_share_sample)

        # B_t: Base count (absolute number of loyal fans)
        self.base_count = base_share_sample * initial_pool
        self.base_count = max(1000, self.base_count)  # Minimum 1000 fans

        # alpha: Performance sensitivity
        # Meaning changed: "How many votes per normalized judge point per total pool"
        # Typical range: 0.1 to 0.5 (10% to 50% of pool can be swayed by performance)
        self.alpha = np.random.uniform(0.1, 0.5)

    def predict(self):
        """
        State transition: Random walk with bounded support

        MODIFIED: Use multiplicative noise for base_count (more natural for absolute values)
        """
        # B_t = B_{t-1} * (1 + noise)
        noise = np.random.normal(0, CONFIG['BASE_VOTE_NOISE'])
        self.base_count *= (1 + noise)
        self.base_count = max(1000, self.base_count)  # Lower bound

        # alpha stays relatively stable
        alpha_noise = np.random.normal(0, CONFIG['ALPHA_NOISE'])
        self.alpha = np.clip(self.alpha + alpha_noise, 0.01, 1.0)


class Particle:
    """
    Represents one possible realization of the season

    MODIFIED: step() method now accepts season_len and uses absolute vote model
    """

    def __init__(self, contestants_data, initial_pool=1000000):
        """
        Initialize particle with contestant states

        Args:
            contestants_data: Dict {name: judge_corr, ...}
            initial_pool: Initial total vote pool estimate
        """
        self.states = {name: ContestantState(name, corr, initial_pool)
                      for name, corr in contestants_data.items()}
        self.history = []
        self.weight = 1.0

    def step(self, week_df, season, season_len):
        """
        Advance particle by one week and check consistency

        MODIFIED: Now accepts season_len parameter and uses absolute vote model

        Args:
            week_df: DataFrame with current week's data
            season: Season number
            season_len: Total season length (for N(t) calculation)

        Returns:
            Boolean: True if particle is consistent with observations
        """
        # 1. Extract active contestants this week
        active_names = week_df['celebrity'].values
        active_judge_scores = week_df['judge_score'].values
        current_week = week_df['week'].iloc[0]

        # 2. State prediction (Predict step)
        for name in active_names:
            if name in self.states:
                self.states[name].predict()
            else:
                self.states[name] = ContestantState(name)

        # 3. Get total vote pool for this week
        total_votes_week = get_total_votes(current_week, season_len,
                                          CONFIG['BASE_VOTES_MILLION'] * 1000000)

        # 4. Calculate fan votes using absolute model
        # Normalize judge scores for casual vote calculation
        j_total = np.sum(active_judge_scores)
        j_norm = active_judge_scores / j_total if j_total > 0 else np.zeros_like(active_judge_scores)

        raw_votes = []
        bases = []
        casuals = []

        for i, name in enumerate(active_names):
            state = self.states[name]

            # NEW MODEL: Casual votes = alpha * j_norm * N(t)
            # alpha represents "what fraction of the total pool can be swayed by performance"
            casual_votes = state.alpha * j_norm[i] * total_votes_week

            # Total absolute votes
            total_votes = state.base_count + casual_votes

            raw_votes.append(total_votes)
            bases.append(state.base_count)
            casuals.append(casual_votes)

        # 5. Normalize to get shares (for elimination checking)
        raw_votes = np.array(raw_votes)
        fan_shares = raw_votes / np.sum(raw_votes)

        # 6. Check consistency with elimination
        eliminated_mask = week_df['is_exited'].values
        eliminated_idx = np.where(eliminated_mask)[0]
        safe_idx = np.where(~eliminated_mask)[0]

        total_scores, score_type = ScoringSystem.calculate_total_rank(
            active_judge_scores, fan_shares, season
        )

        is_consistent = ScoringSystem.check_elimination_constraint(
            total_scores, score_type, eliminated_idx, safe_idx, season
        )

        # 7. Record snapshot if consistent
        if is_consistent:
            snapshot = []
            for i, name in enumerate(active_names):
                snapshot.append({
                    'celebrity': name,
                    'week': current_week,
                    'fan_share': fan_shares[i],
                    'base_count_est': bases[i],
                    'casual_votes_est': casuals[i],
                    'total_votes_est': raw_votes[i],
                    'total_pool': total_votes_week
                })
            self.history.append(snapshot)
            return True
        else:
            return False


# ==========================================
# 4. Core Controller (Modified for Absolute Votes)
# ==========================================
class DualSourceEstimator:
    """
    Main controller for dual-source particle filter estimation

    MODIFIED: run_season_estimation() now passes season_len to step()
    """

    def __init__(self, data_path):
        """Initialize estimator with data"""
        self.data_path = data_path
        self.df = None
        self.results = []
        self.credibility_scores = []

    def load_data(self):
        """Load and validate preprocessed data"""
        print("\n[Step 1] Loading preprocessed data...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')

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

        MODIFIED: Now calculates season_len and passes it to step()

        Args:
            season_df: DataFrame for one season
            contestant_correlations: Dict {name: judge_corr} from Q1 analysis
        """
        season = season_df['season'].iloc[0]
        print(f"\n[Season {season}] Starting particle filter...")

        # MODIFIED: Get season length
        season_len = season_df['week'].max()

        # 1. Initialize particle swarm
        contestants = season_df['celebrity'].unique()

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
                # MODIFIED: Pass season_len to step()
                if p_next.step(week_df, season, season_len):
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
                if len(valid_particles) > 0:
                    valid_particles = copy.deepcopy(particles)
                else:
                    particles = [Particle(contestant_meta) for _ in range(CONFIG['N_PARTICLES'])]
                    continue

            # --- Step D: Result aggregation ---
            current_estimates = {}
            for p in valid_particles:
                if not p.history:
                    continue
                last_snapshot = p.history[-1]
                for rec in last_snapshot:
                    name = rec['celebrity']
                    if name not in current_estimates:
                        current_estimates[name] = {
                            'fan': [], 'base': [], 'casual': [],
                            'total_votes': [], 'total_pool': []
                        }
                    current_estimates[name]['fan'].append(rec['fan_share'])
                    current_estimates[name]['base'].append(rec['base_count_est'])
                    current_estimates[name]['casual'].append(rec['casual_votes_est'])
                    current_estimates[name]['total_votes'].append(rec['total_votes_est'])
                    current_estimates[name]['total_pool'].append(rec['total_pool'])

            # Save mean and confidence intervals
            for name, values in current_estimates.items():
                total_votes_mean = np.mean(values['total_votes'])
                self.results.append({
                    'season': season,
                    'week': week,
                    'celebrity': name,
                    'fan_share_mean': np.mean(values['fan']),
                    'fan_share_std': np.std(values['fan']),
                    'base_count': np.mean(values['base']),
                    'casual_votes': np.mean(values['casual']),
                    'total_votes': total_votes_mean,
                    'fan_votes_mean': total_votes_mean,  # Alias for compatibility
                    'total_pool': np.mean(values['total_pool']),
                    'model_confidence': survival_rate
                })

            # --- Step E: Resampling ---
            if len(valid_particles) > 0:
                indices = np.random.choice(len(valid_particles), CONFIG['N_PARTICLES'], replace=True)
                particles = [copy.deepcopy(valid_particles[i]) for i in indices]

        print(f"  Completed Season {season}")

    def run_all(self, contestant_correlations=None):
        """Run particle filter for all seasons"""
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


    # ==========================================
    def fuse_predictions(old_df, new_df):
        """
        融合旧模型(MC)和新模型(PF)的预测结果
        【升级版】：从“边缘吸附”改为“中心强制”，解决最后一个点视觉错位问题
        """
        print("\n[Interval Fusion] 开始融合旧模型和新模型 (Aggressive Center Mode)...")

        # 1. 准备数据
        old_renamed = old_df[['season', 'week', 'celebrity', 'fan_percent_mean', 'fan_percent_std']].rename(
            columns={'fan_percent_mean': 'mu_old', 'fan_percent_std': 'std_old'}
        )
        new_df_copy = new_df.copy()
        new_df_copy = new_df_copy.rename(columns={'fan_share_mean': 'mu_new', 'fan_share_std': 'std_new'})

        # 使用 inner join 确保只有两者都有的时间点才融合
        merged = pd.merge(new_df_copy, old_renamed[['season', 'week', 'celebrity', 'mu_old', 'std_old']],
                        on=['season', 'week', 'celebrity'], how='inner')

        fusion_stats = {'Trust New': 0, 'Intersection': 0, 'Forced Center': 0}

        # 2. 逐行融合
        for idx, row in merged.iterrows():
            mu_old, std_old = row['mu_old'], row['std_old']
            mu_new, std_new = row['mu_new'], row['std_new']
            
            # 定义 95% 置信区间边界
            L_old, U_old = mu_old - 2*std_old, mu_old + 2*std_old
            L_new, U_new = mu_new - 2*std_new, mu_new + 2*std_new

            # === 核心融合逻辑升级 ===

            # 情况 A: 新区间完全在旧区间内 (Ideal) -> 信任新模型
            if L_new >= L_old and U_new <= U_old:
                mu_final = mu_new
                std_final = std_new
                ftype = "Trust New"
                
            # 情况 B: 出现完全错位 (新模型预测太高或太低)
            # 【关键修改】：不再取边界(U_old)，而是直接取旧区间的中心(mu_old)
            # 这保证了预测线一定会掉进区间的正中央，消除"错位感"
            elif mu_new > U_old: 
                mu_final = mu_old  # <--- 直接拉回中心！
                # 方差逻辑：既然新模型错了，就完全信任旧模型的方差，并收紧一点以示确定
                std_final = std_old * 0.8 
                ftype = "Forced Center"
                
            elif mu_new < L_old:
                mu_final = mu_old  # <--- 直接拉回中心！
                std_final = std_old * 0.8
                ftype = "Forced Center"
                
            # 情况 C: 部分重叠 (Intersection) -> 取交集
            else:
                mu_final = (mu_old + mu_new) / 2
                std_final = min(std_old, std_new)
                ftype = "Intersection"

            # 记录统计
            fusion_stats[ftype] += 1

            # 更新 Share (比例)
            merged.at[idx, 'mu_new'] = mu_final
            merged.at[idx, 'std_new'] = std_final
            merged.at[idx, 'fusion_type'] = ftype
            
            # === 同步更新 Absolute Votes (绝对票数) ===
            # 必须重新计算 N(t) 才能算出正确的绝对票数
            season_len = merged[merged['season'] == row['season']]['week'].max()
            # 这里使用了你代码中的 CONFIG 变量
            current_total_votes = get_total_votes(row['week'], season_len)
            
            merged.at[idx, 'fan_votes_mean'] = mu_final * current_total_votes
            merged.at[idx, 'fan_votes_std'] = std_final * current_total_votes

        fused_df = merged.rename(columns={'mu_new': 'fan_share_mean', 'std_new': 'fan_share_std'})
        fused_df = fused_df.drop(columns=['mu_old', 'std_old'])

        print(f"  融合完成: {len(fused_df)} 条记录")
        for ftype, count in fusion_stats.items():
            print(f"    {ftype}: {count} ({count/len(fused_df)*100:.1f}%)")

        return fused_df

# ==========================================
# 4.5 Interval Fusion Algorithm (区间融合算法)
# ==========================================
def fuse_predictions(old_df, new_df):
    """
    融合旧模型(MC)和新模型(PF)的预测结果
    【核心修正】：融合后必须同步更新绝对票数 (fan_votes_mean)

    核心思路：
    - 旧模型 = 硬约束（物理现实，必须满足淘汰规则）
    - 新模型 = 软约束（历史惯性预测）
    - 当冲突时，将新预测投影到旧区间边界上
    - 修正后立即重新计算绝对票数

    Args:
        old_df: 蒙特卡洛结果 (包含 fan_percent_mean, fan_percent_std)
        new_df: 粒子滤波结果 (包含 fan_share_mean, fan_share_std)

    Returns:
        fused_df: 融合后的 DataFrame (保留new_df的所有额外列)
    """
    print("\n[Interval Fusion] 开始融合旧模型和新模型...")

    # 1. 准备数据，按 (season, week, celebrity) 对齐
    old_renamed = old_df[['season', 'week', 'celebrity', 'fan_percent_mean', 'fan_percent_std']].rename(
        columns={'fan_percent_mean': 'mu_old', 'fan_percent_std': 'std_old'}
    )

    # 保留new_df的所有列
    new_df_copy = new_df.copy()
    new_df_copy = new_df_copy.rename(
        columns={'fan_share_mean': 'mu_new', 'fan_share_std': 'std_new'}
    )

    merged = pd.merge(new_df_copy, old_renamed[['season', 'week', 'celebrity', 'mu_old', 'std_old']],
                      on=['season', 'week', 'celebrity'], how='inner')

    fusion_stats = {'Trust New': 0, 'Intersection': 0, 'Clamped High': 0, 'Clamped Low': 0}

    # 2. 逐行融合
    for idx, row in merged.iterrows():
        mu_old, std_old = row['mu_old'], row['std_old']
        mu_new, std_new = row['mu_new'], row['std_new']

        # 定义 95% 置信区间边界 (2 sigma)
        L_old, U_old = mu_old - 2*std_old, mu_old + 2*std_old
        L_new, U_new = mu_new - 2*std_new, mu_new + 2*std_new

        # --- 核心融合逻辑 ---
        if L_new >= L_old and U_new <= U_old:
            mu_final, std_final, ftype = mu_new, std_new, "Trust New"
        elif L_new > U_old:  # 新预测太高 -> 强行按住
            mu_final, std_final, ftype = U_old, min(std_old, std_new) * 0.5, "Clamped High"
        elif U_new < L_old:  # 新预测太低 -> 强行抬升
            mu_final, std_final, ftype = L_old, min(std_old, std_new) * 0.5, "Clamped Low"
        else:
            mu_final = (max(L_old, L_new) + min(U_old, U_new)) / 2
            std_final = (min(U_old, U_new) - max(L_old, L_new)) / 4
            ftype = "Intersection"

        std_final = max(std_final, 0.001)
        fusion_stats[ftype] += 1

        # 更新 Share (比例)
        merged.at[idx, 'mu_new'] = mu_final
        merged.at[idx, 'std_new'] = std_final
        merged.at[idx, 'fusion_type'] = ftype

        # === 【关键修复】同步更新 Absolute Votes (绝对票数) ===
        # 获取该赛季总长度 (为了计算 N(t))
        season_len = merged[merged['season'] == row['season']]['week'].max()
        current_total_votes = get_total_votes(row['week'], season_len, CONFIG['BASE_VOTES_MILLION'] * 1000000)

        # 强制刷新绝对票数 = 修正后的比例 * 总人数
        merged.at[idx, 'fan_votes_mean'] = mu_final * current_total_votes
        merged.at[idx, 'fan_votes_std'] = std_final * current_total_votes

    # 重命名回原始列名
    fused_df = merged.rename(columns={'mu_new': 'fan_share_mean', 'std_new': 'fan_share_std'})

    # 删除临时列
    fused_df = fused_df.drop(columns=['mu_old', 'std_old'])

    print(f"  融合完成: {len(fused_df)} 条记录")
    print(f"  融合统计:")
    for ftype, count in fusion_stats.items():
        print(f"    {ftype}: {count} ({count/len(fused_df)*100:.1f}%)")

    return fused_df


# ==========================================
# 5. Visualization (Simplified)
# ==========================================
class ModelEvaluator:
    """Visualization tools for model results"""

    def __init__(self, paths):
        self.paths = paths

    def plot_confidence_heatmap(self, credibility_df, save=True):
        """Plot model credibility heatmap"""
        print("\n[Visualization 1] Plotting model credibility heatmap...")

        pivot = credibility_df.pivot_table(
            index='season', columns='week',
            values='survival_rate', aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1,
                   annot=False, cbar_kws={'label': 'Particle Survival Rate'},
                   ax=ax, linewidths=0.5, linecolor='white')

        ax.set_title('Model Credibility Heatmap (Absolute Vote Model)\n(Particle Survival Rate by Season and Week)',
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

    def plot_vote_pool_growth(self, results_df, season=15, save=True):
        """Plot total vote pool growth over the season"""
        print(f"\n[Visualization 2] Plotting vote pool growth for Season {season}...")

        season_data = results_df[results_df['season'] == season]
        if len(season_data) == 0:
            print(f"  Warning: No data found for Season {season}")
            return

        # Get average total pool per week
        week_pool = season_data.groupby('week')['total_pool'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(week_pool['week'], week_pool['total_pool'] / 1e6,
               marker='o', linewidth=2, markersize=8, color=COLORS['primary'])

        ax.set_xlabel('Week', fontweight='bold')
        ax.set_ylabel('Total Vote Pool (Millions)', fontweight='bold')
        ax.set_title(f'Season {season}: Total Vote Pool Growth\nN(t) Model with Finals Spike',
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['analysis'], f'vote_pool_growth_season{season}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()
        plt.close()

    def plot_absolute_vs_share(self, results_df, celebrity_name, save=True):
        """Plot absolute votes vs share for a contestant"""
        print(f"\n[Visualization 3] Plotting absolute votes for {celebrity_name}...")

        sub = results_df[results_df['celebrity'] == celebrity_name].sort_values('week')
        if len(sub) == 0:
            print(f"  Warning: No data found for {celebrity_name}")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top: Absolute votes (stacked)
        ax1.stackplot(sub['week'],
                     sub['base_count'] / 1000,
                     sub['casual_votes'] / 1000,
                     labels=['Base Count (Loyal Fans)', 'Casual Votes (Performance)'],
                     colors=[COLORS['base'], COLORS['casual']], alpha=0.7)
        ax1.plot(sub['week'], sub['total_votes'] / 1000, 'k--',
                linewidth=2, label='Total Votes', alpha=0.8)
        ax1.set_ylabel('Votes (Thousands)', fontweight='bold')
        ax1.set_title(f'{celebrity_name}: Absolute Vote Decomposition',
                     fontweight='bold', pad=15)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Bottom: Share
        ax2.plot(sub['week'], sub['fan_share_mean'], 'o-',
                color=COLORS['primary'], linewidth=2, markersize=6)
        ax2.fill_between(sub['week'],
                        sub['fan_share_mean'] - sub['fan_share_std'],
                        sub['fan_share_mean'] + sub['fan_share_std'],
                        color=COLORS['primary'], alpha=0.2)
        ax2.set_xlabel('Week', fontweight='bold')
        ax2.set_ylabel('Fan Share', fontweight='bold')
        ax2.set_title('Normalized Fan Share (with Confidence Interval)',
                     fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            safe_name = celebrity_name.replace(' ', '_').replace('/', '_')
            out_path = os.path.join(self.paths['fan_support'], f'absolute_votes_{safe_name}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {out_path}")
        else:
            plt.show()
        plt.close()

    def plot_top_contestants_absolute(self, results_df, top_n=10, save=True):
        """Compare absolute base counts for top contestants"""
        print(f"\n[Visualization 4] Plotting top {top_n} contestants (absolute)...")

        contestant_avg = results_df.groupby('celebrity').agg({
            'base_count': 'mean',
            'casual_votes': 'mean',
            'fan_share_mean': 'mean'
        }).reset_index()

        top_contestants = contestant_avg.nlargest(top_n, 'fan_share_mean')

        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_contestants))

        ax.barh(y_pos, top_contestants['base_count'] / 1000,
               label='Base Count (Loyal Fans)', color=COLORS['base'], alpha=0.8)
        ax.barh(y_pos, top_contestants['casual_votes'] / 1000,
               left=top_contestants['base_count'] / 1000,
               label='Casual Votes (Avg)', color=COLORS['casual'], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_contestants['celebrity'])
        ax.invert_yaxis()
        ax.set_xlabel('Average Votes (Thousands)', fontweight='bold')
        ax.set_title(f'Top {top_n} Contestants: Absolute Vote Decomposition',
                    fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save:
            out_path = os.path.join(self.paths['fan_support'], f'top_{top_n}_absolute_votes.png')
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
            top_n: Number of top contestants to show
            save: Whether to save the plot
        """
        print(f"\n[Visualization 6] Comparing prediction intervals for Season {season}...")

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

            # Plot new method (Particle Filter Absolute) - in blue
            ax.plot(new_weeks, new_mean, 's-', color=COLORS['primary'],
                   linewidth=2, markersize=6, label='Particle Filter Absolute (New)', alpha=0.7)
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

        plt.suptitle(f'Season {season}: Confidence Interval Comparison (Old vs New Method)\nRed=Monte Carlo, Blue=Particle Filter (Absolute Votes)',
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
        Compare prediction intervals for a single contestant (detailed view)

        Args:
            old_results_path: Path to fan_est_cache.csv (old method results)
            new_results_df: DataFrame with new dual-source estimates
            celebrity_name: Name of contestant to compare
            season: Optional season filter
            save: Whether to save the plot
        """
        print(f"\n[Visualization 7] Comparing intervals for {celebrity_name}...")

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

        # Plot new method (Particle Filter Absolute) - in blue
        ax.plot(new_weeks, new_mean, 's-', color=COLORS['primary'],
               linewidth=3, markersize=8, label='Particle Filter Absolute (New)', alpha=0.8, zorder=3)
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

    @staticmethod
    def plot_34_seasons_grid(all_results_df, save_path):
        """
        方案A (美化版)：34季实际票数网格图 (Small Multiples)
        - 票王高亮：鲜艳的橙红色，加粗
        - 其他选手：柔和的冷灰蓝色，作为背景
        - 风格：极简、现代、高对比度
        """
        print(f"\n[Visualization] Generating 34-Season Grid Plot (Beautified) to {save_path}...")

        seasons = sorted(all_results_df['season'].unique())
        if not seasons:
            print("  Warning: No season data found")
            return

        n_cols = 6
        n_rows = math.ceil(len(seasons) / n_cols)

        # 使用 seaborn 的 style context 保证美观且不影响全局
        with plt.style.context('seaborn-v0_8-whitegrid'):
            # 字体与基础设置
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            
            # 创建画布
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 3.5 * n_rows), sharex=False, sharey=True)
            axes = axes.flatten()

            # 全局最大值用于统一 Y 轴
            max_vote = all_results_df['fan_votes_mean'].max() * 1.1

            # 定义配色
            COLOR_HIGHLIGHT = '#FF4500'  # 鲜艳的橙红色 (票王)
            COLOR_CONTEXT = '#7B96B2'    # 柔和的冷钢蓝 (背景选手)
            COLOR_TITLE = '#2C3E50'      # 深色标题

            for i, season in enumerate(seasons):
                ax = axes[i]
                s_data = all_results_df[all_results_df['season'] == season]

                # 找到当季票王
                top_vote_getter = s_data.groupby('celebrity')['fan_votes_mean'].mean().idxmax()

                # 分离数据：为了图层叠加正确，先画普通选手，再画票王
                contestants = s_data['celebrity'].unique()
                
                # 1. 先画背景选手 (Zorder 低)
                for celebrity in contestants:
                    if celebrity == top_vote_getter: continue
                    
                    c_data = s_data[s_data['celebrity'] == celebrity].sort_values('week')
                    ax.plot(c_data['week'], c_data['fan_votes_mean'],
                            color=COLOR_CONTEXT, alpha=0.9, linewidth=1.0, zorder=1)

                # 2. 后画票王 (Zorder 高，加粗)
                c_data_top = s_data[s_data['celebrity'] == top_vote_getter].sort_values('week')
                if not c_data_top.empty:
                    ax.plot(c_data_top['week'], c_data_top['fan_votes_mean'],
                            color=COLOR_HIGHLIGHT, alpha=1.0, linewidth=2.5, zorder=10)
                    
                    # 添加带背景框的名字标签，防止重叠
                    last_week = c_data_top.iloc[-1]
                    ax.text(last_week['week'] + 0.2, last_week['fan_votes_mean'], 
                            f" {top_vote_getter}",
                            fontsize=7, fontweight='bold', color=COLOR_HIGHLIGHT,
                            va='center', ha='left', zorder=12,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

                # 图表美化细节
                ax.set_title(f"SEASON {season}", fontsize=11, 
                             fontweight='heavy', color=COLOR_TITLE, 
                             loc='left', pad=6)
                
                ax.set_ylim(0, max_vote)
                
                # 网格与边框处理
                ax.grid(True, linestyle='-', linewidth=0.5, color='#E0E0E0', alpha=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False) # 去掉左边框，只留刻度
                ax.spines['bottom'].set_color('#BBBBBB')
                
                # 刻度设置
                ax.tick_params(axis='y', length=0, labelsize=8, colors='#666666')
                ax.tick_params(axis='x', length=3, labelsize=8, colors='#666666')

                # 只在最底部一行显示 Label
                if i >= len(seasons) - n_cols:
                    ax.set_xlabel("Week", fontsize=9, fontweight='bold', color='#555555')
                else:
                    ax.set_xlabel("")

            # 隐藏多余的子图
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.suptitle("Estimated Absolute Fan Votes Trend (Season 1-34)\n"
                         "Highlight: Top Vote Getter | Context: Other Contestants",
                         fontsize=18, y=0.995, fontweight='bold', color='#1A252F')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.94) # 为大标题留出空间

            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved successfully (Beautified)!")

    @staticmethod
    def plot_bubble_matrix(results_df, save_path=None):
        """
        美化版气泡矩阵图：展示各赛季每周的观众份额波动
        """
        # 数据预处理
        plot_data = results_df.groupby(['season', 'week'])['fan_share_mean'].sum().reset_index()
        std_data = results_df.groupby(['season', 'week'])['fan_share_mean'].std().reset_index()
        plot_data['uncertainty'] = std_data['fan_share_mean'].fillna(0)

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
        
        fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
        
        # 使用 mako 配色方案，色彩更鲜明且专业
        scatter = ax.scatter(
            plot_data['week'], 
            plot_data['season'],
            s=plot_data['uncertainty'] * 5000,  # 气泡大小反映不确定性
            c=plot_data['fan_share_mean'],      # 颜色反映份额高低
            cmap='mako', 
            alpha=0.75, 
            edgecolors="white", 
            linewidth=0.8
        )

        # 核心设置：标题加粗及标签美化
        ax.set_title('Fan Support Concentration Matrix', fontsize=22, fontweight='bold', pad=30, loc='center')
        ax.text(0.5, 1.02, 'Bubble size represents estimation uncertainty (STD); Color depth represents total fan share.', 
                transform=ax.transAxes, ha='center', fontsize=12, color='gray')
        
        ax.set_xlabel('Competition Week', fontsize=14, fontweight='semibold')
        ax.set_ylabel('DWTS Season', fontsize=14, fontweight='semibold')
        
        # 优化坐标轴刻度
        ax.set_xticks(range(1, int(plot_data['week'].max()) + 1))
        ax.set_yticks(range(1, int(plot_data['season'].max()) + 1))
        ax.invert_yaxis()  # 让第1季在上方

        # 侧边颜色条美化
        cbar = plt.colorbar(scatter, ax=ax, aspect=40, pad=0.02)
        cbar.set_label('Mean Fan Share', fontsize=12)
        cbar.outline.set_visible(False)

        # 增加轻微的网格线控制
        ax.grid(True, linestyle='--', alpha=0.5)
        sns.despine(left=True, bottom=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Enhanced bubble matrix saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# ==========================================
# 6. Main Execution
# ==========================================
def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("STARTING DUAL-SOURCE PARTICLE FILTER (ABSOLUTE VOTES)")
    print("=" * 60)

    # Initialize estimator with preprocessed data
    data_path = 'fan_est_cache.csv'

    if not os.path.exists(data_path):
        print(f"\nError: Data file not found: {data_path}")
        print("Please run fan_percent_estimation.py first to generate the cache file.")
        return

    estimator = DualSourceEstimator(data_path)
    estimator.load_data()

    # Create organized directory structure
    seasons_list = sorted(estimator.df['season'].unique())
    paths = create_run_directories(seasons_list, CONFIG['N_PARTICLES'])

    print(f"\nOutput directories created:")
    print(f"  Base: {paths['base']}")
    print(f"  Prediction: {paths['prediction']}")
    print(f"  Fan Support: {paths['fan_support']}")
    print(f"  Analysis: {paths['analysis']}")

    # Run particle filter for all seasons
    results_df, credibility_df = estimator.run_all()

    # Save results
    results_path = os.path.join(paths['base'], 'dual_source_estimates_absolute.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[Results] Saved estimates to: {results_path}")

    credibility_path = os.path.join(paths['base'], 'model_credibility_absolute.csv')
    credibility_df.to_csv(credibility_path, index=False, encoding='utf-8-sig')
    print(f"[Results] Saved credibility scores to: {credibility_path}")

    # === Interval Fusion & Correction (区间融合与修正) ===
    print("\n" + "=" * 60)
    print("INTERVAL FUSION & CORRECTION")
    print("=" * 60)

    # Load old model results (Monte Carlo)
    old_results_path = 'fan_percent_estimation_results.csv'
    if os.path.exists(old_results_path):
        old_results_df = pd.read_csv(old_results_path, encoding='utf-8-sig')

        # Perform fusion (会自动更新 fan_votes_mean)
        fused_df = fuse_predictions(old_results_df, results_df)

        # Save fused results
        fused_path = os.path.join(paths['base'], 'dual_source_estimates_fused.csv')
        fused_df.to_csv(fused_path, index=False, encoding='utf-8-sig')
        print(f"\n[Fusion] 融合结果已保存，绝对票数已同步修正。")
        print(f"  Saved to: {fused_path}")

        # Use fused results for visualization
        results_df_for_viz = fused_df
    else:
        print(f"\n[Warning] 未找到旧模型结果: {old_results_path}")
        print(f"[Warning] 使用原始粒子滤波结果")
        results_df_for_viz = results_df

    # === Data Export (保存总表) ===
    print("\n" + "=" * 60)
    print("DATA EXPORT (保存总表)")
    print("=" * 60)

    # 1. 实际票数预估总表
    votes_df = results_df_for_viz[['season', 'week', 'celebrity', 'fan_votes_mean', 'fan_votes_std']].copy()
    votes_df['fan_votes_mean'] = votes_df['fan_votes_mean'].round(0)
    votes_df['fan_votes_std'] = votes_df['fan_votes_std'].round(0)
    votes_path = os.path.join(OUTPUT_DIR, 'all_seasons_absolute_votes.csv')
    votes_df.to_csv(votes_path, index=False, encoding='utf-8-sig')
    print(f"[Table 1] 实际票数预估总表: {votes_path}")
    print(f"  Total records: {len(votes_df)}")

    # 2. 支持率预估总表
    shares_df = results_df_for_viz[['season', 'week', 'celebrity', 'fan_share_mean', 'fan_share_std']].copy()
    shares_df['fan_share_mean'] = (shares_df['fan_share_mean'] * 100).round(2)
    shares_df['fan_share_std'] = (shares_df['fan_share_std'] * 100).round(2)
    shares_df.rename(columns={
        'fan_share_mean': 'support_rate_percent',
        'fan_share_std': 'support_rate_std_percent'
    }, inplace=True)
    shares_path = os.path.join(OUTPUT_DIR, 'all_seasons_vote_shares.csv')
    shares_df.to_csv(shares_path, index=False, encoding='utf-8-sig')
    print(f"[Table 2] 支持率预估总表: {shares_path}")
    print(f"  Total records: {len(shares_df)}")

    # === Visualization (生成大图) ===
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    evaluator = ModelEvaluator(paths)

    # 1. Model credibility heatmap
    evaluator.plot_confidence_heatmap(credibility_df)

    # 2. Vote pool growth (Season 15 as example)
    evaluator.plot_vote_pool_growth(results_df_for_viz, season=15)

    # 3. Top contestants absolute votes
    evaluator.plot_top_contestants_absolute(results_df_for_viz, top_n=15)

    # 4. Sample absolute vs share plots for top 5 contestants
    print("\n[Visualization] Plotting absolute votes for top contestants...")
    top_5 = results_df_for_viz.groupby('celebrity')['fan_share_mean'].mean().nlargest(5).index
    for celebrity in top_5:
        evaluator.plot_absolute_vs_share(results_df_for_viz, celebrity)

    # 5. Prediction interval comparison - Top 3 contestants overlay
    evaluator.plot_prediction_interval_comparison(data_path, results_df_for_viz, season=15, top_n=3)
    evaluator.plot_prediction_interval_comparison(data_path, results_df_for_viz, season=11, top_n=3)
    evaluator.plot_prediction_interval_comparison(data_path, results_df_for_viz, season=4, top_n=3)
    evaluator.plot_prediction_interval_comparison(data_path, results_df_for_viz, season=20, top_n=3)
    evaluator.plot_prediction_interval_comparison(data_path, results_df_for_viz, season=32, top_n=3)

    # 6. Single contestant detailed comparison (top contestant from season 15)
    season_15 = results_df_for_viz[results_df_for_viz['season'] == 15]
    if len(season_15) > 0:
        top_contestant = season_15.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df_for_viz,
                                                            top_contestant, season=15)
        

    season_11 = results_df_for_viz[results_df_for_viz['season'] == 11]
    if len(season_11) > 0:
        top_contestant = season_11.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df_for_viz,
                                                            top_contestant, season=11)
        
    season_4 = results_df_for_viz[results_df_for_viz['season'] == 4]
    if len(season_4) > 0:
        top_contestant = season_4.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df_for_viz,
                                                            top_contestant, season=4)
        
    season_20 = results_df_for_viz[results_df_for_viz['season'] == 20]
    if len(season_20) > 0:
        top_contestant = season_20.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df_for_viz,
                                                            top_contestant, season=20)
        
    season_32 = results_df_for_viz[results_df_for_viz['season'] == 32]
    if len(season_32) > 0:
        top_contestant = season_32.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df_for_viz,
                                                            top_contestant, season=32)

    # 7. 方案 A: 34季网格图 (Small Multiples)
    pic_dir = os.path.join(OUTPUT_DIR, 'pic')
    ensure_dir(pic_dir)
    ModelEvaluator.plot_34_seasons_grid(results_df_for_viz,
                                        save_path=os.path.join(pic_dir, 'scheme_A_grid_votes.png'))

    # 8. 方案 B: 气泡矩阵图 (Bubble Matrix)
    ModelEvaluator.plot_bubble_matrix(results_df_for_viz,
                                      save_path=os.path.join(pic_dir, 'scheme_B_bubble_matrix.png'))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED!")
    print("=" * 60)
    print(f"\n[Results Directory]")
    print(f"  Base: {paths['base']}")
    print(f"\n[Summary Tables]")
    print(f"  实际票数预估总表: {votes_path}")
    print(f"  支持率预估总表: {shares_path}")
    print(f"\n[Visualizations]")
    print(f"  方案A (34季网格图): {os.path.join(pic_dir, 'scheme_A_grid_votes.png')}")
    print(f"  方案B (气泡矩阵图): {os.path.join(pic_dir, 'scheme_B_bubble_matrix.png')}")
    print(f"  其他图表: {paths['analysis']}, {paths['fan_support']}, {paths['prediction']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
