"""
Dynamic Weight with Judges' Save System - Sensitivity Analysis (Fixed Version)
================================================================================
Fixed Issue: 'Text' object has no property 'pad' (removed pad from suptitle)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm
warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================
# 基础配置（沿用原有配置，新增敏感性分析参数）
# ==========================================
CONFIG = {
    'RAW_DATA_PATH': '2026_MCM_Problem_C_Data.csv',
    'JUDGE_SCORES_PATH': 'fan_est_cache.csv',
    'FAN_VOTES_PATH': 'Q1/dual_source_analysis/seasons_1_to_34_particles_500_absolute/dual_source_estimates_fused.csv',
    'OUTPUT_DIR': 'Q4',
    'SENSITIVITY_DIR': 'Q4/sensitivity_analysis',
    # 原始参数基准值
    'BASE_PARAMS': {
        'BASE_FAN_WEIGHT_START': 0.42,
        'BASE_FAN_WEIGHT_END': 0.34,
        'DISAGREEMENT_BOOST_LOW': 0.18,
        'DISAGREEMENT_BOOST_HIGH': 0.10,
        'DISAGREEMENT_THRESHOLD': 0.4,
        'SAVE_THRESHOLD': 0.7
    },
    # 敏感性分析参数变化范围（基准值±20%、±40%）
    'SENSITIVITY_PARAMS': {
        'DISAGREEMENT_THRESHOLD': [0.2, 0.3, 0.4, 0.5, 0.6],  # 核心参数重点分析
        'SAVE_THRESHOLD': [0.5, 0.6, 0.7, 0.8, 0.9],
        'DISAGREEMENT_BOOST_LOW': [0.10, 0.14, 0.18, 0.22, 0.26],
        'BASE_FAN_WEIGHT_START': [0.34, 0.38, 0.42, 0.46, 0.50]
    }
}

# 可视化配置（修复suptitle参数问题，优化兼容性）
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = {
    'baseline': '#2E86AB',    # 基准参数颜色
    'param_low': '#F18F01',   # 低参数值颜色
    'param_high': '#C73E1D',  # 高参数值颜色
    'other': '#6C757D'        # 其他参数值颜色
}

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'figure.figsize': (14, 10),
    'figure.subplot.hspace': 0.3,  # 增加子图垂直间距（替代pad参数）
    'figure.subplot.wspace': 0.3   # 增加子图水平间距
})

# ==========================================
# 工具函数（复用原有核心函数，确保一致性）
# ==========================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# 创建敏感性分析输出目录
ensure_dir(CONFIG['SENSITIVITY_DIR'])

def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index if isinstance(series, pd.Series) else None)
    return (series - min_val) / (max_val - min_val)

def calculate_dynamic_weights(judge_scores, fan_shares, week, total_weeks, params):
    """修改为接收动态参数"""
    n = len(judge_scores)
    if n < 2:
        return 0.5, 0.5, 0.0, 1.0
    
    judge_ranks = rankdata(-judge_scores, method='min')
    fan_ranks = rankdata(-fan_shares, method='min')
    rho, _ = spearmanr(judge_ranks, fan_ranks)
    rho = 0.0 if np.isnan(rho) else rho
    
    D = (1 - rho) / 2
    progress = (week - 1) / max(1, total_weeks - 1)
    base_fan_weight = (params['BASE_FAN_WEIGHT_START'] - 
                      (params['BASE_FAN_WEIGHT_START'] - params['BASE_FAN_WEIGHT_END']) * progress)
    
    if D <= params['DISAGREEMENT_THRESHOLD']:
        w_f = base_fan_weight + params['DISAGREEMENT_BOOST_LOW'] * (D / params['DISAGREEMENT_THRESHOLD'])
    else:
        w_f = (base_fan_weight + params['DISAGREEMENT_BOOST_LOW'] +
               params['DISAGREEMENT_BOOST_HIGH'] * ((D - params['DISAGREEMENT_THRESHOLD']) /
                                                    (1 - params['DISAGREEMENT_THRESHOLD'])))
    
    w_f = np.clip(w_f, 0.3, 0.7)
    return 1 - w_f, w_f, D, rho

# ==========================================
# 核心分析类（修复图表绘制问题）
# ==========================================
class DynamicWeightSensitivityAnalyzer:
    def __init__(self):
        self.load_data()
        self.preprocess_data()
        self.sensitivity_results = []

    def load_data(self):
        print("[Loading Data]...")
        self.raw_df = pd.read_csv(CONFIG['RAW_DATA_PATH'], encoding='utf-8-sig')
        self.judge_df = pd.read_csv(CONFIG['JUDGE_SCORES_PATH'], encoding='utf-8-sig')
        self.fan_df = pd.read_csv(CONFIG['FAN_VOTES_PATH'], encoding='utf-8-sig')

    def preprocess_data(self):
        self.merged_df = pd.merge(
            self.judge_df[['season', 'week', 'celebrity', 'judge_score', 'is_exited']],
            self.fan_df[['season', 'week', 'celebrity', 'fan_share_mean']],
            on=['season', 'week', 'celebrity'],
            how='inner'
        )
        # 选择重点分析赛季（减少计算量，保证结果代表性）
        self.target_seasons = [5, 15, 27]  # Sabrina、All-Stars、Bobby Bones
        self.merged_df = self.merged_df[self.merged_df['season'].isin(self.target_seasons)]

    def simulate_with_params(self, params, param_name, param_value):
        """使用特定参数模拟所有目标赛季"""
        season_results = []
        
        for season in self.target_seasons:
            season_data = self.merged_df[self.merged_df['season'] == season].copy()
            weeks = sorted(season_data['week'].unique())
            total_weeks = len(weeks)
            remaining_contestants = set(season_data['celebrity'].unique())
            save_used = False
            
            # 跟踪该赛季关键指标
            season_metrics = {
                'param_name': param_name,
                'param_value': param_value,
                'season': season,
                'avg_skill_corr': 0,
                'total_regret': 0,
                'weight_volatility': 0,
                'save_events_count': 0
            }
            
            week_weights = []
            for week in weeks:
                week_data = season_data[season_data['week'] == week].copy()
                week_data = week_data[week_data['celebrity'].isin(remaining_contestants)]
                if len(week_data) < 2:
                    continue
                
                # 计算当周权重和分数
                judge_scores = week_data['judge_score'].values
                fan_shares = week_data['fan_share_mean'].values
                judge_norm = normalize_series(pd.Series(judge_scores)).values
                fan_norm = normalize_series(pd.Series(fan_shares)).values
                
                w_j, w_f, D, rho = calculate_dynamic_weights(
                    judge_scores, fan_shares, week, total_weeks, params
                )
                week_weights.append(w_f)
                
                # 检查评委拯救触发
                new_scores = w_j * judge_norm + w_f * fan_norm
                week_data['new_rank'] = rankdata(-new_scores, method='min')
                lowest_rank_idx = week_data['new_rank'].idxmax()
                
                if (not save_used and
                    2 <= week <= total_weeks - 2 and
                    judge_norm[week_data.index.get_loc(lowest_rank_idx)] > params['SAVE_THRESHOLD']):
                    season_metrics['save_events_count'] += 1
                    save_used = True
            
            # 计算赛季级指标
            if week_weights:
                season_metrics['weight_volatility'] = np.std(week_weights)
            
            # 计算公平性指标
            avg_scores = season_data.groupby('celebrity')['judge_score'].mean()
            actual_exits = season_data[season_data['is_exited']].groupby('celebrity')['week'].first()
            common_contestants = avg_scores.index.intersection(actual_exits.index)
            
            if len(common_contestants) > 2:
                corr, _ = spearmanr(avg_scores[common_contestants], actual_exits[common_contestants])
                season_metrics['avg_skill_corr'] = corr if not np.isnan(corr) else 0
            
            # 计算遗憾案例数
            regret_count = 0
            for celeb in actual_exits.index:
                if celeb in avg_scores.index:
                    skill_rank = avg_scores.rank(ascending=False)[celeb]
                    exit_week = actual_exits[celeb]
                    if skill_rank <= 3 and exit_week <= total_weeks / 2:
                        regret_count += 1
            season_metrics['total_regret'] = regret_count
            
            season_results.append(season_metrics)
        
        return season_results

    def run_sensitivity_analysis(self):
        """执行完整敏感性分析"""
        print("\n[Starting Sensitivity Analysis]...")
        
        # 遍历每个需要分析的参数
        for param_name, param_values in CONFIG['SENSITIVITY_PARAMS'].items():
            print(f"\nAnalyzing parameter: {param_name}")
            
            for param_value in tqdm(param_values, desc=f"Param values {param_values}"):
                # 构建当前参数组合（基于基准参数修改）
                current_params = CONFIG['BASE_PARAMS'].copy()
                # 处理关联参数（保持BASE_FAN_WEIGHT_END与基准比例一致）
                if param_name == 'BASE_FAN_WEIGHT_START':
                    ratio = CONFIG['BASE_PARAMS']['BASE_FAN_WEIGHT_END'] / CONFIG['BASE_PARAMS']['BASE_FAN_WEIGHT_START']
                    current_params['BASE_FAN_WEIGHT_END'] = param_value * ratio
                current_params[param_name] = param_value
                
                # 执行模拟
                results = self.simulate_with_params(current_params, param_name, param_value)
                self.sensitivity_results.extend(results)
        
        # 保存原始结果
        results_df = pd.DataFrame(self.sensitivity_results)
        results_df.to_csv(
            os.path.join(CONFIG['SENSITIVITY_DIR'], 'sensitivity_results.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(f"\nSensitivity results saved to: {CONFIG['SENSITIVITY_DIR']}/sensitivity_results.csv")
        return results_df

    def plot_sensitivity_results(self, results_df):
        """生成敏感性分析图表（修复suptitle参数问题）"""
        print("\n[Generating Sensitivity Plots]...")
        
        # 定义要可视化的指标
        metrics = [
            ('avg_skill_corr', 'Skill-Rank Correlation', 'Higher = More Fair'),
            ('total_regret', 'Regret Case Count', 'Lower = More Fair'),
            ('weight_volatility', 'Weight Volatility', 'Higher = More Dramatic'),
            ('save_events_count', 'Save Events Count', 'Higher = More Drama')
        ]
        
        # 为每个参数生成子图
        for param_name in CONFIG['SENSITIVITY_PARAMS'].keys():
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 增大图幅，提升可读性
            axes = axes.flatten()
            # 修复：移除suptitle的pad参数，使用rcParams的subplot.hspace控制间距
            fig.suptitle(f'Sensitivity Analysis: {param_name} Parameter\n(Base Value: {CONFIG["BASE_PARAMS"][param_name]})',
                        fontsize=16, fontweight='bold')
            
            # 筛选当前参数的结果
            param_results = results_df[results_df['param_name'] == param_name]
            base_value = CONFIG['BASE_PARAMS'][param_name]
            
            for idx, (metric_col, metric_title, metric_note) in enumerate(metrics):
                ax = axes[idx]
                
                # 按赛季分组绘制
                for season in self.target_seasons:
                    season_data = param_results[param_results['season'] == season]
                    x = season_data['param_value']
                    y = season_data[metric_col]
                    
                    # 标记基准参数值
                    if base_value in x.values:
                        base_idx = x[x == base_value].index[0]
                        base_y = y[base_idx]
                        ax.scatter(base_value, base_y, color=COLORS['baseline'], s=100, 
                                  zorder=5, label=f'S{season} (Baseline)' if idx == 0 else "")
                        ax.plot(x, y, marker='o', linewidth=2, alpha=0.7, 
                               label=f'Season {season}' if idx == 0 else "")
                    else:
                        ax.plot(x, y, marker='o', linewidth=2, alpha=0.7)
                
                # 图表样式设置（匹配参考图）
                ax.axvline(x=base_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline' if idx == 0 else "")
                ax.set_xlabel(f'{param_name} Value', fontweight='bold')
                ax.set_ylabel(metric_title, fontweight='bold')
                ax.set_title(f'{metric_title}\n({metric_note})', fontsize=12, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3)
                
                # 只在第一个子图显示图例
                if idx == 0:
                    ax.legend(loc='best', fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为suptitle预留空间
            save_path = os.path.join(CONFIG['SENSITIVITY_DIR'], f'sensitivity_{param_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {save_path}")
        
        # 生成综合对比图（类似参考图的分布对比）
        self.plot_distribution_comparison(results_df)

    def plot_distribution_comparison(self, results_df):
        """生成参数变化对核心指标的分布影响图（与参考图结构一致）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        # 修复：移除suptitle的pad参数
        fig.suptitle('Sensitivity Analysis: Parameter Impact Distribution\n(Baseline vs Param Variations)',
                    fontsize=16, fontweight='bold')
        
        metrics = [
            ('avg_skill_corr', 'Skill-Rank Correlation'),
            ('total_regret', 'Regret Case Count'),
            ('weight_volatility', 'Weight Volatility'),
            ('save_events_count', 'Save Events Count')
        ]
        
        for idx, (metric_col, metric_title) in enumerate(metrics):
            ax = axes[idx]
            
            # 分离基准结果和变化结果
            base_results = results_df[results_df['param_value'] == results_df.apply(
                lambda x: CONFIG['BASE_PARAMS'][x['param_name']], axis=1
            )][metric_col]
            
            variation_results = results_df[results_df['param_value'] != results_df.apply(
                lambda x: CONFIG['BASE_PARAMS'][x['param_name']], axis=1
            )][metric_col]
            
            # 绘制分布直方图（参考图风格：清晰分组、网格线）
            ax.hist(base_results, bins=8, alpha=0.7, color=COLORS['baseline'], 
                   label='Baseline Params', edgecolor='black', linewidth=1)
            ax.hist(variation_results, bins=12, alpha=0.7, color=COLORS['other'], 
                   label='Param Variations', edgecolor='black', linewidth=1)
            
            # 添加统计信息标注
            ax.axvline(base_results.mean(), color=COLORS['baseline'], linestyle='--', 
                      linewidth=2, label=f'Baseline Mean: {base_results.mean():.3f}')
            ax.axvline(variation_results.mean(), color=COLORS['other'], linestyle='--', 
                      linewidth=2, label=f'Variation Mean: {variation_results.mean():.3f}')
            
            ax.set_xlabel(metric_title, fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Distribution of {metric_title}', fontsize=12, fontweight='bold', pad=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为suptitle预留空间
        save_path = os.path.join(CONFIG['SENSITIVITY_DIR'], 'sensitivity_distribution_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution comparison plot: {save_path}")

# ==========================================
# 主执行函数
# ==========================================
def main():
    # 初始化分析器
    analyzer = DynamicWeightSensitivityAnalyzer()
    
    # 执行敏感性分析
    results_df = analyzer.run_sensitivity_analysis()
    
    # 生成可视化图表
    analyzer.plot_sensitivity_results(results_df)
    
    print("\n[SUCCESS] Sensitivity Analysis Completed!")
    print(f"All plots saved to: {CONFIG['SENSITIVITY_DIR']}")

if __name__ == "__main__":
    main()