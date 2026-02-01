"""
选手综合指标分析系统
Contestant Comprehensive Metrics Analysis System

功能模块：
1. 能力指标计算 (Ability Metrics)
2. 人气指标计算 (Popularity Metrics)
3. 选手分类系统 (Classification System)
4. 可视化分析 (Visualization Analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# 统一图片保存路径
METRICS_PIC_ROOT = os.path.join('Q1', 'metrics_analysis')
os.makedirs(METRICS_PIC_ROOT, exist_ok=True)

# 配色方案 - 专业且舒适
COLOR_PALETTE = {
    'primary': '#2E86AB',      # 深蓝色
    'secondary': '#A23B72',    # 紫红色
    'accent': '#F18F01',       # 橙色
    'success': '#06A77D',      # 绿色
    'warning': '#D4AF37',      # 金色
    'danger': '#C73E1D',       # 红色
    'neutral': '#6C757D',      # 灰色
    'light': '#E9ECEF',        # 浅灰
    'dark': '#212529'          # 深灰
}


class MetricsCalculator:
    """选手综合指标计算器"""

    def __init__(self, final_df, monte_carlo_results):
        """
        初始化指标计算器

        参数:
        - final_df: 包含 season, week, celebrity, judge_score (judge_avg)
        - monte_carlo_results: 包含估算的 fan_percent_mean, fan_percent_std
        """
        print("\n" + "=" * 60)
        print("Initializing Metrics Calculator")
        print("=" * 60)

        # 合并数据
        self.data = pd.merge(
            final_df,
            monte_carlo_results[['celebrity', 'season', 'week',
                                'fan_percent_mean', 'fan_percent_std',
                                'judge_percent', 'n_valid_samples']],
            on=['season', 'week', 'celebrity'],
            how='inner'
        )

        print(f"Merged data: {len(self.data)} records")
        print(f"Seasons: {self.data['season'].nunique()}")
        print(f"Celebrities: {self.data['celebrity'].nunique()}")

        # 存储计算结果
        self.ability_metrics = None
        self.popularity_metrics = None
        self.classification_results = None

    def calculate_ability_metrics(self):
        """
        计算能力指标

        指标包括:
        - A_raw: 原始评委分
        - A_norm: 周内归一化评委分 (0-1)
        - A_zscore: 周内Z-score标准化
        - A_core: 核心能力指标 (跨周平均Z-score)
        """
        print("\n" + "=" * 60)
        print("Calculating Ability Metrics")
        print("=" * 60)

        records = []

        # 按 season, week 分组计算
        for (season, week), group in self.data.groupby(['season', 'week']):
            group = group.copy()
            n_contestants = len(group)

            if n_contestants <= 1:
                continue

            # A_raw: 原始评委分
            group['A_raw'] = group['judge_avg']

            # A_norm: 周内归一化 (0-1)
            min_score = group['judge_avg'].min()
            max_score = group['judge_avg'].max()
            if max_score > min_score:
                group['A_norm'] = (group['judge_avg'] - min_score) / (max_score - min_score)
            else:
                group['A_norm'] = 0.5

            # A_zscore: 周内Z-score
            if group['judge_avg'].std() > 0:
                group['A_zscore'] = (group['judge_avg'] - group['judge_avg'].mean()) / group['judge_avg'].std()
            else:
                group['A_zscore'] = 0.0

            for _, row in group.iterrows():
                records.append({
                    'celebrity': row['celebrity'],
                    'season': season,
                    'week': week,
                    'A_raw': row['A_raw'],
                    'A_norm': row['A_norm'],
                    'A_zscore': row['A_zscore']
                })

        ability_df = pd.DataFrame(records)

        # A_core: 跨周平均Z-score (每位选手的核心能力)
        core_ability = ability_df.groupby(['celebrity', 'season'])['A_zscore'].mean().reset_index()
        core_ability.columns = ['celebrity', 'season', 'A_core']

        ability_df = ability_df.merge(core_ability, on=['celebrity', 'season'], how='left')

        self.ability_metrics = ability_df
        print(f"Ability metrics calculated: {len(ability_df)} records")

        return ability_df

    def calculate_popularity_metrics(self):
        """
        计算人气指标

        指标包括:
        - P_raw: 原始观众支持率 (fan_percent_mean)
        - P_avg: 赛季平均观众支持率
        - P_peak: 赛季最高观众支持率
        - P_stability: 人气稳定性 (1 - CV, CV=std/mean)
        - RI: 人气增长指数 (Rising Index)
        """
        print("\n" + "=" * 60)
        print("Calculating Popularity Metrics")
        print("=" * 60)

        records = []

        # 按选手和赛季分组
        for (celebrity, season), group in self.data.groupby(['celebrity', 'season']):
            group = group.sort_values('week')

            # P_raw: 原始观众支持率
            fan_percents = group['fan_percent_mean'].values

            # P_avg: 赛季平均
            p_avg = np.mean(fan_percents)

            # P_peak: 赛季最高
            p_peak = np.max(fan_percents)

            # P_stability: 稳定性 (1 - 变异系数)
            if p_avg > 0:
                cv = np.std(fan_percents) / p_avg
                p_stability = max(0, 1 - cv)
            else:
                p_stability = 0

            # RI: 人气增长指数 (线性回归斜率)
            if len(fan_percents) >= 2:
                weeks = np.arange(len(fan_percents))
                slope, _, _, _, _ = stats.linregress(weeks, fan_percents)
                ri = slope * 100  # 转换为百分比增长率
            else:
                ri = 0

            for _, row in group.iterrows():
                records.append({
                    'celebrity': celebrity,
                    'season': season,
                    'week': row['week'],
                    'P_raw': row['fan_percent_mean'],
                    'P_avg': p_avg,
                    'P_peak': p_peak,
                    'P_stability': p_stability,
                    'RI': ri
                })

        popularity_df = pd.DataFrame(records)
        self.popularity_metrics = popularity_df
        print(f"Popularity metrics calculated: {len(popularity_df)} records")

        return popularity_df

    def calculate_classification_metrics(self):
        """
        计算分类指标并进行选手分类

        指标包括:
        - FDI: 粉丝依赖指数 (Fan Dependency Index)
        - PS: 表现得分 (Performance Score)
        - Category: 选手类型分类
        """
        print("\n" + "=" * 60)
        print("Calculating Classification Metrics")
        print("=" * 60)

        if self.ability_metrics is None or self.popularity_metrics is None:
            raise ValueError("Please calculate ability and popularity metrics first")

        # 合并能力和人气指标
        merged = pd.merge(
            self.ability_metrics,
            self.popularity_metrics,
            on=['celebrity', 'season', 'week'],
            how='inner'
        )

        records = []

        for (celebrity, season), group in merged.groupby(['celebrity', 'season']):
            # FDI: 粉丝依赖指数 = P_avg / (A_core + 0.1)
            # 加0.1避免除零，表示即使能力为0也有基础人气
            a_core = group['A_core'].iloc[0]
            p_avg = group['P_avg'].iloc[0]
            fdi = p_avg / (abs(a_core) + 0.1)

            # PS: 表现得分 = A_core * 0.6 + P_avg * 0.4
            ps = a_core * 0.6 + p_avg * 0.4

            # 分类逻辑
            if a_core > 0.5 and p_avg > 0.15:
                category = "Star"  # 明星型
            elif a_core > 0.5 and p_avg <= 0.15:
                category = "Skilled"  # 实力型
            elif a_core <= 0.5 and p_avg > 0.15:
                category = "Popular"  # 人气型
            else:
                category = "Average"  # 普通型

            for _, row in group.iterrows():
                records.append({
                    'celebrity': celebrity,
                    'season': season,
                    'week': row['week'],
                    'FDI': fdi,
                    'PS': ps,
                    'Category': category
                })

        classification_df = pd.DataFrame(records)
        self.classification_results = classification_df
        print(f"Classification completed: {len(classification_df)} records")
        print(f"Category distribution:")
        print(classification_df.groupby('Category')['celebrity'].nunique())

        return classification_df

    def get_contestant_summary(self, celebrity_name, season=None):
        """
        获取特定选手的完整指标摘要

        参数:
        - celebrity_name: 选手姓名
        - season: 赛季编号 (可选)
        """
        if self.classification_results is None:
            raise ValueError("Please run all calculations first")

        # Merge all metrics to get complete summary
        complete_data = self.ability_metrics.merge(
            self.popularity_metrics, on=['celebrity', 'season', 'week']
        ).merge(
            self.classification_results, on=['celebrity', 'season', 'week']
        )

        query = complete_data['celebrity'] == celebrity_name
        if season is not None:
            query = query & (complete_data['season'] == season)

        summary = complete_data[query].copy()

        if len(summary) == 0:
            print(f"No data found for {celebrity_name}")
            return None

        return summary


class MetricsVisualizer:
    """指标可视化器"""

    def __init__(self, calculator):
        """
        初始化可视化器

        参数:
        - calculator: MetricsCalculator实例
        """
        self.calc = calculator
        self.colors = COLOR_PALETTE

    def plot_metric_confidence_intervals(self, metric_name='P_raw', top_n=15, save=True):
        """
        可视化1: 单个指标的置信区间图
        展示Top N选手的指标值及其置信区间

        参数:
        - metric_name: 指标名称 (P_raw, A_raw, etc.)
        - top_n: 显示前N名选手
        - save: 是否保存图片
        """
        print(f"\n[Visualization 1] Plotting confidence intervals for {metric_name}")

        if self.calc.classification_results is None:
            raise ValueError("Please run all calculations first")

        # 计算每位选手的平均值和标准差
        if metric_name == 'P_raw':
            # Use popularity_metrics for P_raw
            data = self.calc.popularity_metrics.copy()
            summary = data.groupby('celebrity').agg({
                'P_raw': ['mean', 'std'],
                'season': 'first'
            }).reset_index()
            summary.columns = ['celebrity', 'mean', 'std', 'season']
            ylabel = 'Fan Support Rate'
            title = 'Top Contestants by Average Fan Support (with Confidence Intervals)'
        elif metric_name == 'A_raw':
            # Use ability_metrics for A_raw
            data = self.calc.ability_metrics.copy()
            summary = data.groupby('celebrity').agg({
                'A_raw': ['mean', 'std'],
                'season': 'first'
            }).reset_index()
            summary.columns = ['celebrity', 'mean', 'std', 'season']
            ylabel = 'Judge Score'
            title = 'Top Contestants by Average Judge Score (with Confidence Intervals)'
        else:
            print(f"Metric {metric_name} not supported for this visualization")
            return

        # 排序并取Top N
        summary = summary.sort_values('mean', ascending=False).head(top_n)

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))

        x_pos = np.arange(len(summary))
        means = summary['mean'].values
        stds = summary['std'].values

        # 绘制柱状图
        bars = ax.bar(x_pos, means, color=self.colors['primary'],
                     alpha=0.7, edgecolor='white', linewidth=1.5)

        # 添加误差线
        ax.errorbar(x_pos, means, yerr=stds, fmt='none',
                   ecolor=self.colors['danger'], capsize=5,
                   capthick=2, linewidth=2, alpha=0.8)

        # 设置样式
        ax.set_xlabel('Contestant', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['celebrity'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()

        if save:
            out_path = os.path.join(METRICS_PIC_ROOT, f'metric_confidence_{metric_name}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_metric_trends(self, celebrities, metric='P_raw', save=True):
        """
        可视化2: 多选手指标趋势对比图
        展示多位选手在整个赛季中的指标变化趋势

        参数:
        - celebrities: 选手名单 (list)
        - metric: 指标名称
        - save: 是否保存图片
        """
        print(f"\n[Visualization 2] Plotting metric trends for {len(celebrities)} contestants")

        if self.calc.classification_results is None:
            raise ValueError("Please run all calculations first")

        # Merge all metrics dataframes to get all columns
        ability_data = self.calc.ability_metrics.copy()
        popularity_data = self.calc.popularity_metrics.copy()
        classification_data = self.calc.classification_results.copy()

        # Merge on common keys
        merged_data = ability_data.merge(popularity_data, on=['celebrity', 'season', 'week'])
        merged_data = merged_data.merge(classification_data, on=['celebrity', 'season', 'week'])

        fig, ax = plt.subplots(figsize=(14, 8))

        # 为每位选手绘制趋势线
        colors = [self.colors['primary'], self.colors['secondary'],
                 self.colors['accent'], self.colors['success'],
                 self.colors['warning'], self.colors['danger']]

        for i, celebrity in enumerate(celebrities):
            data = merged_data[
                merged_data['celebrity'] == celebrity
            ].sort_values('week')

            if len(data) == 0:
                continue

            color = colors[i % len(colors)]

            # 绘制趋势线
            ax.plot(data['week'], data[metric],
                   marker='o', linewidth=2.5, markersize=8,
                   label=celebrity, color=color, alpha=0.8)

            # 添加阴影区域
            ax.fill_between(data['week'], data[metric], alpha=0.15, color=color)

        # 设置样式
        metric_labels = {
            'P_raw': 'Fan Support Rate',
            'A_raw': 'Judge Score',
            'FDI': 'Fan Dependency Index',
            'PS': 'Performance Score'
        }

        ax.set_xlabel('Week', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=14, fontweight='bold')
        ax.set_title(f'Contestant {metric_labels.get(metric, metric)} Trends Across Season',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save:
            out_path = os.path.join(METRICS_PIC_ROOT, f'metric_trends_{metric}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_metrics_correlation_heatmap(self, save=True):
        """
        可视化3: 指标间相关性热力图
        展示所有关键指标之间的相关系数

        参数:
        - save: 是否保存图片
        """
        print(f"\n[Visualization 3] Plotting metrics correlation heatmap")

        if self.calc.classification_results is None:
            raise ValueError("Please run all calculations first")

        # Merge all metrics dataframes to get all columns
        ability_data = self.calc.ability_metrics.copy()
        popularity_data = self.calc.popularity_metrics.copy()
        classification_data = self.calc.classification_results.copy()

        # Merge on common keys
        data = ability_data.merge(popularity_data, on=['celebrity', 'season', 'week'])
        data = data.merge(classification_data, on=['celebrity', 'season', 'week'])

        # 选择关键指标
        metrics_cols = ['A_raw', 'A_norm', 'A_zscore', 'A_core',
                       'P_raw', 'P_avg', 'P_peak', 'P_stability',
                       'RI', 'FDI', 'PS']

        # 计算相关系数矩阵
        corr_matrix = data[metrics_cols].corr()

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                   cmap='RdYlBu_r', center=0,
                   square=True, linewidths=1,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax, vmin=-1, vmax=1)

        # 设置样式
        ax.set_title('Metrics Correlation Heatmap',
                    fontsize=16, fontweight='bold', pad=20)

        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save:
            out_path = os.path.join(METRICS_PIC_ROOT, 'metrics_correlation_heatmap.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {out_path}")
        else:
            plt.show()

        plt.close()

    def plot_contestant_classification_scatter(self, season=None, save=True):
        """
        可视化4: 选手分类散点图
        基于能力和人气指标对选手进行分类可视化

        参数:
        - season: 指定赛季 (可选，None表示所有赛季)
        - save: 是否保存图片
        """
        print(f"\n[Visualization 4] Plotting contestant classification scatter")

        if self.calc.classification_results is None:
            raise ValueError("Please run all calculations first")

        # Merge all metrics dataframes to get all columns
        ability_data = self.calc.ability_metrics.copy()
        popularity_data = self.calc.popularity_metrics.copy()
        classification_data = self.calc.classification_results.copy()

        # Merge on common keys
        data = ability_data.merge(popularity_data, on=['celebrity', 'season', 'week'])
        data = data.merge(classification_data, on=['celebrity', 'season', 'week'])

        if season is not None:
            data = data[data['season'] == season]
            title_suffix = f' (Season {season})'
        else:
            title_suffix = ' (All Seasons)'

        # 按选手汇总
        summary = data.groupby(['celebrity', 'Category']).agg({
            'A_core': 'first',
            'P_avg': 'first'
        }).reset_index()

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 10))

        # 定义分类颜色
        category_colors = {
            'Star': self.colors['warning'],
            'Skilled': self.colors['primary'],
            'Popular': self.colors['secondary'],
            'Average': self.colors['neutral']
        }

        # 按类别绘制散点
        for category in summary['Category'].unique():
            subset = summary[summary['Category'] == category]
            ax.scatter(subset['A_core'], subset['P_avg'],
                      s=150, alpha=0.7, edgecolors='white', linewidth=2,
                      color=category_colors.get(category, self.colors['neutral']),
                      label=category)

        # 添加分界线
        ax.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        # 设置样式
        ax.set_xlabel('Core Ability (A_core)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Popularity (P_avg)', fontsize=14, fontweight='bold')
        ax.set_title(f'Contestant Classification{title_suffix}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save:
            season_str = f'_season{season}' if season else '_all'
            out_path = os.path.join(METRICS_PIC_ROOT, f'classification_scatter{season_str}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {out_path}")
        else:
            plt.show()

        plt.close()


def run_complete_analysis(data_path='2026_MCM_Problem_C_Data.csv',
                         cache_file='fan_est_cache.csv',
                         sample_celebrities=None):
    """
    运行完整的指标分析流程

    参数:
    - data_path: 原始数据文件路径
    - cache_file: 蒙特卡洛结果缓存文件
    - sample_celebrities: 示例选手列表（用于趋势图）

    返回:
    - calculator: MetricsCalculator实例
    - visualizer: MetricsVisualizer实例
    """
    print("\n" + "=" * 60)
    print("CONTESTANT METRICS ANALYSIS SYSTEM")
    print("=" * 60)

    # 步骤1: 导入必要的模块和数据
    from fan_percent_estimation import FanPercentEstimator

    print("\n[Step 1] Loading data and preprocessing...")
    estimator = FanPercentEstimator(data_path)
    estimator.preprocess_data()

    # 步骤2: 加载或计算蒙特卡洛结果
    print("\n[Step 2] Loading Monte Carlo results...")
    if os.path.exists(cache_file):
        monte_carlo_results = pd.read_csv(cache_file, encoding='utf-8-sig')
        print(f"Loaded {len(monte_carlo_results)} records from cache")
    else:
        print("Cache not found. Please run fan_percent_estimation.py first!")
        return None, None

    # 步骤3: 创建final_df
    print("\n[Step 3] Creating final dataframe...")
    final_df = estimator.create_final_df()

    # 步骤4: 初始化指标计算器
    print("\n[Step 4] Initializing metrics calculator...")
    calculator = MetricsCalculator(final_df, monte_carlo_results)

    # 步骤5: 计算所有指标
    print("\n[Step 5] Calculating all metrics...")
    calculator.calculate_ability_metrics()
    calculator.calculate_popularity_metrics()
    calculator.calculate_classification_metrics()

    # 步骤6: 保存结果
    print("\n[Step 6] Saving results...")
    # Merge all metrics for complete output
    complete_results = calculator.ability_metrics.merge(
        calculator.popularity_metrics, on=['celebrity', 'season', 'week']
    ).merge(
        calculator.classification_results, on=['celebrity', 'season', 'week']
    )
    complete_results.to_csv(
        os.path.join(METRICS_PIC_ROOT, 'contestant_metrics_complete.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"Results saved to: {os.path.join(METRICS_PIC_ROOT, 'contestant_metrics_complete.csv')}")

    # 步骤7: 创建可视化
    print("\n[Step 7] Creating visualizations...")
    visualizer = MetricsVisualizer(calculator)

    # 可视化1: 人气指标置信区间
    visualizer.plot_metric_confidence_intervals(metric_name='P_raw', top_n=15)

    # 可视化2: 能力指标置信区间
    visualizer.plot_metric_confidence_intervals(metric_name='A_raw', top_n=15)

    # 可视化3: 指标相关性热力图
    visualizer.plot_metrics_correlation_heatmap()

    # 可视化4: 选手分类散点图
    visualizer.plot_contestant_classification_scatter(season=None)

    # 可视化5: 示例选手趋势对比
    if sample_celebrities is None:
        # 自动选择Top 5选手
        top_celebrities = calculator.classification_results.groupby('celebrity')['PS'].mean().nlargest(5).index.tolist()
        sample_celebrities = top_celebrities

    if len(sample_celebrities) > 0:
        visualizer.plot_metric_trends(sample_celebrities, metric='P_raw')
        visualizer.plot_metric_trends(sample_celebrities, metric='PS')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED!")
    print(f"All visualizations saved to: {METRICS_PIC_ROOT}")
    print("=" * 60)

    return calculator, visualizer


if __name__ == '__main__':
    # 运行完整分析
    calculator, visualizer = run_complete_analysis()

    # 示例：查看特定选手的指标
    if calculator is not None:
        print("\n" + "=" * 60)
        print("Sample: Top 10 Contestants by Performance Score")
        print("=" * 60)
        # Merge all metrics to access all columns
        complete_data = calculator.ability_metrics.merge(
            calculator.popularity_metrics, on=['celebrity', 'season', 'week']
        ).merge(
            calculator.classification_results, on=['celebrity', 'season', 'week']
        )
        top_10 = complete_data.groupby('celebrity').agg({
            'PS': 'mean',
            'Category': 'first',
            'A_core': 'first',
            'P_avg': 'first'
        }).sort_values('PS', ascending=False).head(10)
        print(top_10)

