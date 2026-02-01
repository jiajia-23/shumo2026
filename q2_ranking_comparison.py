"""
问题2：评分方法对比与排名分析
=================================

任务目标：
1. 计算所有参赛选手的三种排名：
   - 原排名（actual placement）
   - 导师评分排名（judge-based ranking）
   - 另一评分方法排名（alternative scoring method）

2. 模拟比赛过程：
   - 基于比赛信息表格和预测的投票数
   - 按照另一评分法完整模拟比赛
   - 得出最终排名

3. 相关性分析：
   - 计算同季三个排名向量的两两斯皮尔曼系数
   - 评估哪种合并方法更偏向于观众

输出：
- 排名总表：选手名 | 季数 | 原排名 | 导师评分排名 | 另一评分方法排名
- 斯皮尔曼系数表格
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

# Visualization configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# 方案一：学术/莫兰迪配色 (Sophisticated Muted Palette)
# 方案二：北欧工业风格 (Nord-inspired Professional)
COLORS = {
    'actual_judge':  '#BF616A',      # 灰蓝色 (极高级的背景感)
    'actual_alt': '#5E81AC',        # 陶土色 (温暖的对比)
    'judge_alt': '#8FBCBB',         # 冰川青 (通透、轻盈)
    'rank_method': '#BF616A',       # 绛红 (重点突出而不突兀)
    'percent_method': '#5E81AC',    # 钢青 (标准、稳定)
}



# # 方案一：学术/莫兰迪配色 (Sophisticated Muted Palette)
# COLORS = {
#     'actual_judge': '#4A69BD',      # 雅致蓝 (沉稳、专业)
#     'actual_alt': '#E58E26',        # 晚霞橙 (温润、不刺眼)
#     'judge_alt': '#78E08F',         # 极光绿 (清新、辅助感强)
#     'rank_method': '#B33939',       # 砖红 (严肃、历史感)
#     'percent_method': '#218C74',    # 深海绿 (现代、数据感)
# }
print("=" * 60)
print("问题2：评分方法对比与排名分析")
print("=" * 60)


class RankingComparisonAnalyzer:
    """排名对比分析器"""

    def __init__(self, original_data_path, competition_info_path, fan_estimation_path, output_dir='Q2'):
        """
        初始化分析器

        Args:
            original_data_path: 原始数据文件路径
            competition_info_path: 比赛信息表格路径
            fan_estimation_path: 粉丝投票预测结果路径
            output_dir: 输出文件夹路径，默认为'Q2'
        """
        self.original_data_path = original_data_path
        self.competition_info_path = competition_info_path
        self.fan_estimation_path = fan_estimation_path
        self.output_dir = output_dir

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

        self.original_df = None
        self.competition_info = None
        self.fan_estimation = None
        self.processed_data = None
        self.ranking_table = None
        self.correlation_table = None

    def load_data(self):
        """加载所有必要的数据"""
        print("\n[Step 1] 加载数据...")

        # 加载原始数据
        self.original_df = pd.read_csv(self.original_data_path, encoding='utf-8-sig')
        print(f"  原始数据: {len(self.original_df)} 条记录")

        # 加载比赛信息表格
        self.competition_info = pd.read_csv(self.competition_info_path, encoding='utf-8-sig')
        print(f"  比赛信息: {len(self.competition_info)} 条记录")

        # 加载粉丝投票预测结果
        self.fan_estimation = pd.read_csv(self.fan_estimation_path, encoding='utf-8-sig')
        print(f"  粉丝投票预测: {len(self.fan_estimation)} 条记录")

        return self

    def preprocess_data(self):
        """预处理数据，提取每周的评委分和选手信息"""
        print("\n[Step 2] 数据预处理...")

        weekly_data = []

        # 遍历每个选手
        for idx, row in self.original_df.iterrows():
            celebrity = row['celebrity_name']
            season = row['season']
            results = row['results']

            # 获取原始排名（placement）
            placement = None
            if 'placement' in self.original_df.columns and pd.notna(row['placement']):
                try:
                    placement = float(row['placement'])
                except:
                    placement = None

            # 解析results获取最终排名
            final_rank = None
            exit_week = None

            if isinstance(results, str):
                if 'Eliminated Week' in results:
                    exit_week = int(results.split('Week ')[-1])
                elif '1st Place' in results:
                    final_rank = 1
                elif '2nd Place' in results:
                    final_rank = 2
                elif '3rd Place' in results:
                    final_rank = 3
                elif '4th Place' in results:
                    final_rank = 4
                elif results == 'Withdrew':
                    # 退赛选手需要找到退出周次
                    for week in range(1, 12):
                        judge_scores = []
                        for judge_num in range(1, 5):
                            col_name = f'week{week}_judge{judge_num}_score'
                            if col_name in self.original_df.columns:
                                score = row[col_name]
                                if pd.notna(score) and score != 0:
                                    judge_scores.append(float(score))
                        if len(judge_scores) == 0 and week > 1:
                            exit_week = week - 1
                            break

            # 遍历每一周，提取评委分
            for week in range(1, 12):
                judge_scores = []
                for judge_num in range(1, 5):
                    col_name = f'week{week}_judge{judge_num}_score'
                    if col_name in self.original_df.columns:
                        score = row[col_name]
                        if pd.notna(score) and score != 0:
                            judge_scores.append(float(score))

                if len(judge_scores) > 0:
                    avg_judge_score = np.mean(judge_scores)
                    is_exited = (exit_week == week)

                    if exit_week is None or week <= exit_week:
                        weekly_data.append({
                            'celebrity': celebrity,
                            'season': season,
                            'week': week,
                            'judge_score': avg_judge_score,
                            'is_exited': is_exited,
                            'exit_week': exit_week,
                            'final_rank': final_rank,
                            'placement': placement,
                            'results': results
                        })

        self.processed_data = pd.DataFrame(weekly_data)
        print(f"  处理完成: {len(self.processed_data)} 条周记录")

        return self.processed_data

    def calculate_actual_ranking(self):
        """计算原排名（基于placement或final_rank）"""
        print("\n[Step 3] 计算原排名...")

        actual_rankings = []

        for season in sorted(self.processed_data['season'].unique()):
            season_data = self.processed_data[self.processed_data['season'] == season]

            # 获取该赛季所有选手的最终排名
            contestants = season_data.groupby('celebrity').agg({
                'placement': 'first',
                'final_rank': 'first'
            }).reset_index()

            # 使用placement或final_rank作为原排名
            for _, row in contestants.iterrows():
                rank = row['placement'] if pd.notna(row['placement']) else row['final_rank']
                actual_rankings.append({
                    'celebrity': row['celebrity'],
                    'season': season,
                    'actual_rank': rank
                })

        actual_rank_df = pd.DataFrame(actual_rankings)
        print(f"  原排名计算完成: {len(actual_rank_df)} 位选手")

        return actual_rank_df

    def calculate_judge_ranking(self):
        """计算导师评分排名（基于整季平均评委分）"""
        print("\n[Step 4] 计算导师评分排名...")

        judge_rankings = []

        for season in sorted(self.processed_data['season'].unique()):
            season_data = self.processed_data[self.processed_data['season'] == season]

            # 计算每位选手的平均评委分
            avg_scores = season_data.groupby('celebrity')['judge_score'].mean().reset_index()
            avg_scores.columns = ['celebrity', 'avg_judge_score']

            # 根据平均分排名（分数越高，排名越好，rank值越小）
            avg_scores['judge_rank'] = rankdata(-avg_scores['avg_judge_score'], method='min')

            for _, row in avg_scores.iterrows():
                judge_rankings.append({
                    'celebrity': row['celebrity'],
                    'season': season,
                    'judge_rank': row['judge_rank'],
                    'avg_judge_score': row['avg_judge_score']
                })

        judge_rank_df = pd.DataFrame(judge_rankings)
        print(f"  导师评分排名计算完成: {len(judge_rank_df)} 位选手")

        return judge_rank_df

    def simulate_alternative_scoring(self):
        """
        模拟另一评分方法并计算排名

        评分方法：
        - S1-2: 排名系统（Judge Rank + Fan Rank，越小越好）
        - S3-27: 百分比系统（Judge% + Fan%，越大越好）
        - S28+: 排名系统 + Judge Save
        """
        print("\n[Step 5] 模拟另一评分方法...")

        alternative_rankings = []

        for season in sorted(self.processed_data['season'].unique()):
            print(f"  处理 Season {season}...")

            season_data = self.processed_data[self.processed_data['season'] == season].copy()
            season_fan_est = self.fan_estimation[self.fan_estimation['season'] == season].copy()

            # 获取该赛季的所有周次
            weeks = sorted(season_data['week'].unique())

            # 记录每位选手的淘汰周次（模拟结果）
            simulated_exit = {}
            remaining_contestants = set(season_data['celebrity'].unique())

            # 逐周模拟比赛
            for week in weeks:
                week_data = season_data[season_data['week'] == week].copy()
                week_fan = season_fan_est[season_fan_est['week'] == week].copy()

                # 只保留还在比赛中的选手
                week_data = week_data[week_data['celebrity'].isin(remaining_contestants)]

                if len(week_data) == 0:
                    continue

                # 合并评委分和粉丝投票预测
                week_data = week_data.merge(
                    week_fan[['celebrity', 'fan_percent_mean']],
                    on='celebrity',
                    how='left'
                )

                # 填充缺失的粉丝投票（如果有）
                week_data['fan_percent_mean'].fillna(1.0 / len(week_data), inplace=True)

                # 根据赛季确定评分系统
                if 3 <= season < 28:
                    # 百分比系统：归一化评委分和粉丝投票
                    judge_total = week_data['judge_score'].sum()
                    week_data['judge_percent'] = week_data['judge_score'] / judge_total

                    # 粉丝投票已经是百分比
                    week_data['fan_percent'] = week_data['fan_percent_mean']

                    # 总分 = 评委% + 粉丝%（越大越好）
                    week_data['total_score'] = week_data['judge_percent'] + week_data['fan_percent']

                    # 排名（分数越高，排名越好）
                    week_data['rank'] = rankdata(-week_data['total_score'], method='min')
                else:
                    # 排名系统：分别排名后相加
                    week_data['judge_rank'] = rankdata(-week_data['judge_score'], method='min')
                    week_data['fan_rank'] = rankdata(-week_data['fan_percent_mean'], method='min')

                    # 总分 = 评委排名 + 粉丝排名（越小越好）
                    week_data['total_score'] = week_data['judge_rank'] + week_data['fan_rank']

                    # 排名（分数越小，排名越好）
                    week_data['rank'] = rankdata(week_data['total_score'], method='min')

                # 找出本周应该淘汰的选手（根据原始数据）
                original_week = season_data[season_data['week'] == week]
                exited_this_week = original_week[original_week['is_exited'] == True]['celebrity'].tolist()

                # 记录淘汰信息
                for celebrity in exited_this_week:
                    if celebrity in remaining_contestants:
                        simulated_exit[celebrity] = week
                        remaining_contestants.remove(celebrity)

            # 为所有选手分配最终排名
            # 决赛选手（没有被淘汰的）根据最后一周的排名
            finals_week = max(weeks)
            finals_data = season_data[season_data['week'] == finals_week].copy()
            finals_fan = season_fan_est[season_fan_est['week'] == finals_week].copy()

            finals_data = finals_data.merge(
                finals_fan[['celebrity', 'fan_percent_mean']],
                on='celebrity',
                how='left'
            )
            finals_data['fan_percent_mean'].fillna(1.0 / len(finals_data), inplace=True)

            # 计算决赛排名
            if 3 <= season < 28:
                judge_total = finals_data['judge_score'].sum()
                finals_data['judge_percent'] = finals_data['judge_score'] / judge_total
                finals_data['total_score'] = finals_data['judge_percent'] + finals_data['fan_percent_mean']
                finals_data['alternative_rank'] = rankdata(-finals_data['total_score'], method='min')
            else:
                finals_data['judge_rank'] = rankdata(-finals_data['judge_score'], method='min')
                finals_data['fan_rank'] = rankdata(-finals_data['fan_percent_mean'], method='min')
                finals_data['total_score'] = finals_data['judge_rank'] + finals_data['fan_rank']
                finals_data['alternative_rank'] = rankdata(finals_data['total_score'], method='min')

            # 记录所有选手的排名
            all_contestants = season_data['celebrity'].unique()
            for celebrity in all_contestants:
                if celebrity in finals_data['celebrity'].values:
                    rank = finals_data[finals_data['celebrity'] == celebrity]['alternative_rank'].iloc[0]
                else:
                    # 被淘汰的选手，排名根据淘汰顺序
                    exit_week = simulated_exit.get(celebrity, None)
                    if exit_week:
                        # 同一周淘汰的选手排名相同
                        same_week_exits = [c for c, w in simulated_exit.items() if w == exit_week]
                        rank = len(all_contestants) - len(same_week_exits) + 1
                    else:
                        rank = None

                alternative_rankings.append({
                    'celebrity': celebrity,
                    'season': season,
                    'alternative_rank': rank
                })

        alternative_rank_df = pd.DataFrame(alternative_rankings)
        print(f"  另一评分方法排名计算完成: {len(alternative_rank_df)} 位选手")

        return alternative_rank_df

    def calculate_spearman_correlations(self, ranking_table):
        """
        计算同季三个排名向量的两两斯皮尔曼相关系数

        Args:
            ranking_table: 包含三种排名的表格

        Returns:
            相关系数表格
        """
        print("\n[Step 6] 计算斯皮尔曼相关系数...")

        correlation_records = []

        for season in sorted(ranking_table['season'].unique()):
            season_data = ranking_table[ranking_table['season'] == season].copy()

            # 移除缺失值
            season_data = season_data.dropna(subset=['actual_rank', 'judge_rank', 'alternative_rank'])

            if len(season_data) < 3:
                print(f"  Season {season}: 数据不足，跳过")
                continue

            # 计算三个排名向量
            actual = season_data['actual_rank'].values
            judge = season_data['judge_rank'].values
            alternative = season_data['alternative_rank'].values

            # 计算两两相关系数
            corr_actual_judge, p_aj = spearmanr(actual, judge)
            corr_actual_alt, p_aa = spearmanr(actual, alternative)
            corr_judge_alt, p_ja = spearmanr(judge, alternative)

            correlation_records.append({
                'season': season,
                'n_contestants': len(season_data),
                'corr_actual_judge': corr_actual_judge,
                'corr_actual_alternative': corr_actual_alt,
                'corr_judge_alternative': corr_judge_alt,
                'p_value_actual_judge': p_aj,
                'p_value_actual_alternative': p_aa,
                'p_value_judge_alternative': p_ja
            })

        correlation_df = pd.DataFrame(correlation_records)
        print(f"  相关系数计算完成: {len(correlation_df)} 个赛季")

        return correlation_df

    def generate_ranking_table(self):
        """生成排名总表：选手名 | 季数 | 原排名 | 导师评分排名 | 另一评分方法排名"""
        print("\n[Step 7] 生成排名总表...")

        # 计算三种排名
        actual_rank_df = self.calculate_actual_ranking()
        judge_rank_df = self.calculate_judge_ranking()
        alternative_rank_df = self.simulate_alternative_scoring()

        # 合并三种排名
        ranking_table = actual_rank_df.merge(
            judge_rank_df[['celebrity', 'season', 'judge_rank']],
            on=['celebrity', 'season'],
            how='outer'
        )

        ranking_table = ranking_table.merge(
            alternative_rank_df[['celebrity', 'season', 'alternative_rank']],
            on=['celebrity', 'season'],
            how='outer'
        )

        # 重新排列列顺序
        ranking_table = ranking_table[['celebrity', 'season', 'actual_rank', 'judge_rank', 'alternative_rank']]

        # 排序
        ranking_table = ranking_table.sort_values(['season', 'actual_rank'])

        self.ranking_table = ranking_table
        print(f"  排名总表生成完成: {len(ranking_table)} 位选手")

        return ranking_table

    def save_results(self, ranking_output='ranking_comparison_table.csv',
                     correlation_output='spearman_correlation_table.csv'):
        """保存结果到CSV文件"""
        print("\n[Step 8] 保存结果...")

        if self.ranking_table is None or self.correlation_table is None:
            raise ValueError("请先运行 run_analysis() 生成结果")

        # 保存排名总表
        ranking_path = os.path.join(self.output_dir, ranking_output)
        self.ranking_table.to_csv(ranking_path, index=False, encoding='utf-8-sig')
        print(f"  排名总表已保存到: {ranking_path}")

        # 保存相关系数表格
        correlation_path = os.path.join(self.output_dir, correlation_output)
        self.correlation_table.to_csv(correlation_path, index=False, encoding='utf-8-sig')
        print(f"  相关系数表格已保存到: {correlation_path}")

    def visualize_correlation_bars(self, save=True):
        """
        Visualization 1: Bar charts showing 3 Spearman coefficients per season
        Grouped by scoring system (Rank Method vs Percent Method)
        """
        print("\n[Visualization 1] Generating correlation bar charts...")

        if self.correlation_table is None:
            raise ValueError("Please run run_analysis() first")

        # Add scoring method classification
        self.correlation_table['Method'] = self.correlation_table['season'].apply(
            lambda s: 'Rank Method' if (s <= 2 or s >= 28) else 'Percent Method'
        )

        # Create two separate figures for each method
        methods = ['Rank Method', 'Percent Method']

        for method in methods:
            method_data = self.correlation_table[self.correlation_table['Method'] == method].copy()

            if len(method_data) == 0:
                continue

            fig, ax = plt.subplots(figsize=(16, 8))

            seasons = method_data['season'].values
            x = np.arange(len(seasons))
            width = 0.25

            # Three correlations
            corr1 = method_data['corr_actual_judge'].values
            corr2 = method_data['corr_actual_alternative'].values
            corr3 = method_data['corr_judge_alternative'].values

            # Plot bars
            bars1 = ax.bar(x - width, corr1, width, label='Actual vs Judge',
                          color=COLORS['actual_judge'], alpha=0.6, edgecolor='black', linewidth=2)
            bars2 = ax.bar(x, corr2, width, label='Actual vs Alternative',
                          color=COLORS['actual_alt'], alpha=0.6, edgecolor='black', linewidth=2)
            bars3 = ax.bar(x + width, corr3, width, label='Judge vs Alternative',
                          color=COLORS['judge_alt'], alpha=0.6, edgecolor='black', linewidth=2)

            # Formatting
            ax.set_xlabel('Season', fontsize=13, fontweight='bold')
            ax.set_ylabel('Spearman Correlation Coefficient', fontsize=13, fontweight='bold')
            ax.set_title(f'Spearman Correlation Analysis by Season\n({method})',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f'S{s}' for s in seasons], rotation=45, ha='right')
            ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)

            # 方案一：低饱和度专业配色
            # 颜色选取：柔和绿 (#20bf6b), 雅致金 (#f7b731), 烟粉色 (#eb3b5a)'judge_alt': '#8FBCBB',         # 冰川青 (通透、轻盈)
            thresholds = [0.9, 0.7, 0.4]
            colors = ['#218C74', '#BF616A', '#5E81AC']
            labels = ['Strong', 'Moderate', 'Weak']

            for val, col, lab in zip(thresholds, colors, labels):
                # 使用较小的 alpha 和虚线，使其退为背景，不干扰主体数据
                ax.axhline(y=val, color=col, linestyle='--', alpha=0.4, linewidth=2.4, zorder=1)
                
                # 在右侧边缘添加半透明的数值标注，增加可读性
                ax.text(ax.get_xlim()[1], val, f' {val} ({lab})', 
                        color=col, va='center', ha='left', fontsize=9, fontweight='bold', alpha=0.6)

            plt.tight_layout()

            if save:
                filename = f'correlation_bars_{method.replace(" ", "_").lower()}.png'
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            else:
                plt.show()

            plt.close()

        return self

    def visualize_correlation_trends(self, save=True):
        """
        Visualization 2: Line charts showing 3 Spearman coefficients across seasons
        - Different colored/styled lines for each coefficient
        - Background colors to distinguish scoring systems
        - Horizontal zones with labels for correlation strength levels
        """
        print("\n[Visualization 2] Generating correlation trend lines...")

        if self.correlation_table is None:
            raise ValueError("Please run run_analysis() first")

        fig, ax = plt.subplots(figsize=(20, 10))

        # Sort by season
        data = self.correlation_table.sort_values('season').copy()
        seasons = data['season'].values

        # Three correlations with different styles
        corr1 = data['corr_actual_judge'].values
        corr2 = data['corr_actual_alternative'].values
        corr3 = data['corr_judge_alternative'].values

        # Plot lines with different styles
        ax.plot(seasons, corr1, 'o-', linewidth=2.5, markersize=8,
               color='#2E86AB', label='Actual vs Judge', alpha=0.9)
        ax.plot(seasons, corr2, 's--', linewidth=2.5, markersize=8,
               color='#F18F01', label='Actual vs Alternative', alpha=0.9)
        ax.plot(seasons, corr3, '^-.', linewidth=2.5, markersize=8,
               color='#06A77D', label='Judge vs Alternative', alpha=0.9)

        # Add background colors for different scoring systems
        rank_seasons_1 = [s for s in seasons if s <= 2]
        rank_seasons_2 = [s for s in seasons if s >= 28]
        percent_seasons = [s for s in seasons if 3 <= s < 28]

        if len(rank_seasons_1) > 0:
            ax.axvspan(min(rank_seasons_1) - 0.5, max(rank_seasons_1) + 0.5,
                      alpha=0.15, color='#E63946', label='Rank Method')
        if len(percent_seasons) > 0:
            ax.axvspan(min(percent_seasons) - 0.5, max(percent_seasons) + 0.5,
                      alpha=0.15, color='#457B9D', label='Percent Method')
        if len(rank_seasons_2) > 0:
            ax.axvspan(min(rank_seasons_2) - 0.5, max(rank_seasons_2) + 0.5,
                      alpha=0.15, color='#E63946')

        # Add horizontal zones for correlation strength
        ax.axhspan(0.90, 1.00, alpha=0.1, color='green')
        ax.text(seasons[-1] + 1, 0.95, 'Very Strong\n(0.90-1.00)\nHighly Consistent',
               fontsize=10, va='center', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        ax.axhspan(0.70, 0.89, alpha=0.1, color='yellow')
        ax.text(seasons[-1] + 1, 0.795, 'Strong\n(0.70-0.89)\nWell Preserved',
               fontsize=10, va='center', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

        ax.axhspan(0.40, 0.69, alpha=0.1, color='orange')
        ax.text(seasons[-1] + 1, 0.545, 'Moderate\n(0.40-0.69)\nControversial',
               fontsize=10, va='center', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5B4', alpha=0.7))

        ax.axhspan(0.00, 0.39, alpha=0.1, color='red')
        ax.text(seasons[-1] + 1, 0.195, 'Weak\n(0.00-0.39)\nFan Dominated',
               fontsize=10, va='center', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFB6C1', alpha=0.7))

        # Formatting
        ax.set_xlabel('Season', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spearman Correlation Coefficient', fontsize=14, fontweight='bold')
        ax.set_title('Spearman Correlation Trends Across All Seasons\n' +
                    'Background Colors: Scoring System | Horizontal Zones: Correlation Strength',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlim(seasons[0] - 1, seasons[-1] + 8)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper left', fontsize=12, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'correlation_trends_all_seasons.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()
        return self

    def visualize_season_ranking_comparison(self, season, save=True):
        """
        Visualization 3: Ranking comparison line chart for a specific season
        - X-axis: Contestant names (diagonal, bold)
        - Y-axis: Ranking
        - Three lines for three ranking dimensions
        """
        print(f"\n[Visualization 3] Generating ranking comparison for Season {season}...")

        if self.ranking_table is None:
            raise ValueError("Please run run_analysis() first")

        # Filter data for the specific season
        season_data = self.ranking_table[self.ranking_table['season'] == season].copy()
        season_data = season_data.dropna(subset=['actual_rank', 'judge_rank', 'alternative_rank'])

        if len(season_data) == 0:
            print(f"  No data available for Season {season}")
            return self

        # Sort by actual rank
        season_data = season_data.sort_values('actual_rank')

        fig, ax = plt.subplots(figsize=(16, 10))

        celebrities = season_data['celebrity'].values
        x = np.arange(len(celebrities))

        # Three ranking dimensions
        actual_ranks = season_data['actual_rank'].values
        judge_ranks = season_data['judge_rank'].values
        alt_ranks = season_data['alternative_rank'].values

        # Plot lines with markers
        ax.plot(x, actual_ranks, 'o-', linewidth=2.5, markersize=10,
               color='#2E86AB', label='Actual Ranking', alpha=0.9)
        ax.plot(x, judge_ranks, 's--', linewidth=2.5, markersize=10,
               color='#F18F01', label='Judge-Based Ranking', alpha=0.9)
        ax.plot(x, alt_ranks, '^-.', linewidth=2.5, markersize=10,
               color='#06A77D', label='Alternative Ranking', alpha=0.9)

        # Formatting
        ax.set_xlabel('Contestant', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ranking (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_title(f'Ranking Comparison for Season {season}\nThree Ranking Dimensions',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(celebrities, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax.invert_yaxis()  # Lower rank number = better position
        ax.legend(loc='best', fontsize=12, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save:
            filename = f'ranking_comparison_season_{season}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()
        return self

    def generate_special_cases_table(self, save=True):
        """
        Visualization 4: Generate table highlighting special controversial cases
        Identifies contestants with large ranking discrepancies
        """
        print("\n[Visualization 4] Generating special cases table...")

        if self.ranking_table is None:
            raise ValueError("Please run run_analysis() first")

        # Calculate ranking discrepancies
        self.ranking_table['judge_discrepancy'] = abs(
            self.ranking_table['actual_rank'] - self.ranking_table['judge_rank']
        )
        self.ranking_table['alt_discrepancy'] = abs(
            self.ranking_table['actual_rank'] - self.ranking_table['alternative_rank']
        )

        # Find controversial cases
        controversial = self.ranking_table.nlargest(15, 'alt_discrepancy')[
            ['celebrity', 'season', 'actual_rank', 'judge_rank', 'alternative_rank',
             'judge_discrepancy', 'alt_discrepancy']
        ].copy()

        # Create visualization table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = [['Contestant', 'Season', 'Actual\nRank', 'Judge\nRank',
                      'Alternative\nRank', 'Judge\nDiscrepancy', 'Alt\nDiscrepancy']]

        for _, row in controversial.iterrows():
            table_data.append([
                row['celebrity'], f"S{int(row['season'])}", f"{int(row['actual_rank'])}",
                f"{int(row['judge_rank'])}", f"{int(row['alternative_rank'])}",
                f"{int(row['judge_discrepancy'])}", f"{int(row['alt_discrepancy'])}"
            ])

        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(7):
                table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')

        plt.title('Top 15 Controversial Cases: Largest Ranking Discrepancies',
                 fontsize=16, fontweight='bold', pad=20)

        if save:
            png_path = os.path.join(self.output_dir, 'special_cases_table.png')
            csv_path = os.path.join(self.output_dir, 'special_cases_data.csv')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {png_path}")
            controversial.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  Saved: {csv_path}")
        else:
            plt.show()

        plt.close()
        return self

    def visualize_advanced_evaluation(self, save=True):
        """
        Visualization 5: Generate 4-panel advanced evaluation figure
        - Panel 1: Correlation heatmap between different Spearman coefficients
        - Panel 2: Distribution boxplots of correlations
        - Panel 3: Scoring system comparison (average correlations)
        - Panel 4: Consistency score over time
        """
        print("\n[Visualization 5] Creating advanced evaluation figure...")

        if self.correlation_table is None:
            raise ValueError("Please run run_analysis() first")

        # Create working dataframe with renamed columns for consistency
        results_df = self.correlation_table.copy()
        results_df.rename(columns={
            'corr_actual_judge': 'spearman_actual_judge',
            'corr_actual_alternative': 'spearman_actual_alt',
            'corr_judge_alternative': 'spearman_judge_alt'
        }, inplace=True)

        # Add scoring system classification
        results_df['scoring_system'] = results_df['season'].apply(
            lambda s: 'Rank Method' if (s <= 2 or s >= 28) else 'Percent Method'
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Evaluation: Correlation Analysis Across Seasons',
                     fontsize=16, fontweight='bold', y=0.995)

        # Panel 1: Correlation heatmap
        ax1 = axes[0, 0]
        corr_matrix = results_df[['spearman_actual_judge',
                                   'spearman_actual_alt',
                                   'spearman_judge_alt']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Correlation Between Different Spearman Coefficients',
                     fontweight='bold', pad=10)
        ax1.set_xticklabels(['Actual-Judge', 'Actual-Alt', 'Judge-Alt'], rotation=45)
        ax1.set_yticklabels(['Actual-Judge', 'Actual-Alt', 'Judge-Alt'], rotation=0)

        # Panel 2: Distribution of correlations
        ax2 = axes[0, 1]
        data_to_plot = [
            results_df['spearman_actual_judge'].dropna(),
            results_df['spearman_actual_alt'].dropna(),
            results_df['spearman_judge_alt'].dropna()
        ]
        bp = ax2.boxplot(data_to_plot, labels=['Actual-Judge', 'Actual-Alt', 'Judge-Alt'],
                        patch_artist=True, showmeans=True)
        colors = [COLORS['actual_judge'], COLORS['actual_alt'], COLORS['judge_alt']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel('Spearman Correlation', fontweight='bold')
        ax2.set_title('Distribution of Correlation Coefficients', fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong (0.7)')
        ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.4)')
        ax2.legend(loc='lower right', fontsize=8)

        # Panel 3: Scoring system comparison
        ax3 = axes[1, 0]
        rank_seasons = results_df[results_df['scoring_system'] == 'Rank Method']
        percent_seasons = results_df[results_df['scoring_system'] == 'Percent Method']

        x = np.arange(3)
        width = 0.35

        rank_means = [
            rank_seasons['spearman_actual_judge'].mean(),
            rank_seasons['spearman_actual_alt'].mean(),
            rank_seasons['spearman_judge_alt'].mean()
        ]
        percent_means = [
            percent_seasons['spearman_actual_judge'].mean(),
            percent_seasons['spearman_actual_alt'].mean(),
            percent_seasons['spearman_judge_alt'].mean()
        ]

        bars1 = ax3.bar(x - width/2, rank_means, width, label='Rank Method',
                       color=COLORS['rank_method'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, percent_means, width, label='Percent Method',
                       color=COLORS['percent_method'], alpha=0.8)

        ax3.set_ylabel('Average Spearman Correlation', fontweight='bold')
        ax3.set_title('Average Correlations by Scoring System', fontweight='bold', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Actual-Judge', 'Actual-Alt', 'Judge-Alt'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)

        # Panel 4: Consistency score over time
        ax4 = axes[1, 1]
        results_df['consistency_score'] = (
            results_df['spearman_actual_judge'] +
            results_df['spearman_actual_alt'] +
            results_df['spearman_judge_alt']
        ) / 3

        rank_data = results_df[results_df['scoring_system'] == 'Rank Method']
        percent_data = results_df[results_df['scoring_system'] == 'Percent Method']

        ax4.plot(rank_data['season'], rank_data['consistency_score'],
                marker='o', color=COLORS['rank_method'], label='Rank Method',
                linewidth=2, markersize=6)
        ax4.plot(percent_data['season'], percent_data['consistency_score'],
                marker='s', color=COLORS['percent_method'], label='Percent Method',
                linewidth=2, markersize=6)

        ax4.set_xlabel('Season', fontweight='bold')
        ax4.set_ylabel('Average Consistency Score', fontweight='bold')
        ax4.set_title('Ranking Consistency Over Seasons', fontweight='bold', pad=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add correlation strength zones
        ax4.axhspan(0.9, 1.0, alpha=0.1, color='darkgreen', label='Very Strong')
        ax4.axhspan(0.7, 0.9, alpha=0.1, color='green')
        ax4.axhspan(0.4, 0.7, alpha=0.1, color='orange')
        ax4.axhspan(0.0, 0.4, alpha=0.1, color='red')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'advanced_evaluation.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        else:
            plt.show()

        plt.close()
        return self

    def display_summary(self):
        """显示分析结果摘要"""
        print("\n" + "=" * 60)
        print("分析结果摘要")
        print("=" * 60)

        if self.ranking_table is None or self.correlation_table is None:
            print("请先运行 run_analysis() 生成结果")
            return

        # 排名表摘要
        print("\n【排名总表摘要】")
        print(f"总选手数: {len(self.ranking_table)}")
        print(f"总赛季数: {self.ranking_table['season'].nunique()}")
        print(f"\n前10行数据:")
        print(self.ranking_table.head(10).to_string())

        # 相关系数摘要
        print("\n【斯皮尔曼相关系数摘要】")
        print(f"分析赛季数: {len(self.correlation_table)}")
        print(f"\n平均相关系数:")
        print(f"  原排名 vs 导师评分排名: {self.correlation_table['corr_actual_judge'].mean():.4f}")
        print(f"  原排名 vs 另一评分方法: {self.correlation_table['corr_actual_alternative'].mean():.4f}")
        print(f"  导师评分 vs 另一评分方法: {self.correlation_table['corr_judge_alternative'].mean():.4f}")
        print(f"\n相关系数表格:")
        print(self.correlation_table.to_string())

    def run_analysis(self):
        """运行完整的分析流程"""
        print("\n" + "=" * 60)
        print("开始运行完整分析流程")
        print("=" * 60)

        # Step 1-2: 加载和预处理数据
        self.load_data()
        self.preprocess_data()

        # Step 3-7: 生成排名总表
        self.generate_ranking_table()

        # Step 6: 计算相关系数
        self.correlation_table = self.calculate_spearman_correlations(self.ranking_table)

        # Step 8: 保存结果
        self.save_results()

        # 显示摘要
        self.display_summary()

        # Generate visualizations
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        # Visualization 1: Correlation bar charts by scoring method
        self.visualize_correlation_bars(save=True)

        # Visualization 2: Correlation trend lines across all seasons
        self.visualize_correlation_trends(save=True)

        # Visualization 3: Ranking comparison for specific controversial seasons
        # Identify seasons with low correlation (controversial)
        controversial_seasons = self.correlation_table[
            self.correlation_table['corr_actual_alternative'] < 0.7
        ]['season'].tolist()

        if len(controversial_seasons) > 0:
            print(f"\n  Generating ranking comparisons for {len(controversial_seasons)} controversial seasons...")
            for season in controversial_seasons[:5]:  # Limit to top 5 most controversial
                self.visualize_season_ranking_comparison(season, save=True)

        # Visualization 4: Special cases table
        self.generate_special_cases_table(save=True)

        # Visualization 5: Advanced evaluation (4-panel figure)
        self.visualize_advanced_evaluation(save=True)

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("问题2：评分方法对比与排名分析")
    print("=" * 60)

    # 初始化分析器
    analyzer = RankingComparisonAnalyzer(
        original_data_path='2026_MCM_Problem_C_Data.csv',
        competition_info_path='competition_info_table.csv',
        fan_estimation_path='fan_percent_estimation_results.csv'
    )

    # 运行完整分析
    analyzer.run_analysis()


if __name__ == '__main__':
    main()






