"""
观众分估计模型 - 使用蒙特卡洛采样方法
基于评委分和淘汰结果反推观众分

问题设定：
- Total = JudgePercent + FanPercent
- 观众分 FanPercent 不可见，需要估计
- 每周被淘汰的选手 Total 必须是全场最低
- 约束：∑FanPercent = 1.0
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 统一图片保存路径
PIC_ROOT = os.path.join('Q1', 'pic')
os.makedirs(PIC_ROOT, exist_ok=True)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DWTSAdvAnalysis:
    def __init__(self, df):
        self.df = df
        self.indicator_df = None




    def calculate_indicators(self):
        """计算四个维度的导师打分指标"""
        records = []
        # 复用您的预处理逻辑提取每周平均分
        # 此处简化演示，假设 df 已包含基本列: celebrity, season, week, judge_avg, is_eliminated
        
        def get_group_metrics(group):
            group = group.copy()
            # 1. 归一化排名 (0=最高, 1=最低)
            group['judge_rank_norm'] = group['judge_avg'].rank(ascending=False, method='min') / len(group)
            # 2. Z-Score (标准化得分)
            if len(group) > 1 and group['judge_avg'].std() > 0:
                group['z_score'] = (group['judge_avg'] - group['judge_avg'].mean()) / group['judge_avg'].std()
            else:
                group['z_score'] = 0
            # 3. 末位差距
            group['gap_to_bottom'] = group['judge_avg'] - group['judge_avg'].min()
            # 4. 是否为导师最低分
            group['is_judge_lowest'] = (group['judge_avg'] == group['judge_avg'].min()).astype(int)
            return group

        self.indicator_df = self.df.groupby(['season', 'week'], group_keys=False).apply(get_group_metrics)
        print("指标计算完成。")

    def plot_judge_influence_trend(self, save=True):
        """纵向：估算每季导师影响力权重并绘图"""
        influence_data = []
        for s in sorted(self.indicator_df['season'].unique()):
            season_data = self.indicator_df[self.indicator_df['season'] == s]
            # 过滤掉没有淘汰发生的周次以避免计算相关系数失败
            valid_weeks = season_data.groupby('week')['is_eliminated'].sum()
            valid_weeks = valid_weeks[valid_weeks > 0].index
            filtered = season_data[season_data['week'].isin(valid_weeks)]
            if not filtered.empty:
                # 使用 Rank 与 Elimination 的相关性作为影响力指标
                corr = filtered['judge_rank_norm'].corr(filtered['is_eliminated'], method='spearman')
                influence_data.append({'season': s, 'influence_weight': corr})
        inf_df = pd.DataFrame(influence_data)
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=inf_df, x='season', y='influence_weight', marker='o', color='darkred')
        plt.fill_between(inf_df['season'], inf_df['influence_weight'], alpha=0.2, color='red')
        plt.title('Estimated Judge Influence Weight Across Seasons (Spearman Corr)')
        plt.ylabel('Influence Weight (0 to 1)')
        plt.grid(True, linestyle='--', alpha=0.6)
        if save:
            out_path = os.path.join(PIC_ROOT, 'judge_influence_trend.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {out_path}")
        else:
            plt.show()
        plt.close()

    def plot_season_week_heatmap(self, save=True):
        """纵向：季节-周次 淘汰者导师排名热力图"""
        elim_data = self.indicator_df[self.indicator_df['is_eliminated'] == 1]
        pivot_table = elim_data.pivot_table(index='season', columns='week', values='judge_rank_norm')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Judge Rank (1.0 = Lowest)'})
        plt.title('Relative Judge Rank of Eliminated Contestants\n(Redder = Judges accurately predicted the exit)')
        if save:
            out_path = os.path.join(PIC_ROOT, 'season_week_heatmap.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {out_path}")
        else:
            plt.show()
        plt.close()

    def plot_celebrity_score_heatmap(self, target_season, save=True):
        """横向：单季内不同选手的导师分走势热力图"""
        season_df = self.indicator_df[self.indicator_df['season'] == target_season]
        # 转换数据：选手为行，周次为列，数值为导师平均分
        pivot_scores = season_df.pivot(index='celebrity', columns='week', values='judge_avg')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_scores, cmap='Blues', annot=True, fmt=".1f")
        plt.title(f'Judge Scores Heatmap - Season {target_season}')
        if save:
            out_path = os.path.join(PIC_ROOT, f'celebrity_score_heatmap_season{target_season}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {out_path}")
        else:
            plt.show()
        plt.close()

    def plot_comprehensive_season_overview(self, estimation_results=None, save=True):
        """
        综合可视化：展示34季的评分规则、淘汰动态和决赛结构

        参数:
        - estimation_results: FanPercentEstimator的estimation_results DataFrame
        - save: 是否保存图片
        """
        print("\n" + "=" * 60)
        print("Creating Comprehensive Season Overview Visualization")
        print("=" * 60)

        # 使用estimation_results或self.df
        if estimation_results is not None:
            data = estimation_results.copy()
            # 确保有is_exited列，如果没有则尝试使用is_eliminated
            if 'is_exited' not in data.columns and 'is_eliminated' in data.columns:
                data['is_exited'] = data['is_eliminated']
        else:
            data = self.df.copy()
            if 'is_exited' not in data.columns and 'is_eliminated' in data.columns:
                data['is_exited'] = data['is_eliminated']

        # 设置专业样式
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 10))

        # 定义颜色方案
        color_rank_system = '#E8F4F8'  # 浅蓝色 - Rank System
        color_percentage_system = '#FFF4E6'  # 浅橙色 - Percentage System
        color_elimination = '#FF6B6B'  # 红色 - 正常淘汰
        color_no_elimination = '#95E1D3'  # 青色 - 无淘汰周
        color_text = '#2C3E50'  # 深灰色 - 文字

        # 第一部分：背景分区（评分系统）
        ax.axvspan(0.5, 2.5, facecolor=color_rank_system, alpha=0.3, zorder=0)
        ax.axvspan(2.5, 27.5, facecolor=color_percentage_system, alpha=0.3, zorder=0)
        ax.axvspan(27.5, 34.5, facecolor=color_rank_system, alpha=0.3, zorder=0)

        # 添加系统标注
        ax.text(1.5, ax.get_ylim()[1] * 0.95, 'Rank\nSystem',
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=color_text, bbox=dict(boxstyle='round,pad=0.5',
                facecolor='white', edgecolor=color_text, alpha=0.8))
        ax.text(15, ax.get_ylim()[1] * 0.95, 'Percentage System',
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=color_text, bbox=dict(boxstyle='round,pad=0.5',
                facecolor='white', edgecolor=color_text, alpha=0.8))
        ax.text(30.5, ax.get_ylim()[1] * 0.95, 'Rank\nSystem',
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=color_text, bbox=dict(boxstyle='round,pad=0.5',
                facecolor='white', edgecolor=color_text, alpha=0.8))

        print("Background zones created successfully")

        # 第二部分：计算每个赛季的淘汰数据
        elimination_data = []
        finalist_data = []

        for season in range(1, 35):
            season_data = data[data['season'] == season]
            if len(season_data) == 0:
                continue

            # 按周次分组统计淘汰人数
            weekly_exits = season_data.groupby('week')['is_exited'].sum().to_dict()
            max_week = season_data['week'].max()

            # 统计决赛人数（最后一周的参赛人数）
            final_week_data = season_data[season_data['week'] == max_week]
            n_finalists = len(final_week_data)
            finalist_data.append({'season': season, 'n_finalists': n_finalists})

            # 记录每周的淘汰情况
            for week in range(1, int(max_week) + 1):
                n_exits = weekly_exits.get(week, 0)
                elimination_data.append({
                    'season': season,
                    'week': week,
                    'n_exits': n_exits,
                    'has_elimination': n_exits > 0
                })

        elim_df = pd.DataFrame(elimination_data)
        finalist_df = pd.DataFrame(finalist_data)

        print(f"Processed {len(elim_df)} week records across {len(finalist_df)} seasons")

        # 第三部分：绘制淘汰事件标记
        # 按赛季汇总：每个赛季有多少周有淘汰，多少周无淘汰
        season_summary = elim_df.groupby('season').agg({
            'has_elimination': 'sum',  # 有淘汰的周数
            'week': 'count'  # 总周数
        }).reset_index()
        season_summary.columns = ['season', 'weeks_with_elimination', 'total_weeks']
        season_summary['weeks_no_elimination'] = season_summary['total_weeks'] - season_summary['weeks_with_elimination']

        # 绘制堆叠柱状图
        x_positions = season_summary['season'].values
        y_elimination = season_summary['weeks_with_elimination'].values
        y_no_elimination = season_summary['weeks_no_elimination'].values

        # 淘汰周（底部）
        bars1 = ax.bar(x_positions, y_elimination, width=0.7,
                       color=color_elimination, alpha=0.8, label='Weeks with Elimination',
                       edgecolor='white', linewidth=1.5, zorder=3)

        # 无淘汰周（顶部）
        bars2 = ax.bar(x_positions, y_no_elimination, width=0.7,
                       bottom=y_elimination, color=color_no_elimination, alpha=0.8,
                       label='Weeks without Elimination', edgecolor='white',
                       linewidth=1.5, zorder=3)

        print("Bar charts plotted successfully")

        # 第四部分：添加淘汰人数注释
        # 在每个赛季的柱子上标注总淘汰周数
        for i, (season, weeks_elim) in enumerate(zip(x_positions, y_elimination)):
            if weeks_elim > 0:
                # 在淘汰周柱子的中间位置添加文字
                y_pos = weeks_elim / 2
                ax.text(season, y_pos, f'{int(weeks_elim)}',
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='white', zorder=5)

        print("Annotations added successfully")

        # 第五部分：决赛结构分析
        # 识别决赛人数变化点
        finalist_df['n_finalists_prev'] = finalist_df['n_finalists'].shift(1)
        finalist_df['structure_change'] = (finalist_df['n_finalists'] != finalist_df['n_finalists_prev'])

        # 在图表顶部添加决赛结构标记
        change_points = finalist_df[finalist_df['structure_change'] == True]

        # 绘制决赛人数趋势线（在柱状图上方）
        ax2 = ax.twinx()
        ax2.plot(finalist_df['season'], finalist_df['n_finalists'],
                color='#8B4513', linewidth=2.5, marker='D', markersize=6,
                label='Number of Finalists', alpha=0.7, zorder=4)
        ax2.set_ylabel('Number of Finalists', fontsize=14, fontweight='bold', color='#8B4513')
        ax2.tick_params(axis='y', labelsize=12, colors='#8B4513')
        ax2.set_ylim(0, max(finalist_df['n_finalists']) + 2)

        # 标注决赛结构变化点
        for idx, row in change_points.iterrows():
            if pd.notna(row['n_finalists_prev']):
                ax2.annotate(f"{int(row['n_finalists'])} finalists",
                           xy=(row['season'], row['n_finalists']),
                           xytext=(row['season'], row['n_finalists'] + 1),
                           fontsize=9, ha='center', color='#8B4513',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='#8B4513', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='#8B4513', lw=1.5))

        print("Finalist structure analysis completed")

        # 第六部分：最终样式设置
        # 设置主轴标签和标题
        ax.set_xlabel('Season', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Weeks', fontsize=16, fontweight='bold')
        ax.set_title('Comprehensive Overview: Scoring Systems, Elimination Dynamics & Finalist Structure\nAcross 34 Seasons of Dancing with the Stars',
                    fontsize=18, fontweight='bold', pad=20)

        # 设置X轴刻度
        ax.set_xticks(range(1, 35))
        ax.set_xticklabels(range(1, 35), fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # 设置网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # 移除顶部和右侧边框
        sns.despine(ax=ax, top=True, right=False)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                 loc='upper left', fontsize=12, frameon=True,
                 fancybox=True, shadow=True)

        # 调整布局
        plt.tight_layout()

        print("Final styling completed")

        # 保存图片
        if save:
            out_path = os.path.join(PIC_ROOT, 'comprehensive_season_overview.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive overview saved to: {out_path}")
        else:
            plt.show()

        plt.close()
        print("=" * 60)
        print("Comprehensive Season Overview Visualization Completed!")
        print("=" * 60)



class FanPercentEstimator:
    """观众分估计器"""

    def __init__(self, data_path):
        """初始化并加载数据"""
        self.df = pd.read_csv(data_path, encoding='utf-8-sig')
        self.processed_data = None
        self.estimation_results = None

    def preprocess_data(self):
        """数据预处理：提取每周的评委分、淘汰信息和排名"""
        print("=" * 60)
        print("Step 1: Data Preprocessing (Enhanced with Placement)")
        print("=" * 60)

        # 存储每周每个选手的数据
        weekly_data = []

        # 遍历每一行（每个选手）
        for idx, row in self.df.iterrows():
            celebrity = row['celebrity_name']
            season = row['season']
            results = row['results']

            # 获取placement并转换为数值
            placement = None
            if 'placement' in self.df.columns and pd.notna(row['placement']):
                try:
                    placement = float(row['placement'])
                except:
                    placement = None

            # 确定该选手的退出周次和最终排名
            exit_week = None
            final_rank = None

            if isinstance(results, str):
                if 'Eliminated Week' in results:
                    # 正常淘汰：从results解析
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
                    # 退出选手：需要找到第一个分数为0或NaN的周次
                    for week in range(1, 12):
                        judge_scores = []
                        for judge_num in range(1, 5):
                            col_name = f'week{week}_judge{judge_num}_score'
                            if col_name in self.df.columns:
                                score = row[col_name]
                                if pd.notna(score) and score != 0:
                                    judge_scores.append(float(score))

                        # 如果该周没有有效分数，说明已退出
                        if len(judge_scores) == 0 and week > 1:
                            exit_week = week - 1  # 上一周是最后参赛周
                            break

            # 遍历每一周
            for week in range(1, 12):  # 最多11周
                # 获取该周的评委分
                judge_scores = []
                for judge_num in range(1, 5):  # 最多4个评委
                    col_name = f'week{week}_judge{judge_num}_score'
                    if col_name in self.df.columns:
                        score = row[col_name]
                        if pd.notna(score) and score != 0:
                            judge_scores.append(float(score))

                # 如果该周有评委分
                if len(judge_scores) > 0:
                    avg_judge_score = np.mean(judge_scores)

                    # 判断该选手在该周是否退出
                    is_exited = (exit_week == week)

                    # 判断该选手在该周是否还在比赛中
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

        # 识别每个赛季的决赛周次
        self.finals_week = self.processed_data.groupby('season')['week'].max().to_dict()

        # 为决赛选手设置exit_week
        for season in self.finals_week:
            finals_week = self.finals_week[season]
            mask = (self.processed_data['season'] == season) & \
                   (self.processed_data['final_rank'].notna()) & \
                   (self.processed_data['exit_week'].isna())
            self.processed_data.loc[mask, 'exit_week'] = finals_week

        print(f"Processing completed: {len(self.processed_data)} weekly records")
        print(f"Seasons: {self.processed_data['season'].nunique()}")
        print(f"Celebrities: {self.processed_data['celebrity'].nunique()}")
        print(f"Finals identified for {len(self.finals_week)} seasons")
        print(f"Withdrew contestants handled: {(self.processed_data['results'] == 'Withdrew').sum()}")
        print("\nFirst 5 records:")
        print(self.processed_data.head())

        return self.processed_data

    def normalize_judge_scores(self, week_data):
        """将评委分归一化为百分比（总和为1）"""
        total_score = week_data['judge_score'].sum()
        if total_score > 0:
            return week_data['judge_score'] / total_score
        else:
            # 如果总分为0，均分
            return pd.Series([1.0 / len(week_data)] * len(week_data), index=week_data.index)

    def convert_percent_to_rank(self, percent_array):
        """
        将百分比转换为排名
        最高百分比 = Rank 1 (最好)
        最低百分比 = Rank N (最差)

        参数：
        - percent_array: 百分比数组

        返回：
        - 排名数组 (1-based, 1是最好)
        """
        # 使用argsort两次来获得排名
        # 降序排列：最高分数排名为1
        temp = percent_array.argsort()[::-1]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1, len(percent_array) + 1)
        return ranks

    def check_percentage_constraints(self, judge_percent, fan_percent, week_data, current_week, is_finals):
        """
        检查百分比制约束 (Seasons 3-27) - 基于Placement排序

        约束逻辑：
        1. 识别本周退出者 (exit_week == current_week)
        2. 识别幸存者 (exit_week > current_week 或 exit_week is None)
        3. 全局约束：所有幸存者的 Total % 必须 > 所有退出者的 Total %
        4. 退出者内部约束：Total % 必须符合 placement 顺序（placement小的Total%应该更高）
        5. 决赛周特殊处理：按 final_rank 排序

        参数：
        - judge_percent: 评委百分比数组
        - fan_percent: 观众百分比数组
        - week_data: 该周数据
        - current_week: 当前周次
        - is_finals: 是否为决赛周

        返回：
        - True/False
        """
        total_percent = judge_percent + fan_percent

        if is_finals:
            # 决赛周：按 final_rank 排序
            ranked_contestants = week_data[week_data['final_rank'].notna()].copy()
            if len(ranked_contestants) > 0:
                ranked_contestants = ranked_contestants.sort_values('final_rank')
                indices = ranked_contestants.index.tolist()

                # 检查 Total % 是否按 final_rank 递减
                for i in range(len(indices) - 1):
                    idx_current = week_data.index.get_loc(indices[i])
                    idx_next = week_data.index.get_loc(indices[i + 1])

                    if total_percent[idx_current] <= total_percent[idx_next]:
                        return False
                return True
            else:
                return True
        else:
            # 常规周：基于 placement 的约束
            # 识别本周退出者和幸存者
            exited_mask = week_data['is_exited'] == True
            survivor_mask = week_data['is_exited'] == False

            exited_indices = week_data[exited_mask].index.tolist()
            survivor_indices = week_data[survivor_mask].index.tolist()

            if len(exited_indices) == 0:
                # 没有人退出，接受样本
                return True

            # 约束1：所有幸存者的 Total % 必须 > 所有退出者的 Total %
            if len(survivor_indices) > 0:
                survivor_totals = [total_percent[week_data.index.get_loc(idx)] for idx in survivor_indices]
                exited_totals = [total_percent[week_data.index.get_loc(idx)] for idx in exited_indices]

                min_survivor_total = min(survivor_totals)
                max_exited_total = max(exited_totals)

                if min_survivor_total <= max_exited_total:
                    return False

            # 约束2：退出者内部按 placement 排序
            if len(exited_indices) > 1:
                exited_data = week_data.loc[exited_indices].copy()
                # 过滤掉没有placement的数据
                exited_with_placement = exited_data[exited_data['placement'].notna()]

                if len(exited_with_placement) > 1:
                    # 按 placement 排序（小的placement应该有更高的Total%）
                    exited_with_placement = exited_with_placement.sort_values('placement')
                    sorted_indices = exited_with_placement.index.tolist()

                    for i in range(len(sorted_indices) - 1):
                        idx_current = week_data.index.get_loc(sorted_indices[i])
                        idx_next = week_data.index.get_loc(sorted_indices[i + 1])

                        # placement小的应该有更高的Total%
                        if total_percent[idx_current] <= total_percent[idx_next]:
                            return False

            return True

    def check_rank_constraints(self, judge_percent, fan_percent, week_data, current_week, is_finals):
        """
        检查排名制约束 (Seasons 1-2, 28-34) - 基于Placement排序

        约束逻辑：
        1. 识别本周退出者 (exit_week == current_week)
        2. 识别幸存者 (exit_week > current_week 或 exit_week is None)
        3. 全局约束：所有幸存者的 Total Rank Sum 必须 < 所有退出者的 Total Rank Sum
        4. 退出者内部约束：Total Rank Sum 必须符合 placement 顺序（placement小的Rank Sum应该更低）
        5. 决赛周特殊处理：按 final_rank 排序

        参数：
        - judge_percent: 评委百分比数组
        - fan_percent: 观众百分比数组
        - week_data: 该周数据
        - current_week: 当前周次
        - is_finals: 是否为决赛周

        返回：
        - True/False
        """
        # 转换为排名
        judge_rank = self.convert_percent_to_rank(judge_percent)
        fan_rank = self.convert_percent_to_rank(fan_percent)
        total_rank_sum = judge_rank + fan_rank

        if is_finals:
            # 决赛周：按 final_rank 排序
            ranked_contestants = week_data[week_data['final_rank'].notna()].copy()
            if len(ranked_contestants) > 0:
                ranked_contestants = ranked_contestants.sort_values('final_rank')
                indices = ranked_contestants.index.tolist()

                # 检查 Total Rank Sum 是否按 final_rank 递增
                for i in range(len(indices) - 1):
                    idx_current = week_data.index.get_loc(indices[i])
                    idx_next = week_data.index.get_loc(indices[i + 1])

                    if total_rank_sum[idx_current] >= total_rank_sum[idx_next]:
                        return False
                return True
            else:
                return True
        else:
            # 常规周：基于 placement 的约束
            # 识别本周退出者和幸存者
            exited_mask = week_data['is_exited'] == True
            survivor_mask = week_data['is_exited'] == False

            exited_indices = week_data[exited_mask].index.tolist()
            survivor_indices = week_data[survivor_mask].index.tolist()

            if len(exited_indices) == 0:
                # 没有人退出，接受样本
                return True

            # 约束1：所有幸存者的 Total Rank Sum 必须 < 所有退出者的 Total Rank Sum
            if len(survivor_indices) > 0:
                survivor_rank_sums = [total_rank_sum[week_data.index.get_loc(idx)] for idx in survivor_indices]
                exited_rank_sums = [total_rank_sum[week_data.index.get_loc(idx)] for idx in exited_indices]

                max_survivor_rank = max(survivor_rank_sums)
                min_exited_rank = min(exited_rank_sums)

                if max_survivor_rank >= min_exited_rank:
                    return False

            # 约束2：退出者内部按 placement 排序
            if len(exited_indices) > 1:
                exited_data = week_data.loc[exited_indices].copy()
                # 过滤掉没有placement的数据
                exited_with_placement = exited_data[exited_data['placement'].notna()]

                if len(exited_with_placement) > 1:
                    # 按 placement 排序（小的placement应该有更低的Rank Sum）
                    exited_with_placement = exited_with_placement.sort_values('placement')
                    sorted_indices = exited_with_placement.index.tolist()

                    for i in range(len(sorted_indices) - 1):
                        idx_current = week_data.index.get_loc(sorted_indices[i])
                        idx_next = week_data.index.get_loc(sorted_indices[i + 1])

                        # placement小的应该有更低的Rank Sum（更好）
                        if total_rank_sum[idx_current] >= total_rank_sum[idx_next]:
                            return False

            return True

    def monte_carlo_sampling(self, week_data, season, week, n_samples=10000):
        """
        蒙特卡洛采样估计观众分 (支持两种评分系统 + Placement约束)

        评分系统：
        - Seasons 3-27: 百分比制 (Judge % + Fan % = Total %)
        - Seasons 1-2, 28-34: 排名制 (Judge Rank + Fan Rank = Total Rank Sum)

        约束条件：
        1. ∑FanPercent = 1.0
        2. 幸存者 vs 退出者：幸存者的得分/排名必须优于所有退出者
        3. 退出者内部：按 placement 排序
        4. 决赛周：按 final_rank 排序

        参数：
        - week_data: 某一周的选手数据
        - season: 赛季编号
        - week: 周次
        - n_samples: 采样次数

        返回：
        - 每位选手的观众分样本
        """
        n_contestants = len(week_data)

        # 归一化评委分为百分比
        judge_percent = self.normalize_judge_scores(week_data).values

        # 判断是否为决赛周
        is_finals = (week == self.finals_week.get(season, 999))

        # 判断使用哪种评分系统
        use_percentage_system = (3 <= season <= 27)

        # 存储有效样本
        valid_samples = []

        # 蒙特卡洛采样
        for _ in range(n_samples):
            # 使用Dirichlet分布生成满足∑=1约束的观众分
            # alpha参数控制分布的均匀程度，这里使用1表示均匀先验
            fan_percent = np.random.dirichlet(np.ones(n_contestants))

            # 根据赛季选择约束检查方法
            if use_percentage_system:
                # Seasons 3-27: 百分比制
                is_valid = self.check_percentage_constraints(
                    judge_percent, fan_percent, week_data, week, is_finals
                )
            else:
                # Seasons 1-2, 28-34: 排名制
                is_valid = self.check_rank_constraints(
                    judge_percent, fan_percent, week_data, week, is_finals
                )

            if is_valid:
                valid_samples.append(fan_percent)

        return np.array(valid_samples)

    def estimate_fan_percent(self, n_samples=10000):
        """估计所有周次的观众分 (支持两种评分系统 + Placement约束)"""
        print("\n" + "=" * 60)
        print("Step 2: Monte Carlo Sampling for Fan Percent Estimation")
        print("Step 2: (Enhanced with Dual Scoring Systems + Placement)")
        print("=" * 60)

        if self.processed_data is None:
            raise ValueError("Please run preprocess_data() first")

        results = []

        # 按赛季和周次分组
        grouped = self.processed_data.groupby(['season', 'week'])

        total_groups = len(grouped)
        print(f"Total groups to process: {total_groups}")
        print(f"Percentage System (Seasons 3-27)")
        print(f"Rank System (Seasons 1-2, 28-34)")

        for i, ((season, week), week_data) in enumerate(grouped):
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{total_groups}")

            # 重置索引以确保索引连续
            week_data = week_data.reset_index(drop=True)

            # 蒙特卡洛采样 (传入 season 和 week 参数)
            samples = self.monte_carlo_sampling(week_data, season, week, n_samples)

            if len(samples) == 0:
                # 如果没有有效样本，使用均匀分布
                print(f"Warning: Season {season} Week {week} - no valid samples, using uniform distribution")
                n_contestants = len(week_data)
                fan_mean = np.ones(n_contestants) / n_contestants
                fan_std = np.zeros(n_contestants)
            else:
                # 计算均值和标准差
                fan_mean = samples.mean(axis=0)
                fan_std = samples.std(axis=0)

            # 保存结果
            for j, (idx, row) in enumerate(week_data.iterrows()):
                results.append({
                    'celebrity': row['celebrity'],
                    'season': season,
                    'week': week,
                    'judge_score': row['judge_score'],
                    'judge_percent': self.normalize_judge_scores(week_data).iloc[j],
                    'fan_percent_mean': fan_mean[j],
                    'fan_percent_std': fan_std[j],
                    'is_exited': row['is_exited'],
                    'exit_week': row['exit_week'],
                    'final_rank': row['final_rank'],
                    'placement': row['placement'],
                    'n_valid_samples': len(samples)
                })

        self.estimation_results = pd.DataFrame(results)
        print(f"\nEstimation completed! Total records: {len(self.estimation_results)}")

        return self.estimation_results
    
    def create_final_df(self):
        """
        基于类中加载的 self.df 生成长表格式的 final_df
        """
        rows = []
        for _, row in self.df.iterrows():
            celebrity = row['celebrity_name']
            season = row['season']
            results = row['results']
            
            # 1. 解析淘汰周次
            eliminated_week = None
            if isinstance(results, str) and 'Eliminated Week' in results:
                try:
                    eliminated_week = int(results.split('Week ')[-1])
                except:
                    pass

            # 2. 遍历每周数据 (1-11周)
            for week in range(1, 12):
                scores = []
                for j in range(1, 5):
                    col = f'week{week}_judge{j}_score'
                    if col in self.df.columns and pd.notna(row[col]) and row[col] != 0:
                        scores.append(float(row[col]))
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    # 判定当前周是否为该选手的淘汰周
                    is_eliminated = (eliminated_week == week)
                    
                    # 只保留选手还在参赛或刚刚被淘汰的数据（过滤掉后续的0分记录）
                    if eliminated_week is None or week <= eliminated_week:
                        rows.append({
                            'celebrity': celebrity,
                            'season': season,
                            'week': week,
                            'judge_avg': avg_score,
                            'is_eliminated': int(is_eliminated)
                        })
        
        return pd.DataFrame(rows)

    def calculate_upset_rate(self):
        """
        计算"爆冷率"
        定义：评委分最低者与实际退出者不一致的比例
        """
        print("\n" + "=" * 60)
        print("Step 3: Calculate Upset Rate")
        print("=" * 60)

        if self.estimation_results is None:
            raise ValueError("Please run estimate_fan_percent() first")

        upset_count = 0
        total_exit_weeks = 0
        upset_details = []

        # 按赛季和周次分组
        grouped = self.estimation_results.groupby(['season', 'week'])

        for (season, week), week_data in grouped:
            # 检查该周是否有人退出
            if week_data['is_exited'].any():
                total_exit_weeks += 1

                # 找出评委分最低的选手
                min_judge_idx = week_data['judge_percent'].idxmin()
                judge_lowest = week_data.loc[min_judge_idx, 'celebrity']

                # 找出实际退出的选手
                exited_idx = week_data[week_data['is_exited']].index[0]
                actual_exited = week_data.loc[exited_idx, 'celebrity']

                # 如果不一致，则为"爆冷"
                if judge_lowest != actual_exited:
                    upset_count += 1
                    upset_details.append({
                        'season': season,
                        'week': week,
                        'judge_lowest': judge_lowest,
                        'actual_exited': actual_exited
                    })

        upset_rate = upset_count / total_exit_weeks if total_exit_weeks > 0 else 0

        # 将详细信息保存到文件
        if len(upset_details) > 0:
            upset_df = pd.DataFrame(upset_details)
            upset_df.to_csv('upset_details.csv', index=False, encoding='utf-8-sig')
            print(f"Upset details saved to: upset_details.csv")

        print(f"\nTotal exit weeks: {total_exit_weeks}")
        print(f"Upset count: {upset_count}")
        print(f"Upset rate: {upset_rate:.2%}")

        return upset_rate

    def save_results(self, output_path='fan_percent_estimation_results.csv'):
        """保存估计结果到CSV"""
        if self.estimation_results is None:
            raise ValueError("Please run estimate_fan_percent() first")

        self.estimation_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_path}")

    def plot_fan_support_trend(self, season=1, output_path='fan_support_trend.png'):
        """
        绘制某一季中选手的观众支持度随时间变化的趋势图（带置信区间）

        参数：
        - season: 要绘制的赛季编号
        - output_path: 输出图片路径
        """
        print("\n" + "=" * 60)
        print(f"Step 4: Plotting Fan Support Trend for Season {season}")
        print("=" * 60)

        if self.estimation_results is None:
            raise ValueError("Please run estimate_fan_percent() first")

        # 筛选指定赛季的数据
        season_data = self.estimation_results[self.estimation_results['season'] == season]

        if len(season_data) == 0:
            print(f"Warning: No data for Season {season}")
            return

        # 创建图表
        plt.figure(figsize=(14, 8))

        # 获取该赛季的所有选手
        celebrities = season_data['celebrity'].unique()

        # 为每位选手绘制趋势线
        for celebrity in celebrities:
            celeb_data = season_data[season_data['celebrity'] == celebrity].sort_values('week')

            weeks = celeb_data['week'].values
            fan_mean = celeb_data['fan_percent_mean'].values
            fan_std = celeb_data['fan_percent_std'].values

            # 绘制均值线
            plt.plot(weeks, fan_mean, marker='o', label=celebrity, linewidth=2)

            # 绘制置信区间（均值 ± 1个标准差）
            plt.fill_between(weeks,
                            fan_mean - fan_std,
                            fan_mean + fan_std,
                            alpha=0.2)

        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Fan Support Percent', fontsize=12)
        plt.title(f'Season {season} - Fan Support Trend with Confidence Intervals', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图表
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("Fan Percent Estimation Model - Monte Carlo Sampling")
    print("=" * 60)

    # 初始化估计器
    estimator = FanPercentEstimator('2026_MCM_Problem_C_Data.csv')

    # 步骤1：数据预处理
    estimator.preprocess_data()

    # 步骤2：估计观众分
    estimator.estimate_fan_percent(n_samples=10000)

    # 步骤3：计算爆冷率
    upset_rate = estimator.calculate_upset_rate()

    # 步骤4：保存结果
    estimator.save_results('fan_percent_estimation_results.csv')

    # 步骤5：绘制趋势图（以第1季为例）
    # estimator.plot_fan_support_trend(season=1, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season1.png'))
    # estimator.plot_fan_support_trend(season=2, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season2.png'))
    # estimator.plot_fan_support_trend(season=3, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season3.png'))
    # estimator.plot_fan_support_trend(season=28, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season28.png'))
    # estimator.plot_fan_support_trend(season=30, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season30.png'))
    # estimator.plot_fan_support_trend(season=21, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season21.png'))#withdrew
    # estimator.plot_fan_support_trend(season=18, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season18.png'))#withdrew
    # estimator.plot_fan_support_trend(season=15, output_path=os.path.join(PIC_ROOT, 'fan_support_trend_season15.png'))#全明星阵容

    #导师相关指标分析
   
    # 使用示例 (假设您已经有了预处理后的 final_df)
    print("正在生成分析用 DataFrame...")
    final_df = estimator.create_final_df()
    analyzer = DWTSAdvAnalysis(final_df)
    analyzer.calculate_indicators()
    analyzer.plot_judge_influence_trend(save=True)  # 纵向：导师话语权趋势
    analyzer.plot_season_week_heatmap(save=True)    # 纵向：淘汰命中分析
    analyzer.plot_celebrity_score_heatmap(27, save=True) # 横向：第27季选手表现对比

      # Call the new comprehensive visualization
    analyzer.plot_comprehensive_season_overview(
        estimation_results=estimator.estimation_results,
        save=True
    )

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
