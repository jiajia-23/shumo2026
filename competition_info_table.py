"""
比赛信息表格生成器
基于 fan_percent_estimation.py 的数据处理逻辑
生成每季每周的详细比赛信息表格

表格包含：
1. 每季每周信息
2. 每季的初始人数
3. 每周是比赛周还是决赛周
4. 是否有人离开
5. 是否有人退赛
6. 走了多少人
7. 决赛周人数
"""

import pandas as pd
import numpy as np
import os

class CompetitionInfoGenerator:
    """比赛信息表格生成器"""

    def __init__(self, data_path):
        """初始化并加载数据"""
        self.df = pd.read_csv(data_path, encoding='utf-8-sig')
        self.processed_data = None
        self.competition_info = None

    def preprocess_data(self):
        """数据预处理：提取每周的评委分、淘汰信息和排名"""
        print("=" * 60)
        print("Step 1: Data Preprocessing")
        print("=" * 60)

        # 存储每周每个选手的数据
        weekly_data = []

        # 遍历每一行（每个选手）
        for idx, row in self.df.iterrows():
            celebrity = row['celebrity_name']
            season = row['season']
            results = row['results']

            # 确定该选手的退出周次和最终排名
            exit_week = None
            final_rank = None
            is_withdrew = False

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
                    is_withdrew = True
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
                            'is_withdrew': is_withdrew,
                            'exit_week': exit_week,
                            'final_rank': final_rank,
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

        return self.processed_data

    def generate_competition_info(self):
        """生成比赛信息表格"""
        print("\n" + "=" * 60)
        print("Step 2: Generating Competition Info Table")
        print("=" * 60)

        if self.processed_data is None:
            raise ValueError("Please run preprocess_data() first")

        competition_records = []

        # 遍历每个赛季
        for season in sorted(self.processed_data['season'].unique()):
            season_data = self.processed_data[self.processed_data['season'] == season]

            # 获取该赛季的所有周次
            weeks = sorted(season_data['week'].unique())
            finals_week = max(weeks)

            # 计算初始人数（第一周的参赛人数）
            initial_count = len(season_data[season_data['week'] == 1])

            # 遍历每一周
            for week in weeks:
                week_data = season_data[season_data['week'] == week]

                # 当前周参赛人数
                current_count = len(week_data)

                # 判断是否为决赛周
                is_finals = (week == finals_week)
                week_type = "决赛周" if is_finals else "比赛周"

                # 统计本周离开的人数（is_exited == True）
                eliminated_this_week = week_data[week_data['is_exited'] == True]
                eliminated_count = len(eliminated_this_week)

                # 统计本周退赛的人数（is_withdrew == True 且 is_exited == True）
                withdrew_this_week = eliminated_this_week[eliminated_this_week['is_withdrew'] == True]
                withdrew_count = len(withdrew_this_week)

                # 检查下一周是否有人退赛（需要算进本周离开的人里）
                next_week_withdrew_count = 0
                if week < finals_week:
                    # 查找下一周退赛的人
                    next_week = week + 1
                    next_week_data = season_data[season_data['week'] == next_week]

                    # 找出在本周还在，但下一周退赛的人
                    current_celebrities = set(week_data['celebrity'].unique())
                    next_week_celebrities = set(next_week_data['celebrity'].unique())

                    # 找出在本周之后、下一周之前退赛的人
                    # 这些人在本周还有数据，但在下一周就没有了（且是退赛）
                    for celebrity in current_celebrities:
                        celeb_data = season_data[season_data['celebrity'] == celebrity]
                        # 如果这个人是退赛的，且退出周次是下一周
                        if celeb_data['is_withdrew'].any() and celeb_data['exit_week'].iloc[0] == next_week:
                            next_week_withdrew_count += 1

                # 总离开人数 = 本周淘汰/退赛人数 + 下一周退赛人数
                total_left = eliminated_count + next_week_withdrew_count

                # 是否有人离开
                has_exit = (total_left > 0)

                # 是否有人退赛（本周）
                has_withdrew = (withdrew_count > 0)

                # 决赛周人数（如果是决赛周，就是当前人数）
                finals_count = current_count if is_finals else None

                competition_records.append({
                    'season': season,
                    'week': week,
                    'initial_count': initial_count,
                    'current_count': current_count,
                    'week_type': week_type,
                    'is_finals': is_finals,
                    'has_exit': has_exit,
                    'has_withdrew': has_withdrew,
                    'eliminated_count': eliminated_count,
                    'withdrew_count': withdrew_count,
                    'next_week_withdrew_count': next_week_withdrew_count,
                    'total_left': total_left,
                    'finals_count': finals_count
                })

        self.competition_info = pd.DataFrame(competition_records)

        print(f"Competition info table generated: {len(self.competition_info)} records")
        print(f"Seasons: {self.competition_info['season'].nunique()}")
        print(f"Total weeks: {len(self.competition_info)}")

        return self.competition_info

    def save_results(self, output_path='competition_info_table.csv'):
        """保存比赛信息表格到CSV文件"""
        if self.competition_info is None:
            raise ValueError("Please run generate_competition_info() first")

        self.competition_info.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n比赛信息表格已保存到: {output_path}")

    def display_summary(self):
        """显示比赛信息表格的摘要统计"""
        if self.competition_info is None:
            raise ValueError("Please run generate_competition_info() first")

        print("\n" + "=" * 60)
        print("Competition Info Summary")
        print("=" * 60)

        # 按赛季统计
        print("\n按赛季统计:")
        season_summary = self.competition_info.groupby('season').agg({
            'initial_count': 'first',
            'week': 'count',
            'total_left': 'sum',
            'withdrew_count': 'sum',
            'finals_count': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
        }).rename(columns={
            'initial_count': '初始人数',
            'week': '周数',
            'total_left': '总离开人数',
            'withdrew_count': '退赛人数',
            'finals_count': '决赛人数'
        })
        print(season_summary)

        # 整体统计
        print("\n整体统计:")
        print(f"总赛季数: {self.competition_info['season'].nunique()}")
        print(f"总周数: {len(self.competition_info)}")
        print(f"有人离开的周数: {self.competition_info['has_exit'].sum()}")
        print(f"有人退赛的周数: {self.competition_info['has_withdrew'].sum()}")
        print(f"总离开人数: {self.competition_info['total_left'].sum()}")
        print(f"总退赛人数: {self.competition_info['withdrew_count'].sum()}")

        # 显示前10行
        print("\n前10行数据:")
        print(self.competition_info.head(10).to_string())


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("比赛信息表格生成器")
    print("=" * 60)

    # 初始化生成器
    generator = CompetitionInfoGenerator('2026_MCM_Problem_C_Data.csv')

    # 数据预处理
    generator.preprocess_data()

    # 生成比赛信息表格
    generator.generate_competition_info()

    # 显示摘要
    generator.display_summary()

    # 保存结果
    generator.save_results('competition_info_table.csv')

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()


