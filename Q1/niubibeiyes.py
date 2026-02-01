# ==========================================
# 0. 配置与工具函数
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import copy
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    'N_PARTICLES': 1000,      # 粒子数量
    'SYSTEM_CHANGE_S3': 3,    # 百分比制起始赛季
    'SYSTEM_CHANGE_S28': 28,  # 裁判拯救环节起始赛季
    'JUDGE_SAVE_ERA_RETURN_TO_RANK': True,  # S28+回归排名制
    'BASE_VOTE_NOISE': 0.05,  # 铁粉基数周际波动幅度
    'PIC_ROOT': 'DWTS_Visualizations',  # 图片保存根目录
    'CACHE_DIR': 'DWTS_Cache',          # 缓存目录
    'DATA_PATH': '2026_MCM_Problem_C_Data.csv',  # 原始数据路径
    'OUTPUT_PATH': 'dual_source_estimation_results.csv'  # 结果输出路径
}

# 创建必要目录
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for dir_path in [CONFIG['PIC_ROOT'], CONFIG['CACHE_DIR']]:
    ensure_dir(dir_path)

# 设置中文字体与绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

# 定义专业配色方案
COLORS = {
    'base': '#2E86AB',      # 深蓝色-铁粉基数
    'casual': '#A23B72',    # 深红色-路人增量
    'total': '#F18F01',     # 橙色-总人气
    'confidence_high': '#28A745',  # 深绿色-高可信度
    'confidence_low': '#DC3545',   # 深红色-低可信度
    'rank_system': '#E9F1F7',     # 浅蓝色-排名制背景
    'percent_system': '#FFF3CD',   # 浅橙色-百分比制背景
    'elimination': '#FFE8E8',      # 浅红色-淘汰周标记
}

# ==========================================
# 1. 数据预处理工具类
# ==========================================
class DataPreprocessor:
    """数据预处理：将原始CSV转换为长表格式"""
    def __init__(self, data_path):
        self.raw_df = pd.read_csv(data_path, encoding='utf-8-sig')
        self.processed_df = None

    def preprocess(self):
        """核心预处理逻辑"""
        print("开始数据预处理...")
        weekly_data = []
        
        # 遍历每位选手
        for idx, row in tqdm(self.raw_df.iterrows(), total=len(self.raw_df), desc="预处理选手数据"):
            celebrity = row['celebrity_name']
            season = row['season']
            results = row['results']
            placement = row['placement'] if pd.notna(row['placement']) else None
            
            # 解析退出周次
            exit_week = None
            final_rank = None
            if isinstance(results, str):
                if 'Eliminated Week' in results:
                    try:
                        exit_week = int(results.split('Week ')[-1])
                    except:
                        exit_week = None
                elif '1st Place' in results:
                    final_rank = 1
                elif '2nd Place' in results:
                    final_rank = 2
                elif '3rd Place' in results:
                    final_rank = 3
                elif '4th Place' in results:
                    final_rank = 4
                elif results == 'Withdrew':
                    # 处理退赛选手：找到最后有有效分数的周次
                    for week in range(1, 12):
                        judge_scores = []
                        for judge_num in range(1, 5):
                            col_name = f'week{week}_judge{judge_num}_score'
                            if col_name in self.raw_df.columns:
                                score = row[col_name]
                                if pd.notna(score) and score != 0:
                                    judge_scores.append(float(score))
                        if len(judge_scores) == 0 and week > 1:
                            exit_week = week - 1
                            break
            
            # 遍历每周数据
            for week in range(1, 12):
                judge_scores = []
                # 收集该周有效评委分
                for judge_num in range(1, 5):
                    col_name = f'week{week}_judge{judge_num}_score'
                    if col_name in self.raw_df.columns:
                        score = row[col_name]
                        if pd.notna(score) and score != 0:
                            judge_scores.append(float(score))
                
                # 只保留有有效分数且在退赛前的记录
                if len(judge_scores) > 0:
                    avg_judge_score = np.mean(judge_scores)
                    is_exited = (exit_week == week)
                    
                    if exit_week is None or week <= exit_week:
                        weekly_data.append({
                            'season': season,
                            'week': week,
                            'celebrity': celebrity,
                            'judge_score': avg_judge_score,
                            'is_exited': is_exited,
                            'exit_week': exit_week,
                            'final_rank': final_rank,
                            'placement': placement,
                            'results': results
                        })
        
        self.processed_df = pd.DataFrame(weekly_data)
        
        # 识别决赛周次
        finals_week = self.processed_df.groupby('season')['week'].max().to_dict()
        for season in finals_week:
            f_week = finals_week[season]
            mask = (self.processed_df['season'] == season) & \
                   (self.processed_df['final_rank'].notna()) & \
                   (self.processed_df['exit_week'].isna())
            self.processed_df.loc[mask, 'exit_week'] = f_week
        
        # 计算每个选手的评委相关性先验
        self.processed_df['judge_corr'] = self._calculate_judge_correlation()
        
        print(f"预处理完成：{len(self.processed_df)} 条周度记录")
        print(f"涵盖赛季：{self.processed_df['season'].nunique()} 个")
        print(f"涵盖选手：{self.processed_df['celebrity'].nunique()} 位")
        
        # 保存预处理结果到缓存
        self.processed_df.to_csv(os.path.join(CONFIG['CACHE_DIR'], 'processed_data.csv'), 
                               index=False, encoding='utf-8-sig')
        return self.processed_df

    def _calculate_judge_correlation(self):
        """计算选手评委分与最终排名的相关性（作为先验）"""
        judge_corr_dict = {}
        
        for season in self.processed_df['season'].unique():
            season_df = self.processed_df[self.processed_df['season'] == season]
            
            for celeb in season_df['celebrity'].unique():
                celeb_df = season_df[season_df['celebrity'] == celeb].sort_values('week')
                
                # 提取评委分序列和对应的排名（反向转换：分数越高排名越前）
                judge_scores = celeb_df['judge_score'].values
                if len(judge_scores) < 2:
                    judge_corr_dict[celeb] = 0.5
                    continue
                
                # 计算该选手在各周的相对排名
                weekly_ranks = []
                for week in celeb_df['week'].values:
                    week_df = season_df[season_df['week'] == week]
                    # 分数越高排名越小（1为最高）
                    ranks = rankdata(-week_df['judge_score'], method='min')
                    celeb_rank = ranks[week_df['celebrity'] == celeb][0]
                    weekly_ranks.append(celeb_rank)
                
                # 计算排名与周次的相关性（负相关表示排名提升）
                if len(weekly_ranks) >= 2:
                    corr = np.corrcoef(range(len(weekly_ranks)), weekly_ranks)[0, 1]
                    # 归一化到[0,1]区间
                    corr = (corr + 1) / 2
                else:
                    corr = 0.5
                
                judge_corr_dict[celeb] = corr
        
        # 映射到DataFrame
        self.processed_df['judge_corr'] = self.processed_df['celebrity'].map(judge_corr_dict)
        return self.processed_df['judge_corr']

# ==========================================
# 2. 评分系统逻辑（策略模式）
# ==========================================
class ScoringSystem:
    @staticmethod
    def calculate_total_rank(judge_scores, fan_shares, season):
        """
        计算总排名
        参数:
            judge_scores: 评委分列表
            fan_shares: 观众分占比列表
            season: 赛季编号
        返回:
            total_scores: 总得分
            score_type: 得分类型 ("Higher is Better" / "Lower is Better")
        """
        n = len(judge_scores)
        if n == 0:
            return np.array([]), "Higher is Better"
        
        # 评委分归一化
        j_sum = np.sum(judge_scores)
        j_share = judge_scores / j_sum if j_sum > 0 else np.ones(n) / n
        
        # 观众分已归一化
        f_share = fan_shares
        
        # 百分比制（S3-S27）
        if 3 <= season < 28:
            total_score = j_share + f_share
            return total_score, "Higher is Better"
        # 排名制（S1-S2, S28+）
        else:
            # 分数越高排名越小（1为最高）
            j_rank = rankdata(-j_share, method='min')
            f_rank = rankdata(-f_share, method='min')
            total_score = j_rank + f_rank  # 越小越好
            return total_score, "Lower is Better"

    @staticmethod
    def check_elimination_constraint(total_scores, score_type, eliminated_indices, safe_indices, season):
        """
        检查淘汰约束是否满足
        """
        # 无人淘汰周直接通过
        if len(eliminated_indices) == 0:
            return True
        
        # 提取相关分数
        elim_scores = total_scores[eliminated_indices]
        safe_scores = total_scores[safe_indices] if len(safe_indices) > 0 else np.array([])
        
        # 决赛周逻辑（无幸存者，单独处理）
        if len(safe_scores) == 0:
            return True
        
        # S28+裁判拯救环节（Bottom 2规则）
        if season >= 28:
            # 计算有多少幸存者比淘汰者表现差
            if score_type == "Higher is Better":
                worse_survivors = sum(s <= max(elim_scores) for s in safe_scores)
            else:
                worse_survivors = sum(s >= min(elim_scores) for s in safe_scores)
            return worse_survivors <= 1
        
        # 常规淘汰规则（所有淘汰者表现差于所有幸存者）
        if score_type == "Higher is Better":
            return np.min(safe_scores) > np.max(elim_scores)
        else:
            return np.max(safe_scores) < np.min(elim_scores)

# ==========================================
# 3. 粒子滤波核心模型
# ==========================================
class ContestantState:
    """单个选手的状态类"""
    def __init__(self, name, judge_corr=0.5):
        self.name = name
        self.judge_corr = judge_corr
        
        # 隐变量初始化：相关性越低，初始铁粉基数越高
        base_prior = 0.18 if judge_corr < 0.3 else 0.08 if judge_corr < 0.7 else 0.04
        self.base_share = np.random.beta(2.5, 25) + base_prior  # 贝塔分布初始化（更符合比例特性）
        self.base_share = max(0.001, min(0.5, self.base_share))  # 限制在合理范围
        
        # 表现转化率：相关性越高，转化率越高
        alpha_prior = 0.4 if judge_corr > 0.7 else 0.25 if judge_corr > 0.3 else 0.15
        self.alpha = np.random.uniform(alpha_prior - 0.1, alpha_prior + 0.1)
        self.alpha = max(0.05, min(0.6, self.alpha))  # 限制范围

    def predict(self):
        """状态转移：随机游走"""
        # 铁粉基数随机游走（带边界约束）
        noise = np.random.normal(0, CONFIG['BASE_VOTE_NOISE'])
        self.base_share = max(0.001, min(0.5, self.base_share + noise))
        
        # 表现转化率微小扰动
        alpha_noise = np.random.normal(0, 0.008)
        self.alpha = max(0.05, min(0.6, self.alpha + alpha_noise))

class Particle:
    """粒子类：代表一个可能的状态组合"""
    def __init__(self, contestants_data):
        """
        参数:
            contestants_data: {celebrity_name: judge_corr, ...}
        """
        self.states = {
            name: ContestantState(name, corr) 
            for name, corr in contestants_data.items()
        }
        self.history = []  # 记录历史状态
        self.weight = 1.0  # 粒子权重

    def step(self, week_df, season):
        """
        粒子推演一步
        返回: 是否符合约束
        """
        # 提取当周有效数据
        active_names = week_df['celebrity'].values
        active_judge_scores = week_df['judge_score'].values
        n_active = len(active_names)
        
        # 状态预测
        for name in active_names:
            if name in self.states:
                self.states[name].predict()
            else:
                # 处理新出现的选手
                judge_corr = week_df[week_df['celebrity'] == name]['judge_corr'].iloc[0]
                self.states[name] = ContestantState(name, judge_corr)
        
        # 计算原始观众票（双源动力模型）
        j_total = np.sum(active_judge_scores)
        j_norm = active_judge_scores / j_total if j_total > 0 else np.ones(n_active) / n_active
        
        raw_votes = []
        bases = []
        casuals = []
        
        for i, name in enumerate(active_names):
            state = self.states[name]
            casual_vote = state.alpha * j_norm[i]
            total_vote = state.base_share + casual_vote
            raw_votes.append(total_vote)
            bases.append(state.base_share)
            casuals.append(casual_vote)
        
        # 归一化为占比
        raw_votes = np.array(raw_votes)
        fan_shares = raw_votes / np.sum(raw_votes) if np.sum(raw_votes) > 0 else np.ones(n_active) / n_active
        
        # 检查约束
        eliminated_mask = week_df['is_exited'].values
        eliminated_idx = np.where(eliminated_mask)[0]
        safe_idx = np.where(~eliminated_mask)[0]
        
        total_scores, score_type = ScoringSystem.calculate_total_rank(
            active_judge_scores, fan_shares, season
        )
        
        is_consistent = ScoringSystem.check_elimination_constraint(
            total_scores, score_type, eliminated_idx, safe_idx, season
        )
        
        # 记录历史（仅保留符合约束的）
        if is_consistent:
            snapshot = [{
                'celebrity': name,
                'week': week_df['week'].iloc[0],
                'fan_share': fan_shares[i],
                'base_share_est': bases[i],
                'casual_share_est': casuals[i],
                'total_score': total_scores[i],
                'score_type': score_type
            } for i, name in enumerate(active_names)]
            self.history.append(snapshot)
            return True
        return False

# ==========================================
# 4. 核心控制器
# ==========================================
class DualSourceEstimator:
    def __init__(self, processed_df):
        self.processed_df = processed_df
        self.results = []
        self.credibility_records = []
        self.contestant_profiles = {}  # 选手特征档案

    def run_season_estimation(self, season):
        """运行单个赛季的估计"""
        season_df = self.processed_df[self.processed_df['season'] == season].copy()
        if len(season_df) == 0:
            print(f"警告：赛季 {season} 无有效数据")
            return
        
        print(f"\n=== 处理赛季 {season} ===")
        
        # 准备选手元数据（姓名: 评委相关性）
        contestant_meta = season_df.groupby('celebrity')['judge_corr'].first().to_dict()
        
        # 初始化粒子群
        particles = [Particle(contestant_meta) for _ in range(CONFIG['N_PARTICLES'])]
        
        # 按周推演
        weeks = sorted(season_df['week'].unique())
        for week in tqdm(weeks, desc=f"赛季 {season} 周次推演"):
            week_df = season_df[season_df['week'] == week].copy()
            
            # 粒子传播与筛选
            valid_particles = []
            for p in particles:
                p_copy = copy.deepcopy(p)
                if p_copy.step(week_df, season):
                    valid_particles.append(p_copy)
            
            # 计算可信度
            survival_rate = len(valid_particles) / len(particles) if particles else 0
            self.credibility_records.append({
                'season': season,
                'week': week,
                'survival_rate': survival_rate,
                'n_valid_particles': len(valid_particles),
                'n_total_particles': len(particles)
            })
            
            # 处理粒子崩溃情况
            if len(valid_particles) == 0:
                print(f"  警告：周 {week} 无有效粒子，使用随机重启")
                valid_particles = [Particle(contestant_meta) for _ in range(CONFIG['N_PARTICLES'][:200])]
                survival_rate = 0.0
            
            # 结果聚合
            self._aggregate_results(valid_particles, season, week)
            
            # 重采样（保持粒子数量）
            if len(valid_particles) > 0:
                sample_probs = np.ones(len(valid_particles)) / len(valid_particles)
                selected_indices = np.random.choice(
                    len(valid_particles), 
                    size=CONFIG['N_PARTICLES'],
                    p=sample_probs,
                    replace=True
                )
                particles = [copy.deepcopy(valid_particles[i]) for i in selected_indices]
        
        # 生成选手特征档案
        self._generate_contestant_profiles(season)

    def _aggregate_results(self, valid_particles, season, week):
        """聚合有效粒子的结果"""
        if len(valid_particles) == 0:
            return
        
        # 按选手分组统计
        contestant_stats = {}
        for p in valid_particles:
            if not p.history:
                continue
            week_snapshot = p.history[-1]  # 最新一周的快照
            for record in week_snapshot:
                celeb = record['celebrity']
                if celeb not in contestant_stats:
                    contestant_stats[celeb] = {
                        'fan_shares': [],
                        'base_components': [],
                        'casual_components': [],
                        'total_scores': []
                    }
                contestant_stats[celeb]['fan_shares'].append(record['fan_share'])
                contestant_stats[celeb]['base_components'].append(record['base_share_est'])
                contestant_stats[celeb]['casual_components'].append(record['casual_share_est'])
                contestant_stats[celeb]['total_scores'].append(record['total_score'])
        
        # 计算统计量并保存
        survival_rate = len(valid_particles) / CONFIG['N_PARTICLES']
        for celeb, stats in contestant_stats.items():
            self.results.append({
                'season': season,
                'week': week,
                'celebrity': celeb,
                'fan_share_mean': np.mean(stats['fan_shares']),
                'fan_share_std': np.std(stats['fan_shares']),
                'fan_share_25p': np.percentile(stats['fan_shares'], 25),
                'fan_share_75p': np.percentile(stats['fan_shares'], 75),
                'base_component': np.mean(stats['base_components']),
                'casual_component': np.mean(stats['casual_components']),
                'total_score_mean': np.mean(stats['total_scores']),
                'model_confidence': survival_rate,
                'n_valid_samples': len(stats['fan_shares'])
            })

    def _generate_contestant_profiles(self, season):
        """生成选手特征档案"""
        season_results = [r for r in self.results if r['season'] == season]
        if not season_results:
            return
        
        for celeb in set(r['celebrity'] for r in season_results):
            celeb_results = [r for r in season_results if r['celebrity'] == celeb]
            if len(celeb_results) < 2:
                continue
            
            # 计算核心特征
            base_components = [r['base_component'] for r in celeb_results]
            casual_components = [r['casual_component'] for r in celeb_results]
            fan_shares = [r['fan_share_mean'] for r in celeb_results]
            
            # 铁粉指数（FDI）：铁粉占总人气的比例
            fdi = np.mean([b/(b+c+1e-8) for b, c in zip(base_components, casual_components)])
            
            # 实力敏感度（PS）：路人增量与评委分的相关性
            judge_scores = self.processed_df[
                (self.processed_df['season'] == season) & 
                (self.processed_df['celebrity'] == celeb)
            ]['judge_score'].values
            ps = np.corrcoef(casual_components, judge_scores[:len(casual_components)])[0, 1] if len(casual_components) >= 2 else 0
            
            # 稳定性指数（SI）：人气波动系数
            si = np.std(fan_shares) / np.mean(fan_shares) if np.mean(fan_shares) > 0 else 0
            
            self.contestant_profiles[(season, celeb)] = {
                'season': season,
                'celebrity': celeb,
                'fan_dominance_index': fdi,
                'performance_sensitivity': ps,
                'stability_index': si,
                'avg_base_component': np.mean(base_components),
                'avg_casual_component': np.mean(casual_components),
                'total_weeks': len(celeb_results),
                'final_placement': self.processed_df[
                    (self.processed_df['season'] == season) & 
                    (self.processed_df['celebrity'] == celeb)
                ]['placement'].iloc[0]
            }

    def run_all_seasons(self):
        """运行所有赛季的估计"""
        seasons = sorted(self.processed_df['season'].unique())
        print(f"开始处理 {len(seasons)} 个赛季")
        
        for season in seasons:
            self.run_season_estimation(season)
        
        # 转换结果为DataFrame并保存
        results_df = pd.DataFrame(self.results)
        credibility_df = pd.DataFrame(self.credibility_records)
        profiles_df = pd.DataFrame(self.contestant_profiles.values())
        
        # 保存到文件
        results_df.to_csv(CONFIG['OUTPUT_PATH'], index=False, encoding='utf-8-sig')
        credibility_df.to_csv(os.path.join(CONFIG['CACHE_DIR'], 'model_credibility.csv'), index=False)
        profiles_df.to_csv(os.path.join(CONFIG['CACHE_DIR'], 'contestant_profiles.csv'), index=False)
        
        print(f"\n所有赛季处理完成！")
        print(f"结果已保存到：{CONFIG['OUTPUT_PATH']}")
        return results_df, credibility_df, profiles_df

# ==========================================
# 5. 可视化与模型评价
# ==========================================
class ModelVisualizer:
    """可视化工具类"""
    def __init__(self, results_df, credibility_df, profiles_df):
        self.results_df = results_df
        self.credibility_df = credibility_df
        self.profiles_df = profiles_df
        self.controversial_contestants = [
            ('Jerry Rice', 2),
            ('Billy Ray Cyrus', 4),
            ('Bristol Palin', 11),
            ('Bobby Bones', 27)
        ]

    def plot_model_credibility_heatmap(self):
        """绘制模型可信度热力图"""
        pivot_df = self.credibility_df.pivot_table(
            index='season', 
            columns='week', 
            values='survival_rate',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(16, 10))
        mask = pivot_df.isna()
        
        # 创建热力图
        sns.heatmap(
            pivot_df,
            mask=mask,
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            center=0.5,
            annot=False,
            square=True,
            linewidths=0.1,
            cbar_kws={'label': '模型可信度 (粒子存活率)', 'shrink': 0.8}
        )
        
        # 添加评分系统分区
        max_week = pivot_df.columns.max()
        plt.axvspan(-0.5, max_week + 0.5, xmin=0, xmax=2/34, alpha=0.1, color=COLORS['rank_system'], label='排名制 (S1-S2)')
        plt.axvspan(-0.5, max_week + 0.5, xmin=2/34, xmax=27/34, alpha=0.1, color=COLORS['percent_system'], label='百分比制 (S3-S27)')
        plt.axvspan(-0.5, max_week + 0.5, xmin=27/34, xmax=1, alpha=0.1, color=COLORS['rank_system'], label='排名制 (S28+)')
        
        plt.title('DWTS模型可信度热力图（按赛季-周次）', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('周次', fontsize=12, fontweight='bold')
        plt.ylabel('赛季', fontsize=12, fontweight='bold')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['rank_system'], alpha=0.5, label='排名制 (S1-S2, S28+)'),
            Patch(facecolor=COLORS['percent_system'], alpha=0.5, label='百分比制 (S3-S27)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['PIC_ROOT'], 'model_credibility_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("模型可信度热力图已保存")

    def plot_controversial_contestants_dynamics(self):
        """绘制争议选手双源动力动态图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (celeb, season) in enumerate(self.controversial_contestants):
            # 筛选数据
            celeb_data = self.results_df[
                (self.results_df['celebrity'] == celeb) & 
                (self.results_df['season'] == season)
            ].sort_values('week')
            
            if len(celeb_data) == 0:
                axes[idx].text(0.5, 0.5, f'无 {celeb} (S{season}) 数据', ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{celeb} (S{season})')
                continue
            
            # 绘制堆叠面积图
            weeks = celeb_data['week'].values
            base = celeb_data['base_component'].values
            casual = celeb_data['casual_component'].values
            
            axes[idx].stackplot(
                weeks,
                base,
                casual,
                labels=['铁粉基数', '路人增量'],
                colors=[COLORS['base'], COLORS['casual']],
                alpha=0.7
            )
            
            # 绘制总人气趋势线
            total = base + casual
            axes[idx].plot(weeks, total, color=COLORS['total'], linestyle='--', linewidth=2, label='总人气')
            
            # 添加置信区间
            fan_mean = celeb_data['fan_share_mean'].values
            fan_std = celeb_data['fan_share_std'].values
            axes[idx].errorbar(
                weeks, fan_mean, yerr=fan_std, fmt='o', color='black', alpha=0.5, 
                capsize=3, label='观众分占比（含误差）'
            )
            
            # 标记淘汰周
            exit_week = self.results_df[
                (self.results_df['celebrity'] == celeb) & 
                (self.results_df['season'] == season)
            ]['week'].max()
            axes[idx].axvline(x=exit_week, color='red', linestyle=':', linewidth=2, label=f'淘汰周 ({exit_week})')
            
            # 设置标题和标签
            axes[idx].set_title(f'{celeb} (S{season}) 人气动态分解', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('周次', fontsize=10)
            axes[idx].set_ylabel('分数/占比', fontsize=10)
            axes[idx].legend(loc='upper left', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['PIC_ROOT'], 'controversial_contestants_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("争议选手人气动态图已保存")

    def plot_contestant_classification_scatter(self):
        """绘制选手分类散点图"""
        plt.figure(figsize=(14, 10))
        
        # 过滤有效数据
        valid_profiles = self.profiles_df[
            (self.profiles_df['fan_dominance_index'].notna()) &
            (self.profiles_df['performance_sensitivity'].notna())
        ].copy()
        
        # 选手分类
        valid_profiles['type'] = '双轮驱动型'
        valid_profiles.loc[valid_profiles['fan_dominance_index'] > 0.7, 'type'] = '粉丝决定型'
        valid_profiles.loc[valid_profiles['fan_dominance_index'] < 0.3, 'type'] = '实力决定型'
        
        # 绘制散点图
        scatter = sns.scatterplot(
            data=valid_profiles,
            x='performance_sensitivity',
            y='fan_dominance_index',
            hue='type',
            size='stability_index',
            sizes=(20, 200),
            alpha=0.7,
            palette={
                '粉丝决定型': COLORS['casual'],
                '实力决定型': COLORS['base'],
                '双轮驱动型': COLORS['total']
            },
            edgecolors='black',
            linewidth=0.5
        )
        
        # 标记争议选手
        for celeb, season in self.controversial_contestants:
            mask = (valid_profiles['celebrity'] == celeb) & (valid_profiles['season'] == season)
            if mask.any():
                row = valid_profiles[mask].iloc[0]
                plt.scatter(
                    row['performance_sensitivity'],
                    row['fan_dominance_index'],
                    color='red',
                    s=200,
                    marker='*',
                    edgecolors='black',
                    linewidth=2,
                    label=f'{celeb} (S{season})'
                )
                plt.annotate(
                    celeb,
                    xy=(row['performance_sensitivity'], row['fan_dominance_index']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )
        
        # 添加分类边界线
        plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
        
        # 设置标签和标题
        plt.title('DWTS选手分类散点图（粉丝主导度 vs 实力敏感度）', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('实力敏感度（PS）', fontsize=12, fontweight='bold')
        plt.ylabel('粉丝主导度（FDI）', fontsize=12, fontweight='bold')
        plt.xlim(-0.3, 1.0)
        plt.ylim(-0.1, 1.1)
        
        # 添加区域标注
        plt.text(0.05, 0.85, '粉丝决定型\n(Fan-Driven)', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['casual'], alpha=0.2))
        plt.text(0.6, 0.2, '实力决定型\n(Judge-Driven)', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['base'], alpha=0.2))
        plt.text(0.6, 0.85, '双轮驱动型\n(Hybrid)', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['total'], alpha=0.2))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['PIC_ROOT'], 'contestant_classification_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("选手分类散点图已保存")

    def plot_scoring_system_comparison(self):
        """绘制两种评分系统对比图"""
        # 按评分系统分组
        self.results_df['scoring_system'] = '排名制'
        self.results_df.loc[(self.results_df['season'] >= 3) & (self.results_df['season'] < 28), 'scoring_system'] = '百分比制'
        
        # 计算每周的平均观众分占比和标准差
        system_stats = self.results_df.groupby(['scoring_system', 'season', 'week']).agg({
            'fan_share_mean': 'mean',
            'fan_share_std': 'mean',
            'model_confidence': 'mean'
        }).reset_index()
        
        plt.figure(figsize=(16, 8))
        
        # 绘制箱线图
        ax1 = plt.subplot(1, 2, 1)
        sns.boxplot(
            data=self.results_df,
            x='scoring_system',
            y='fan_share_mean',
            palette=[COLORS['rank_system'], COLORS['percent_system']],
            ax=ax1
        )
        ax1.set_title('两种评分系统下观众分占比分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('评分系统', fontsize=10)
        ax1.set_ylabel('观众分占比均值', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 绘制可信度对比
        ax2 = plt.subplot(1, 2, 2)
        sns.barplot(
            data=system_stats,
            x='scoring_system',
            y='model_confidence',
            palette=[COLORS['rank_system'], COLORS['percent_system']],
            ax=ax2
        )
        ax2.set_title('两种评分系统下模型平均可信度', fontsize=12, fontweight='bold')
        ax2.set_xlabel('评分系统', fontsize=10)
        ax2.set_ylabel('模型可信度（粒子存活率）', fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for p in ax2.patches:
            height = p.get_height()
            ax2.text(p.get_x() + p.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('DWTS两种评分系统对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['PIC_ROOT'], 'scoring_system_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("评分系统对比图已保存")

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("\n开始生成可视化图表...")
        self.plot_model_credibility_heatmap()
        self.plot_controversial_contestants_dynamics()
        self.plot_contestant_classification_scatter()
        self.plot_scoring_system_comparison()
        print("所有可视化图表生成完成！")

# ==========================================
# 6. 模型评价指标
# ==========================================
class ModelEvaluator:
    """模型评价类"""
    def __init__(self, results_df, credibility_df):
        self.results_df = results_df
        self.credibility_df = credibility_df

    def calculate_consistency_metrics(self):
        """计算模型一致性指标"""
        # 1. 整体可信度
        overall_credibility = self.credibility_df['survival_rate'].mean()
        
        # 2. 周度可信度稳定性
        weekly_credibility_std = self.credibility_df.groupby('week')['survival_rate'].std().mean()
        
        # 3. 赛季可信度差异
        season_credibility_range = self.credibility_df.groupby('season')['survival_rate'].mean().max() - \
                                 self.credibility_df.groupby('season')['survival_rate'].mean().min()
        
        # 4. 观众分估计不确定性
        avg_fan_std = self.results_df['fan_share_std'].mean()
        std_variation = self.results_df['fan_share_std'].std()
        
        metrics = {
            '整体模型可信度': overall_credibility,
            '周度可信度稳定性（标准差）': weekly_credibility_std,
            '赛季可信度差异范围': season_credibility_range,
            '平均观众分估计标准差': avg_fan_std,
            '估计不确定性波动': std_variation
        }
        
        # 打印结果
        print("\n=== 模型一致性评价指标 ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        # 保存到文件
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标名称', '数值'])
        metrics_df.to_csv(os.path.join(CONFIG['CACHE_DIR'], 'model_evaluation_metrics.csv'), index=False, encoding='utf-8-sig')
        
        return metrics

# ==========================================
# 主程序入口
# ==========================================
def main():
    """主运行函数"""
    print("="*60)
    print("DWTS双源动力粒子滤波观众分估计模型")
    print("="*60)
    
    # 步骤1：数据预处理
    print("\n【步骤1/4】数据预处理...")
    preprocessor = DataPreprocessor(CONFIG['DATA_PATH'])
    processed_df = preprocessor.preprocess()
    
    # 步骤2：模型训练与估计
    print("\n【步骤2/4】模型训练与观众分估计...")
    estimator = DualSourceEstimator(processed_df)
    results_df, credibility_df, profiles_df = estimator.run_all_seasons()
    
    # 步骤3：模型评价
    print("\n【步骤3/4】模型评价...")
    evaluator = ModelEvaluator(results_df, credibility_df)
    evaluator.calculate_consistency_metrics()
    
    # 步骤4：可视化生成
    print("\n【步骤4/4】可视化生成...")
    visualizer = ModelVisualizer(results_df, credibility_df, profiles_df)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"结果文件：{CONFIG['OUTPUT_PATH']}")
    print(f"可视化图表：{CONFIG['PIC_ROOT']}")
    print("="*60)

if __name__ == "__main__":
    main()