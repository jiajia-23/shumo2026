import pandas as pd
import numpy as np
import re
from scipy.stats import dirichlet

# ==========================================
# 1. 增强版数据预处理模块
# ==========================================
class DWTSAdvancedProcessor:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)

    def _parse_elim_week(self, result_str):
        if pd.isna(result_str): return 99
        match = re.search(r'Eliminated Week (\d+)', str(result_str))
        return int(match.group(1)) if match else 99

    def process(self):
        melted_list = []
        score_cols = [c for c in self.raw_data.columns if 'judge' in c and '_score' in c]
        
        for _, row in self.raw_data.iterrows():
            elim_week = self._parse_elim_week(row['results'])
            for week in range(1, 12):
                week_prefix = f'week{week}_'
                v_scores = [row[c] for c in score_cols if c.startswith(week_prefix) and not pd.isna(row[c])]
                if not v_scores: continue 
                
                melted_list.append({
                    'season': int(row['season']),
                    'week': week,
                    'name': row['celebrity_name'],
                    'judge_sum': sum(v_scores),
                    'is_eliminated': 1 if elim_week == week else 0
                })
        
        df = pd.DataFrame(melted_list)

        # 定义评分机制
        def apply_mechanism(group):
            s = group.name[0]
            # 1. 排名制 (1-2 季, 28-34 季)
            if s <= 2 or s >= 28:
                # 分数越高，排名越小(1为最佳)
                group['judge_val'] = group['judge_sum'].rank(ascending=False, method='min')
                group['mode'] = 'rank'
            # 2. 百分比制 (3-27 季)
            else:
                group['judge_val'] = group['judge_sum'] / group['judge_sum'].sum()
                group['mode'] = 'percent'
            return group

        df = df.groupby(['season', 'week'], group_keys=False).apply(apply_mechanism)
        return df

# ==========================================
# 2. 三阶段贝叶斯推理引擎
# ==========================================
class TriplePhaseInferenceEngine:
    def __init__(self, n_iter=3000):
        self.n_iter = n_iter

    def infer(self, df):
        final_results = []
        for (s, w), group in df.groupby(['season', 'week']):
            n = len(group)
            mode = group['mode'].iloc[0]
            j_vals = group['judge_val'].values
            elim_mask = group['is_eliminated'].values
            names = group['name'].values
            
            # 默认值
            fan_est = np.full(n, 1.0/n)
            
            if elim_mask.any():
                elim_idx = np.where(elim_mask == 1)[0][0]
                
                # 采样逻辑：根据不同赛制设定约束
                samples = dirichlet.rvs([1.0]*n, size=self.n_iter)
                
                if mode == 'percent':
                    # 约束：总百分比最低者淘汰
                    totals = j_vals + samples
                    valid = samples[np.argmin(totals, axis=1) == elim_idx]
                elif s <= 2:
                    # 约束：总排名(JudgeRank + FanRank)最高者淘汰
                    fan_ranks = ((-samples).argsort(axis=1).argsort(axis=1) + 1)
                    totals = j_vals + fan_ranks
                    valid = samples[np.argmax(totals, axis=1) == elim_idx]
                else: # 28季及以后
                    # 约束：淘汰者必须在总排名的最后两名 (Bottom 2)
                    fan_ranks = ((-samples).argsort(axis=1).argsort(axis=1) + 1)
                    totals = j_vals + fan_ranks
                    # 找到每一行中排名最大的两个索引
                    bottom_two_indices = np.argsort(-totals, axis=1)[:, :2]
                    valid = samples[np.any(bottom_two_indices == elim_idx, axis=1)]

                if len(valid) > 0:
                    fan_est = valid.mean(axis=0)
                else:
                    # 鲁棒性回退逻辑
                    fan_est = self._robust_fallback(j_vals, elim_idx, mode, n)

            for i in range(n):
                final_results.append({
                    'season': s, 'week': w, 'name': names[i], 
                    'est_fan_percent': fan_est[i], 'mode': mode
                })
        
        return pd.merge(df, pd.DataFrame(final_results), on=['season', 'week', 'name', 'mode'])

    def _robust_fallback(self, j_vals, elim_idx, mode, n):
        # 简单启发式：给淘汰者分配较少的模拟权重
        res = np.ones(n)
        res[elim_idx] = 0.5
        return res / res.sum()

# ==========================================
# 3. 运行与验证
# ==========================================
processor = DWTSAdvancedProcessor('./2026_MCM_Problem_C_Data.csv')
full_data = processor.process()

engine = TriplePhaseInferenceEngine()
results = engine.infer(full_data)

# 查看第 28 季（新规第一季）的结果
print(results[results['season'] == 28][['week', 'name', 'judge_val', 'est_fan_percent', 'is_eliminated']].head(10))
results.to_csv('./Q1/DWTS_TriplePhase_Results.csv', index=False)