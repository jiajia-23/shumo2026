import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings

# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 适配字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载数据...")
try:
    weights_df = pd.read_csv('./Q4/tables/dynamic_weights_history.csv')
    fairness_df = pd.read_csv('./Q4/tables/fairness_metrics.csv')
    entertainment_df = pd.read_csv('./Q4/tables/entertainment_metrics.csv')
    saves_df = pd.read_csv('./Q4/tables/judges_save_events.csv')
except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保所有CSV文件都在当前目录下。")
    exit()

# 合并数据用于赛季级别的分析
season_metrics = pd.merge(fairness_df, entertainment_df, on='season')
# 计算爆冷率 (Upset Rate)
season_metrics['upset_rate'] = season_metrics['upset_weeks'] / season_metrics['total_weeks']
# 计算每个赛季的平均粉丝权重
season_metrics['avg_fan_weight'] = weights_df.groupby('season')['w_fan'].mean().values

# ==========================================
# 2. 可视化一：公平与娱乐的权衡矩阵 (3D Bubble Chart)
# ==========================================
print("正在生成图表 1: 公平与娱乐权衡气泡图...")

plt.figure(figsize=(12, 8))

# 绘制散点图
# X轴: 公平性 (相关系数)
# Y轴: 娱乐性 (权重波动性)
# 大小: 紧张度 (平均分歧指数)
# 颜色: 爆冷率
scatter = plt.scatter(
    x=season_metrics['skill_rank_correlation'],
    y=season_metrics['weight_volatility'],
    s=season_metrics['avg_disagreement'] * 1500, # 放大尺寸以便观察
    c=season_metrics['upset_rate'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='w',
    linewidth=1
)

# 标注几个关键赛季
interesting_seasons = [27, 5, 1, 34] # Bobby Bones, Sabrina, 第一季, 最新季
for season in interesting_seasons:
    if season in season_metrics['season'].values:
        row = season_metrics[season_metrics['season'] == season].iloc[0]
        plt.annotate(f"S{season}", 
                     (row['skill_rank_correlation'], row['weight_volatility']),
                     xytext=(0, 5), textcoords='offset points', 
                     fontsize=11, fontweight='bold', color='black')

# 添加标签和标题
plt.title('Trade-off Matrix: Fairness vs Entertainment (All 34 Seasons)\n(Bubble size = disagreement, color = upset rate)', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Fairness: skill-rank correlation (higher = fairer)', fontsize=12)
plt.ylabel('Entertainment: weight volatility (higher = more dynamic)', fontsize=12)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Upset rate (fan votes changing rankings)', rotation=270, labelpad=15)

# 添加网格和保存
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./Q4/figures/viz_1_bubble_tradeoff.png', dpi=300)
print("  - 已保存为 viz_1_bubble_tradeoff.png")
plt.close()


# ==========================================
# 3. 可视化二：粉丝权力演变热力图 (Heatmap)
# ==========================================
print("正在生成图表 2: 粉丝权重演变热力图...")

# 转换数据格式用于热力图 (行=赛季, 列=周次)
heatmap_data = weights_df.pivot(index='season', columns='week', values='w_fan')

plt.figure(figsize=(14, 10))
sns.set(font_scale=1.0) # 重置字体大小
# 绘制热力图，使用冷暖色调
ax = sns.heatmap(heatmap_data, cmap="RdYlBu_r", center=0.5, 
                 cbar_kws={'label': 'Fan weight ($w_{fan}$)'},
                 linewidths=.5, linecolor='gray',
                 vmin=0.3, vmax=0.7) # 锁定范围使颜色对比更明显

# 叠加评委拯救事件 (星号标记)
for idx, row in saves_df.iterrows():
    # 热力图的坐标是从0开始的，需要调整
    # Y轴: 赛季 (Season 1 对应索引 0)
    # X轴: 周次 (Week 1 对应索引 0)
    # 注意：如果您的赛季是从1开始的，需要减去热力图index的最小值来对齐
    try:
        y_loc = heatmap_data.index.get_loc(row['season']) + 0.5
        x_loc = heatmap_data.columns.get_loc(row['week']) + 0.5
        
        plt.plot(x_loc, y_loc, marker='*', color='gold', markersize=18, 
                 markeredgecolor='black', markeredgewidth=1.5)
    except KeyError:
        continue # 如果某个周次数据缺失则跳过

# 自定义图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='*', color='w', label='Judges\' Save triggered',
                          markerfacecolor='gold', markersize=15, markeredgecolor='black')]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.title('Competition "Heat": Fan-Power Dynamics Across 34 Seasons', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Week', fontsize=12)
plt.ylabel('Season', fontsize=12)
plt.tight_layout()
plt.savefig('./Q4/figures/viz_2_heatmap_weights.png', dpi=300)
print("  - 已保存为 viz_2_heatmap_weights.png")
plt.close()


# ==========================================
# 4. 可视化三：赛季特征指纹 (Radar Chart)
# ==========================================
print("正在生成图表 3: 赛季特征雷达图...")

# 选择特征并进行归一化处理
features = ['skill_rank_correlation', 'weight_volatility', 'avg_disagreement', 'upset_rate', 'avg_fan_weight']
feature_labels = ['Fairness\n(skill correlation)', 'Entertainment\n(weight volatility)', 'Tension\n(disagreement)', 'Upset\n(rate)', 'Fan Power\n(avg weight)']

# 数据归一化 (Min-Max Scaling) 以便在雷达图上展示
normalized_df = season_metrics.copy()
for feature in features:
    min_val = season_metrics[feature].min()
    max_val = season_metrics[feature].max()
    # 避免除以零
    if max_val - min_val == 0:
        normalized_df[feature] = 0.5
    else:
        normalized_df[feature] = (season_metrics[feature] - min_val) / (max_val - min_val)

# 准备绘图数据
# 选取 Season 5 (Sabrina - 遗憾), Season 27 (Bobby - 争议), 以及 全局平均
seasons_to_plot = [5, 27]
radar_data = []

# 获取选定赛季的数据
for s in seasons_to_plot:
    if s in normalized_df['season'].values:
        radar_data.append(normalized_df[normalized_df['season'] == s][features].iloc[0].values.tolist())
    else:
        print(f"警告: 赛季 {s} 数据不存在，跳过。")

# 获取平均数据
radar_data.append(normalized_df[features].mean().values.tolist()) 

# 构建雷达图
num_vars = len(features)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += [angles[0]] # 闭合回路

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 设置角度方向 (顺时针，顶部为0度)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 绘制轴标签
plt.xticks(angles[:-1], feature_labels, size=11)

# 绘制Y轴刻度
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=8)
plt.ylim(0, 1)

# 定义样式
colors = ['#FF5733', '#33FF57', '#3357FF'] # 红(S5), 绿(S27), 蓝(Avg)
labels = ['Season 5 (Sabrina, upset)', 'Season 27 (Bobby, controversy)', 'Historical average']

# 循环绘制
for i, d in enumerate(radar_data):
    val = d + [d[0]] # 闭合数据
    ax.plot(angles, val, linewidth=2, linestyle='solid', label=labels[i], color=colors[i])
    ax.fill(angles, val, color=colors[i], alpha=0.1)

plt.title('Season "Personality" Fingerprint: Controversial vs Heartbreak vs Average', size=15, y=1.1, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('./Q4/figures/viz_3_radar_profiles.png', dpi=300)
print("  - 已保存为 viz_3_radar_profiles.png")
plt.close()

print("\n所有图表生成完毕！")
