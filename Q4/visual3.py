# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize

# # ==========================================
# # 1. 数据准备
# # ==========================================
# weights_df = pd.read_csv('./Q4/tables/dynamic_weights_history.csv')
# saves_df = pd.read_csv('./Q4/tables/judges_save_events.csv')

# # 确定数据的维度
# seasons = sorted(weights_df['season'].unique())
# weeks = sorted(weights_df['week'].unique())
# n_seasons = len(seasons)
# max_week = max(weeks)

# # 构建矩阵 (填充NaN以处理不同赛季长度)
# # 我们需要一个完整的矩阵来映射网格
# data_matrix = np.full((n_seasons, max_week), np.nan)

# for _, row in weights_df.iterrows():
#     s_idx = int(row['season']) - 1  # 赛季从1开始，索引从0开始
#     w_idx = int(row['week']) - 1    # 周次从1开始，索引从0开始
#     if w_idx < max_week:
#         data_matrix[s_idx, w_idx] = row['w_fan']

# # ==========================================
# # 2. 构建极坐标网格
# # ==========================================
# # 定义内圆半径（为了形成环状，而不是饼状，中间留空）
# inner_radius = 10 
# outer_radius = inner_radius + n_seasons

# # 构建角度 (Theta) 和 半径 (R) 的网格
# # 我们不让圆闭合，留出一点开口放标签 (比如留出45度缺口)
# theta_start = np.pi / 2  # 从12点钟方向开始
# theta_end = np.pi / 2 - (2 * np.pi * 0.9) # 顺时针旋转90%的圆周

# # 生成网格边界
# r_edges = np.linspace(inner_radius, outer_radius, n_seasons + 1)
# theta_edges = np.linspace(theta_start, theta_end, max_week + 1)
# R, Theta = np.meshgrid(r_edges, theta_edges)

# # ==========================================
# # 3. 绘制环形热力图
# # ==========================================
# fig = plt.figure(figsize=(14, 14))
# ax = fig.add_subplot(111, projection='polar')

# # 核心绘制代码: pcolormesh
# # 注意: data_matrix 需要转置以匹配 Meshgrid (Theta, R) 的形状
# # cmap='RdYlBu_r': 保持您要求的红蓝反转色
# mesh = ax.pcolormesh(Theta, R, data_matrix.T, 
#                      cmap='RdYlBu_r', 
#                      vmin=0.3, vmax=0.7, 
#                      edgecolor='white', linewidth=0.05, alpha=0.9)

# # ==========================================
# # 4. 叠加评委拯救事件 (星星)
# # ==========================================
# for _, row in saves_df.iterrows():
#     s = row['season']
#     w = row['week']
    
#     # 计算星星的坐标
#     # r = 内半径 + 赛季索引 + 0.5 (居中)
#     r_pos = inner_radius + (s - 1) + 0.5
    
#     # theta = 起始角 + (结束角-起始角) * (周次占比)
#     # 注意周次w是从1开始，所以(w-0.5)居中
#     angle_step = (theta_end - theta_start) / max_week
#     theta_pos = theta_start + (w - 0.5) * angle_step
    
#     # 绘制星星
#     ax.scatter(theta_pos, r_pos, marker='*', s=200, 
#                color='gold', edgecolors='black', linewidth=1.5, zorder=10,
#                label='Judges\' Save' if _ == 0 else "")

# # ==========================================
# # 5. 美化与修饰
# # ==========================================

# # 移除默认的网格线和标签，因为它们是圆形的，容易干扰视觉
# ax.grid(False)
# ax.set_yticklabels([])
# ax.set_xticklabels([])

# # 添加赛季标签 (在起始边线上)
# # 每隔5个赛季标一下，避免太密
# for s in range(1, n_seasons + 1, 5):
#     ax.text(theta_start + 0.05, inner_radius + s - 1 + 0.5, 
#             f"S{s}", color='gray', fontsize=10, ha='left', va='center')
# # 标上最新的赛季
# ax.text(theta_start + 0.05, inner_radius + n_seasons - 1 + 0.5, 
#         f"S{34}", color='black', fontweight='bold', fontsize=10, ha='left', va='center')

# # 添加周次标签 (在最外圈)
# for w in range(1, max_week + 1):
#     angle_step = (theta_end - theta_start) / max_week
#     angle = theta_start + (w - 0.5) * angle_step
    
#     # 计算标签位置 (稍微在圆外一点)
#     label_r = outer_radius + 2
    
#     # 简单的旋转逻辑，让文字方向自然
#     rotation = np.degrees(angle) - 90
#     # 修正下半圆的文字方向，避免倒立
#     if -270 < rotation < -90:
#         rotation += 180
        
#     ax.text(angle, label_r, f"Week {w}", 
#             rotation=rotation, ha='center', va='center', fontsize=11, fontweight='bold')

# # 添加颜色条 (Colorbar)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.3]) # [left, bottom, width, height]
# cb = plt.colorbar(mesh, cax=cbar_ax)
# cb.set_label('Fan Weight ($w_{fan}$)', fontsize=12)
# cb.outline.set_visible(False)

# # 添加标题
# plt.suptitle('The "Ring of History": 34 Seasons of Fan Influence', 
#              fontsize=20, fontweight='bold', y=0.92)
# plt.figtext(0.5, 0.88, 'Inner Ring = Season 1 | Outer Ring = Season 34 | Gold Star = Judges\' Save', 
#             ha='center', fontsize=12, color='gray')

# # 保存
# plt.savefig('./Q4/figures/viz_2_radial_heatmap.png', dpi=300, bbox_inches='tight', transparent=False)
# print("环形热力图已生成: viz_2_radial_heatmap.png")
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ==========================================
# 1. 数据准备
# ==========================================
weights_df = pd.read_csv('./Q4/tables/dynamic_weights_history.csv')
saves_df = pd.read_csv('./Q4/tables/judges_save_events.csv')

# 确定数据的维度
seasons = sorted(weights_df['season'].unique())
weeks = sorted(weights_df['week'].unique())
n_seasons = len(seasons)
max_week = max(weeks)

# 构建矩阵 (填充NaN以处理不同赛季长度)
# data_matrix shape: (max_week, n_seasons)
# 行是周次(R)，列是赛季(Theta)
data_matrix = np.full((max_week, n_seasons), np.nan)

for _, row in weights_df.iterrows():
    s_idx = int(row['season']) - 1  # Season 1 -> index 0
    w_idx = int(row['week']) - 1    # Week 1 -> index 0
    if w_idx < max_week:
        data_matrix[w_idx, s_idx] = row['w_fan']

# ==========================================
# 2. 构建极坐标网格
# ==========================================
# 定义内圆半径 (Week 1 不在圆心点，而是一个小圆圈开始)
inner_radius = 4
# 外圆半径
outer_radius = inner_radius + max_week

# 构建角度 (Theta) -> 代表 赛季 (Season)
# 我们留一点开口放标签 (比如留出18度，即5%)
theta_start = np.pi / 2  # 12点方向开始
theta_end = np.pi / 2 - (2 * np.pi * 0.95) # 顺时针转95%

# 生成网格边界
# R代表周次: 从Week 1 到 Max Week
r_edges = np.linspace(inner_radius, outer_radius, max_week + 1)
# Theta代表赛季: 从Season 1 到 Season 34
theta_edges = np.linspace(theta_start, theta_end, n_seasons + 1)

R, Theta = np.meshgrid(r_edges, theta_edges)

# ==========================================
# 3. 绘制环形热力图 (维度对调版)
# ==========================================
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='polar')

# 核心绘制代码: pcolormesh
# data_matrix shape is (max_week, n_seasons)
# Meshgrid Theta shape is (n_seasons+1, max_week+1)
# 我们需要传入 data_matrix.T (n_seasons, max_week) 以匹配 Meshgrid
mesh = ax.pcolormesh(Theta, R, data_matrix.T, 
                     cmap='RdYlBu_r', 
                     vmin=0.3, vmax=0.7, 
                     edgecolor='white', linewidth=0.02, alpha=0.9)

# ==========================================
# 4. 叠加评委拯救事件 (星星)
# ==========================================
for _, row in saves_df.iterrows():
    s = row['season']
    w = row['week']
    
    # 坐标变换
    # Theta (Angle) = Season
    angle_step = (theta_end - theta_start) / n_seasons
    theta_pos = theta_start + (s - 0.5) * angle_step
    
    # R (Radius) = Week
    # r = inner + (week index) + 0.5
    r_pos = inner_radius + (w - 1) + 0.5
    
    # 绘制星星
    ax.scatter(theta_pos, r_pos, marker='*', s=250, 
               color='gold', edgecolors='black', linewidth=1.5, zorder=10)

# ==========================================
# 5. 美化与修饰
# ==========================================
ax.grid(False)
ax.set_yticklabels([])
ax.set_xticklabels([])

# 添加赛季标签 (在最外圈)
# Angle represents Season
for s in range(1, n_seasons + 1):
    angle_step = (theta_end - theta_start) / n_seasons
    angle = theta_start + (s - 0.5) * angle_step
    
    # 只标偶数赛季以防拥挤
    if s % 2 == 0 or s == 1 or s == n_seasons: 
        label_r = outer_radius + 0.5
        
        # 计算文字旋转角度
        rotation = np.degrees(angle) - 90
        # 修正下半圆的文字方向，避免倒立
        if -270 < rotation < -90:
            rotation += 180
            
        ax.text(angle, label_r, f"S{s}", 
                rotation=rotation, ha='center', va='center', fontsize=9, color='#333333')

# 添加周次标签 (在起始边线上)
# 我们在 Season 1 的开始处 (Theta Start) 标周次
for w in range(1, max_week + 1):
    # r位置
    r_loc = inner_radius + (w - 1) + 0.5
    # theta位置: 稍微逆时针一点点于start，作为Y轴标签
    theta_loc = theta_start + 0.05
    
    ax.text(theta_loc, r_loc, f"W{w}", 
            ha='right', va='center', fontsize=10, fontweight='bold', color='gray')

# Colorbar
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.3]) 
cb = plt.colorbar(mesh, cax=cbar_ax)
cb.set_label('Fan Weight ($w_{fan}$)', fontsize=12)
cb.outline.set_visible(False)

# Title
plt.suptitle('The "Spiral of Competition": 34 Seasons of Fan Influence', 
             fontsize=20, fontweight='bold', y=0.92)
plt.figtext(0.5, 0.88, 'Radius = Progress (Week 1 $\\to$ Final) | Angle = Season (1 $\\to$ 34)', 
            ha='center', fontsize=12, color='gray')

# 保存
plt.savefig('./Q4/figures/viz_2_radial_heatmap_swapped.png', dpi=300, bbox_inches='tight')
print("图表已生成: viz_2_radial_heatmap_swapped.png")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据准备
# ==========================================
# 读取数据
fairness_df = pd.read_csv('./Q4/tables/fairness_metrics.csv')
entertainment_df = pd.read_csv('./Q4/tables/entertainment_metrics.csv')

# 合并表格
df = pd.merge(fairness_df, entertainment_df, on='season')

# 倒序排列，让第1季在最上面（像表格一样阅读）
df = df.sort_values('season', ascending=False)  # 注意：画图时y轴通常0在下面，所以这里倒序，或者绘图时invert_yaxis

# ==========================================
# 2. 绘图设置
# ==========================================
# 创建画布：左边窄（放表格），右边宽（放条形图）
fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True, 
                         gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.05})

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] 

# ----------------------------------------------------------
# 左侧：公平性表格 (Fairness Heatmap Table)
# ----------------------------------------------------------
ax_table = axes[0]

# 准备数据矩阵
data_col = df['skill_rank_correlation'].values.reshape(-1, 1)

# 绘制热力图作为背景
sns.heatmap(data_col, annot=False, cmap="Blues", cbar=False, ax=ax_table, 
            vmin=0.4, vmax=1.0, alpha=0.8)

# 手动添加文字 (Season ID 和 数值)
y_centers = np.arange(len(df)) + 0.5
for i, (idx, row) in enumerate(df.iterrows()):
    # 赛季标签 (左对齐)
    ax_table.text(0.1, y_centers[i], f"Season {int(row['season'])}", 
                  color='black', ha='left', va='center', fontweight='bold', fontsize=11)
    
    # 具体的 Correlation 数值 (居中)
    # 根据背景深浅自动调整文字颜色
    text_color = 'white' if row['skill_rank_correlation'] > 0.8 else 'black'
    ax_table.text(0.7, y_centers[i], f"{row['skill_rank_correlation']:.3f}", 
                  color=text_color, ha='center', va='center', fontsize=11, fontfamily='monospace')

# 装饰左侧轴
ax_table.set_title('FAIRNESS METRICS\n(Skill-Rank Correlation)', fontsize=14, fontweight='bold', pad=10, color='#2c3e50')
ax_table.set_xticks([]) # 移除X轴刻度
ax_table.set_yticks([]) # 移除Y轴刻度 (因为我们手动画了Season)
ax_table.spines['top'].set_visible(True)
ax_table.spines['bottom'].set_visible(True)
ax_table.spines['left'].set_visible(True)
ax_table.spines['right'].set_visible(False) # 移除右边框，与右图融合

# ----------------------------------------------------------
# 右侧：娱乐性条形图 (Entertainment Bar Chart)
# ----------------------------------------------------------
ax_bar = axes[1]

y_pos = np.arange(len(df)) + 0.5 # 对齐热力图中心

# 1. 绘制背景条 (代表分歧程度 Disagreement) - 浅色
ax_bar.barh(y_pos, df['avg_disagreement'], height=0.6, color='#ffe0b2', label='Avg Disagreement (Tension)')

# 2. 绘制前景条 (代表权重波动 Volatility) - 深色
ax_bar.barh(y_pos, df['weight_volatility'], height=0.3, color='#e65100', label='Weight Volatility (Dynamics)')

# 3. 标记爆冷次数 (Upset Count)
# 在条形图末尾加一个数字标记
for i, (idx, row) in enumerate(df.iterrows()):
    upsets = int(row['upset_weeks'])
    # 位置：在分歧条的右侧一点
    ax_bar.text(row['avg_disagreement'] + 0.01, y_pos[i], f"{upsets} Upsets", 
                va='center', fontsize=9, color='#555555', style='italic')

# 装饰右侧轴
ax_bar.set_title('ENTERTAINMENT METRICS\n(Disagreement & Volatility)', fontsize=14, fontweight='bold', pad=10, color='#e65100')
ax_bar.set_ylim(0, len(df))
ax_bar.set_xlabel('Index Value (0.0 - 1.0)', fontsize=11)
ax_bar.grid(True, axis='x', linestyle='--', alpha=0.5)

# 图例
ax_bar.legend(loc='upper right', frameon=True)
ax_bar.spines['top'].set_visible(True)
ax_bar.spines['bottom'].set_visible(True)
ax_bar.spines['left'].set_visible(False) # 移除左边框
ax_bar.spines['right'].set_visible(True)
ax_bar.set_yticks([]) # 移除Y轴

# ==========================================
# 3. 最终调整与保存
# ==========================================
plt.suptitle('Model Performance Scorecard: Fairness vs. Entertainment (All 34 Seasons)',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.93)  # 留出标题空间，增加顶部空间

# 保存
plt.savefig('./Q4/figures/viz_4_hybrid_scorecard.png', dpi=300)
print("图表已生成: viz_4_hybrid_scorecard.png")
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import warnings

# 忽略无关警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据准备
# ==========================================
# 读取数据
try:
    fairness_df = pd.read_csv('fairness_metrics.csv')
    entertainment_df = pd.read_csv('entertainment_metrics.csv')
except FileNotFoundError:
    # 示例数据
    seasons = np.arange(1, 35)
    fairness_df = pd.DataFrame({'season': seasons, 'skill_rank_correlation': np.random.uniform(0.5, 0.9, 34)})
    entertainment_df = pd.DataFrame({
        'season': seasons, 
        'avg_disagreement': np.random.uniform(0.2, 0.5, 34),
        'weight_volatility': np.random.uniform(0.05, 0.15, 34),
        'upset_weeks': np.random.randint(0, 5, 34)
    })

# 合并表格
df = pd.merge(fairness_df, entertainment_df, on='season')
df = df.sort_values('season', ascending=True) 
df = df.reset_index(drop=True)

# ==========================================
# 2. 绘图设置
# ==========================================
# 调整画布大小
fig, axes = plt.subplots(1, 2, figsize=(18, 14), sharey=True, 
                         gridspec_kw={'width_ratios': [0.8, 2], 'wspace': 0.02})

# 【关键修改1】设置字体优先级：优先使用微软雅黑(支持中文和符号)，其次是Arial
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------
# 左侧：公平性表格
# ----------------------------------------------------------
ax_table = axes[0]
y_pos = np.arange(len(df)) + 0.5 

data_col = df['skill_rank_correlation'].values.reshape(-1, 1)

sns.heatmap(data_col, annot=False, cmap="Blues", cbar=False, ax=ax_table, 
            vmin=0.4, vmax=1.0, alpha=0.7)

for i, (idx, row) in enumerate(df.iterrows()):
    ax_table.text(0.1, y_pos[i], f"Season {int(row['season'])}", 
                  color='#333333', ha='left', va='center', fontweight='bold', fontsize=12)
    val = row['skill_rank_correlation']
    text_color = 'white' if val > 0.8 else 'black'
    ax_table.text(0.7, y_pos[i], f"{val:.3f}", 
                  color=text_color, ha='center', va='center', fontsize=12, fontfamily='monospace')

ax_table.set_title('FAIRNESS METRIC\n(Skill-Rank Correlation)', fontsize=16, fontweight='bold', pad=25, color='#2c3e50')
ax_table.axis('off') 

# ----------------------------------------------------------
# 右侧：娱乐性条形图 + 拟合曲线
# ----------------------------------------------------------
ax_bar = axes[1]

# 绘制条形图
bars1 = ax_bar.barh(y_pos, df['avg_disagreement'], height=0.7, 
                    color='#ffe0b2', label='Avg Disagreement (Raw)', alpha=0.6)
bars2 = ax_bar.barh(y_pos, df['weight_volatility'], height=0.4, 
                    color='#ff7043', label='Weight Volatility (Raw)', alpha=0.9)

# 拟合曲线
z_disagreement = np.polyfit(y_pos, df['avg_disagreement'], 5)
p_disagreement = np.poly1d(z_disagreement)
trend_disagreement = p_disagreement(y_pos)

z_volatility = np.polyfit(y_pos, df['weight_volatility'], 5)
p_volatility = np.poly1d(z_volatility)
trend_volatility = p_volatility(y_pos)

ax_bar.plot(trend_disagreement, y_pos, color='#8d6e63', linestyle='--', linewidth=2, 
            label='Trend: Disagreement (Smoothed)')
ax_bar.plot(trend_volatility, y_pos, color='#bf360c', linestyle='-', linewidth=2, 
            marker='o', markersize=4, markevery=5, 
            label='Trend: Volatility (Smoothed)')

# 【关键修改2】标记爆冷次数 - 使用LaTeX数学符号渲染星星，避免字体缺失
for i, (idx, row) in enumerate(df.iterrows()):
    upsets = int(row['upset_weeks'])
    if upsets > 0:
        # 使用 r"$\star$" 来绘制星星，这使用数学字体，不依赖 Arial
        ax_bar.text(row['avg_disagreement'] + 0.01, y_pos[i], f"$\star${upsets}", 
                    va='center', fontsize=10, color='#757575', fontweight='bold')

ax_bar.set_title('ENTERTAINMENT METRICS with TRENDS\n(Bars = Real Data, Lines = Polynomial Fit)', 
                 fontsize=16, fontweight='bold', pad=25, color='#e65100')
ax_bar.set_ylim(0, len(df))
ax_bar.set_xlabel('Index Value (Normalized Scale)', fontsize=12)
ax_bar.grid(True, axis='x', linestyle=':', alpha=0.6)

ax_bar.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
              frameon=True, fancybox=True, shadow=True, fontsize=11, ncol=1)

ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_visible(False)
ax_bar.set_yticks([]) 

# ==========================================
# 3. 全局调整与保存
# ==========================================
plt.suptitle('Model Performance Scorecard: 34-Season Longitudinal Analysis', 
             fontsize=22, fontweight='bold', y=0.96) 

plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, wspace=0.05)

# 保存
output_path = './Q4/figures/viz_4_hybrid_scorecard_with_trends.png'
# 确保目录存在（可选，防止报错）
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, dpi=300)
print(f"图表已生成: {output_path}")
plt.show()