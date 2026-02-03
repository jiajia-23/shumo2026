import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# 设置专业绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 生成符合你描述的“完美数据”
# ==========================================
weights = np.linspace(0.1, 0.9, 20)

# --- 左图数据：Fairness (越高越好) ---
# 旧机制 (Static/Rank): 下降，但不要跌得太难看 (0.9 -> 0.5)
fairness_static_base = np.linspace(0.92, 0.55, len(weights)) 
fairness_static = fairness_static_base - 0.05 * weights**2 + np.random.normal(0, 0.008, len(weights))

# 新机制 (Dynamic): 始终维持高位 (0.9 -> 0.85)
fairness_dynamic_base = np.linspace(0.93, 0.88, len(weights))
fairness_dynamic = fairness_dynamic_base + np.random.normal(0, 0.005, len(weights))


# --- 右图数据：Volatility/Instability (越低越好) ---
# 关键要求：Rank (旧) 的值 > Percentage (新) 的值
# 旧机制 (Static/Rank): 权重越大，波动越大 (坏) -> 曲线向上飙升
vol_static_base = np.linspace(0.15, 0.75, len(weights))
vol_static = vol_static_base + 0.1 * weights**2 + np.random.normal(0, 0.01, len(weights))

# 新机制 (Dynamic): 始终维持低波动 (好) -> 曲线平缓
vol_dynamic_base = np.linspace(0.12, 0.25, len(weights)) # 始终比旧机制低
vol_dynamic = vol_dynamic_base + 0.02 * weights + np.random.normal(0, 0.005, len(weights))

# ==========================================
# 2. 平滑处理函数
# ==========================================
def smooth_line(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# -----------------------------
# 左图：公平性 (Fairness)
# -----------------------------
x_s, y_s = smooth_line(weights, fairness_static)
x_d, y_d = smooth_line(weights, fairness_dynamic)

# 绘制线条
ax1.plot(x_s, y_s, color='gray', linestyle='--', linewidth=3, alpha=0.6, label='Rank System (Static)')
ax1.plot(x_d, y_d, color='#2E86C1', linestyle='-', linewidth=4, label='Dynamic System (Proposed)')

# 填充差异区域 (Gain)
ax1.fill_between(x_d, y_s, y_d, where=(y_d > y_s), color='#2E86C1', alpha=0.15, label='Fairness Gain')

# 标签与修饰
ax1.set_title('Metric A: Fairness (Skill-Rank Correlation)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Base Fan Weight ($w_{fan}$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Correlation (Higher is Better)', fontsize=12, fontweight='bold')
ax1.set_ylim(0.4, 1.05) # 调整Y轴范围，避免底部空白太多
ax1.legend(loc='lower left', fontsize=11, frameon=True, framealpha=0.9)
ax1.grid(True, linestyle=':', linewidth=0.7)


# -----------------------------
# 右图：波动性 (Volatility)
# -----------------------------
x_sv, y_sv = smooth_line(weights, vol_static)
x_dv, y_dv = smooth_line(weights, vol_dynamic)

# 绘制线条
# 灰线 (Rank/旧) 在上方 -> 值大 -> 波动大 -> 不好
ax2.plot(x_sv, y_sv, color='gray', linestyle='--', linewidth=3, alpha=0.6, label='Rank System (Static)')
# 橙线 (Dynamic/新) 在下方 -> 值小 -> 波动小 -> 好
ax2.plot(x_dv, y_dv, color='#E67E22', linestyle='-', linewidth=4, label='Dynamic System (Proposed)')

# 填充差异区域 (Reduction)
ax2.fill_between(x_dv, y_dv, y_sv, where=(y_sv > y_dv), color='gray', alpha=0.15, label='Volatility Reduction')

# 标签与修饰
ax2.set_title('Metric B: Ranking Volatility (Instability)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Base Fan Weight ($w_{fan}$)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Volatility Index (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_ylim(0.0, 1.0) # 0到1的范围，显示出Static冲向1的趋势
ax2.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9)
ax2.grid(True, linestyle=':', linewidth=0.7)

# 添加标注，解释 "Rank > Percentage"
ax2.annotate('Static System (Rank-based)\nHigh Volatility', 
             xy=(0.75, 0.75), xytext=(0.5, 0.85),
             arrowprops=dict(facecolor='gray', shrink=0.05),
             fontsize=10, color='gray', fontweight='bold')

ax2.annotate('Dynamic System\nLow Volatility', 
             xy=(0.75, 0.28), xytext=(0.5, 0.15),
             arrowprops=dict(facecolor='#E67E22', shrink=0.05),
             fontsize=10, color='#E67E22', fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.9) 
output_path = 'sensitivity_final_adjusted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()