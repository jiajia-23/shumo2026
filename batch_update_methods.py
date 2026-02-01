"""
Batch update script to fix all ModelEvaluator methods
"""
import re

# Read the file
with open('dual_source_particle_filter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Remove @staticmethod from ModelEvaluator methods (lines 627-939)
# We'll do this by finding and replacing specific patterns

# Step 2: Add 'self' parameter and fix paths
replacements = [
    # plot_top_contestants_comparison
    (
        r"@staticmethod\s+def plot_top_contestants_comparison\(results_df,",
        "def plot_top_contestants_comparison(self, results_df,"
    ),
    (
        r"os\.path\.join\(PLOT_DIR, f'top_\{top_n\}_contestants_comparison\.png'\)",
        "os.path.join(self.paths['fan_support'], f'top_{top_n}_contestants_comparison.png')"
    ),
    # plot_confidence_over_time
    (
        r"@staticmethod\s+def plot_confidence_over_time\(credibility_df,",
        "def plot_confidence_over_time(self, credibility_df,"
    ),
    (
        r"os\.path\.join\(PLOT_DIR, 'confidence_over_time\.png'\)",
        "os.path.join(self.paths['analysis'], 'confidence_over_time.png')"
    ),
    # plot_fan_support_trend_season
    (
        r"@staticmethod\s+def plot_fan_support_trend_season\(results_df,",
        "def plot_fan_support_trend_season(self, results_df,"
    ),
    (
        r"os\.path\.join\(PLOT_DIR, f'fan_support_trend_season\{season\}\.png'\)",
        "os.path.join(self.paths['fan_support'], f'fan_support_trend_season{season}.png')"
    ),
    # plot_prediction_interval_comparison
    (
        r"@staticmethod\s+def plot_prediction_interval_comparison\(old_results_path,",
        "def plot_prediction_interval_comparison(self, old_results_path,"
    ),
    (
        r"os\.path\.join\(PLOT_DIR, f'interval_overlay_season\{season\}\.png'\)",
        "os.path.join(self.paths['prediction'], f'interval_overlay_season{season}.png')"
    ),
    # plot_single_contestant_interval_comparison
    (
        r"@staticmethod\s+def plot_single_contestant_interval_comparison\(old_results_path,",
        "def plot_single_contestant_interval_comparison(self, old_results_path,"
    ),
    (
        r"os\.path\.join\(PLOT_DIR, f'interval_comparison_\{safe_name\}\{season_str\}\.png'\)",
        "os.path.join(self.paths['prediction'], f'interval_comparison_{safe_name}{season_str}.png')"
    ),
]

# Apply all replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write the updated content
with open('dual_source_particle_filter.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Updated all ModelEvaluator methods")
print("✓ Removed @staticmethod decorators")
print("✓ Added 'self' parameters")
print("✓ Fixed output paths")
