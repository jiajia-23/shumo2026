"""
Script to update ModelEvaluator methods with proper paths
"""
import re

# Read the file
with open('dual_source_particle_filter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define path mappings for each visualization method
path_mappings = {
    'plot_latent_dynamics': 'fan_support',
    'plot_top_contestants_comparison': 'fan_support',
    'plot_confidence_over_time': 'analysis',
    'plot_fan_support_trend_season': 'fan_support',
    'plot_prediction_interval_comparison': 'prediction',
    'plot_single_contestant_interval_comparison': 'prediction',
}

# Step 1: Remove @staticmethod decorators from ModelEvaluator class methods
# (but not from ScoringSystem class)
in_model_evaluator = False
lines = content.split('\n')
new_lines = []

for i, line in enumerate(lines):
    if 'class ModelEvaluator:' in line:
        in_model_evaluator = True
    elif line.startswith('class ') and 'ModelEvaluator' not in line:
        in_model_evaluator = False

    # Remove @staticmethod if we're in ModelEvaluator class
    if in_model_evaluator and line.strip() == '@staticmethod':
        continue  # Skip this line

    new_lines.append(line)

content = '\n'.join(new_lines)

# Step 2: Add 'self' parameter to method definitions in ModelEvaluator
for method_name in path_mappings.keys():
    # Pattern: def method_name(param1, param2, ...)
    pattern = rf'(def {method_name}\()([^)]+\))'

    def add_self(match):
        method_def = match.group(1)
        params = match.group(2)
        # Add 'self, ' at the beginning of parameters
        return f"{method_def}self, {params}"

    content = re.sub(pattern, add_self, content)

# Step 3: Replace PLOT_DIR with appropriate self.paths[...]
for method_name, path_type in path_mappings.items():
    # Find the method and replace PLOT_DIR within it
    # This is a simplified approach - we'll replace all PLOT_DIR in the file
    # with the appropriate path based on the method context
    pass

# For now, let's do a simple replacement of PLOT_DIR
# We'll need to be more careful about which path to use
content = content.replace('PLOT_DIR', "self.paths['PLACEHOLDER']")

# Write back
with open('dual_source_particle_filter_updated.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated file saved as dual_source_particle_filter_updated.py")
print("Please review and manually adjust path types (prediction/fan_support/analysis)")
