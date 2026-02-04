import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# 1. Load Data
# Ensure 'merged_simulation_results.csv' is in the same folder
df = pd.read_csv('merged_simulation_results.csv')

# 2. Parse 'simul_id' into readable categories
def parse_simul_id(sim_id):
    # Extracts Control, Pattern, Scenario, Condition from the ID string
    match = re.match(r'(adaptive_tl|fixed_tl|rbl)_(.*)_(Sc\d+)_(.*)', sim_id)
    if match:
        return match.groups()
    return None, None, None, None

parsed_data = df['simul_id'].apply(parse_simul_id)
df[['Control_Method', 'Traffic_Pattern', 'Scenario', 'Condition']] = pd.DataFrame(
    parsed_data.tolist(), index=df.index
)

# 3. Set Plotting Style
sns.set_theme(style="whitegrid")

# --- PLOT 1: Average Travel Time Comparison ---
print("Generating avg_travel_time_comparison.png...")
g = sns.catplot(
    data=df, kind="bar",
    x="Scenario", y="avg_travel_time", hue="Control_Method",
    col="Traffic_Pattern", col_wrap=2,
    height=4, aspect=1.5,
    palette="viridis",
    alpha=0.9
)
g.set_axis_labels("Scenario", "Average Travel Time (s)")
g.set_titles("{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Average Travel Time by Control Method')
# Save figure
plt.savefig('avg_travel_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 2: Throughput vs Efficiency Scatter ---
print("Generating throughput_vs_efficiency.png...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="number_of_vehicles_left_successfully",
    y="avg_travel_time",
    hue="Control_Method",
    style="Traffic_Pattern",
    s=100,
    alpha=0.8,
    palette="deep"
)
plt.title("Throughput vs. Average Travel Time")
plt.xlabel("Number of Vehicles Left Successfully")
plt.ylabel("Average Travel Time (s)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
# Save figure
plt.savefig('throughput_vs_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 3: Intersection Delay (Control Zone Time) ---
print("Generating control_zone_delay.png...")
g = sns.catplot(
    data=df, kind="bar",
    x="Scenario", y="avg_time_in_control_zone", hue="Control_Method",
    col="Traffic_Pattern", col_wrap=2,
    height=4, aspect=1.5,
    palette="magma",
    alpha=0.9
)
g.set_axis_labels("Scenario", "Avg Time in Control Zone (s)")
g.set_titles("{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Intersection Delay (Control Zone Time)')
# Save figure
plt.savefig('control_zone_delay.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved successfully.")
