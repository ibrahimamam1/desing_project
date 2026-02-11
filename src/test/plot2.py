import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & STYLE
# =============================================================================
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Updated Scenarios for Fixed TL Phase Comparison
SCENARIOS_MAP = {
    'FTL10': 'Fixed TL (10s)',
    'FTL20': 'Fixed TL (20s)',
    'FTL30': 'Fixed TL (30s)',
    'FTL40': 'Fixed TL (40s)'
}

INTENTIONS_MAP = {
    'all_straight': 'All Straight',
    'all_left': 'All Left',
    'uniform_random': 'Uniform Random',
    'assymetric_random': 'Asymmetric Random'
}

# Sequential Red Palette to represent increasing time durations
SCENARIO_COLORS = {
    'FTL10': '#fadbd8', 
    'FTL20': '#f1948a',
    'FTL30': '#e74c3c',
    'FTL40': '#943126'
}

# Directory Setup
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
results_dir = os.path.join(root_dir, "results/tls")
output_dir = os.path.join(root_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# DATA LOADING & PARSING
# =============================================================================
def parse_filename(filename):
    """Parse CSV filename to extract metadata based on scenario/intention maps"""
    basename = os.path.basename(filename).replace('.csv', '')
    
    scenario = None
    intention = None
    
    # Extract scenario from key maps
    for scen in SCENARIOS_MAP.keys():
        if scen in basename:
            scenario = scen
            remaining = basename.replace(scen, '').strip('_')
            break

    if scenario:
        for intent in INTENTIONS_MAP.keys():
            if intent in remaining:
                intention = intent
                traffic_rate = remaining.replace(intent, '').strip('_')
                break
        else:
            traffic_rate = remaining
    else:
        traffic_rate = basename

    return {
        'scenario': scenario,
        'intention': intention,
        'traffic_rate': traffic_rate,
        'group_name': basename
    }

def load_data():
    """Load all CSV files and combine into single DataFrame"""
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {results_dir}")

    if not csv_files:
        return None

    all_data = []
    for csv_file in csv_files:
        try:
            meta = parse_filename(csv_file)
            df = pd.read_csv(csv_file)
            df['scenario'] = meta['scenario']
            df['intention'] = meta['intention']
            df['traffic_rate'] = meta['traffic_rate']
            df['group_name'] = meta['group_name']
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined)} total records")
        return combined
    return None

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_intention_scenario_bars(df):
    """4 Subplots (Intentions), grouped bars for the 4 FTL timings"""
    intentions = [i for i in INTENTIONS_MAP.keys() if i in df['intention'].unique()]
    if not intentions: return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    scenario_order = list(SCENARIOS_MAP.keys())

    for idx, intention in enumerate(intentions):
        ax = axes[idx]
        intent_data = df[df['intention'] == intention]
        grouped = intent_data.groupby(['traffic_rate', 'scenario'])['travel_time'].mean().reset_index()
        pivot = grouped.pivot(index='traffic_rate', columns='scenario', values='travel_time')
        
        # Ensure we only plot what we have
        cols_present = [s for s in scenario_order if s in pivot.columns]
        pivot = pivot.reindex(columns=cols_present)

        x = np.arange(len(pivot.index))
        width = 0.2  # Thinner for 4 scenarios

        for i, col in enumerate(pivot.columns):
            offset = width * (i - 1.5)
            ax.bar(x + offset, pivot[col], width, 
                   label=SCENARIOS_MAP[col], color=SCENARIO_COLORS.get(col, '#95a5a6'))

        ax.set_xlabel('Traffic Scenario', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Travel Time (s)', fontsize=10, fontweight='bold')
        ax.set_title(f'Intention: {INTENTIONS_MAP.get(intention, intention)}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
        ax.legend(title='FTL Duration', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Traffic Light Duration Performance comparison\n(Grouped by Intention)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "1A_FTL_duration_bars.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_control_type_heatmaps(df):
    """Heatmaps for each timing side by side"""
    scenarios = [s for s in SCENARIOS_MAP.keys() if s in df['scenario'].unique()]
    if not scenarios: return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 6))
    if len(scenarios) == 1: axes = [axes]

    vmin, vmax = df['travel_time'].min(), df['travel_time'].max()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_data = df[df['scenario'] == scenario]
        pivot = scenario_data.groupby(['intention', 'traffic_rate'])['travel_time'].mean().unstack()
        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                    ax=ax, cbar=(idx == len(scenarios)-1), 
                    vmin=vmin, vmax=vmax, linewidths=0.5)
        
        ax.set_title(f'{SCENARIOS_MAP[scenario]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Traffic Scenario', fontsize=11)
        if idx == 0: ax.set_ylabel('Intention Type', fontsize=11)
        else: ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1B_FTL_duration_heatmaps.png"), dpi=300)
    plt.close()

def plot_scenario_boxplots_and_cdf(df):
    """Distribution of travel times for each timing"""
    scenarios = [s for s in SCENARIOS_MAP.keys() if s in df['scenario'].unique()]
    n_scen = len(scenarios)
    if n_scen == 0: return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Individual Boxplots
    for i, scenario in enumerate(scenarios):
        ax = fig.add_axes([0.05 + i*(0.9/n_scen), 0.55, 0.9/n_scen*0.8, 0.35])
        scen_data = df[df['scenario'] == scenario]['travel_time'].dropna()
        
        ax.boxplot(scen_data, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor=SCENARIO_COLORS.get(scenario, '#95a5a6'), alpha=0.7),
                   medianprops=dict(color='black', linewidth=2))
        
        mean_val = scen_data.mean()
        ax.plot(1, mean_val, marker='*', color='blue', markersize=12, label=f'Mean: {mean_val:.1f}s')
        
        ax.set_title(SCENARIOS_MAP[scenario], fontweight='bold')
        ax.set_ylim(0, df['travel_time'].max() * 1.05)
        ax.set_xticklabels([''])
        ax.legend(loc='upper right', fontsize=8)

    # Combined CDF
    ax_cdf = fig.add_subplot(gs[1])
    for scenario in scenarios:
        scen_data = df[df['scenario'] == scenario]['travel_time'].dropna()
        sorted_data = np.sort(scen_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax_cdf.plot(sorted_data, cdf, label=SCENARIOS_MAP[scenario], 
                    color=SCENARIO_COLORS.get(scenario, '#95a5a6'), linewidth=2.5)
    
    ax_cdf.set_title('Cumulative Distribution: Impact of TL Timing', fontsize=12, fontweight='bold')
    ax_cdf.set_xlabel('Travel Time (s)')
    ax_cdf.set_ylabel('Probability')
    ax_cdf.legend()
    
    plt.savefig(os.path.join(output_dir, "4_FTL_boxplot_cdf.png"), dpi=300)
    plt.close()

def plot_summary_heatmap(df):
    """Detailed summary heatmap of all combinations"""
    fig, ax = plt.subplots(figsize=(14, 8))
    df['scenario_traffic'] = df['scenario'] + ' | ' + df['traffic_rate']
    pivot = df.groupby(['intention', 'scenario_traffic'])['travel_time'].mean().unstack()
    pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Travel Time (s)'})
    ax.set_title('Full Performance Summary (Duration vs Traffic)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_summary_heatmap.png"), dpi=300)
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*60)
    print("SUMO Simulation Analysis - Duration Comparison Mode")
    print("="*60)

    df = load_data()
    if df is None:
        print("Error: No data found. Check your results folder.")
        return

    print("\nGenerating visualizations...")
    plot_intention_scenario_bars(df)
    plot_control_type_heatmaps(df)
    plot_scenario_boxplots_and_cdf(df)
    plot_summary_heatmap(df)

    # Stats Summary
    summary_csv = os.path.join(output_dir, "duration_comparison_stats.csv")
    df.groupby(['scenario', 'intention', 'traffic_rate'])['travel_time'].mean().to_csv(summary_csv)

    print("\n" + "="*60)
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
