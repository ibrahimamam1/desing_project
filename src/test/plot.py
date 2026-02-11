import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Get paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
results_dir = os.path.join(root_dir, "results")
output_dir = os.path.join(root_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

print(f"Results directory: {results_dir}")
print(f"Output directory: {output_dir}")

# Scenarios and intentions reference
SCENARIOS_MAP = {
    'fixed_tl': 'Fixed TL',
    'adaptive_tl': 'Adaptive TL', 
    'rbl': 'Right Before Left'
}

INTENTIONS_MAP = {
    'all_straight': 'All Straight',
    'all_left': 'All Left',
    'uniform_random': 'Uniform Random',
    'assymetric_random': 'Asymmetric Random'
}

SCENARIO_COLORS = {
    'fixed_tl': '#e74c3c',
    'adaptive_tl': '#3498db',
    'rbl': '#2ecc71'
}

def parse_filename(filename):
    """Parse CSV filename to extract metadata"""
    basename = os.path.basename(filename).replace('.csv', '')

    scenarios = ['fixed_tl', 'adaptive_tl', 'rbl']
    intentions = ['all_straight', 'all_left', 'uniform_random', 'assymetric_random']

    scenario = None
    intention = None

    # Extract scenario
    for scen in scenarios:
        if scen in basename:
            scenario = scen
            remaining = basename.replace(scen, '').strip('_')
            break

    # Extract intention
    if scenario:
        for intent in intentions:
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
    print(f"Found {len(csv_files)} CSV files")

    if not csv_files:
        print("No CSV files found!")
        return None

    all_data = []
    for csv_file in csv_files:
        try:
            meta = parse_filename(csv_file)
            df = pd.read_csv(csv_file)

            # Add metadata
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
# PLOT 1: Per Intention, Per Scenario Group Bar Charts - 4 SUBPLOTS IN 1 FIGURE
# =============================================================================
def plot_intention_scenario_bars(df):
    """
    Single figure with 4 subplots (2x2 grid).
    Each subplot shows one intention with 7 scenarios on x-axis, 
    grouped bars for 3 control types.
    """
    intentions = df['intention'].dropna().unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    scenario_order = ['fixed_tl', 'adaptive_tl', 'rbl']

    for idx, intention in enumerate(intentions):
        ax = axes[idx]
        intent_data = df[df['intention'] == intention]

        # Group by traffic_rate and scenario, calculate mean travel time
        grouped = intent_data.groupby(['traffic_rate', 'scenario'])['travel_time'].mean().reset_index()

        # Pivot for plotting
        pivot = grouped.pivot(index='traffic_rate', columns='scenario', values='travel_time')

        # Reorder columns for consistent colors
        pivot = pivot.reindex(columns=[s for s in scenario_order if s in pivot.columns])

        # Plot
        x = np.arange(len(pivot.index))
        width = 0.25

        for i, col in enumerate(pivot.columns):
            offset = width * (i - 1)
            bars = ax.bar(x + offset, pivot[col], width, 
                          label=SCENARIOS_MAP[col], color=SCENARIO_COLORS[col])

        ax.set_xlabel('Traffic Scenario', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Travel Time (s)', fontsize=10, fontweight='bold')
        ax.set_title(f'{INTENTIONS_MAP.get(intention, intention)}', 
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
        ax.legend(title='Control Type', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Traffic Controller Performance on each scenario\n(Grouped by Intention)', 
                  fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = "1A_intention_scenario_bars.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# =============================================================================
# PLOT 2: Per Intention, Per Scenario HEATMAP
# =============================================================================

def plot_control_type_heatmaps(df):
    """
    Three heatmaps side by side, one for each control type.
    Y-axis: Intentions
    X-axis: Traffic scenarios
    Values: Average travel time
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = ['fixed_tl', 'adaptive_tl', 'rbl']
    
    # Find global min for consistent color scale
    vmin = df['travel_time'].min()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        # Filter data for this control type
        scenario_data = df[df['scenario'] == scenario]
        
        # Create pivot table: intentions (rows) x traffic_rates (columns)
        pivot = scenario_data.groupby(['intention', 'traffic_rate'])['travel_time'].mean().unstack()
        
        # Calculate vmax as the maximum of the grouped means for this specific heatmap
        vmax = pivot.max().max()
        
        # Map intention names for display
        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]
        
        # Sort traffic rates if they're numeric-like
        try:
            sorted_cols = sorted(pivot.columns, key=lambda x: float(x) if x.replace('.','').isdigit() else x)
            pivot = pivot[sorted_cols]
        except:
            pass
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                    ax=ax, cbar_kws={'label': 'Travel Time (s)'}, 
                    linewidths=0.5, vmin=vmin, vmax=vmax)
        
        ax.set_title(f'{SCENARIOS_MAP[scenario]}', 
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Traffic Scenario', fontsize=11, fontweight='bold')
        
        # Only show y-label on leftmost plot
        if idx == 0:
            ax.set_ylabel('Intention Type', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
    
    fig.suptitle('Heatmap showing how intention and scenario affect travel time', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = "1B_intention_scenario_heatmaps.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# PLOT 3: Travel Time Heatmap
# =============================================================================
def plot_heatmap(df):
    """
    Comprehensive heatmap showing travel time for all combinations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap 1: Average travel time by intention and scenario (averaged across traffic rates)
    pivot1 = df.groupby(['intention', 'scenario'])['travel_time'].mean().unstack()
    pivot1.index = [INTENTIONS_MAP.get(i, i) for i in pivot1.index]
    pivot1.columns = [SCENARIOS_MAP.get(c, c) for c in pivot1.columns]
    
    # Calculate vmax for this heatmap
    vmax1 = pivot1.max().max()

    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Travel Time (s)'}, linewidths=0.5,
                vmax=vmax1)
    axes[0].set_title('Travel Time Heatmap\n(Intention vs Control Type)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Control Type', fontsize=11)
    axes[0].set_ylabel('Intention', fontsize=11)

    # Heatmap 2: Include traffic scenarios as well (more detailed)
    df['scenario_traffic'] = df['scenario'] + ' | ' + df['traffic_rate']
    pivot2 = df.groupby(['intention', 'scenario_traffic'])['travel_time'].mean().unstack()
    pivot2.index = [INTENTIONS_MAP.get(i, i) for i in pivot2.index]
    
    # Calculate vmax for this heatmap
    vmax2 = pivot2.max().max()

    sns.heatmap(pivot2, annot=False, fmt='.1f', cmap='YlOrRd', 
                ax=axes[1], cbar_kws={'label': 'Travel Time (s)'}, vmax=vmax2)
    axes[1].set_title('Summary of traffic controller performance for all intentions and scenarios', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Control Type + Traffic Rate', fontsize=11)
    axes[1].set_ylabel('Intention', fontsize=11)
    axes[1].tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()
    filename = "3_summary_heatmap.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# PLOT 4: Per Scenario Box Plots and CDF - FIXED
# =============================================================================
def plot_scenario_boxplots_and_cdf(df):
    """Box plots and CDF for each scenario"""
    scenarios = df['scenario'].dropna().unique()
    n_scen = len(scenarios)
    
    # Create figure with box plots on top row and single CDF plot on bottom
    fig = plt.figure(figsize=(max(4*n_scen, 12), 10))
    
    # Create grid: top row for box plots, bottom row for single CDF
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top subplot for box plots
    ax_box_container = fig.add_subplot(gs[0])
    ax_box_container.axis('off')
    
    # Create individual box plot axes
    box_axes = []
    for i in range(n_scen):
        ax = fig.add_axes([0.1 + i*(0.8/n_scen), 0.55, 0.8/n_scen*0.9, 0.35])
        box_axes.append(ax)
    
    # Plot box plots
    for idx, scenario in enumerate(scenarios):
        scen_data = df[df['scenario'] == scenario]['travel_time'].dropna()
        color = SCENARIO_COLORS.get(scenario, '#95a5a6')
        mean_val = scen_data.mean()
        
        # Box plot - STANDARD
        ax_box = box_axes[idx]
        bp = ax_box.boxplot(scen_data, 
                            patch_artist=True,
                            widths=0.5,
                            showfliers=True,
                            boxprops=dict(facecolor=color, alpha=0.7),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        # Add mean as a star
        ax_box.plot(1, mean_val, marker='*', color='red', markersize=15, 
                   markeredgecolor='darkred', markeredgewidth=1.5, zorder=3,
                   label=f'Mean: {mean_val:.1f}s')
        
        ax_box.set_title(f'{SCENARIOS_MAP.get(scenario, scenario)}', 
                         fontsize=11, fontweight='bold')
        ax_box.set_ylabel('Travel Time (s)', fontsize=10)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.set_ylim(0, 200)
        ax_box.set_xticklabels([''])
        ax_box.legend(loc='upper right', fontsize=8)
    
    # Single CDF plot (bottom row)
    ax_cdf = fig.add_subplot(gs[1])
    
    for idx, scenario in enumerate(scenarios):
        scen_data = df[df['scenario'] == scenario]['travel_time'].dropna()
        color = SCENARIO_COLORS.get(scenario, '#95a5a6')
        sorted_data = np.sort(scen_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax_cdf.plot(sorted_data, cdf, linewidth=2.5, color=color, 
                   label=SCENARIOS_MAP.get(scenario, scenario))
    
    ax_cdf.set_xlabel('Travel Time (s)', fontsize=12)
    ax_cdf.set_ylabel('CDF', fontsize=12)
    ax_cdf.set_xlim(0, 250)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_title('Cumulative Distribution Functions - All Scenarios', 
                     fontsize=12, fontweight='bold')
    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    fig.suptitle('Travel time distribution by controller type on all scenarios + intentions', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    filename = "4_scenario_boxplot_cdf.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# PLOT 4: Per Intention Box Plots and CDF - FIXED BOX PLOTS (NO FANCINESS)
# =============================================================================
def plot_intention_boxplots_and_cdf(df):
    """
    Box plots and CDF for each intention.
    Box plots show: min, 1st quartile, median, 3rd quartile, max (standard).
    All CDFs plotted on a single plot for easy comparison.
    """
    intentions = df['intention'].dropna().unique()
    n_intents = len(intentions)
    
    # Create figure with box plots on top row and single CDF plot on bottom
    fig = plt.figure(figsize=(max(4*n_intents, 12), 10))
    
    # Create grid: top row for box plots, bottom row for single CDF
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top subplot for box plots
    ax_box_container = fig.add_subplot(gs[0])
    ax_box_container.axis('off')
    
    # Create individual box plot axes
    box_axes = []
    for i in range(n_intents):
        ax = fig.add_axes([0.1 + i*(0.8/n_intents), 0.55, 0.8/n_intents*0.9, 0.35])
        box_axes.append(ax)
    
    colors = sns.color_palette("Set2", n_intents)
    
    # Plot box plots
    for idx, intention in enumerate(intentions):
        intent_data = df[df['intention'] == intention]['travel_time'].dropna()
        mean_val = intent_data.mean()
        
        # Box plot (top row) - STANDARD BOX PLOT
        ax_box = box_axes[idx]
        bp = ax_box.boxplot(intent_data, 
                            patch_artist=True,
                            widths=0.5,
                            showfliers=True,  # Show outliers as points
                            boxprops=dict(facecolor=colors[idx], alpha=0.7),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        # Add mean as an orange star
        ax_box.plot(1, mean_val, marker='*', color='orange', markersize=15, 
                   markeredgecolor='darkorange', markeredgewidth=1.5, zorder=3,
                   label=f'Mean: {mean_val:.1f}s')
        
        ax_box.set_title(f'{INTENTIONS_MAP.get(intention, intention)}', 
                         fontsize=11, fontweight='bold')
        ax_box.set_ylabel('Travel Time (s)', fontsize=10)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.set_ylim(0, 200)
        ax_box.set_xticklabels([''])
        ax_box.legend(loc='upper right', fontsize=8)
    
    # Single CDF plot (bottom row)
    ax_cdf = fig.add_subplot(gs[1])
    
    for idx, intention in enumerate(intentions):
        intent_data = df[df['intention'] == intention]['travel_time'].dropna()
        sorted_data = np.sort(intent_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax_cdf.plot(sorted_data, cdf, linewidth=2.5, color=colors[idx], 
                   label=INTENTIONS_MAP.get(intention, intention))
    
    ax_cdf.set_xlabel('Travel Time (s)', fontsize=12)
    ax_cdf.set_ylabel('CDF', fontsize=12)
    ax_cdf.set_xlim(0, 250)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_title('Travel Time distribution for each intention(all the scenarios summarised)', 
                     fontsize=12, fontweight='bold')
    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    fig.suptitle('Travel Time Distribution by Intention(all controllers + all scenarios)\n)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    filename = "5_intention_boxplot_cdf.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# =============================================================================
# STATISTICS SUMMARY
# =============================================================================
def generate_statistics(df):
    """Generate and save statistical summary"""
    stats = {}

    # Overall stats
    stats['overall'] = df['travel_time'].describe()

    # By intention
    stats['by_intention'] = df.groupby('intention')['travel_time'].agg(['mean', 'std', 'min', 'max', 'count'])

    # By scenario
    stats['by_scenario'] = df.groupby('scenario')['travel_time'].agg(['mean', 'std', 'min', 'max', 'count'])

    # By combination
    stats['by_combination'] = df.groupby(['scenario', 'intention'])['travel_time'].agg(['mean', 'std', 'count'])

    # Save to text file
    stats_file = os.path.join(output_dir, "statistics_summary.txt")
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SUMO SIMULATION STATISTICS SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-"*60 + "\n")
        f.write(stats['overall'].to_string())

        f.write("\n\n" + "="*60 + "\n")
        f.write("BY INTENTION\n")
        f.write("-"*60 + "\n")
        f.write(stats['by_intention'].to_string())

        f.write("\n\n" + "="*60 + "\n")
        f.write("BY SCENARIO (CONTROL TYPE)\n")
        f.write("-"*60 + "\n")
        f.write(stats['by_scenario'].to_string())

        f.write("\n\n" + "="*60 + "\n")
        f.write("BY COMBINATION (Scenario + Intention)\n")
        f.write("-"*60 + "\n")
        f.write(stats['by_combination'].to_string())

    # Also save main results to CSV for easy import
    summary_csv = os.path.join(output_dir, "summary_by_scenario_intention.csv")
    stats['by_combination'].reset_index().to_csv(summary_csv, index=False)

    print(f"Saved statistics: statistics_summary.txt and summary_by_scenario_intention.csv")
    return stats

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("SUMO Simulation Analysis - Updated Version")
    print("="*60)

    # Load data
    df = load_data()
    if df is None:
        return

    print(f"\nData summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Scenarios: {df['scenario'].unique()}")
    print(f"  Intentions: {df['intention'].unique()}")
    print(f"  Traffic rates: {df['traffic_rate'].nunique()}")

    # Generate all plots
    print("\nGenerating visualizations...")

    plot_intention_scenario_bars(df)
    plot_control_type_heatmaps(df)
    plot_intention_boxplots_and_cdf(df)
    plot_scenario_boxplots_and_cdf(df)
    plot_heatmap(df)

    print("\nGenerating statistics...")
    generate_statistics(df)

    print("\n" + "="*60)
    print(f"Analysis complete! Output saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
