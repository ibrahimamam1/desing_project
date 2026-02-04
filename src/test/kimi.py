import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================================
# CONFIGURATION
# ============================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# File paths
INPUT_FILE = 'merged_simulation_results.csv'
OUTPUT_DIR = './'

# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

def load_and_preprocess_data(filepath):
    """Load data and extract components from simulation IDs."""
    df = pd.read_csv(filepath)
    
    # Parse the simul_id to extract components
    def parse_simul_id(simul_id):
        parts = simul_id.split('_')
        
        # Control strategy (first 2 parts)
        control_strategy = '_'.join(parts[:2])
        
        # Traffic pattern
        if 'all_straight' in simul_id:
            traffic_pattern = 'all_straight'
        elif 'all_left' in simul_id:
            traffic_pattern = 'all_left'
        elif 'uniform_random' in simul_id:
            traffic_pattern = 'uniform_random'
        elif 'assymetric_random' in simul_id:
            traffic_pattern = 'assymetric_random'
        else:
            traffic_pattern = 'unknown'
        
        # Scenario (starts with Sc)
        scenario = None
        for i, part in enumerate(parts):
            if part.startswith('Sc'):
                scenario = '_'.join(parts[i:])
                break
        
        return control_strategy, traffic_pattern, scenario
    
    # Apply parsing
    df[['control_strategy', 'traffic_pattern', 'scenario']] = df['simul_id'].apply(
        lambda x: pd.Series(parse_simul_id(x))
    )
    
    # Clean up control strategy names
    df['control_strategy_clean'] = df['control_strategy'].replace({
        'adaptive_tl': 'Adaptive TL',
        'fixed_tl': 'Fixed TL', 
        'rbl_all': 'RBL (All)',
        'rbl_uniform': 'RBL (Uniform)',
        'rbl_assymetric': 'RBL (Asymmetric)'
    })
    
    # Calculate efficiency metric
    df['efficiency'] = df['number_of_vehicles_left_successfully'] / df['avg_travel_time']
    
    # Create scenario ordering (Low → High traffic)
    scenario_order = ['Sc1_All_low', 'Sc5_Mixed_1H', 'Sc3_All_medium', 
                      'Sc4_Mixed_2H', 'Sc6_Mixed_ML', 'Sc2_All_high', 'Sc7_Mixed_3H']
    df['scenario_order'] = pd.Categorical(df['scenario'], categories=scenario_order, ordered=True)
    
    return df

# ============================================
# VISUALIZATION 1: DASHBOARD
# ============================================

def create_dashboard(df, output_path):
    """Create comprehensive dashboard with multiple views."""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Throughput comparison by control strategy (top-left)
    ax1 = plt.subplot(3, 3, 1)
    sns.boxplot(data=df, x='control_strategy_clean', y='number_of_vehicles_left_successfully', ax=ax1)
    ax1.set_title('Throughput by Control Strategy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Vehicles Successfully Left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Average travel time comparison (top-middle)
    ax2 = plt.subplot(3, 3, 2)
    sns.boxplot(data=df, x='control_strategy_clean', y='avg_travel_time', ax=ax2)
    ax2.set_title('Average Travel Time by Control Strategy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Avg Travel Time (s)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Time in control zone (top-right)
    ax3 = plt.subplot(3, 3, 3)
    sns.boxplot(data=df, x='control_strategy_clean', y='avg_time_in_control_zone', ax=ax3)
    ax3.set_title('Time in Control Zone by Strategy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('Avg Time in Control Zone (s)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Throughput heatmap (middle-left, larger)
    ax4 = plt.subplot(3, 3, (4, 5))
    pivot_throughput = df.pivot_table(
        values='number_of_vehicles_left_successfully',
        index='scenario',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    sns.heatmap(pivot_throughput, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax4)
    ax4.set_title('Throughput Heatmap: Scenario vs Control Strategy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Control Strategy')
    ax4.set_ylabel('Scenario')
    
    # 5. Travel time heatmap (middle-right, larger)
    ax5 = plt.subplot(3, 3, (6, 9))
    pivot_travel = df.pivot_table(
        values='avg_travel_time',
        index='scenario',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    sns.heatmap(pivot_travel, annot=True, fmt='.1f', cmap='YlOrRd_r', ax=ax5)
    ax5.set_title('Avg Travel Time Heatmap: Scenario vs Control Strategy', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Control Strategy')
    ax5.set_ylabel('Scenario')
    
    # 6. Traffic pattern impact on throughput (bottom-left)
    ax6 = plt.subplot(3, 3, 7)
    sns.barplot(data=df, x='traffic_pattern', y='number_of_vehicles_left_successfully', ax=ax6, palette='Set2')
    ax6.set_title('Throughput by Traffic Pattern', fontsize=12, fontweight='bold')
    ax6.set_xlabel('')
    ax6.set_ylabel('Vehicles Successfully Left')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Throughput vs Travel Time trade-off (bottom-middle)
    ax7 = plt.subplot(3, 3, 8)
    scatter = ax7.scatter(
        df['number_of_vehicles_left_successfully'],
        df['avg_travel_time'],
        c=df['control_strategy_clean'].astype('category').cat.codes,
        cmap='tab10',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax7.set_xlabel('Throughput (Vehicles Successfully Left)')
    ax7.set_ylabel('Avg Travel Time (s)')
    ax7.set_title('Throughput vs Travel Time Trade-off', fontsize=12, fontweight='bold')
    
    # Add legend for scatter plot
    handles = []
    for i, strategy in enumerate(df['control_strategy_clean'].unique()):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.tab10(i), markersize=10, label=strategy))
    ax7.legend(handles=handles, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/traffic_simulation_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Dashboard saved to {output_path}/traffic_simulation_dashboard.png")

# ============================================
# VISUALIZATION 2: SCENARIO COMPARISON
# ============================================

def create_scenario_comparison(df, output_path):
    """Create scenario comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    df_sorted = df.sort_values('scenario_order')
    
    # 1. Grouped bar chart: Throughput by Scenario
    ax1 = axes[0, 0]
    pivot_data = df_sorted.pivot_table(
        values='number_of_vehicles_left_successfully',
        index='scenario_order',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    pivot_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Throughput Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Traffic Scenario (Low → High)')
    ax1.set_ylabel('Vehicles Successfully Left')
    ax1.legend(title='Control Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Grouped bar chart: Travel Time by Scenario
    ax2 = axes[0, 1]
    pivot_travel = df_sorted.pivot_table(
        values='avg_travel_time',
        index='scenario_order',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    pivot_travel.plot(kind='bar', ax=ax2, width=0.8, 
                      color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    ax2.set_title('Average Travel Time Across Scenarios', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Traffic Scenario (Low → High)')
    ax2.set_ylabel('Avg Travel Time (s)')
    ax2.legend(title='Control Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Violin plot: Distribution of travel times
    ax3 = axes[1, 0]
    sns.violinplot(data=df, x='control_strategy_clean', y='avg_travel_time', 
                   ax=ax3, palette='Set2', inner='box')
    ax3.set_title('Distribution of Average Travel Times', fontsize=14, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('Avg Travel Time (s)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Empty subplot (can be used for efficiency or removed)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Scenario comparison saved to {output_path}/scenario_comparison.png")

# ============================================
# VISUALIZATION 3: EFFICIENCY COMPARISON
# ============================================

def create_efficiency_comparison(df, output_path):
    """Create efficiency comparison chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    df_sorted = df.sort_values('scenario_order')
    pivot_eff = df_sorted.pivot_table(
        values='efficiency',
        index='scenario_order',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    
    pivot_eff.plot(kind='bar', ax=ax, width=0.8, 
                   color=['#1abc9c', '#e67e22', '#34495e', '#e74c3c', '#95a5a6'])
    ax.set_title('Efficiency: Throughput / Travel Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Traffic Scenario (Low → High)')
    ax.set_ylabel('Efficiency (veh/s)')
    ax.legend(title='Control Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Efficiency comparison saved to {output_path}/efficiency_comparison.png")

# ============================================
# VISUALIZATION 4: SUMMARY AND CORRELATION
# ============================================

def create_summary_and_correlation(df, output_path):
    """Create correlation matrix and summary table."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Correlation matrix
    ax1 = axes[0]
    numeric_cols = [
        'number_of_vehicles_left_successfully', 
        'avg_travel_time', 
        'total_travel_time',
        'avg_time_in_control_zone',
        'total_time_in_control_zone'
    ]
    corr_matrix = df[numeric_cols].corr()
    
    # Rename for display
    corr_display = corr_matrix.copy()
    corr_display.index = ['Throughput', 'Avg Travel Time', 'Total Travel Time', 
                          'Avg Time in Zone', 'Total Time in Zone']
    corr_display.columns = ['Throughput', 'Avg Travel Time', 'Total Travel Time', 
                            'Avg Time in Zone', 'Total Time in Zone']
    
    sns.heatmap(corr_display, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax1.set_title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
    
    # 2. Summary table
    ax2 = axes[1]
    summary_stats = df.groupby('control_strategy_clean').agg({
        'number_of_vehicles_left_successfully': ['mean', 'std'],
        'avg_travel_time': ['mean', 'std'],
        'efficiency': ['mean', 'std']
    }).round(1)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    table_data = []
    for _, row in summary_stats.iterrows():
        table_data.append([
            row['control_strategy_clean'],
            f"{row['number_of_vehicles_left_successfully_mean']:.0f} ± {row['number_of_vehicles_left_successfully_std']:.0f}",
            f"{row['avg_travel_time_mean']:.1f} ± {row['avg_travel_time_std']:.1f}",
            f"{row['efficiency_mean']:.1f} ± {row['efficiency_std']:.1f}"
        ])
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(
        cellText=table_data,
        colLabels=['Control Strategy', 'Throughput\n(mean ± std)', 
                   'Avg Travel Time (s)\n(mean ± std)', 'Efficiency\n(mean ± std)'],
        cellLoc='center',
        loc='center',
        colColours=['#4472C4']*4,
        colWidths=[0.3, 0.25, 0.25, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor('#4472C4')
    
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax2.set_title('Performance Summary by Control Strategy', fontsize=14, fontweight='bold', y=0.75)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/summary_and_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary and correlation saved to {output_path}/summary_and_correlation.png")

# ============================================
# VISUALIZATION 5: TRAFFIC PATTERN ANALYSIS
# ============================================

def create_traffic_pattern_analysis(df, output_path):
    """Create traffic pattern impact analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Throughput by Traffic Pattern and Strategy
    ax1 = axes[0, 0]
    pivot_pattern = df.pivot_table(
        values='number_of_vehicles_left_successfully',
        index='traffic_pattern',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    pivot_pattern.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Throughput by Traffic Pattern and Strategy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Traffic Pattern')
    ax1.set_ylabel('Vehicles Successfully Left')
    ax1.legend(title='Control Strategy', fontsize=8, loc='upper right')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Travel Time by Traffic Pattern
    ax2 = axes[0, 1]
    pivot_pattern_travel = df.pivot_table(
        values='avg_travel_time',
        index='traffic_pattern',
        columns='control_strategy_clean',
        aggfunc='mean'
    )
    pivot_pattern_travel.plot(kind='bar', ax=ax2, width=0.8, 
                              color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    ax2.set_title('Travel Time by Traffic Pattern and Strategy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Traffic Pattern')
    ax2.set_ylabel('Avg Travel Time (s)')
    ax2.legend(title='Control Strategy', fontsize=8, loc='upper right')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Best strategy by scenario
    ax3 = axes[1, 0]
    best_by_scenario = df.loc[df.groupby('scenario')['efficiency'].idxmax()][['scenario', 'control_strategy_clean', 'efficiency']]
    pivot_best = best_by_scenario.pivot_table(
        values='efficiency',
        index='scenario',
        columns='control_strategy_clean',
        aggfunc='first'
    ).fillna(0)
    
    annot_array = pivot_best.copy().astype(str)
    annot_array[:] = ''
    for scenario in pivot_best.index:
        best_col = pivot_best.loc[scenario].idxmax()
        best_val = pivot_best.loc[scenario].max()
        if best_val > 0:
            annot_array.loc[scenario, best_col] = '★'
    
    sns.heatmap(pivot_best, annot=annot_array, fmt='', cmap='YlGnBu', ax=ax3, 
                cbar_kws={'label': 'Efficiency'})
    ax3.set_title('Best Performing Strategy by Scenario (★ = Best)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Control Strategy')
    ax3.set_ylabel('Scenario')
    
    # 4. Max travel time analysis
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='control_strategy_clean', y='max_travel_time', ax=ax4, palette='Set3')
    ax4.set_title('Maximum Travel Time by Strategy (Worst Case)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('Max Travel Time (s)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/traffic_pattern_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Traffic pattern analysis saved to {output_path}/traffic_pattern_analysis.png")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main function to generate all visualizations."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(INPUT_FILE)
    print(f"✓ Loaded {len(df)} simulations")
    
    print("\nGenerating visualizations...")
    create_dashboard(df, OUTPUT_DIR)
    create_scenario_comparison(df, OUTPUT_DIR)
    create_efficiency_comparison(df, OUTPUT_DIR)
    create_summary_and_correlation(df, OUTPUT_DIR)
    create_traffic_pattern_analysis(df, OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
