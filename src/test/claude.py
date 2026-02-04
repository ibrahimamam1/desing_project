#!/usr/bin/env python3
"""
Traffic Intersection Simulation Visualization Script
Generates comprehensive visualizations comparing different control strategies
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def load_data(filepath):
    """Load and prepare the simulation data"""
    df = pd.read_csv(filepath)
    
    # Extract control type - handle both traffic light and right-before-left types
    df['control_type'] = df['simul_id'].apply(lambda x: 
        'Adaptive TL' if x.startswith('adaptive_tl') else
        'Fixed TL' if x.startswith('fixed_tl') else
        'Right Before Left'
    )
    
    # Extract traffic pattern
    df['traffic_pattern'] = df['simul_id'].str.extract(r'(?:_tl|rbl)_(.+?)_Sc')[0]
    
    # Extract scenario
    df['scenario'] = df['simul_id'].str.extract(r'(Sc\d+_[^_]+(?:_[^_]+)?)$')[0]
    
    return df

def create_performance_comparison(df, output_dir):
    """Create grouped bar charts comparing control types"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison Across Control Types', fontsize=16, fontweight='bold')
    
    metrics = [
        ('number_of_vehicles_left_successfully', 'Throughput (vehicles)', axes[0, 0]),
        ('avg_travel_time', 'Average Travel Time (s)', axes[0, 1]),
        ('avg_time_in_control_zone', 'Average Time in Control Zone (s)', axes[1, 0]),
        ('total_travel_time', 'Total Travel Time (s)', axes[1, 1])
    ]
    
    for metric, label, ax in metrics:
        summary = df.groupby('control_type')[metric].agg(['mean', 'std']).reset_index()
        
        x = np.arange(len(summary))
        width = 0.6
        
        bars = ax.bar(x, summary['mean'], width, yerr=summary['std'], 
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Color bars
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Control Type', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary['control_type'], rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, summary['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_comparison.png")
    plt.close()

def create_scatter_analysis(df, output_dir):
    """Create scatter plots showing relationships between metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Relationship Analysis Between Key Metrics', fontsize=16, fontweight='bold')
    
    scatter_configs = [
        ('number_of_vehicles_left_successfully', 'avg_travel_time', 
         'Throughput vs Avg Travel Time', axes[0, 0]),
        ('number_of_vehicles_left_successfully', 'avg_time_in_control_zone',
         'Throughput vs Time in Control Zone', axes[0, 1]),
        ('avg_travel_time', 'avg_time_in_control_zone',
         'Avg Travel Time vs Time in Control Zone', axes[1, 0]),
        ('max_travel_time', 'avg_travel_time',
         'Max Travel Time vs Avg Travel Time', axes[1, 1])
    ]
    
    colors = {'Adaptive TL': '#2ecc71', 'Fixed TL': '#3498db', 'Right Before Left': '#e74c3c'}
    
    for x_metric, y_metric, title, ax in scatter_configs:
        for control_type in df['control_type'].unique():
            mask = df['control_type'] == control_type
            ax.scatter(df.loc[mask, x_metric], df.loc[mask, y_metric],
                      label=control_type, alpha=0.6, s=100, 
                      color=colors[control_type], edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scatter_analysis.png")
    plt.close()

def create_heatmap_performance(df, output_dir):
    """Create heatmap showing performance across scenarios and control types"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Performance Heatmap: Control Type vs Scenario', fontsize=16, fontweight='bold')
    
    # Heatmap 1: Average Travel Time
    pivot_travel = df.pivot_table(
        values='avg_travel_time', 
        index='control_type', 
        columns='scenario', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_travel, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                ax=axes[0], cbar_kws={'label': 'Seconds'}, linewidths=0.5)
    axes[0].set_title('Average Travel Time', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Scenario', fontweight='bold')
    axes[0].set_ylabel('Control Type', fontweight='bold')
    
    # Heatmap 2: Throughput
    pivot_throughput = df.pivot_table(
        values='number_of_vehicles_left_successfully', 
        index='control_type', 
        columns='scenario', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_throughput, annot=True, fmt='.0f', cmap='YlGnBu', 
                ax=axes[1], cbar_kws={'label': 'Vehicles'}, linewidths=0.5)
    axes[1].set_title('Throughput (Vehicles)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Scenario', fontweight='bold')
    axes[1].set_ylabel('Control Type', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: heatmap_performance.png")
    plt.close()

def create_boxplot_comparison(df, output_dir):
    """Create box plots comparing distributions across control types"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution Comparison Across Control Types (All Scenarios)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('avg_travel_time', 'Average Travel Time (s)', axes[0, 0]),
        ('avg_time_in_control_zone', 'Average Time in Control Zone (s)', axes[0, 1]),
        ('number_of_vehicles_left_successfully', 'Throughput (vehicles)', axes[1, 0]),
        ('max_travel_time', 'Maximum Travel Time (s)', axes[1, 1])
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for metric, label, ax in metrics:
        bp = ax.boxplot([df[df['control_type'] == ct][metric].values 
                         for ct in ['Adaptive TL', 'Fixed TL', 'Right Before Left']],
                        tick_labels=['Adaptive TL', 'Fixed TL', 'Right Before Left'],
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: boxplot_comparison.png")
    plt.close()

def create_per_scenario_boxplots(df, output_dir):
    """Create boxplots for specific traffic scenarios with uniform_random pattern"""
    # Filter for uniform_random pattern and specific scenarios
    scenarios_of_interest = [
        ('Sc1_All_low', 'Low Traffic'),
        ('Sc3_All_medium', 'Medium Traffic'),
        ('Sc2_All_high', 'High Traffic')
    ]
    
    df_filtered = df[
        (df['traffic_pattern'] == 'uniform_random') & 
        (df['scenario'].isin([s[0] for s in scenarios_of_interest]))
    ].copy()
    
    if len(df_filtered) == 0:
        print("⚠ No uniform_random data for specified scenarios")
        return
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Travel Time by Control Type: Uniform Random Traffic Pattern', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    control_types = ['Adaptive TL', 'Fixed TL', 'Right Before Left']
    
    for idx, (scenario_id, scenario_name) in enumerate(scenarios_of_interest):
        ax = axes[idx]
        scenario_data = df_filtered[df_filtered['scenario'] == scenario_id]
        
        # Prepare data for boxplot using min, avg, max
        box_data = []
        positions = []
        box_colors = []
        
        for i, ct in enumerate(control_types):
            ct_data = scenario_data[scenario_data['control_type'] == ct]
            
            if len(ct_data) > 0:
                min_val = ct_data['min_travel_time'].values[0]
                avg_val = ct_data['avg_travel_time'].values[0]
                max_val = ct_data['max_travel_time'].values[0]
                
                # Create box plot statistics manually
                # Format: [min, 25th percentile, median, 75th percentile, max]
                # Since we only have min, avg, max - we'll approximate:
                # Use min as whisker, avg as median, max as whisker
                # Estimate quartiles between min-avg and avg-max
                q1 = min_val + (avg_val - min_val) * 0.5
                q3 = avg_val + (max_val - avg_val) * 0.5
                
                stats = {
                    'med': avg_val,
                    'q1': q1, 
                    'q3': q3,
                    'whislo': min_val,
                    'whishi': max_val,
                    'mean': avg_val,
                    'fliers': []  # No outliers
                }
                
                box_data.append(stats)
                positions.append(i + 1)
                box_colors.append(colors[i])
        
        if box_data:
            # Create the boxplot from statistics
            bp = ax.bxp(box_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Travel Time (s)', fontweight='bold', fontsize=11)
            ax.set_title(scenario_name, fontsize=13, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels(control_types, rotation=20, ha='right')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {scenario_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(scenario_name, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_per_scenario.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: boxplot_per_scenario.png")
    plt.close()

def create_scenario_performance(df, output_dir):
    """Create line charts showing performance across scenarios"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Trends Across Traffic Scenarios', fontsize=16, fontweight='bold')
    
    metrics = [
        ('number_of_vehicles_left_successfully', 'Throughput (vehicles)', axes[0, 0]),
        ('avg_travel_time', 'Average Travel Time (s)', axes[0, 1]),
        ('avg_time_in_control_zone', 'Average Time in Control Zone (s)', axes[1, 0]),
        ('max_travel_time', 'Maximum Travel Time (s)', axes[1, 1])
    ]
    
    colors = {'Adaptive TL': '#2ecc71', 'Fixed TL': '#3498db', 'Right Before Left': '#e74c3c'}
    markers = {'Adaptive TL': 'o', 'Fixed TL': 's', 'Right Before Left': '^'}
    
    for metric, label, ax in metrics:
        for control_type in ['Adaptive TL', 'Fixed TL', 'Right Before Left']:
            control_df = df[df['control_type'] == control_type]
            scenario_avg = control_df.groupby('scenario')[metric].mean().sort_index()
            
            ax.plot(range(len(scenario_avg)), scenario_avg.values, 
                   marker=markers[control_type], linewidth=2, markersize=8,
                   label=control_type, color=colors[control_type], alpha=0.8)
        
        ax.set_xlabel('Scenario', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(scenario_avg)))
        ax.set_xticklabels(scenario_avg.index, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scenario_performance.png")
    plt.close()

def create_traffic_pattern_analysis(df, output_dir):
    """Analyze performance by traffic pattern"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance by Traffic Pattern', fontsize=16, fontweight='bold')
    
    # Filter out None values
    df_filtered = df[df['traffic_pattern'].notna()].copy()
    
    if len(df_filtered) == 0:
        print("⚠ No traffic pattern data available")
        plt.close()
        return
    
    colors_map = {'Adaptive TL': '#2ecc71', 'Fixed TL': '#3498db', 'Right Before Left': '#e74c3c'}
    
    # Plot 1: Throughput by pattern
    pivot_data = df_filtered.pivot_table(
        values='number_of_vehicles_left_successfully',
        index='traffic_pattern',
        columns='control_type',
        aggfunc='mean'
    )
    pivot_data.plot(kind='bar', ax=axes[0, 0], color=[colors_map[c] for c in pivot_data.columns], 
                    edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[0, 0].set_title('Throughput by Traffic Pattern', fontweight='bold')
    axes[0, 0].set_ylabel('Vehicles', fontweight='bold')
    axes[0, 0].set_xlabel('Traffic Pattern', fontweight='bold')
    axes[0, 0].legend(title='Control Type', framealpha=0.9)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Avg travel time by pattern
    pivot_data2 = df_filtered.pivot_table(
        values='avg_travel_time',
        index='traffic_pattern',
        columns='control_type',
        aggfunc='mean'
    )
    pivot_data2.plot(kind='bar', ax=axes[0, 1], color=[colors_map[c] for c in pivot_data2.columns],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[0, 1].set_title('Avg Travel Time by Traffic Pattern', fontweight='bold')
    axes[0, 1].set_ylabel('Seconds', fontweight='bold')
    axes[0, 1].set_xlabel('Traffic Pattern', fontweight='bold')
    axes[0, 1].legend(title='Control Type', framealpha=0.9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Time in control zone
    pivot_data3 = df_filtered.pivot_table(
        values='avg_time_in_control_zone',
        index='traffic_pattern',
        columns='control_type',
        aggfunc='mean'
    )
    pivot_data3.plot(kind='bar', ax=axes[1, 0], color=[colors_map[c] for c in pivot_data3.columns],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1, 0].set_title('Avg Time in Control Zone by Pattern', fontweight='bold')
    axes[1, 0].set_ylabel('Seconds', fontweight='bold')
    axes[1, 0].set_xlabel('Traffic Pattern', fontweight='bold')
    axes[1, 0].legend(title='Control Type', framealpha=0.9)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Efficiency metric (throughput / avg_travel_time)
    df_filtered['efficiency'] = (df_filtered['number_of_vehicles_left_successfully'] / 
                                  df_filtered['avg_travel_time'])
    pivot_data4 = df_filtered.pivot_table(
        values='efficiency',
        index='traffic_pattern',
        columns='control_type',
        aggfunc='mean'
    )
    pivot_data4.plot(kind='bar', ax=axes[1, 1], color=[colors_map[c] for c in pivot_data4.columns],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1, 1].set_title('Efficiency by Traffic Pattern (Throughput/Time)', fontweight='bold')
    axes[1, 1].set_ylabel('Vehicles/Second', fontweight='bold')
    axes[1, 1].set_xlabel('Traffic Pattern', fontweight='bold')
    axes[1, 1].legend(title='Control Type', framealpha=0.9)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: traffic_pattern_analysis.png")
    plt.close()

def create_correlation_matrix(df, output_dir):
    """Create correlation heatmap for key independent metrics"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select only the key independent metrics
    numeric_cols = [
        'number_of_vehicles_left_successfully',
        'total_travel_time',
        'total_time_in_control_zone'
    ]
    
    # Create readable labels
    labels = [
        'Vehicles\nEmitted',
        'Total\nTravel Time',
        'Total Time in\nControl Zone'
    ]
    
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, linewidths=2,
                cbar_kws={'label': 'Correlation Coefficient'},
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1)
    
    ax.set_title('Correlation Matrix: Key Performance Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: correlation_matrix.png")
    plt.close()

def create_summary_statistics(df, output_dir):
    """Generate summary statistics table"""
    summary = df.groupby('control_type').agg({
        'number_of_vehicles_left_successfully': ['mean', 'std', 'min', 'max'],
        'avg_travel_time': ['mean', 'std', 'min', 'max'],
        'avg_time_in_control_zone': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Save to CSV
    summary.to_csv(output_dir / 'summary_statistics.csv')
    print(f"✓ Saved: summary_statistics.csv")
    
    # Create a visual table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Control Type', 'Metric', 'Mean', 'Std Dev', 'Min', 'Max'])
    
    metrics_info = [
        ('number_of_vehicles_left_successfully', 'Throughput'),
        ('avg_travel_time', 'Avg Travel Time'),
        ('avg_time_in_control_zone', 'Avg Time in Zone')
    ]
    
    for control_type in summary.index:
        for i, (metric_col, metric_name) in enumerate(metrics_info):
            # Access values using the column multi-index
            mean_val = summary.loc[control_type, (metric_col, 'mean')]
            std_val = summary.loc[control_type, (metric_col, 'std')]
            min_val = summary.loc[control_type, (metric_col, 'min')]
            max_val = summary.loc[control_type, (metric_col, 'max')]
            
            table_data.append([
                control_type if i == 0 else '',
                metric_name,
                f"{mean_val:.1f}",
                f"{std_val:.1f}",
                f"{min_val:.1f}",
                f"{max_val:.1f}"
            ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary Statistics by Control Type', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: summary_table.png")
    plt.close()

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("Traffic Intersection Simulation Visualization")
    print("="*60 + "\n")
    
    # Setup paths
    data_path = Path('/home/rgb/Desktop/research/alpha_model/merged_simulation_results.csv')
    output_dir = Path('/home/rgb/Desktop/research/alpha_model/plots3')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    print(f"✓ Loaded {len(df)} simulation results\n")
    
    # Display basic info
    print(f"Control Types: {df['control_type'].unique().tolist()}")
    print(f"Number of scenarios: {df['scenario'].nunique()}")
    print(f"Traffic patterns: {df['traffic_pattern'].nunique()} unique patterns\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    create_performance_comparison(df, output_dir)
    create_boxplot_comparison(df, output_dir)
    create_per_scenario_boxplots(df, output_dir)
    create_scatter_analysis(df, output_dir)
    create_heatmap_performance(df, output_dir)
    create_scenario_performance(df, output_dir)
    create_traffic_pattern_analysis(df, output_dir)
    create_correlation_matrix(df, output_dir)
    create_summary_statistics(df, output_dir)
    
    print("\n" + "="*60)
    print("✓ All visualizations completed successfully!")
    print(f"✓ Output directory: {output_dir}")
    print("="*60 + "\n")
    
    # List all generated files
    print("Generated files:")
    for file in sorted(output_dir.glob('*.png')) + sorted(output_dir.glob('*.csv')):
        print(f"  • {file.name}")
    print()

if __name__ == "__main__":
    main()
