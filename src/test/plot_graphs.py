import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Configuration
results_dir = "results/processed"  # Adjust to your directory
scenarios = ["right_before_left", "fixed_traffic_light", "adaptive_traffic_light"]
traffic_rates = [1000, 1200, 1400, 1600, 1800, 2000]

# Colors for each scenario
colors = {
    "right_before_left": "#3498db",  # Blue
    "fixed_traffic_light": "#e74c3c",           # Red
    "adaptive_traffic_light": "#2ecc71"         # Green
}

# Nice labels for scenarios
scenario_labels = {
    "right_before_left": "Right Before Left",
    "fixed_traffic_light": "Fixed Traffic Light",
    "adaptive_traffic_light": "Adaptive Traffic Light"
}

def load_and_aggregate_data(scenario, traffic_rate):
    """Load CSV and compute aggregate metrics"""
    filename = os.path.join(results_dir, f"{scenario}_{traffic_rate}veh_hr.csv")
    
    try:
        df = pd.read_csv(filename)
        
        # Compute aggregate metrics
        metrics = {
            'total_vehicles': len(df),
            'avg_waiting_time': df['total_waiting_time'].mean(),
            'avg_travel_time': df['travel_time'].mean(),
            'avg_speed': df['avg_speed'].mean()
        }
        return metrics
    except FileNotFoundError:
        print(f"Warning: File not found - {filename}")
        return None

def collect_all_data():
    """Collect data for all scenarios and traffic rates"""
    data = {scenario: {rate: None for rate in traffic_rates} for scenario in scenarios}
    
    for scenario in scenarios:
        for rate in traffic_rates:
            data[scenario][rate] = load_and_aggregate_data(scenario, rate)
    
    return data

def plot_metric(data, metric_key, ylabel, title, filename):
    """Create bar chart for a specific metric"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(traffic_rates))
    width = 0.25  # Width of bars
    
    # Plot bars for each scenario
    for i, scenario in enumerate(scenarios):
        values = []
        for rate in traffic_rates:
            if data[scenario][rate] is not None:
                values.append(data[scenario][rate][metric_key])
            else:
                values.append(0)  # Use 0 if data is missing
        
        offset = width * (i - 1)  # Center the middle bar
        bars = ax.bar(x + offset, values, width, 
                     label=scenario_labels[scenario],
                     color=colors[scenario],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Traffic Rate (vehicles/hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(traffic_rates)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add some padding to y-axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def create_all_plots(output_dir="results/plots"):
    """Generate all comparison plots"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data
    print("Loading data from CSV files...")
    data = collect_all_data()
    
    # Define metrics to plot
    metrics = [
        {
            'key': 'total_vehicles',
            'ylabel': 'Number of Vehicles',
            'title': 'Total Number of Vehicles Processed by Traffic Rate',
            'filename': os.path.join(output_dir, 'total_vehicles.png')
        },
        {
            'key': 'avg_waiting_time',
            'ylabel': 'Average Waiting Time (seconds)',
            'title': 'Average Waiting Time by Traffic Rate',
            'filename': os.path.join(output_dir, 'avg_waiting_time.png')
        },
        {
            'key': 'avg_travel_time',
            'ylabel': 'Average Travel Time (seconds)',
            'title': 'Average Travel Time by Traffic Rate',
            'filename': os.path.join(output_dir, 'avg_travel_time.png')
        },
        {
            'key': 'avg_speed',
            'ylabel': 'Average Speed (m/s)',
            'title': 'Average Speed by Traffic Rate',
            'filename': os.path.join(output_dir, 'avg_speed.png')
        }
    ]
    
    # Create each plot
    print("\nGenerating plots...")
    for metric in metrics:
        plot_metric(data, metric['key'], metric['ylabel'], 
                   metric['title'], metric['filename'])
    
    print(f"\nAll plots saved to: {output_dir}")

def print_summary_table(data):
    """Print a summary table of all metrics"""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    for metric_name, metric_key in [
        ("Total Vehicles", "total_vehicles"),
        ("Avg Waiting Time (s)", "avg_waiting_time"),
        ("Avg Travel Time (s)", "avg_travel_time"),
        ("Avg Speed (m/s)", "avg_speed")
    ]:
        print(f"\n{metric_name}:")
        print(f"{'Traffic Rate':<15}", end="")
        for scenario in scenarios:
            print(f"{scenario_labels[scenario]:<25}", end="")
        print()
        print("-" * 90)
        
        for rate in traffic_rates:
            print(f"{rate:<15}", end="")
            for scenario in scenarios:
                if data[scenario][rate] is not None:
                    value = data[scenario][rate][metric_key]
                    print(f"{value:<25.2f}", end="")
                else:
                    print(f"{'N/A':<25}", end="")
            print()
    
    print("="*100)

if __name__ == "__main__":
    # Generate all plots
    create_all_plots()
    
    # Also print a summary table
    data = collect_all_data()
    print_summary_table(data)
