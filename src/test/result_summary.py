import pandas as pd
import os
from datetime import datetime

def process_emission_data(emission_csv_path, output_csv_path=None):
    """
    Process SUMO emission data to extract vehicle summary statistics.
    Optimized with vectorized pandas operations.
    """
    if not os.path.exists(emission_csv_path):
        print(f"Error: File {emission_csv_path} not found.")
        return None

    print(f"Reading emission data from {emission_csv_path}...")
    df = pd.read_csv(emission_csv_path)
    
    # Vectorized calculation: Group by vehicle 'id' and aggregate
    summary_df = df.sort_values('time').groupby('id').agg(
        entry_time=('time', 'min'),
        exit_time=('time', 'max'),
        total_waiting_time=('waiting_time', 'max'),
        avg_speed=('speed', 'mean'),
        max_speed=('speed', 'max'),
        entry_edge=('edge_id', 'first'),
        exit_edge=('edge_id', 'last')
    ).reset_index()

    # Calculate travel time
    summary_df['travel_time'] = summary_df['exit_time'] - summary_df['entry_time']
    summary_df.rename(columns={'id': 'vehicle_id'}, inplace=True)

    # Print overall statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total vehicles: {len(summary_df)}")
    print(f"Simulation duration: {df['time'].min():.1f}s to {df['time'].max():.1f}s")
    print(f"Average travel time: {summary_df['travel_time'].mean():.2f}s")
    print(f"Average waiting time: {summary_df['total_waiting_time'].mean():.2f}s")
    print("="*60)
    
    # Handle Output Path
    if output_csv_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(root_dir, "results/processed")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = f'{results_dir}/summary_{timestamp}.csv'
    
    summary_df.to_csv(output_csv_path, index=False)
    print(f"\nSummary saved to: {output_csv_path}")
    
    return summary_df

