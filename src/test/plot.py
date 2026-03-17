import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
root_dir    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
results_dir = os.path.join(root_dir, "output")
output_dir  = os.path.join(root_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

print(f"Results directory: {results_dir}")
print(f"Output  directory: {output_dir}")

# ---------------------------------------------------------------------------
# Reference maps
# ---------------------------------------------------------------------------
SCENARIOS_MAP = {
    'allway_stop': 'Allway Stop',
    'fixed_tl':    'Fixed TL',
    'rbl':         'Right Before Left',
}

INTENTIONS_MAP = {
    'all_straight':      'All Straight',
    'all_left':          'All Left',
    'uniform_random':    'Uniform Random',
    'assymetric_random': 'Asymmetric Random',
}

SCENARIO_COLORS = {
    'allway_stop': '#9b59b6',
    'fixed_tl':    '#e74c3c',
    'rbl':         '#2ecc71',
}

# ---------------------------------------------------------------------------
# Metric config  —  each metric maps to three summary CSV columns (_min/_avg/_max)
# and one per-vehicle CSV column.
# ---------------------------------------------------------------------------
METRIC_CONFIG = {
    'travel_time': {
        'label':              'Travel Time (s)',
        'title':              'Travel Time',
        'filename_prefix':    'travel_time',
        # summary CSV columns (for bar / heatmap / ribbon plots)
        'col_avg':            'travel_time_avg',
        'col_min':            'travel_time_min',
        'col_max':            'travel_time_max',
        # per-vehicle CSV column (for CDF / box plots)
        'col_vehicle':        'travel_time',
    },
    'waiting_time': {
        'label':              'Waiting Time (s)',
        'title':              'Waiting Time',
        'filename_prefix':    'waiting_time',
        'col_avg':            'waiting_time_avg',
        'col_min':            'waiting_time_min',
        'col_max':            'waiting_time_max',
        'col_vehicle':        'waiting_time',
    },
}


# ---------------------------------------------------------------------------
# Filename parser  (works for both "group.csv" and "group_vehicles.csv")
# ---------------------------------------------------------------------------
def parse_filename(filename):
    basename   = os.path.basename(filename).replace('_vehicles.csv', '').replace('.csv', '')
    scenarios  = list(SCENARIOS_MAP.keys())
    intentions = list(INTENTIONS_MAP.keys())

    scenario = None
    remaining = basename
    for scen in scenarios:
        if scen in basename:
            scenario  = scen
            remaining = basename.replace(scen, '').strip('_')
            break

    intention    = None
    traffic_rate = remaining
    for intent in intentions:
        if intent in remaining:
            intention    = intent
            traffic_rate = remaining.replace(intent, '').strip('_')
            break

    return {'scenario': scenario, 'intention': intention,
            'traffic_rate': traffic_rate, 'group_name': basename}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_summary_data():
    """
    Load per-run summary CSVs (one row per run with _min/_avg/_max columns).
    Used by: bar charts, heatmaps, ribbon plot, statistics.
    """
    # Exclude _vehicles.csv files
    csv_files = [f for f in glob(os.path.join(results_dir, "*.csv"))
                 if not f.endswith("_vehicles.csv")]
    print(f"Found {len(csv_files)} summary CSV files")
    if not csv_files:
        print("No summary CSV files found!")
        return None

    all_data = []
    for csv_file in csv_files:
        try:
            meta = parse_filename(csv_file)
            df   = pd.read_csv(csv_file)
            for col in ['travel_time_min', 'travel_time_avg', 'travel_time_max',
                        'waiting_time_min', 'waiting_time_avg', 'waiting_time_max',
                        'n_vehicles']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['scenario']     = meta['scenario']
            df['intention']    = meta['intention']
            df['traffic_rate'] = meta['traffic_rate']
            df['group_name']   = meta['group_name']
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} runs across {len(csv_files)} groups (summary)")
    return combined


def load_vehicle_data():
    """
    Load per-vehicle CSVs (*_vehicles.csv).  One row per completed vehicle trip.
    Columns: run, vehicle_id, travel_time, waiting_time, time_loss,
             depart, arrival, route_length  + scenario / intention / traffic_rate.
    Used by: CDF plots, box plots.
    """
    vehicle_files = glob(os.path.join(results_dir, "*_vehicles.csv"))
    print(f"Found {len(vehicle_files)} vehicle CSV files")
    if not vehicle_files:
        print("No per-vehicle CSV files found!  "
              "Re-run run_simulation.py to generate them.")
        return None

    all_data = []
    for vfile in vehicle_files:
        try:
            meta = parse_filename(vfile)
            df   = pd.read_csv(vfile)
            for col in ['travel_time', 'waiting_time', 'time_loss',
                        'depart', 'arrival', 'route_length']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['scenario']     = meta['scenario']
            df['intention']    = meta['intention']
            df['traffic_rate'] = meta['traffic_rate']
            df['group_name']   = meta['group_name']
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {vfile}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined):,} vehicle trips across "
          f"{len(vehicle_files)} groups (per-vehicle)")
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_summary(df, metric):
    return df.dropna(subset=[METRIC_CONFIG[metric]['col_avg']])

def _clean_vehicle(df, metric):
    return df.dropna(subset=[METRIC_CONFIG[metric]['col_vehicle']])

def _save(fig, filename):
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PLOT 1 — Bar chart with min–max error bars  (uses summary data)
# ---------------------------------------------------------------------------
def plot_intention_scenario_bars(df, metric):
    cfg      = METRIC_CONFIG[metric]
    col_avg, col_min, col_max = cfg['col_avg'], cfg['col_min'], cfg['col_max']
    df_clean  = _clean_summary(df, metric)
    intentions = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    scenario_order = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, intention in enumerate(intentions):
        if idx >= len(axes): break
        ax = axes[idx]
        grp = (df_clean[df_clean['intention'] == intention]
               .groupby(['traffic_rate', 'scenario'])
               .agg(bar=(col_avg, 'mean'), lo=(col_min, 'min'), hi=(col_max, 'max'))
               .reset_index())

        pivot_bar = grp.pivot(index='traffic_rate', columns='scenario', values='bar')
        pivot_lo  = grp.pivot(index='traffic_rate', columns='scenario', values='lo')
        pivot_hi  = grp.pivot(index='traffic_rate', columns='scenario', values='hi')

        for p in (pivot_bar, pivot_lo, pivot_hi):
            p = p.reindex(columns=[s for s in scenario_order if s in p.columns])

        x     = np.arange(len(pivot_bar.index))
        width = 0.25
        n_cols = len([s for s in scenario_order if s in pivot_bar.columns])

        for i, col in enumerate([s for s in scenario_order if s in pivot_bar.columns]):
            offset  = width * (i - (n_cols - 1) / 2)
            heights = pivot_bar[col].fillna(0).values
            lo      = np.where(np.isnan(pivot_lo[col].values), 0,
                               heights - pivot_lo[col].values)
            hi      = np.where(np.isnan(pivot_hi[col].values), 0,
                               pivot_hi[col].values - heights)
            ax.bar(x + offset, heights, width,
                   label=SCENARIOS_MAP.get(col, col),
                   color=SCENARIO_COLORS.get(col, '#333'),
                   yerr=[lo, hi], capsize=3,
                   error_kw=dict(elinewidth=1, alpha=0.6))

        ax.set_xlabel('Traffic Scenario', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Avg {cfg["label"]}', fontsize=10, fontweight='bold')
        ax.set_title(INTENTIONS_MAP.get(intention, intention), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_bar.index, rotation=45, ha='right', fontsize=8)
        ax.legend(title='Control Type', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(len(intentions), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{cfg["title"]} — avg ± [min, max] per run\n(Grouped by Intention)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"{cfg['filename_prefix']}_1A_intention_scenario_bars.png")


# ---------------------------------------------------------------------------
# PLOT 2 — Heatmap per scenario  (uses summary data)
# ---------------------------------------------------------------------------
def plot_control_type_heatmaps(df, metric):
    cfg      = METRIC_CONFIG[metric]
    col_avg  = cfg['col_avg']
    df_clean  = _clean_summary(df, metric)
    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6))
    if len(scenarios) == 1:
        axes = [axes]

    vmin = df_clean[col_avg].min()
    vmax = df_clean.groupby(['intention', 'traffic_rate'])[col_avg].mean().max()

    for idx, scenario in enumerate(scenarios):
        ax    = axes[idx]
        pivot = (df_clean[df_clean['scenario'] == scenario]
                 .groupby(['intention', 'traffic_rate'])[col_avg].mean()
                 .unstack())
        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]
        try:
            pivot = pivot[sorted(pivot.columns,
                                 key=lambda x: float(x) if str(x).replace('.','').isdigit() else x)]
        except Exception:
            pass
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': cfg['label']}, linewidths=0.5,
                    vmin=vmin, vmax=vmax)
        ax.set_title(SCENARIOS_MAP.get(scenario, scenario), fontsize=13, fontweight='bold')
        ax.set_xlabel('Traffic Scenario', fontsize=11, fontweight='bold')
        ax.set_ylabel('Intention Type' if idx == 0 else '', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0,  labelsize=10)

    fig.suptitle(f'Heatmap: avg {cfg["title"]} by Intention & Traffic Rate',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, f"{cfg['filename_prefix']}_1B_intention_scenario_heatmaps.png")


# ---------------------------------------------------------------------------
# PLOT 3 — Summary heatmaps  (uses summary data)
# ---------------------------------------------------------------------------
def plot_heatmap(df, metric):
    cfg     = METRIC_CONFIG[metric]
    col_avg = cfg['col_avg']
    df_clean = _clean_summary(df, metric)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pivot1         = df_clean.groupby(['intention', 'scenario'])[col_avg].mean().unstack()
    pivot1.index   = [INTENTIONS_MAP.get(i, i) for i in pivot1.index]
    pivot1.columns = [SCENARIOS_MAP.get(c, c) for c in pivot1.columns]
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': cfg['label']}, linewidths=0.5)
    axes[0].set_title(f'{cfg["title"]} — Intention vs Control Type', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Control Type', fontsize=11)
    axes[0].set_ylabel('Intention',    fontsize=11)

    df_temp = df_clean.copy()
    df_temp['scenario_traffic'] = df_temp['scenario'] + ' | ' + df_temp['traffic_rate'].astype(str)
    pivot2       = df_temp.groupby(['intention', 'scenario_traffic'])[col_avg].mean().unstack()
    pivot2.index = [INTENTIONS_MAP.get(i, i) for i in pivot2.index]
    sns.heatmap(pivot2, annot=False, cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': cfg['label']})
    axes[1].set_title(f'{cfg["title"]} — Detailed (Control + Traffic Rate)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Control Type + Traffic Rate', fontsize=11)
    axes[1].set_ylabel('Intention', fontsize=11)
    axes[1].tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()
    _save(fig, f"{cfg['filename_prefix']}_3_summary_heatmap.png")


# ---------------------------------------------------------------------------
# PLOT 4 — Box plots + CDF per scenario  (CDF now uses per-vehicle data)
#
#   Box plots: one box per scenario, each data point = one vehicle's value
#              (pooled across all runs / traffic rates / intentions for
#               that scenario).  This gives a true distributional picture.
#   CDF:       empirical CDF over all individual vehicle trips per scenario.
# ---------------------------------------------------------------------------
def plot_scenario_boxplots_and_cdf(df_vehicle, metric):
    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)
    scenarios   = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]
    n_scen      = len(scenarios)

    fig = plt.figure(figsize=(max(4 * n_scen, 12), 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
    fig.add_subplot(gs[0]).axis('off')

    box_axes = [fig.add_axes([0.1 + i*(0.8/n_scen), 0.55, 0.8/n_scen*0.9, 0.35])
                for i in range(n_scen)]

    for idx, scenario in enumerate(scenarios):
        data     = df_clean[df_clean['scenario'] == scenario][col_vehicle].dropna()
        color    = SCENARIO_COLORS.get(scenario, '#95a5a6')
        mean_val = data.mean()
        ax       = box_axes[idx]
        ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markersize=2, alpha=0.3))
        ax.plot(1, mean_val, marker='*', color='red', markersize=15,
                markeredgecolor='darkred', markeredgewidth=1.5, zorder=3,
                label=f'Mean: {mean_val:.1f}s')
        n_veh = len(data)
        ax.set_title(f'{SCENARIOS_MAP.get(scenario, scenario)}\n(n={n_veh:,} vehicles)',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(cfg['label'], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels([''])
        ax.legend(loc='upper right', fontsize=8)

    # --- CDF — empirical, one curve per scenario, one point per vehicle ---
    ax_cdf = fig.add_subplot(gs[1])
    for scenario in scenarios:
        data = np.sort(
            df_clean[df_clean['scenario'] == scenario][col_vehicle].dropna().values
        )
        cdf  = np.arange(1, len(data) + 1) / len(data)
        ax_cdf.plot(data, cdf, linewidth=2.5,
                    color=SCENARIO_COLORS.get(scenario, '#95a5a6'),
                    label=f"{SCENARIOS_MAP.get(scenario, scenario)} (n={len(data):,})")

    ax_cdf.set_xlabel(cfg['label'], fontsize=12)
    ax_cdf.set_ylabel('CDF', fontsize=12)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_title(f'Empirical CDF — {cfg["title"]} per Vehicle (All Scenarios)',
                     fontsize=12, fontweight='bold')
    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)

    n_runs = df_clean['run'].nunique() if 'run' in df_clean.columns else '?'
    fig.suptitle(
        f'{cfg["title"]} Distribution by Controller Type\n'
        f'(each data point = one individual vehicle trip; '
        f'pooled across all runs / traffic rates)',
        fontsize=14, fontweight='bold', y=0.98
    )
    _save(fig, f"{cfg['filename_prefix']}_4_scenario_boxplot_cdf.png")


# ---------------------------------------------------------------------------
# PLOT 5 — Box plots + CDF per intention  (CDF now uses per-vehicle data)
# ---------------------------------------------------------------------------
def plot_intention_boxplots_and_cdf(df_vehicle, metric):
    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)
    intentions  = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    n_intents   = len(intentions)
    colors      = sns.color_palette("Set2", n_intents)

    fig = plt.figure(figsize=(max(4 * n_intents, 12), 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
    fig.add_subplot(gs[0]).axis('off')

    box_axes = [fig.add_axes([0.1 + i*(0.8/n_intents), 0.55, 0.8/n_intents*0.9, 0.35])
                for i in range(n_intents)]

    for idx, intention in enumerate(intentions):
        data     = df_clean[df_clean['intention'] == intention][col_vehicle].dropna()
        mean_val = data.mean()
        ax       = box_axes[idx]
        ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                   boxprops=dict(facecolor=colors[idx], alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markersize=2, alpha=0.3))
        ax.plot(1, mean_val, marker='*', color='orange', markersize=15,
                markeredgecolor='darkorange', markeredgewidth=1.5, zorder=3,
                label=f'Mean: {mean_val:.1f}s')
        n_veh = len(data)
        ax.set_title(f'{INTENTIONS_MAP.get(intention, intention)}\n(n={n_veh:,} vehicles)',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(cfg['label'], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels([''])
        ax.legend(loc='upper right', fontsize=8)

    # --- CDF — empirical, one curve per intention, one point per vehicle ---
    ax_cdf = fig.add_subplot(gs[1])
    for idx, intention in enumerate(intentions):
        data = np.sort(
            df_clean[df_clean['intention'] == intention][col_vehicle].dropna().values
        )
        cdf  = np.arange(1, len(data) + 1) / len(data)
        ax_cdf.plot(data, cdf, linewidth=2.5, color=colors[idx],
                    label=f"{INTENTIONS_MAP.get(intention, intention)} (n={len(data):,})")

    ax_cdf.set_xlabel(cfg['label'], fontsize=12)
    ax_cdf.set_ylabel('CDF', fontsize=12)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_title(f'Empirical CDF — {cfg["title"]} per Vehicle',
                     fontsize=12, fontweight='bold')
    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)

    fig.suptitle(
        f'{cfg["title"]} Distribution by Intention\n'
        f'(each data point = one individual vehicle trip; '
        f'pooled across all runs / traffic rates)',
        fontsize=14, fontweight='bold', y=0.98
    )
    _save(fig, f"{cfg['filename_prefix']}_5_intention_boxplot_cdf.png")


# ---------------------------------------------------------------------------
# PLOT 6 — Min / Avg / Max ribbon per traffic rate and scenario
#          (uses summary data — ribbon is across runs, not vehicles)
# ---------------------------------------------------------------------------
def plot_min_avg_max_ribbon(df, metric):
    cfg                      = METRIC_CONFIG[metric]
    col_avg, col_min, col_max = cfg['col_avg'], cfg['col_min'], cfg['col_max']
    df_clean  = _clean_summary(df, metric)
    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        grp = (df_clean[df_clean['scenario'] == scenario]
               .groupby('traffic_rate')
               .agg(avg=(col_avg, 'mean'), lo=(col_min, 'min'), hi=(col_max, 'max'))
               .reset_index()
               .sort_values('traffic_rate'))
        x     = np.arange(len(grp))
        color = SCENARIO_COLORS.get(scenario, '#555')
        ax.fill_between(x, grp['lo'], grp['hi'], alpha=0.25, color=color, label='min–max range')
        ax.plot(x, grp['avg'], marker='o', linewidth=2,   color=color, label='avg of runs')
        ax.plot(x, grp['lo'],  marker='v', linewidth=1, linestyle='--', color=color, alpha=0.6)
        ax.plot(x, grp['hi'],  marker='^', linewidth=1, linestyle='--', color=color, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(grp['traffic_rate'], rotation=45, ha='right', fontsize=8)
        ax.set_title(SCENARIOS_MAP.get(scenario, scenario), fontsize=12, fontweight='bold')
        ax.set_xlabel('Traffic Rate', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(cfg['label'], fontsize=11)
    fig.suptitle(f'{cfg["title"]} — Min / Avg / Max per Traffic Rate & Scenario',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, f"{cfg['filename_prefix']}_6_min_avg_max_ribbon.png")


# ---------------------------------------------------------------------------
# Statistics summary  (uses summary data)
# ---------------------------------------------------------------------------
def generate_statistics(df, metric):
    cfg                      = METRIC_CONFIG[metric]
    col_avg, col_min, col_max = cfg['col_avg'], cfg['col_min'], cfg['col_max']
    df_clean = _clean_summary(df, metric)

    def summarise(groupby_cols):
        return df_clean.groupby(groupby_cols).agg(
            runs=(col_avg, 'count'),
            avg= (col_avg, 'mean'),
            std= (col_avg, 'std'),
            min= (col_min, 'min'),
            max= (col_max, 'max'),
        )

    stats = {
        'overall':        df_clean[[col_avg, col_min, col_max]].describe(),
        'by_intention':   summarise('intention'),
        'by_scenario':    summarise('scenario'),
        'by_combination': summarise(['scenario', 'intention']),
    }

    txt_path = os.path.join(output_dir, f"{cfg['filename_prefix']}_statistics_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"SIMULATION STATISTICS: {cfg['title'].upper()}\n")
        f.write("(Derived from per-run min / avg / max columns)\n")
        f.write("=" * 60 + "\n\n")
        for title, table in [
            ("OVERALL",                          stats['overall']),
            ("BY INTENTION",                     stats['by_intention']),
            ("BY SCENARIO (CONTROL TYPE)",       stats['by_scenario']),
            ("BY COMBINATION (Scenario+Intent)", stats['by_combination']),
        ]:
            f.write(f"{title}\n{'-'*60}\n{table.to_string()}\n\n{'='*60}\n\n")

    csv_path = os.path.join(output_dir, f"{cfg['filename_prefix']}_summary_by_scenario_intention.csv")
    stats['by_combination'].reset_index().to_csv(csv_path, index=False)
    print(f"Saved: {os.path.basename(txt_path)} + {os.path.basename(csv_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Simulation Analysis — min/avg/max summary + per-vehicle format")
    print("=" * 60)

    # Load both datasets independently
    df_summary = load_summary_data()
    df_vehicle = load_vehicle_data()

    if df_summary is None and df_vehicle is None:
        print("No data found in either summary or vehicle CSVs.  Exiting.")
        return

    if df_summary is not None:
        print(f"\nSummary data:")
        print(f"  Total runs : {len(df_summary)}")
        print(f"  Scenarios  : {df_summary['scenario'].dropna().unique()}")
        print(f"  Intentions : {df_summary['intention'].dropna().unique()}")
        print(f"  Rate groups: {df_summary['traffic_rate'].nunique()}")

    if df_vehicle is not None:
        print(f"\nPer-vehicle data:")
        print(f"  Total trips: {len(df_vehicle):,}")
        print(f"  Scenarios  : {df_vehicle['scenario'].dropna().unique()}")
        print(f"  Intentions : {df_vehicle['intention'].dropna().unique()}")

    for metric in ['travel_time', 'waiting_time']:
        cfg = METRIC_CONFIG[metric]
        print(f"\n{'='*60}")
        print(f"Generating plots for: {cfg['title']} ...")

        # Plots that need summary data
        if df_summary is not None:
            missing = [cfg[c] for c in ('col_avg', 'col_min', 'col_max')
                       if cfg[c] not in df_summary.columns]
            if missing:
                print(f"  [SKIP] Summary columns not found: {missing}")
            else:
                plot_intention_scenario_bars(df_summary, metric)
                plot_control_type_heatmaps(df_summary, metric)
                plot_heatmap(df_summary, metric)
                plot_min_avg_max_ribbon(df_summary, metric)
                print(f"Generating statistics for: {cfg['title']} ...")
                generate_statistics(df_summary, metric)

        # Plots that need per-vehicle data (CDF + box)
        if df_vehicle is not None:
            col_v = cfg['col_vehicle']
            if col_v not in df_vehicle.columns:
                print(f"  [SKIP] Vehicle column '{col_v}' not found in vehicle CSVs")
            else:
                plot_scenario_boxplots_and_cdf(df_vehicle, metric)
                plot_intention_boxplots_and_cdf(df_vehicle, metric)
        else:
            print(f"  [SKIP] No per-vehicle data — box/CDF plots skipped.\n"
                  f"         Run run_simulation.py to generate *_vehicles.csv files.")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
