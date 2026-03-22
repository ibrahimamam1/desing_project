import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
root_dir    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
results_dir = os.path.join(root_dir, "telemetry")
output_dir  = os.path.join(root_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

print(f"Results directory: {results_dir}")
print(f"Output  directory: {output_dir}")

# ---------------------------------------------------------------------------
# Reference maps
# ---------------------------------------------------------------------------
SCENARIOS_MAP = {
    # 'fixed_tl':    'Fixed TL',
    'fixed_tl_30s_20s':    'Fixed TL 30s-20s',
    'fixed_tl_25s_15s':    'Fixed TL 25s-15s',
    'fixed_tl_15s_10s':    'Fixed TL 15s-10s',
    'rbl':         'Right Before Left',
    'fcfs': 'First Come First Serve',
}

INTENTIONS_MAP = {
    'all_straight':      'All Straight',
    'all_left':          'All Left',
    'uniform_random':    'Uniform Random',
    'assymetric_random': 'Asymmetric Random',
}

SCENARIO_COLORS = {

    'fixed_tl_30s_20s': '#cc9d2e',   # orange
    'fixed_tl_25s_15s': '#cc2e3e',   # red
    'fixed_tl_15s_10s': '#ffe32a',   # yellow
    'rbl': '#6bcc2e',                # green
    'fcfs': '#9d2ecc'                # purple
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
# Derive summary statistics from per-vehicle data
# ---------------------------------------------------------------------------
def derive_summary_from_vehicle(df_vehicle):
    """
    Build a run-level summary DataFrame from per-vehicle data, computing
    _min / _avg / _max for travel_time and waiting_time per (run, group).
    This replaces the pre-summarised summary CSV for all bar/heatmap/ribbon plots.
    """
    group_cols = ['run', 'scenario', 'intention', 'traffic_rate', 'group_name']
    # Only keep group cols that actually exist in the frame
    group_cols = [c for c in group_cols if c in df_vehicle.columns]

    agg_dict = {}
    for metric, cfg in METRIC_CONFIG.items():
        col = cfg['col_vehicle']
        if col in df_vehicle.columns:
            agg_dict[cfg['col_avg']] = (col, 'mean')
            agg_dict[cfg['col_min']] = (col, 'min')
            agg_dict[cfg['col_max']] = (col, 'max')

    if not agg_dict:
        return None

    # n_vehicles per run
    if 'vehicle_id' in df_vehicle.columns:
        agg_dict['n_vehicles'] = ('vehicle_id', 'count')
    else:
        first_metric_col = list(METRIC_CONFIG.values())[0]['col_vehicle']
        agg_dict['n_vehicles'] = (first_metric_col, 'count')

    df_summary = (
        df_vehicle
        .groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    print(f"Derived summary: {len(df_summary)} run-level rows from vehicle data")
    return df_summary


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

    cfg = METRIC_CONFIG[metric]
    col_avg, col_min, col_max = cfg['col_avg'], cfg['col_min'], cfg['col_max']

    df_clean = _clean_summary(df, metric)

    intentions = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    scenario_order = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, intention in enumerate(intentions):

        if idx >= len(axes):
            break

        ax = axes[idx]

        grp = (
            df_clean[df_clean['intention'] == intention]
            .groupby(['traffic_rate', 'scenario'])
            .agg(
                bar=(col_avg, 'mean'),
                lo=(col_min, 'min'),
                hi=(col_max, 'max')
            )
            .reset_index()
        )

        pivot_bar = grp.pivot(index='traffic_rate', columns='scenario', values='bar')
        pivot_lo  = grp.pivot(index='traffic_rate', columns='scenario', values='lo')
        pivot_hi  = grp.pivot(index='traffic_rate', columns='scenario', values='hi')

        # keep scenario order consistent
        pivot_bar = pivot_bar.reindex(columns=[s for s in scenario_order if s in pivot_bar.columns])
        pivot_lo  = pivot_lo.reindex(columns=[s for s in scenario_order if s in pivot_lo.columns])
        pivot_hi  = pivot_hi.reindex(columns=[s for s in scenario_order if s in pivot_hi.columns])

        x = np.arange(len(pivot_bar.index))

        # -----------------------------
        # Dynamic bar width adjustment
        # -----------------------------
        n_cols = len(pivot_bar.columns)   # number of control types
        width = 0.8 / n_cols              # keep bars within 80% of group width

        for i, col in enumerate(pivot_bar.columns):

            offset = width * (i - (n_cols - 1) / 2)

            heights = pivot_bar[col].fillna(0).values

            lo = np.where(
                np.isnan(pivot_lo[col].values),
                0,
                heights - pivot_lo[col].values
            )

            hi = np.where(
                np.isnan(pivot_hi[col].values),
                0,
                pivot_hi[col].values - heights
            )

            ax.bar(
                x + offset,
                heights,
                width,
                label=SCENARIOS_MAP.get(col, col),
                color=SCENARIO_COLORS.get(col, '#333'),
                yerr=[lo, hi],
                capsize=3,
                error_kw=dict(elinewidth=1, alpha=0.6)
            )

        ax.set_xlabel('Traffic Scenario', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Avg {cfg["label"]}', fontsize=10, fontweight='bold')
        ax.set_title(
            INTENTIONS_MAP.get(intention, intention),
            fontsize=12,
            fontweight='bold'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(pivot_bar.index, rotation=45, ha='right', fontsize=8)

        ax.legend(title='Control Type', fontsize=8)

        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplot panels
    for idx in range(len(intentions), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{cfg["title"]} — avg ± [min, max] per Traffic Rate\n(Grouped by Intention)',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save(fig, f"{cfg['filename_prefix']}_1A_intention_scenario_bars.png")


# ---------------------------------------------------------------------------
# PLOT 2A-Avg — Heatmap per scenario  (uses summary data)
# All heatmaps share the same color scale
# ---------------------------------------------------------------------------
def plot_control_type_heatmaps_avg(df, metric):

    cfg      = METRIC_CONFIG[metric]
    col_avg  = cfg['col_avg']

    df_clean  = _clean_summary(df, metric)

    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6))

    if len(scenarios) == 1:
        axes = [axes]

    # ------------------------------------------------
    # GLOBAL color scale (shared across all heatmaps)
    # ------------------------------------------------
    vmin = df_clean[col_avg].min()
    vmax = df_clean[col_avg].max()

    for idx, scenario in enumerate(scenarios):

        ax = axes[idx]

        pivot = (
            df_clean[df_clean['scenario'] == scenario]
            .groupby(['intention', 'traffic_rate'])[col_avg]
            .mean()
            .unstack()
        )

        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]

        try:
            pivot = pivot[
                sorted(
                    pivot.columns,
                    key=lambda x: float(x)
                    if str(x).replace('.', '').isdigit()
                    else x
                )
            ]
        except Exception:
            pass

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            ax=ax,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
            cbar=(idx == len(scenarios) - 1),   # only show one colorbar
            cbar_kws={'label': cfg['label']}
        )

        ax.set_title(
            SCENARIOS_MAP.get(scenario, scenario),
            fontsize=13,
            fontweight='bold'
        )

        ax.set_xlabel('Traffic Scenario', fontsize=11, fontweight='bold')

        ax.set_ylabel(
            'Intention Type' if idx == 0 else '',
            fontsize=11,
            fontweight='bold'
        )

        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

    fig.suptitle(
        f'Heatmap: avg {cfg["title"]} by Intention & Traffic Rate',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_2A-Avg_intention_scenario_heatmaps.png")

# ---------------------------------------------------------------------------
# PLOT 2B-Max — Heatmap per scenario (uses summary data)
# ---------------------------------------------------------------------------
def plot_control_type_heatmaps_max(df, metric):

    cfg      = METRIC_CONFIG[metric]
    col_max  = cfg['col_max']

    df_clean = df.dropna(subset=[col_max])

    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6))

    if len(scenarios) == 1:
        axes = [axes]

    vmin = df_clean[col_max].min()
    vmax = df_clean[col_max].max()

    for idx, scenario in enumerate(scenarios):

        ax = axes[idx]

        pivot = (
            df_clean[df_clean['scenario'] == scenario]
            .groupby(['intention', 'traffic_rate'])[col_max]
            .max()
            .unstack()
        )

        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]

        try:
            pivot = pivot[
                sorted(
                    pivot.columns,
                    key=lambda x: float(x)
                    if str(x).replace('.', '').isdigit()
                    else x
                )
            ]
        except Exception:
            pass

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            ax=ax,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
            cbar=(idx == len(scenarios) - 1),
            cbar_kws={'label': cfg['label']}
        )

        ax.set_title(
            f"{SCENARIOS_MAP.get(scenario, scenario)} (Max)",
            fontsize=13,
            fontweight='bold'
        )

        ax.set_xlabel('Traffic Scenario', fontsize=11, fontweight='bold')

        ax.set_ylabel(
            'Intention Type' if idx == 0 else '',
            fontsize=11,
            fontweight='bold'
        )

        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

    fig.suptitle(
        f'Heatmap: max {cfg["title"]} by Intention & Traffic Rate',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_2B-Max_intention_scenario_heatmaps_max.png")

# ---------------------------------------------------------------------------
# PLOT 2C-STD — Heatmap per scenario (STD across runs)
# ---------------------------------------------------------------------------
def plot_control_type_heatmaps_std(df, metric):

    cfg      = METRIC_CONFIG[metric]
    col_avg  = cfg['col_avg']

    df_clean = _clean_summary(df, metric)

    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6))

    if len(scenarios) == 1:
        axes = [axes]

    vmax = df_clean.groupby(['scenario','intention','traffic_rate'])[col_avg].std().max()
    vmin = 0

    for idx, scenario in enumerate(scenarios):

        ax = axes[idx]

        pivot = (
            df_clean[df_clean['scenario'] == scenario]
            .groupby(['intention', 'traffic_rate'])[col_avg]
            .std()
            .unstack()
        )

        pivot.index = [INTENTIONS_MAP.get(i, i) for i in pivot.index]

        try:
            pivot = pivot[
                sorted(
                    pivot.columns,
                    key=lambda x: float(x)
                    if str(x).replace('.', '').isdigit()
                    else x
                )
            ]
        except Exception:
            pass

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            ax=ax,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
            cbar=(idx == len(scenarios) - 1),
            cbar_kws={'label': f"Std {cfg['label']}"}
        )

        ax.set_title(
            f"{SCENARIOS_MAP.get(scenario, scenario)} (Std)",
            fontsize=13,
            fontweight='bold'
        )

        ax.set_xlabel('Traffic Scenario', fontsize=11, fontweight='bold')

        ax.set_ylabel(
            'Intention Type' if idx == 0 else '',
            fontsize=11,
            fontweight='bold'
        )

        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

    fig.suptitle(
        f'Heatmap: std {cfg["title"]} by Intention & Traffic Rate',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_2C-STD_intention_scenario_heatmaps_std.png")



# ---------------------------------------------------------------------------
# PLOT 3A-Avg— Summary heatmaps  (uses summary data)
# All heatmaps share the same dynamic color scale
# ---------------------------------------------------------------------------
def plot_heatmap_avg(df, metric):

    cfg     = METRIC_CONFIG[metric]
    col_avg = cfg['col_avg']

    df_clean = _clean_summary(df, metric).copy()

    # ------------------------------------------------
    # Enforce desired scenario ordering (same as plot_heatmap_max)
    # ------------------------------------------------
    scenario_order = list(SCENARIOS_MAP.keys())

    df_clean['scenario'] = pd.Categorical(
        df_clean['scenario'],
        categories=scenario_order,
        ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------------------------------------
    # Global dynamic scaling (shared color scale)
    # ------------------------------------------------
    vmin = df_clean[col_avg].min()
    vmax = df_clean[col_avg].max()

    # ------------------------------------------------
    # Heatmap 1 — Intention vs Control Type
    # ------------------------------------------------
    pivot1 = (
        df_clean
        .groupby(['intention', 'scenario'])[col_avg]
        .mean()
        .unstack()
    )

    pivot1.index   = [INTENTIONS_MAP.get(i, i) for i in pivot1.index]
    pivot1.columns = [SCENARIOS_MAP.get(c, c) for c in pivot1.columns]

    sns.heatmap(
        pivot1,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        ax=axes[0],
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        cbar=False
    )

    axes[0].set_title(
        f'Avg {cfg["title"]} — Intention vs Control Type',
        fontsize=12,
        fontweight='bold'
    )

    axes[0].set_xlabel('Control Type', fontsize=11)
    axes[0].set_ylabel('Intention', fontsize=11)

    # ------------------------------------------------
    # Heatmap 2 — Detailed (Control + Traffic Rate)
    # ------------------------------------------------
    df_temp = df_clean.copy()

    df_temp['scenario_traffic'] = (
        df_temp['scenario'].astype(str) + ' | ' + df_temp['traffic_rate'].astype(str)
    )

    pivot2 = (
        df_temp
        .groupby(['intention', 'scenario_traffic'])[col_avg]
        .mean()
        .unstack()
    )

    pivot2.index = [INTENTIONS_MAP.get(i, i) for i in pivot2.index]

    # ------------------------------------------------
    # Sort detailed columns by scenario order then traffic rate
    # ------------------------------------------------
    ordered_cols = []

    for scen in scenario_order:
        for col in pivot2.columns:
            if col.startswith(scen + " |"):
                ordered_cols.append(col)

    pivot2 = pivot2[ordered_cols]

    sns.heatmap(
        pivot2,
        annot=False,
        cmap='YlOrRd',
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cfg['label']}
    )

    axes[1].set_title(
        f'Avg {cfg["title"]} — Detailed (Control + Traffic Rate)',
        fontsize=12,
        fontweight='bold'
    )

    axes[1].set_xlabel('Control Type + Traffic Rate', fontsize=11)
    axes[1].set_ylabel('Intention', fontsize=11)

    axes[1].tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3A_Avg_summary_heatmap.png")

# ---------------------------------------------------------------------------
# PLOT 3B-Max— Summary heatmaps (MAX values instead of AVG)
# All heatmaps share the same dynamic color scale
# ---------------------------------------------------------------------------
def plot_heatmap_max(df, metric):

    cfg     = METRIC_CONFIG[metric]
    col_max = cfg['col_max']

    df_clean = df.dropna(subset=[col_max]).copy()

    # ------------------------------------------------
    # Enforce desired scenario ordering
    # ------------------------------------------------
    scenario_order = list(SCENARIOS_MAP.keys())

    df_clean['scenario'] = pd.Categorical(
        df_clean['scenario'],
        categories=scenario_order,
        ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------------------------------------
    # Global dynamic scaling
    # ------------------------------------------------
    vmin = df_clean[col_max].min()
    vmax = df_clean[col_max].max()

    # ------------------------------------------------
    # Heatmap 1 — Intention vs Control Type
    # ------------------------------------------------
    pivot1 = (
        df_clean
        .groupby(['intention', 'scenario'])[col_max]
        .max()
        .unstack()
    )

    pivot1.index   = [INTENTIONS_MAP.get(i, i) for i in pivot1.index]
    pivot1.columns = [SCENARIOS_MAP.get(c, c) for c in pivot1.columns]

    sns.heatmap(
        pivot1,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        ax=axes[0],
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        cbar=False
    )

    axes[0].set_title(
        f'Max {cfg["title"]} — Intention vs Control Type',
        fontsize=12,
        fontweight='bold'
    )

    axes[0].set_xlabel('Control Type', fontsize=11)
    axes[0].set_ylabel('Intention', fontsize=11)

    # ------------------------------------------------
    # Heatmap 2 — Detailed (Control + Traffic Rate)
    # ------------------------------------------------
    df_temp = df_clean.copy()

    df_temp['scenario_traffic'] = (
        df_temp['scenario'].astype(str) + ' | ' + df_temp['traffic_rate'].astype(str)
    )

    pivot2 = (
        df_temp
        .groupby(['intention', 'scenario_traffic'])[col_max]
        .max()
        .unstack()
    )

    pivot2.index = [INTENTIONS_MAP.get(i, i) for i in pivot2.index]

    # ------------------------------------------------
    # Sort detailed columns by scenario order then traffic rate
    # ------------------------------------------------
    ordered_cols = []

    for scen in scenario_order:
        for col in pivot2.columns:
            if col.startswith(scen + " |"):
                ordered_cols.append(col)

    pivot2 = pivot2[ordered_cols]

    sns.heatmap(
        pivot2,
        annot=False,
        cmap='YlOrRd',
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cfg['label']}
    )

    axes[1].set_title(
        f'Max {cfg["title"]} — Detailed (Control + Traffic Rate)',
        fontsize=12,
        fontweight='bold'
    )

    axes[1].set_xlabel('Control Type + Traffic Rate', fontsize=11)
    axes[1].set_ylabel('Intention', fontsize=11)

    axes[1].tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3B_Max_summary_heatmap.png")

# ---------------------------------------------------------------------------
# PLOT 3C-STD — Summary heatmaps (STD across runs)
# ---------------------------------------------------------------------------
def plot_heatmap_std(df, metric):

    cfg     = METRIC_CONFIG[metric]
    col_avg = cfg['col_avg']   # STD computed from avg values across runs

    df_clean = _clean_summary(df, metric).copy()

    # ------------------------------------------------
    # Enforce desired scenario ordering
    # ------------------------------------------------
    scenario_order = list(SCENARIOS_MAP.keys())

    df_clean['scenario'] = pd.Categorical(
        df_clean['scenario'],
        categories=scenario_order,
        ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------------------------------------
    # Global dynamic scaling
    # ------------------------------------------------
    vmin = 0
    vmax = df_clean.groupby(['intention','scenario'])[col_avg].std().max()

    # ------------------------------------------------
    # Heatmap 1 — Intention vs Control Type
    # ------------------------------------------------
    pivot1 = (
        df_clean
        .groupby(['intention','scenario'])[col_avg]
        .std()
        .unstack()
    )

    pivot1.index   = [INTENTIONS_MAP.get(i,i) for i in pivot1.index]
    pivot1.columns = [SCENARIOS_MAP.get(c,c) for c in pivot1.columns]

    sns.heatmap(
        pivot1,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=axes[0],
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        cbar=False
    )

    axes[0].set_title(
        f'Std {cfg["title"]} — Intention vs Control Type',
        fontsize=12,
        fontweight='bold'
    )

    axes[0].set_xlabel('Control Type', fontsize=11)
    axes[0].set_ylabel('Intention', fontsize=11)

    # ------------------------------------------------
    # Heatmap 2 — Detailed (Control + Traffic Rate)
    # ------------------------------------------------
    df_temp = df_clean.copy()

    df_temp['scenario_traffic'] = (
        df_temp['scenario'].astype(str) + ' | ' + df_temp['traffic_rate'].astype(str)
    )

    pivot2 = (
        df_temp
        .groupby(['intention','scenario_traffic'])[col_avg]
        .std()
        .unstack()
    )

    pivot2.index = [INTENTIONS_MAP.get(i,i) for i in pivot2.index]

    # ------------------------------------------------
    # Sort detailed columns by scenario order
    # ------------------------------------------------
    ordered_cols = []

    for scen in scenario_order:
        for col in pivot2.columns:
            if col.startswith(scen + " |"):
                ordered_cols.append(col)

    pivot2 = pivot2[ordered_cols]

    sns.heatmap(
        pivot2,
        annot=False,
        cmap='YlOrRd',
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': f"Std {cfg['label']}"}
    )

    axes[1].set_title(
        f'Std {cfg["title"]} — Detailed (Control + Traffic Rate)',
        fontsize=12,
        fontweight='bold'
    )

    axes[1].set_xlabel('Control Type + Traffic Rate', fontsize=11)
    axes[1].set_ylabel('Intention', fontsize=11)

    axes[1].tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3C_STD_summary_heatmap.png")



# ---------------------------------------------------------------------------
# PLOT 3D — Scenario vs Traffic Rate heatmaps (Avg / Max / Std)
#           All intentions combined
# ---------------------------------------------------------------------------
def plot_scenario_traffic_heatmap_stats(df, metric):

    cfg = METRIC_CONFIG[metric]
    col_avg = cfg['col_avg']
    col_max = cfg['col_max']

    df_clean = _clean_summary(df, metric).copy()

    # ------------------------------------------------
    # Enforce scenario ordering
    # ------------------------------------------------
    scenario_order = list(SCENARIOS_MAP.keys())

    df_clean['scenario'] = pd.Categorical(
        df_clean['scenario'],
        categories=scenario_order,
        ordered=True
    )

    # ------------------------------------------------
    # Create pivot tables
    # ------------------------------------------------
    pivot_avg = (
        df_clean
        .groupby(['scenario', 'traffic_rate'])[col_avg]
        .mean()
        .unstack()
    )

    pivot_max = (
        df_clean
        .groupby(['scenario', 'traffic_rate'])[col_max]
        .max()
        .unstack()
    )

    pivot_std = (
        df_clean
        .groupby(['scenario', 'traffic_rate'])[col_avg]
        .std()
        .unstack()
    )

    # ------------------------------------------------
    # Sort traffic_rate numerically
    # ------------------------------------------------
    try:
        sorted_cols = sorted(
            pivot_avg.columns,
            key=lambda x: float(x) if str(x).replace('.', '').isdigit() else x
        )

        pivot_avg = pivot_avg[sorted_cols]
        pivot_max = pivot_max[sorted_cols]
        pivot_std = pivot_std[sorted_cols]

    except Exception:
        pass

    # Rename scenario labels
    pivot_avg.index = [SCENARIOS_MAP.get(i, i) for i in pivot_avg.index]
    pivot_max.index = [SCENARIOS_MAP.get(i, i) for i in pivot_max.index]
    pivot_std.index = [SCENARIOS_MAP.get(i, i) for i in pivot_std.index]

    # Swap axes (traffic rate vertical)
    pivot_avg = pivot_avg.T
    pivot_max = pivot_max.T
    pivot_std = pivot_std.T

    # =========================================================
    # 1️⃣ Average Heatmap
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        pivot_avg,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': cfg['label']},
        ax=ax
    )

    ax.set_title(f'Average {cfg["title"]}\n(All Intentions Combined)')
    ax.set_xlabel("Control Type")
    ax.set_ylabel("Traffic Rate")

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3D_avg_scenario_traffic_heatmap.png")

    # =========================================================
    # 2️⃣ Maximum Heatmap
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        pivot_max,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': cfg['label']},
        ax=ax
    )

    ax.set_title(f'Maximum {cfg["title"]}\n(All Intentions Combined)')
    ax.set_xlabel("Control Type")
    ax.set_ylabel("Traffic Rate")

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3D_max_scenario_traffic_heatmap.png")

    # =========================================================
    # 3️⃣ Standard Deviation Heatmap
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        pivot_std,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': f"Std {cfg['label']}"},
        ax=ax
    )

    ax.set_title(f'Standard Deviation of {cfg["title"]}\n(All Intentions Combined)')
    ax.set_xlabel("Control Type")
    ax.set_ylabel("Traffic Rate")

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_3D_std_scenario_traffic_heatmap.png")


# ---------------------------------------------------------------------------
# PLOT 4 — Box plots + CDF per scenario
#
#   Box plots: one box per scenario, all plotted on the same axis
#              (each data point = one vehicle trip)
#
#   CDF:       empirical CDF per scenario
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PLOT — Box plot per scenario (vehicle-level distribution)
# ---------------------------------------------------------------------------
def plot_scenario_boxplots(df_vehicle, metric):

    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)

    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, ax_box = plt.subplots(figsize=(12,6))

    data_list = []
    labels    = []

    for scenario in scenarios:
        data = df_clean[df_clean['scenario'] == scenario][col_vehicle].dropna()
        data_list.append(data)
        labels.append(SCENARIOS_MAP.get(scenario, scenario))

    # ------------------------------------------------
    # Compute adaptive y-axis range
    # ------------------------------------------------
    all_values = np.concatenate(data_list)

    y_min = np.min(all_values)
    y_max = np.max(all_values)

    y_range = y_max - y_min

    y_min_plot = max(0, y_min - 0.05 * y_range)
    y_max_plot = y_max + 0.15 * y_range

    # ------------------------------------------------
    # Boxplot
    # ------------------------------------------------
    bp = ax_box.boxplot(
        data_list,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.3)
    )

    ax_box.set_ylim(y_min_plot, y_max_plot)
    ax_box.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # ------------------------------------------------
    # Color boxes
    # ------------------------------------------------
    for patch, scenario in zip(bp['boxes'], scenarios):
        patch.set_facecolor(SCENARIO_COLORS.get(scenario, '#95a5a6'))
        patch.set_alpha(0.7)

    # ------------------------------------------------
    # Mean marker + annotation
    # ------------------------------------------------
    ymax = ax_box.get_ylim()[1]

    for i, (data, scenario) in enumerate(zip(data_list, scenarios)):

        mean_val = data.mean()
        n_veh    = len(data)

        ax_box.plot(
            i + 1,
            mean_val,
            marker='*',
            color='red',
            markersize=14,
            markeredgecolor='darkred',
            markeredgewidth=1.5,
            zorder=3
        )

        ax_box.text(
            i + 1,
            ymax * 0.95,
            f"Mean: {mean_val:.1f}s\n(n={n_veh:,})",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold'
        )

    ax_box.set_xticklabels(labels, rotation=20)
    ax_box.set_ylabel(cfg['label'], fontsize=11)

    ax_box.set_title(
        f'{cfg["title"]} Distribution by Controller Type',
        fontsize=13,
        fontweight='bold'
    )

    ax_box.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_4A_scenario_boxplot.png")

# ---------------------------------------------------------------------------
# PLOT — CDF per scenario (vehicle-level distribution)
# ---------------------------------------------------------------------------
def plot_scenario_cdf(df_vehicle, metric):

    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)

    scenarios = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, ax_cdf = plt.subplots(figsize=(12,6))

    for scenario in scenarios:

        data = np.sort(
            df_clean[df_clean['scenario'] == scenario][col_vehicle].dropna().values
        )

        cdf = np.arange(1, len(data) + 1) / len(data)

        ax_cdf.plot(
            data,
            cdf,
            linewidth=2.5,
            color=SCENARIO_COLORS.get(scenario, '#95a5a6'),
            label=f"{SCENARIOS_MAP.get(scenario, scenario)} (n={len(data):,})"
        )

    ax_cdf.set_xlabel(cfg['label'], fontsize=12)
    ax_cdf.set_ylabel("CDF", fontsize=12)
    ax_cdf.set_ylim(0, 1)

    ax_cdf.set_title(
        f'Empirical CDF — {cfg["title"]} per Vehicle (All Scenarios)',
        fontsize=13,
        fontweight='bold'
    )

    ax_cdf.grid(alpha=0.3)

    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_4B_scenario_cdf.png")

# ---------------------------------------------------------------------------
# PLOT 5 — Box plots + CDF per intention
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PLOT — Box plot per intention
# ---------------------------------------------------------------------------
def plot_intention_boxplots(df_vehicle, metric):

    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)

    intentions = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    n_intents  = len(intentions)

    colors = sns.color_palette("Set2", n_intents)

    fig, ax_box = plt.subplots(figsize=(12,6))

    data_list = []
    labels = []

    for intention in intentions:
        data = df_clean[df_clean['intention'] == intention][col_vehicle].dropna()
        data_list.append(data)
        labels.append(INTENTIONS_MAP.get(intention, intention))

    bp = ax_box.boxplot(
        data_list,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.3)
    )

    # apply colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.margins(y=0.20)
    ymax = ax_box.get_ylim()[1]

    # mean markers + annotations
    for i, (data, intention) in enumerate(zip(data_list, intentions)):

        mean_val = data.mean()
        n_veh    = len(data)

        ax_box.plot(
            i + 1,
            mean_val,
            marker='*',
            color='orange',
            markersize=14,
            markeredgecolor='darkorange',
            markeredgewidth=1.5,
            zorder=3
        )

        ax_box.text(
            i + 1,
            ymax * 0.95,
            f"Mean: {mean_val:.1f}s\n(n={n_veh:,} vehicles)",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold'
        )

    ax_box.set_xticklabels(labels, rotation=20)
    ax_box.set_ylabel(cfg['label'], fontsize=11)

    ax_box.set_title(
        f'{cfg["title"]} Distribution by Intention',
        fontsize=13,
        fontweight='bold'
    )

    ax_box.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_5A_intention_boxplot.png")

# ---------------------------------------------------------------------------
# PLOT — CDF per intention
# ---------------------------------------------------------------------------
def plot_intention_cdf(df_vehicle, metric):

    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean    = _clean_vehicle(df_vehicle, metric)

    intentions = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    n_intents  = len(intentions)

    colors = sns.color_palette("Set2", n_intents)

    fig, ax_cdf = plt.subplots(figsize=(12,6))

    for idx, intention in enumerate(intentions):

        data = np.sort(
            df_clean[df_clean['intention'] == intention][col_vehicle].dropna().values
        )

        cdf = np.arange(1, len(data) + 1) / len(data)

        ax_cdf.plot(
            data,
            cdf,
            linewidth=2.5,
            color=colors[idx],
            label=f"{INTENTIONS_MAP.get(intention, intention)} (n={len(data):,})"
        )

    ax_cdf.set_xlabel(cfg['label'], fontsize=12)
    ax_cdf.set_ylabel('CDF', fontsize=12)
    ax_cdf.set_ylim(0, 1)

    ax_cdf.set_title(
        f'Empirical CDF — {cfg["title"]} per Vehicle',
        fontsize=13,
        fontweight='bold'
    )

    ax_cdf.grid(alpha=0.3)

    ax_cdf.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    _save(fig, f"{cfg['filename_prefix']}_5B_intention_cdf.png")


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
    fig.suptitle(f'{cfg["title"]} — Min / Avg / Max per Traffic Rate (Considering all Intentions)',
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
        # 'overall':        df_clean[[col_avg, col_min, col_max]].describe(),
        'by_intention':   summarise('intention'),
        'by_scenario':    summarise('scenario'),
        'by_combination': summarise(['scenario', 'intention']),
        # NEW — used for PLOT 6
        'by_scenario_traffic_rate':
            df_clean.groupby(['scenario', 'traffic_rate']).agg(
                runs=(col_avg, 'count'),
                avg=(col_avg, 'mean'),
                std=(col_avg, 'std'),
                min=(col_min, 'min'),
                max=(col_max, 'max'),
            )
    }

    txt_path = os.path.join(output_dir, f"{cfg['filename_prefix']}_statistics_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"SIMULATION STATISTICS: {cfg['title'].upper()}\n")
        f.write("(Derived from per-run min / avg / max columns)\n")
        f.write("=" * 60 + "\n\n")
        for title, table in [
            # ("OVERALL",                          stats['overall']),
            ("BY INTENTION",                     stats['by_intention']),
            ("BY SCENARIO (CONTROL TYPE)",       stats['by_scenario']),
            ("BY COMBINATION (Scenario+Intent)", stats['by_combination']),
            # NEW section for Plot 6
            ("BY SCENARIO AND TRAFFIC RATE (USED IN PLOT 6)",
             stats['by_scenario_traffic_rate'])
        ]:
            f.write(f"{title}\n{'-'*60}\n{table.to_string()}\n\n{'='*60}\n\n")

    csv_path = os.path.join(output_dir, f"{cfg['filename_prefix']}_summary_by_scenario_intention.csv")
    stats['by_combination'].reset_index().to_csv(csv_path, index=False)
    # NEW CSV for Plot 6
    csv_plot6 = os.path.join(
        output_dir,
        f"{cfg['filename_prefix']}_summary_by_scenario_traffic_rate.csv"
    )
    stats['by_scenario_traffic_rate'].reset_index().to_csv(csv_plot6, index=False)
    print(f"Saved: {os.path.basename(txt_path)} + {os.path.basename(csv_path)}")

# ---------------------------------------------------------------------------
# Statistics summary — per-vehicle data
# ---------------------------------------------------------------------------
def generate_vehicle_statistics(df_vehicle, metric):

    cfg         = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']

    df_clean = _clean_vehicle(df_vehicle, metric)

    # ------------------------------------------------
    # Aggregations
    # ------------------------------------------------
    def summarise(groupby_cols):
        return df_clean.groupby(groupby_cols).agg(
            trips=('vehicle_id', 'count') if 'vehicle_id' in df_clean.columns else (col_vehicle, 'count'),
            avg=(col_vehicle, 'mean'),
            std=(col_vehicle, 'std'),
            min=(col_vehicle, 'min'),
            max=(col_vehicle, 'max'),
            p95=(col_vehicle, lambda x: np.percentile(x, 95))
        )

    stats = {
        'by_scenario': summarise('scenario'),
        'by_intention': summarise('intention'),
        'by_combination': summarise(['scenario', 'intention']),
        'by_scenario_traffic_rate': summarise(['scenario', 'traffic_rate'])
    }

    # ------------------------------------------------
    # Save CSV files
    # ------------------------------------------------
    base = cfg['filename_prefix']

    stats['by_scenario'].reset_index().to_csv(
        os.path.join(output_dir, f"{base}_vehicle_stats_by_scenario.csv"),
        index=False
    )

    stats['by_intention'].reset_index().to_csv(
        os.path.join(output_dir, f"{base}_vehicle_stats_by_intention.csv"),
        index=False
    )

    stats['by_combination'].reset_index().to_csv(
        os.path.join(output_dir, f"{base}_vehicle_stats_by_scenario_intention.csv"),
        index=False
    )

    stats['by_scenario_traffic_rate'].reset_index().to_csv(
        os.path.join(output_dir, f"{base}_vehicle_stats_by_scenario_traffic_rate.csv"),
        index=False
    )

    # ------------------------------------------------
    # Optional TXT summary (like your existing style)
    # ------------------------------------------------
    txt_path = os.path.join(output_dir, f"{base}_vehicle_statistics_summary.txt")

    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"VEHICLE-LEVEL STATISTICS: {cfg['title'].upper()}\n")
        f.write("(Each row = one vehicle trip)\n")
        f.write("=" * 60 + "\n\n")

        for title, table in [
            ("BY SCENARIO", stats['by_scenario']),
            ("BY INTENTION", stats['by_intention']),
            ("BY SCENARIO + INTENTION", stats['by_combination']),
            ("BY SCENARIO + TRAFFIC RATE", stats['by_scenario_traffic_rate'])
        ]:
            f.write(f"{title}\n{'-'*60}\n{table.to_string()}\n\n{'='*60}\n\n")

    print(f"Saved vehicle stats CSV + TXT for {cfg['title']}")


# ---------------------------------------------------------------------------
# NEW PLOTS: Max, 95th Percentile, and STD (vehicle-level distributions)
# ---------------------------------------------------------------------------
def _plot_custom_stat_bars(df_vehicle, metric, stat_name, stat_func, filename_suffix):
    cfg = METRIC_CONFIG[metric]
    col_vehicle = cfg['col_vehicle']
    df_clean = _clean_vehicle(df_vehicle, metric)

    intentions = [i for i in INTENTIONS_MAP if i in df_clean['intention'].values]
    scenario_order = [s for s in SCENARIOS_MAP if s in df_clean['scenario'].values]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, intention in enumerate(intentions):
        if idx >= len(axes):
            break
        
        ax = axes[idx]

        grp = (
            df_clean[df_clean['intention'] == intention]
            .groupby(['traffic_rate', 'scenario'])[col_vehicle]
            .agg(stat_func)
            .reset_index(name='val')
        )

        pivot_bar = grp.pivot(index='traffic_rate', columns='scenario', values='val')

        # keep scenario order consistent
        pivot_bar = pivot_bar.reindex(columns=[s for s in scenario_order if s in pivot_bar.columns])

        x = np.arange(len(pivot_bar.index))

        # Dynamic bar width adjustment
        n_cols = len(pivot_bar.columns)
        width = 0.8 / n_cols if n_cols > 0 else 0.8

        for i, col in enumerate(pivot_bar.columns):
            offset = width * (i - (n_cols - 1) / 2)
            heights = pivot_bar[col].fillna(0).values

            ax.bar(
                x + offset,
                heights,
                width,
                label=SCENARIOS_MAP.get(col, col),
                color=SCENARIO_COLORS.get(col, '#333')
            )

        ax.set_xlabel('Traffic Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{stat_name} {cfg["label"]}', fontsize=10, fontweight='bold')
        ax.set_title(
            INTENTIONS_MAP.get(intention, intention),
            fontsize=12,
            fontweight='bold'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(pivot_bar.index, rotation=45, ha='right', fontsize=8)
        ax.legend(title='Control Type', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplot panels
    for idx in range(len(intentions), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{cfg["title"]} — {stat_name} per Traffic Rate\n(Grouped by Intention)',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"{cfg['filename_prefix']}_{filename_suffix}.png")


def plot_intention_scenario_bars_max(df_vehicle, metric):
    _plot_custom_stat_bars(df_vehicle, metric, 'Max', 'max', '7A_max_intention_scenario_bars')

def plot_intention_scenario_bars_p95(df_vehicle, metric):
    _plot_custom_stat_bars(df_vehicle, metric, '95th Percentile', lambda x: np.percentile(x, 95), '7B_p95_intention_scenario_bars')

def plot_intention_scenario_bars_std(df_vehicle, metric):
    _plot_custom_stat_bars(df_vehicle, metric, 'Standard Deviation of', 'std', '7C_std_intention_scenario_bars')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Simulation Analysis — all plots derived from per-vehicle data")
    print("=" * 60)

    # Load per-vehicle data (primary source for ALL plots)
    df_vehicle = load_vehicle_data()

    # Also load the pre-summarised CSV (used only for generate_statistics text report)
    df_summary_file = load_summary_data()

    if df_vehicle is None:
        print("No per-vehicle data found.  Exiting.")
        return

    # Derive run-level summary from vehicle data — used by bars / heatmaps / ribbon
    df_summary = derive_summary_from_vehicle(df_vehicle)

    print(f"\nPer-vehicle data:")
    print(f"  Total trips: {len(df_vehicle):,}")
    print(f"  Scenarios  : {df_vehicle['scenario'].dropna().unique()}")
    print(f"  Intentions : {df_vehicle['intention'].dropna().unique()}")

    if df_summary is not None:
        print(f"\nDerived run-level summary (from vehicle data):")
        print(f"  Total runs : {len(df_summary)}")
        print(f"  Rate groups: {df_summary['traffic_rate'].nunique()}")

    for metric in ['travel_time', 'waiting_time']:
        cfg = METRIC_CONFIG[metric]
        print(f"\n{'='*60}")
        print(f"Generating plots for: {cfg['title']} ...")

        # ------------------------------------------------------------------
        # Bar charts, heatmaps, ribbon — now all use vehicle-derived summary
        # ------------------------------------------------------------------
        if df_summary is not None:
            missing = [cfg[c] for c in ('col_avg', 'col_min', 'col_max')
                       if cfg[c] not in df_summary.columns]
            if missing:
                print(f"  [SKIP] Derived summary columns not found: {missing}")
            else:
                plot_intention_scenario_bars(df_summary, metric)

                plot_control_type_heatmaps_avg(df_summary, metric)
                plot_control_type_heatmaps_max(df_summary, metric)
                plot_control_type_heatmaps_std(df_summary, metric)
                plot_scenario_traffic_heatmap_stats(df_summary, metric)

                plot_heatmap_avg(df_summary, metric)
                plot_heatmap_max(df_summary, metric)
                plot_heatmap_std(df_summary, metric)

                plot_min_avg_max_ribbon(df_summary, metric)

                # Statistics text report: prefer file summary if available,
                # fall back to derived summary
                stats_source = df_summary_file if df_summary_file is not None else df_summary
                print(f"Generating statistics for: {cfg['title']} ...")
                generate_statistics(stats_source, metric)

        # ------------------------------------------------------------------
        # Box plots, CDFs — always used per-vehicle data
        # ------------------------------------------------------------------
        col_v = cfg['col_vehicle']
        if col_v not in df_vehicle.columns:
            print(f"  [SKIP] Vehicle column '{col_v}' not found in vehicle CSVs")
        else:
            plot_scenario_boxplots(df_vehicle, metric)
            plot_scenario_cdf(df_vehicle, metric)
            plot_intention_boxplots(df_vehicle, metric)
            plot_intention_cdf(df_vehicle, metric)

            print(f"Generating vehicle-level statistics for: {cfg['title']} ...")
            generate_vehicle_statistics(df_vehicle, metric)

            # --- NEW METRIC PLOTS ADDED HERE ---
            print(f"Generating Max, P95, and STD plots for: {cfg['title']} ...")
            plot_intention_scenario_bars_max(df_vehicle, metric)
            plot_intention_scenario_bars_p95(df_vehicle, metric)
            plot_intention_scenario_bars_std(df_vehicle, metric)

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
