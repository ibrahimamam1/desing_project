"""
plot_results.py  –  reads TensorBoard event files and saves training plots to:
    <root_dir>/outputs/train/<exp_name>/

Usage (standalone):
    python plot_results.py \
        --logdir  tensorboard_logs/v0_1_heuristic_discrete/<run_name> \
        --out     outputs/train/<run_name> \
        --exp-name "AlphaV0.1 Heuristic Discrete"

Or imported and called at the end of train():
    from plot_results import plot_results
    plot_results(logdir=TENSORBOARD_RUN_DIR, output_dir=plot_out, exp_name=RUN_NAME)
"""

import os
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── TensorBoard tag → (y-axis label, panel title, higher-is-better) ──────────
# Adjust tag names to match exactly what RLlib logs to your TensorBoard.
METRICS = [
    ("ray/tune/episode_reward_mean",                "Mean Reward",          "Episode Reward",   True),
    ("ray/tune/custom_metrics/avg_speed_mean",      "Avg Speed (m/s)",      "Average Speed",    True),
    ("ray/tune/custom_metrics/collision_mean",      "Collision Rate",        "Collision Rate",   False),
    ("ray/tune/custom_metrics/success_mean",        "Success Rate",          "Success Rate",     True),
    ("ray/tune/custom_metrics/waiting_time_mean",   "Avg Waiting Time (s)", "Waiting Time",     False),
]

STAGE_COLORS = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]
SMOOTH_WIN   = 20
SMOOTH_ALPHA = 0.35


def _rolling(values, window):
    out, buf = [], []
    for v in values:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def _load_tensorboard(logdir: str) -> dict:
    """
    Walk logdir recursively to find all event files.
    Returns dict: { tag: [(step, value), ...] } sorted by step.
    """
    print(f"[plot_results] Loading TensorBoard events from: {logdir}")
    all_data = {}

    for root, _, files in os.walk(logdir):
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if not event_files:
            continue

        ea = EventAccumulator(root)
        ea.Reload()
        available = ea.Tags().get("scalars", [])

        for tag in available:
            events = ea.Scalars(tag)
            points = [(e.step, e.value) for e in events]
            if tag not in all_data:
                all_data[tag] = points
            else:
                all_data[tag].extend(points)

    # sort each tag by step
    for tag in all_data:
        all_data[tag].sort(key=lambda x: x[0])

    if all_data:
        print(f"  Found {len(all_data)} tags:")
        for t in sorted(all_data.keys()):
            print(f"    {t}  ({len(all_data[t])} points)")
    else:
        print("  WARNING: No scalar data found.")

    return all_data


def _get_series(all_data: dict, tag: str):
    """Return (steps, values) for a tag, or ([], []) if missing."""
    if tag not in all_data:
        # fuzzy match on the final tag component
        key = tag.split("/")[-1]
        matches = [k for k in all_data if key in k]
        if matches:
            tag = matches[0]
            print(f"  [~] Fuzzy matched '{tag}'")
        else:
            print(f"  [!] Tag not found: {tag}")
            return [], []
    pts = all_data[tag]
    return [p[0] for p in pts], [p[1] for p in pts]


def _plot_ax(ax, steps, values, color, label):
    smoothed = _rolling(values, SMOOTH_WIN)
    ax.plot(steps, values,   color=color, linewidth=1,   alpha=SMOOTH_ALPHA)
    ax.plot(steps, smoothed, color=color, linewidth=2.2, alpha=1.0, label=label)


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.set_title(title, color="#e6edf3", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="#30363d")


def plot_results(logdir: str, output_dir: str, exp_name: str):
    all_data = _load_tensorboard(logdir)
    os.makedirs(output_dir, exist_ok=True)

    # ── dashboard ─────────────────────────────────────────────────────────────
    n_cols = 3
    n_rows = math.ceil(len(METRICS) / n_cols)
    fig    = plt.figure(figsize=(7 * n_cols, 4.5 * n_rows), facecolor="#0f1117")
    gs     = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    for idx, (tag, ylabel, title, _) in enumerate(METRICS):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        _style_ax(ax, title, "Step", ylabel)
        steps, values = _get_series(all_data, tag)
        if steps:
            _plot_ax(ax, steps, values, STAGE_COLORS[0], "value")
            ax.legend(fontsize=8, framealpha=0.2, labelcolor="#e6edf3",
                      facecolor="#161b22", edgecolor="#30363d")

    fig.suptitle(f"Training Results  ·  {exp_name}", color="#e6edf3",
                 fontsize=16, fontweight="bold", y=1.01)

    dashboard_path = os.path.join(output_dir, "dashboard.png")
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [✓] dashboard → {dashboard_path}")

    # ── individual plots ──────────────────────────────────────────────────────
    for tag, ylabel, title, higher_better in METRICS:
        steps, values = _get_series(all_data, tag)

        fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor="#0f1117")
        _style_ax(ax2, title, "Step", ylabel)

        if steps:
            _plot_ax(ax2, steps, values, STAGE_COLORS[0], "value")

            best_val  = max(values) if higher_better else min(values)
            best_step = steps[values.index(best_val)]
            ax2.axhline(best_val, color="#f0c040", linewidth=1, linestyle=":", alpha=0.6)
            ax2.text(best_step, best_val, f"  best: {best_val:.3f}",
                     color="#f0c040", fontsize=9, va="bottom")

            ax2.legend(fontsize=9, framealpha=0.2, labelcolor="#e6edf3",
                       facecolor="#161b22", edgecolor="#30363d")

        fig2.tight_layout()
        fname    = tag.replace("/", "_").replace(" ", "_") + ".png"
        out_path = os.path.join(output_dir, fname)
        fig2.savefig(out_path, dpi=150, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"  [✓] {title:<22} → {out_path}")

    print(f"\n[plot_results] Done. All plots saved to: {output_dir}")


# ── standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir",   required=True, help="TensorBoard run directory")
    p.add_argument("--out",      required=True, help="Output directory for plots")
    p.add_argument("--exp-name", default="experiment", help="Experiment name for plot title")
    a = p.parse_args()
    plot_results(logdir=a.logdir, output_dir=a.out, exp_name=a.exp_name)
