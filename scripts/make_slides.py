#!/usr/bin/env python3
"""
Build docs/Shielded_PPO_Slides.pdf (4 slides, 16:9) plus one PNG per slide in
experiments/submission/slides/ for pasting into PowerPoint.

    python scripts/make_slides.py
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "eval_results")
OUT = os.path.join(ROOT, "docs", "Shielded_PPO_Slides.pdf")
PNG = os.path.join(ROOT, "experiments", "submission", "slides")
os.makedirs(PNG, exist_ok=True)

SIZE = (13.33, 7.5)
INK, MUTED = "#12263a", "#5b6b7c"
BLUE, GREEN, RED, ORANGE, GREY = "#1f5fa8", "#2e7d32", "#c62828", "#ef6c00", "#9aa5b1"


def load(tag):
    f = os.path.join(RES, f"{tag}_results.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    g = lambda k: sum(float(r.get(k) or 0) for r in rows)
    return dict(col=100 * g("collisions") / g("n_episodes"), c=int(g("collisions")),
                n=int(g("n_episodes")), tt=g("avg_travel_time") / len(rows),
                succ=g("success_rate") / len(rows))


def slide(pdf, name, title, subtitle=None):
    fig = plt.figure(figsize=SIZE)
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.94, title, fontsize=25, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.045, 0.875, subtitle, fontsize=13, color=MUTED, va="top")
    fig.text(0.955, 0.035, name, fontsize=9, color=MUTED, ha="right")
    return fig


def box(ax, x, y, w, h, label, fc, ec=None, fs=10.5, tc="white", weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec or fc, lw=1.6))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.0, label=None, fs=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.022, label, ha="center", va="bottom",
                fontsize=fs, color=color)


def finish(pdf, fig, name):
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=170, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


with PdfPages(OUT) as pdf:
    # ================================================== SLIDE 1: architecture
    fig = slide(pdf, "1 / 4", "Shielded PPO at an unsignalised junction",
                "The learned policy proposes an acceleration; the shield inspects it and may override it before SUMO applies it")
    ax = fig.add_axes([0.03, 0.06, 0.40, 0.76]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # junction sketch
    ax.add_patch(Rectangle((0.38, 0.0), 0.24, 1.0, fc="#eef1f5", ec="none"))
    ax.add_patch(Rectangle((0.0, 0.38), 1.0, 0.24, fc="#eef1f5", ec="none"))
    ax.add_patch(Circle((0.5, 0.5), 0.30, fill=False, ec=BLUE, ls=":", lw=1.6))
    ax.text(0.5, 0.835, "perception radius 50 m", ha="center", fontsize=9, color=BLUE)
    ax.add_patch(Rectangle((0.44, 0.10), 0.055, 0.10, fc=BLUE, ec=BLUE))
    ax.text(0.5, 0.055, "ego (RL)", ha="center", fontsize=9.5, color=BLUE, weight="bold")
    for (x, y, dx, dy) in [(0.70, 0.455, -1, 0), (0.28, 0.505, 1, 0), (0.455, 0.72, 0, -1)]:
        ax.add_patch(Rectangle((x, y), 0.055 if dy else 0.08, 0.08 if dy else 0.055, fc=RED, ec=RED))
        ax.annotate("", xy=(x + 0.10 * dx, y + 0.10 * dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=RED))
    ax.plot([0.5], [0.5], marker="X", ms=13, color=ORANGE)
    ax.text(0.53, 0.47, "conflict point", fontsize=9, color=ORANGE)
    ax.text(0.5, -0.04, "right-before-left priority, no traffic light", ha="center",
            fontsize=9.5, color=MUTED)

    ax2 = fig.add_axes([0.45, 0.06, 0.53, 0.78]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    box(ax2, 0.0, 0.83, 1.0, 0.13,
        "STATE  s  (37 values)\nego 4: dist-to-goal, speed, sin θ, cos θ   |   leader 3: gap, speed, TTC\n"
        "5 neighbours × 5: dist-to-conflict-point, speed, Δ arrival time, sin Δθ, cos Δθ   |   mask 5",
        "#e8eef6", ec=BLUE, tc=INK, fs=9.3, weight="normal")
    arrow(ax2, 0.5, 0.82, 0.5, 0.76)
    box(ax2, 0.12, 0.62, 0.76, 0.13,
        "Multi-head cross-attention encoder\nego = query, neighbours = keys/values, masked → 256-d context", BLUE, fs=10)
    arrow(ax2, 0.5, 0.61, 0.5, 0.55)
    box(ax2, 0.17, 0.45, 0.66, 0.12, "PPO policy  π(a|s)\nActor 256×256   Critic 256×256", "#26456e", fs=10)
    arrow(ax2, 0.5, 0.44, 0.5, 0.375)
    ax2.text(0.5, 0.405, "proposed action  a ∈ [−1, 1]", ha="center", va="center",
             fontsize=9.5, color=MUTED, bbox=dict(fc="white", ec="none", pad=1.5))
    box(ax2, 0.05, 0.135, 0.90, 0.225,
        "SAFETY SHIELD   (checks the proposed acceleration)\n\n"
        "① TTC: brake if time-to-conflict < 2 s along the path\n"
        "② RSS: keep v·t_react + v²/2a + gap clear, else brake\n"
        "③ Right-of-way: yield when a vehicle arrives from the right\n\n"
        "extensions:  commit zone — never brake past the point of no return\n"
        "rear-aware — cap braking when a follower is close",
        "#fdf0e6", ec=ORANGE, tc=INK, fs=9.3, weight="normal")
    arrow(ax2, 0.5, 0.125, 0.5, 0.07, color=ORANGE)
    box(ax2, 0.13, 0.0, 0.74, 0.065,
        "ACTION applied in SUMO:   a ≥ 0 → a × 2.6 m/s²      a < 0 → a × 4.5 m/s²",
        "#eaf3ea", ec=GREEN, tc=INK, fs=10, weight="normal")
    finish(pdf, fig, "slide1_architecture")

    # ================================================== SLIDE 2: reward
    fig = slide(pdf, "2 / 4", "Reward function",
                "Sparse terminal outcome plus dense per-step shaping; the shield adds a small penalty for over-reliance")
    axl = fig.add_axes([0.045, 0.10, 0.50, 0.71]); axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    box(axl, 0.0, 0.80, 1.0, 0.16,
        "TERMINAL\n\ncollision  r = −10        goal reached  r = +15", "#f7e9e9", ec=RED, tc=INK, fs=12, weight="normal")
    box(axl, 0.0, 0.40, 1.0, 0.34,
        "PER STEP\n\n"
        "r  =  10·Δprogress        progress toward the exit\n"
        "      +  1·safety_penalty    conflict proximity\n"
        "      −  0.01                 time penalty",
        "#e8eef6", ec=BLUE, tc=INK, fs=12, weight="normal")
    box(axl, 0.0, 0.10, 1.0, 0.24,
        "safety_penalty  =  Σ  −exp(−10·|Δη|)   for neighbours with |Δη| < 0.2\n"
        "                    −exp(−10·TTC)      when a leader is closer than TTC 0.2\n\n"
        "Δη = normalised difference in arrival time at the shared conflict point",
        "#fdf6e6", ec=ORANGE, tc=INK, fs=10.5, weight="normal")
    box(axl, 0.0, 0.0, 1.0, 0.07,
        "shielded variant:  additional −0.005 when the shield overrides > 50% of steps",
        "#eef1f5", ec=GREY, tc=INK, fs=10, weight="normal")

    axr = fig.add_axes([0.62, 0.30, 0.33, 0.44])
    d = np.linspace(0.001, 0.4, 400)
    pen = np.where(d < 0.2, -np.exp(-10 * d), 0)
    axr.plot(d, pen, lw=2.6, color=ORANGE)
    axr.axvline(0.2, ls=":", color=MUTED)
    axr.text(0.205, -0.15, "cut-off |Δη| = 0.2", fontsize=9, color=MUTED)
    axr.set_xlabel("|Δη|   (0 = both arrive at the same instant)", fontsize=10)
    axr.set_ylabel("safety penalty per neighbour", fontsize=10)
    axr.set_title("Penalty spikes as arrival times coincide", fontsize=11.5, weight="bold", color=INK)
    axr.grid(alpha=0.3)
    fig.text(0.62, 0.20,
             "The penalty grows exponentially as two vehicles are\n"
             "projected to reach the conflict point together, so the\n"
             "policy learns to break the tie by slowing or clearing.",
             fontsize=10.5, color=MUTED, va="top")
    finish(pdf, fig, "slide2_reward")

    # ================================================== SLIDE 3: results, 275 pipeline
    B = {"Heuristic\n+ Discrete": load("heuristic_discrete"),
         "Heuristic\n+ Continuous": load("heuristic_continous"),
         "Attention\n+ Continuous": load("attention_continous"),
         "Attention\n+ Discrete": load("attention_discrete"),
         "Attention + Cont\n+ SHIELD": load("shielded_attention_continous__policy_attention_continous")}
    B = {k: v for k, v in B.items() if v}
    fig = slide(pdf, "3 / 4", "Results in our training setup (275 veh/h, rule-abiding traffic)",
                "Evaluated on the pre-defence benchmark: 6 scenarios × 4 intentions × 42 episodes = 1,008 episodes per controller")
    ax = fig.add_axes([0.07, 0.30, 0.52, 0.48])
    keys = list(B)
    vals = [B[k]["col"] for k in keys]
    cols = [GREY, GREY, BLUE, BLUE, GREEN]
    ax.bar(keys, vals, color=cols[:len(keys)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=11, weight="bold", color=INK)
    ax.set_ylabel("Collision rate (%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Shield reaches the lowest rate, tied with Attention + Discrete",
                 fontsize=12, weight="bold", color=INK)
    sh, off = B.get("Attention + Cont\n+ SHIELD"), B.get("Attention\n+ Continuous")
    fig.text(0.64, 0.74, "Why the shield works here", fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.635, 0.685,
             f"• Shield halves its own baseline:\n"
             f"   {off['col']:.2f}% → {sh['col']:.2f}%  (10 → 4 collisions of 1,008;\n"
             f"   p = 0.18, not significant)\n\n"
             "• Background vehicles obey SUMO safety\n"
             "   rules (speed_mode 31), so they brake for\n"
             "   the ego car. Shield braking is therefore\n"
             "   safe: it removes crossing conflicts\n"
             "   without creating rear-end conflicts.\n\n"
             f"• Cost: travel time {off['tt']:.1f} s → {sh['tt']:.1f} s, success\n"
             f"   {off['succ']:.1f}% → {sh['succ']:.1f}%, 18.5% of actions overridden\n\n"
             "• Pre-defence report, same benchmark:\n"
             "   best controller 0.88%; heuristic\n"
             "   baselines 5.31% and 6.73%",
             fontsize=10.8, color=INK, va="top", linespacing=1.5)
    fig.text(0.07, 0.135,
             "Honest reading: the three attention-based controllers (0.40%, 0.40%, 0.99%) are statistically\n"
             "indistinguishable (all pairwise p ≥ 0.17). The shield is in the best tier and ties Attention + Discrete;\n"
             "it is not proven better than either attention controller.",
             fontsize=10, color=MUTED, va="top", linespacing=1.6)
    finish(pdf, fig, "slide3_results_ours")

    # ================================================== SLIDE 4: Ibrahima's scenario
    C = {"No shield": load("attention_continous__policy_attention_continous__complex"),
         "Shield": load("shielded_attention_continous__policy_attention_continous__complex"),
         "Rear-aware": load("shielded_attention_continous__policy_attention_continous__complex_rear"),
         "Rear-aware\n+ commit zone": load("shielded_attention_continous__policy_attention_continous__complex_commit_rear")}
    C = {k: v for k, v in C.items() if v}
    fig = slide(pdf, "4 / 4", "Why the shield cannot win in the pre-defence scenario",
                "400 veh/h cross traffic, traffic in the agent's own lane, background vehicles that ignore safety rules (speed_mode 0)")
    # mechanism drawing
    ax = fig.add_axes([0.045, 0.34, 0.45, 0.44]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.0, 0.34), 1.0, 0.30, fc="#eef1f5", ec="none"))
    ax.plot([0, 1], [0.49, 0.49], ls="--", color="white", lw=2)
    ax.add_patch(Rectangle((0.56, 0.40), 0.12, 0.09, fc=BLUE, ec=BLUE))
    ax.text(0.62, 0.53, "ego", ha="center", fontsize=10, color=BLUE, weight="bold")
    ax.text(0.78, 0.31, "shield brakes for\na crossing car", ha="center", fontsize=9.5, color=ORANGE, va="top")
    ax.annotate("", xy=(0.70, 0.445), xytext=(0.80, 0.445),
                arrowprops=dict(arrowstyle="-|>", lw=3, color=ORANGE))
    ax.add_patch(Rectangle((0.22, 0.40), 0.12, 0.09, fc=RED, ec=RED))
    ax.text(0.28, 0.53, "follower (speed_mode 0)", ha="center", fontsize=10, color=RED, weight="bold")
    ax.annotate("", xy=(0.54, 0.445), xytext=(0.36, 0.445),
                arrowprops=dict(arrowstyle="-|>", lw=3, color=RED))
    ax.text(0.30, 0.31, "does not brake", ha="center", fontsize=9.5, color=RED, va="top")
    ax.plot([0.55], [0.445], marker="*", ms=26, color="#ffcc00", mec=RED, mew=1.5)
    ax.text(0.5, 0.88, "The shield's only tool is braking", ha="center", fontsize=12.5,
            weight="bold", color=INK)
    ax.text(0.5, 0.78, "Slowing down in front of traffic that never yields\nconverts a crossing conflict into a rear-end collision.",
            ha="center", fontsize=10.5, color=MUTED, va="top", linespacing=1.5)

    ax2 = fig.add_axes([0.57, 0.30, 0.38, 0.47])
    keys = list(C)
    vals = [C[k]["col"] for k in keys]
    cols = [BLUE, RED, RED, GREEN][:len(keys)]
    ax2.bar(keys, vals, color=cols)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=10.5, weight="bold", color=INK)
    ax2.axhline(C["No shield"]["col"], ls=":", color=BLUE)
    ax2.set_ylabel("Collision rate (%)", fontsize=11)
    ax2.set_ylim(0, max(vals) * 1.28)
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_title("Pre-defence environment (1,008 episodes each)", fontsize=11.5, weight="bold", color=INK)

    fig.text(0.045, 0.245,
             "• Original shield made it significantly worse: 6.35% → 9.72% (p = 0.007). Left turns, where it brakes most, went 11.1% → 23.8%.\n"
             "• Capping the braking (rear-aware) removed some rear-end crashes but let crossing conflicts through: 9.42%, still worse (p = 0.013).\n"
             "• Adding the commit zone — never brake once the car cannot stop before the conflict point — removed the harm entirely: 6.35%,\n"
             "   identical to no shield, with overrides down from 8.8% to 2.9% and travel time from 26.7 s to 23.3 s.\n"
             "• Training the policy with the shield active (824k steps, half the budget) reached 12.70%: inconclusive, not a fix.",
             fontsize=10.6, color=INK, va="top", linespacing=1.65)
    fig.text(0.045, 0.075,
             "Conclusion: the shield helps only where surrounding traffic cooperates. In non-cooperative traffic the best refined\n"
             "shield matches the unshielded controller rather than beating it; a net gain needs the policy to be trained with the\n"
             "shield from the start.",
             fontsize=10.4, color=MUTED, va="top", linespacing=1.6)
    finish(pdf, fig, "slide4_ibrahima")

print("wrote", OUT)
print("slide PNGs in", PNG)
