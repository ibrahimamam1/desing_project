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
INK, MUTED = "#0d1b2a", "#31465a"
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
        fig.text(0.045, 0.875, subtitle, fontsize=14, color=MUTED, va="top")
    fig.text(0.985, 0.018, name, fontsize=9.5, color=MUTED, ha="right")
    return fig


def box(ax, x, y, w, h, label, fc, ec=None, fs=10.5, tc="white", weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec or fc, lw=2.2, clip_on=False))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, linespacing=1.45, clip_on=False)


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
    ax = fig.add_axes([0.03, 0.155, 0.38, 0.65]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    CX_, CY_ = 0.50, 0.52          # junction centre
    DK = "#a8332a"                 # leader colour

    # roads and lane markings
    ax.add_patch(Rectangle((0.42, 0.0), 0.16, 1.0, fc="#e9edf2", ec="none"))
    ax.add_patch(Rectangle((0.0, 0.44), 1.0, 0.16, fc="#e9edf2", ec="none"))
    for x1, y1, x2, y2 in [(CX_, 0.0, CX_, 0.42), (CX_, 0.62, CX_, 1.0),
                           (0.0, CY_, 0.40, CY_), (0.60, CY_, 1.0, CY_)]:
        ax.plot([x1, x2], [y1, y2], ls=(0, (5, 5)), color="white", lw=1.6)
    for t, x, y in [("N", CX_, 0.99), ("S", CX_, 0.005), ("E", 0.985, CY_), ("W", 0.015, CY_)]:
        ax.text(x, y, t, ha="center", va="center", fontsize=11, weight="bold", color=MUTED)

    # perception radius, drawn around the ego
    ax.add_patch(Circle((0.49, 0.42), 0.34, fill=False, ec=BLUE, ls=":", lw=1.8, clip_on=False))
    ax.text(0.5, 0.80, "perception radius 50 m", ha="center", fontsize=10.5,
            color=BLUE, weight="semibold")

    # ego, its leader, and the route it intends to take
    ax.add_patch(Rectangle((0.455, 0.10), 0.05, 0.085, fc=BLUE, ec=BLUE, zorder=3))
    ax.text(0.435, 0.142, "ego\n(RL)", ha="right", va="center", fontsize=11,
            color=BLUE, weight="bold", linespacing=1.3)
    ax.add_patch(Rectangle((0.455, 0.235), 0.05, 0.085, fc=DK, ec=DK, zorder=3))
    ax.text(0.525, 0.277, "leader:  gap, speed, TTC", ha="left", va="center",
            fontsize=9.8, color=DK, weight="semibold")
    ax.plot([0.480, 0.480, 0.10], [0.19, CY_, CY_], ls="--", lw=2.0, color=BLUE)
    ax.annotate("", xy=(0.07, CY_), xytext=(0.12, CY_),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=BLUE))
    ax.text(0.13, 0.625, "intended route", fontsize=9.8, color=BLUE, weight="semibold")

    # background vehicles; line thickness to the ego shows the attention weight
    for (x, y), (dx, dy), lw in [((0.505, 0.665), (0, -1), 3.2),
                                 ((0.665, 0.455), (-1, 0), 1.8),
                                 ((0.245, 0.495), (1, 0), 0.9)]:
        w_, h_ = (0.05, 0.085) if dy else (0.085, 0.05)
        ax.add_patch(Rectangle((x, y), w_, h_, fc=RED, ec=RED, zorder=3))
        cx, cy = x + w_ / 2, y + h_ / 2
        ax.annotate("", xy=(cx + 0.095 * dx, cy + 0.095 * dy), xytext=(cx + 0.05 * dx, cy + 0.05 * dy),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=RED))
        ax.plot([0.480, cx], [0.185, cy], color=ORANGE, lw=lw, alpha=0.6, zorder=2)

    # conflict point
    ax.plot([CX_], [CY_], marker="X", ms=15, color=ORANGE, zorder=5)
    ax.text(0.545, 0.565, "conflict point", fontsize=9.8, color=ORANGE, weight="semibold")

    # what the state records for each neighbour
    ax.annotate("per neighbour:\ndist-to-conflict-point, speed,\nΔ arrival time, sin Δθ, cos Δθ",
                xy=(0.50, 0.755), xytext=(0.0, 0.90), ha="left", va="center", fontsize=9.6,
                color=INK, weight="semibold", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=1.0, color=MUTED))

    ax.text(0.5, -0.05, "blue = RL ego     red = background traffic (IDM)",
            ha="center", va="top", fontsize=9.6, color=INK, clip_on=False)
    ax.text(0.5, -0.105, "orange line thickness = attention weight",
            ha="center", va="top", fontsize=9.6, color=INK, clip_on=False)
    ax.text(0.5, -0.16, "right-before-left priority, no traffic light",
            ha="center", va="top", fontsize=9.6, color=MUTED, clip_on=False)

    ax2 = fig.add_axes([0.45, 0.075, 0.53, 0.76]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    # Vertical layout, top to bottom, with a clear 0.045 gap for every arrow.
    box(ax2, 0.0, 0.825, 1.0, 0.155,
        "STATE  s   (37 values)\n"
        "ego 4:  dist-to-goal, speed, sin θ, cos θ        leader 3:  gap, speed, TTC\n"
        "5 neighbours × 5:  dist-to-conflict-point, speed, Δ arrival time, sin Δθ, cos Δθ\n"
        "mask 5  (which neighbour slots are real)",
        "#e8eef6", ec=BLUE, tc=INK, fs=10.5, weight="semibold")
    arrow(ax2, 0.5, 0.820, 0.5, 0.735)
    box(ax2, 0.04, 0.625, 0.92, 0.105,
        "Multi-head cross-attention encoder\nego = query, neighbours = keys / values → 256-d context",
        BLUE, fs=11)
    arrow(ax2, 0.5, 0.620, 0.5, 0.540)
    box(ax2, 0.20, 0.430, 0.60, 0.105,
        "PPO policy  π(a|s)\nActor 256×256    Critic 256×256", "#26456e", fs=11)
    arrow(ax2, 0.5, 0.425, 0.5, 0.358)
    ax2.text(0.53, 0.385, "  proposed action  a ∈ [−1, 1]  ", ha="left", va="center",
             fontsize=10.5, color=INK, weight="semibold")
    box(ax2, 0.01, 0.135, 0.98, 0.215,
        "SAFETY SHIELD   checks the proposed acceleration\n\n"
        "① TTC:  brake if time-to-conflict < 2 s along the path\n"
        "② RSS:  keep v·t_react + v²/2a + gap clear, else brake\n"
        "③ Right-of-way:  yield to a vehicle arriving from the right\n"
        "+ commit zone · rear-aware braking cap",
        "#fdf0e6", ec=ORANGE, tc=INK, fs=10.5, weight="semibold")
    arrow(ax2, 0.5, 0.130, 0.5, 0.098, color=ORANGE, lw=2.4)
    box(ax2, 0.04, 0.022, 0.92, 0.072,
        "ACTION in SUMO:   a ≥ 0 → a × 2.6 m/s²      a < 0 → a × 4.5 m/s²",
        "#eaf3ea", ec=GREEN, tc=INK, fs=11, weight="semibold")
    finish(pdf, fig, "slide1_architecture")

    # ================================================== SLIDE 2: reward
    fig = slide(pdf, "2 / 4", "Reward function",
                "Sparse terminal outcome plus dense per-step shaping; the shield adds a small penalty for over-reliance")
    axl = fig.add_axes([0.045, 0.10, 0.50, 0.71]); axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    box(axl, 0.0, 0.79, 1.0, 0.17,
        "TERMINAL\n\ncollision  r = −10        goal reached  r = +15", "#f7e9e9", ec=RED, tc=INK, fs=14, weight="semibold")
    box(axl, 0.0, 0.40, 1.0, 0.34,
        "PER STEP\n\n"
        "r  =  10 · Δprogress          progress toward the exit\n"
        "      +  1 · safety_penalty     conflict proximity\n"
        "      −  0.01                      time penalty",
        "#e8eef6", ec=BLUE, tc=INK, fs=14, weight="semibold")
    box(axl, 0.0, 0.11, 1.0, 0.24,
        "safety_penalty =  Σ −exp(−10·|Δη|)  over neighbours with |Δη| < 0.2\n"
        "                          −exp(−10·TTC)  when a leader is within TTC 0.2\n\n"
        "Δη = normalised gap in arrival time at the shared conflict point",
        "#fdf6e6", ec=ORANGE, tc=INK, fs=12, weight="semibold")
    box(axl, 0.0, -0.01, 1.0, 0.09,
        "shielded variant:  extra −0.005 when the shield overrides > 50% of steps",
        "#eef1f5", ec=GREY, tc=INK, fs=11.5, weight="semibold")

    axr = fig.add_axes([0.62, 0.30, 0.33, 0.44])
    d = np.linspace(0.001, 0.4, 400)
    pen = np.where(d < 0.2, -np.exp(-10 * d), 0)
    axr.plot(d, pen, lw=2.6, color=ORANGE)
    axr.axvline(0.2, ls=":", color=MUTED)
    axr.text(0.207, -0.15, "cut-off |Δη| = 0.2", fontsize=11, color=INK)
    axr.set_xlabel("|Δη|   (0 = both arrive at the same instant)", fontsize=12)
    axr.set_ylabel("safety penalty per neighbour", fontsize=12)
    axr.set_title("Penalty spikes as arrival times coincide", fontsize=13.5, weight="bold", color=INK)
    axr.grid(alpha=0.3)
    fig.text(0.62, 0.20,
             "The penalty grows exponentially as two vehicles are\n"
             "projected to reach the conflict point together, so the\n"
             "policy learns to break the tie by slowing or clearing.",
             fontsize=12, color=INK, va="top", linespacing=1.5)
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
        ax.text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=13.5, weight="bold", color=INK)
    ax.set_ylabel("Collision rate (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Shield reaches the lowest rate, tied with Attention + Discrete",
                 fontsize=13.5, weight="bold", color=INK)
    sh, off = B.get("Attention + Cont\n+ SHIELD"), B.get("Attention\n+ Continuous")
    fig.text(0.635, 0.745, "Why the shield works here", fontsize=16, weight="bold", color=INK, va="top")
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
             fontsize=12, color=INK, va="top", linespacing=1.55)
    fig.text(0.07, 0.135,
             "Honest reading: the three attention-based controllers (0.40%, 0.40%, 0.99%) are statistically\n"
             "indistinguishable (all pairwise p ≥ 0.17). The shield is in the best tier and ties Attention + Discrete;\n"
             "it is not proven better than either attention controller.",
             fontsize=11.5, color=INK, va="top", linespacing=1.6)
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
    ax.text(0.62, 0.56, "ego", ha="center", fontsize=12.5, color=BLUE, weight="bold")
    ax.text(0.80, 0.29, "shield brakes for\na crossing car", ha="center", fontsize=11.5, color=ORANGE, va="top", weight="semibold")
    ax.annotate("", xy=(0.70, 0.445), xytext=(0.80, 0.445),
                arrowprops=dict(arrowstyle="-|>", lw=3, color=ORANGE))
    ax.add_patch(Rectangle((0.22, 0.40), 0.12, 0.09, fc=RED, ec=RED))
    ax.text(0.26, 0.56, "follower (speed_mode 0)", ha="center", fontsize=12.5, color=RED, weight="bold")
    ax.annotate("", xy=(0.54, 0.445), xytext=(0.36, 0.445),
                arrowprops=dict(arrowstyle="-|>", lw=3, color=RED))
    ax.text(0.28, 0.29, "does not brake", ha="center", fontsize=11.5, color=RED, va="top", weight="semibold")
    ax.plot([0.55], [0.445], marker="*", ms=26, color="#ffcc00", mec=RED, mew=1.5)
    ax.text(0.5, 0.90, "The shield's only tool is braking", ha="center", fontsize=15,
            weight="bold", color=INK)
    ax.text(0.5, 0.79, "Slowing down in front of traffic that never yields\nconverts a crossing conflict into a rear-end collision.",
            ha="center", fontsize=12.5, color=INK, va="top", linespacing=1.5)

    ax2 = fig.add_axes([0.57, 0.335, 0.38, 0.45])
    keys = list(C)
    vals = [C[k]["col"] for k in keys]
    cols = [BLUE, RED, RED, GREEN][:len(keys)]
    ax2.bar(keys, vals, color=cols)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=12.5, weight="bold", color=INK)
    ax2.axhline(C["No shield"]["col"], ls=":", color=BLUE)
    ax2.set_ylabel("Collision rate (%)", fontsize=13)
    ax2.set_ylim(0, max(vals) * 1.28)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.set_title("Pre-defence environment (1,008 episodes each)", fontsize=13.5, weight="bold", color=INK)

    fig.text(0.045, 0.245,
             "• Original shield made it significantly worse: 6.35% → 9.72% (p = 0.007). Left turns, where it brakes most, went 11.1% → 23.8%.\n"
             "• Capping the braking (rear-aware) removed some rear-end crashes but let crossing conflicts through: 9.42%, still worse (p = 0.013).\n"
             "• Adding the commit zone — never brake once the car cannot stop before the conflict point — removed the harm entirely: 6.35%,\n"
             "   identical to no shield, with overrides down from 8.8% to 2.9% and travel time from 26.7 s to 23.3 s.\n"
             "• Training the policy with the shield active (824k steps, half the budget) reached 12.70%: inconclusive, not a fix.",
             fontsize=11.8, color=INK, va="top", linespacing=1.7)
    fig.text(0.045, 0.085,
             "Conclusion: the shield helps only where surrounding traffic cooperates. In non-cooperative traffic the best refined\n"
             "shield matches the unshielded controller rather than beating it; a net gain needs the policy to be trained with the\n"
             "shield from the start.",
             fontsize=11.6, color=INK, va="top", linespacing=1.6)
    finish(pdf, fig, "slide4_ibrahima")

print("wrote", OUT)
print("slide PNGs in", PNG)
