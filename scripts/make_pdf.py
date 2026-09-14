#!/usr/bin/env python3
"""
Build docs/Shielded_PPO_Results.pdf: text summary, tables, heatmaps and charts,
all generated from eval_results/. Uses only matplotlib.

    python scripts/make_pdf.py
"""
import csv
import math
import os
import statistics
import textwrap
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n, k):
        return 0 if k < 0 or k > n else factorial(n) // (factorial(k) * factorial(n - k))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "eval_results")
OUT_PDF = os.path.join(ROOT, "docs", "Shielded_PPO_Results.pdf")
FIGDIR = os.path.join(ROOT, "experiments", "submission")
os.makedirs(FIGDIR, exist_ok=True)

A4 = (8.27, 11.69)
GREY, BLUE, GREEN, RED, ORANGE = "#8a96a3", "#1f5fa8", "#2e7d32", "#c62828", "#ef6c00"

SCEN = ["Sc1_All_Low", "Sc6_Mixed_ML", "Sc3_All_Med", "Sc4_Mixed_2H", "Sc5_Mixed_1H", "Sc2_All_High"]
SCEN_LBL = ["Sc1\nAll Low", "Sc6\nMixed ML", "Sc3\nAll Med", "Sc4\nMixed 2H", "Sc5\nMixed 1H", "Sc2\nAll High"]
INT = ["all_straight", "all_left", "uniform_random", "asymmetric_random"]
INT_LBL = ["All straight", "All left", "Uniform random", "Asymmetric random"]

PREDEF = {
    "Heuristic + Discrete":   [3.38, 5.50, 5.25, 5.75, 6.50, 5.50],
    "Heuristic + Continuous": [4.50, 6.12, 6.50, 8.50, 7.88, 6.88],
    "Attention + Discrete":   [0.50, 1.88, 1.75, 1.38, 3.00, 2.50],
    "Attention + Continuous": [0.38, 1.12, 0.88, 0.88, 1.12, 0.88],
}


# ------------------------------------------------------------------ data
def fisher(a, b, n1, n2):
    A, B, C, D = a, n1 - a, b, n2 - b
    N = A + B + C + D
    p = lambda x: comb(A + B, x) * comb(C + D, A + C - x) / comb(N, A + C)
    lo, hi = max(0, A + C - (C + D)), min(A + B, A + C)
    p0 = p(A)
    return min(sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12), 1.0)


def wilson(c, n, z=1.96):
    ph = c / n
    d = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, ctr - hw), 100 * (ctr + hw)


def load(tag):
    f = os.path.join(RES, f"{tag}_results.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    k = len(rows)
    by_sc, by_int = {}, {}
    for r in rows:
        by_sc.setdefault(r["scenario"], []).append(float(r["collision_rate"]))
        by_int.setdefault(r["intention"], []).append(float(r["collision_rate"]))
    g = lambda key: sum(float(r.get(key) or 0) for r in rows)
    return dict(
        c=int(g("collisions")), n=int(g("n_episodes")),
        col=100 * g("collisions") / g("n_episodes"),
        succ=g("success_rate") / k, tt=g("avg_travel_time") / k,
        ovr=100 * g("shield_override_rate") / k,
        ttc=int(g("shield_ttc_overrides")), rss=int(g("shield_rss_overrides")),
        row=int(g("shield_row_overrides")),
        by_sc={s: statistics.mean(v) for s, v in by_sc.items()},
        by_int={s: statistics.mean(v) for s, v in by_int.items()},
    )


STD = {
    "Heuristic + Discrete": load("heuristic_discrete"),
    "Heuristic + Continuous": load("heuristic_continous"),
    "Attention + Discrete": load("attention_discrete"),
    "Attention + Continuous": load("attention_continous"),
    "Att + Cont + Shield": load("shielded_attention_continous__policy_attention_continous"),
}
CX = {
    "No shield": load("attention_continous__policy_attention_continous__complex"),
    "Shield": load("shielded_attention_continous__policy_attention_continous__complex"),
    "Rear-aware": load("shielded_attention_continous__policy_attention_continous__complex_rear"),
    "Rear-aware +\ncommit zone": load("shielded_attention_continous__policy_attention_continous__complex_commit_rear"),
    "Shield-aware\ntraining (824k)": load("shielded_attention_continous__policy_shielded_attention_continous__complex_shieldtrained"),
}
CX = {k: v for k, v in CX.items() if v}
ST = {
    "Heuristic + Discrete": load("heuristic_discrete__policy_heuristic_discrete__stress"),
    "Heuristic + Continuous": load("heuristic_continous__policy_heuristic_continous__stress"),
    "Attention + Discrete": load("attention_discrete__policy_attention_discrete__stress"),
    "Attention + Continuous": load("attention_continous__policy_attention_continous__stress_1500k"),
    "Att + Cont + Shield": load("shielded_attention_continous__policy_attention_continous__stress_1500k"),
}
ST = {k: v for k, v in ST.items() if v}


# ------------------------------------------------------------------ page helpers
def text_page(pdf, title, blocks):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold", va="top")
    y = 0.90
    for kind, content in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.08, y, content, fontsize=11.5, weight="bold", va="top")
            y -= 0.03
        else:
            for line in textwrap.wrap(content, 96):
                fig.text(0.08, y, line, fontsize=9.3, va="top")
                y -= 0.019
            y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, header, rows, note=None, col_w=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold", va="top")
    ax = fig.add_axes([0.06, 0.45, 0.88, 0.45])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                 colWidths=col_w)
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.6)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e3e8ee")
            cell.set_text_props(weight="bold")
    if note:
        y = 0.42
        for line in textwrap.wrap(note, 100):
            fig.text(0.08, y, line, fontsize=9, va="top")
            y -= 0.018
    pdf.savefig(fig)
    plt.close(fig)


def fig_page(pdf, fig, name, caption):
    fig.text(0.08, 0.03, "\n".join(textwrap.wrap(caption, 110)), fontsize=8.5, va="bottom")
    fig.savefig(os.path.join(FIGDIR, name), dpi=180, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def heatmap(ax, data, rows, cols, title, vmax=None, cmap="YlOrRd", fmt="{:.2f}"):
    vmax = vmax or max(max(r) for r in data)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5)
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 * vmax else "black")
    ax.set_title(title, fontsize=10.5, weight="bold")
    return im


# ------------------------------------------------------------------ build
with PdfPages(OUT_PDF) as pdf:
    # 1. Title and summary
    cx0, cxc = CX.get("No shield"), CX.get("Rear-aware +\ncommit zone")
    cxs = CX.get("Shield")
    text_page(pdf, "Shielded PPO Extension: Results", [
        ("p", "Muntasir Hossain (210041265). Thesis: Navigating Unsignalised Intersections in Frenet "
              "Space with PPO and Attention, CSE 4700, Islamic University of Technology. Scope: "
              "Section 11.2 of the pre-defence report."),
        ("h", "Summary"),
        ("p", "1. A three-layer post-decision safety shield (path-based time-to-conflict braking, an "
              "RSS safe-distance override, right-before-left yielding) was implemented on the "
              "attention-based PPO controller, with every intervention counted per layer."),
        ("p", f"2. In the pre-defence environment, used for both training and evaluation, the original "
              f"shield increased collisions from {cx0['col']:.2f}% to {cxs['col']:.2f}% "
              f"(p = {fisher(cx0['c'], cxs['c'], cx0['n'], cxs['n']):.3f}). Background vehicles there "
              f"do not brake, so shield braking causes rear-end collisions. Combining rear-aware "
              f"braking with a commit zone removed that harm: {cxc['col']:.2f}%, identical to the "
              f"controller without a shield, and significantly safer than the original shield."),
        ("p", "3. On the pre-defence benchmark, with the revised training pipeline, the shielded "
              "controller recorded 0.40% collisions against 0.88% reported for Attention + Continuous "
              "in the pre-defence study. Against identical weights without the shield (0.99%) the "
              "reduction is not statistically significant (p = 0.18)."),
        ("p", "4. The shield costs efficiency in every setting: longer travel time and a lower "
              "success rate, with the RSS layer producing most interventions."),
        ("p", "5. Attention-based controllers significantly outperform heuristic controllers "
              "(p < 0.001), confirming the central pre-defence finding."),
        ("p", "6. A first attempt at shield-aware training was stopped at 824k steps for the "
              "deadline and collided more often; with half the training budget this is inconclusive."),
        ("h", "Conclusion"),
        ("p", "A post-decision shield did not yield a measurable net safety gain for this controller. "
              "Its original design harms safety in aggressive traffic; the rear-aware and commit-zone "
              "refinements remove that harm. Full-length shield-aware training against a matched "
              "baseline is the next step."),
    ])

    # 2. Environment tables
    table_page(pdf, "Evaluation and training environments",
               ["Setting", "Pre-defence training", "Revised pipeline training", "Evaluation benchmark"],
               [["Cross traffic", "400 veh/h", "275 veh/h", "150 / 275 / 400 veh/h"],
                ["Traffic in agent's lane", "275 veh/h", "none", "none"],
                ["Background speed_mode", "0 (ignores safety)", "31", "31"],
                ["RL spawn probability", "0.3", "0.8", "0.8"],
                ["Warmup", "5 steps", "50 steps", "50 steps"],
                ["Episodes", "-", "-", "42 per configuration"]],
               note="The evaluation benchmark is identical to the pre-defence protocol "
                    "(src/test/v0_1_evaluate.py). Results labelled 'pre-defence environment' use the "
                    "pre-defence training environment for both training and evaluation, reproduced "
                    "from src/configs/v0_1_single_agent.py. The observation additionally includes "
                    "three leader features added after the pre-defence report.",
               col_w=[0.24, 0.24, 0.26, 0.26])

    # 3. Heatmaps: pre-defence reported vs revised pipeline
    fig, axes = plt.subplots(2, 1, figsize=A4, gridspec_kw=dict(hspace=0.45, top=0.93, bottom=0.12))
    names = list(PREDEF)
    heatmap(axes[0], [PREDEF[k] for k in names], names, SCEN_LBL,
            "Pre-defence study (reported, Figure 9.2)", vmax=8.5)
    names2 = [k for k in STD if STD[k]]
    im = heatmap(axes[1], [[STD[k]["by_sc"][s] for s in SCEN] for k in names2], names2, SCEN_LBL,
                 "This work, revised pipeline, same benchmark", vmax=8.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Collision rate (%)")
    fig.suptitle("Collision rate by controller and traffic scenario", fontsize=13, weight="bold")
    fig_page(pdf, fig, "heatmap_scenarios.png",
             "Collision rate (%) averaged over intention settings. Top: values reported in the "
             "pre-defence report. Bottom: this work on the same evaluation benchmark.")

    # 4. Cross-study bar
    cross = [(k, statistics.mean(v), "p") for k, v in PREDEF.items()]
    if STD.get("Att + Cont + Shield"):
        cross.append(("Attention + Continuous + Shield",
                      statistics.mean(STD["Att + Cont + Shield"]["by_sc"][s] for s in SCEN), "t"))
    cross.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(A4[0], 5.5))
    ax.barh([k for k, _, _ in cross], [v for _, v, _ in cross],
            color=[GREY if s == "p" else GREEN for *_, s in cross])
    for y, (_, v, _) in enumerate(cross):
        ax.text(v + 0.08, y, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_xlim(0, 7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean collision rate (%)")
    ax.set_title("Pre-defence benchmark: reported baselines and this work", weight="bold")
    plt.subplots_adjust(left=0.32, bottom=0.22)
    fig_page(pdf, fig, "cross_study.png",
             "Grey: pre-defence report. Green: shielded controller from this work, trained in the "
             "revised pipeline. The 0.40% vs 0.88% difference is not statistically significant.")

    # 5. CI plot, revised pipeline
    items = sorted([(k, v) for k, v in STD.items() if v], key=lambda kv: -kv[1]["col"])
    fig, ax = plt.subplots(figsize=(A4[0], 5))
    for y, (k, s) in enumerate(items):
        lo, hi = wilson(s["c"], s["n"])
        col = GREEN if "Shield" in k else (GREY if "Heuristic" in k else BLUE)
        ax.plot([lo, hi], [y, y], color=col, lw=3)
        ax.plot(s["col"], y, "o", color=col, ms=7)
        ax.text(hi + 0.1, y, f"{s['col']:.2f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("Collision rate (%) with 95% confidence interval")
    ax.set_title("Two tiers: heuristic vs attention-based controllers", weight="bold")
    plt.subplots_adjust(left=0.3, bottom=0.25)
    fig_page(pdf, fig, "ci_revised.png",
             "Revised pipeline, 1,008 episodes per controller. Heuristic controllers are significantly "
             "worse (p < 0.001); the attention-based controllers overlap and are not distinguishable.")

    # 6. Shield on/off on the benchmark + override breakdown
    off, on = STD.get("Attention + Continuous"), STD.get("Att + Cont + Shield")
    if off and on:
        fig, axes = plt.subplots(2, 2, figsize=A4, gridspec_kw=dict(hspace=0.45, wspace=0.35, top=0.92, bottom=0.12))
        for ax, key, lbl in [(axes[0, 0], "col", "Collision rate (%)"), (axes[0, 1], "succ", "Success rate (%)"),
                             (axes[1, 0], "tt", "Travel time (s)")]:
            vals = [off[key], on[key]]
            ax.bar(["Shield off", "Shield on"], vals, color=[BLUE, GREEN])
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(lbl, fontsize=10)
            if key == "succ":
                ax.set_ylim(90, 100)
        parts = [("TTC", on["ttc"]), ("RSS", on["rss"]), ("Right-of-way", on["row"])]
        axes[1, 1].bar([p[0] for p in parts], [p[1] for p in parts], color=[ORANGE, RED, BLUE])
        for i, (_, v) in enumerate(parts):
            axes[1, 1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        axes[1, 1].set_title("Shield overrides by layer", fontsize=10)
        fig.suptitle("Same policy weights, shield off vs on (benchmark)", fontsize=13, weight="bold")
        fig_page(pdf, fig, "shield_on_off_benchmark.png",
                 f"Collisions {off['c']} vs {on['c']} of 1,008 (p = {fisher(off['c'], on['c'], off['n'], on['n']):.2f}). "
                 "The RSS layer produces most interventions; the right-of-way layer rarely fires.")

    # 7. Pre-defence environment: bars with CI
    if CX:
        fig, ax = plt.subplots(figsize=(A4[0], 5.5))
        keys = list(CX)
        vals = [CX[k]["col"] for k in keys]
        errs = [[CX[k]["col"] - wilson(CX[k]["c"], CX[k]["n"])[0] for k in keys],
                [wilson(CX[k]["c"], CX[k]["n"])[1] - CX[k]["col"] for k in keys]]
        cols = [BLUE] + [GREEN if "commit" in k else (GREY if "training" in k else RED) for k in keys[1:]]
        ax.bar(keys, vals, yerr=errs, capsize=5, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=9)
        ax.set_ylabel("Collision rate (%) with 95% CI")
        ax.set_title("Pre-defence environment: training and evaluation", weight="bold")
        ax.tick_params(axis="x", labelsize=8.5)
        plt.subplots_adjust(bottom=0.3)
        lines = [f"{k.replace(chr(10), ' ')}: {CX[k]['c']}/{CX[k]['n']}"
                 + (f", p = {fisher(CX['No shield']['c'], CX[k]['c'], CX['No shield']['n'], CX[k]['n']):.3f} vs no shield"
                    if k != "No shield" else "") for k in keys]
        fig_page(pdf, fig, "predefence_env_bars.png",
                 "; ".join(lines) + ". Shield-aware training was stopped at 824k steps against 1.5M.")

        # 8. Heatmap intention x variant
        fig, ax = plt.subplots(figsize=(A4[0], 5.5))
        data = [[CX[k]["by_int"].get(i, 0) for i in INT] for k in keys]
        im = heatmap(ax, data, [k.replace("\n", " ") for k in keys], INT_LBL,
                     "Collision rate (%) by intention, pre-defence environment", fmt="{:.1f}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Collision rate (%)")
        plt.subplots_adjust(left=0.3, bottom=0.25)
        fig_page(pdf, fig, "heatmap_predefence_intentions.png",
                 "Left turns are hardest and deteriorate most under shield braking, because the agent "
                 "is struck from behind by background vehicles that do not brake.")

        # 9. Safety-efficiency trade-off
        fig, ax = plt.subplots(figsize=(A4[0], 6))
        for k, s in CX.items():
            ax.scatter(s["tt"], s["col"], s=120, color=GREEN if "commit" in k else (BLUE if k == "No shield" else RED))
            ax.annotate(k.replace("\n", " "), (s["tt"], s["col"]), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
        if off and on:
            for lbl, s in [("Benchmark, no shield", off), ("Benchmark, shield", on)]:
                ax.scatter(s["tt"], s["col"], s=90, marker="s", color=GREY)
                ax.annotate(lbl, (s["tt"], s["col"]), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
        ax.set_xlabel("Average travel time (s)")
        ax.set_ylabel("Collision rate (%)")
        ax.set_title("Safety vs efficiency", weight="bold")
        ax.grid(alpha=0.3)
        plt.subplots_adjust(bottom=0.2)
        fig_page(pdf, fig, "tradeoff.png",
                 "Lower-left is better. Circles: pre-defence environment. Squares: benchmark. "
                 "Shield variants move right (slower); only the rear-aware + commit-zone variant avoids "
                 "moving up (more collisions).")

    # 10. High-flow
    if ST:
        fig, ax = plt.subplots(figsize=(A4[0], 5))
        keys = list(ST)
        ax.bar(keys, [ST[k]["col"] for k in keys],
               color=[GREY if "Heuristic" in k else (GREEN if "Shield" in k else BLUE) for k in keys])
        for i, k in enumerate(keys):
            ax.text(i, ST[k]["col"] + 0.05, f"{ST[k]['col']:.2f}%", ha="center", fontsize=9)
        ax.set_ylabel("Collision rate (%)")
        ax.set_title("High-flow traffic (550 / 700 / 850 veh/h)", weight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        plt.subplots_adjust(bottom=0.3)
        fig_page(pdf, fig, "highflow.png",
                 "Higher nominal flow did not raise collision rates, most likely because congested "
                 "approaches limit how many vehicles SUMO inserts. The shield did not reduce collisions here.")

    # 11. Limitations
    text_page(pdf, "Limitations and future work", [
        ("h", "Limitations"),
        ("p", "Sample size: at collision rates below 1%, differences of a few collisions cannot be "
              "resolved with 1,008 episodes."),
        ("p", "Training environment: benchmark results use a revised pipeline whose training "
              "environment is easier than the pre-defence one. The pre-defence environment results "
              "remove this difference for Attention + Continuous only."),
        ("p", "Shield variants in the pre-defence environment (rear-aware, commit zone) were designed "
              "after the original shield's failures were observed. All variants run are reported."),
        ("p", "Shield-aware training was stopped at 824k steps against 1.5M for the unshielded model, "
              "and no matched unshielded checkpoint was available."),
        ("p", "Each controller was trained with a single seed."),
        ("h", "Future work"),
        ("p", "Full-length shield-aware training against a matched baseline; calibrating the RSS "
              "trigger distance to the perception radius; multiple seeds and larger evaluation budgets; "
              "stress scenarios that raise conflict density rather than nominal inflow."),
        ("h", "Reproducibility"),
        ("p", "All numbers are generated from eval_results/ by scripts/make_report.py and "
              "scripts/make_pdf.py. Setup and every option are documented in SETUP.md."),
    ])

print("wrote", OUT_PDF)
