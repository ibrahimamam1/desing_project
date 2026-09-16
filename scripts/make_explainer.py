#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build docs/Shielded_PPO_Explained.pdf: a four-page plain-language explanation
for the supervisor. All text is black and bold.

Page 1  what we are doing, what the shield is, how it works
Page 2  state space, with a diagram of the 37-value observation
Page 3  action space and reward function, as formulas
Page 4  results: where it worked, where it failed, and why

    python scripts/make_explainer.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["mathtext.default"] = "bf"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "Shielded_PPO_Explained.pdf")
A4 = (8.27, 11.69)
BLACK = "#000000"
EGO_C, LEAD_C, NB_C, MASK_C = "#cfe0f3", "#f6ddc9", "#d8ecd6", "#e6e2ef"
L, R, TOP, BOT = 0.085, 0.93, 0.945, 0.05
LH = 0.0169
NPAGES = 4


class Page:
    def __init__(self, pdf, title):
        self.pdf = pdf
        self.fig = plt.figure(figsize=A4)
        self.fig.patch.set_facecolor("white")
        self.y = TOP
        self.fig.text(L, self.y, title, fontsize=16.5, weight="bold", color=BLACK, va="top")
        self.y -= 0.033
        self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y], color=BLACK, lw=1.4))
        self.y -= 0.016

    def h2(self, text):
        self.y -= 0.006
        self.fig.text(L, self.y, text, fontsize=12.5, weight="bold", color=BLACK, va="top")
        self.y -= 0.024

    def para(self, text, width=90, size=9.5):
        for line in textwrap.wrap(text, width):
            self.fig.text(L, self.y, line, fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.006

    def bullet(self, text, width=89, size=9.5):
        for i, line in enumerate(textwrap.wrap(text, width)):
            self.fig.text(L + (0.0 if i == 0 else 0.017), self.y,
                          ("•  " if i == 0 else "") + line,
                          fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.004

    def kv(self, key, val, kw=0.27, width=61, size=9.5):
        self.fig.text(L, self.y, key, fontsize=size, weight="bold", color=BLACK, va="top")
        for line in textwrap.wrap(val, width):
            self.fig.text(L + kw, self.y, line, fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.004

    def math(self, rows, size=12.0, x=0.125, xc=0.50, lh=0.024, pre=0.004, post=0.006):
        """Display formulas: rows of (formula, condition) in mathtext, paper style."""
        self.y -= pre
        for formula, cond in rows:
            if formula:
                self.fig.text(x, self.y, formula, fontsize=size, color=BLACK, va="top")
            if cond:
                self.fig.text(xc, self.y + 0.002, cond, fontsize=9.5, weight="bold",
                              color=BLACK, va="top")
            self.y -= lh if formula or cond else 0.010
        self.y -= post

    def axes(self, height):
        """Reserve vertical space and return an axes spanning the text column."""
        ax = self.fig.add_axes([L, self.y - height, R - L, height])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.y -= height + 0.010
        return ax

    def close(self, num):
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        worst = 0.0
        for t in self.fig.texts:
            x1 = t.get_window_extent(rend).transformed(self.fig.transFigure.inverted()).x1
            worst = max(worst, x1)
        if worst > R + 0.004:
            print(f"  WARNING: page {num} text runs {worst - R:.3f} past the right margin")
            for t in self.fig.texts:
                x1 = t.get_window_extent(rend).transformed(
                    self.fig.transFigure.inverted()).x1
                if x1 > R + 0.004:
                    print(f"    [{x1 - R:+.3f}] {t.get_text()[:70]!r}")
        if self.y < BOT:
            print(f"  WARNING: page {num} overflows by {BOT - self.y:.3f}")
        else:
            print(f"  page {num}: {self.y - BOT:.3f} spare")
        self.fig.text(R, 0.022, f"page {num} of {NPAGES}", fontsize=9.5, weight="bold",
                      color=BLACK, ha="right")
        self.pdf.savefig(self.fig, facecolor="white")
        plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ================================================================ page 1
    p = Page(pdf, "Shielded PPO: what we built and how it works")

    p.h2("1.  What we are doing")
    p.para("A single self-driving car has to cross a four-way junction with no traffic light, "
           "while cars keep arriving from the other three roads. It must decide, moment by "
           "moment, how hard to accelerate or brake so it crosses without hitting anyone and "
           "without waiting longer than necessary.")
    p.para("The car is driven by a neural network trained with reinforcement learning (PPO) in "
           "SUMO. The pre-defence report proposed adding a safety shield on top of it. Building, "
           "testing and reporting that shield is the work described here.")

    p.h2("2.  What the shield is")
    p.para("The shield is a set of fixed safety rules sitting between the network and the "
           "simulator. The network does not know about it: it simply proposes an acceleration. "
           "The shield checks that proposal against the nearby cars, then either passes it "
           "through unchanged or replaces it with a safer value, normally braking.")
    p.para("The aim is a safety guarantee that does not depend on learning.")

    p.h2("3.  How the shield works, step by step")
    p.para("Every simulation step, in this order:")
    p.bullet("The network outputs one number in [-1, +1], turned into an acceleration: positive "
             "accelerates, negative brakes.")
    p.bullet("Layer 1, time-to-conflict. For each nearby car the shield computes how long our car "
             "needs to reach the point where the two paths cross. If that is under 2 seconds and "
             "the other car arrives at about the same time, it brakes hard enough to stop short "
             "of that point.")
    p.bullet("Layer 2, safe distance (RSS). It adds reaction distance, braking distance and a "
             "3 metre margin. If our car is closer to the conflict point than that, it brakes.")
    p.bullet("Layer 3, right of way. If two cars would arrive together and the other comes from "
             "the right, our car yields, because the junction uses right-before-left priority.")
    p.bullet("Two later additions: the commit zone stops the shield braking once the car can no "
             "longer stop before the conflict point, and the rear-aware cap limits braking when "
             "a car is close behind.")
    p.bullet("The surviving value is sent to SUMO. Every intervention is counted per layer, so we "
             "can report how often the shield acts and which rule fired.")

    p.h2("4.  Training and evaluation setup")
    p.kv("Algorithm", "PPO (stable-baselines3); attention encoder producing 256 values, then "
                      "actor and critic of 256 x 256.")
    p.kv("Hyper-parameters", "learning rate 3e-4 to 1e-5, batch 8192, minibatch 256, 10 epochs, "
                             "discount 0.98, GAE 0.95, clip 0.25, entropy 0.01.")
    p.kv("Training", "1.5 million simulation steps per controller, 8 parallel simulators.")
    p.kv("Evaluation", "6 traffic levels x 4 turning patterns x 42 episodes = 1,008 episodes per "
                       "controller, policy deterministic. Same protocol as the pre-defence report.")
    p.close(1)

    # ================================================================ page 2
    p = Page(pdf, "State space: what the car sees (37 numbers)")
    p.para("The network receives one flat vector of 37 numbers each step. Every value is "
           "normalised, so they all sit roughly between -1 and +1. The car looks 50 metres around "
           "itself and keeps the five closest vehicles.")

    ax = p.axes(0.118)
    n = 37
    w = 1.0 / n
    for a, b, c, lab in [(0, 4, EGO_C, "ego (4)"), (4, 7, LEAD_C, "leader (3)"),
                         (7, 32, NB_C, "5 neighbours x 5 values = 25"), (32, 37, MASK_C, "mask (5)")]:
        ax.add_patch(Rectangle((a * w, 0.40), (b - a) * w, 0.32, fc=c, ec=BLACK, lw=1.3))
        ax.text((a + b) / 2 * w, 0.78, lab, ha="center", va="bottom", fontsize=9,
                weight="bold", color=BLACK)
        ax.text((a + b) / 2 * w, 0.33, f"index {a}-{b - 1}", ha="center", va="top",
                fontsize=8, weight="bold", color=BLACK)
    for i in range(1, n):
        ax.plot([i * w, i * w], [0.40, 0.72], color="#999999", lw=0.5, zorder=0)
    for i in range(7, 32, 5):
        ax.plot([i * w, i * w], [0.40, 0.72], color=BLACK, lw=1.3)
    for k in range(5):
        ax.text((7 + 5 * k + 2.5) * w, 0.56, f"n{k + 1}", ha="center", va="center",
                fontsize=8, weight="bold", color=BLACK)
    ax.text(0.5, 0.04, "one observation vector, 37 values, fed to the attention encoder",
            ha="center", va="bottom", fontsize=8.5, weight="bold", color=BLACK)

    p.kv("ego (4 values)", "distance still to drive to the exit as a fraction of the route, "
                           "speed / 55, and our heading as sin and cos.")
    p.kv("leader (3 values)", "the car directly ahead in our own lane: gap / 50 m, its speed / 55, "
                              "and time-to-collision with it. All three are 1.0 when the lane "
                              "ahead is empty.")
    p.kv("each neighbour (5)", "distance to the point where our two paths cross (/ 50 m), its "
                               "speed / 55, the difference in arrival time at that point, and its "
                               "heading relative to ours as sin and cos.")
    p.kv("mask (5 values)", "1 if that neighbour slot holds a real car, 0 if it is empty padding. "
                            "The attention layer ignores the 0 slots.")

    p.h2("The two values that make this work")
    p.para("Each neighbour is not described by its raw position on the map, but by where our two "
           "routes cross and when each of us gets there:")

    ax = p.axes(0.300)
    ROAD = "#e9edf2"
    BLUE, RED, ORG = "#1f5fa8", "#c62828", "#e06c00"
    # the two roads
    ax.add_patch(Rectangle((0.02, 0.44), 0.96, 0.15, fc=ROAD, ec="#b9c2cc", lw=0.8))
    ax.add_patch(Rectangle((0.50, 0.44), 0.10, 0.54, fc=ROAD, ec="#b9c2cc", lw=0.8))
    ax.plot([0.02, 0.98], [0.515, 0.515], color="#ffffff", lw=1.4, ls=(0, (6, 6)), zorder=1)
    ax.plot([0.55, 0.55], [0.59, 0.98], color="#ffffff", lw=1.4, ls=(0, (6, 6)), zorder=1)
    # our car, travelling east
    ax.add_patch(Rectangle((0.10, 0.465), 0.055, 0.075, fc=BLUE, ec=BLUE, zorder=3))
    ax.text(0.128, 0.615, "our car", ha="center", va="bottom", fontsize=9.5, weight="bold",
            color=BLUE)
    ax.plot([0.155, 0.545], [0.503, 0.503], color=BLUE, lw=2.0, ls="--", zorder=2)
    ax.annotate("", xy=(0.545, 0.503), xytext=(0.49, 0.503),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=BLUE))
    # the neighbour, travelling south
    ax.add_patch(Rectangle((0.525, 0.80), 0.055, 0.085, fc=RED, ec=RED, zorder=3))
    ax.text(0.60, 0.845, "neighbour", ha="left", va="center", fontsize=9.5, weight="bold",
            color=RED)
    ax.plot([0.552, 0.552], [0.80, 0.53], color=RED, lw=2.0, ls="--", zorder=2)
    ax.annotate("", xy=(0.552, 0.53), xytext=(0.552, 0.60),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=RED))
    # the conflict point
    ax.plot([0.552], [0.503], marker="X", ms=16, color=ORG, zorder=4)
    ax.annotate("conflict point:\nwhere the two routes cross",
                xy=(0.565, 0.495), xytext=(0.64, 0.66), fontsize=9.5, weight="bold",
                color=ORG, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=1.2, color=ORG))
    # value 1: distance to the conflict point
    ax.annotate("", xy=(0.548, 0.33), xytext=(0.128, 0.33),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.4, color=BLACK))
    ax.plot([0.128, 0.128], [0.33, 0.465], color="#777777", lw=0.8, ls=":")
    ax.plot([0.552, 0.552], [0.33, 0.49], color="#777777", lw=0.8, ls=":")
    ax.text(0.338, 0.295, "value 1:  distance along our route to the conflict point  (/ 50 m)",
            ha="center", va="top", fontsize=9.5, weight="bold", color=BLACK)
    # value 2: arrival-time difference
    ax.text(0.0, 0.17, "value 2:  arrival-time difference   =   (our time to reach the X)   -   "
                       "(their time to reach the X)",
            fontsize=9.5, weight="bold", color=BLACK, va="top")
    ax.text(0.03, 0.085, "close to 0  ->  both cars arrive together, a collision course",
            fontsize=9.5, weight="bold", color=BLACK, va="top")
    ax.text(0.03, 0.015, "large  ->  one car is safely through before the other arrives",
            fontsize=9.5, weight="bold", color=BLACK, va="top")

    p.y -= 0.016
    p.para("Measuring the distance to the shared conflict point rather than the straight-line "
           "distance between the cars is what makes left turns work: two cars can be far apart "
           "and still head for the same spot at the same instant. These two values also feed the "
           "shield's rules and the reward's safety term.")
    p.close(2)

    # ================================================================ page 3
    p = Page(pdf, "Action space and reward function, as implemented")

    p.h2("5.  Action space")
    p.para("The action is one continuous number per step; the car does not steer, and its route "
           "is fixed when it enters the network.")
    p.math([(r"$a_t \ \in \ \mathcal{A} \ = \ [\,-1,\ +1\,] \ \subset \ \mathbb{R}$",
             "one continuous value per step")])
    p.para("That number is mapped to a physical acceleration with two different scales, because "
           "the car can brake harder than it can accelerate:")
    p.math([(r"$u_t \ = \ a_{max}\,\cdot\,a_t \ = \ 2.6\,a_t$", r"if  $a_t \geq 0$   (accelerate)"),
            (r"$u_t \ = \ b_{max}\,\cdot\,a_t \ = \ 4.5\,a_t$", r"if  $a_t < 0$   (brake)")])
    p.para(u"So a = +1 is full acceleration at 2.6 m/s\u00b2, a = -1 is full emergency braking "
           u"at 4.5 m/s\u00b2, and a = 0 holds speed. The command u is exactly what the shield "
           u"inspects, and may replace, before the car executes it.")
    p.kv("Discrete version", "for comparison we also trained a five-way action, a in "
                             "{-1, -0.5, 0, +0.5, +1}, mapped with the same two scales. Coarser "
                             "control, but an easier learning problem.")

    p.h2("6.  Reward function")
    p.para("The reward paid at each step is:")
    p.math([(r"$r_t \ = \ -10$", "if a collision occurs (episode ends)"),
            (r"$r_t \ = \ +15$", "if the car reaches its exit (episode ends)"),
            (r"$r_t \ = \ 10\,\Delta p_t \ + \ s_t \ - \ 0.01$", "otherwise, at every step")])
    p.para("with the progress term and the safety term defined as:")
    p.math([(r"$\Delta p_t \ = \ p_t \ - \ p_{t-1}$",
             "fraction of the route covered this step"),
            (r"$s_t \ = \ -\sum_{j \in N_t} e^{-10\,|\Delta\tau_j|} \ - \ e^{-10\,TTC_t}$",
             r"first sum over neighbours with $|\Delta\tau_j| < 0.2$,"),
            (None, r"second term only if $TTC_t < 0.2$"),
            (None, None)])
    p.kv(u"Meaning of the terms",
         u"\u0394p is progress towards the exit, \u0394\u03c4 for neighbour j is the gap between "
         u"our arrival time and theirs at their shared conflict point, and TTC is the "
         u"time-to-collision with the car directly ahead.")
    p.kv("-10  and  +15", "the only large rewards: crashing must be worse than being slow, and "
                          "reaching the exit is the goal.")
    p.kv("Progress term", "a full crossing sums to about +10, so the car is rewarded for "
                          "approaching the exit rather than for raw speed.")
    p.kv("Safety term", "-1.0 per neighbour when the two arrival times are identical, about -0.14 "
                        "at a difference of 0.2, and zero beyond it. This punishes near-misses long "
                        "before any crash happens.")
    p.kv("Time cost", "the constant -0.01 per step stops the car from waiting at the junction "
                      "indefinitely.")
    p.kv("Shielded version", "a further -0.005 whenever the shield has overridden more than half of "
                             "the recent steps, to discourage leaning on it.")
    p.close(3)

    # ================================================================ page 4
    p = Page(pdf, "Results: where it worked and where it failed")

    p.h2("7.  Where it worked: our training setup")
    p.para("Here the surrounding traffic obeys the simulator's safety rules, so those cars brake "
           "for our car. Traffic arrives at 275 vehicles/hour on the crossing roads and the car's "
           "own lane is clear.")
    p.kv("Shield off", "0.99% of episodes ended in a collision (10 of 1,008).")
    p.kv("Shield on", "0.40% (4 of 1,008), travel time 12.9 to 16.3 seconds, 18.5% of actions "
                      "overridden.")
    p.kv("For comparison", "heuristic controllers without attention 3.17% and 3.97%; best "
                           "controller in the pre-defence study 0.88%.")
    p.para("Why it worked: the shield's only tool is braking, and braking is safe when the cars "
           "around you yield. Slowing down removed crossing conflicts without creating new ones.")
    p.para("Limitation: 10 collisions against 4 is not statistically significant (p = 0.18). All "
           "three attention controllers sit between 0.40% and 0.99% and cannot be separated at "
           "this sample size, so the shield is in the best group but not proven the best.")

    p.h2("8.  Where it failed: the pre-defence scenario")
    p.para("That setup is harsher: 400 vehicles/hour on the crossing roads, traffic in the car's "
           "own lane, and background drivers that ignore the simulator's safety checks, so they "
           "never brake. We retrained and tested the controller in that environment.")
    p.kv("Shield off", "6.35% collisions (64 of 1,008).")
    p.kv("Shield on", "9.72% (98). Significantly worse, p = 0.007.")
    p.kv("Rear-aware cap", "9.42% (95). Still significantly worse, p = 0.013.")
    p.kv("+ commit zone", "6.35% (64). Back to the unshielded level, p = 1.00, with overrides down "
                          "from 8.8% to 2.9%.")
    p.kv("Trained with shield", "12.70%, but training was cut to 824k steps by the deadline against "
                                "1.5M for the baseline, so this is inconclusive.")
    p.para("Why it failed: the cars behind do not brake. When the shield slows our car for a "
           "crossing car, the car behind drives into it. The damage was worst on left turns, where "
           "the shield brakes most: 11.1% without it against 23.8% with it. Only the commit zone, "
           "which stops the shield braking inside the junction, removed the harm.")

    ax = p.axes(0.150)
    ROAD2 = "#e9edf2"
    ax.add_patch(Rectangle((0.02, 0.46), 0.96, 0.26, fc=ROAD2, ec="#b9c2cc", lw=0.8))
    ax.add_patch(Rectangle((0.58, 0.46), 0.10, 0.54, fc=ROAD2, ec="#b9c2cc", lw=0.8))
    # the crossing car, which never brakes
    ax.add_patch(Rectangle((0.605, 0.80), 0.05, 0.14, fc="#c62828", ec="#c62828", zorder=3))
    ax.plot([0.630, 0.630], [0.80, 0.66], color="#c62828", lw=1.8, ls="--", zorder=2)
    ax.text(0.70, 0.86, "crossing car: never brakes", ha="left", va="center", fontsize=9.5,
            weight="bold", color="#c62828")
    # our car, braking because of the shield
    ax.add_patch(Rectangle((0.40, 0.52), 0.065, 0.14, fc="#1f5fa8", ec="#1f5fa8", zorder=3))
    ax.text(0.4325, 0.42, "our car: the shield brakes", ha="center", va="top", fontsize=9.5,
            weight="bold", color="#1f5fa8")
    # the follower, which also never brakes
    ax.add_patch(Rectangle((0.10, 0.52), 0.065, 0.14, fc="#424242", ec="#424242", zorder=3))
    ax.text(0.1325, 0.42, "car behind", ha="center", va="top", fontsize=9.5,
            weight="bold", color="#424242")
    ax.annotate("", xy=(0.385, 0.59), xytext=(0.172, 0.59),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#424242"))
    ax.plot([0.393], [0.59], marker="*", ms=19, color="#e06c00", zorder=4)
    ax.annotate("rear-end collision", xy=(0.390, 0.66), xytext=(0.25, 0.87),
                fontsize=9.5, weight="bold", color="#e06c00", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=1.1, color="#e06c00"))
    ax.text(0.0, 0.20, "The shield can only brake, and braking is safe only if the car behind "
                       "brakes too.",
            fontsize=9.5, weight="bold", color=BLACK, va="top")
    ax.text(0.0, 0.05, "It does in our setup; in the pre-defence setup it does not, so the "
                       "crash moves to the rear.",
            fontsize=9.5, weight="bold", color=BLACK, va="top")

    p.h2("9.  Conclusion and next step")
    p.para("A braking-only shield added on top of a trained controller helps when the "
           "surrounding traffic cooperates and hurts when it does not. The refined shield is safe "
           "in both environments but does not improve on the controller alone. The next step is to "
           "train with the shield on from the start, on a matched budget for a fair comparison.")
    p.close(4)

print("wrote", OUT)
