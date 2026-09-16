#!/usr/bin/env python3
"""
Build docs/Shielded_PPO_Explained.pdf: a three-page plain-language explanation
for the supervisor. All text is black and bold.

    python scripts/make_explainer.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "Shielded_PPO_Explained.pdf")
A4 = (8.27, 11.69)
BLACK = "#000000"
L, R, TOP, BOT = 0.085, 0.93, 0.945, 0.05
LH = 0.0169                                   # line height


class Page:
    def __init__(self, pdf, title):
        self.pdf = pdf
        self.fig = plt.figure(figsize=A4)
        self.fig.patch.set_facecolor("white")
        self.y = TOP
        self.fig.text(L, self.y, title, fontsize=17, weight="bold", color=BLACK, va="top")
        self.y -= 0.034
        self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y], color=BLACK, lw=1.4))
        self.y -= 0.016

    def h2(self, text):
        self.y -= 0.006
        self.fig.text(L, self.y, text, fontsize=12.5, weight="bold", color=BLACK, va="top")
        self.y -= 0.024

    def para(self, text, width=82, size=9.5):
        for line in textwrap.wrap(text, width):
            self.fig.text(L, self.y, line, fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.006

    def bullet(self, text, width=78, size=9.5):
        for i, line in enumerate(textwrap.wrap(text, width)):
            self.fig.text(L + (0.0 if i == 0 else 0.017), self.y,
                          ("•  " if i == 0 else "") + line,
                          fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.004

    def kv(self, key, val, kw=0.30, width=52, size=9.5):
        self.fig.text(L, self.y, key, fontsize=size, weight="bold", color=BLACK, va="top")
        for line in textwrap.wrap(val, width):
            self.fig.text(L + kw, self.y, line, fontsize=size, weight="bold", color=BLACK, va="top")
            self.y -= LH
        self.y -= 0.004

    def close(self, num):
        # Measure what was actually rendered: wrapping by character count is only
        # an estimate, so check every line against the right margin.
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        worst = 0.0
        for t in self.fig.texts:
            x1 = t.get_window_extent(rend).transformed(self.fig.transFigure.inverted()).x1
            worst = max(worst, x1)
        if worst > R + 0.004:
            bad = [t.get_text()[:70] for t in self.fig.texts
                   if t.get_window_extent(rend).transformed(
                       self.fig.transFigure.inverted()).x1 > R + 0.004]
            print(f"  WARNING: page {num} text runs {worst - R:.3f} past the right margin")
            for b in bad[:4]:
                print(f"      offending: {b!r}")
        if self.y < BOT:
            print(f"  WARNING: page {num} overflows by {BOT - self.y:.3f}")
        else:
            print(f"  page {num}: {self.y - BOT:.3f} spare")
        self.fig.text(R, 0.022, f"page {num} of 3", fontsize=9.5, weight="bold",
                      color=BLACK, ha="right")
        self.pdf.savefig(self.fig, facecolor="white")
        plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ------------------------------------------------------------- page 1
    p = Page(pdf, "Shielded PPO: what we built and how it works")
    p.para("Muntasir Hossain (210041265). Section 11.2 of the pre-defence report. "
           "Prepared for supervisor review.")

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

    p.h2("4.  Action space: what the car controls")
    p.kv("Continuous", "one number in [-1, +1]; positive is multiplied by 2.6 m/s2 to "
                              "accelerate, negative by 4.5 m/s2 to brake.")
    p.kv("Discrete", "five fixed levels (-1, -0.5, 0, +0.5, +1), scaled the same way: "
                                "coarser but easier to learn.")
    p.para("The car controls acceleration only; it does not steer.")
    p.close(1)

    # ------------------------------------------------------------- page 2
    p = Page(pdf, "The learning problem: inputs, outputs and reward")

    p.h2("5.  State space: what the car sees (37 numbers)")
    p.para("All values are normalised. The car sees vehicles within 50 metres and keeps the five "
           "closest.")
    p.kv("Own state (4)", "distance left to the exit, speed, heading as sin and cos.")
    p.kv("Car ahead (3)", "gap, its speed, time-to-collision with it.")
    p.kv("Neighbours (5 x 5)", "for each: distance to the point where the paths cross, its "
                                    "speed, the difference in arrival time at that point, and "
                                    "relative heading as sin and cos.")
    p.kv("Mask (5)", "which neighbour slots hold a real car, so empty slots are ignored.")
    p.para("The attention layer learns which nearby car matters most at that moment, instead of "
           "treating all five equally.")

    p.h2("6.  Reward function")
    p.kv("End of episode", "collision -10, reaching the exit +15.")
    p.kv("Every step", "10 x progress along the route, plus the safety penalty, minus 0.01 for time.")
    p.kv("Safety penalty", "-exp(-10 x |arrival-time difference|) for each neighbour due at the "
                           "conflict point at nearly the same moment. It rises steeply as the two "
                           "arrival times converge, so near-misses are punished well before a "
                           "crash. The same applies to the car ahead when time-to-collision is "
                           "small.")
    p.kv("Shielded version", "a further -0.005 when the shield overrides more than half of recent "
                             "steps, to discourage leaning on it.")

    p.h2("7.  Training and evaluation setup")
    p.kv("Algorithm", "PPO (stable-baselines3); attention encoder producing 256 values, then "
                      "actor and critic of 256 x 256.")
    p.kv("Hyper-parameters", "learning rate 3e-4 to 1e-5, batch 8192, minibatch 256, 10 epochs, "
                            "discount 0.98, GAE 0.95, clip 0.25, entropy 0.01.")
    p.kv("Training", "1.5 million simulation steps per controller, 8 parallel simulators.")
    p.kv("Evaluation", "6 traffic levels x 4 turning patterns x 42 episodes = 1,008 episodes per "
                       "controller, policy acting deterministically. Same protocol as the "
                       "pre-defence report, so the numbers are comparable.")
    p.kv("Measured", "collision rate, success rate, travel time, waiting time, and interventions "
                     "per shield layer.")
    p.close(2)

    # ------------------------------------------------------------- page 3
    p = Page(pdf, "Results: where it worked and where it failed")

    p.h2("8.  Where it worked: our training setup")
    p.para("Here the surrounding traffic obeys the simulator's safety rules, so those cars brake "
           "for our car. Traffic arrives at 275 vehicles/hour on the crossing roads and the car's "
           "own lane is clear.")
    p.kv("Shield off", "0.99% of episodes ended in a collision (10 of 1,008).")
    p.kv("Shield on", "0.40% (4 of 1,008), travel time 12.9 to 16.3 seconds, 18.5% "
                                    "of actions overridden.")
    p.kv("For comparison", "heuristic controllers without attention 3.17% and 3.97%; best "
                           "controller in the pre-defence study 0.88%.")
    p.para("Why it worked: the shield's only tool is braking, and braking is safe when the cars "
           "around you yield. Slowing down removed crossing conflicts without creating new ones.")
    p.para("Limitation: 10 collisions against 4 is not statistically significant (p = 0.18). The "
           "three attention controllers all sit between 0.40% and 0.99% and cannot be separated "
           "at this sample size. The shield is in the best group, not proven the single best.")

    p.h2("9.  Where it failed: the pre-defence scenario")
    p.para("The pre-defence setup is harsher: 400 vehicles per hour on the crossing roads, traffic "
           "in the car's own lane, and background drivers that ignore the simulator's safety "
           "checks, so they never brake. We retrained the controller in that exact environment "
           "and tested it there.")
    p.kv("Shield off", "6.35% collisions (64 of 1,008).")
    p.kv("Shield on", "9.72% (98). Significantly worse, p = 0.007.")
    p.kv("Rear-aware cap", "9.42% (95). Still significantly worse, p = 0.013.")
    p.kv("+ commit zone", "6.35% (64). Back to the unshielded level, p = 1.00, with "
                                     "overrides down from 8.8% to 2.9%.")
    p.kv("Trained with shield", "12.70%, but training was cut to 824k steps by the deadline "
                                    "against 1.5M for the baseline, so this is inconclusive.")
    p.para("Why it failed: the cars behind do not brake. When the shield slows our car for a "
           "crossing car, the car behind drives into it. The damage was worst on left turns, where "
           "the shield brakes most: 11.1% without it against 23.8% with it. Capping the braking "
           "stopped some rear-end crashes but let crossing conflicts through; only the commit "
           "zone removed the harm.")

    p.h2("10.  Conclusion and next step")
    p.para("A braking-only shield added on top of a trained controller helps when the surrounding "
           "traffic cooperates and hurts when it does not. The refined shield is safe in both "
           "environments but does not improve on the controller alone.")
    p.para("The next step is to train the controller with the shield switched on from the start, "
           "so it learns to work with the interventions, using a matched training budget for a "
           "fair comparison.")
    p.close(3)

print("wrote", OUT)
