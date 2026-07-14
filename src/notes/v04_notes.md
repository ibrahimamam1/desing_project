# v0_4 — Reward Routing Fix

Branched from **v0_2** (not v0_3, to avoid tangling with the Lagrangian safety layer).

## Problem
v0_2's evaluation showed elevated collision rates in some scenario/intention combos
(directional collision issue), despite good training curves.

## Root cause (hypothesis)
v0_2 used a two-stream reward split (`r_traj` = long-horizon, `r_cruise` = short-horizon,
each with its own critic/GAE). But:
- `progress_delta` (dense, per-step) lived in `r_traj` — wrong stream for a per-step signal.
- `fail` (crash) only fed `r_cruise`, giving the crash penalty no long-horizon credit
  assignment, even though the causal decision behind a collision can happen many steps
  before `r_cruise`'s short (~33-step) horizon reaches back.

## Changes applied (in `alpha_env_v04*.py`, all 3 files with their own `compute_reward()`)

**Edit 1 — fail block routes to long-horizon only:**
```python
if fail:
    self.last_r_traj, self.last_r_cruise = -10.0, 0.0
    return -10.0
if goal_reached:
    self.last_r_traj, self.last_r_cruise = 15.0, 0.0
    return 15.0
```
Makes `fail` symmetric with `goal_reached` — both terminal/sparse events go to `r_traj`
only. Avoids injecting a large discrete -10 spike into `r_cruise`, which is meant to stay
dense and small-magnitude (so it doesn't distort short-horizon GAE near crashes).

**Edit 2 — stream routing swap:**
```python
r_traj   = 0.0                                                   # reserved for terminal outcomes only
r_cruise = 10.0 * progress_delta + 1.0 * safety_penalty - 0.01   # dense per-step signal
```
Moves `progress_delta` into `r_cruise` (where dense signals belong) and leaves `r_traj`
purely for terminal outcomes (`fail`, `goal_reached`), letting the high-γ long-horizon
GAE propagate the crash penalty back to the actual upstream decision.

`alpha_env_v04_attention_discrete.py` — no edit, inherits from `alpha_env_v04_attention_continous.py`.

## Why branch from v0_2, not v0_3
v0_3 adds a Lagrangian safety constraint — a separate mechanism. This experiment is only
about reward-stream routing, so branching from v0_2 isolates the one variable being tested.

## Comparison structure
| Version | Description |
|---|---|
| v01 | untouched baseline |
| v02 | multi-discount GAE, fail penalty split `(0, -10)` — untouched, live comparison |
| v03 | Lagrangian safety, built on v02 |
| v04 | fail penalty `(-10, 0)` + progress_delta moved to r_cruise — this experiment |

## Status
Edits applied and grep-verified in all v04 env files. Imports in `v0_4_single_agent.py`
and `v0_4_evaluate.py` updated to point at v04 modules. v0_2 left untouched as baseline.
Training in progress (`heuristic_discrete`, started 2026-07-13 19:56).
