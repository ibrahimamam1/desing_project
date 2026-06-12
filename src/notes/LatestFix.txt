# v03 — Lagrangian Safety (Single Agent PPO)

**Branch:** `shakkhar_2.0_multi_discount_single_agent_clean`
**Date:** May 14, 2026
**Built on top of:** v02 (Multi-Discount GAE)

---

## What Was Built

Lagrangian safety constraint added to single-agent PPO:

```
L_safe = L_PPO − λc · C(s, a)
Ct = c1 · 1[|d_eta| < τ_crit] + c2 · 1[collision]
```

λc auto-adjusts every update:
- If mean Ct > 0 → λc increases (more safety pressure)
- If mean Ct = 0 → λc decreases (relax, optimize efficiency)
- λc clipped to [0, ∞)

Reference: Achiam et al., 2017 (CPO) — thesis page 19-20.

---

## Files

| Action | File |
|--------|------|
| Modified | `src/envs/base_env_single.py` — Ct injected into infos |
| New | `src/ppo/lagrangian_buffer.py` — stores Ct per step |
| New | `src/ppo/lagrangian_ppo.py` — λc logic, modified loss |
| New | `src/configs/v0_3_single_agent.py` — wires everything |

**Untouched:** all v01 files, all v02 files, `v0_2_single_agent.py`

---

## Architecture

```
base_env_single.py step()
    → computes Ct from last_neighbors_info + crashed flag
    → infos["Ct"] = Ct  (v01/v02 never read this — zero effect on them)

LagrangianPPO.collect_rollouts()
    → reads infos["Ct"]
    → calls rollout_buffer.add_cost(pos, ct_arr)
    → also reads r_cruise, r_traj (inherited from MultiDiscountPPO)

LagrangianRolloutBuffer
    → inherits dual GAE from MultiDiscountRolloutBuffer
    → adds self.costs array
    → get_mean_cost() returns scalar for λc update

LagrangianPPO.train()
    → calls super().train()  (full standard PPO update)
    → then adjusts λc based on mean_cost
    → logs lagrangian/lambda_c and lagrangian/mean_cost to tensorboard
```

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| c1 | 1.0 | Cost per dangerous neighbor (\|d_eta\| < 0.2) |
| c2 | 5.0 | Cost per collision |
| tau_crit | 0.2 | d_eta danger threshold (validated in thesis) |
| lambda_c | 0.1 | Initial Lagrange multiplier |
| lambda_lr | 0.01 | λc learning rate |
| gamma1 | 0.90 | Short horizon discount (safety) — inherited from v02 |
| gamma2 | 0.99 | Long horizon discount (progress) — inherited from v02 |
| w1 | 0.4 | Weight for safety advantage — inherited from v02 |
| w2 | 0.6 | Weight for progress advantage — inherited from v02 |

---

## Run Commands

```bash
# Training
python src/configs/v0_3_single_agent.py --train --version heuristic_discrete

# Evaluation
python src/configs/v0_3_single_agent.py --eval checkpoints/v0_3/.../final_model.zip --version heuristic_discrete
```

Version options: `heuristic_discrete`, `heuristic_continuous`, `attention_discrete`, `attention_continous`

---

## Key Design Decisions

**Why `base_env_single.py` and not v02 env files:**
`crashed` and `last_neighbors_info` are both in scope there. One change covers all 4 env variants automatically. v01 and v02 are unaffected — they never read `infos["Ct"]`.

**Why no separate env files (v03 variants):**
Unlike v02 where reward split required per-variant changes in `compute_reward()`, Ct lives in the base class. No per-variant env files needed.

**Why `super().train()` first, then λc update:**
SB3's loss computation is inside the parent's `train()`. Adjusting λc after and applying next update is the standard Lagrangian PPO practice (one-step lag is invisible at 1.5M timesteps with `lambda_lr=0.01`).

**SB3 never broken:**
Environment still returns scalar reward. SubprocVecEnv works normally. PPO update loop unchanged. Dual GAE and Ct handling happen only inside the buffer and PPO subclasses.

---

## Smoke Test Results

| Metric | Value |
|--------|-------|
| Timesteps | 5120 |
| Crashes | 0 |
| ep_rew_mean (start) | -0.07 |
| ep_rew_mean (end) | +0.90 |
| success | 1 |
| lambda_c | 0.1 (stable, few violations in short run) |
| mean_cost | 0.01 |

Basic learning signal confirmed. Full training to be run on GPU cluster.

---

## Versioning Convention (reminder)

- v01 files → never touch (baseline)
- v02 files → never touch (multi-discount GAE baseline)
- v03 files → this experiment
- If new experiment needed → create v04, copy from v03 or v02
