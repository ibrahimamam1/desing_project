# v03 — Lagrangian Safety (Single Agent PPO)

**Built on top of:** v02 (Multi-Discount GAE) — v02 files are **never modified**.

---

## What It Does

```
L_safe = L_PPO − λc · C(s, a)          (Achiam et al., 2017 — CPO)
Ct = c1 · 1[|d_eta| < τ_crit] + c2 · 1[collision]
```

λc auto-adjusts every update: increases when mean(Ct) > 0, decreases otherwise, clipped ≥ 0.

An earlier version of this implementation computed `Ct` and `λc` correctly but never
actually applied the penalty to the policy loss — the constraint was fully logged but had
zero effect on training. The fixes below close that gap.

---

## Files

| Action | File |
|---|---|
| Modified | `src/envs/base_env_single.py` — computes `Ct`, injects into `infos` |
| Modified | `src/ppo/lagrangian_buffer.py` — stores `Ct`, adds cost-GAE |
| Modified | `src/ppo/lagrangian_ppo.py` — applies λc·cost penalty, then updates λc |
| **New** | `src/envs/alpha_env_v03_discrete.py` — cancels v02's duplicate safety penalty, without touching v02 |
| Modified | `src/configs/v0_3_single_agent.py` — points to the new v03 env class |

**Untouched:** all v01 files, all v02 files (`alpha_env_v02_discrete.py`, `multi_discount_ppo.py`, `multi_discount_buffer.py`), `v0_2_single_agent.py`.

---

## Architecture

```
alpha_env_v03_discrete.py   (NEW — v03 only)
    → calls v02's compute_reward(), then cancels the crossing-conflict
      penalty already baked into r_cruise (Ct/λc now own that signal)

base_env_single.py :: step()
    → Ct = Σ c1·1[|d_eta|<τ_crit] + c2·1[crashed]  → infos["Ct"]
      (v01/v02 never read this key — zero effect on them)

LagrangianPPO.collect_rollouts()
    → reads infos["Ct"] → rollout_buffer.add_cost()

LagrangianRolloutBuffer.compute_returns_and_advantage()
    → calls super() first (fixed v02 GAE, unchanged: self.returns, self.advantages)
    → then computes cost_advantages via GAE on self.costs (γ1, λ1 — reused from v02)

LagrangianPPO.train()
    → self.advantages -= λc * cost_advantages     [THE FIX — this line was missing]
    → super().train()   (PPO update now genuinely safety-penalized)
    → λc updated from mean_cost, logged to tensorboard
```

---

## Why This Design

**Cost-GAE, not a trained cost-critic:** true CPO trains a separate `Vc(s)` value head.
Minimum-effort alternative here: a baseline-free discounted cost-to-go (GAE with V=0) —
unbiased, higher variance, zero architecture changes, reuses existing `γ1`/`λ1`.

**Penalty applied to `self.advantages`, not raw reward:** `compute_returns_and_advantage()`
only reads `rewards_cruise`/`rewards_traj`, never SB3's base `rewards` field — shaping the
scalar reward would have no effect. `self.advantages` is the exact tensor PPO's clipped
surrogate loss consumes, so that's where the penalty must land. `self.returns` (value target)
is left untouched, keeping the value function's target stable as λc drifts over training.

**New env file instead of editing `alpha_env_v02_discrete.py`:** the crossing-conflict
penalty already in v02's `compute_reward()` would double-count once `Ct`/λc also penalize
the same `|d_eta| < 0.2` signal. Editing the v02 file directly would fix v03 but silently
regress v02 (no compensating mechanism exists there). Instead, `alpha_env_v03_discrete.py`
subclasses the v02 env, calls its `compute_reward()` unchanged, and mathematically cancels
just the duplicate term — v02 stays byte-for-byte untouched and fully independent.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| c1, c2, τ_crit | 1.0, 5.0, 0.2 |
| λc init, λc lr | 0.1, 0.01 |
| γ1, γ2 | 0.90, 0.99 (inherited from v02, reused for cost-GAE) |
| λ1, λ2 | 0.90, 0.95 (inherited from v02, reused for cost-GAE) |
| w1, w2 | 0.4, 0.6 (inherited from v02) |

No new hyperparameters introduced.

---

## Validation

Watch `lagrangian/mean_cost` in TensorBoard — it should now visibly trend down over
training, since the policy has a genuine gradient incentive to reduce `Ct` for the first
time. Pre-fix, this metric could stay flat/noisy indefinitely regardless of policy behavior.
Previously logged smoke-test numbers (5120 steps, `lambda_c` stable at 0.1) reflect the
broken, pre-fix pipeline and should be re-run before being treated as representative of the
Lagrangian constraint actually working.

Note: this doc/fix set covers `heuristic_discrete` only. Other v03 variants
(`attention_discrete`, `attention_continous`, `heuristic_continuous`) need the same
thin-wrapper pattern (`alpha_env_v03_*.py`) repeated once each if used with v03.
