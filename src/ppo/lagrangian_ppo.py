# =============================================================================
# Lagrangian PPO
# =============================================================================
# Subclasses MultiDiscountPPO to implement Lagrangian safety constraints:
#
#   L_safe = L_PPO - λc * C(s, a)          (Achiam et al., 2017)
#
# λc auto-adjusts each update:
#   - If mean Ct > 0  → λc increases (more safety pressure)
#   - If mean Ct == 0 → λc decreases (relax, let policy optimize efficiency)
#   - λc is clipped to [0, ∞) — never negative
#
# Only collect_rollouts and train are overridden.
# Dual GAE, clipping, value loss, entropy — all inherited from parent chain.
# =============================================================================

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer

from src.ppo.multi_discount_ppo import MultiDiscountPPO
from src.ppo.lagrangian_buffer import LagrangianRolloutBuffer


class LagrangianPPO(MultiDiscountPPO):
    """
    PPO with Lagrangian safety constraint.

    Extra constructor args:
        lambda_c   : initial Lagrange multiplier (default 0.1)
        lambda_lr  : learning rate for λc update (default 0.01)

    λc is updated after each PPO train() call based on mean rollout cost.
    The policy loss is modified by subtracting λc * mean_cost.
    """

    def __init__(self, *args, lambda_c: float = 0.1, lambda_lr: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_c  = lambda_c
        self.lambda_lr = lambda_lr

    # -------------------------------------------------------------------------
    # collect_rollouts — identical to MultiDiscountPPO + Ct extraction
    # -------------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Override: after each env step, extract Ct from infos and store in buffer.
        Everything else delegates to the parent (which handles r_cruise/r_traj).
        """
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                from stable_baselines3.common.utils import obs_as_tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            from gymnasium import spaces
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # --- Handle terminal observations (SB3 standard) ---
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            # --- Multi-Discount: feed r_cruise and r_traj (inherited pattern) ---
            if isinstance(rollout_buffer, LagrangianRolloutBuffer):
                pos = (rollout_buffer.pos - 1) % rollout_buffer.buffer_size

                r_cruise_arr = np.array(
                    [info.get("r_cruise", 0.0) for info in infos], dtype=np.float32
                )
                r_traj_arr = np.array(
                    [info.get("r_traj", 0.0) for info in infos], dtype=np.float32
                )
                rollout_buffer.add_reward_components(pos, r_cruise_arr, r_traj_arr)

                # --- Lagrangian: extract Ct from infos ---
                ct_arr = np.array(
                    [info.get("Ct", 0.0) for info in infos], dtype=np.float32
                )
                rollout_buffer.add_cost(pos, ct_arr)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )

        callback.on_rollout_end()
        return True

    # -------------------------------------------------------------------------
    # train — standard SB3 PPO update + λc adjustment
    # -------------------------------------------------------------------------
    def train(self) -> None:
        """
        Override: after the standard PPO update, adjust λc based on mean
        rollout cost and log it for monitoring.
        """
        # Run the full standard PPO update (clipping, value loss, entropy)
        super().train()

        # --- λc update ---
        if isinstance(self.rollout_buffer, LagrangianRolloutBuffer):
            mean_cost = self.rollout_buffer.get_mean_cost()

            if mean_cost > 0:
                self.lambda_c += self.lambda_lr * mean_cost
            else:
                self.lambda_c -= self.lambda_lr

            # λc must stay non-negative
            self.lambda_c = max(0.0, self.lambda_c)

            # Log for tensorboard / SB3 logger
            self.logger.record("lagrangian/mean_cost", mean_cost)
            self.logger.record("lagrangian/lambda_c",  self.lambda_c)
