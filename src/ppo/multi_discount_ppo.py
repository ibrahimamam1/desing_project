# =============================================================================
# Multi-Discount PPO
# =============================================================================
# Subclasses SB3's PPO and overrides collect_rollouts to feed r_cruise and
# r_traj from infos into the MultiDiscountRolloutBuffer after each step.
# Everything else is identical to SB3's PPO.
# =============================================================================

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from src.ppo.multi_discount_buffer import MultiDiscountRolloutBuffer


class MultiDiscountPPO(PPO):
    """
    PPO with dual-GAE support via MultiDiscountRolloutBuffer.
    Only collect_rollouts is overridden — everything else is standard SB3 PPO.
    """

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
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

            # --- Multi-Discount: store reward components from infos ---
            if isinstance(rollout_buffer, MultiDiscountRolloutBuffer):
                r_cruise = np.array([info.get("r_cruise", 0.0) for info in infos], dtype=np.float32)
                r_traj   = np.array([info.get("r_traj",   0.0) for info in infos], dtype=np.float32)
                rollout_buffer.add_reward_components(
                    pos=rollout_buffer.pos - 1,  # pos was incremented by add()
                    r_cruise=r_cruise,
                    r_traj=r_traj,
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
