# =============================================================================
# Multi-Discount Rollout Buffer for PPO
# =============================================================================
# Implements the Multi-Discount GAE proposed in the thesis (page 18):
#
#   Â(1)_t = GAE(γ1, λ1) on r_cruise  (short horizon: safety, TTC avoidance)
#   Â(2)_t = GAE(γ2, λ2) on r_traj   (long horizon: progress, efficiency)
#   Â_t    = w1 * Â(1)_t + w2 * Â(2)_t
#
# r_cruise and r_traj are passed through infos dict from the environment.
# SubprocVecEnv still receives a single scalar reward — interface unchanged.
# =============================================================================

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces


class MultiDiscountRolloutBuffer(RolloutBuffer):
    """
    Extension of SB3's RolloutBuffer that computes two separate GAE
    advantages with different (gamma, lambda) pairs and combines them.

    Extra constructor args:
        gamma1  : discount factor for r_cruise (short horizon, safety)
        gamma2  : discount factor for r_traj   (long horizon, progress)
        lambda1 : GAE lambda for r_cruise
        lambda2 : GAE lambda for r_traj
        w1      : weight for advantage_cruise
        w2      : weight for advantage_traj
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        gae_lambda=0.95,   # kept for SB3 compatibility, not used in dual GAE
        gamma=0.99,        # kept for SB3 compatibility, not used in dual GAE
        n_envs=1,
        # --- Multi-Discount Parameters ---
        gamma1=0.90,       # short horizon gamma (safety)
        gamma2=0.99,       # long horizon gamma (progress)
        lambda1=0.90,      # short horizon lambda
        lambda2=0.95,      # long horizon lambda
        w1=0.4,            # weight for safety advantage
        w2=0.6,            # weight for progress advantage
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        # Store multi-discount hyperparameters
        self.gamma1  = gamma1
        self.gamma2  = gamma2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w1      = w1
        self.w2      = w2

        # Two separate reward buffers — filled from infos dict
        self.rewards_cruise = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards_traj   = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def reset(self):
        """Reset all buffers including the two reward component buffers."""
        self.rewards_cruise = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rewards_traj   = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add_reward_components(self, pos, r_cruise, r_traj):
        """
        Store the two reward components at the given buffer position.
        Called from the custom PPO collect_rollouts after each env step.

        pos     : current buffer position
        r_cruise: np.array shape (n_envs,) — safety + time penalty
        r_traj  : np.array shape (n_envs,) — progress reward
        """
        self.rewards_cruise[pos] = r_cruise.copy()
        self.rewards_traj[pos]   = r_traj.copy()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        Override SB3's GAE computation.
        Runs GAE twice — once per reward component — then combines.
        """
        last_values = last_values.clone().cpu().numpy().flatten()

        # --- GAE for r_cruise (short horizon: gamma1, lambda1) ---
        last_gae_cruise = 0
        advantages_cruise = np.zeros_like(self.rewards_cruise)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards_cruise[step]
                + self.gamma1 * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_cruise = (
                delta
                + self.gamma1 * self.lambda1 * next_non_terminal * last_gae_cruise
            )
            advantages_cruise[step] = last_gae_cruise

        # --- GAE for r_traj (long horizon: gamma2, lambda2) ---
        last_gae_traj = 0
        advantages_traj = np.zeros_like(self.rewards_traj)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards_traj[step]
                + self.gamma2 * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_traj = (
                delta
                + self.gamma2 * self.lambda2 * next_non_terminal * last_gae_traj
            )
            advantages_traj[step] = last_gae_traj

        # --- Combine: Â_t = w1 * Â_cruise + w2 * Â_traj ---
        self.advantages = self.w1 * advantages_cruise + self.w2 * advantages_traj

        # --- Returns for value function — computed from total reward with gamma2 (consistent) ---
        total_rewards = self.rewards_cruise + self.rewards_traj
        last_gae_combined = 0
        advantages_combined = np.zeros_like(total_rewards)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                total_rewards[step]
                + self.gamma2 * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_combined = (
                delta
                + self.gamma2 * self.lambda2 * next_non_terminal * last_gae_combined
            )
            advantages_combined[step] = last_gae_combined
        self.returns = advantages_combined + self.values
