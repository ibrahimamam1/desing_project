"""Test environment used to run simulations in the absence of autonomy."""
from flow.envs.base_new import Env_N as Env
from gymnasium.spaces import Box
import numpy as np


class TestEnv(Env):
    """Test environment used to run simulations in the absence of autonomy.
    
    Required from env_params
        None
    
    Optional from env_params
        reward_fn : A reward function which takes an input the environment
        class and returns a real number.
    
    States
        States are an empty list.
    
    Actions
        No actions are provided to any RL agent.
    
    Rewards
        The reward is zero at every step.
    
    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """Apply RL actions (no-op for test environment)."""
        return

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        if "reward_fn" in self.env_params.additional_params:
            return self.env_params.additional_params["reward_fn"](self)
        else:
            return 0

    def get_state(self, **kwargs):
        """See class definition."""
        return np.array([])
    
    def step(self, rl_actions):
        """Advance the environment by one step.
        
        Overrides parent to ensure proper return types for non-RL simulation.
        """
        # Call parent step method
        obs, reward, terminated, truncated, info = super().step(rl_actions)
        
        # Convert dict returns to scalar values if needed
        if isinstance(obs, dict):
            obs = np.array([])
        
        if isinstance(reward, dict):
            # Sum all rewards or just return 0 for non-RL
            reward = 0.0
        
        if isinstance(terminated, dict):
            # Check if any agent is terminated
            terminated = any(terminated.values()) if terminated else False
            
        if isinstance(truncated, dict):
            # Check if any agent is truncated
            truncated = any(truncated.values()) if truncated else False
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def reset(self, seed=None, options=None):
        """Reset the environment.
        
        Overrides parent to ensure proper return types for non-RL simulation.
        """
        # Call parent reset - handle both old and new API
        try:
            result = super().reset(seed=seed, options=options)
        except TypeError:
            # Fallback if parent doesn't accept seed/options
            result = super().reset()
        
        # Handle return value
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Convert dict obs to array if needed
        if isinstance(obs, dict):
            obs = np.array([])
        
        return obs, info
