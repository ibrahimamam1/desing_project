#============HEURISTIC + ATTENTION VARIANT (CONTINUOUS)============
# Uses conflict-detection heuristic to filter neighbors AND adds
# an attention mask for the attention model.
# Obs: [ego(2)] + [neighbor(3) × 5] + [mask(5)] = 22

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01 import AlphaEnv_v01

class AlphaEnv_v01_HeuristicAttention(AlphaEnv_v01):
    """
    Combines conflict-detection heuristic (from v01) with attention mask.
    - Neighbors are filtered using the conflict map (like v01)
    - A neighbor mask is appended to observation (like attention variant)
    - The attention model uses the mask in its forward pass
    
    Obs = [d_norm, v_norm] + 5×[v, d, ttc] + 5×[mask]  = 22
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        # Override observation space to include mask
        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours) + self.max_neighbours
        self.observation_space = Box(
            low=-1.0, high=1.0,
            shape=(total_obs_len, ),
            dtype=np.float32)
        
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _get_local_observation(self, ego_id):
        # Get the base observation from AlphaEnv_v01 (heuristic-filtered, 17 dims)
        base_obs = super()._get_local_observation(ego_id)
        
        # Count actual neighbors by checking non-padded slots
        # Base obs layout: [ego(2)] + N×[v, d, ttc]
        # Padded neighbors have [v=0, d=1, ttc=1]
        num_actual = 0
        for i in range(self.max_neighbours):
            start = self.ego_obs_features + i * self.neighbour_obs_features
            v_val = base_obs[start]      # v
            d_val = base_obs[start + 1]  # d
            ttc_val = base_obs[start + 2] # ttc
            # A real neighbor unlikely to have exactly v=0, d=1, ttc=1
            if not (v_val == 0.0 and d_val == 1.0 and ttc_val == 1.0):
                num_actual += 1
            else:
                break  # padded neighbors are at the end (sorted by distance)
        
        # Create mask: 1.0 for real neighbors, 0.0 for padded
        mask = [1.0] * num_actual + [0.0] * (self.max_neighbours - num_actual)
        
        # Append mask to base observation
        obs_with_mask = np.concatenate([base_obs, np.array(mask, dtype=np.float32)])
        
        return obs_with_mask
