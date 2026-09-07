#============HEURISTIC + ATTENTION VARIANT (CONTINUOUS)============
# Uses conflict-detection heuristic to filter neighbors AND adds
# an attention mask for the attention model.
# Obs: [ego(4)] + [leader(3)] + [neighbor(5) x 5] + [mask(5)] = 37

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

    Obs = [d_norm, v_norm, sin, cos] + [gap, leader_v, ttc]
          + 5x[dist_to_cp, v, d_eta, sin, cos] + 5x[mask] = 37
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        # Override observation space to include the mask. The parent layout is
        # [ego][leader][neighbours]; the leader block must be counted here or
        # the declared space is shorter than what _get_local_observation emits.
        total_obs_len = (self.ego_obs_features
                         + self.leader_obs_features
                         + (self.neighbour_obs_features * self.max_neighbours)
                         + self.max_neighbours)
        self.observation_space = Box(
            low=-1.0, high=1.0,
            shape=(total_obs_len, ),
            dtype=np.float32)

        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _get_local_observation(self, ego_id):
        # Parent returns (obs_array, neighbors_info), heuristic-filtered.
        result = super()._get_local_observation(ego_id)

        if isinstance(result, tuple):
            base_obs, neighbors_info = result
        else:
            base_obs, neighbors_info = result, []

        base_obs = np.asarray(base_obs, dtype=np.float32)

        # The parent bails out with the cached observation when the ego has no
        # valid 2D position. That cache is already mask-extended, so appending
        # a second mask would change the shape.
        if base_obs.shape[0] == self.observation_space.shape[0]:
            return base_obs, neighbors_info

        # Count real neighbours from the list the parent already truncated,
        # rather than sniffing padded slots. The padding sentinel lives in the
        # parent and has changed before; the list length cannot drift.
        if not isinstance(neighbors_info, list):
            neighbors_info = []
        num_actual = min(len(neighbors_info), self.max_neighbours)

        mask = np.zeros(self.max_neighbours, dtype=np.float32)
        mask[:num_actual] = 1.0

        return np.concatenate([base_obs, mask]), neighbors_info
