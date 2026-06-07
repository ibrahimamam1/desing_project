#============DISCRETE ACTION SPACE VARIANT OF V0.1 ENV============
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from alpha_env_v02 import AlphaEnv_v01
class AlphaEnv_v01_Discrete(AlphaEnv_v01):
    """
    Same as AlphaEnv_v01 but with a discrete action space.
    5 bins: [-1.0, -0.5, 0.0, 0.5, 1.0]
    Mapped to: [-max_decel, -max_decel/2, 0, max_accel/2, max_accel]
    """

    ACCEL_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        # Override action space to discrete
        self.action_space = Discrete(len(self.ACCEL_BINS))
    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']
        # Map discrete action index to normalized value
        action_idx = int(rl_action)
        action_val = self.ACCEL_BINS[action_idx]

        # Denormalize from [-1, 1] to [-max_decel, max_accel]
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel
        rl_ids = []
        rl_ids.append(self.agent_id)
        if not rl_ids:
            return
        self.k.vehicle.apply_acceleration(rl_ids, [real_action])
    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        # 1. Sparse Terminal Rewards
        if fail:
            self.last_r_traj, self.last_r_cruise = 0.0, -10.0
            return -10.0
        if goal_reached:
            self.last_r_traj, self.last_r_cruise = 15.0, 0.0
            return 15.0

        # Fallback to empty list if no neighbors
        obs_info = getattr(self, 'last_neighbors_info', [])

        # 2. Progress Reward
        ego_dis = self.k.vehicle.get_distance(agent_id)
        if ego_dis == -1001: ego_dis = 0.0
        route = self.k.vehicle.get_route(agent_id)
        total_route_length = max(sum([self.k.network.edge_length(e) for e in route]), 1e-4)

        progress_norm = np.clip(ego_dis / total_route_length, 0.0, 1.0)

        # Initialize last_progress if it doesn't exist yet
        if not hasattr(self, 'last_progress'):
            self.last_progress = progress_norm

        progress_delta = progress_norm - self.last_progress
        self.last_progress = progress_norm

        # 3. Safety Penalty
        safety_penalty = 0.0
        # A. Crossing conflicts
        for n in obs_info:
            # n['d_eta'] is normalized [-1, 1]. Close to 0 is highly dangerous.
            # We take the absolute value, so 0 is a crash, 1 is perfectly safe.
            abs_d_eta = abs(n['d_eta'])

            # Only penalize if they are projected to arrive within a tight window of each other
            if abs_d_eta < 0.2:
                # Exponential penalty: spikes hard as d_eta approaches 0
                safety_penalty += -np.exp(-abs_d_eta * 10.0)

        # B. Car-following safety penalty
        leader_info = getattr(self, 'last_leader_info', {})
        if leader_info.get('has_leader', False):
            abs_ttc = leader_info['ttc_norm']
            if abs_ttc < 0.2:
                safety_penalty += -np.exp(-abs_ttc * 10.0)

        # 4. Dense Reward Assembly — Split for Multi-Discount GAE
        r_traj   = 10.0 * progress_delta        # Long horizon: progress toward goal
        r_cruise = 1.0 * safety_penalty - 0.01  # Short horizon: safety + time penalty

        # Store components so base_env can pass them through infos
        self.last_r_traj   = r_traj
        self.last_r_cruise = r_cruise

        return r_cruise + r_traj  # Single scalar — SB3 interface unchanged
