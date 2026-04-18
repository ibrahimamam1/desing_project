#============DISCRETE ACTION SPACE VARIANT OF V0.1 ENV============

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01 import AlphaEnv_v01

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
        # 1. Terminal conditions (Normalized to larger, but bounded values)
        # Assuming your dense step rewards will be roughly in the [-1, 1] range.
        if fail:
            return -1.0  
        if goal_reached:
            return 1.0   
            
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        # --- POTENTIAL FUNCTION CALCULATION ---
       
        obs_info = self.last_neighbors_info
        # A. Progress Potential (0.0 at start, 1.0 at goal)
        ego_dis = self.k.vehicle.get_distance(agent_id)
        if ego_dis == -1001: ego_dis = 0.0
        route = self.k.vehicle.get_route(agent_id)
        total_route_length = max(sum([self.k.network.edge_length(e) for e in route]), 1e-4)
        
        progress_norm = np.clip(ego_dis / total_route_length, 0.0, 1.0)
        
        # B. Safety Potential (1.0 is perfectly safe, 0.0 is imminent crash)
        # We extract this directly from the `neighbors_info` list you generated in _get_local_observation
        min_delta_eta_norm = 1.0
        
        if obs_info and len(obs_info) > 0:
            # For crossing conflicts, we want delta_eta away from 0. 
            # abs(delta_eta_norm) close to 0 is dangerous. 
            min_delta_eta_norm = min([abs(n['d_eta']) for n in obs_info])

        # Combine into total state potential
        # Weights: 0.4 for progress, 0.6 for safety (prioritize not crashing)
        current_potential = (0.4 * progress_norm) + (0.6 * min_delta_eta_norm)
        
        # --- PBRS STEP REWARD ---
        gamma = 0.98 # Make sure this matches your RL algorithm's gamma
        reward_pbrs = (gamma * current_potential) - self.last_potential
        self.last_potential = current_potential

        # We don't strictly need a time penalty anymore, as the progress potential 
        # naturally encourages reaching the goal to maximize the discounted potential.
        
        return reward_pbrs
