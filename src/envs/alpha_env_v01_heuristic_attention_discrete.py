#============HEURISTIC + ATTENTION VARIANT (DISCRETE)============
# Conflict-detection heuristic + attention mask + discrete action space.

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01_heuristic_attention_continous import AlphaEnv_v01_HeuristicAttention

class AlphaEnv_v01_HeuristicAttentionDiscrete(AlphaEnv_v01_HeuristicAttention):
    """
    Combines conflict-detection heuristic + attention mask + discrete actions.
    5 bins: [-1.0, -0.5, 0.0, 0.5, 1.0]
    """
    
    ACCEL_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.action_space = Discrete(len(self.ACCEL_BINS))

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        action_idx = int(rl_action)
        action_val = self.ACCEL_BINS[action_idx]
        
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        rl_ids = self.sorted_ids
        if not rl_ids:
            return
        self.k.vehicle.apply_acceleration(rl_ids, [real_action])

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        if fail:
            return -10.0  
        if goal_reached:
            return 15.0   
            
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        speed = self.k.vehicle.get_speed(agent_id)
        max_speed = self.k.network.max_speed()

        speed_reward = 0.05 * (speed / max_speed)
        time_penalty = -0.02 
        
        action_penalty = 0.0
        if current_action is not None:
            action_idx = int(current_action)
            action_val = self.ACCEL_BINS[action_idx]
            action_penalty = -0.02 * abs(action_val - self.last_action)
            self.last_action = action_val
        
        return speed_reward + time_penalty + action_penalty
