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
        # 1. Terminal conditions
        if fail:
            self.telemetry["reward_terminal_total"] -= 10.0
            return -10.0  
        if goal_reached:
            self.telemetry["reward_terminal_total"] += 15.0
            return 15.0   
            
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        # 2. Step components
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
        
        # 3. Accumulate in telemetry
        self.telemetry["reward_speed_total"] += speed_reward
        self.telemetry["reward_time_total"] += time_penalty
        self.telemetry["reward_action_total"] += action_penalty
        
        return speed_reward + time_penalty + action_penalty
