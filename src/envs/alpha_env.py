#============NEW ACCEL ENV ENVIRONMENT BY IBRAHIMA============
from flow.core import rewards

import gymnasium as gym
import numpy as np
import random
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env import Env_N
import numpy as np
import gymnasium as gym
from copy import deepcopy

class AlphaEnv(Env_N):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        
        self.ego_obs_features = 4 # S_ego = [Vel, Accel, Dis along path, path] 
        self.neighbour_obs_features = 3
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = gym.spaces.Box(
            low=-1.0,  # Normalized
            high=1.0,   # Normalized
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space
        total_obs_len = self.ego_obs_features
        self.observation_space = gym.spaces.Box(
            low=-10.0,  # Bounded instead of inf
            high=10.0,   # Bounded instead of inf
            shape=(total_obs_len, ),
            dtype=np.float32)
   
    def get_state(self):
        """
        Return the state of the simulation as a DICTIONARY.
        Format: { "veh_id_1": [obs_array], "veh_id_2": [obs_array] }
        """
        obs_dict = {}
        for veh_id in self.k.vehicle.get_rl_ids():
                obs = self._get_local_observation(veh_id)
                #Clip observations to prevent extreme values
                obs = np.clip(obs, -10.0, 10.0)
                obs_dict[veh_id] = obs
            
        return obs_dict

    def _get_local_observation(self, ego_id):
        """
        Internal helper to build the observation array for ONE agent.
        """
        try:
            # --- 1. Get Ego State ---
            max_speed = self.k.network.max_speed()
            ego_speed = self.k.vehicle.get_speed(ego_id)
            # normalize speed
            ego_speed = ego_speed / max_speed
            
            # Normalized Accel
            max_accel = self.env_params.additional_params.get('max_accel', 3.0)
            realized_accel = self.k.vehicle.get_realized_accel(ego_id)
            ego_accel = realized_accel / max_accel
            
            # Ego path encoding
            try:
                route = self.k.vehicle.get_route(ego_id)
                if len(route) > 0:
                    start_edge = route[0]
                    end_edge = route[-1]
                    route_hash = (hash(start_edge) + hash(end_edge)) % 12
                    ego_path = route_hash / 12.0  # Normalize to 0-1
                else:
                    ego_path = 0.0
            except Exception:
                ego_path = 0.0
           
            # Distance traveled
            ego_dis = self.k.vehicle.get_distance(ego_id)
           
            # Start the vector
            obs_vector = [ego_speed, ego_accel, ego_dis, ego_path]
            return np.array(obs_vector, dtype=np.float32)
            
        except Exception as e:
            # FIX: If anything goes wrong, return safe zero observation
            print(f"Warning: Error getting observation for {ego_id}: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _apply_rl_actions(self, rl_actions_dict):
        """
        Apply RL actions to the vehicles.
        Input: Dictionary { "veh_id": action_value }
        """
        if not rl_actions_dict:
            return

        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        actions_to_apply = []
        ids_to_apply = [] 

        for veh_id, action in rl_actions_dict.items():
            if veh_id not in self.k.vehicle.get_ids():
                continue
            
            action_val = float(action)
            # Denormalize from [-1, 1] to [max_decel, max_accel]
            if action_val >= 0:
                real_action = action_val * max_accel
            else:
                real_action = action_val * max_decel
                
            # Extra safety clip
            real_action = np.clip(real_action, -max_decel, max_accel)
            actions_to_apply.append(real_action)
            ids_to_apply.append(veh_id)

        if ids_to_apply:
            try:
                self.k.vehicle.apply_acceleration(ids_to_apply, actions_to_apply)
            except Exception as e:
                print(f"Warning: Error applying actions: {e}")

    def compute_reward(self, agent_id, fail, goal_reached):
        """Reward with intermediate feedback for learning"""
        if fail:
            return -50.0
        if goal_reached:
            return 10.0
        
        return -1 #Delay Penalty

    def additional_command(self):
        """
        Update the sorting of vehicles using the self.sorted_ids variable.
        """
        try:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

            for veh_id in self.k.vehicle.get_ids():
                this_pos = self.k.vehicle.get_x_by_id(veh_id)

                if this_pos == -1001:
                    self.absolute_position[veh_id] = -1001
                else:
                    change = this_pos - self.prev_pos.get(veh_id, this_pos)
                    self.absolute_position[veh_id] = \
                        (self.absolute_position.get(veh_id, this_pos) + change) \
                        % self.k.network.length()
                    self.prev_pos[veh_id] = this_pos
        except Exception as e:
            print(f"Warning in additional_command: {e}")

    def _get_abs_position(self, veh_id):
        return self.absolute_position.get(veh_id, -1001)
