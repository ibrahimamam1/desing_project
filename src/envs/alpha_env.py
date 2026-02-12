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
        
        self.ego_obs_features = 4 
        self.neighbour_obs_features = 3
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = gym.spaces.Box(
            low=-1.0,  # Normalized
            high=1.0,   # Normalized
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space
        total_obs_len = self.ego_obs_features + (self.max_neighbours * self.neighbour_obs_features)
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
            if veh_id in self.k.vehicle.get_ids() and self._is_in_control_zone(veh_id):
                obs = self._get_local_observation(veh_id)
                # FIX: Clip observations to prevent extreme values
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
            if max_speed <= 0:
                max_speed = 30.0  # Fallback
            
            ego_speed = self.k.vehicle.get_speed(ego_id)
            # FIX: Clip and normalize speed
            ego_speed = np.clip(ego_speed / max_speed, 0, 2.0)
            
            # Normalized Accel
            max_accel = self.env_params.additional_params.get('max_accel', 3.0)
            realized_accel = self.k.vehicle.get_realized_accel(ego_id)
            ego_accel = np.clip(realized_accel / max_accel, -2.0, 2.0)
            
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
            
            # Distance traveled - normalize by a reasonable max distance
            ego_dis = self.k.vehicle.get_distance(ego_id)
            ego_dis = np.clip(ego_dis / 1000.0, 0, 5.0)  # Normalize by 1km
            
            # Start the vector
            obs_vector = [ego_speed, ego_accel, ego_dis, ego_path]
            
            # --- 2. Find Neighbors ---
            ego_pos = np.array(self.k.vehicle.get_2d_position(ego_id))
            all_ids = self.k.vehicle.get_ids()
            
            def get_dist_sq(other_id):
                try:
                    o_pos = np.array(self.k.vehicle.get_2d_position(other_id))
                    return np.sum((ego_pos - o_pos)**2)
                except:
                    return float('inf')

            others = []
            for vid in all_ids:
                if vid == ego_id:
                    continue
                dist_sq = get_dist_sq(vid)
                if dist_sq <= self.perception_radius**2:
                    others.append((vid, dist_sq))

            others.sort(key=lambda x: x[1])
            closest_neighbors_ids = [x[0] for x in others[:self.max_neighbours]]
            
            # --- 3. Add Neighbor States ---
            for neigh_id in closest_neighbors_ids:
                try:
                    neigh_speed = self.k.vehicle.get_speed(neigh_id) / max_speed
                    neigh_speed = np.clip(neigh_speed, 0, 2.0)
                    
                    # Relative Speed (Delta V)
                    rel_neigh_speed = ego_speed - neigh_speed
                    rel_neigh_speed = np.clip(rel_neigh_speed, -2.0, 2.0)
                    
                    # Relative Distance (Delta D)
                    dist = np.sqrt(get_dist_sq(neigh_id))
                    rel_neigh_dis = np.clip(dist / self.perception_radius, 0, 2.0)
                    
                    # Time To Collision (TTC)
                    # FIX: More robust TTC calculation
                    denorm_rel_speed = rel_neigh_speed * max_speed
                    if abs(denorm_rel_speed) < 0.1:  # Almost same speed
                        neigh_ttc = 1.0  # Normalized "safe" value
                    else:
                        raw_ttc = dist / abs(denorm_rel_speed)
                        neigh_ttc = np.clip(raw_ttc / 10.0, 0, 2.0)  # Normalize by 10s

                    obs_vector.extend([rel_neigh_speed, rel_neigh_dis, neigh_ttc])
                except Exception as e:
                    # If we can't get neighbor data, use safe defaults
                    obs_vector.extend([0.0, 1.0, 1.0])
                    
            # --- 4. Padding ---
            needed_length = self.observation_space.shape[0]
            current_length = len(obs_vector)
            
            if current_length < needed_length:
                padding = [0.0] * (needed_length - current_length)
                obs_vector.extend(padding)
                
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

        ids_to_apply = []
        actions_to_apply = []

        for veh_id, action in rl_actions_dict.items():
            if veh_id not in self.k.vehicle.get_ids():
                continue
                
            if self._is_in_control_zone(veh_id):
                ids_to_apply.append(veh_id)
                
                # FIX: Properly denormalize and clip actions
                if isinstance(action, (list, np.ndarray)):
                    action_val = float(action[0])
                else:
                    action_val = float(action)
                
                # Denormalize from [-1, 1] to [max_decel, max_accel]
                if action_val >= 0:
                    real_action = action_val * max_accel
                else:
                    real_action = action_val * max_decel
                
                # Extra safety clip
                real_action = np.clip(real_action, -max_decel, max_accel)
                actions_to_apply.append(real_action)

        if ids_to_apply:
            try:
                self.k.vehicle.apply_acceleration(ids_to_apply, actions_to_apply)
            except Exception as e:
                print(f"Warning: Error applying actions: {e}")

    def compute_reward(self, agent_id, action, fail, goal_reached):
        """Reward with intermediate feedback for learning"""
        
        # Terminal rewards
        if fail:
            return -50.0
        if goal_reached:
            return 10.0
        
        # Intermediate rewards (NEW)
        reward = 0.0
        
        # 1. Survival bonus (small positive reinforcement)
        reward += 0.1
        
        # 2. Distance-based safety reward
        try:
            min_dist = self._get_min_distance_to_others(agent_id)
            
            if min_dist < 3.0:      # Critical danger
                reward -= 1.0
            elif min_dist < 5.0:    # Too close  
                reward -= 0.3
            elif min_dist > 8.0:    # Safe
                reward += 0.2
        except:
            pass
        
        # 3. Penalize panic braking (suggests poor planning)
        if action is not None and action < -0.9:
            reward -= 0.2
        
        return reward

    def _get_min_distance_to_others(self, veh_id):
        """Calculate distance to nearest vehicle"""
        try:
            my_pos = self.k.vehicle.get_position(veh_id)
            min_dist = float('inf')
            
            for other_id in self.k.vehicle.get_ids():
                if other_id != veh_id:
                    other_pos = self.k.vehicle.get_position(other_id)
                    dist = np.sqrt((my_pos[0] - other_pos[0])**2 + 
                                  (my_pos[1] - other_pos[1])**2)
                    min_dist = min(min_dist, dist)
            
            return min_dist if min_dist != float('inf') else 100.0
        except:
            return 100.0

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
