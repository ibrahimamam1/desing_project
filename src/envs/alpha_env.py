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
        self.max_neighbours = 3
        self.perception_radius = 50
        
        self.ego_obs_features = 4 # S_ego = [Vel, Accel, Dis along path, path] 
        self.neighbour_obs_features = 3 #S_neigh = [delta_V, Delta_D, TTC]
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = gym.spaces.Box(
            low=-1.0,  # Normalized
            high=1.0,   # Normalized
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space
        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
        self.observation_space = gym.spaces.Box(
            low=-1,  
            high=1,   
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
                obs = np.clip(obs, -1, 1)
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
            # normalize speed to [0,1]
            ego_speed = ego_speed / max_speed
            
            # get and Normalized Accel to [0,1]
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
            route = self.k.vehicle.get_route(ego_id)
            total_route_length = sum([self.k.network.edge_length(edge) for edge in route])
            # Get cumulative distance traveled
            ego_dis = self.k.vehicle.get_distance(ego_id)
            # Normalize: 0.0 (start) to 1.0 (end of route)
            ego_dis_norm = ego_dis / total_route_length if total_route_length > 0 else 0.0            
            
            # Start the vector
            obs_vector = [ego_speed, ego_accel, ego_dis_norm, ego_path]
            
            # --- 2. Get Neighbour state ---
            # Get ego position
            ego_x, ego_y = self.k.vehicle.get_2d_position(ego_id)
            ego_pos = np.array([ego_x, ego_y])
            
            # Collect neighbors within perception radius
            neighbors_info = []
            agent_ids = self.k.vehicle.get_rl_ids()
            
            for agent_id in agent_ids:
                if agent_id == ego_id:
                    continue 
                
                # Get agent's position
                agent_x, agent_y = self.k.vehicle.get_2d_position(agent_id)
                agent_pos = np.array([agent_x, agent_y])
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(ego_pos - agent_pos)
                
                # Check if agent is within perception range
                if distance <= self.perception_radius:
                    # Get agent velocity
                    agent_speed = self.k.vehicle.get_speed(agent_id)
                    
                    # Compute relative velocity (delta_V)
                    delta_v = agent_speed - (ego_speed * max_speed)  # Denormalize ego_speed
                    # Normalize delta_v to [-1, 1] range
                    delta_v_norm = delta_v / max_speed
                    
                    # Compute relative distance (Delta_D)
                    # Normalize distance to [0, 1] based on perception radius
                    delta_d_norm = distance / self.perception_radius
                    
                    # Compute Time-To-Collision (TTC)
                    # TTC = distance / relative_velocity (only if approaching)
                    relative_velocity = abs(delta_v)
                    if relative_velocity > 0.1 and delta_v < 0:  # Vehicles approaching
                        ttc = distance / relative_velocity
                        max_ttc = self.perception_radius/max_speed
                        ttc_norm = min(ttc / max_ttc, 1.0)
                    else:
                        # Not approaching or stationary - set to max value
                        ttc_norm = 1.0
                    
                    # Store neighbor info: [delta_V, Delta_D, TTC, distance for sorting]
                    neighbors_info.append({
                        'delta_v': delta_v_norm,
                        'delta_d': delta_d_norm,
                        'ttc': ttc_norm,
                        'distance': distance
                    })
            
            # Sort neighbors by distance (closest first) and take top max_neighbours
            neighbors_info.sort(key=lambda x: x['distance'])
            neighbors_info = neighbors_info[:self.max_neighbours]
            
            # Add neighbor observations to vector
            for neighbor in neighbors_info:
                obs_vector.extend([
                    neighbor['delta_v'],
                    neighbor['delta_d'],
                    neighbor['ttc']
                ])
            
            # Pad with zeros if fewer than max_neighbours
            num_actual_neighbors = len(neighbors_info)
            if num_actual_neighbors < self.max_neighbours:
                padding_length = (self.max_neighbours - num_actual_neighbors) * self.neighbour_obs_features
                obs_vector.extend([0.0] * padding_length)
            
            return np.array(obs_vector, dtype=np.float32)
        
        except Exception as e:
            # Return zero observation if error occurs
            print(f"Error getting observation for {ego_id}: {e}")
            total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
            return np.zeros(total_obs_len, dtype=np.float32)


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
        if fail:
            return -150.0               # still large, but not apocalyptic

        if goal_reached:
            return +120.0               # slightly higher incentive

        speed = self.k.vehicle.get_speed(agent_id)
        max_speed = self.k.network.max_speed()

        # Dense speed reward — encourage being close to target speed
        target_speed = 0.85 * max_speed          # slightly below max to avoid constant acceleration
        speed_reward = -0.5 * (speed - target_speed)**2 / (max_speed**2)   # quadratic → peaks at target

        # Very strong proximity penalty (near-miss shaping)
        ego_pos = np.array(self.k.vehicle.get_2d_position(agent_id))
        safety_penalty = 0.0

        for other_id in self.k.vehicle.get_rl_ids():
            if other_id == agent_id: continue
            other_pos = np.array(self.k.vehicle.get_2d_position(other_id))
            dist = np.linalg.norm(ego_pos - other_pos)

            if dist < 15.0:                     # larger zone than 6 m
                safety_penalty -= 8.0 * np.exp(-dist / 4.0)   # exponential decay, strong when very close

        # Small liveliness bonus (discourage stopping forever)
        if speed < 0.5:
            safety_penalty -= 0.3

        return speed_reward + safety_penalty

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
