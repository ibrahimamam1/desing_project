
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
import gymnasium as gym  # Or 'import gym' depending on your Ray/Flow version
from copy import deepcopy

# Assuming Env_N is defined as provided in your context

class AlphaEnv(Env_N):
    """
    Multi-Agent Alpha environment.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        # Variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        
        # Define how many features we track per vehicle 
        # ego = (V, d, p, a)
        # neighbour = (delta_v, delta_d, TTC)
        self.ego_obs_features = 4 
        self.neighbour_obs_features = 3
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space
        self.action_space = gym.spaces.Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space
        total_obs_len = self.ego_obs_features + (self.max_neighbours * self.neighbour_obs_features)
        self.observation_space = gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(total_obs_len, ),
            dtype=np.float32)
   
    def get_state(self):
        """
        Return the state of the simulation as a DICTIONARY.
        Format: { "veh_id_1": [obs_array], "veh_id_2": [obs_array] }
        """
        obs_dict = {}
        # Only get observations for RL controlled vehicles
        for veh_id in self.k.vehicle.get_rl_ids():
            # Check if active and in control zone
            if veh_id in self.k.vehicle.get_ids() and self._is_in_control_zone(veh_id):
                obs_dict[veh_id] = self._get_local_observation(veh_id)
            
        return obs_dict

    def _get_local_observation(self, ego_id):
        """
        Internal helper to build the observation array for ONE agent.
        """
        # --- 1. Get Ego State ---
        
        # Normalized speed
        max_speed = self.k.network.max_speed()
        ego_speed = self.k.vehicle.get_speed(ego_id) / max_speed
        
        # Normalized Accel (using realized acceleration from previous step)
        # We divide by max_accel to normalize approx between -1 and 1
        max_accel = self.env_params.additional_params.get('max_accel', 3.0)
        realized_accel = self.k.vehicle.get_realized_accel(ego_id)
        ego_accel = realized_accel / max_accel if max_accel > 0 else 0.0
        
        # Ego path encoding
        # We hash the (Origin, Destination) edge pair to an integer 0-11
        try:
            route = self.k.vehicle.get_route(ego_id)
            if len(route) > 0:
                # Simple consistent hash for the route
                start_edge = route[0]
                end_edge = route[-1]
                # Combine characters of edge IDs to generate a deterministic int
                route_hash = (hash(start_edge) + hash(end_edge)) % 12
                ego_path = route_hash
            else:
                ego_path = 0
        except Exception:
            ego_path = 0
        
        # Distance ego traveled along its path
        # get_distance returns total distance traveled since insertion
        ego_dis = self.k.vehicle.get_distance(ego_id)
        
        # Start the vector
        obs_vector = [ego_speed, ego_accel, ego_dis, ego_path]
        
        # --- 2. Find Neighbors ---
        
        ego_pos = np.array(self.k.vehicle.get_2d_position(ego_id))
        all_ids = self.k.vehicle.get_ids()
        
        # Helper to calculate Euclidean distance
        def get_dist_sq(other_id):
            o_pos = np.array(self.k.vehicle.get_2d_position(other_id))
            return np.sum((ego_pos - o_pos)**2)

        # Filter neighbors:
        # 1. Not self
        # 2. Within perception radius
        others = []
        for vid in all_ids:
            if vid == ego_id:
                continue
            dist_sq = get_dist_sq(vid)
            if dist_sq <= self.perception_radius**2:
                others.append((vid, dist_sq))

        # Sort by distance (closest first)
        others.sort(key=lambda x: x[1])
        
        # Take top N neighbors (unzip to get just IDs)
        closest_neighbors_ids = [x[0] for x in others[:self.max_neighbours]]
        
        # --- 3. Add Neighbor States ---
        
        for neigh_id in closest_neighbors_ids:
            neigh_speed = self.k.vehicle.get_speed(neigh_id) / max_speed
            
            # Relative Speed (Delta V)
            rel_neigh_speed = ego_speed - neigh_speed
            
            # Relative Distance (Delta D)
            # Using Euclidean distance as "distance to conflict" proxy
            # Ideally, this should be distance along the lane, but Euclidean is robust
            dist = np.sqrt(get_dist_sq(neigh_id))
            rel_neigh_dis = dist 
            
            # Time To Collision (TTC)
            # TTC = Distance / Relative Speed (approaching speed)
            # If rel_neigh_speed is positive, ego is faster (approaching rear) or moving toward head-on
            # We treat TTC as a positive value. High value if moving away.
            if abs(rel_neigh_speed) < 1e-3:
                neigh_ttc = 100.0 # Effectively infinity
            else:
                # We normalize TTC roughly so it fits in NN inputs (e.g., cap at 10s)
                raw_ttc = dist / (abs(rel_neigh_speed) * max_speed) # denorm speed for calc
                neigh_ttc = min(raw_ttc, 10.0)

            obs_vector.extend([rel_neigh_speed, rel_neigh_dis, neigh_ttc])
            
        # --- 4. Padding ---
        # If we have fewer neighbors than max_neighbours, pad with zeros
        needed_length = self.observation_space.shape[0]
        current_length = len(obs_vector)
        
        if current_length < needed_length:
            padding = [0.0] * (needed_length - current_length)
            obs_vector.extend(padding)
            
        return np.array(obs_vector, dtype=np.float32)

    def _apply_rl_actions(self, rl_actions_dict):
        """
        Apply RL actions to the vehicles.
        Input: Dictionary { "veh_id": action_value }
        """
        if not rl_actions_dict:
            return

        ids_to_apply = []
        actions_to_apply = []

        # Iterate through the dictionary provided by RLlib
        for veh_id, action in rl_actions_dict.items():
            if self._is_in_control_zone(veh_id):
                # print(f'{veh_id} control by Agent')
                ids_to_apply.append(veh_id)
                # Unpack the action (it comes as a list/array from RLlib)
                if isinstance(action, (list, np.ndarray)):
                    actions_to_apply.append(action[0])
                else:
                    actions_to_apply.append(action)
            else:
                # Vehicle is outside control zone, do nothing let SUMO control
                # print(f'{veh_id} control by SUMO')
                pass

        if ids_to_apply:
            self.k.vehicle.apply_acceleration(ids_to_apply, actions_to_apply)

    def compute_reward(self, veh_id, rl_action, **kwargs):
        reward = 0
        
        # 1. Collision is the ultimate failure
        if kwargs.get("fail", False):
            return -50.0 

        # 2. Encourage speed (Progress toward goal)
        # We want them to move at the target/max speed
        max_speed = self.k.network.max_speed()
        v_ego = self.k.vehicle.get_speed(veh_id)
        
        # Simple linear reward for speed (normalized 0 to 1)
        reward += (v_ego / max_speed) * 0.5 

        # 3. Small penalty for excessive braking (improves traffic flow)
        if rl_action < 0:
            reward -= 0.05 * abs(rl_action)

        return reward

    def additional_command(self):
        """
        See parent class.
        Update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_human_ids():
            self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    # Helper for logic if needed, though mostly handled by get_state now
    def _get_abs_position(self, veh_id):
        return self.absolute_position.get(veh_id, -1001)
