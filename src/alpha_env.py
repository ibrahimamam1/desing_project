
#============NEW ACCEL ENV ENVIRONMENT BY IBRAHIMA============
from flow.core import rewards

import gymnasium as gym
import numpy as np
import random
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env import Env_N
class AlphaEnv(Env_N):
    """
    Multi-Agent Alpha environment.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        
        # Define how many features we track per vehicle (Speed, x, y, angle, intention)
        self.num_obs_features = 5 
        
        super().__init__(env_params, sim_params, network, simulator)
        
        #Defining action space
        self.action_space = gym.spaces.Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space
        total_obs_len = (1 + self.max_neighbours) * self.num_obs_features
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
            obs_dict[veh_id] = self._get_local_observation(veh_id)
            
        return obs_dict

    def _get_local_observation(self, ego_id):
        """
        Internal helper to build the observation array for ONE agent.
        """
        # 1. Get Ego State
        # Normalized speed
        ego_speed = self.k.vehicle.get_speed(ego_id) / self.k.network.max_speed()
        
        # Position normalization
        pos = self.k.vehicle.get_2d_position(ego_id)
        max_len = self.k.network.length()
        ego_x = pos[0] / max_len
        ego_y = pos[1] / max_len
        
        ego_angle = self.k.vehicle.get_heading(ego_id) / 360.0
        ego_intention = 1.0 # Placeholder from original code
        
        # Start the vector
        obs_vector = [ego_speed, ego_x, ego_y, ego_angle, ego_intention]
        
        # 2. Find Neighbors
        # We find the closest vehicles to the ego vehicle
        all_ids = self.k.vehicle.get_ids()
        
        # Simple distance lambda
        def get_dist(other_id):
            o_pos = self.k.vehicle.get_2d_position(other_id)
            return np.sqrt((pos[0] - o_pos[0])**2 + (pos[1] - o_pos[1])**2)

        # Filter out self, sort by distance
        others = [vid for vid in all_ids if vid != ego_id]
        others.sort(key=get_dist)
        
        # Take top N neighbors
        closest_neighbors = others[:self.max_neighbours]
        
        # 3. Add Neighbor States
        for neigh_id in closest_neighbors:
            n_speed = self.k.vehicle.get_speed(neigh_id) / self.k.network.max_speed()
            n_pos = self.k.vehicle.get_2d_position(neigh_id)
            n_x = n_pos[0] / max_len
            n_y = n_pos[1] / max_len
            n_angle = self.k.vehicle.get_heading(neigh_id) / 360.0
            n_intention = 1.0
            
            obs_vector.extend([n_speed, n_x, n_y, n_angle, n_intention])
            
        # 4. Padding (Zero-pad if we don't have enough neighbors)
        needed_length = self.observation_space.shape[0]
        current_length = len(obs_vector)
        
        if current_length < needed_length:
            padding = [0.0] * (needed_length - current_length)
            obs_vector.extend(padding)
            
        return np.array(obs_vector, dtype=np.float32)

    def _apply_rl_actions(self, rl_actions_dict):
        """
        Apply RL actions to the vehicles.
        
        Input: Dictionary { "veh_id": [action_value] }
        """
        if not rl_actions_dict:
            return

        ids_to_apply = []
        actions_to_apply = []

        # Iterate through the dictionary provided by RLlib
        for veh_id, action in rl_actions_dict.items():
            
            # OPTIONAL: Keep your "Control Zone" logic if desired
            # This replicates the logic from your original step() function
            position = self.k.vehicle.get_2d_position(veh_id)
            in_box_x = -12 <= position[0] <= 12
            in_box_y = -12 <= position[1] <= 12
            
            if in_box_x and in_box_y:
                ids_to_apply.append(veh_id)
                # Unpack the action (it comes as a list/array from RLlib)
                if isinstance(action, (list, np.ndarray)):
                    actions_to_apply.append(action[0])
                else:
                    actions_to_apply.append(action)
            else:
                # Vehicle is outside control zone, do nothing (or let SUMO control)
                pass

        if ids_to_apply:
            self.k.vehicle.apply_acceleration(ids_to_apply, actions_to_apply)

    def compute_reward(self, veh_id, rl_action, **kwargs):
        """
        Compute the reward for a specific agent.
        """
        # Basic penalty for time step
        reward = -0.1
        
        # Collision penalty
        # Note: 'fail' is passed from the base class step() method
        if kwargs.get("fail", False):
            reward -= 100

        # Optional: Add goal reward if vehicle reaches specific edge/position
        # if self.k.vehicle.get_edge(veh_id) == "goal_edge":
        #     reward += 10
            
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
