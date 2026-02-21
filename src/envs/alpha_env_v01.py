#============NEW ACCEL ENV ENVIRONMENT BY IBRAHIMA============

import gymnasium as gym
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env import Env_N

class AlphaEnv_v01(Env_N):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        
        # Ego-centric observation: S_ego = [v_norm, cos θ, sin θ]
        self.ego_obs_features = 3
        # Per-neighbor (ego-relative): S_i = [dx, dy, v, cos Δθ, sin Δθ]
        self.neighbour_obs_features = 5
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, ), 
            dtype=np.float32)

        # Define Observation Space: 3 ego + 5*5 neighbors = 28
        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
        self.observation_space = gym.spaces.Box(
            low=-1.0,  
            high=1.0,   
            shape=(total_obs_len, ),
            dtype=np.float32)

        # Reward shaping parameters
        self.safe_distance = 10.0  # meters — proximity penalty kicks in below this
   
    def get_state(self):
        """
        Return the observation for the single RL agent.
        Returns a flat np.array matching self.observation_space.
        """
        # If RL_0 is no longer in the network, return zeros (terminal step)
        if 'RL_0' not in self.k.vehicle.get_ids():
            total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
            return np.zeros(total_obs_len, dtype=np.float32)

        return self._get_local_observation('RL_0')

    def _get_local_observation(self, ego_id):
        # --- 1. Ego State ---
        pos_ret = self.k.vehicle.get_2d_position(ego_id)
        ego_x, ego_y = pos_ret

        # Ego Speed (normalized, clipped)
        ego_speed = self.k.vehicle.get_speed(ego_id)
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)
        
        # Ego heading (SUMO heading → standard math angle)
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
        
        # Ego-centric obs: only speed and heading (position is the reference frame)
        obs_vector = [ego_speed_norm, ego_cos, ego_sin]
        
        # --- 2. Neighbor States (ego-centric relative frame) ---
        neighbors_info = []
        all_ids = self.k.vehicle.get_ids()
        
        for other_id in all_ids:
            if other_id == ego_id:
                continue
            
            other_pos = self.k.vehicle.get_2d_position(other_id)
            if other_pos is None or other_pos == -1001:
                continue

            other_x, other_y = other_pos
            dx_world = other_x - ego_x
            dy_world = other_y - ego_y
            distance = np.sqrt(dx_world**2 + dy_world**2)
            
            if distance <= self.perception_radius:
                # Rotate world-frame delta into ego frame
                dx_ego = dx_world * ego_cos + dy_world * ego_sin
                dy_ego = -dx_world * ego_sin + dy_world * ego_cos
                
                # Normalize relative position by perception radius → [-1, 1]
                dx_norm = np.clip(dx_ego / self.perception_radius, -1.0, 1.0)
                dy_norm = np.clip(dy_ego / self.perception_radius, -1.0, 1.0)

                # Neighbor speed (normalized)
                other_speed = self.k.vehicle.get_speed(other_id)
                other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)
                
                # Relative heading (neighbor heading - ego heading)
                other_heading = self.k.vehicle.get_heading(other_id)
                other_angle_rad = np.radians((-other_heading) + 90)
                delta_angle = other_angle_rad - ego_angle_rad
                cos_delta = np.cos(delta_angle)
                sin_delta = np.sin(delta_angle)
                
                neighbors_info.append({
                    'dx': dx_norm,
                    'dy': dy_norm,
                    'v': other_speed_norm,
                    'cos_delta': cos_delta,
                    'sin_delta': sin_delta,
                    'distance': distance,
                })
        
        # Sort by distance (closest first), take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
        
        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor['dx'],
                neighbor['dy'],
                neighbor['v'],
                neighbor['cos_delta'],
                neighbor['sin_delta'],
            ])
        
        # Pad with zeros if fewer than max_neighbours
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            padding = (self.max_neighbours - num_actual) * self.neighbour_obs_features
            obs_vector.extend([0.0] * padding)
        
        return np.array(obs_vector, dtype=np.float32)

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        action_val = float(rl_action)
        # Denormalize from [-1, 1] to [-max_decel, max_accel]
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        rl_ids = self.sorted_ids
        if not rl_ids:
            return
        self.k.vehicle.apply_acceleration(rl_ids, [real_action])

    def compute_reward(self, agent_id, fail, goal_reached):
        if fail:
            return -10.0

        if goal_reached:
            return +10.0
        
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        speed = self.k.vehicle.get_speed(agent_id)
        max_speed = self.k.network.max_speed()

        # 1. Speed reward: encourage progress (0.0 to 1.0)
        speed_reward = 1.0 * (speed / max_speed)
        
        return speed_reward

    def additional_command(self):
        """
        Update the sorting of vehicles using the self.sorted_ids variable.
        """
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

    def _get_abs_position(self, veh_id):
        return self.absolute_position.get(veh_id, -1001)
