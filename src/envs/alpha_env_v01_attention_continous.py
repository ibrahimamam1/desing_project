#============ATTENTION-BASED VARIANT OF V0.1 ENV (NO CONFLICT HEURISTIC)============

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env_single import Env_N

class AlphaEnv_v01_Attention(Env_N):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 8
        self.perception_radius = 80
       
        # Ego-centric observation: S_ego = [d_norm, v_norm, cos θ, sin θ]
        self.ego_obs_features = 4
        # Per-neighbor (ego-relative): S_i = [dist_to_cp, v, delta_eta, cos Δθ, sin Δθ]
        self.neighbour_obs_features = 5
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1, ), 
            dtype=np.float32)

        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours) + self.max_neighbours
        self.observation_space = Box(
            low=-1.0,  
            high=1.0,   
            shape=(total_obs_len, ),
            dtype=np.float32)

        self.last_action = 0.0
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def get_state(self):
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.agent_id not in rl_ids:
            return self.last_obs
     
        obs, neighbors_info = self._get_local_observation(self.agent_id)
        self.last_obs = obs
        self.last_neighbors_info = neighbors_info  # cache for terminal step
        return obs

    def _get_local_observation(self, ego_id):
        # --- 1. Ego State ---
        # Ego Distance to goal
        route = self.k.vehicle.get_route(ego_id)
        total_route_length = sum([self.k.network.edge_length(edge) for edge in route])
        total_route_length = max(total_route_length, 1e-4)

        ego_dis = self.k.vehicle.get_distance(ego_id)
        # Also protect against Flow's default -1001 for missing vehicles
        if ego_dis == -1001: 
            ego_dis = 0.0 

        dis_to_goal = total_route_length - ego_dis
        dis_to_goal = np.clip(dis_to_goal / total_route_length, 0, 1.0)        # Ego Speed (normalized, clipped)
        
        ego_speed = self.k.vehicle.get_speed(ego_id)
        max_speed = self.k.network.max_speed()
        ego_speed = np.clip(ego_speed / max_speed, 0, 1.0)
        
        # Ego heading (Convert SUMO North=0, CW to Math East=0, CCW)
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)    
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
        
        # Add heading to the ego observation vector (Now 4 features)
        obs_vector = [dis_to_goal, ego_speed, ego_sin, ego_cos]
        
        # --- 2. Neighbor States (ego-centric relative frame) ---
        neighbors_info = []
        all_ids = self.k.vehicle.get_ids()
        
        pos_ret = self.k.vehicle.get_2d_position(ego_id)
        if pos_ret is None or pos_ret == -1001 or pos_ret == (-1001.0, -1001.0):
            return self.last_obs 
        ego_x, ego_y = pos_ret

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
                edge = self.k.vehicle.get_edge(other_id)
                
                # Neighbor speed (normalized)
                other_speed = self.k.vehicle.get_speed(other_id)
                other_speed = np.clip(other_speed / max_speed, 0.0, 1.0)
                
                # Neighbor heading to Math angle
                other_heading = self.k.vehicle.get_heading(other_id)
                other_angle_rad = np.radians((-other_heading) + 90)
                other_sin = np.sin(other_angle_rad)
                other_cos = np.cos(other_angle_rad)
                
                # --- Compute conflict point ---
                vx_ego, vy_ego = ego_cos, ego_sin
                vx_other, vy_other = other_cos, other_sin

                det = (-ego_cos * other_sin) + (ego_sin * other_cos)

                if abs(det) < 0.05:
                    # If parallel, ego is following other on same lane
                    dx = other_x - ego_x
                    dy = other_y - ego_y
                
                    # Dot product gives the projection (longitudinal distance)
                    # t1 is how far ego must travel to reach 'other'
                    t1 = dx * vx_ego + dy * vy_ego
                
                    # In a following scenario, the lead vehicle is already "at" the conflict
                    # relative to its own path start, so we set its distance to 0.
                    other_dist_to_cp = 0.0
                
                    # Apply a 5.0m buffer for the lead vehicle's physical length
                    ego_dist_to_cp = max(0, t1)
                    ego_dist_to_cp = np.clip(ego_dist_to_cp/self.perception_radius, 0, 1)
                else:  # Intersecting Case
                    dx = other_x - ego_x
                    dy = other_y - ego_y
                
                    t1 = (dx * (-vy_other) - dy * (-vx_other)) / det
                    t2 = (dx * vy_ego - dy * vx_ego) / det
            
                    # Lane width buffer (1.5m offset from center of 3m lane)
                    ego_dist_to_cp = max(0, t1)
                    other_dist_to_cp = max(0, t2)
            
                    # Normalise dist_to_cp
                    ego_dist_to_cp = np.clip(ego_dist_to_cp/self.perception_radius, 0, 1)
                    other_dist_to_cp = np.clip(other_dist_to_cp/self.perception_radius, 0, 1)
                    edge = self.k.vehicle.get_edge(other_id)
            
                rel_speed = ego_speed - other_speed 

            
                # 3. Delta ETA (Difference in arrival times at Conflict Point)
                ego_eta = ego_dist_to_cp / max(ego_speed, 0.5)
                other_eta = other_dist_to_cp / max(other_speed, 0.5)
                delta_eta = ego_eta - other_eta
                delta_eta_norm =  np.tanh(delta_eta / 2.0)

                neighbors_info.append({
                    'ego_dist_to_cp':        ego_dist_to_cp,
                    'v':        other_speed,
                    'delta_eta':        delta_eta_norm,
                    'sin':     other_sin,
                    'cos':     other_cos,
                    'edge':     edge,
                    'distance': distance,
                })

        # Sort by physical distance (closest first), take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
        
        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor['ego_dist_to_cp'],
                neighbor['v'],
                neighbor['delta_eta'],
                neighbor['sin'],
                neighbor['cos'],
            ])
        
        # Pad if fewer than max_neighbours: [ego_d_to_cp=1(safe), v=0, ttc=1(safe), delta_eta=1(safe), sin=0, cos=0]
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            missing_count = self.max_neighbours - num_actual
            for _ in range(missing_count):
                obs_vector.extend([1.0, 0.0, 1.0, 0.0, 0.0])
        
        # Append neighbor mask: 1.0 = real neighbor, 0.0 = padded
        neighbor_mask = [1.0] * num_actual + [0.0] * (self.max_neighbours - num_actual)
        obs_vector.extend(neighbor_mask)
        

        # --- THE FAILSAFE ---
        obs_array = np.array(obs_vector, dtype=np.float32)
        return obs_array, neighbors_info 

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        # 1. Safely extract and sanitize the action
        try:
            action_val = float(rl_action[0]) if isinstance(rl_action, (list, np.ndarray)) else float(rl_action)
        except (TypeError, ValueError):
            action_val = 0.0
            
        if np.isnan(action_val) or np.isinf(action_val):
            action_val = 0.0  # Fallback to zero acceleration if NaN

        # Denormalize from [-1, 1] to [-max_decel, max_accel]
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        rl_ids = [self.agent_id]
        if self.agent_id in self.k.vehicle.get_ids():
            self.k.vehicle.apply_acceleration(rl_ids, [real_action])

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        if fail: return -10.0  
        if goal_reached: return 15.0   
            
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        speed = self.k.vehicle.get_speed(agent_id)
        # Prevent NaN speed from SUMO glitch
        if speed == -1001 or np.isnan(speed): 
            speed = 0.0

        max_speed = self.k.network.max_speed()

        speed_reward = 0.05 * (speed / max_speed)
        time_penalty = -0.02 
        
        action_penalty = 0.0
        if current_action is not None:
            # 2. Sanitize action penalty calculation
            action_val = float(current_action[0]) if isinstance(current_action, (list, np.ndarray)) else float(current_action)
            if not np.isnan(action_val) and not np.isinf(action_val):
                action_penalty = -0.02 * abs(action_val - self.last_action)
                self.last_action = action_val
        
        total_reward = speed_reward + time_penalty + action_penalty
        
        # 3. Final failsafe before handing reward back to RLlib
        if np.isnan(total_reward) or np.isinf(total_reward):
            return 0.0
            
        return total_reward

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
