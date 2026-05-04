import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env_multi import Env_Multi

class AlphaEnv_v03(Env_Multi):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        
        # Ego-centric observation: S_ego = [d_norm, v_norm]
        self.ego_obs_features = 2
        # Per-neighbor (ego-relative): S_i = [dv, dd]
        self.neighbour_obs_features = 2
        
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

        # Multi-agent historical tracking
        self.last_action = {}
        self.last_obs = {}

    def get_state(self, agent_id):
        """
        Return the observation for a specific RL agent.
        """
        # Get current observation for this specific agent
        obs = self._get_local_observation(agent_id)
        
        # Cache the valid observation
        self.last_obs[agent_id] = obs
        
        return obs

    def _get_local_observation(self, ego_id):
        # --- 1. Ego State ---
        # Ego Distance to goal
        route = self.k.vehicle.get_route(ego_id)
        total_route_length = sum([self.k.network.edge_length(edge) for edge in route])

        ego_dis = self.k.vehicle.get_distance(ego_id) # distance traveled by vehicle
        dis_to_goal = total_route_length - ego_dis
        dis_to_goal_norm = np.clip(dis_to_goal / total_route_length, -1, 1) # normalise distance to goal

        # Ego Speed (normalized, clipped)
        ego_speed = self.k.vehicle.get_speed(ego_id)
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)
        
        # Ego heading (SUMO heading → standard math angle)
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
        
        # Convert SUMO heading (North=0, CW) to Math (East=0, CCW)
        ego_angle_rad = np.radians((-ego_heading) + 90)    
        
        obs_vector = [dis_to_goal_norm, ego_speed_norm]
        
        # --- 2. Neighbor States (ego-centric relative frame) ---
        neighbors_info = []
        all_ids = self.k.vehicle.get_ids()
        
        pos_ret = self.k.vehicle.get_2d_position(ego_id)
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
                # Rotate world-frame delta into ego frame
               # dx_ego = dx_world * ego_cos + dy_world * ego_sin
               # dy_ego = -dx_world * ego_sin + dy_world * ego_cos
               # 
               # # Normalize relative position by perception radius → [-1, 1]
               # dx_norm = np.clip(dx_ego / self.perception_radius, -1.0, 1.0)
               # dy_norm = np.clip(dy_ego / self.perception_radius, -1.0, 1.0)

                # Neighbor speed (normalized)
                other_speed = self.k.vehicle.get_speed(other_id)
                other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)
               
                other_ttc_norm = (distance / self.perception_radius) / (np.abs(ego_speed_norm - other_speed_norm))
                
                # Relative heading (neighbor heading - ego heading)
                other_heading = self.k.vehicle.get_heading(other_id)
                other_angle_rad = np.radians((-other_heading) + 90)
                
                # --- GEOMETRIC PROJECTION ---
                t, u = self._get_distances_to_collision_point(
                    ego_x, ego_y, ego_angle_rad,
                    other_x, other_y, other_angle_rad
                )
                
                # Default "safe" values (max distance)
                d_ego_conf = self.perception_radius
                d_other_conf = self.perception_radius 
                
               # INTERSECTING PATHS
                if t is not None:
                    # We only care if intersection is in the FUTURE (t > 0, u > 0)
                    if t > -2.0 and u > -2.0:
                        d_ego_conf = t
                        d_other_conf = u
                    else:
                        d_ego_conf = self.perception_radius
                        d_other_conf = self.perception_radius 
                    
                    # Normalize [-1, 1]
                    d_ego_norm = np.clip(d_ego_conf / self.perception_radius, 0.0, 1.0)
                    d_other_norm = np.clip(d_other_conf / self.perception_radius, 0.0, 1.0)
                    d = d_ego_norm - d_other_norm

                # PARALLEL PATHS (Leading / Following)
                else:
                    # Project relative position onto Ego's forward direction
                    longitudinal_proj = dx_world * ego_cos + dy_world * ego_sin
                    
                    # Determine sign: 
                    # +1 if neighbor is in front (Ego is behind)
                    # -1 if neighbor is behind (Ego is ahead)
                    sign = np.sign(longitudinal_proj)
                    
                    # d = Signed Euclidean Distance
                    d_raw = sign * distance
                    
                    # Normalize by perception radius to keep input within [-1, 1] range
                    # If neighbor is 20m behind and radius is 50m -> d = -0.4
                    d = np.clip(d_raw / self.perception_radius, -1.0, 1.0)

                # delta_angle = other_angle_rad - ego_angle_rad
               # cos_delta = np.cos(delta_angle)
               # sin_delta = np.sin(delta_angle)
                
                neighbors_info.append({
                    #'dx': dx_norm,
                    #'dy': dy_norm,
                    'v': other_speed_norm,
                    #'cos_delta': cos_delta,
                    #'sin_delta': sin_delta,
                    'd': d,
                    'distance': distance,
                })
        
        # Sort by distance (closest first), take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
        
        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor['v'],
                neighbor['d'],
            ])
        
        # Pad with zeros if fewer than max_neighbours
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            missing_count = self.max_neighbours - num_actual
            for _ in range(missing_count):
                obs_vector.extend([0.0, 1.0])
        
        # Append neighbor mask: 1.0 = real neighbor, 0.0 = padded
        neighbor_mask = [1.0] * num_actual + [0.0] * (self.max_neighbours - num_actual)
        obs_vector.extend(neighbor_mask)
        
        return np.array(obs_vector, dtype=np.float32)
    
    def _get_distances_to_collision_point(self, x1, y1, theta1, x2, y2, theta2):
        """
        Calculates the distance from (x1, y1) to the collision point and 
        (x2, y2) to the collision point.
        theta1, theta2 are in standard radians (counter-clockwise from East).
        
        Returns:
            t (float): Distance for vehicle 1 to collision.
            u (float): Distance for vehicle 2 to collision.
            Returns (None, None) if lines are parallel.
        """
        # Direction vectors
        dx1 = np.cos(theta1)
        dy1 = np.sin(theta1)
        dx2 = np.cos(theta2)
        dy2 = np.sin(theta2)

        # Determinant (cross product of direction vectors)
        # det = dx1 * dy2 - dx2 * dy1
        # This is equivalent to sin(theta2 - theta1)
        det = dx1 * dy2 - dy1 * dx2

        # Check for parallel lines (det close to 0)
        if abs(det) < 1e-6:
            return None, None

        # Delta position
        dx_delta = x2 - x1
        dy_delta = y2 - y1

        # Cramers rule / Cross product solution for t and u
        # t = (Delta x Dir2) / det
        t = (dx_delta * dy2 - dy_delta * dx2) / det
        
        # u = (Delta x Dir1) / det
        u = (dx_delta * dy1 - dy_delta * dx1) / det

        return t, u
    
    def _apply_rl_actions(self, rl_actions_dict):
        """
        Applies a dictionary of actions to their respective agents.
        """
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        target_ids = []
        target_actions = []

        for agent_id, action in rl_actions_dict.items():
            if agent_id not in self.k.vehicle.get_ids():
                continue

            # In gym, Box(1,) shape means action is usually an array/list
            action_val = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            
            # Denormalize from [-1, 1] to [-max_decel, max_accel]
            if action_val >= 0:
                real_action = action_val * max_accel
            else:
                real_action = action_val * max_decel
            
            target_ids.append(agent_id)
            target_actions.append(real_action)

        # Apply batch accelerations
        if target_ids:
            self.k.vehicle.apply_acceleration(target_ids, target_actions)

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        """
        Calculates the reward for a specific agent.
        """
        # Clean up cache on exit to prevent memory leaks in long episodes
        if fail or goal_reached:
            self.last_action.pop(agent_id, None)
            self.last_obs.pop(agent_id, None)

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
        
        # Action smoothness penalty
        action_penalty = 0.0
        if current_action is not None:
            action_val = float(current_action[0]) if isinstance(current_action, (list, np.ndarray)) else float(current_action)
            
            prev_action = self.last_action.get(agent_id, 0.0)
            action_penalty = -0.02 * abs(action_val - prev_action)
            
            # Update cache for next step
            self.last_action[agent_id] = action_val
        
        return speed_reward + time_penalty + action_penalty


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
