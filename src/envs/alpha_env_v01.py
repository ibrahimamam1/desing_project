#============NEW ACCEL ENV ENVIRONMENT BY IBRAHIMA============

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from base_env_single import Env_N

class AlphaEnv_v01(Env_N):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
       
        #### V1
        # Ego-centric observation: S_ego = [d_norm, v_norm, cos θ, sin θ]
        self.ego_obs_features = 4
        # Per-neighbor (ego-relative): S_i = [dx, dy, v, cos Δθ, sin Δθ]
        self.neighbour_obs_features = 5
        
        ### V2
        # Ego-centric observation: S_ego = [d_norm, v_norm]
        # self.ego_obs_features = 2
        # Per-neighbor (ego-relative): S_i = [v, d, ttc]
        # self.neighbour_obs_features = 3

        ### V3 
        # Ego-centric observation: S_ego = [d_norm, v_norm, heading_norm]
        #self.ego_obs_features = 3
        # Per-neighbor (ego-relative): S_i = [v_norm, d_norm, heading_norm]
        #self.neighbour_obs_features = 3
        
        ### V4 
        # Ego-centric observation: S_ego = [d_norm, v_norm, sin , cos]
        #self.ego_obs_features = 4
        # Per-neighbor (ego-relative): S_i = [v_norm, d_norm, ttc, sin, cos]
        #self.neighbour_obs_features = 5
        

        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Initialize the static conflict map
        self.conflict_map = self._build_conflict_map()

        # Defining action space - KEEP NORMALIZED
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1, ), 
            dtype=np.float32)

        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
        self.observation_space = Box(
            low=-1.0,  
            high=1.0,   
            shape=(total_obs_len, ),
            dtype=np.float32)

        self.last_action = 0.0
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def get_state(self):
        """
        Return the observation for the single RL agent.
        Returns a flat np.array matching self.observation_space.
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        
        # If the vehicle despawned (reached goal or crashed), return the 
        # last valid observation to prevent value function spikes.
        if not self.agent_id in rl_ids:
            return self.last_obs

        # Get current observation
        obs = self._get_local_observation(self.agent_id)
        
        # Cache the valid observation for the terminal step
        self.last_obs = obs
        
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
        dis_to_goal_norm = np.clip(dis_to_goal / total_route_length, -1.0, 1.0)        # Ego Speed (normalized, clipped)
        
        ego_speed = self.k.vehicle.get_speed(ego_id)
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)
        
        # Ego heading (Convert SUMO North=0, CW to Math East=0, CCW)
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)    
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
        
        # Add heading to the ego observation vector (Now 4 features)
        obs_vector = [dis_to_goal_norm, ego_speed_norm, ego_sin, ego_cos]
        
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
            
            if self._is_conflicting(ego_id, other_id) and distance <= self.perception_radius:
                edge = self.k.vehicle.get_edge(other_id)
                
                # Neighbor speed (normalized)
                other_speed = self.k.vehicle.get_speed(other_id)
                other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)
                
                # Neighbor heading to Math angle
                other_heading = self.k.vehicle.get_heading(other_id)
                other_angle_rad = np.radians((-other_heading) + 90)
                other_sin = np.sin(other_angle_rad)
                other_cos = np.cos(other_angle_rad)
                
                # --- GEOMETRIC PROJECTION ---
                t, u = self._get_distances_to_collision_point(
                    ego_x, ego_y, ego_angle_rad,
                    other_x, other_y, other_angle_rad
                )
                
                # INTERSECTING PATHS
                if t is not None:
                    if t > -2.0 and u > -2.0:
                        d_raw = t - u
                        d = np.clip(d_raw / self.perception_radius, -1.0, 1.0)
                        
                        # Calculate TTC based on Ego's distance to the conflict point (t)
                        # We clip to 10 seconds max to normalize it properly.
                        raw_ttc = max(0.0, t) / (ego_speed + 1e-6)
                        ttc = np.clip(raw_ttc / 10.0, 0.0, 1.0) 
                    else:
                        d = 1.0  # Safe
                        ttc = 1.0 # Safe

                # PARALLEL PATHS (Leading / Following)
                else:
                    longitudinal_proj = dx_world * ego_cos + dy_world * ego_sin
                    sign = np.sign(longitudinal_proj)
                    d_raw = sign * abs(longitudinal_proj)
                    d = np.clip(d_raw / self.perception_radius, -1.0, 1.0)
                    
                    # TTC based on distance to the car directly ahead/behind
                    raw_ttc = abs(longitudinal_proj) / (ego_speed + 1e-6)
                    ttc = np.clip(raw_ttc / 10.0, 0.0, 1.0)

                # Append the 5 features
                neighbors_info.append({
                    'v': other_speed_norm,
                    'dx': dx_world/self.perception_radius,
                    'dy': dy_world/self.perception_radius,
                    'ttc': ttc,
                    'd': d,
                    'sinx': other_sin,
                    'cosx': other_cos,
                    'edge': edge,
                    'distance': distance,
                })

        # Sort by physical distance (closest first), take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
        
        print("=========NEIGHBOUR INFO==============") 
        for neighbor in neighbors_info:
            print(f'edge={neighbor["edge"]}, dist={neighbor["distance"]:.2f}, dx={neighbor["dx"]:.2f}, dy={neighbor["dy"]:.2f}, ttc={neighbor["ttc"]:.2f}, STC={neighbor["d"]:.2f}')
            obs_vector.extend([
                neighbor['v'],
                neighbor['d'],
                neighbor['ttc'],
                neighbor['sinx'],
                neighbor['cosx']
            ])
        
        # Pad if fewer than max_neighbours: [v=0, d=1(safe), ttc=1(safe), sin=0, cos=0]
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            missing_count = self.max_neighbours - num_actual
            for _ in range(missing_count):
                obs_vector.extend([0.0, 1.0, 1.0, 0.0, 0.0])
        
        # --- THE FAILSAFE ---
        obs_array = np.array(obs_vector, dtype=np.float32)
        
        if np.isnan(obs_array).any() or np.isinf(obs_array).any():
            print(f"WARNING: NaN/Inf generated in observation for {ego_id}! Sanitizing array.")
            # Convert NaNs to 0.0, +Inf to 1.0, -Inf to -1.0
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return obs_array 

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
    
    def _is_conflicting(self, veh1, veh2):
        """
        Determines if two vehicles have a conflicting path.
        Returns True if:
        1. They are currently on the same edge.
        2. Their routes (source to destination) share any common edges (e.g. merging or shared goal).
        """

        edge1 = self.k.vehicle.get_edge(veh1)
        edge2 = self.k.vehicle.get_edge(veh2)

        # --- Condition A: Currently on the same edge ---
        if edge1 == edge2:
            return True
        
        # Vehicles that are not on the same lane and either already crossed cannot be conflicting
        if edge1.startswith('E#X') or edge2.startswith('E#X'):
            return False  

        # 2. JUNCTION FIX: If either vehicle is inside the junction (internal SUMO edge), 
        # consider them conflicting. Your geometric projection will filter out the safe ones.
        if edge1.startswith(':') or edge2.startswith(':'):
            return True 
         
        # 3. Get Routes
        # Note: get_route returns a tuple/list of edges from current position to destination
        route1 = self.k.vehicle.get_route(veh1)
        route2 = self.k.vehicle.get_route(veh2)

        # If routes are unavailable for some reason, assume no conflict to avoid errors
        if not route1 or not route2:
            return False

        # Extract (Source, Destination) tuple for both vehicles
        # route[0] is source edge, route[-1] is destination edge
        pattern_1 = (route1[0], route1[-1])
        pattern_2 = (route2[0], route2[-1])

        # Retrieve the list of patterns that conflict with Vehicle 1
        # defaults to empty list if pattern is not in map
        conflicting_patterns = self.conflict_map.get(pattern_1, [])

        if pattern_2 in conflicting_patterns:
            return True

        return False

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        action_val = float(rl_action)
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
        if fail:
            return -10.0  
        if goal_reached:
            return 15.0   
            
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0

        speed = self.k.vehicle.get_speed(agent_id)
        max_speed = self.k.network.max_speed()

        # Step rewards balanced to prevent "suicide loophole"
        speed_reward = 0.05 * (speed / max_speed)
        time_penalty = -0.02 
        
        # Action smoothness penalty
        action_penalty = 0.0
        if current_action is not None:
            action_val = float(current_action[0])
            action_penalty = -0.02 * abs(action_val - self.last_action)
            self.last_action = action_val
        
        return speed_reward + time_penalty + action_penalty

    def _build_conflict_map(self):
        """
        Statically maps a (Source, Destination) pair to a list of conflicting 
        (Source, Destination) pairs.
        
        Format:
        {
            (My_Source, My_Dest): [ (Enemy_Source, Enemy_Dest), ... ]
        }
        """
        # REPLACE THESE WITH YOUR REAL SUMO EDGE IDs
        # Example assumes a 4-way intersection
        N_in, N_out = 'E#T-X', 'E#X-T'
        S_in, S_out = 'E#D-X', 'E#X-D'
        E_in, E_out = 'E#R-X',  'E#X-R'
        W_in, W_out = 'E#L-X',  'E#X-L'

        # Define Flows (Source, Destination)
        # Straight Flows
        NS = (N_in, S_out) # North to South
        SN = (S_in, N_out) # South to North
        EW = (E_in, W_out) # East to West
        WE = (W_in, E_out) # West to East
        
        # Left Turns (usually conflict with straights)
        NE = (N_in, E_out) # North turning Left to East
        SW = (S_in, W_out) # South turning Left to West
        WN = (W_in, N_out) # West turning Left to North
        ES = (E_in, S_out) # East turning Left to South

        # The Conflict Dictionary
        mapping = {}

        # 1. North-South Straight Conflicts
        # Conflicts with: West-East, East-West, and Left turns crossing it
        mapping[NS] = [WE, EW, SW, ES] 
        mapping[SN] = [WE, EW, NE, WN]

        # 2. East-West Straight Conflicts
        # Conflicts with: North-South, South-North, and Left turns crossing it
        mapping[EW] = [NS, SN, NE, SW]
        mapping[WE] = [NS, SN, ES, WN]

        # 3. Left Turn Conflicts 
        # (Conflict with oncoming straight and crossing straights)
        mapping[NE] = [SN, WE, EW] 
        mapping[SW] = [NS, WE, EW]
        mapping[WN] = [ES, NS, SN]
        mapping[ES] = [WN, NS, SN]

        return mapping 

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
