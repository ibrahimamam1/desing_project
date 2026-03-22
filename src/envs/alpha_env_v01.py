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
        self.last_neighbors_info = []

    def get_state(self):
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.agent_id not in rl_ids:
            return self.last_obs, self.last_neighbors_info  # also cache neighbors
     
        obs, neighbors_info = self._get_local_observation(self.agent_id)
        self.last_obs = obs
        self.last_neighbors_info = neighbors_info  # cache for terminal step
        return obs

    def _get_local_observation(self, ego_id):

        # --- 1. Ego State ---
        route = self.k.vehicle.get_route(ego_id)
        total_route_length = sum([self.k.network.edge_length(edge) for edge in route])
        total_route_length = max(total_route_length, 1e-4)
    
        ego_dis = self.k.vehicle.get_distance(ego_id)
        if ego_dis == -1001:
            ego_dis = 0.0
    
        dis_to_goal = total_route_length - ego_dis
        dis_to_goal_norm = np.clip(dis_to_goal / total_route_length, -1.0, 1.0)
    
        ego_speed = self.k.vehicle.get_speed(ego_id)
        if ego_speed is None or ego_speed < 0:
            ego_speed = 0.0
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)
    
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
    
        obs_vector = [dis_to_goal_norm, ego_speed_norm, ego_sin, ego_cos]
    
        # --- 2. Neighbor States (Frenet-based) ---
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
            distance = np.sqrt((other_x - ego_x)**2 + (other_y - ego_y)**2)
    
            if not (self._is_conflicting(ego_id, other_id) and distance <= self.perception_radius):
                continue
    
            # Neighbor speed
            other_speed = self.k.vehicle.get_speed(other_id)
            if other_speed is None or other_speed < 0:
                other_speed = 0.0
            other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)
    
            # Neighbor heading
            other_heading = self.k.vehicle.get_heading(other_id)
            other_angle_rad = np.radians((-other_heading) + 90)
            other_sin = np.sin(other_angle_rad)
            other_cos = np.cos(other_angle_rad)
    
            # --- FRENET CONFLICT ---
            conflict = self._get_frenet_conflict(ego_id, other_id)
    
            if conflict is not None:
                ego_dist_to_cp, other_dist_to_cp = conflict
    
                # s: signed lead/lag — positive means ego arrives first (safe)
                #    negative means neighbor arrives first (ego must yield)
                s_raw = ego_dist_to_cp - other_dist_to_cp
                s = np.clip(s_raw / self.perception_radius, -1.0, 1.0)
    
                # TTC: ego's remaining distance to conflict point / ego speed
                raw_ttc = ego_dist_to_cp / (ego_speed + 1e-6)
                ttc = np.clip(raw_ttc / 10.0, 0.0, 1.0)
            else:
                # Truly non-conflicting routes (e.g. same-direction following)
                # Fall back to longitudinal gap along ego's heading
                longitudinal_proj = (other_x - ego_x) * ego_cos + (other_y - ego_y) * ego_sin
                s_raw = np.sign(longitudinal_proj) * abs(longitudinal_proj)
                s = np.clip(s_raw / self.perception_radius, -1.0, 1.0)
                raw_ttc = abs(longitudinal_proj) / (ego_speed + 1e-6)
                ttc = np.clip(raw_ttc / 10.0, 0.0, 1.0)
    
            edge = self.k.vehicle.get_edge(other_id)
    
            neighbors_info.append({
                'v':        other_speed_norm,
                's':        s,
                'ttc':      ttc,
                'sinx':     other_sin,
                'cosx':     other_cos,
                'edge':     edge,
                'distance': distance,
            })
    
        # Sort by physical distance, take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
    
        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor['v'],
                neighbor['s'],
                neighbor['ttc'],
                neighbor['sinx'],
                neighbor['cosx'],
            ])
    
        # Pad missing neighbors: [v=0, s=1(safe), ttc=1(safe), sin=0, cos=0]
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            for _ in range(self.max_neighbours - num_actual):
                obs_vector.extend([0.0, 1.0, 1.0, 0.0, 0.0])
    
        obs_array = np.array(obs_vector, dtype=np.float32)
        assert np.all(np.isfinite(obs_array)), f"Non-finite obs: {obs_array}"
    
        return obs_array, neighbors_info

    def _get_route_waypoints(self, route):
        """
        Build a flat list of (x, y, cumulative_s) waypoints for a route.
        Uses SUMO's laneShape for the first lane of each edge.
        """
        waypoints = []
        s = 0.0
        for edge_id in route:
            try:
                # get shape of lane 0 of this edge
                shape = self.k.kernel_api.lane.getShape(edge_id + "_0")
            except Exception:
                print('get_route_waypoint: Failed to get waypoint')
                continue
            for i, (x, y) in enumerate(shape):
                if i == 0 and len(waypoints) > 0:
                    # skip duplicate junction point
                    pass
                else:
                    waypoints.append((x, y, s))
                if i < len(shape) - 1:
                    nx, ny = shape[i + 1]
                    s += np.sqrt((nx - x)**2 + (ny - y)**2)
        return waypoints  # list of (x, y, cumulative_s_from_route_start)


    def _segments_intersect(self, p1, p2, p3, p4):
        """
        Find intersection of segment p1->p2 and segment p3->p4.
        Returns (t, u) parameters (both in [0,1]) or None.
        t: fraction along p1->p2, u: fraction along p3->p4.
        """
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None  # parallel
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return t, u
        return None
    
    def _get_frenet_conflict(self, ego_id, other_id):
        """
        Find the conflict point between ego and neighbor using route polylines.
        Returns (ego_s_to_cp, other_s_to_cp) where s is arc-length remaining
        to the conflict point, or None if routes don't intersect.
        """
        ego_route   = self.k.vehicle.get_route(ego_id)
        other_route = self.k.vehicle.get_route(other_id)
    
        ego_wp   = self._get_route_waypoints(ego_route)
        other_wp = self._get_route_waypoints(other_route)
    
        if len(ego_wp) < 2 or len(other_wp) < 2:
            return None
    
        # Current odometer readings
        ego_s_traveled   = self.k.vehicle.get_distance(ego_id)
        other_s_traveled = self.k.vehicle.get_distance(other_id)
        if ego_s_traveled == -1001: ego_s_traveled = 0.0
        if other_s_traveled == -1001: other_s_traveled = 0.0
    
        # Search all segment pairs for first intersection
        best = None  # (ego_s_at_cp, other_s_at_cp)
    
        for i in range(len(ego_wp) - 1):
            ex1, ey1, es1 = ego_wp[i]
            ex2, ey2, es2 = ego_wp[i + 1]
            seg_len_e = es2 - es1
    
            for j in range(len(other_wp) - 1):
                ox1, oy1, os1 = other_wp[j]
                ox2, oy2, os2 = other_wp[j + 1]
                seg_len_o = os2 - os1
    
                result = self._segments_intersect(
                    (ex1, ey1), (ex2, ey2),
                    (ox1, oy1), (ox2, oy2)
                )
                if result is not None:
                    t, u = result
                    ego_s_at_cp   = es1 + t * seg_len_e  # s from route start
                    other_s_at_cp = os1 + u * seg_len_o
    
                    # Only care about conflict points ahead of both vehicles
                    if ego_s_at_cp >= ego_s_traveled and other_s_at_cp >= other_s_traveled:
                        ego_remaining   = ego_s_at_cp   - ego_s_traveled
                        other_remaining = other_s_at_cp - other_s_traveled
                        # keep the earliest conflict point for ego
                        if best is None or ego_remaining < best[0]:
                            best = (ego_remaining, other_remaining)
    
        return best  # (ego_dist_to_cp, other_dist_to_cp) or None

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
            return -1  
        if goal_reached:
            return 1   
            
        if agent_id not in self.k.vehicle.get_rl_ids():
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
