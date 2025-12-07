#============NEW ACCEL ENV ENVIRONMENT BY IBRAHIMA============

"""Environment for training Alpha model."""

from flow.core import rewards
from flow.envs.base_new import Env_N

import gymnasium as gym
import numpy as np
import random

ADDITIONAL_ENV_PARAMS = {
    'max_accel': 10.0,
    'max_decel': 4.5,
    'sort_vehicles': False
}

class AlphaEnv(Env_N):
    """Alpha environment.
    States
        The state consists of the velocities, absolute positions, heading angle, and moving intention of all
        vehicles in the perception radius of the ego vehicle.

    Actions
        Actions is the acceleration bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is -0.1 each time step, -100 if collision occurs and +10 if the desired goal is reached

    Termination
        A rollout is terminated if the time horizon is reached, if two
        vehicles collide into one another or if vehicle reaches desired goal.

    Attributes
    ----------
    prev_pos : dict
        dictionary keeping track of each veh_id's previous position
    absolute_position : dict
        dictionary keeping track of each veh_id's absolute position
    obs_var_labels : list of str
        referenced in the visualizer. Tells the visualizer which
        metrics to track
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return gym.spaces.Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'x_pos', 'y_pos', 'Heading_angle', 'Moving_intention']
        return gym.spaces.Box(
            #low=0,
            low=float("-inf"),
            high=float("inf"),
            #high=1,
            shape=(5 * max(1, self.max_neighbours), ), #the observation contains state of 5 neighbours and each state has 4 values
            dtype=np.float32)
    
    def setup_initial_state(self):
        #RANDOMIZE STARTING EDGE
        if isinstance(self.initial_config.edges_distribution, list):
            random.shuffle(self.initial_config.edges_distribution)
        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        print(f"@@@ DEBUG setup_initial_state: start_pos = {start_pos}")
        print(f"@@@ DEBUG setup_initial_state: start_lanes = {start_lanes}")
        
        #===================ADDED BY IBRAHIMA STARTS=================#
        
        # Track occupied edges
        occupied_edges = set()

        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            lane = start_lanes[i]
            speed = self.k.vehicle.get_initial_speed(veh_id)
        
            # Randomly select an edge that hasn't been used yet
            available_edges = [e for e in self.initial_config.edges_distribution if e not in occupied_edges]
        
            if available_edges:
                edge = random.choice(available_edges)
            else:
                # If all edges are occupied, just pick any edge
                edge = random.choice(self.initial_config.edges_distribution)
        
            # Mark this edge as occupied
            occupied_edges.add(edge)
        
            print(f"@@@ DEBUG setup_initial_state: veh_id={veh_id}, edge={edge}, lane={lane} (type: {type(lane)}), pos={pos}")
            self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)  
        
    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to avs. Actions that are not assigned are left to the
        control of the simulator.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        # ANSI color codes
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        self.step_counter_within_rl_step=0
        for inner_step in range(self.env_params.sims_per_step):
            self.time_counter += self.sim_step
            self.step_counter += 1
            self.step_counter_within_rl_step = inner_step
            # input("PRESS")
            print(f"\n{CYAN}@ Simulation Time:{self.time_counter}s #Inner-Step Count within RL-iteration:{inner_step}{RESET}")

            ##### perform lane change actions for Controlled Human-Driven Vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)
            #### Before Applying Acceleration #########
            ## Added by Ashraf: Starts
            # for veh_id in self.k.vehicle.get_rl_ids():
            #     speed=self.k.vehicle.get_speed(veh_id)
            #     print(f"\t#veh_id:{veh_id} speed:{speed} Before applying acceleration at {self.step_counter}")
            ## Added by Ashraf: Ends

            ###@@@###
            self.apply_rl_actions(rl_actions)
            ###@@@###
            # if self.step_counter>20:
            #     self.check_about_to_enter_junction_functionality()

            self.additional_command()

            # ## Added by Ashraf: Starts
            # ###@@@###
            # # @ Need to STORE new observations related to Vehicle and Junction Object
            # previous_grid_observation_data, previous_lane_observation_data = self.k.junction.save_observation_data()
            # self.k.junction.initialize_all_observation_data()
            # self.update_coordiantion_data_functionality()
            # ###@@@###
            # ## Added by Ashraf: Ends

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            ###@@@###
            # #@ Need to STORE new observations related to Vehicle and Junction Object
            # self.k.juction.initialize_all_observation_data()
            # time.sleep(0.05)  # sleep 50 ms
            self.k.update(reset=False)
            ###@@@###

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # ## Added by Ashraf: Starts
            # ###@@@###
            # #@ Need to STORE new observations related to Vehicle and Junction Object
            # previous_grid_observation_data,previous_lane_observation_data=self.k.junction.save_observation_data()
            # self.k.junction.initialize_all_observation_data()
            # self.update_coordiantion_data_functionality()
            # ###@@@###
            # ## Added by Ashraf: Ends

            # crash encodes whether the simulator experienced a collision
            ## Added by Ashraf: Starts
            # crash = self.k.simulation.check_collision() # Commented by Ashraf
            crash = False
            if self.k.kernel_api.simulation.getCollidingVehiclesNumber():
                crash = True
            ## Added by Ashraf: Ends
            # stop collecting new simulation steps if there is a collision
            # if crash:
            #     colliding_vehicles_number = self.k.kernel_api.simulation.getCollidingVehiclesNumber()
            #     colliding_vehicles_id_list = self.k.kernel_api.simulation.getCollidingVehiclesIDList()
            #     starting_teleport_id_list = self.k.kernel_api.simulation.getStartingTeleportIDList()
            #     print(f"\n !!!! Colliding Vehicles Count:{colliding_vehicles_number} IDs:{colliding_vehicles_id_list}:\n\t Starting Teleport IDs:{starting_teleport_id_list}")
            if crash:
                colliding_vehicles_number = self.k.kernel_api.simulation.getCollidingVehiclesNumber()
                colliding_vehicles_id_list = self.k.kernel_api.simulation.getCollidingVehiclesIDList()

                # Get unique vehicle IDs
                unique_colliding_vehicles = set(colliding_vehicles_id_list)
                # starting_teleport_id_list = self.k.kernel_api.simulation.getStartingTeleportIDList()
                # Dictionary to store speed modes for each unique vehicle
                colliding_vehicle_speed_modes = {}

                for colliding_veh_id in unique_colliding_vehicles:
                    try:
                        colliding_veh_speed_mode = self.k.kernel_api.vehicle.getSpeedMode(colliding_veh_id)
                        colliding_vehicle_speed_modes[colliding_veh_id] = colliding_veh_speed_mode
                    except Exception as e:
                        print(f"Could not get speed mode for vehicle {veh_id}: {e}")
                        colliding_vehicle_speed_modes[colliding_veh_id] = None  # or some default value

                print(
                    f"\n {RED}!!!! Colliding Vehicles Count:{len(unique_colliding_vehicles)} Unique IDs:{unique_colliding_vehicles}: Speed Modes:{colliding_vehicle_speed_modes}\n{RESET}") #\t Starting Teleport IDs:{starting_teleport_id_list}")

                # Now also print the position of each unique collided vehicle
                for veh_id in unique_colliding_vehicles:
                    try:
                        x, y = self.k.kernel_api.vehicle.getPosition(veh_id)
                        print(f"\t{RED} Vehicle {veh_id} position: ({x:.2f}, {y:.2f}){RESET}")
                    except Exception as e:
                        print(f"\t Could not get position for {veh_id}: {e}")

            # render a frame
            self.render()

            ## All Tranning Related Tasks at this particular Simulation Step is Done #######
            # for veh_id in self.k.vehicle.get_rl_ids():
            #     speed = self.k.vehicle.get_speed(veh_id)
            #     print(f"\t@veh_id:{veh_id} speed:{speed} AFTER applying acceleration at {self.step_counter}")

        ## A particulat RL Tranning Step Done: ##

        states = self.get_state()
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        goal_reached = False
        
        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.sims_per_step *
                (self.env_params.warmup_steps + self.env_params.horizon)
                or crash or goal_reached)

        # compute the info for each agent
        infos = {}

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)
        print(f"Step applied. action = {rl_actions}, observation = {next_observation}, reward = {reward}")
        #====ORIGINAL RETURN 4 TUPLE====
        #return next_observation, reward, done, infos
        #====MODIFIED By Ibrahima RETURN 5 TUPLE====
        truncated = False #TO MODIFY
        return next_observation, reward, done, truncated, infos
        


    def _apply_rl_actions(self, rl_actions):
        """Apply RL actions to the sorted RL vehicles."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        if rl_actions is None or len(rl_actions) != len(sorted_rl_ids):
            print(f"⚠️ RL actions mismatch: expected {len(sorted_rl_ids)} actions but got {len(rl_actions) if rl_actions is not None else 'None'}")
            return  # or apply zero acceleration if needed

        print('Applying RL actions to RL vehicles')
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions)


    def compute_reward(self, rl_actions, **kwargs):
        """See class definition"""
        reward = -0.1 #teme step penalty
        
        if self.k.kernel_api.simulation.getCollidingVehiclesNumber():
            #collision penalty
            reward -= 100

        return reward

    def get_state(self):
        """See class definition."""
        
        sorted_ids = self.sorted_ids
        if not sorted_ids:
            # Return zero vector of expected shape if no vehicles are present
            obs_len = self.observation_space.shape[0]
            return np.zeros(obs_len, dtype=np.float32)
        
        sorted_ids = sorted_ids[:self.max_neighbours]
        
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in sorted_ids]
        
        positions = [self.k.vehicle.get_2d_position(veh_id) for veh_id in sorted_ids]
        network_length = self.k.network.length()
        x = [pos[0] / self.k.network.length() for pos in positions]
        y = [pos[1] / self.k.network.length() for pos in positions]
        
        heading_angle = [self.k.vehicle.get_heading(veh_id) / 360
                         for veh_id in sorted_ids]
        moving_intention = [1
                            for veh_id in sorted_ids]

        state = np.array(speed + x + y + heading_angle + moving_intention, dtype=np.float32)

        # Pad with zeros if actual state is smaller than observation space
        obs_len = self.observation_space.shape[0]
        if state.shape[0] < obs_len:
            padding = np.zeros(obs_len - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding])

        return state

    def get_turn_signal(self, veh_id):
        """
        Get simplified turn signal.
        Returns: 0=straight, 1=left, 2=right
        """
        signals = self.k.vehicle.get_signals(veh_id)
    
        if signals & 2:  # Left turn signal (bit 1)
            return 1
        elif signals & 1:  # Right turn signal (bit 0)
            return 2
        else:
            return 0

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
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

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """ 
        
        ids = self.k.vehicle.get_ids()
            
        if self.env_params.additional_params['sort_vehicles']:
           return sorted(ids, key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)
    
    
    def reset(self, *, seed=None, options=None):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        super().reset()
        obs = self.get_state()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
        
        info = {} 
        return obs, info
