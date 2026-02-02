"""Base environment class. This is the parent of all other environments."""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import os
import sys
import atexit
import time
import traceback
import numpy as np
import random
import shutil
import subprocess
import csv
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
from flow.utils.flow_warnings import deprecated_attribute

import gymnasium as gym
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete, Dict
from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException

import sumolib
from flow.core.util import ensure_dir
from flow.core.kernel import Kernel
from flow.utils.exceptions import FatalFlowError

# ANSI color codes for debugging
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

class Env_N(MultiAgentEnv, metaclass=ABCMeta):
    """
    Multi-Agent Base environment class.
    Adapted from Flow to work with Ray RLlib MultiAgentEnv.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 env_params,
                 sim_params,
                 network=None,
                 simulator='traci',
                 scenario=None,
                 render_mode=None,
                 output_file="simulation_telemetry.csv"  # <--- CSV File Path
                 ):
        
        # Initialize Ray MultiAgentEnv
        super().__init__()
    
        self.env_params = env_params
        if scenario is not None:
            deprecated_attribute(self, "scenario", "network")
        self.network = scenario if scenario is not None else network
        self.net_params = self.network.net_params
        self.initial_config = self.network.initial_config
        self.sim_params = deepcopy(sim_params)
        
        # Rendering setup
        self.should_render = self.sim_params.render
        self.sim_params.render = False 
        
        # Unique port generation to prevent collisions during parallel training
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()
        
        self.time_counter = 0
        self.step_counter = 0
        self.step_counter_within_rl_step = 0
        self.initial_state = {}
        self.state = None

        self.sim_step = sim_params.sim_step
        self.simulator = simulator

        # --- TELEMETRY SETUP ---
        self.telemetry_file = output_file
        
        # Check/Create CSV header
        if not os.path.exists(self.telemetry_file):
            with open(self.telemetry_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    "episode_duration", 
                    "number_of_vehicles_left_succesfully",
                    "number_of_collisions",
                    "min_travel_time", 
                    "max_travel_time", 
                    "avg_travel_time", 
                    "total_travel_time",             
                    "min_time_in_control_zone", 
                    "max_time_in_control_zone", 
                    "avg_time_in_control_zone", 
                    "total_time_in_control_zone"
                ]
                writer.writerow(header)

        # Telemetry Accumulators
        self.telemetry = None 

        # --- FLOW KERNEL INITIALIZATION ---
        self.k = Kernel(simulator=self.simulator, sim_params=self.sim_params)
        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        
        self.k.pass_api(kernel_api)
        self.available_routes = self.k.network.rts
        self.initial_ids = deepcopy(self.network.vehicles.ids)

        # Snapshot for restarts
        self.k.vehicle.kernel_api = None
        self.k.vehicle.master_kernel = None
        self.initial_vehicles = deepcopy(self.k.vehicle)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # Snapshot for junctions
        self.k.junction.kernel_api = None
        self.k.junction.master_kernel = None
        self.initial_junction = deepcopy(self.k.junction)
        self.k.junction.kernel_api = self.k.kernel_api
        self.k.junction.master_kernel = self.k

        self.setup_initial_state()

        # Renderer Setup
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sim_params.save_render
            sight_radius = self.sim_params.sight_radius
            pxpm = self.sim_params.pxpm
            show_radius = self.sim_params.show_radius
            network = []
            for lane_id in self.k.kernel_api.lane.getIDList():
                _lane_poly = self.k.kernel_api.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)
            self.renderer = Renderer(
                network,
                self.sim_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)
            self.render(reset=True)
            self.path = os.path.expanduser('~')+'/flow_rendering/' + self.network.name
            os.makedirs(self.path, exist_ok=True)
        elif self.sim_params.render in [True, False]:
            self.path = os.path.expanduser('~')+'/flow_rendering/' + self.network.name
            os.makedirs(self.path, exist_ok=True)
        else:
             raise FatalFlowError('Mode %s is not supported!' % self.sim_params.render)
        
        atexit.register(self.terminate)

    def restart_simulation(self, sim_params, render=None):
        """Restart simulation logic (Kept identical to original)."""
        self.k.close()
        if self.simulator == 'traci':
            self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sim_params.render = render
        if sim_params.emission_path is not None:
            ensure_dir(sim_params.emission_path)
            self.sim_params.emission_path = sim_params.emission_path

        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        self.k.pass_api(kernel_api)
        self.setup_initial_state()

    # --- TELEMETRY HELPERS ---

    def _init_telemetry(self):
        """Resets telemetry storage for a new episode."""
        self.telemetry = {
            "entry_times": {},        # {veh_id: float (time_step)}
            "travel_times": [],       # [float] (duration for finished vehicles)
            "zone_durations": {},     # {veh_id: float} (accumulated time in zone)
            "collisions": 0           # int
        }

    def _is_in_control_zone(self, veh_id):
        """
        Determines if a vehicle is in the control zone.
        """
        position = self.k.vehicle.get_2d_position(veh_id)
        in_box_x = -12 <= position[0] <= 12
        in_box_y = -12 <= position[1] <= 12
            
        return in_box_x and in_box_y
             
    
    def _update_telemetry_step(self):
            """
            Updates internal accumulators. 
            Uses get_ids() (all active vehicles) to robustly detect entries.
            """
            current_time = self.time_counter
            
            # Get all vehicles currently in the network
            current_ids = self.k.vehicle.get_ids()

            # 1. Track Entries
            # If we see a vehicle ID we haven't seen before, mark it as entered now.
            for veh_id in current_ids:
                if veh_id not in self.telemetry["entry_times"]:
                    self.telemetry["entry_times"][veh_id] = current_time
                    self.telemetry["zone_durations"][veh_id] = 0.0

                # 2. Track Control Zone Time 
                if self._is_in_control_zone(veh_id):
                    self.telemetry["zone_durations"][veh_id] += self.sim_step

            # 3. Track Successful Exits
            # get_arrived_ids returns vehicles that reached their destination (not crashed)
            newly_arrived = self.k.vehicle.get_arrived_ids()
            for veh_id in newly_arrived:
                if veh_id in self.telemetry["entry_times"]:
                    duration = current_time - self.telemetry["entry_times"][veh_id]
                    self.telemetry["travel_times"].append(duration)
                    # Cleanup
                    del self.telemetry["entry_times"][veh_id]

            # 4. Track Collisions
            n_colliding = self.k.kernel_api.simulation.getCollidingVehiclesNumber()
            if n_colliding > 0:
                self.telemetry["collisions"] += n_colliding

    def _finalize_and_write_telemetry(self):
        """
        Calculates aggregates and writes ONE row to the CSV.
        Called only when terminated is True.
        """
        travel_times = self.telemetry["travel_times"]
        zone_durations = list(self.telemetry["zone_durations"].values())
        
        # --- Travel Time Stats ---
        num_completed = len(travel_times)
        if num_completed > 0:
            min_tt = min(travel_times)
            max_tt = max(travel_times)
            total_tt = sum(travel_times) 
            avg_tt = total_tt / num_completed
        else:
            min_tt = max_tt = total_tt = avg_tt = 0

        # --- Control Zone Stats ---
        valid_zone_times = [t for t in zone_durations if t > 0]
        if valid_zone_times:
            min_zone = min(valid_zone_times)
            max_zone = max(valid_zone_times)
            total_zone = sum(valid_zone_times)
            avg_zone = total_zone / len(valid_zone_times)
        else:
            min_zone = max_zone = total_zone = avg_zone = 0

        row = [
            self.time_counter,      # Episode Duration
            num_completed,          # Emitted/Successful Exits (Constraint 1)
            self.telemetry["collisions"],
            min_tt, 
            max_tt, 
            avg_tt, 
            total_tt,
            min_zone, 
            max_zone, 
            avg_zone, 
            total_zone
        ]

        # --- Write Single Row
        try:
            with open(self.telemetry_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"{RED}Error writing telemetry: {e}{RESET}")


    def step(self, action_dict):
        """
        Advance the environment by one step.
        """
        self.step_counter_within_rl_step = 0
        
        sorted_ids = set(self.sorted_ids)

        self.apply_rl_actions(None) 
        self.additional_command()
        
        # 2. Simulation Step (Inner Loop)
        for inner_step in range(self.env_params.sims_per_step):
            self.time_counter += self.sim_step
            self.step_counter += 1
            self.step_counter_within_rl_step = inner_step
            
            self._apply_non_rl_controls()

            # Advance Simulator
            self.k.simulation.simulation_step()
            self.k.update(reset=False)
            
            # --- UPDATE TELEMETRY (Per Sim Step) ---
            self._update_telemetry_step()
            # ---------------------------------------

            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()
        
        new_sorted_ids = set(self.sorted_ids)
        agents_that_left = sorted_ids - new_sorted_ids

        # 3. Retrieve Observations and Rewards
        states = self.get_state() 
        
        crash = self.k.kernel_api.simulation.getCollidingVehiclesNumber() > 0
        
        terminated = (self.time_counter >= (self.env_params.sims_per_step * 
                     (self.env_params.warmup_steps + self.env_params.horizon))
                     or crash)
        
        # --- SAVE TELEMETRY (Once at end) ---
        if terminated:
            self._finalize_and_write_telemetry()
        # ------------------------------------

        # Construct Multi-Agent Returns
        obs_dict = states if isinstance(states, dict) else {} 
        reward_dict = {}
        infos = {}
        
        for agent_id in agents_that_left:
            reward_dict[agent_id] = self.compute_reward(agent_id, action_dict.get(agent_id), 
                                                        fail=crash, goal_reached=True)
            infos[agent_id] = {}
            if hasattr(self, 'observation_space'):
                 obs_dict[agent_id] = np.zeros(self.observation_space.shape, dtype=np.float32)

        for agent_id in obs_dict.keys():
            if agent_id not in agents_that_left:
                agent_action = action_dict.get(agent_id) 
                reward_dict[agent_id] = self.compute_reward(agent_id, agent_action, fail=crash, goal_reached=False)
                infos[agent_id] = {}

        return obs_dict, reward_dict, terminated, {}, {}

    def _apply_non_rl_controls(self):
        """Helper to handle IDM/LaneChange controllers for non-RL vehicles."""
        if len(self.k.vehicle.get_controlled_ids()) > 0:
            accel = []
            for veh_id in self.k.vehicle.get_controlled_ids():
                action = self.k.vehicle.get_acc_controller(veh_id).get_action(self)
                accel.append(action)
            self.k.vehicle.apply_acceleration(
                self.k.vehicle.get_controlled_ids(), accel)

        if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
            direction = []
            for veh_id in self.k.vehicle.get_controlled_lc_ids():
                target_lane = self.k.vehicle.get_lane_changing_controller(veh_id).get_action(self)
                direction.append(target_lane)
            self.k.vehicle.apply_lane_change(
                self.k.vehicle.get_controlled_lc_ids(), direction=direction)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.
        """
        # --- RESET TELEMETRY ---
        self._init_telemetry()
        # -----------------------

        super().reset(seed=seed)
        
        self.time_counter = 0
        if self.should_render:
            self.sim_params.render = True
            self.restart_simulation(self.sim_params)

        if self.sim_params.restart_instance or (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            self.sim_params.seed = random.randint(0, 1e5)
            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            self.k.junction = deepcopy(self.initial_junction)
            self.k.junction.master_kernel = self.k
            self.restart_simulation(self.sim_params)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        if self.simulator == 'traci':
            try:
                for veh_id in self.k.kernel_api.vehicle.getIDList():
                    self.k.vehicle.remove(veh_id)
            except:
                pass

        self.k.vehicle.reset()

        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = self.initial_state[veh_id]
            try:
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)
            except (FatalTraCIError, TraCIException):
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)

        self.k.simulation.simulation_step()
        self.k.update(reset=True)
        
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        states = self.get_state()
        obs_dict = states if isinstance(states, dict) else {}
        
        return obs_dict, {}
    
    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.""" 
        ids = self.k.vehicle.get_ids()
        rl_ids = []
        for id in ids:
            if id.startswith("RL"):
                rl_ids.append(id)
        return rl_ids
    
    def apply_rl_actions(self, action_dict):
        if action_dict is None:
            return

        clipped_dict = {}
        for agent_id, action in action_dict.items():
            if isinstance(self.action_space, Box):
                clipped_dict[agent_id] = np.clip(
                    action, self.action_space.low, self.action_space.high)
            else:
                clipped_dict[agent_id] = action
                
        self._apply_rl_actions(clipped_dict)

    @abstractmethod
    def _apply_rl_actions(self, rl_actions):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def compute_reward(self, agent_id, rl_action, **kwargs):
        pass

    def setup_initial_state(self):
        if isinstance(self.initial_config.edges_distribution, list):
            random.shuffle(self.initial_config.edges_distribution)

        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        occupied_edges = set()

        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            speed = self.k.vehicle.get_initial_speed(veh_id)

            available_edges = [e for e in self.initial_config.edges_distribution if e not in occupied_edges]
            if available_edges:
                edge = random.choice(available_edges)
            else:
                edge = random.choice(self.initial_config.edges_distribution)
            occupied_edges.add(edge)

            self.initial_state[veh_id] = (type_id, edge, 0, pos, speed)

    def additional_command(self):
        pass
    
    def terminate(self):
        try:
            self.k.close()
            if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
                self.renderer.close()
        except:
            pass
    def render(self, reset=False, buffer_length=5):
        """Render a frame.

        Parameters
        ----------
        reset : bool
            set to True to reset the buffer
        buffer_length : int
            length of the buffer
        """
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    self.frame_buffer.append(self.frame.copy())
                    self.sights_buffer.append(self.sights.copy())
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)
        elif (self.sim_params.render is True) and self.sim_params.save_render:
            # sumo-gui render
            self.k.kernel_api.gui.screenshot("View #0", self.path+"/frame_%06d.png" % self.time_counter)

    def pyglet_render(self):
        """Render a frame using pyglet."""
        # get human and RL simulation status
        human_idlist = self.k.vehicle.get_human_ids()
        machine_idlist = self.k.vehicle.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.k.network.max_speed()
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.k.vehicle.get_orientation(id))
                machine_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.k.vehicle.get_orientation(id))
                human_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.k.vehicle.get_timestep(id),
                 self.k.vehicle.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.k.vehicle.get_orientation(id))
            machine_dynamics.append(
                self.k.vehicle.get_speed(id)/max_speed)

        # step the renderer
        self.frame = self.renderer.render(human_orientations,
                                          machine_orientations,
                                          human_dynamics,
                                          machine_dynamics,
                                          human_logs,
                                          machine_logs)

        # get local observation of RL vehicles
        self.sights = []
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if "track" in id:
                orientation = self.k.vehicle.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.k.vehicle.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)


