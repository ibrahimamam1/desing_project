import os
import sys

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams,SumoCarFollowingParams

import glob, time, random
from flow.controllers import RLController # for RL controlled Vehicles
from flow.controllers import IDMController # for NON-RL controlled Vehicles

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


################ Acceleration Controllers #######################
IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

SPEED_MODES = {
    "disable_right_of_way": 55, #Newly Defined at /flow/core/params.py
    "all_checks_off": 32,     #Newly Defined at /flow/core/params.py
    "aggressive": 0,
    "obey_safe_speed": 1,
    "no_collide": 7,
    "right_of_way": 25,
    "all_checks": 31
}
min_gap=0.9 #Default 2.5 #min_gap_to_avoid_collision
max_accel=2.6 #Default 2.6
max_decel=4.5 #Default 4.5
max_speed=30 #Default 30m/s 108km/h
initial_speed = 0
speed_factor=1.0
speed_dev=0.0
impatience=0.0 #Default 0.5
car_follow_model="IDM" # Default "IDM"
sigma=0 #Default 0.5
tau=0.8 # past 1 at sim_step=0.1 you no longer see waves

##### Car Following Params Configuration  #####

#number of Vehicles at the begining of Simulation
max_vehicle_count_in_inflow = 4
num_inflows_vehicles= random.randint(1, max_vehicle_count_in_inflow) #1
num_rl_vehicles= 2
num_non_rl_vehicles= 0



vehicles = VehicleParams()

RL_car_following_params=SumoCarFollowingParams(
    accel=max_accel,
    decel=max_decel,
    sigma=sigma,
    tau=tau,  # past 1 at sim_step=0.1 you no longer see waves
    min_gap=min_gap,
    max_speed=max_speed,
    speed_factor=speed_factor,
    speed_dev=speed_dev,
    impatience=impatience,
    car_follow_model=car_follow_model,
    )

vehicles.add(
     veh_id="RL",
     acceleration_controller=(RL_vehicle_acceleration_controller, {}),
     #lane_change_controller=(SimLaneChangeController, {}),
     #routing_controller=(ContinuousRouter, {}),
     initial_speed=initial_speed,
     num_vehicles=num_rl_vehicles,
     car_following_params=RL_car_following_params,
     lane_change_params=None,
     color="blue"
     )

NonRL_car_following_params=SumoCarFollowingParams(
    accel=max_accel,
    decel=max_decel,
    sigma=sigma,
    tau=tau,  # past 1 at sim_step=0.1 you no longer see waves
    min_gap=min_gap,
    max_speed=max_speed,
    speed_factor=speed_factor,
    speed_dev=speed_dev,
    impatience=impatience,
    car_follow_model=car_follow_model,
    )

vehicles.add(
     veh_id="NonRL",
     acceleration_controller=(IDM_acceleration_controller, {}),#v0=30), #{}),
     #lane_change_controller=(SimLaneChangeController, {}),
     #routing_controller=(ContinuousRouter, {}),
     initial_speed=initial_speed,
     num_vehicles=num_non_rl_vehicles,
     car_following_params=NonRL_car_following_params,
     lane_change_params=None,
     color="red"
     )

############################# InFlow  Configuration  #########################
from flow.core.params import InFlows
inflow = InFlows()
############################# Probability Distribution Parameter ##########
####### Discrete Integer probability steps (e.g., multiples of 1)
# pick randomly from a set of values between 1 and 6
probability_discrete_int = random.choice([i for i in range(1, 7)])

####### Discrete probability steps (e.g., multiples of 0.01)
# pick randomly from a set of values between 0.01 and 0.10
probability_discrete = random.choice([i/100 for i in range(1, 11)])

####### Normal (Gaussian) variations
value = random.gauss(0.05, 0.02)   # mean=0.05, std=0.02
probability_gaussian = max(0.01, min(0.10, value))  # clamp to [0.01, 0.10]

###### Uniform Distributions
# probability between 0.01 and 0.10 (inclusive of bounds)
probability_uniform = random.uniform(0.01, 0.10)

inflow.add(veh_type="NonRL",
           edge="E#T-X",
           # period=1,
           # vehs_per_hour=200,
           probability= probability_discrete,
           # probability= 0.0001,#probability_discrete,
           depart_lane=0,  # right lane
           depart_speed= initial_speed, #initial_speed, #"max","random"
           begin=probability_discrete_int,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           # number=7,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#R-X",
           # period=1,
           # vehs_per_hour=200,
           probability= probability_discrete,
           depart_lane=0,  # right lane
           depart_speed= initial_speed, #initial_speed, #"max","random"
           begin=probability_discrete_int,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#D-X",
           # period=1,
           # vehs_per_hour=200,
           probability= probability_discrete,
           # probability= 0.0001,#probability_discrete,
           depart_lane=0,  # right lane
           depart_speed= initial_speed, #initial_speed, #"max","random"
           begin=probability_discrete_int,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="NonRL",
           edge="E#L-X",
           # period=1,
           # vehs_per_hour=200,
           probability= probability_discrete,
           # probability= probability_discrete,
           depart_lane=0,  # right lane
           depart_speed= initial_speed, #initial_speed, #"max","random"
           begin=probability_discrete_int,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           # number=2,
           color="green"
           )


################ NETWORK Description #######################
from flow.networks.all_turning_intersection import AllTurningIntersectionNetwork as myNet

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_file_dir =  os.path.join(root_dir, "results")
net_file_dir = os.path.join(root_dir, "networks")



net_file_name="50m_right_before_left.net.xml" 
net_file =os.path.join(net_file_dir, net_file_name)

############ NetParams Configuration  #############
net_params = NetParams(
     inflows = inflow,
     osm_path = None,
     template = net_file,
)
EDGES_DISTRIBUTION = [
    "E#D-X",
    "E#L-X",
    "E#R-X",
    "E#T-X",
]

# Shuffle edge distibuton to randomize which edge is first
random.shuffle(EDGES_DISTRIBUTION)

initial_config = InitialConfig(
                 shuffle=False,
                 spacing="random", #"random",#"uniform",
                 min_gap=12, #min_gap, #minimum gap between two vehicles upon initialization, in meters.Default is 0 m.
                 perturbation=0.0,
                 x0=10,
                 bunching=100,
                 lanes_distribution=float("inf"),
                 edges_distribution=EDGES_DISTRIBUTION,
                 additional_params=None
                 )

from src.alpha_env import AlphaEnv as myEnv
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
#myEnv=AccelEnv
myTag="1_RL_PPO_AccelEnv_500mLane_12mJunction_RightBeforeLeft_Junction"

############################## Environemnt Configuration  ###############################
#number of time steps
horizon=260

#simimualtion step length
sim_step=0.5

number_of_sim_steps_per_RlAction_step=1
env_params = EnvParams(
             # additional_params=ADDITIONAL_ENV_PARAMS,
             additional_params={
                 'max_accel': max_accel,
                 'max_decel': max_decel,
                 'target_velocity': max_speed,
                 'sort_vehicles': False
             },
             horizon=horizon,
             warmup_steps=0,
             sims_per_step=number_of_sim_steps_per_RlAction_step,
             evaluate=False,
             clip_actions=True)

############################## Sumo Params Configuration  ###############################
teleport_time = 0
sim_params = SumoParams(
                 port=None, #Port for Traci to connect to; finds an empty port by default
                 ###############
                 sim_step=sim_step,
                 ##############
                 emission_path=output_file_dir,
                 #############
                 lateral_resolution=None,
                 no_step_log=True,#False,
                 ###########
                 # render=False + save_render=True + restart_instance=True : Simulation Tested & Render Tested okey
                 # render=True/"drgb" + save_render=False/True(*warning) + restart_instance=True : Simulation Tested & Render Tested okey
                 render= True, #True, #"drgb" is tested and it works fine with render true #specifies whether to save rendering data to disk
                 #######SET -'True' while Render False.... :'False' while Render True ############
                 save_render=False,
                 ###################
                 sight_radius=25, #sets the radius of observation for RL vehicles (meter)
                 show_radius=False, #specifies whether to render the radius of RL observation
                 pxpm=2, #specifies rendering resolution (pixel / meter)
                 force_color_update=False, #whether or not to automatically color vehicles according to their typ
                 overtake_right=False, #whether vehicles are allowed to overtake on the right as well as the left
                 seed=None, #seed for sumo instance

                 ##################Important: restart_instance=True (for all case) (specially for RL and Render False)######
                 restart_instance=True,
                 #specifies whether to restart a sumo instance upon reset. Restarting
                 #the instance helps avoid slowdowns cause by excessive inflows over
                 #large experiment runtimes, but also require the gui to be started
                 #after every reset if "render" is set to True.

                 print_warnings=True,
                 teleport_time=teleport_time,
                 #If negative, vehicles don't teleport in gridlock. If positive,
                 #they teleport after teleport_time seconds

                 num_clients=1, #Number of clients that will connect to Traci
                 color_by_speed=False, #whether to color the vehicles by the speed they are moving at the current time step
                 use_ballistic=False #If true, use a ballistic integration step instead of an euler step
                 )############################## FLOW Configuration for simulation/Training ###############################

flow_params = dict(
    exp_tag=myTag,
    env_name=myEnv,  # using my new environment for the simulation
    network=myNet,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

###############################  Running RL experiments in Ray #####################################

####  Import  ####################
import json
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

################################ Initializing Ray ####################
ray.init(local_mode=True)  # FOR DEBUGGING

N_CPUS = 2
N_ROLLOUTS = 1

# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)

config = (PPOConfig()
          .environment(env=gym_name)
          .training(
            lr=0.001,
            clip_param=0.2,
          )
          .resources(num_gpus=0) 
          .rollouts(num_rollout_workers=0, num_envs_per_worker=1) 
          ) 

algo = config.build()
algo.train()
