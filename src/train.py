from alpha_env import AlphaEnv
from copy import deepcopy
import numpy as np
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import ray
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from src.alpha_env import AlphaEnv as myEnv
from flow.networks.all_turning_intersection import AllTurningIntersectionNetwork as myNet
from flow.core.params import InFlows
import os
import sys

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams, SumoCarFollowingParams

from flow.controllers import RLController  # for RL controlled Vehicles
from flow.controllers import IDMController  # for NON-RL controlled Vehicles

import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

################ Acceleration Controllers #######################
IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

min_gap = 0.9
max_accel = 2.6
max_decel = 4.5
max_speed = 30
initial_speed = 5
speed_factor = 1.0
speed_dev = 0.0
impatience = 0.0
car_follow_model = "IDM"
sigma = 0
tau = 0.8
period = 0.5

######### Car Following Params Configuration  #######

max_vehicle_count_in_inflow = 20
num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)  # 1
num_rl_vehicles = 2
num_non_rl_vehicles = 6

vehicles = VehicleParams()

rl_speed_mode = 32  # 32 = safety check of, 31 = safety check on
RL_car_following_params = SumoCarFollowingParams(
    speed_mode=rl_speed_mode,
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
    initial_speed=0,
    num_vehicles=num_rl_vehicles,
    car_following_params=RL_car_following_params,
    lane_change_params=None,
    color="blue"
)

non_rl_speed_mode = 31  # 32 = safety check of, 31 = safety check on
NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=non_rl_speed_mode,
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
    acceleration_controller=(IDM_acceleration_controller, {}),  # v0=30), #{}),
    initial_speed=initial_speed,
    num_vehicles=num_non_rl_vehicles,
    car_following_params=NonRL_car_following_params,
    lane_change_params=None,
    color="red"
)

############################# InFlow  Configuration  #########################
inflow = InFlows()

random_begin_time = random.choice([i for i in range(1, 7)])
probability_discrete = random.choice([i/100 for i in range(1, 11)])

inflow.add(veh_type="NonRL",
           edge="E#T-X",
           period=period,
           # vehs_per_hour=200,
           # probability= probability_discrete,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           # number=7,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#R-X",
           period=period,
           # vehs_per_hour=200,
           # probability= probability_discrete,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#D-X",
           period=period,
           # vehs_per_hour=200,
           # probability= probability_discrete,
           # probability= 0.0001,#probability_discrete,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="RL",
           edge="E#L-X",
           period=period,
           # vehs_per_hour=200,
           # probability= probability_discrete,
           # probability= probability_discrete,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )


################ NETWORK Description #######################

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir = os.path.join(root_dir, "networks")


net_file_name = "50m_right_before_left.net.xml"
net_file = os.path.join(net_file_dir, net_file_name)

############ NetParams Configuration  #############
net_params = NetParams(
    # inflows = inflow,
    osm_path=None,
    template=net_file,
)
EDGES_DISTRIBUTION = [
    "E#D-X",
    "E#L-X",
    "E#R-X",
    "E#T-X",
]

initial_config = InitialConfig(
    shuffle=False,
    spacing="uniform",  # "random",#"uniform",
    # min_gap, #minimum gap between two vehicles upon initialization, in meters.Default is 0 m.
    min_gap=12,
    perturbation=0.0,
    x0=10,
    bunching=0,
    lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION,
    additional_params=None
)

# myEnv=AccelEnv
myTag = "Alpha Experiment"

############################## Environemnt Configuration  ###############################
horizon = 260
sim_step = 0.5
number_of_sim_steps_per_RlAction_step = 1

env_params = EnvParams(
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
    port=None,
    sim_step=sim_step,
    emission_path=output_file_dir,
    lateral_resolution=None,
    no_step_log=True,
    render=True,
    save_render=False,
    sight_radius=25,
    show_radius=False,
    pxpm=2,  # specifies rendering resolution (pixel / meter)
    force_color_update=False,
    overtake_right=False,
    seed=42,

    restart_instance=True,
    # specifies whether to restart a sumo instance upon reset. Restarting
    # the instance helps avoid slowdowns cause by excessive inflows over
    # large experiment runtimes, but also require the gui to be started
    # after every reset if "render" is set to True.

    print_warnings=True,
    teleport_time=teleport_time,

    num_clients=1,  # Number of clients that will connect to Traci
    # whether to color the vehicles by the speed they are moving at the current time step
    color_by_speed=False,
    use_ballistic=False  # If true, use a ballistic integration step instead of an euler step
)  # FLOW Configuration for simulation/Training ###############################

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


sys.path.append(os.path.dirname(__file__))

################################ Initializing Ray ####################
ray.init(local_mode=True, ignore_reinit_error=True)

# We cannot use flow.utils.registry.make_create_env because it wraps the env
# in a Single-Agent wrapper. We need our raw MultiAgent AlphaEnv.


def create_flow_env(env_config):
    params = env_config["flow_params"]

    vehicles = deepcopy(params['veh'])
    net_params = params['net']
    sim_params = deepcopy(params['sim'])

    network_class = params["network"]
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    # 3. Initialize the network with the variables defined above
    network = network_class(
        name='AlphaEnv-exp',
        vehicles=vehicles,      # Now this variable actually exists
        net_params=net_params,  # Now this variable actually exists
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    # 4. Initialize the Environment
    env = AlphaEnv(
        env_params=params['env'],
        sim_params=sim_params,
        network=network,
        simulator=params['simulator']
    )

    return env


# register env with ray multiagent
env_name = "alpha_multiagent_v0"
register_env(env_name, create_flow_env)

# DEFINE PARAMETER SHARING (PPO CONFIG)

# Define the shapes of observation and action spaces
# Obs: (1 ego + 5 neighbors) * 5 features = 30
obs_dim = 30
dummy_obs_space = gym.spaces.Box(
    low=float("-inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32)

# Act: 1 value (acceleration)
dummy_act_space = gym.spaces.Box(
    low=-4.5, high=10.0, shape=(1,), dtype=np.float32)

# Define the "Shared Policy"
policies = {
    "shared_policy": (
        None,             # Use default PPO Policy class
        dummy_obs_space,
        dummy_act_space,
        {}                # Extra config
    )
}

# We map ALL agents to the SAME policy (Parameter Sharing)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

# 3. CONFIGURE AND TRAIN PPO


ray.init(local_mode=True, ignore_reinit_error=True)

config = (PPOConfig()
          .environment(
              env=env_name,
              # Pass the flow_params dictionary so the factory can use it
              env_config={"flow_params": flow_params},
              # Disable Gym API checks (critical for Flow)
              disable_env_checking=True
)
    .framework("torch")
    .training(
    lr=0.001,
    clip_param=0.2,
    train_batch_size=4000,
    sgd_minibatch_size=128,
    num_sgd_iter=10
)
    .multi_agent(
              policies=policies,
              policy_mapping_fn=policy_mapping_fn,
              policies_to_train=["shared_policy"],
)
    # Set to 0 GPUs and 0 workers for easier debugging/local testing
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=0, num_envs_per_worker=1)
)

algo = config.build()

print("Starting Training...")
for i in range(10):
    result = algo.train()
    print(f"Iteration: {i}, Mean Reward: {result['episode_reward_mean']}")

algo.stop()
