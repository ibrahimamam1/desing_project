from copy import deepcopy
from ray.rllib.algorithms.ppo import PPOConfig
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

from flow.controllers import RLController
from flow.controllers import IDMController

import random

################ Acceleration Controllers #######################
IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

min_gap = 0.9
max_accel = 2.6
max_decel = 4.5
max_speed = 30
initial_speed = 0
speed_factor = 1.0
speed_dev = 0.0
impatience = 0.0
car_follow_model = "IDM"
sigma = 0
tau = 0.8
period = 0.5

######### Car Following Params Configuration  #######

max_vehicle_count_in_inflow = 20
num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)
num_rl_vehicles = 1
num_non_rl_vehicles = 0

vehicles = VehicleParams()

speed_mode = 0
RL_car_following_params = SumoCarFollowingParams(
    speed_mode=speed_mode,
    accel=max_accel,
    decel=max_decel,
    sigma=sigma,
    tau=tau,
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

NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=speed_mode,
    accel=max_accel,
    decel=max_decel,
    sigma=sigma,
    tau=tau,
    min_gap=min_gap,
    max_speed=max_speed,
    speed_factor=speed_factor,
    speed_dev=speed_dev,
    impatience=impatience,
    car_follow_model=car_follow_model,
)

vehicles.add(
    veh_id="NonRL",
    acceleration_controller=(IDM_acceleration_controller, {}),
    initial_speed=initial_speed,
    num_vehicles=num_non_rl_vehicles,
    car_following_params=NonRL_car_following_params,
    lane_change_params=None,
    color="red"
)

############################# InFlow Configuration #########################
inflow = InFlows()

#### TRAFFIC RATES
high = 500
medium = 300
low = 150

traffic_rate = {"N": medium, "S": medium, "W": medium, "E": medium}

inflow.add(veh_type="NonRL",
           edge="E#T-X",
           probability=traffic_rate["N"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="NonRL",
           edge="E#R-X",
           probability=traffic_rate["E"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="NonRL",
           edge="E#D-X",
           probability=traffic_rate["S"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="RL",
           edge="E#L-X",
           probability=traffic_rate["W"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           number=num_inflows_vehicles,
           color="green"
           )

################ NETWORK Description #######################

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir = os.path.join(root_dir, "networks")

net_file_name = "100m_unregulated.net.xml"
net_file = os.path.join(net_file_dir, net_file_name)

############ NetParams Configuration #############
net_params = NetParams(
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
    spacing="uniform",
    min_gap=12,
    perturbation=5.0,
    x0=5,
    bunching=0,
    lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION,
    additional_params=None
)

myTag = "AlphaV0.1"

############################## Environment Configuration ###############################
horizon = 70
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

############################## Sumo Params Configuration ###############################
teleport_time = 0
sim_params = SumoParams(
    port=None,
    sim_step=sim_step,
    emission_path=output_file_dir,
    lateral_resolution=None,
    no_step_log=True,
    render=False,
    save_render=False,
    sight_radius=25,
    show_radius=False,
    pxpm=2,
    force_color_update=False,
    overtake_right=False,
    seed=42,
    restart_instance=True,
    print_warnings=True,
    teleport_time=teleport_time,
    num_clients=1,
    color_by_speed=False,
    use_ballistic=False
)

flow_params = dict(
    exp_tag=myTag,
    network=myNet,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

###############################  Running RL experiments in Ray #####################################
from ray.tune.registry import register_env
import ray 
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np
import os
import shutil 
from datetime import datetime 

TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs")
RUN_NAME = f"flow_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)

def create_flow_env(env_config):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from envs.alpha_env import AlphaEnv 
    
    params = flow_params
    vehicles = deepcopy(params['veh'])
    net_params = params['net']
    sim_params = deepcopy(params['sim'])
    network_class = params["network"]
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    network = network_class(
        name='AlphaEnv-Check',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    env = AlphaEnv(
        env_params=params['env'],
        sim_params=sim_params,
        network=network,
        simulator=params['simulator']
    )
    return env

register_env("flow_intersection", create_flow_env)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

# CRITICAL FIXES:
config = (
    PPOConfig()
    .environment(env="flow_intersection")
    .framework("torch")
    
    .rollouts(
        num_rollout_workers=6, 
        rollout_fragment_length='auto',
        num_envs_per_worker = 1,
    )
    
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=policy_mapping_fn,
    )
    
    # FIX 1: Lower learning rate significantly
    .training(
        train_batch_size=4000,  # Increase from 2000
        sgd_minibatch_size=256,  # Increase from 128
        num_sgd_iter=10,
        lr=3e-4,  # INCREASE from 5e-5 (too conservative)
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.3,  # Increase from 0.2 for faster learning
        vf_clip_param=10.0,
        grad_clip=0.5,
        kl_coeff=0.2,
        kl_target=0.01,
        entropy_coeff=0.05,  # INCREASE from 0.01 for more exploration
    )    
    # FIX 3: Add evaluation for monitoring
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=5,
        evaluation_num_workers=1,
    )
    
    .debugging(log_level="WARN")  # Reduce log noise
    .reporting(
        metrics_num_episodes_for_smoothing=10,
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=2000,
    )
    
    # FIX 4: Add resource allocation
    .resources(
        num_gpus=0,  # Set to 1 if you have GPU
    )
)

print("--- BUILDING ALGORITHM ---")
algo = config.build(logger_creator=lambda config: \
    ray.tune.logger.UnifiedLogger(config, TENSORBOARD_RUN_DIR, loggers=None))

CHECKPOINT_ROOT = os.path.join(os.getcwd(), "checkpoints")
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True)

print("--- STARTING TRAINING WITH STABILITY FIXES ---")
print("Key changes:")
print("  - Learning rate: 0.0003 -> 5e-5 (60x lower)")
print("  - Added gradient clipping at 0.5")
print("  - Added KL penalty and value function clipping")
print("  - Increased batch size for stability")
print("")

for i in range(100):
    result = algo.train()
    # Save checkpoint every 10 iterations
    if i % 10 == 0 or i == 199:
        save_dir = algo.save(checkpoint_dir=CHECKPOINT_ROOT)
        print(f"    --> Checkpoint saved to: {save_dir}")

print("\n--- TRAINING COMPLETE ---")
ray.shutdown()
