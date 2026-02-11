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

from flow.controllers import RLController  # for RL controlled Vehicles
from flow.controllers import IDMController  # for NON-RL controlled Vehicles

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
num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)  # 1
num_rl_vehicles = 8
num_non_rl_vehicles = 0

vehicles = VehicleParams()

speed_mode = 0  # 32 = safety check off, 31 = safety check on
RL_car_following_params = SumoCarFollowingParams(
    speed_mode=speed_mode,
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

NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=speed_mode,
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

#### TRAFFIC RATES
high = 500
medium = 300
low = 150

traffic_rate = {"N": medium, "S": medium, "W": medium, "E": medium}

inflow.add(veh_type="NonRL",
           edge="E#T-X",
           probability= traffic_rate["N"]/3600,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           # number=7,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#R-X",
           probability= traffic_rate["E"]/3600,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )


inflow.add(veh_type="NonRL",
           edge="E#D-X",
           probability= traffic_rate["S"]/3600,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )

inflow.add(veh_type="RL",
           edge="E#L-X",
           probability= traffic_rate["W"]/3600,
           depart_lane=0,  # right lane
           depart_speed=initial_speed,  # initial_speed, #"max","random"
           begin=1,  # rand[1,6] unit in minute
           number=num_inflows_vehicles,
           color="green"
           )


################ NETWORK Description #######################

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir = os.path.join(root_dir, "networks")


net_file_name = "100m_unregulated.net.xml"
net_file = os.path.join(net_file_dir, net_file_name)

############ NetParams Configuration  #############
net_params = NetParams(
    #inflows = inflow,
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
    perturbation=5.0,
    x0=5,
    bunching=0,
    lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION,
    additional_params=None
)

# myEnv=AccelEnv
myTag = "AlphaV0.1"

############################## Environemnt Configuration  ###############################
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

############################## Sumo Params Configuration  ###############################
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

# 2. Define the Environment Creator (Same as yours, just ensured imports are safe)
def create_flow_env(env_config):
    import sys
    # Add path to your envs folder if not already in pythonpath
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from envs.alpha_env import AlphaEnv 
    
    # Re-construct necessary params inside the worker
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

# 3. Define Policy Mapping (Parameter Sharing)
# This maps ALL vehicles (Agent IDs) to a single policy named "shared_policy"
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

# 4. Configure PPO for SANITY CHECK
config = (
    PPOConfig()
    .environment(env="flow_intersection")
    .framework("torch")
    
    # CRITICAL CHANGE: Run on main thread to see print() statements
    .rollouts(
        num_rollout_workers=4, 
        rollout_fragment_length='auto' # Short rollouts for quick updates
    )
    
    # CRITICAL CHANGE: Multi-Agent Setup
    .multi_agent(
        policies={"shared_policy"}, # Let RLlib infer observation/action space from env
        policy_mapping_fn=policy_mapping_fn,
    )
    
    # Simplify training params for quick feedback
    .training(
        train_batch_size=1000, 
        sgd_minibatch_size=64,
        num_sgd_iter=5,
        lr=0.0003,
    )
    .debugging(log_level="INFO")
)

print("--- BUILDING ALGORITHM ---")
algo = config.build()

CHECKPOINT_ROOT = os.path.join(os.getcwd(), "checkpoints")
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True) # Clean up old runs

print("--- STARTING SANITY CHECK LOOP ---")

for i in range(1000):
    result = algo.train()
    
    reward_mean = result.get('episode_reward_mean')
    len_mean = result.get('episode_len_mean')
    
    print(f"Iter: {i} | Reward: {reward_mean:.2f} | Len: {len_mean:.2f}")
    
    # Save checkpoint every 50 iterations (and always at the end)
    if i % 5 == 0 or i == 9:
        save_dir = algo.save(checkpoint_dir=CHECKPOINT_ROOT)
        print(f"--> Saved checkpoint to: {save_dir}")

print("--- SANITY CHECK COMPLETE ---")
ray.shutdown()
