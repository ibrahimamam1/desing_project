import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
import shutil
import random
import math 
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.uniform_random import UniformRandomNetwork

parser = argparse.ArgumentParser(description="Train or evaluate the AlphaEnv PPO agent.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true", help="Run training loop.")
group.add_argument("--eval",  metavar="CHECKPOINT_PATH",
                   help="Path to a checkpoint directory to load and evaluate.")
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.all_straight import AllStraghtNetwork as myNet
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

myTag = "AlphaV0.1"
min_gap       = 2.6
max_accel     = 2.6
max_decel     = 4.5
max_speed     = 30
initial_speed = 0
speed_factor  = 1.0
speed_dev     = 0.0
impatience    = 0.0
car_follow_model = "IDM"
sigma = 0
tau   = 0.8
horizon = 900
sim_step = 0.25
warmup_steps = 50
number_of_sim_steps_per_RlAction_step = 1
RENDER_MODE = False

############### VEHICLE Configuration ##########################
num_rl_vehicles      = 0
num_non_rl_vehicles  = 0

rl_speed_mode    = 0
non_rl_speed_mode = 0

vehicles = VehicleParams()

RL_car_following_params = SumoCarFollowingParams(
    speed_mode=rl_speed_mode,
    accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau,
    min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev,
    impatience=impatience, car_follow_model=car_follow_model,
)
NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=non_rl_speed_mode,
    accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau,
    min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev,
    impatience=impatience, car_follow_model=car_follow_model,
)

vehicles.add(
    veh_id="RL",
    acceleration_controller=(RL_vehicle_acceleration_controller, {}),
    initial_speed=0,
    num_vehicles=num_rl_vehicles,
    car_following_params=RL_car_following_params,
    lane_change_params=None,
    color="blue",
)
vehicles.add(
    veh_id="NonRL",
    acceleration_controller=(IDM_acceleration_controller, {}),
    initial_speed=initial_speed,
    num_vehicles=num_non_rl_vehicles,
    car_following_params=NonRL_car_following_params,
    lane_change_params=None,
    color="red",
)

############################# InFlow Configuration #########################
inflow = InFlows()

#### TRAFFIC RATES
high = 500
medium = 300
low = 150

traffic_rate = {"N": medium, "S": medium, "W": medium, "E": medium}

inflow.add(veh_type="RL",
           edge="E#T-X",
           probability=traffic_rate["N"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           color="red"
           )

inflow.add(veh_type="RL",
           edge="E#R-X",
           probability=traffic_rate["E"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           color="red"
           )

inflow.add(veh_type="RL",
           edge="E#D-X",
           probability=traffic_rate["S"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           color="red"
           )

inflow.add(veh_type="RL",
           edge="E#L-X",
           probability=traffic_rate["W"]/3600,
           depart_lane=0,
           depart_speed=initial_speed,
           begin=1,
           color="red"
           )

root_dir        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir    = os.path.join(root_dir, "networks")

net_file_name = "100m_right_before_left.net.xml"
net_file= os.path.join(net_file_dir, net_file_name)

net_params = NetParams(
    osm_path=None,
    template=net_file,
    inflows=inflow
)

EDGES_DISTRIBUTION = ["E#D-X", "E#L-X", "E#R-X", "E#T-X"]

initial_config = InitialConfig(
    shuffle=False,
    spacing="uniform",
    min_gap=12,
    perturbation=5.0,
    x0=5,
    bunching=0,
    lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION,
    additional_params=None,
)
env_params = EnvParams(
    additional_params={
        "max_accel": max_accel,
        "max_decel": max_decel,
        "target_velocity": max_speed,
        "sort_vehicles": False,
    },
    horizon=horizon,
    warmup_steps=warmup_steps,
    sims_per_step=number_of_sim_steps_per_RlAction_step,
    evaluate=False,
    clip_actions=True,
)

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
    print_warnings=False,
    teleport_time=0,
    num_clients=1,
    color_by_speed=False,
    use_ballistic=False,
)

flow_params = dict(
    exp_tag=myTag,
    network=myNet,
    simulator="traci",
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# ─────────────────────────────────────────────
# Ray / RLlib setup
# ─────────────────────────────────────────────
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec # <-- Added import
from typing import Any, Dict

class TrafficCallbacks(DefaultCallbacks):

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        **kwargs,
    ) -> None:
        envs = base_env.get_sub_environments()
        env = envs[0] 
        
        telemetry = env._compute_telemetry_stats()

        if not telemetry:
            return

        episode.custom_metrics["collisions"] = float(telemetry.get("number_of_collisions", 0))
        episode.custom_metrics["avg_speed"] = float(telemetry.get("avg_speed", 0.0))

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        keep = {
            "episode_reward_mean",
            "episode_len_mean",
            "custom_metrics",
            "evaluation",
            "info",
            "training_iteration",
            "timesteps_total",
        }
        keys_to_delete = [k for k in result if k not in keep]
        for k in keys_to_delete:
            result.pop(k)

        if "custom_metrics" in result and isinstance(result["custom_metrics"], dict):
            custom_keep = {"avg_speed_mean", "collisions_mean"}
            custom_keys_to_delete = [k for k in result["custom_metrics"] if k not in custom_keep]
            for k in custom_keys_to_delete:
                result["custom_metrics"].pop(k)

        if "evaluation" in result and isinstance(result["evaluation"], dict):
            eval_keep = {"episode_reward_mean", "episode_len_mean"}
            eval_keys_to_delete = [k for k in result["evaluation"] if k not in eval_keep]
            for k in eval_keys_to_delete:
                result["evaluation"].pop(k)

        if "info" in result and isinstance(result["info"], dict):
            info = result["info"]
            if "learner" in info and isinstance(info["learner"], dict):
                learner = info["learner"]
                # Adjusted for parameter sharing naming
                if "shared_policy" in learner and isinstance(learner["shared_policy"], dict):
                    info_keep = {"entropy", "mean_kl_loss", "policy_loss", "total_loss", "vf_loss", "vf_explained_var"}
                    info_keys_to_delete = [k for k in learner["shared_policy"] if k not in info_keep]
                    for k in info_keys_to_delete:
                        learner["shared_policy"].pop(k)
                 
                info_keys_to_delete = [k for k in learner if k != "shared_policy"]
                for k in info_keys_to_delete:
                    learner.pop(k) 
            
            info_keys_to_delete = [k for k in result["info"] if k != "learner"]
            for k in info_keys_to_delete:
                result["info"].pop(k)        

def create_flow_env(env_config):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.alpha_env_v02 import AlphaEnv_v02

    params       = flow_params
    _vehicles    = deepcopy(params["veh"])
    _net_params  = params["net"]
    _sim_params  = deepcopy(params["sim"])
    _sim_params.render = env_config.get("render", False)
    network_class = params["network"]
    _initial_config = params.get("initial", InitialConfig())
    traffic_lights  = params.get("tls", TrafficLightParams())

    network = network_class(
        name="AlphaEnv-Check",
        vehicles=_vehicles,
        net_params=_net_params,
        initial_config=_initial_config,
        traffic_lights=traffic_lights,
    )
    return AlphaEnv_v02(
        env_params=params["env"],
        sim_params=_sim_params,
        network=network,
        simulator=params["simulator"],
        )


register_env("alpha_env_v02", create_flow_env)

def build_config(num_workers: int = 7, render: bool = False) -> PPOConfig:
        cfg = (
            PPOConfig()
            .environment(env="alpha_env_v02", env_config={"render": render}, disable_env_checking=True)
            .framework("torch")
            .multi_agent(
                policies={"shared_policy": PolicySpec()}, 
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
                count_steps_by="env_steps",  
            )
            .rollouts(
                num_rollout_workers=num_workers,
                rollout_fragment_length="auto",
                num_envs_per_worker=1,
                batch_mode="complete_episodes",
            )            .training(
                train_batch_size=2048,
                sgd_minibatch_size=256,
                num_sgd_iter=10,
                
                lr=[[0, 3e-4], [2_000_000, 1e-5]], 
                entropy_coeff=[[0, 0.02], [2_000_000, 0.0]], 
                
                gamma=0.995, 
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=50.0, 
                grad_clip=0.5,
                kl_coeff=0.2,
                kl_target=0.01,
            )
            .evaluation(
                evaluation_interval=100,
                evaluation_duration=10,
                evaluation_num_workers=1,
            )
            .debugging(log_level="WARN")
            .reporting(
                metrics_num_episodes_for_smoothing=10,
                min_time_s_per_iteration=0,
                min_sample_timesteps_per_iteration=2000,
            )
            .resources(num_gpus=0)
            .callbacks(TrafficCallbacks)
        )
        return cfg

# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
ENV_NAME  = "alpha_env_v02"
ALGO_NAME = "PPO"

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(),
    "checkpoints/v0_2",
    f"{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
FINAL_MODEL_DIR = os.path.join(CHECKPOINT_ROOT, "final")
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "best")

TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs/v0_2")
RUN_NAME = f"flow_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)

def train():
    ray.init(ignore_reinit_error=True)

    algo = build_config(num_workers=7).build(
        logger_creator=lambda cfg: ray.tune.logger.UnifiedLogger(
            cfg, TENSORBOARD_RUN_DIR, loggers=None
        )
    )

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n--- TRAINING START ---")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}\n")
    
    num_iterations = 1000
    best_reward = -float('inf')

    for i in range(num_iterations):
        result = algo.train()

        current_reward = result.get("episode_reward_mean")

        if current_reward is not None and not math.isnan(current_reward):
            if current_reward > best_reward:
                best_reward = current_reward
                
                if os.path.exists(BEST_CHECKPOINT_DIR):
                    shutil.rmtree(BEST_CHECKPOINT_DIR)
                
                best_save_path = algo.save(checkpoint_dir=BEST_CHECKPOINT_DIR)
                print(f"  [⭐ NEW BEST] Iteration: {i:4d} | Reward: {best_reward:.3f} | Saved to: {best_save_path}")

    print("\n--- TRAINING COMPLETE ---")
   
    save_path = algo.save(checkpoint_dir=FINAL_MODEL_DIR)
    print(f"Final Model → {FINAL_MODEL_DIR}")
    print(f"Best Model  → {BEST_CHECKPOINT_DIR}")
    print(f"Tensorboard  → {TENSORBOARD_RUN_DIR}")
    ray.shutdown()

def evaluate(checkpoint_path: str, num_iterations: int = 20):
    # 1. Define Scenarios
    intentions = {
        "all_straight": AllStraghtNetwork,  
        "all_left": AllLeftNetwork,     
        "uniform_random": UniformRandomNetwork,
    }

    # Traffic rates definitions
    high_rate = 500
    medium_rate = 300
    low_rate = 150

    traffic_rates = {
        #"Sc1_All_low" : [{"N": low_rate, "S": low_rate, "W": low_rate, "E": low_rate}],
        #"Sc3_All_medium": [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}], 
        "Sc2_All_high_3H" : [{"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                      {"N": high_rate, "S": high_rate, "W": high_rate, "E": medium_rate},
                      {"N": high_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
                      {"N": high_rate, "S": medium_rate, "W": high_rate, "E": high_rate},
                      {"N": medium_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                      {"N": high_rate, "S": high_rate, "W": high_rate, "E": low_rate},
                      {"N": high_rate, "S": high_rate, "W": low_rate, "E": high_rate},
                      {"N": high_rate, "S": low_rate, "W": high_rate, "E": high_rate},
                      {"N": low_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                    ],
        #"Sc4_Mixed_2H": [ 
                 #   {"N": high_rate, "S": high_rate, "W": low_rate, "E": low_rate},
                #    {"N": low_rate, "S": low_rate, "W": high_rate, "E": high_rate},
               #     {"N": high_rate, "S": low_rate, "W": high_rate, "E": low_rate},
              #      {"N": low_rate, "S": high_rate, "W": low_rate, "E": high_rate},
             #       {"N": high_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
            #        {"N": high_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
           #         {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
          #          {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
               # ],
        #"Sc5_Mixed_1H": [ 
        #            {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
       #             {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
       #             {"N": medium_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
       #             {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
       #             {"N": high_rate, "S": low_rate, "W": low_rate, "E": low_rate},
       #             {"N": low_rate, "S": high_rate, "W": low_rate, "E": low_rate},
       #             {"N": low_rate, "S": low_rate, "W": high_rate, "E": low_rate},
          #          {"N": low_rate, "S": low_rate, "W": low_rate, "E": high_rate},
               # ],
        #"Sc6_Mixed_ML": [ 
        #            {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
       #             {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
       #             {"N": medium_rate, "S": low_rate, "W": low_rate, "E": medium_rate},
       #             {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
       #             {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
       #             {"N": medium_rate, "S": low_rate, "W": low_rate, "E": low_rate},
                #],
    }
     
    # Map directions to your specific Edge IDs (based on your Inflow config)
    direction_map = {
        "N": "E#T-X",
        "E": "E#R-X",
        "S": "E#D-X",
        "W": "E#L-X"
    }

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ray.init(ignore_reinit_error=True)
    
    print(f"\n--- EVALUATION START ---")
    print(f"Checkpoint: {checkpoint_path}")

    for int_key, int_class in intentions.items():
        for sc_name, rate_configs in traffic_rates.items():
            
            print(f"\n>>> Evaluating: Intention=[{int_key}] | Scenario=[{sc_name}]")

            # 3. Inner Loop: Iterate through specific configs in the scenario list
            for config_idx, current_rates in enumerate(rate_configs):
                
                # 1. Update Network Class
                flow_params["network"] = int_class
                
                # 2. Update Inflows dynamically based on the current rates
                new_inflow = InFlows()
                for direction, rate in current_rates.items():
                    edge_id = direction_map[direction]
                    new_inflow.add(
                        veh_type="RL",
                        edge=edge_id,
                        probability=rate / 3600,
                        depart_lane=0,
                        depart_speed=initial_speed,
                        begin=1,
                        color="red"
                    )
                flow_params["net"].inflows = new_inflow

                # --- B: REBUILD ALGORITHM ---
                # We must rebuild the algo/env so it picks up the new flow_params
                # num_workers=0 ensures we use the local worker (easier for evaluation)
                algo = build_config(num_workers=0, render=True).build()
                algo.restore(checkpoint_path)
                
                env = algo.workers.local_worker().env
                
                # --- C: RUN EPISODES ---
                episode_rewards = []
                obs_dict, info_dict = env.reset()
                done = False
                total_reward = 0.0
                step = 0

                while not done:
                    action_dict = {}
                        
                    for agent_id, obs in obs_dict.items():
                        action_dict[agent_id] = algo.compute_single_action(
                            observation=obs,
                            policy_id="shared_policy",
                            explore=False  # Deterministic for evaluation
                        )

                        obs_dict, reward_dict, terminateds, truncateds, infos = env.step(action_dict)

                        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
                        step_reward = sum(reward_dict.values())
                        total_reward += step_reward
                        step += 1
                    
                episode_rewards.append(total_reward)
                # Calculate stats for this specific configuration
                avg_rew = sum(episode_rewards) / len(episode_rewards)
                print(f"  [RESULT] {int_key} - {sc_name} (Cfg {config_idx}): Avg Reward = {avg_rew:.3f}")

                # Stop the algo to free memory before the next scenario re-creates it
                algo.stop()

    print("\n--- EVALUATION COMPLETE ---")
    ray.shutdown()

if __name__ == "__main__":
    if args.train:
        train()
    else:
        evaluate(checkpoint_path=args.eval)
