import argparse
import os
import sys
import csv
from copy import deepcopy
from datetime import datetime
import shutil
import random
import math 

parser = argparse.ArgumentParser(description="Train or evaluate the AlphaEnv PPO agent (Discrete).")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true", help="Run training loop.")
group.add_argument("--eval",  metavar="CHECKPOINT_PATH",
                   help="Path to a checkpoint directory to load and evaluate.")
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from networks.uniform_random import UniformRandomNetwork as myNet
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from utils.plot_train_curves import plot_results
IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

myTag = "AlphaV0.1_Heuristic_Discrete"
min_gap       = 2.5
max_accel     = 2.6
max_decel     = 4.5
max_speed     = 55
initial_speed = 0
speed_factor  = 1.0
speed_dev     = 0.1
impatience    = 0.0
car_follow_model = "IDM"
sigma = 0
tau   = 0.8
horizon = 180
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
high = 500
medium = 300

traffic_rates = [
    {"N": medium, "S": medium, "W": medium, "E": medium},
    {"N": high, "S": medium, "W": medium, "E": medium},
    {"N": high, "S": high, "W": medium, "E": medium},
    {"N": high, "S": high, "W": high, "E": medium},
]

def _build_inflows(traffic_rate):
    """Build an InFlows object from a traffic rate dict."""
    inflow = InFlows()
    
    inflow.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"]/3600,
               depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"]/3600,
               depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"]/3600,
               depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#L-X", probability=traffic_rate["W"]/3600,
               depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inflow.add(veh_type="RL", edge="E#L-X", probability=0.8,
               depart_lane=0, depart_speed=initial_speed, begin=warmup_steps,
                color="green")
    

    return inflow

root_dir        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir    = os.path.join(root_dir, "networks")

net_file_name = "100m_right_before_left.net.xml"
net_file= os.path.join(net_file_dir, net_file_name)

net_params = NetParams(osm_path=None, template=net_file, inflows=_build_inflows(traffic_rates[0]))

EDGES_DISTRIBUTION = ["E#D-X", "E#L-X", "E#R-X", "E#T-X"]

initial_config = InitialConfig(
    shuffle=False, spacing="uniform", min_gap=12, perturbation=5.0,
    x0=5, bunching=0, lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION, additional_params=None,
)
env_params = EnvParams(
    additional_params={"max_accel": max_accel, "max_decel": max_decel,
                       "target_velocity": max_speed, "sort_vehicles": False},
    horizon=horizon, warmup_steps=5, sims_per_step=number_of_sim_steps_per_RlAction_step,
    evaluate=False, clip_actions=True,
)

sim_params = SumoParams(
    port=None, sim_step=sim_step, emission_path=output_file_dir,
    lateral_resolution=None, no_step_log=True, render=False, save_render=False,
    sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
    overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
    teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False,
)

flow_params = dict(
    exp_tag=myTag, network=myNet, simulator="traci",
    sim=sim_params, env=env_params, net=net_params, veh=vehicles, initial=initial_config,
)

# ─────────────────────────────────────────────
# Ray / RLlib setup
# ─────────────────────────────────────────────
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Any, Dict

class TrafficCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs) -> None:
        info = episode.last_info_for()
        if info is None or "telemetry" not in info:
            return
        telemetry = info["telemetry"]
        if telemetry is None:
            return
        episode.custom_metrics["collision"] = 1.0 if telemetry.get("agent_collision", False) else 0.0
        episode.custom_metrics["success"] = 1.0 if telemetry.get("agent_success", False) else 0.0
        episode.custom_metrics["avg_speed"] = float(telemetry.get("agent_avg_speed", 0.0))
        episode.custom_metrics["travel_time"] = float(telemetry.get("agent_travel_time", 0.0))
        episode.custom_metrics["waiting_time"] = float(telemetry.get("agent_waiting_time", 0.0))

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        keep = {"episode_reward_mean", "episode_len_mean", "custom_metrics",
                "info", "training_iteration", "timesteps_total"}
        for k in [k for k in result if k not in keep]:
            result.pop(k)
        if "custom_metrics" in result and isinstance(result["custom_metrics"], dict):
            custom_keep = {"collision_mean", "success_mean", "avg_speed_mean",
                           "travel_time_mean", "waiting_time_mean"}
            for k in [k for k in result["custom_metrics"] if k not in custom_keep]:
                result["custom_metrics"].pop(k)
        if "info" in result and isinstance(result["info"], dict):
            info = result["info"]
            if "learner" in info and isinstance(info["learner"], dict):
                learner = info["learner"]
                if "default_policy" in learner and isinstance(learner["default_policy"], dict):
                    info_keep = {"entropy", "mean_kl_loss", "policy_loss", "total_loss", "vf_loss", "vf_explained_var"}
                    for k in [k for k in learner["default_policy"] if k not in info_keep]:
                        learner["default_policy"].pop(k)
                for k in [k for k in learner if k != "default_policy"]:
                    learner.pop(k) 
            for k in [k for k in result["info"] if k != "learner"]:
                result["info"].pop(k)        

def create_flow_env(env_config):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.envs.alpha_env_v01_discrete import AlphaEnv_v01_Discrete

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
    return AlphaEnv_v01_Discrete(
        env_params=params["env"],
        sim_params=_sim_params,
        network=network,
        simulator=params["simulator"],
    )

register_env("alpha_env_v01_discrete", create_flow_env)

def build_config(num_workers: int = 7, render: bool = False) -> PPOConfig:
        cfg = (
            PPOConfig()
            .environment(env="alpha_env_v01_discrete", env_config={"render": render}, disable_env_checking=True)
            .framework("torch")
            .rollouts(
                num_rollout_workers=num_workers,
                rollout_fragment_length="auto",
                num_envs_per_worker=1,
            )
            .training(
                train_batch_size=2048, sgd_minibatch_size=256, num_sgd_iter=10,
                lr=[[0, 3e-4], [2_000_000, 1e-5]], 
                entropy_coeff=[[0, 0.02], [2_000_000, 0.0]], 
                gamma=0.995, lambda_=0.95, clip_param=0.2,
                vf_clip_param=50.0, grad_clip=0.5, kl_coeff=0.2, kl_target=0.01,
            )
            .evaluation(evaluation_interval=100, evaluation_duration=10, evaluation_num_workers=1)
            .debugging(log_level="ERROR")
            .reporting(metrics_num_episodes_for_smoothing=10, min_time_s_per_iteration=0,
                       min_sample_timesteps_per_iteration=2000)
            .resources(num_gpus=0)
            .callbacks(TrafficCallbacks)
        )
        return cfg

# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
ENV_NAME  = "alpha_env_v01_heuristic_discrete"
ALGO_NAME = "PPO"

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(), "checkpoints/v0_1_heuristic_discrete",
    f"{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
FINAL_MODEL_DIR = os.path.join(CHECKPOINT_ROOT, "final")
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "best")
TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs/v0_1_heuristic_discrete")
RUN_NAME = f"flow_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)
TRAIN_STATS_CSV = os.path.join(CHECKPOINT_ROOT, "train_stats.csv")
TRAIN_STATS_HEADER = [
    "iteration", "timesteps_total", "stage", "traffic_rate",
    "collision_rate", "success_rate", "avg_speed", "avg_travel_time", "avg_waiting_time",
]

def train():
    import logging
    ray.init(
        ignore_reinit_error=True, 
        log_to_driver=False,        
        logging_level=logging.ERROR    
     )
    

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n--- TRAINING START (Discrete - Curriculum) ---")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}")
    print(f"Train Stats CSV → {TRAIN_STATS_CSV}\n")
    
    num_iterations = 1600
    num_stages = len(traffic_rates)
    iters_per_stage = num_iterations // num_stages
    best_reward = -float('inf')
    checkpoint_path = None

    for stage_idx, traffic_rate in enumerate(traffic_rates):
        print(f"\n{'='*60}")
        print(f"  STAGE {stage_idx+1}/{num_stages}: traffic_rate = {traffic_rate}")
        print(f"{'='*60}\n")

        flow_params["net"] = NetParams(osm_path=None, template=net_file,
                                       inflows=_build_inflows(traffic_rate))
        algo = build_config(num_workers=7).build(
            logger_creator=lambda cfg: ray.tune.logger.UnifiedLogger(cfg, TENSORBOARD_RUN_DIR, loggers=None)
        )
        if checkpoint_path is not None:
            algo.restore(checkpoint_path)

        for i in range(iters_per_stage):
            global_iter = stage_idx * iters_per_stage + i
            result = algo.train()
            
            current_reward = result.get("episode_reward_mean")
            custom = result.get("custom_metrics", {})
            
            learner_stats = result.get("info", {}).get("learner", {}).get("default_policy", {})
            explained_var = learner_stats.get("vf_explained_var", float("nan"))
            entropy       = learner_stats.get("entropy", float("nan"))
            total_loss    = learner_stats.get("total_loss", float("nan"))
            # -----------------------

            print(f"stage={stage_idx}, iter={i}, reward={current_reward:.3f}, "
                  f"loss={total_loss:.4f}, entropy={entropy:.4f}, expl_var={explained_var:.4f}", 
                  flush=True)

        checkpoint_path = algo.save(checkpoint_dir=os.path.join(CHECKPOINT_ROOT, f"stage_{stage_idx}"))
        algo.stop()

    print("\n--- TRAINING COMPLETE ---")
    print(f"Saved Models  → {CHECKPOINT_ROOT}")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}")
    
    plot_out = os.path.join(root_dir, "outputs", "train", RUN_NAME)
    plot_results(logdir=TENSORBOARD_RUN_DIR, output_dir=plot_out, exp_name=RUN_NAME)
    ray.shutdown()

def evaluate(checkpoint_path: str, num_iterations: int = 20):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ray.init(ignore_reinit_error=True)
    algo = build_config(num_workers=0, render=True).build()
    algo.restore(checkpoint_path)
    env = algo.workers.local_worker().env

    print(f"\n--- EVALUATION START (Discrete) ---")
    print(f"Loaded checkpoint: {checkpoint_path}")

    rewards = []
    for episode in range(num_iterations):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        print(f"  Episode {episode+1}: reward={total_reward:.3f}")
        rewards.append(total_reward)

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"\n  Average reward: {avg:.3f}")
    print("--- EVALUATION COMPLETE ---\n")
    ray.shutdown()

if __name__ == "__main__":
    if args.train:
        train()
    else:
        evaluate(checkpoint_path=args.eval)
