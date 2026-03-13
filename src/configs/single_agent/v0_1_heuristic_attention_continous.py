import argparse
import os
import sys
import csv
from copy import deepcopy
from datetime import datetime
import shutil
import math 

parser = argparse.ArgumentParser(description="Train or evaluate (Heuristic+Attention Continuous).")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--eval", metavar="CHECKPOINT_PATH")
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.uniform_random import UniformRandomNetwork as myNet
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

myTag = "AlphaV0.1_HeuristicAttention_Continous"
min_gap=2.5
max_accel=2.6
max_decel=4.5
max_speed=55
initial_speed=0
speed_factor=1.0
speed_dev=0.1
impatience=0.0
car_follow_model="IDM"
sigma=0
tau=0.8
horizon=180
sim_step=0.25
warmup_steps=5
number_of_sim_steps_per_RlAction_step=1

vehicles = VehicleParams()
RL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev, impatience=impatience,
    car_follow_model=car_follow_model)
NonRL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev, impatience=impatience,
    car_follow_model=car_follow_model)
vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
    initial_speed=0, num_vehicles=0, car_following_params=RL_cfp,
    lane_change_params=None, color="blue")
vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
    initial_speed=initial_speed, num_vehicles=0, car_following_params=NonRL_cfp,
    lane_change_params=None, color="red")


high=500; medium=300; low=150
traffic_rates = [
    {"N": medium, "S": medium, "W": medium, "E": medium},
    {"N": high, "S": medium, "W": medium, "E": medium},
    {"N": high, "S": high, "W": medium, "E": medium},
    {"N": high, "S": high, "W": high, "E": medium},
]

def _build_inflows(tr):
    inf = InFlows()
    inf.add(veh_type="NonRL", edge="E#T-X", probability=tr["N"]/3600, depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#R-X", probability=tr["E"]/3600, depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#D-X", probability=tr["S"]/3600, depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#L-X", probability=tr["W"]/3600, depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="RL", edge="E#L-X", probability=tr["W"]/3600, depart_lane=0, depart_speed=initial_speed, begin=warmup_steps, number=1, color="red")
    return inf

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
net_file = os.path.join(root_dir, "networks", "100m_right_before_left.net.xml")

net_params = NetParams(osm_path=None, template=net_file, inflows=_build_inflows(traffic_rates[0]))

EDGES_DISTRIBUTION = ["E#D-X", "E#L-X", "E#R-X", "E#T-X"]

initial_config = InitialConfig(shuffle=False, spacing="uniform", min_gap=12, perturbation=5.0,
    x0=5, bunching=0, lanes_distribution=float("inf"), edges_distribution=EDGES_DISTRIBUTION)

env_params = EnvParams(additional_params={"max_accel": max_accel, "max_decel": max_decel,
    "target_velocity": max_speed, "sort_vehicles": False}, horizon=horizon, warmup_steps=5,
    sims_per_step=number_of_sim_steps_per_RlAction_step, evaluate=False, clip_actions=True)

sim_params = SumoParams(port=None, sim_step=sim_step, emission_path=os.path.join(root_dir, "results"),
    lateral_resolution=None, no_step_log=True, render=False, save_render=False,
    sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
    overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
    teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False)

flow_params = dict(exp_tag=myTag, network=myNet, simulator="traci",
    sim=sim_params, env=env_params, net=net_params, veh=vehicles, initial=initial_config)

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.attention_model import AttentionPolicyModel
ModelCatalog.register_custom_model("attention_policy", AttentionPolicyModel)

class TrafficCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        info = episode.last_info_for()
        if not info or "telemetry" not in info: return
        t = info["telemetry"]
        if not t: return
        episode.custom_metrics["collision"] = 1.0 if t.get("agent_collision") else 0.0
        episode.custom_metrics["success"] = 1.0 if t.get("agent_success") else 0.0
        episode.custom_metrics["avg_speed"] = float(t.get("agent_avg_speed", 0.0))
        episode.custom_metrics["travel_time"] = float(t.get("agent_travel_time", 0.0))
        episode.custom_metrics["waiting_time"] = float(t.get("agent_waiting_time", 0.0))

    def on_train_result(self, *, algorithm, result, **kwargs):
        keep = {"episode_reward_mean","episode_len_mean","custom_metrics","info","training_iteration","timesteps_total"}
        for k in [k for k in result if k not in keep]: result.pop(k)
        if "custom_metrics" in result:
            ck = {"collision_mean","success_mean","avg_speed_mean","travel_time_mean","waiting_time_mean"}
            for k in [k for k in result["custom_metrics"] if k not in ck]: result["custom_metrics"].pop(k)

def create_flow_env(env_config):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.alpha_env_v01_heuristic_attention_continous import AlphaEnv_v01_HeuristicAttention
    p = flow_params
    _v = deepcopy(p["veh"]); _n = p["net"]; _s = deepcopy(p["sim"])
    _s.render = env_config.get("render", False)
    net = p["network"](name="AlphaEnv-Check", vehicles=_v, net_params=_n,
        initial_config=p.get("initial", InitialConfig()), traffic_lights=p.get("tls", TrafficLightParams()))
    return AlphaEnv_v01_HeuristicAttention(env_params=p["env"], sim_params=_s, network=net, simulator=p["simulator"])

register_env("alpha_env_v01_heuristic_attention_continous", create_flow_env)

def build_config(num_workers=7, render=False):
    return (PPOConfig()
        .environment(env="alpha_env_v01_heuristic_attention_continous", env_config={"render": render}, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers, rollout_fragment_length="auto", num_envs_per_worker=1)
        .training(train_batch_size=2048, sgd_minibatch_size=256, num_sgd_iter=10,
            lr=[[0, 3e-4], [2_000_000, 1e-5]], entropy_coeff=[[0, 0.02], [2_000_000, 0.0]],
            gamma=0.995, lambda_=0.95, clip_param=0.2, vf_clip_param=50.0, grad_clip=0.5,
            kl_coeff=0.2, kl_target=0.01,
            model={"custom_model": "attention_policy", "custom_model_config": {
                "ego_features": 2, "neighbor_features": 3, "max_neighbors": 5,
                "embed_dim": 64, "num_heads": 4, "mlp_hidden": 256}})
        .evaluation(evaluation_interval=100, evaluation_duration=10, evaluation_num_workers=1)
        .debugging(log_level="ERROR")
        .reporting(metrics_num_episodes_for_smoothing=10, min_time_s_per_iteration=0, min_sample_timesteps_per_iteration=2000)
        .resources(num_gpus=0).callbacks(TrafficCallbacks))


ENV_NAME  = "alpha_env_v01"
ALGO_NAME = "PPO"

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(),
    "checkpoints/v0_1",
    f"{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
FINAL_MODEL_DIR = os.path.join(CHECKPOINT_ROOT, "final")
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "best")

TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs/v0_1")
RUN_NAME = f"flow_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)

# CSV for training stats
TRAIN_STATS_CSV = os.path.join(CHECKPOINT_ROOT, "train_stats.csv")
TRAIN_STATS_HEADER = [
    "iteration", "timesteps_total", "stage", "traffic_rate",
    "collision_rate", "success_rate", "avg_speed", "avg_travel_time", "avg_waiting_time",
]

def _write_train_stats_row(csv_path, row_dict):
    """Append one row to the training stats CSV."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_STATS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def train():
    import logging
    import warnings
    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    logging.getLogger("ray.tune.trainable").setLevel(logging.ERROR)

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
    
    num_iterations = 10
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

            _write_train_stats_row(TRAIN_STATS_CSV, {
                "iteration": global_iter,
                "timesteps_total": result.get("timesteps_total", 0),
                "stage": stage_idx,
                "traffic_rate": str(traffic_rate),
                "collision_rate": f"{custom.get('collision_mean', 0.0):.4f}",
                "success_rate": f"{custom.get('success_mean', 0.0):.4f}",
                "avg_speed": f"{custom.get('avg_speed_mean', 0.0):.4f}",
                "avg_travel_time": f"{custom.get('travel_time_mean', 0.0):.4f}",
                "avg_waiting_time": f"{custom.get('waiting_time_mean', 0.0):.4f}",
            })

            if current_reward is not None and not math.isnan(current_reward):
                if current_reward > best_reward:
                    best_reward = current_reward
                    if os.path.exists(BEST_CHECKPOINT_DIR):
                        shutil.rmtree(BEST_CHECKPOINT_DIR)
                    algo.save(checkpoint_dir=BEST_CHECKPOINT_DIR)
                    print(f"  [⭐ NEW BEST]")

        checkpoint_path = algo.save(checkpoint_dir=os.path.join(CHECKPOINT_ROOT, f"stage_{stage_idx}"))
        algo.stop()

    print("\n--- TRAINING COMPLETE ---")
    print(f"Best Model  → {BEST_CHECKPOINT_DIR}")
    print(f"Stats CSV   → {TRAIN_STATS_CSV}")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}")
    ray.shutdown()

def evaluate(checkpoint_path, num_iterations=20):
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(checkpoint_path)
    ray.init(ignore_reinit_error=True)
    algo = build_config(0, True).build(); algo.restore(checkpoint_path)
    env = algo.workers.local_worker().env
    print(f"\n--- EVAL (Heuristic+Attention Continuous) --- Checkpoint: {checkpoint_path}")
    rews = []
    for ep in range(num_iterations):
        obs, _ = env.reset(); done=False; tr=0.0
        while not done:
            a = algo.compute_single_action(obs); obs, rew, term, trunc, info = env.step(a)
            done = term or trunc; tr += rew
        print(f"  Ep {ep+1}: {tr:.3f}"); rews.append(tr)
    print(f"\n  Avg: {sum(rews)/max(len(rews),1):.3f}\n--- DONE ---"); ray.shutdown()

if __name__ == "__main__":
    if args.train: train()
    else: evaluate(args.eval)
