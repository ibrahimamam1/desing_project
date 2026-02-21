import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime

import random
import numpy as np

parser = argparse.ArgumentParser(description="Train or evaluate the AlphaEnv PPO agent (SB3).")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true", help="Run training loop.")
group.add_argument("--eval",  metavar="CHECKPOINT_PATH",
                   help="Path to a .zip model file to load and evaluate.")
args = parser.parse_args()

from flow.networks.all_turning_intersection import AllTurningIntersectionNetwork as myNet
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

min_gap       = 0.9
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
RENDER_MODE = False

max_vehicle_count_in_inflow = 20
num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)
num_rl_vehicles      = 1
num_non_rl_vehicles  = 7

rl_speed_mode    = 0
non_rl_speed_mode = 31

vehicles = VehicleParams()

RL_car_following_params = SumoCarFollowingParams(
    speed_mode=rl_speed_mode,
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

NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=non_rl_speed_mode,
    accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau,
    min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev,
    impatience=impatience, car_follow_model=car_follow_model,
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

root_dir        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir    = os.path.join(root_dir, "networks")

net_file_name = "100m_unregulated.net.xml"
net_file      = os.path.join(net_file_dir, net_file_name)

net_params = NetParams(
    osm_path=None,
    template=net_file,
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

myTag = "AlphaV0.1"
horizon = 200
sim_step = 0.25
number_of_sim_steps_per_RlAction_step = 1

env_params = EnvParams(
    additional_params={
        "max_accel": max_accel,
        "max_decel": max_decel,
        "target_velocity": max_speed,
        "sort_vehicles": False,
    },
    horizon=horizon,
    warmup_steps=0,
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
# Environment factory
# ─────────────────────────────────────────────
def create_flow_env(render=False):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.alpha_env_v01 import AlphaEnv_v01

    params       = flow_params
    _vehicles    = deepcopy(params["veh"])
    _net_params  = params["net"]
    _sim_params  = deepcopy(params["sim"])
    _sim_params.render = render
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
    return AlphaEnv_v01(
        env_params=params["env"],
        sim_params=_sim_params,
        network=network,
        simulator=params["simulator"],
    )


# ─────────────────────────────────────────────
# Stable Baselines 3 setup
# ─────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.logger import configure


class TelemetryCallback(BaseCallback):
    """
    Logs custom telemetry metrics (collisions, avg_speed) to TensorBoard
    at the end of each episode, mirroring TrafficCallbacks from the RLlib config.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []

    def _on_step(self) -> bool:
        # Check for episode end via 'dones'
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][idx]
                telemetry = info.get("telemetry", None)
                if telemetry is not None:
                    collisions = telemetry.get("number_of_collisions", 0)
                    avg_speed = telemetry.get("avg_speed", 0.0)
                    self.logger.record("custom/collisions", float(collisions))
                    self.logger.record("custom/avg_speed", float(avg_speed))
        return True


# ─────────────────────────────────────────────
# Checkpoint / logging paths
# ─────────────────────────────────────────────
ENV_NAME  = "alpha_env_v01"
ALGO_NAME = "PPO_SB3"

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(),
    "checkpoints",
    f"{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs")
RUN_NAME = f"sb3_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)

TOTAL_TIMESTEPS = 4_000_000       # Comparable to ~1000 RLlib iters × 2000 steps/iter
CHECKPOINT_FREQ = 20_000          # Save every 20k steps
EVAL_FREQ       = 10_000          # Evaluate every 10k steps
EVAL_EPISODES   = 5


def train():
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    os.makedirs(TENSORBOARD_RUN_DIR, exist_ok=True)

    env = create_flow_env(render=False)
    eval_env = create_flow_env(render=False)

    # PPO hyperparameters — matched to the RLlib config
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,              # Rollout buffer size per update
        batch_size=256,            # Minibatch size (was sgd_minibatch_size)
        n_epochs=10,               # SGD epochs per update (was num_sgd_iter)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,            # PPO clip parameter
        clip_range_vf=10.0,        # Value function clip
        max_grad_norm=0.5,         # Gradient clipping (was grad_clip)
        ent_coef=0.01,             # Entropy coefficient
        vf_coef=0.5,               # Value function loss weight
        target_kl=0.01,            # Early stopping if KL exceeds this
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        seed=42,
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_ROOT,
        name_prefix="ppo_alpha",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(CHECKPOINT_ROOT, "best_model"),
        log_path=os.path.join(CHECKPOINT_ROOT, "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    telemetry_cb = TelemetryCallback()

    print(f"\n--- TRAINING START (Stable Baselines 3) ---")
    print(f"Checkpoints → {CHECKPOINT_ROOT}")
    print(f"TensorBoard → {TENSORBOARD_DIR}/{RUN_NAME}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, telemetry_cb],
        tb_log_name=RUN_NAME,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_ROOT, "ppo_alpha_final")
    model.save(final_path)
    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Final model saved: {final_path}.zip")

    env.close()
    eval_env.close()


def evaluate(checkpoint_path: str, num_episodes: int = 20):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    env = create_flow_env(render=True)
    model = PPO.load(checkpoint_path, env=env)

    print(f"\n--- EVALUATION START (Stable Baselines 3) ---")
    print(f"Loaded model: {checkpoint_path}")

    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n=== Episode {episode + 1} ===")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            print(f"  Step {step:4d} | obs={obs} | action={action} | reward={reward:.4f}")

        print(f"  --> Episode total reward: {total_reward:.3f}")
        rewards.append(total_reward)

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"\n  Average reward over {num_episodes} episodes: {avg:.3f}")
    print("--- EVALUATION COMPLETE ---\n")

    env.close()


if __name__ == "__main__":
    if args.train:
        train()
    else:
        evaluate(checkpoint_path=args.eval)
