"""
v0.2 Single-Agent PPO Config — highway-env intersection
========================================================
"""

import argparse
import os
import sys
from tqdm import tqdm
from datetime import datetime
import torch
import time
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train or evaluate AlphaEnv v0.2 PPO agent.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true", help="Run training loop.")
group.add_argument("--eval", metavar="CHECKPOINT_PATH",
                   help="Path to a checkpoint directory to load and evaluate.")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Ray / RLlib setup
# ─────────────────────────────────────────────
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

# Ensure envs package is importable both locally and on Ray workers
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)
from envs.alpha_env_v02 import AlphaEnvV02


# ─────────────────────────────────────────────
# Callbacks 
# ─────────────────────────────────────────────
class TrafficCallbacks(RLlibCallback):

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        # In the new API stack, episode info lives in the episode object.
        # Our env attaches telemetry to the info dict on terminal steps.
        info = episode.get_infos()
        if not info:
            return
        # get_infos() returns a list; grab the last info dict
        last_info = info[-1] if isinstance(info, list) else info
        telemetry = last_info.get("telemetry") if isinstance(last_info, dict) else None
        if telemetry is None:
            return

        metrics_logger.log_value(
            "collisions",
            float(telemetry.get("number_of_collisions", 0)),
            reduce="mean",
            window=10,
        )
        metrics_logger.log_value(
            "avg_speed",
            float(telemetry.get("avg_speed", 0.0)),
            reduce="mean",
            window=10,
        )

    def on_train_result(self, *, algorithm, metrics_logger, result: dict, **kwargs):
        # Strip noisy TB keys
        for key in [
            "sampler_results", "connector_metrics", "sampler_perf", "perf",
            "agent_timesteps_total", "episodes_this_iter",
            "config", "date", "timestamp",
            "time_this_iter_s", "time_total_s",
            "pid", "hostname", "node_ip", "trial_id", "experiment_id", "done",
            "timers", "counters",
            "num_healthy_workers", "num_recreated_workers",
            "num_agent_steps_sampled", "num_agent_steps_trained",
            "timesteps_total", "time_since_restore",
        ]:
            result.pop(key, None)

        if "evaluation" in result and isinstance(result["evaluation"], dict):
            eval_dict = result["evaluation"]
            for key in [
                "sampler_results", "agent_timesteps_total", "episodes_this_iter",
                "connector_metrics", "sampler_perf",
                "num_healthy_workers", "num_recreated_workers",
                "num_agent_steps_sampled", "num_agent_steps_trained",
            ]:
                eval_dict.pop(key, None)


# ─────────────────────────────────────────────
# Environment registration
# ─────────────────────────────────────────────
ENV_NAME = "alpha_env_v02"

def create_env(env_config):
    return AlphaEnvV02(env_config)

register_env(ENV_NAME, create_env)


# ─────────────────────────────────────────────
# PPO config builder 
# ─────────────────────────────────────────────
def build_config(num_workers: int = 7, render: bool = False) -> PPOConfig:
    env_cfg = {}
    if render:
        env_cfg["render_mode"] = "human"

    cfg = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_cfg)
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
        )
        .training(
            train_batch_size_per_learner=2048,
            minibatch_size=256,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,
            kl_coeff=0.2,
            kl_target=0.01,
            entropy_coeff=0.01,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
            evaluation_num_env_runners=1,
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
# Checkpoint & TensorBoard paths
# ─────────────────────────────────────────────
ALGO_NAME = "PPO"
STORAGE_DIR = os.path.join(os.getcwd(), "ray_results")

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(),
    "checkpoints/v0_2/",
    f"{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)


# ─────────────────────────────────────────────
# Train / Evaluate
# ─────────────────────────────────────────────
def train():
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": SRC_DIR})

    algo = build_config(num_workers=4).build()

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    print(f"\n--- TRAINING START ---")
    print(f"Checkpoints → {CHECKPOINT_ROOT}")
    print(f"TensorBoard → tensorboard --logdir {STORAGE_DIR}\n")

    iters = 300 
    for i in tqdm(range(iters)):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        print(f"  Iter {i+1:3d} | mean_reward={mean_reward:.3f}")

        if i % 10 == 0 or i == iters-1:
            save_path = algo.save(checkpoint_dir=CHECKPOINT_ROOT)
            print(f"    --> Checkpoint saved: {save_path}")

    print("\n--- TRAINING COMPLETE ---")
    ray.shutdown()


def evaluate(checkpoint_path: str, num_episodes: int = 20):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": SRC_DIR})

    algo = build_config(num_workers=0, render=True).build()
    algo.restore(os.path.abspath(checkpoint_path))

    # Create env directly (algo.workers is deprecated)
    env = AlphaEnvV02({"render_mode": "human"})

    print(f"\n--- EVALUATION START ---")
    print(f"Loaded checkpoint: {checkpoint_path}")

    rewards = []
    module = algo.get_module()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n=== Episode {episode + 1} ===")

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                result = module.forward_inference({"obs": obs_tensor})
                action = int(torch.argmax(result["action_dist_inputs"], dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if step % 5 == 0:
                print(f"  Step {step:4d} | reward={reward:.4f}")

        print(f"  --> Episode total reward: {total_reward:.3f}")
        rewards.append(total_reward)

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"\n  Average reward over {num_episodes} episodes: {avg:.3f}")
    print("--- EVALUATION COMPLETE ---\n")

    env.close()
    ray.shutdown()


if __name__ == "__main__":
    if args.train:
        train()
    else:
        evaluate(checkpoint_path=args.eval)
