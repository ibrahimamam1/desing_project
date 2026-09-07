#!/usr/bin/env python3
"""
Quick evaluation — loads the trained model and runs 3 episodes.
"""
import sys, os, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Must set paths BEFORE loading the model (cloudpickle needs to find modules)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "envs"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))

from copy import deepcopy
from networks.uniform_random import UniformRandomNetwork
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
from src.models.attention_model import AttentionFeatureExtractor

# ─── Config ───
NET_FILE = os.path.join(ROOT, "networks", "100m_right_before_left.net.xml")
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "checkpoints", "quick_test", "test_20260426_001251", "quick_model.zip")
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# ─── Environment Setup ───
vehicles = VehicleParams()
RL_cfp = SumoCarFollowingParams(speed_mode=0, accel=2.6, decel=4.5, sigma=0, tau=0.8,
    min_gap=2.5, max_speed=55, speed_factor=1.0, speed_dev=0.1,
    impatience=0.0, car_follow_model="IDM")
NonRL_cfp = SumoCarFollowingParams(speed_mode=0, accel=2.6, decel=4.5, sigma=0, tau=0.8,
    min_gap=2.5, max_speed=55, speed_factor=1.0, speed_dev=0.1,
    impatience=0.0, car_follow_model="IDM")
vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
    initial_speed=0, num_vehicles=0, car_following_params=RL_cfp,
    lane_change_params=None, color="blue")
vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
    initial_speed=0, num_vehicles=0, car_following_params=NonRL_cfp,
    lane_change_params=None, color="red")

inflow = InFlows()
inflow.add(veh_type="NonRL", edge="E#T-X", probability=275/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#R-X", probability=275/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#D-X", probability=275/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="RL", edge="E#L-X", probability=0.3,
           depart_lane=0, depart_speed=0, begin=50, color="green")

net_params = NetParams(osm_path=None, template=NET_FILE, inflows=inflow)
sim_params = SumoParams(port=None, sim_step=0.25, lateral_resolution=None,
    no_step_log=True, render=False, save_render=False, sight_radius=25,
    show_radius=False, pxpm=2, force_color_update=False, overtake_right=False,
    seed=42, restart_instance=True, print_warnings=False, teleport_time=0,
    num_clients=1, color_by_speed=False, use_ballistic=False)
env_params = EnvParams(
    additional_params={"max_accel": 2.6, "max_decel": 4.5,
                       "target_velocity": 55, "sort_vehicles": False},
    horizon=180, warmup_steps=5, sims_per_step=1,
    evaluate=False, clip_actions=True)
initial_config = InitialConfig(shuffle=False, spacing="uniform", min_gap=12,
    perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
    edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"])

def make_env():
    network = UniformRandomNetwork(name="eval", vehicles=deepcopy(vehicles),
        net_params=net_params, initial_config=initial_config,
        traffic_lights=TrafficLightParams())
    return AlphaEnv_v01_Attention(env_params=env_params, sim_params=deepcopy(sim_params),
        network=network, simulator="traci")

# ─── Load & Evaluate ───
print(f"\n{'='*60}")
print(f"  QUICK EVALUATION")
print(f"  Model: {MODEL_PATH}")
print(f"  Episodes: {N_EPISODES}")
print(f"{'='*60}\n")

model = PPO.load(MODEL_PATH)
print("Model loaded OK!\n")

env = DummyVecEnv([make_env])
results = []

for ep in range(N_EPISODES):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if dones[0]:
            info = infos[0]
            telemetry = info.get("telemetry", {})
            collision = telemetry.get("agent_collision", False)
            success = telemetry.get("agent_success", False)
            avg_speed = telemetry.get("agent_avg_speed", 0)
            travel_time = telemetry.get("agent_travel_time", 0)
            
            result = {
                "ep": ep+1, "steps": steps, "reward": total_reward,
                "collision": collision, "success": success,
                "avg_speed": avg_speed, "travel_time": travel_time,
            }
            results.append(result)
            
            status = "💥 CRASH" if collision else ("✅ GOAL" if success else "⏰ TIMEOUT")
            print(f"  Ep {ep+1:2d}: {status:12s} | steps={steps:3d} | "
                  f"R={total_reward:+7.3f} | spd={avg_speed:.1f} | tt={travel_time:.1f}s")
            break

env.close()

# ─── Summary ───
import numpy as np
collisions = sum(1 for r in results if r["collision"])
successes = sum(1 for r in results if r["success"])
avg_reward = np.mean([r["reward"] for r in results])

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"  Collisions: {collisions}/{N_EPISODES} ({100*collisions/N_EPISODES:.0f}%)")
print(f"  Successes:  {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.0f}%)")
print(f"  Avg Reward: {avg_reward:.3f}")
print(f"{'='*60}\n")
