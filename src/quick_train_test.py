#!/usr/bin/env python3
"""
Quick training test — short PPO run to verify everything works end-to-end.
Trains Attention+Continuous for 20,000 timesteps (~5 min).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))

from copy import deepcopy
from datetime import datetime

from networks.uniform_random import UniformRandomNetwork
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from models.attention_model import AttentionFeatureExtractor

# ─── Parameters ───
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_FILE = os.path.join(ROOT_DIR, "networks", "100m_right_before_left.net.xml")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "quick_test",
                               f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
TB_DIR = os.path.join(ROOT_DIR, "tensorboard_logs", "quick_test")

# Simulation
max_accel = 2.6; max_decel = 4.5; max_speed = 55; sim_step = 0.25
horizon = 180; warmup_steps = 50

# Traffic
traffic_rate = {"N": 275, "S": 275, "W": 275, "E": 275}

# ─── Vehicle Setup ───
vehicles = VehicleParams()
RL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
    sigma=0, tau=0.8, min_gap=2.5, max_speed=max_speed,
    speed_factor=1.0, speed_dev=0.1, impatience=0.0, car_follow_model="IDM")
NonRL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
    sigma=0, tau=0.8, min_gap=2.5, max_speed=max_speed,
    speed_factor=1.0, speed_dev=0.1, impatience=0.0, car_follow_model="IDM")
vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
    initial_speed=0, num_vehicles=0, car_following_params=RL_cfp,
    lane_change_params=None, color="blue")
vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
    initial_speed=0, num_vehicles=0, car_following_params=NonRL_cfp,
    lane_change_params=None, color="red")

# ─── InFlows ───
inflow = InFlows()
inflow.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"]/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"]/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"]/3600,
           depart_lane=0, depart_speed=0, begin=1, color="green")
inflow.add(veh_type="RL", edge="E#L-X", probability=0.3,
           depart_lane=0, depart_speed=0, begin=warmup_steps, color="green")

# ─── Flow Params ───
net_params = NetParams(osm_path=None, template=NET_FILE, inflows=inflow)
sim_params = SumoParams(port=None, sim_step=sim_step, lateral_resolution=None,
    no_step_log=True, render=False, save_render=False, sight_radius=25,
    show_radius=False, pxpm=2, force_color_update=False, overtake_right=False,
    seed=42, restart_instance=True, print_warnings=False, teleport_time=0,
    num_clients=1, color_by_speed=False, use_ballistic=False)
env_params = EnvParams(
    additional_params={"max_accel": max_accel, "max_decel": max_decel,
                       "target_velocity": max_speed, "sort_vehicles": False},
    horizon=horizon, warmup_steps=5, sims_per_step=1,
    evaluate=False, clip_actions=True)
initial_config = InitialConfig(shuffle=False, spacing="uniform", min_gap=12,
    perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
    edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"])

flow_params = dict(
    network=UniformRandomNetwork, sim=sim_params, env=env_params,
    net=net_params, veh=vehicles, initial=initial_config,
)

# ─── Environment Factory ───
def make_env():
    from envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
    p = flow_params
    _vehicles = deepcopy(p["veh"])
    _sim_params = deepcopy(p["sim"])
    network = p["network"](
        name="QuickTest", vehicles=_vehicles, net_params=p["net"],
        initial_config=p.get("initial", InitialConfig()),
        traffic_lights=TrafficLightParams())
    env = AlphaEnv_v01_Attention(
        env_params=p["env"], sim_params=_sim_params,
        network=network, simulator="traci")
    return Monitor(env)

# ─── Callback ───
class QuickCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.ep_count = 0
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "telemetry" in info:
                t = info["telemetry"]
                self.ep_count += 1
                col = "CRASH" if t.get("agent_collision") else "OK"
                suc = "GOAL" if t.get("agent_success") else "---"
                print(f"  Ep {self.ep_count:3d} | {col:5s} {suc:4s} | "
                      f"spd={t.get('agent_avg_speed',0):.1f} tt={t.get('agent_travel_time',0):.1f}")
        return True

# ─── Main ───
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    NUM_WORKERS = 4
    TOTAL_TIMESTEPS = 20_000  # ~5 min with 4 workers
    
    print(f"\n{'='*60}")
    print(f"  QUICK TRAINING TEST — Attention+Continuous")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS}")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")
    
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_WORKERS)])
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256, ego_features=4, neighbor_features=5,
                max_neighbors=5, embed_dim=64, num_heads=4),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.25,
        max_grad_norm=0.5,
        ent_coef=0.01,
        tensorboard_log=TB_DIR,
        verbose=1,
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=QuickCallback(), progress_bar=True)
    
    save_path = os.path.join(CHECKPOINT_DIR, "quick_model")
    model.save(save_path)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Model saved: {save_path}.zip")
    print(f"{'='*60}\n")
    
    vec_env.close()
