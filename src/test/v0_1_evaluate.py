"""
V0.1 Evaluation Script — mirrors run_flow_tests.py naming convention.

Usage:
    python v0_1_evaluate.py --checkpoint PATH --version continuous
    python v0_1_evaluate.py --checkpoint PATH --version discrete
    python v0_1_evaluate.py --checkpoint PATH --version attention
    python v0_1_evaluate.py --checkpoint PATH --version attention_discrete
"""
import argparse
import os
import math
import sys
import csv
import random
import gc
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from networks.uniform_random import UniformRandomNetwork
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.asymetric_random import AsymmetricRandomNetwork

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from plot_eval_results import plot_eval_results

# ─────────────────────── CLI ───────────────────────
parser = argparse.ArgumentParser(description="Evaluate trained v0.1 agent.")
parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
parser.add_argument("--version", required=True,
                    choices=["heuristic_continous", "heuristic_discrete",
                             "attention_continous", "attention_discrete",
                             "heuristic_attention_continous", "heuristic_attention_discrete"])
parser.add_argument("--n_sims", type=int, default=42, help="Runs per scenario combo.")
parser.add_argument("--render", action="store_true", default=False)
args = parser.parse_args()

# ─────────────── Sim Params ────────────────────
min_gap=2.5; max_accel=2.6; max_decel=4.5; max_speed=55; initial_speed=0
speed_factor=1.0; speed_dev=0.1; sigma=0; tau=0.8; horizon=180; sim_step=0.25
warmup_steps=50

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
net_file = os.path.join(root_dir, "networks", "100m_right_before_left.net.xml")
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# ─────────────── Scenarios (same as run_flow_tests) ──────────
scenarios = {"rbl": "100m_right_before_left.net.xml"}

intentions = {
   # "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
   # "uniform_random": UniformRandomNetwork,
   # "asymetric_random": AsymmetricRandomNetwork 
}

high_rate=500; medium_rate=300; low_rate=150
traffic_rates = {
    "Sc1_All_low":    [{"N": low_rate, "S": low_rate, "W": low_rate, "E": low_rate}],
    "Sc3_All_medium": [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}],
    "Sc2_All_high_3H": [
        {"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
        {"N": high_rate, "S": high_rate, "W": high_rate, "E": medium_rate},
        {"N": high_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
        {"N": high_rate, "S": medium_rate, "W": high_rate, "E": high_rate},
        {"N": medium_rate, "S": high_rate, "W": high_rate, "E": high_rate},
    ],
    "Sc4_Mixed_2H": [
        {"N": high_rate, "S": high_rate, "W": low_rate, "E": low_rate},
        {"N": low_rate, "S": low_rate, "W": high_rate, "E": high_rate},
        {"N": high_rate, "S": low_rate, "W": high_rate, "E": low_rate},
        {"N": low_rate, "S": high_rate, "W": low_rate, "E": high_rate},
    ],
    "Sc5_Mixed_1H": [
        {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
    ],
    "Sc6_Mixed_ML": [
        {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
        {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
        {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
    ],
}

CSV_HEADER = [
    "run", "collision", "success", "avg_speed", "travel_time", "waiting_time",
]

# ─────────────── Version-specific setup ──────────────
def _get_env_class(version):
    if version == "heuristic_continous":
        from envs.alpha_env_v01 import AlphaEnv_v01
        return AlphaEnv_v01, "alpha_env_v01_eval"
    elif version == "heuristic_discrete":
        from envs.alpha_env_v01_discrete import AlphaEnv_v01_Discrete
        return AlphaEnv_v01_Discrete, "alpha_env_v01_discrete_eval"
    elif version == "attention_continous":
        from envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
        return AlphaEnv_v01_Attention, "alpha_env_v01_attention_eval"
    elif version == "attention_discrete":
        from envs.alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete
        return AlphaEnv_v01_AttentionDiscrete, "alpha_env_v01_attn_disc_eval"
    elif version == "heuristic_attention_continous":
        from envs.alpha_env_v01_heuristic_attention_continous import AlphaEnv_v01_HeuristicAttention
        return AlphaEnv_v01_HeuristicAttention, "alpha_env_v01_heur_attn_eval"
    elif version == "heuristic_attention_discrete":
        from envs.alpha_env_v01_heuristic_attention_discrete import AlphaEnv_v01_HeuristicAttentionDiscrete
        return AlphaEnv_v01_HeuristicAttentionDiscrete, "alpha_env_v01_heur_attn_disc_eval"

def _build_inflows(traffic_rate):
    inf = InFlows()
    inf.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="NonRL", edge="E#L-X", probability=traffic_rate["W"]/3600,
            depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
    inf.add(veh_type="RL", edge="E#L-X", probability=0.8,
            depart_lane=0, depart_speed=initial_speed, begin=warmup_steps, color="green")
    return inf

def _risk_bar(value, width=10):
    """value in [0,1] where 0=dangerous, 1=safe. Returns a colored bar string."""
    filled = int((1 - value) * width)
    bar = "█" * filled + "░" * (width - filled)
    if value < 0.3:
        color = "\033[91m"   # red
    elif value < 0.6:
        color = "\033[93m"   # yellow
    else:
        color = "\033[92m"   # green
    return f"{color}{bar}\033[0m"

def _angle_arrow(sin_val, cos_val):
    angle_deg = math.degrees(math.atan2(sin_val, cos_val))
    # atan2: east=0°, north=90°, west=±180°, south=-90°
    # Shift so that east (0°) maps to index 0, going CCW
    idx = round(angle_deg / 45) % 8
    arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    return arrows[idx]

def print_neighbor_table(step_num, obs, reward, neighbors_info, terminated, truncated):
    os.system("cls" if os.name == "nt" else "clear")

    # --- Ego stats from obs vector ---
    dis_to_goal = obs[0]
    ego_speed = obs[1]
    ego_sin, ego_cos = obs[2], obs[3]
    ego_dir = _angle_arrow(ego_sin, ego_cos)

    print(f"╔{'═'*72}╗")
    print(f"║  Step {step_num:<6}   Reward: {reward:+.3f}   "
          f"{'TERMINATED' if terminated else 'TRUNCATED' if truncated else 'running  ':<12}║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  EGO   dir:{ego_dir}  speed:{ego_speed:.2f}  dist_to_goal:{dis_to_goal:.2f}   ║")
    print(f"╠══════════════════════════════════════════════════════════╣")

    if not neighbors_info:
        print(f"║  No conflicting neighbors in perception radius.          ║")
    else:
        print(f"║  {'#':<3} {'dir':<4} {'speed':>6} {'dist':>6} {'TTC':>6} {'d(gap)':>8}  risk-bar      {'edge':<12}║")
        print(f"║  {'─'*70}║")
        for i, n in enumerate(neighbors_info):
            direction = _angle_arrow(n['sinx'], n['cosx'])
            speed_pct = n['v']
            dist_norm = n['distance']
            ttc = n['ttc']
            d = n['s']
            bar = _risk_bar(ttc)
            d_label = f"{d:+.2f}"
            label = (f"({'behind' if d < -0.1 else 'ahead' if d > 0.1 else 'cross':>6})")
            edge = n['edge'][:12].ljust(12)   # truncate long edge IDs to keep layout stable

            print(f"║  {i+1:<3} {direction:<4} {speed_pct:>6.2f} {dist_norm:>6.2f}"
                  f" {ttc:>6.2f} {d_label:>8} {label}  {bar}  {edge}║")
    
    print(f"╚{'═'*72}╝")
    print()

def main():
    version = args.version
    checkpoint_path = args.checkpoint
    n_sims = args.n_sims

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    EnvClass, env_name = _get_env_class(version)

    # Register attention model if needed
    if "attention" in version:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"))
        from models.attention_model import AttentionPolicyModel
        ModelCatalog.register_custom_model("attention_policy", AttentionPolicyModel)

    # Ray is initialised once for the whole script and shut down at the very end.
    # We do NOT call ray.shutdown() between runs — that would be expensive and
    # can leave orphaned processes. Instead we destroy only the algo + env objects.
    ray.init(ignore_reinit_error=True)

    vehicles = VehicleParams()
    RL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev, impatience=0.0,
        car_follow_model="IDM")
    NonRL_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
        sigma=sigma, tau=tau, min_gap=min_gap, max_speed=max_speed,
        speed_factor=speed_factor, speed_dev=speed_dev, impatience=0.0,
        car_follow_model="IDM")
    vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
        initial_speed=0, num_vehicles=0, car_following_params=RL_cfp,
        lane_change_params=None, color="blue")
    vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
        initial_speed=initial_speed, num_vehicles=0, car_following_params=NonRL_cfp,
        lane_change_params=None, color="red")

    sim_params = SumoParams(
        port=None, sim_step=sim_step, lateral_resolution=None,
        no_step_log=True, render=args.render, save_render=False,
        sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
        overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
        teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False)

    env_params = EnvParams(
        additional_params={"max_accel": max_accel, "max_decel": max_decel,
                           "target_velocity": max_speed, "sort_vehicles": False},
        horizon=horizon, warmup_steps=warmup_steps,
        sims_per_step=1, evaluate=False, clip_actions=True)

    initial_config = InitialConfig(shuffle=False, spacing="uniform", min_gap=12,
        perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
        edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"])

    print(f"\n--- EVALUATION START ---")
    print(f"Version: {version}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sims per scenario: {n_sims}\n")

    for scen_key, scen_net_file in scenarios.items():
        for int_key, int_class in intentions.items():
            for rate_key, rate_list in traffic_rates.items():
                group_name = f"{scen_key}_{int_key}_{rate_key}_{version}"
                csv_path = os.path.join(output_dir, f"{group_name}.csv")

                with open(csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

                print(f"\n>>> {group_name} ({n_sims} runs)")

                for run_idx in range(n_sims):
                    current_flow = random.choice(rate_list)

                    _net_file = os.path.join(root_dir, "networks", scen_net_file)
                    _net_params = NetParams(osm_path=None, template=_net_file,
                                           inflows=_build_inflows(current_flow))

                    flow_params = dict(
                        exp_tag="eval", network=int_class, simulator="traci",
                        sim=sim_params, env=env_params, net=_net_params,
                        veh=vehicles, initial=initial_config,
                    )

                    def _make_env(env_config, fp=flow_params, EC=EnvClass):
                        p = fp
                        _v = deepcopy(p["veh"]); _n = p["net"]; _s = deepcopy(p["sim"])
                        _s.render = env_config.get("render", False)
                        net = p["network"](name="eval", vehicles=_v, net_params=_n,
                            initial_config=p.get("initial", InitialConfig()),
                            traffic_lights=p.get("tls", TrafficLightParams()))
                        return EC(env_params=p["env"], sim_params=_s, network=net, simulator=p["simulator"])

                    register_env(env_name, _make_env)

                    cfg = (PPOConfig()
                        .environment(env=env_name, env_config={"render": args.render}, disable_env_checking=True)
                        .framework("torch").
                        rl_module(_enable_rl_module_api=False)
                        .training(_enable_learner_api=False)
                        .rollouts(num_rollout_workers=0)
                        .resources(num_gpus=0))

                    if "attention" in version:
                        cfg = cfg.training( _enable_learner_api=False,  model={
                            "custom_model": "attention_policy",
                            "custom_model_config": {
                                "ego_features": 2, "neighbor_features": 3,
                                "max_neighbors": 5, "embed_dim": 64,
                                "num_heads": 4, "mlp_hidden": 256}})

                    # ── run with guaranteed teardown ──────────────────────────
                    algo = None
                    env  = None
                    row  = None
                    try:
                        algo = cfg.build()
                        algo.restore(checkpoint_path)
                        env = algo.workers.local_worker().env

                        obs, _ = env.reset()
                        done = False
                        step_num = 0
                        while not done:
                             action = algo.compute_single_action(obs, explore=False)
                             obs, reward, terminated, truncated, info = env.step(action)
                             done = terminated or truncated
                             step_num += 1
                            
                             neighbors_info = info.get("neighbors", [])
                             print_neighbor_table(step_num, obs, reward, neighbors_info, terminated, truncated)
                        telemetry = info.get("telemetry", {})
                        row = {
                            "run":          run_idx,
                            "collision":    1 if telemetry.get("agent_collision", False) else 0,
                            "success":      1 if telemetry.get("agent_success",   False) else 0,
                            "avg_speed":    f"{telemetry.get('agent_avg_speed',    0.0):.4f}",
                            "travel_time":  f"{telemetry.get('agent_travel_time',  0.0):.4f}",
                            "waiting_time": f"{telemetry.get('agent_waiting_time', 0.0):.4f}",
                        }

                    except Exception as exc:
                        print(f"  Run {run_idx:02d} | ERROR: {exc}")

                    finally:
                        # Always terminate SUMO/traci first, then stop the algo.
                        # Each step is wrapped individually so one failure doesn't
                        # prevent the others from running.
                        try:
                            if env is not None:
                                env.terminate()
                        except Exception:
                            pass

                        try:
                            if algo is not None:
                                algo.stop()
                        except Exception:
                            pass

                        # Drop all references so Python's GC can reclaim memory.
                        del env, algo
                        gc.collect()
                    # ── end of guaranteed teardown ────────────────────────────

                    if row is None:
                        print(f"  Run {run_idx:02d} | SKIPPED (no telemetry — see error above)")
                        continue

                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

                    print(f"  Run {run_idx:02d} | col={row['collision']} suc={row['success']}"
                          f" spd={row['avg_speed']} tt={row['travel_time']}")

                print(f"  [CSV] → {csv_path}")

    print("\n--- EVALUATION COMPLETE ---")
    plot_eval_results(
    output_dir=output_dir,
    version_filter=version,          # e.g. "heuristic_discrete"
    save_path=os.path.join(output_dir, f"results_{version}.png"),
    show=False,                       # set True if you have a display
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
