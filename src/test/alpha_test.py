import csv
import os
import sys
import random
from copy import deepcopy
import gc

from flow.core.params import InFlows, VehicleParams, NetParams
from flow.controllers import IDMController
from flow.utils.registry import make_create_env

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flow.envs.alpha_env import AlphaEnv

# ---------------------------------------------------------------------------
# Shared CSV header — must match run_sims.py exactly
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "run",
    "n_vehicles",
    "travel_time_min", "travel_time_avg", "travel_time_max",
    "waiting_time_min", "waiting_time_avg", "waiting_time_max",
]


def _stats(values: list):
    """Return (min, avg, max) for a non-empty list, else (None, None, None)."""
    if not values:
        return None, None, None
    return min(values), sum(values) / len(values), max(values)


def run_sim(
        run_idx,
        scenario_name,
        net_file_name,
        network,
        flow_dist,
        initial_config,
        car_follow_params,
        sim_params,
        env_params,
):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net_file = os.path.join(root_dir, "networks", net_file_name)

    # One CSV per group (scenario_name is the group key)
    csv_path = os.path.join(root_dir, "output", scenario_name, ".csv")

    # Write header only if the file is new
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="NonRL",
        acceleration_controller=(IDMController, {}),
        car_following_params=car_follow_params,
        num_vehicles=0,
    )

    # Add ±40 vph randomness to each entry flow
    inflow = InFlows()
    for edge, cardinal in [("E#T-X", "N"), ("E#R-X", "S"),
                            ("E#D-X", "W"), ("E#L-X", "E")]:
        rate = max(0, flow_dist[cardinal] + random.uniform(-40, 40))
        inflow.add(
            veh_type="NonRL",
            edge=edge,
            probability=rate / 3600.0,
            depart_lane="free",
            depart_speed=0,
            begin=1,
            end=3600,
        )

    net_params = NetParams(
        inflows=inflow,
        osm_path=None,
        template=net_file,
    )

    flow_params = dict(
        exp_tag="Benchmark Experiment",
        env_name=AlphaEnv,
        network=network,
        simulator="traci",
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )

    create_env, _ = make_create_env(params=flow_params, version=0)
    try:
        env = create_env()
    except Exception as e:
        print(f"Direct call failed: {e}")
        env = create_env(flow_params)

    sim_complete = False
    max_attempts = 20

    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, trunc, info = env.step([])

        # ----------------------------------------------------------------
        # Validate episode completed fully
        # ----------------------------------------------------------------
        if "__common__" not in info or "telemetry" not in info["__common__"]:
            print(f"  [attempt {attempt}] No telemetry — retrying...")
            continue

        telemetry       = info["__common__"]["telemetry"]
        episode_dur     = telemetry["episode_duration"]
        duration_ratio  = episode_dur / env_params.horizon

        if duration_ratio < 0.99:
            print(f"  [attempt {attempt}] Premature end "
                  f"({episode_dur:.1f}s, {telemetry['number_of_collisions']} collisions) — retrying...")
            continue

        # ----------------------------------------------------------------
        # Summarise per-vehicle data into min / avg / max
        # ----------------------------------------------------------------
        travel_times  = telemetry.get("per_vehicle_travel_times",  {})
        waiting_times = telemetry.get("per_vehicle_waiting_times", {})

        # Use waiting_times keys as the master vehicle set (includes all vehicles)
        all_veh_ids = list(waiting_times.keys())
        n_vehicles  = len(all_veh_ids)

        # Only include numeric travel times (exclude "N/A" for unfinished trips)
        tt_values = [v for v in travel_times.values() if isinstance(v, (int, float))]
        wt_values = [waiting_times.get(vid, 0.0) for vid in all_veh_ids]

        tt_min, tt_avg, tt_max = _stats(tt_values)
        wt_min, wt_avg, wt_max = _stats(wt_values)

        row = {
            "run":              run_idx,
            "n_vehicles":       n_vehicles,
            "travel_time_min":  tt_min,
            "travel_time_avg":  tt_avg,
            "travel_time_max":  tt_max,
            "waiting_time_min": wt_min,
            "waiting_time_avg": wt_avg,
            "waiting_time_max": wt_max,
        }

        # ----------------------------------------------------------------
        # Append single summary row to group CSV
        # ----------------------------------------------------------------
        try:
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writerow(row)
            print(
                f"  Run {run_idx:02d} | n={n_vehicles} | "
                f"travel: {tt_min:.1f}/{tt_avg:.1f}/{tt_max:.1f}s | "
                f"wait: {wt_min:.1f}/{wt_avg:.1f}/{wt_max:.1f}s"
            )
            sim_complete = True
        except Exception as e:
            print(f"  Error writing CSV: {e}")
            sim_complete = True  # don't retry on write error

        break  # exit attempt loop on success or write error

    if not sim_complete:
        print(f"  Warning: max attempts ({max_attempts}) reached for run {run_idx}.")

    env.close()
    env = None
    gc.collect()
