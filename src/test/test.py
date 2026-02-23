import csv
import os
import sys
import random
from copy import deepcopy
import gc 

from flow.core.params import InFlows, VehicleParams, NetParams
from flow.controllers import IDMController
from flow.utils.registry import make_create_env

# Adjust imports based on your file structure
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.alpha_env_v01 import AlphaEnv_v01

def run_sim(
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
    net_file_dir = os.path.join(root_dir, "networks")
    net_file = os.path.join(net_file_dir, net_file_name)
    
    # CHANGE 1: Update filename to indicate per-vehicle data
    telemetry_result_file = scenario_name + '.csv'
    
    # We check if file exists so we don't overwrite headers if running multiple batches
    if not os.path.exists(telemetry_result_file):
        with open(telemetry_result_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "vehicle_id",
                "travel_time",
                "zone_time",
                "total_collisions_in_episode",
            ]
            writer.writerow(header)
    
    myTag = "Benchmark Experiment"
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="NonRL",
        acceleration_controller=(IDMController, {}),
        car_following_params=car_follow_params,
        num_vehicles=0
    )
        
    inflow = InFlows()
    initial_speed = 0
    
    # Add randomness to flow rates (±40)
    random_flow_n = flow_dist["N"] + random.uniform(-40, 40)
    random_flow_s = flow_dist["S"] + random.uniform(-40, 40)
    random_flow_w = flow_dist["W"] + random.uniform(-40, 40)
    random_flow_e = flow_dist["E"] + random.uniform(-40, 40)
    
    print(f"Randomized flow rates - N: {random_flow_n:.1f}, S: {random_flow_s:.1f}, W: {random_flow_w:.1f}, E: {random_flow_e:.1f}")
    
    inflow.add(veh_type="NonRL", edge="E#T-X", probability=random_flow_n / 3600.0, depart_lane="free", depart_speed=initial_speed, begin=1, end=3600)
    inflow.add(veh_type="NonRL", edge="E#R-X", probability=random_flow_s / 3600.0, depart_lane="free", depart_speed=initial_speed, begin=1, end=3600)
    inflow.add(veh_type="NonRL", edge="E#D-X", probability=random_flow_w / 3600.0, depart_lane="free", depart_speed=initial_speed, begin=1, end=3600)
    inflow.add(veh_type="NonRL", edge="E#L-X", probability=random_flow_e / 3600.0, depart_lane="free", depart_speed=initial_speed, begin=1, end=3600)
        
    net_params = NetParams(
        inflows=inflow,
        osm_path=None,
        template=net_file,
    )
        
    flow_params = dict(
        exp_tag=myTag,
        env_name=AlphaEnv_v01,
        network=network,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )
        
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    try:
        env = create_env()
    except Exception as e:
        print(f"Direct call failed: {e}")
        env = create_env(flow_params) # Fallback
    
    sim_complete = False 
    max_attempts = 20 
    attempts = 0
    
    while not sim_complete and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset()
        done = False 
        
        while not done:
            obs, reward, done, trunc, info = env.step([])
        
        # Check completion
        if "__common__" in info and "telemetry" in info["__common__"]:
            telemetry = info["__common__"]["telemetry"]
            episode_duration = telemetry["episode_duration"]
            
            duration_ratio = episode_duration / env_params.horizon
            
            # If successful episode
            if duration_ratio >= 0.99: 
                print(f"Episode completed successfully. Duration: {episode_duration:.2f}s")
                
                # --- Process Per-Vehicle Data ---
                travel_times = telemetry.get("per_vehicle_travel_times", {})
                zone_times = telemetry.get("per_vehicle_zone_times", {})
                collisions = telemetry.get("number_of_collisions", 0)
                
                rows_to_write = []
                
                # We iterate through successful vehicles (those in travel_times)
                for veh_id, t_time in travel_times.items():
                    z_time = zone_times.get(veh_id, 0.0)
                    
                    row = [
                        veh_id,
                        t_time,
                        z_time,
                        collisions,        # Same value for all cars in this run
                    ]
                    rows_to_write.append(row)
                
                # Write all rows for this episode
                try:
                    with open(telemetry_result_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_write)
                        
                    print(f"Telemetry for {len(rows_to_write)} vehicles written to {telemetry_result_file}")
                    sim_complete = True
                except Exception as e:
                    print(f"Error writing telemetry: {e}")
                    sim_complete = True 
            else:
                print(f"Episode ended prematurely. Duration: {episode_duration:.2f}s. Collisions: {telemetry['number_of_collisions']}. Retrying...")
        else:
            print("Warning: No telemetry data found in info dict. Retrying...")
    
    if attempts >= max_attempts:
        print(f"Warning: Maximum attempts ({max_attempts}) reached.")
    
    env.close()
    
    env = None 
    gc.collect()
