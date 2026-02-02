from copy import deepcopy
from flow.core.params import InFlows
import os
import sys
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.controllers import IDMController  # for NON-RL controlled Vehicles
import random
import csv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alpha_env import AlphaEnv


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
  
    telemetry_result_file = scenario_name + '.csv'
    
    # Create CSV file with header if it doesn't exist
    if not os.path.exists(telemetry_result_file):
        with open(telemetry_result_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "episode_duration", 
                "number_of_vehicles_left_successfully",
                "number_of_collisions",
                "min_travel_time", 
                "max_travel_time", 
                "avg_travel_time", 
                "total_travel_time",             
                "min_time_in_control_zone", 
                "max_time_in_control_zone", 
                "avg_time_in_control_zone", 
                "total_time_in_control_zone"
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
    
    # Add randomness to flow rates (±40 from specified value)
    random_flow_n = flow_dist["N"] + random.uniform(-40, 40)
    random_flow_s = flow_dist["S"] + random.uniform(-40, 40)
    random_flow_w = flow_dist["W"] + random.uniform(-40, 40)
    random_flow_e = flow_dist["E"] + random.uniform(-40, 40)
    
    print(f"Randomized flow rates - N: {random_flow_n:.1f}, S: {random_flow_s:.1f}, W: {random_flow_w:.1f}, E: {random_flow_e:.1f}")
    
    inflow.add(
                veh_type="NonRL",
                edge="E#T-X",
                probability=random_flow_n / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#R-X",
                probability=random_flow_s / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#D-X",
                probability=random_flow_w / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#L-X",
                probability=random_flow_e / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
        
    net_params = NetParams(
                inflows=inflow,
                osm_path=None,
                template=net_file,
            )
        
    flow_params = dict(
                exp_tag=myTag,
                env_name=AlphaEnv,
                network=network,
                simulator='traci',
                sim=sim_params,
                env=env_params,
                net=net_params,
                veh=vehicles,
                initial=initial_config,
            )
        
        
    # Run experiment
    from flow.utils.registry import make_create_env
    from flow.utils.rllib import FlowParamsEncoder
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    try:
        env = create_env()
        print(f"Environment created successfully: {type(env)}")
    except Exception as e:
        print(f"Direct call failed: {e}")
        try:
            env = create_env(flow_params)
            print(f"Environment created with params: {type(env)}")
        except Exception as e2:
            print(f"Call with params failed: {e2}")
            print(f"create_env type: {type(create_env)}")
            print(f"create_env: {create_env}")
            raise Exception("Could not create environment")
    
    # Calculate expected episode duration based on horizon
    expected_duration = env_params.sims_per_step * (env_params.warmup_steps + env_params.horizon) * sim_params.sim_step
    
    sim_complete = False 
    max_attempts = 100  # Prevent infinite loops
    attempts = 0
    
    while not sim_complete and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset()
        done = False 
        
        while not done:
            obs, reward, done, trunc, info = env.step([])
        
        # Check if episode completed successfully (ran for full horizon)
        # The episode is complete if it reached the expected duration
        if "__common__" in info and "telemetry" in info["__common__"]:
            telemetry = info["__common__"]["telemetry"]
            episode_duration = telemetry["episode_duration"]
            
            duration_ratio = episode_duration / expected_duration
            if duration_ratio >= 0.99:  # 99% of expected duration (accounts for floating point precision)
                # Episode completed successfully - write to CSV
                print(f"Episode completed successfully. Duration: {episode_duration:.2f}s (expected: {expected_duration:.2f}s)")
                
                row = [
                    telemetry["episode_duration"],
                    telemetry["number_of_vehicles_left_successfully"],
                    telemetry["number_of_collisions"],
                    telemetry["min_travel_time"],
                    telemetry["max_travel_time"],
                    telemetry["avg_travel_time"],
                    telemetry["total_travel_time"],
                    telemetry["min_time_in_control_zone"],
                    telemetry["max_time_in_control_zone"],
                    telemetry["avg_time_in_control_zone"],
                    telemetry["total_time_in_control_zone"]
                ]
                
                try:
                    with open(telemetry_result_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Telemetry written to {telemetry_result_file}")
                    sim_complete = True
                except Exception as e:
                    print(f"Error writing telemetry: {e}")
                    sim_complete = True  # Don't retry on file write errors
            else:
                # Episode ended prematurely (likely due to crash)
                print(f"Episode ended prematurely. Duration: {episode_duration:.2f}s (expected: {expected_duration:.2f}s). Retrying...")
                print(f"Collisions: {telemetry['number_of_collisions']}")
        else:
            # No telemetry in info (shouldn't happen, but handle gracefully)
            print("Warning: No telemetry data found in info dict. Retrying...")
    
    if attempts >= max_attempts:
        print(f"Warning: Maximum attempts ({max_attempts}) reached. Could not complete a full episode.")
    
    # Clean up
    env.close()
