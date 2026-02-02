from copy import deepcopy
from flow.core.params import InFlows
import os
import sys

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.controllers import IDMController  # for NON-RL controlled Vehicles

import random
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
    inflow.add(
                veh_type="NonRL",
                edge="E#T-X",
                probability=flow_dist["N"] / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#R-X",
                probability=flow_dist["S"] / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#D-X",
                probability=flow_dist["W"] / 3600.0,
                depart_lane="free",
                depart_speed=initial_speed,
                begin=1,
                end=3600,
            )
    inflow.add(
                veh_type="NonRL",
                edge="E#L-X",
                probability=flow_dist["E"] / 3600.0,
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
                telemetry=telemetry_result_file,
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

    obs, info = env.reset()
    done = False 

    while not done:
        obs,reward,done,_,_ = env.step([])
