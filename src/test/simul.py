import os
import sys
import random

from non_rl_test_env import TestEnv
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import SumoCarFollowingParams
from flow.controllers import IDMController
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(root_dir, "results")
net_file_dir = os.path.join(root_dir, "networks")
    

def run_simulation(scenario, net_file_name, traffic_rate, intentions, initial_config, env_params, sim_params, car_following_params):
    print(f'net_file = {net_file_name}, intention={intentions}')

    expTag = scenario
    net_file_path = os.path.join(net_file_dir, net_file_name)

    max_vehicle_count_in_inflow = 20
    num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)  # 1

    vehicles = VehicleParams()
    vehicles.add(
            veh_id="NonRL",
            acceleration_controller=(IDMController, {}),
            car_following_params= car_following_params,
            num_vehicles=0
        )
    inflow = InFlows()
    inflow.add(
            veh_type="NonRL",
            edge="E#T-X",
            probability= 800 / 3600.0,
            depart_lane="free",
            depart_speed=5,
            begin=1,
            end=3600,
        )
    inflow.add(
            veh_type="NonRL",
            edge="E#R-X",
            probability= 800 / 3600.0,
            depart_lane="free",
            depart_speed=5,
            begin=1,
            end=3600,
        )
    inflow.add(
            veh_type="NonRL",
            edge="E#D-X",
            probability= 800 / 3600.0,
            depart_lane="free",
            depart_speed=5,
            begin=1,
            end=3600,
        )
    inflow.add(
            veh_type="NonRL",
            edge="E#L-X",
            probability= 800 / 3600.0,
            depart_lane="free",
            depart_speed=5,
            begin=1,
            end=3600,
        )
    
    net_params = NetParams(
            inflows=inflow,
            osm_path=None,
            template=net_file_path,
        )

    flow_params = dict(
            exp_tag=expTag,
            env_name=TestEnv,
            network=intentions,
            simulator='traci',
            sim=sim_params,
            env=env_params,
            net=net_params,
            veh=vehicles,
            initial=initial_config,
        )
    
    
    # Run experiment
    from flow.core.experiment_new import Experiment
    exp = Experiment(flow_params)
    _ = exp.run(1, convert_to_csv=True)

