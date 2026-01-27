from flow.networks.all_turning_intersection import AllTurningIntersectionNetwork as myNet
from flow.core.params import InFlows
import os
import sys

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams, SumoCarFollowingParams

from flow.controllers import IDMController  # for NON-RL controlled Vehicles

import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def run_simulation(scenario, traffic_rate, traffic_ratio, intentions):
    
    ################ NETWORK Description #######################
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(root_dir, "results")
    emmision_dir = os.path.join(results_dir, "emmisions")
    net_file_dir = os.path.join(root_dir, "networks")
    net_file_name = scenario[1]
    net_file = os.path.join(net_file_dir, net_file_name)
    
    ################ Acceleration Controllers #######################
    IDM_acceleration_controller = IDMController

    min_gap = 0.9
    max_accel = 2.6
    max_decel = 4.5
    max_speed = 30
    initial_speed = 0
    speed_factor = 1.0
    speed_dev = 0.0
    impatience = 0.0
    car_follow_model = "IDM"
    sigma = 0
    tau = 0.8
    period = 0.5

    speed_mode = "all_checks"  # 32 = safety check of, 31 = safety check on
    
    ############ Initial Configuration ###########
    EDGES_DISTRIBUTION = [
    "E#D-X",
    "E#L-X",
    "E#R-X",
    "E#T-X",
    ]

    initial_config = InitialConfig(
        shuffle=False,
        spacing="uniform",  # "random",#"uniform",
        min_gap=12,
        perturbation=5.0,
        x0=5,
        bunching=0,
        lanes_distribution=float("inf"),
        edges_distribution=EDGES_DISTRIBUTION,
        additional_params=None
    )

############################## Environment Configuration  ###############################

    myTag = scenario[0]
    horizon = 3000
    sim_step = 0.5
    number_of_sim_steps_per_RlAction_step = 1

    env_params = EnvParams(
        additional_params={
            'max_accel': max_accel,
            'max_decel': max_decel,
            'target_velocity': max_speed,
            'sort_vehicles': False
        },
        horizon=horizon,
        warmup_steps=0,
        sims_per_step=number_of_sim_steps_per_RlAction_step,
        evaluate=False,
        clip_actions=True)

############################## Sumo Params Configuration  ###############################
    teleport_time = 0
    sim_params = SumoParams(
        port=None,
        sim_step=sim_step,
        emission_path=emmision_dir,
        lateral_resolution=None,
        no_step_log=True,
        render=True, 
        save_render=False,
        sight_radius=25,
        show_radius=False,
        pxpm=2,  # specifies rendering resolution (pixel / meter)
        force_color_update=False,
        overtake_right=False,
        seed=42,

        restart_instance=True,
        print_warnings=True,
        teleport_time=teleport_time,

        num_clients=1,  # Number of clients that will connect to Traci
        color_by_speed=False,
        use_ballistic=False, 
    )

######################### TRAFFIC NETWORK ###############################
    max_vehicle_count_in_inflow = 20
    num_inflows_vehicles = random.randint(1, max_vehicle_count_in_inflow)  # 1

    vehicles = VehicleParams()
    vehicles.add(
            veh_id="NonRL",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=min_gap,
                max_speed=max_speed,
                speed_mode=speed_mode,
                accel=max_accel,
                decel=max_decel,
                sigma=sigma,
                tau=tau,
            ),
            num_vehicles=0  # FIXED: Proper indentation
        )
    
    inflow = InFlows()
    
    inflow.add(
            veh_type="NonRL",
            edge="E#T-X",
            probability= (traffic_ratio["N"] * traffic_rate) / 3600.0,
            depart_lane="free",
            depart_speed=initial_speed,
            begin=1,
            end=3600,
        )
    
    inflow.add(
            veh_type="NonRL",
            edge="E#R-X",
            probability= (traffic_ratio["E"] * traffic_rate) / 3600.0,
            depart_lane="free",
            depart_speed=initial_speed,
            begin=1,
            end=3600,
        )
    
    inflow.add(
            veh_type="NonRL",
            edge="E#D-X",
            probability= (traffic_ratio["S"] * traffic_rate) / 3600.0,
            depart_lane="free",
            depart_speed=initial_speed,
            begin=1,
            end=3600,
        )
    
    inflow.add(
            veh_type="NonRL",
            edge="E#L-X",
            probability= (traffic_ratio["W"] * traffic_rate) / 3600.0,
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
###################### FLOW PARAMS ###################### 

    flow_params = dict(
            exp_tag=myTag,
            env_name=TestEnv,
            network=myNet,
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
 
