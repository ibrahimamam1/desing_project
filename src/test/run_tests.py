import sys 
import os 
from copy import deepcopy 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.uniform_random import UniformRandomNetwork
from networks.asymetric_random import AsymmetricRandomNetwork
from simul import run_simulation 

### Scenarios and their network file names 
scenarios = {
    "fixed_tl": "50m_fixed_tl.net.xml",
    "adaptive_tl": "50m_adaptive_tl.net.xml",
    "rbl_stop": "50m_right_before_left.net.xml",
    "rbl_": "50m_right_before_left.net.xml",
}

### Traffic rates
high_rate = 1000
medium_rate = 600
low_rate = 200

traffic_rates = {
    "4H" : {"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
    "4M": {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}, 
    "2H_2M": {"N": high_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
    "2H_2M_Cross": {"N": high_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
    "1H_3M": {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
    "2M_2L": {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
    "2M_2L_Cross": {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
}

### Intentions 
intentions = {
    "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
    "uniform_random": UniformRandomNetwork,
    "assymetric_random": AsymmetricRandomNetwork,
}


from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams, SumoCarFollowingParams

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(root_dir, "results")
emmision_dir = os.path.join(results_dir, "emmisions")

### Traffic Parameters
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
speed_mode = "all_checks"  
horizon = 3000
sim_step = 0.5
number_of_sim_steps_per_RlAction_step = 1
teleport_time = 0
EDGES_DISTRIBUTION = ["E#D-X", "E#L-X", "E#R-X", "E#T-X"]

############ Initial Configuration ###########
initial_config = InitialConfig(
        shuffle=False,
        spacing="uniform",  
        min_gap=12,
        perturbation=5.0,
        x0=5,
        bunching=0,
        lanes_distribution=float("inf"),
        edges_distribution=EDGES_DISTRIBUTION,
        additional_params=None
    )

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

car_follow_params = SumoCarFollowingParams(
                min_gap=min_gap,
                max_speed=max_speed,
                speed_mode=speed_mode,
                accel=max_accel,
                decel=max_decel,
                sigma=sigma,
                tau=tau,
            )
# Run All Possible combinations
for scenario in scenarios:
    for traffic_rate in traffic_rates:
        for intention in intentions:
            print(f"--- Starting: {scenario} | {traffic_rate} | {intention} ---")
            
            # CRITICAL FIX: Use deepcopy() here
            run_simulation(
                scenario, 
                scenarios[scenario], 
                traffic_rates[traffic_rate], 
                intentions[intention], 
                deepcopy(initial_config), 
                deepcopy(env_params), 
                deepcopy(sim_params), 
                deepcopy(car_follow_params)
            )
            
            print(f"--- Finished: {scenario} | {traffic_rate} | {intention} ---")
