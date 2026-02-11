import sys 
import os 
import random
import psutil
import signal

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.uniform_random import UniformRandomNetwork
from networks.asymetric_random import AsymmetricRandomNetwork
from test import run_sim 
from copy import deepcopy
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams, SumoCarFollowingParams

def kill_sumo_processes():
    """Force kill all SUMO processes"""
    for proc in psutil.process_iter(['name']):
        try:
            if 'sumo' in proc.info['name'].lower():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

min_gap = 2.0
max_accel = 2.6
max_decel = 4.5
max_speed = 25
initial_speed = 0
speed_factor = 1.0
speed_dev = 0.0
impatience = 0.0
car_follow_model = "IDM"
sigma = 0
tau = 1.5
period = 0.5
speed_mode = 31

horizon = 1200 # 10 minutes
sim_step = 0.5
number_of_sim_steps_per_RlAction_step = 1


EDGES_DISTRIBUTION = [
        "E#D-X",
        "E#L-X",
        "E#R-X",
        "E#T-X",
    ]

initial_config = InitialConfig(
    shuffle=False,
    spacing="uniform",
    min_gap=12,
    perturbation=0,
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

############################## Sumo Params Configuration  ###############################
teleport_time = 0
sim_params = SumoParams(
    port=None,
    sim_step=sim_step,
    lateral_resolution=None,
    no_step_log=True,
    render=False, 
    save_render=False,
    sight_radius=25,
    show_radius=False,
    pxpm=2, 
    force_color_update=False,
    overtake_right=False,
    seed=42,
    restart_instance=True,
    print_warnings=False,
    teleport_time=teleport_time,
    num_clients=1,  
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
                    car_follow_model="IDM", 
                )

### Scenarios and their network file names 
scenarios = {
    "FTL10": "100m_fixed_tl10.net.xml",
    "FTL20": "100m_fixed_tl20.net.xml",
    "FTL30": "100m_fixed_tl30.net.xml",
    "FTL40": "100m_fixed_tl40.net.xml",
}

### INTENTIONS
intentions = {
    "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
    "uniform_random": UniformRandomNetwork,
    "assymetric_random": AsymmetricRandomNetwork,
}

### Traffic rates
high_rate = 500
medium_rate = 300
low_rate = 150

traffic_rates = {
    "Sc1_All_low" : [{"N": low_rate, "S": low_rate, "W": low_rate, "E": low_rate}],
    "Sc3_All_medium": [{"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}], 
    "Sc2_All_high_3H" : [{"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                      {"N": high_rate, "S": high_rate, "W": high_rate, "E": medium_rate},
                      {"N": high_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
                      {"N": high_rate, "S": medium_rate, "W": high_rate, "E": high_rate},
                      {"N": medium_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                      {"N": high_rate, "S": high_rate, "W": high_rate, "E": low_rate},
                      {"N": high_rate, "S": high_rate, "W": low_rate, "E": high_rate},
                      {"N": high_rate, "S": low_rate, "W": high_rate, "E": high_rate},
                      {"N": low_rate, "S": high_rate, "W": high_rate, "E": high_rate},
                    ],
    "Sc4_Mixed_2H": [ 
                    {"N": high_rate, "S": high_rate, "W": low_rate, "E": low_rate},
                    {"N": low_rate, "S": low_rate, "W": high_rate, "E": high_rate},
                    {"N": high_rate, "S": low_rate, "W": high_rate, "E": low_rate},
                    {"N": low_rate, "S": high_rate, "W": low_rate, "E": high_rate},
                    {"N": high_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
                    {"N": high_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
                    {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": high_rate},
                    {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
                ],
    "Sc5_Mixed_1H": [ 
                    {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
                    {"N": medium_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
                    {"N": medium_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
                    {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
                    {"N": high_rate, "S": low_rate, "W": low_rate, "E": low_rate},
                    {"N": low_rate, "S": high_rate, "W": low_rate, "E": low_rate},
                    {"N": low_rate, "S": low_rate, "W": high_rate, "E": low_rate},
                    {"N": low_rate, "S": low_rate, "W": low_rate, "E": high_rate},
                ],
  "Sc6_Mixed_ML": [ 
                    {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
                    {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
                    {"N": medium_rate, "S": low_rate, "W": low_rate, "E": medium_rate},
                    {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
                    {"N": low_rate, "S": low_rate, "W": medium_rate, "E": medium_rate},
                    {"N": medium_rate, "S": low_rate, "W": low_rate, "E": low_rate},
                ],
}
#run_sim(
#                   'test',       # Name for output file 
#                    '100m_right_before_left.net.xml',         # The .net.xml file
#                    UniformRandomNetwork,        # The intention Network class 
#                    {"N": low_rate, "S": low_rate, "W": low_rate, "E": low_rate}, 
#                    deepcopy(initial_config),
#                    deepcopy(car_follow_params),
#                    deepcopy(sim_params),
#                    deepcopy(env_params)
#                )
#

n_sims_per_scenario = 42 #Total of 7 hours per scenario 
for scen_key, net_file in scenarios.items():
    for int_key, int_class in intentions.items():
        for rate_key, rate_list in traffic_rates.items():
           
            group_name = f"{scen_key}_{int_key}_{rate_key}"
            for i in range(n_sims_per_scenario):
               # 1. Randomly pick one flow configuration from the list
                current_flow = random.choice(rate_list)
                print(f"--- Starting: {group_name} Run {i} ---")
               
                run_sim(
                    group_name,       # Name for output file 
                    net_file,         # The .net.xml file
                    int_class,        # The intention Network class 
                    current_flow,      # The flow dictionary
                    deepcopy(initial_config),
                    deepcopy(car_follow_params),
                    deepcopy(sim_params),
                    deepcopy(env_params)
                )
                kill_sumo_processes()                 
                print(f"--- Finished Run {i} ---")

from plot2 import main
main()
