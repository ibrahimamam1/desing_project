import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.uniform_random import UniformRandomNetwork
from networks.asymetric_random import AsymmetricRandomNetwork
from test import run_sim 

### Scenarios and their network file names 
scenarios = {
    "fixed_tl": "50m_fixed_tl.net.xml",
    "adaptive_tl": "50m_adaptive_tl.net.xml",
    "rbl_stop": "50m_right_before_left.net.xml",
    "rbl_": "50m_right_before_left.net.xml",
}

### INTENTIONS
intentions = {
    "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
    "uniform_random": UniformRandomNetwork,
    "assymetric_random": AsymmetricRandomNetwork,
}

### Traffic rates
high_rate = 200
medium_rate = 300
low_rate = 400

traffic_rates = {
    "4H" : {"N": high_rate, "S": high_rate, "W": high_rate, "E": high_rate},
    "4M": {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate}, 
    "2H_2M": {"N": high_rate, "S": high_rate, "W": medium_rate, "E": medium_rate},
    "2H_2M_Cross": {"N": high_rate, "S": medium_rate, "W": high_rate, "E": medium_rate},
    "1H_3M": {"N": high_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
    "2M_2L": {"N": medium_rate, "S": medium_rate, "W": low_rate, "E": low_rate},
    "2M_2L_Cross": {"N": medium_rate, "S": low_rate, "W": medium_rate, "E": low_rate},
}



# Run All Possible combinations
for scenario in scenarios:
    for intention in intentions:
            print(f"--- Starting: {scenario} ---")
            run_sim(
                scenario, 
                scenarios[scenario],
                intention,
                intentions[intention],
            )
            print(f"--- Finished: {scenario} ---")
