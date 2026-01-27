### Scenarios and their network file names 
scenarios = {
    "fixed_tl": "50m_fixed_tl.net.xml",
    "adaptive_tl": "50m_adaptive_tl.net.xml",
    "rbl_stop": "50m_right_before_left.net.xml",
    "rbl_": "50m_right_before_left.net.xml",
}

### Traffic rates
traffic_rates = {
    "low" : 1000
    "medium": 1200 
    "medium-high": 1500
    "high": 1800
    "very-high": 2200
}

# Traffic rate ratio per lane
traffic_ratio_per_lane = {
    "uniform" : {"N": 0.25, "S": 0.25, "W": 0.25, "E":0.25},
    "NS_heavy": {"N": 0.4, "S": 0.4, "W": 0.1, "E": 0.1},
    "EW_heavy": {"N": 0.1, "S": 0.1, "W": 0.4, "E": 0.4},
}

### Intentions 
intentions = {
    "all_straight":
    "N_W_S_E":
    "N_E_S_E":
    "N_S_"
}


# Run All Possible combinations
for scenario in scenarios:
    for traffic_rate in traffic_rates:
        for traffic_ratio in traffic_ratio_per_lane:
            for intention in intentions:
                run_simulation(scenario, traffic_rate, traffic_ratio, intention)
