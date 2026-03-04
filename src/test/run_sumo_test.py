import csv
import random
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent.parent
NETS_DIR   = BASE_DIR / "networks"
ROUTES_DIR = BASE_DIR / "networks"
OUTPUT_DIR = BASE_DIR / "output"
TMP_DIR    = BASE_DIR / "tmp"

for d in (NETS_DIR, ROUTES_DIR, OUTPUT_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenarios  ->  net file names  
# ---------------------------------------------------------------------------
scenarios = {
    "allway_stop": "100m_allway_stop_fcfs_junction.net.xml",
    "fixed_tl":    "100m_fixed_tl_junction.net.xml",
    "rbl":         "100m_right_before_left_junction.net.xml",
}

# ---------------------------------------------------------------------------
# Intentions  ->  base route file names 
# ---------------------------------------------------------------------------
intentions = {
    "all_straight":      "routes_all_straight.rou.xml",
    "all_left":          "routes_all_left.rou.xml",
    "uniform_random":    "routes_uniform_random.rou.xml",
    "assymetric_random": "routes_nonuniform_random.rou.xml",
}

high_rate   = 500
medium_rate = 300
low_rate    = 150

traffic_rates = {
    "Sc1_All_low": [
        {"N": low_rate,    "S": low_rate,    "W": low_rate,    "E": low_rate},
    ],
    "Sc3_All_medium": [
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": medium_rate},
    ],
    "Sc2_All_high_3H": [
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": medium_rate},
        {"N": high_rate,   "S": high_rate,   "W": medium_rate, "E": high_rate},
        {"N": high_rate,   "S": medium_rate, "W": high_rate,   "E": high_rate},
        {"N": medium_rate, "S": high_rate,   "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": high_rate,   "W": high_rate,   "E": low_rate},
        {"N": high_rate,   "S": high_rate,   "W": low_rate,    "E": high_rate},
        {"N": high_rate,   "S": low_rate,    "W": high_rate,   "E": high_rate},
        {"N": low_rate,    "S": high_rate,   "W": high_rate,   "E": high_rate},
    ],
    "Sc4_Mixed_2H": [
        {"N": high_rate,   "S": high_rate,   "W": low_rate,    "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": high_rate,   "E": high_rate},
        {"N": high_rate,   "S": low_rate,    "W": high_rate,   "E": low_rate},
        {"N": low_rate,    "S": high_rate,   "W": low_rate,    "E": high_rate},
        {"N": high_rate,   "S": high_rate,   "W": medium_rate, "E": medium_rate},
        {"N": high_rate,   "S": medium_rate, "W": high_rate,   "E": medium_rate},
        {"N": medium_rate, "S": high_rate,   "W": medium_rate, "E": high_rate},
        {"N": high_rate,   "S": medium_rate, "W": medium_rate, "E": high_rate},
    ],
    "Sc5_Mixed_1H": [
        {"N": high_rate,   "S": medium_rate, "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": high_rate,   "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": high_rate,   "E": medium_rate},
        {"N": medium_rate, "S": medium_rate, "W": medium_rate, "E": high_rate},
        {"N": high_rate,   "S": low_rate,    "W": low_rate,    "E": low_rate},
        {"N": low_rate,    "S": high_rate,   "W": low_rate,    "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": high_rate,   "E": low_rate},
        {"N": low_rate,    "S": low_rate,    "W": low_rate,    "E": high_rate},
    ],
    "Sc6_Mixed_ML": [
        {"N": medium_rate, "S": medium_rate, "W": low_rate,    "E": low_rate},
        {"N": medium_rate, "S": low_rate,    "W": medium_rate, "E": low_rate},
        {"N": medium_rate, "S": low_rate,    "W": low_rate,    "E": medium_rate},
        {"N": low_rate,    "S": low_rate,    "W": medium_rate, "E": medium_rate},
        {"N": low_rate,    "S": low_rate,    "W": medium_rate, "E": medium_rate},
        {"N": medium_rate, "S": low_rate,    "W": low_rate,    "E": low_rate},
    ],
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIM_DURATION = 360   # seconds
N_SIMS       = 42     # runs per scenario x intention x traffic-rate group

EDGE_MAP = {
    "N": "E#T-X",  # North = Top
    "S": "E#D-X",  # South = Down
    "W": "E#L-X",  # West  = Left
    "E": "E#R-X",  # East  = Right
}

FLOW_ID_MAP = {
    "E#T-X": "flow_T",
    "E#D-X": "flow_D",
    "E#L-X": "flow_L",
    "E#R-X": "flow_R",
}

# CSV written incrementally; one file per group
CSV_HEADER = [
    "run",
    "n_vehicles",
    "travel_time_min", "travel_time_avg", "travel_time_max",
    "waiting_time_min", "waiting_time_avg", "waiting_time_max",
]


# ---------------------------------------------------------------------------
# Helper: veh/h  ->  per-second spawn probability
# ---------------------------------------------------------------------------
def vph_to_probability(vph: int) -> float:
    return round(vph / 3600.0, 6)


# ---------------------------------------------------------------------------
# Build a modified .rou.xml with updated flow probabilities
# ---------------------------------------------------------------------------
def build_route_file(base_rou_path: Path, flow_config: dict,
                     out_path: Path) -> None:
    """
    flow_config: {"N": vph, "S": vph, "W": vph, "E": vph}
    Writes a modified copy of base_rou_path to out_path with updated
    probability attributes on each <flow> element.
    """
    tree = ET.parse(base_rou_path)
    root = tree.getroot()

    prob_lookup = {
        FLOW_ID_MAP[EDGE_MAP[cardinal]]: vph_to_probability(vph)
        for cardinal, vph in flow_config.items()
    }

    for flow_elem in root.iter("flow"):
        fid = flow_elem.get("id")
        if fid in prob_lookup:
            flow_elem.set("probability", str(prob_lookup[fid]))
            flow_elem.set("end", str(SIM_DURATION))

    tree.write(out_path, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Build a .sumocfg file
# ---------------------------------------------------------------------------
def build_sumocfg(net_path: Path, rou_path: Path,
                  tripinfo_path: Path,
                  cfg_path: Path) -> None:
    """
    Writes a SUMO config with only tripinfo-output enabled.
    tripinfo provides per-vehicle duration, waitingTime, and timeLoss.
    """
    cfg_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_path.resolve()}"/>
        <route-files value="{rou_path.resolve()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_DURATION}"/>
    </time>
    <o>
        <!-- per-vehicle travel time, waiting time, time loss -->
        <tripinfo-output value="{tripinfo_path.resolve()}"/>
    </o>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    cfg_path.write_text(cfg_xml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Parse tripinfo XML -> min / avg / max for travel time, waiting, time loss
# ---------------------------------------------------------------------------
def parse_tripinfo(tripinfo_path: Path) -> dict:
    """
    Reads a SUMO tripinfo XML and computes min/avg/max for:
      - travel time  ('duration'    attribute, seconds)
      - waiting time ('waitingTime' attribute, seconds)
      - time loss    ('timeLoss'    attribute, seconds)

    Returns a dict whose keys match the CSV_HEADER metric columns.
    Values are None if no completed trips exist.
    """
    if not tripinfo_path.exists():
        return {
            "n_vehicles":       0,
            "travel_time_min":  None, "travel_time_avg":  None, "travel_time_max":  None,
            "waiting_time_min": None, "waiting_time_avg": None, "waiting_time_max": None,
        }

    durations, waiting = [], []

    for trip in ET.parse(tripinfo_path).getroot().iter("tripinfo"):
        durations.append(float(trip.get("duration",    0)))
        waiting.append(  float(trip.get("waitingTime", 0)))

    def stats(values):
        if not values:
            return None, None, None
        return min(values), sum(values) / len(values), max(values)

    tt_min, tt_avg, tt_max = stats(durations)
    wt_min, wt_avg, wt_max = stats(waiting)

    return {
        "n_vehicles":       len(durations),
        "travel_time_min":  tt_min,
        "travel_time_avg":  tt_avg,
        "travel_time_max":  tt_max,
        "waiting_time_min": wt_min,
        "waiting_time_avg": wt_avg,
        "waiting_time_max": wt_max,
    }


# ---------------------------------------------------------------------------
# Run a single SUMO simulation
# ---------------------------------------------------------------------------
def run_sumo(cfg_path: Path, group_name: str, run_idx: int) -> None:
    """
    Launch sumo-gui with the given config.
    Switch "sumo-gui" -> "sumo" for headless batch runs.
    A unique --seed is passed each run so that the stochastic flow
    spawning produces different vehicle patterns every time.
    """
    seed = random.randint(0, 2**31 - 1)
    cmd = [
        "sumo",
        "-c", str(cfg_path.resolve()),
        "--seed", str(seed),
        "--quit-on-end",
        "--no-warnings",
    ]
    print(f"    CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] SUMO exited with code {result.returncode} "
              f"for {group_name} run {run_idx}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    for scen_key, net_file in scenarios.items():
        net_path = NETS_DIR / net_file
        if not net_path.exists():
            print(f"[SKIP] Net file not found: {net_path}")
            continue

        for int_key, rou_file in intentions.items():
            base_rou_path = ROUTES_DIR / rou_file
            if not base_rou_path.exists():
                print(f"[SKIP] Route template not found: {base_rou_path}")
                continue

            for rate_key, rate_list in traffic_rates.items():
                group_name = f"{scen_key}_{int_key}_{rate_key}"

                # One CSV per group, header written once, rows appended each run
                csv_path = OUTPUT_DIR / f"{group_name}.csv"
                csv_file = csv_path.open("w", newline="", encoding="utf-8")
                writer   = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
                writer.writeheader()

                for i in range(N_SIMS):
                    current_flow = random.choice(rate_list)
                    print(f"--- Starting: {group_name}  Run {i:02d}  "
                          f"flow={current_flow} ---")

                    # Build modified route file
                    tmp_rou = TMP_DIR / f"{group_name}_run{i}.rou.xml"
                    build_route_file(base_rou_path, current_flow, tmp_rou)

                    # Single reusable tripinfo file in tmp/ (overwritten each run)
                    tripinfo_out = TMP_DIR / "current_tripinfo.xml"

                    # Build sumocfg
                    tmp_cfg = TMP_DIR / f"{group_name}_run{i}.sumocfg"
                    build_sumocfg(net_path, tmp_rou, tripinfo_out, tmp_cfg)

                    # Run SUMO
                    run_sumo(tmp_cfg, group_name, i)

                    # Parse tripinfo and write CSV row
                    metrics = parse_tripinfo(tripinfo_out)
                    row = {
                        "run": i,
                        **metrics,
                    }
                    writer.writerow(row)
                    csv_file.flush()  # persist immediately in case of crash

                    # Console summary
                    def fmt(v):
                        return f"{v:.2f}s" if v is not None else "N/A"

                    print(
                        f"    travel_time : "
                        f"min={fmt(metrics['travel_time_min'])}  "
                        f"avg={fmt(metrics['travel_time_avg'])}  "
                        f"max={fmt(metrics['travel_time_max'])}"
                    )
                    print(
                        f"    waiting_time: "
                        f"min={fmt(metrics['waiting_time_min'])}  "
                        f"avg={fmt(metrics['waiting_time_avg'])}  "
                        f"max={fmt(metrics['waiting_time_max'])}"
                    )
                    print(f"--- Finished: {group_name}  Run {i:02d} ---\n")

                csv_file.close()
                print(f"[CSV] Written: {csv_path}\n")
if __name__ == "__main__":
    main()
