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
    #"all_straight":      "routes_all_straight.rou.xml",
    #"all_left":          "routes_all_left.rou.xml",
    #"uniform_random":    "routes_uniform_random.rou.xml",
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
N_SIMS       = 42    # runs per scenario x intention x traffic-rate group

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

# Summary CSV — one row per run (unchanged)
CSV_HEADER = [
    "run",
    "n_vehicles",
    "travel_time_min", "travel_time_avg", "travel_time_max",
    "waiting_time_min", "waiting_time_avg", "waiting_time_max",
]

# Per-vehicle CSV — one row per completed vehicle trip, all runs appended
VEHICLE_CSV_HEADER = [
    "run",
    "vehicle_id",
    "travel_time",   # tripinfo 'duration'   attribute (seconds)
    "waiting_time",  # tripinfo 'waitingTime' attribute (seconds)
    "time_loss",     # tripinfo 'timeLoss'    attribute (seconds)
    "depart",        # departure time (seconds into simulation)
    "arrival",       # arrival  time (seconds into simulation)
    "route_length",  # routeLength (metres)
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
# Parse tripinfo XML
#   Returns:
#     summary  (dict)  — min/avg/max aggregates for the summary CSV
#     vehicles (list)  — one dict per completed trip for the vehicle CSV
# ---------------------------------------------------------------------------
def parse_tripinfo(tripinfo_path: Path, run_idx: int) -> tuple[dict, list]:
    """
    Reads a SUMO tripinfo XML.

    Summary dict keys match CSV_HEADER metric columns.
    Vehicle list contains one dict per <tripinfo> element with keys
    matching VEHICLE_CSV_HEADER.

    Only vehicles that actually *arrived* (arrival >= 0) are included;
    vehicles still in the network at simulation end are excluded unless
    --tripinfo-output.write-unfinished was set.
    """
    empty_summary = {
        "n_vehicles":       0,
        "travel_time_min":  None, "travel_time_avg":  None, "travel_time_max":  None,
        "waiting_time_min": None, "waiting_time_avg": None, "waiting_time_max": None,
    }

    if not tripinfo_path.exists():
        return empty_summary, []

    vehicle_rows = []
    durations    = []
    waiting      = []

    for trip in ET.parse(tripinfo_path).getroot().iter("tripinfo"):
        arrival = float(trip.get("arrival", -1))
        if arrival < 0:
            # Vehicle did not finish — skip for arrived-only statistics
            continue

        dur  = float(trip.get("duration",    0))
        wait = float(trip.get("waitingTime", 0))
        loss = float(trip.get("timeLoss",    0))
        dep  = float(trip.get("depart",      0))
        rlen = float(trip.get("routeLength", 0))
        vid  = trip.get("id", "")

        durations.append(dur)
        waiting.append(wait)

        vehicle_rows.append({
            "run":          run_idx,
            "vehicle_id":   vid,
            "travel_time":  dur,
            "waiting_time": wait,
            "time_loss":    loss,
            "depart":       dep,
            "arrival":      arrival,
            "route_length": rlen,
        })

    def stats(values):
        if not values:
            return None, None, None
        return min(values), sum(values) / len(values), max(values)

    tt_min, tt_avg, tt_max = stats(durations)
    wt_min, wt_avg, wt_max = stats(waiting)

    summary = {
        "n_vehicles":       len(durations),
        "travel_time_min":  tt_min,
        "travel_time_avg":  tt_avg,
        "travel_time_max":  tt_max,
        "waiting_time_min": wt_min,
        "waiting_time_avg": wt_avg,
        "waiting_time_max": wt_max,
    }

    return summary, vehicle_rows


# ---------------------------------------------------------------------------
# Run a single SUMO simulation
# ---------------------------------------------------------------------------
def run_sumo(cfg_path: Path, group_name: str, run_idx: int) -> None:
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

                # ----------------------------------------------------------
                # Open both CSVs once; append rows incrementally
                # ----------------------------------------------------------
                summary_csv_path = OUTPUT_DIR / f"{group_name}.csv"
                vehicle_csv_path = OUTPUT_DIR / f"{group_name}_vehicles.csv"

                summary_file = summary_csv_path.open("w", newline="", encoding="utf-8")
                vehicle_file = vehicle_csv_path.open("w", newline="", encoding="utf-8")

                summary_writer = csv.DictWriter(summary_file, fieldnames=CSV_HEADER)
                vehicle_writer = csv.DictWriter(vehicle_file, fieldnames=VEHICLE_CSV_HEADER)

                summary_writer.writeheader()
                vehicle_writer.writeheader()

                for i in range(N_SIMS):
                    current_flow = random.choice(rate_list)
                    print(f"--- Starting: {group_name}  Run {i:02d}  "
                          f"flow={current_flow} ---")

                    # Build modified route file
                    tmp_rou = TMP_DIR / f"{group_name}_run{i}.rou.xml"
                    build_route_file(base_rou_path, current_flow, tmp_rou)

                    # Each run gets its own tripinfo file so per-vehicle data
                    # is never overwritten before it has been parsed.
                    tripinfo_out = TMP_DIR / f"{group_name}_run{i}_tripinfo.xml"

                    # Build sumocfg
                    tmp_cfg = TMP_DIR / f"{group_name}_run{i}.sumocfg"
                    build_sumocfg(net_path, tmp_rou, tripinfo_out, tmp_cfg)

                    # Run SUMO
                    run_sumo(tmp_cfg, group_name, i)

                    # Parse tripinfo — get both summary and per-vehicle data
                    metrics, vehicle_rows = parse_tripinfo(tripinfo_out, run_idx=i)

                    # Write summary row
                    summary_writer.writerow({"run": i, **metrics})
                    summary_file.flush()

                    # Append all per-vehicle rows for this run
                    vehicle_writer.writerows(vehicle_rows)
                    vehicle_file.flush()

                    # Console summary
                    def fmt(v):
                        return f"{v:.2f}s" if v is not None else "N/A"

                    print(
                        f"    vehicles    : {metrics['n_vehicles']}  "
                        f"(wrote {len(vehicle_rows)} rows to vehicle CSV)"
                    )
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

                summary_file.close()
                vehicle_file.close()
                print(f"[CSV] Summary : {summary_csv_path}")
                print(f"[CSV] Vehicles: {vehicle_csv_path}\n")


if __name__ == "__main__":
    main()
