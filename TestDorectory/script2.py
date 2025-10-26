import traci
import sumolib
import sys
import os


SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "Sample.sumocfg" 

def run_sumo():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    sumo_cmd = [SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--quit-on-end"]
    traci.start(sumo_cmd)

    # To store vehicle data
    vehicle_entry_time = {}  # {veh_id: enter_time}
    vehicle_exit_time = {}   # {veh_id: exit_time}
    vehicle_data = []        # list of dicts for each completed vehicle

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Get all vehicles currently in the simulation
        current_vehicles = set(traci.vehicle.getIDList())

        # Record entry time for new vehicles
        for veh_id in current_vehicles:
            if veh_id not in vehicle_entry_time:
                vehicle_entry_time[veh_id] = traci.simulation.getTime()

        # Detect vehicles that have left the simulation
        all_known = set(vehicle_entry_time.keys())
        exited_vehicles = all_known - current_vehicles
        for veh_id in exited_vehicles:
            exit_time = traci.simulation.getTime()
            vehicle_exit_time[veh_id] = exit_time

            travel_time = exit_time - vehicle_entry_time[veh_id]
            # Get the last speed before exit (approximation using exit time)
            try:
                speed = traci.vehicle.getSpeed(veh_id)
            except traci.TraCIException:
                speed = 0.0  # if vehicle already removed

            vehicle_data.append({
                "id": veh_id,
                "enter_time": vehicle_entry_time[veh_id],
                "exit_time": exit_time,
                "travel_time": travel_time,
                "speed": speed
            })

            # Remove from active tracking
            del vehicle_entry_time[veh_id]

        step += 1

    traci.close()
    return vehicle_data

# -------------------------------
def analyze_travel_times(vehicle_data):
    if not vehicle_data:
        print("No vehicle data collected.")
        return

    travel_times = [v["travel_time"] for v in vehicle_data]
    avg_tt = sum(travel_times) / len(travel_times)

    print(f"\n--- Travel Time Statistics ---")
    print(f"Total Vehicles Completed: {len(vehicle_data)}") #y-axis
    print(f"Average Travel Time: {avg_tt:.2f} s") #x-axis

if __name__ == "__main__":
    print("Running SUMO simulation...")
    data = run_sumo()
    analyze_travel_times(data)

