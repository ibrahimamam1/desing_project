import os
import sys

sys.argv = ['test_env_inflow.py', '--train']
sys.path.append(os.path.join(os.getcwd(), 'src', 'configs'))
sys.path.append(os.path.join(os.getcwd(), 'src'))
import v0_1_single_agent
import time

print("Building env")
env = v0_1_single_agent.create_flow_env({"render": False})
env.reset()
print("Env reset.")
for i in range(15):
    obs, reward, terminated, truncated, infos = env.step([0])
    vehicles = env.k.vehicle.get_ids()
    rl_ids = env.k.vehicle.get_rl_ids()
    print(f"Step {i}: rl_ids={rl_ids}, all={vehicles}, terminated={terminated}")
    if terminated:
        print("TERMINATED early")
        break
env.terminate()
