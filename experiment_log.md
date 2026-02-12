# Experiment Log

## 2026-02-11: Env and Train Sanity Check (v0.1.0)
- **Goal**: Ensure Enviroment is bug free and training works using multi agent PPO with param sharing
- **Next steps**: Multi Agent Baseline. PPO with param sharing, unlimited agents. 
- **Config**: configs/sanity_check.py
- **Git commit**: ced4818
- **Tensorboard dir**: tensorboard_logs/flow_ppo_20260212_101820/ 
- **Results**: 
  - Collision rate: 
  - Avg throughput: 
  - Training time:
- **Notes**: 
    - Env is too hard for the agents to learn. Range of actions preventing collision from start config is too low.
    - Success/Failure is heavily dependent on initial configuration agent cannot learn good and bad actions.
    - Need to try making env easier or good reward shaping.

## 2026-02-11: Simplest Env and Train Sanity Check (v0.1.1)
- **Description Of Exp**: 1 single agent whose task is to learn how to reach goal as soon as possible.
- **Goal**: Test the simplest environment possible and ensure learning happens.
- **Next steps**: 4 Agents learning to safely cross as soon as possible. 
- **Config**: configs/sanity_check.py
- **Git commit**: 
- **Tensorboard dir**: tensorboard_logs/flow_ppo_20260212_140003/  
- **Results**: 
  - Collision rate: 
  - Avg throughput: 
  - Training run: 400k steps
- **Notes**: 
    - At first the agent could not even accelerate but after 400k steps the agent could cross the intersection and learned to apply max accelerations
    - The Rewards and actions are applied correctly and reflect in the observations.
