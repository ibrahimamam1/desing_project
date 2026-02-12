# Experiment Log

## 2026-02-10: Env and Train Sanity Check (v0.1)
- **Goal**: Ensure Enviroment is bug free and training works using multi agent PPO with param sharing
- **Config**: configs/sanity_check.py
- **Git commit**: 
- **Tensorboard dir**: 
- **Results**: 
  - Collision rate: 
  - Avg throughput: 
  - Training time:
- **Notes**: 
    - Env is too hard for the agents to learn. Range of actions preventing collision from start config is too low.
    - Success/Failure is heavily dependent on initial configuration agent cannot learn good and bad actions.
    - Need to try making env easier or good reward shaping.
- **Next steps**: Multi Agent Baseline. PPO with param sharing, unlimited agents. 

