# Experiment Log

## 2026-02-11: Single agent PPO (v0.1)
- **Goal**: Ensure Enviroment is bug free and training works by using PPO to train a single agent
- **Next steps**: Multi Agent Baseline. PPO with param sharing, 2 agents. 
- **Config**: configs/v0_1_single_agent.py
- **Env**: envs/alpha_env_v01.py 
- **Git commit**: ced4818
- **Tensorboard dir**: tensorboard_logs/v0_1/ 
- **Results**: 
  - Collision rate: 0.04 
  - Avg Speed: 7.9
  - Avg Reward: 16 
  - Training time: 2M Steps

 ## 2026-02-11: MultiAgent agent Sanity Check (v0.2)
- **Goal**: Train 2, 4, 8 then all_vehicles as agents using PPO with parameter sharing
- **Next steps**: Add the attention mecanism as collision detection and input to the PPO Network
- **Config**: configs/v0_2_multi_agent.py
- **Env**: envs/alpha_env_v02.py 
- **Git commit**: 
- **Tensorboard dir**: tensorboard_logs/v0_2/ 
- **Results**: 
  - Collision rate: 
  - Avg Speed: 
  - Avg Reward:  
  - Training time:  
