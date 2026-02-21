# Experiment Log

## 2026-02-11: Single agent PPO (v0.1)
- **Goal**: Ensure Enviroment is bug free and training works by using PPO to train a single agent
- **Next steps**: Multi Agent Baseline. PPO with param sharing, 2 agents. 
- **Config**: configs/v0_1_single_agent.py
- **Env**: envs/alpha_env_v01.py 
- **Git commit**: ced4818
- **Tensorboard dir**: tensorboard_logs/v0_1/ 
- **Results**: 
  - Collision rate: 
  - Avg Speed:
  - Avg Reward:
  - Training time:
- **Notes**: 
  - Initially : speed_reward = 0.5 * (speed / max_speed). The dense reward was consistently very small caussing the entropy to outweight the reward and the policy optimizer to just continue taking random actions. I changed the reward scale to speed_reward = 2.0 * (speed / max_speed).
  - Experiment Failed. Both SB3 and rllib could not learn a policy that made sense. My guess is the underlying environemnt is flawed. Swithcing to highway env

## 2026-02-21: Highway env PPO (v0.2)
- **Goal**: Setup highway_env Enviroment and Rllib training and ensure it is bug free and training works on a single agent using PPO
- **Next steps**: Multi Agent Baseline. PPO with param sharing, 2 agents. 
- **Config**: configs/v0_2_single_agent.py
- **Env**: envs/alpha_env_v02.py 
- **Git commit**: 
- **Tensorboard dir**: 
- **Results**: 
  - Collision rate: 
  - Avg Speed:
  - Avg Reward:
  - Training time:
- **Notes**: 
   
