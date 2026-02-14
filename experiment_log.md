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

## 2026-02-11: Simple Env and Train Sanity Check (v0.1.1)
- **Description Of Exp**: 1 single agent whose task is to learn how to reach goal as soon as possible.
- **Goal**: Test the simplest environment possible and see if learning happens.
- **Next step**: 4 Agents learning to safely cross as soon as possible. 
- **Config**: configs/v0_1_sanity_check.py
- **Git commit**: 48e3438 
- **Tensorboard dir**: tensorboard_logs/flow_ppo_20260212_140003/  
- **Results**: 
  - Collision rate: 
  - Avg throughput: 
  - Training run: 400k steps
- **Notes**: 
    - At first the agent could not even accelerate but after 400k steps the agent could cross the intersection and learned to apply max accelerations
    - The Rewards and actions are applied correctly and reflect in the observations.

## 2026-02-11: MultiAgent Env with shared parameter policy (v0.2.0)
- **Description Of Exp**: 4 agents whose task is to learn how to safely cross the intersection and reach the goal as soon as possible.
- **Goal**: Test if multiple agents can learn a shared policy simultanously.
- **Next steps**: Move from 4 to 8 agents. 
- **Config**: configs/v0_1_sanity_check.py
- **Git commit**: 
- **Tensorboard dir**: tensorboard_logs/flow_ppo_20260215_001651 
- **Results**: 
  - Collision rate: < 0.05
  - Avg throughput: 
  - Training run: 400k steps
- **Notes**:
    - After 1 training run i found that the agents do not learn how to avoid collision rather they found that rushing to crash is the best way to maximise reward based on how the reward functions is designed. I hence modified the reward to penalise crashes more heavily than driving slowly would:
        current reward = +10 success, -50 collision, -1 every time step. it is better to crash after 10 time step(-60 reward) than drive slowly for 61 steps(-61 reward).
        new reward = +100 success, -200 crash, every timestep +(current speed/max speed)
    - With the changes above the agents learned that slowing down is better than speeding into each other which is good but still canot learn how to completely avoid collision. I will modify the reward to give -5 reward for not maintaining safe gap.
    - Even the proximity reward did not result in safe behaviour. From the training curves the possible causes are a broken value function(due to wrong training parameters) and a noisy reward(sparse reward too large compared to dense reward).
    - Changing the reward to more dense and smothing the proximity penalty(providing a gradient for the neural net to learn from) fixed the problem. Now mean collisions are down to 0.05. Agents learned to slow down when necessary. 
    
    
