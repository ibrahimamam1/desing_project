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

 ## 2026-02-11: MultiAgent agent parameter sharing (v0.2)
- **Goal**: Train 2, 4, 8 then all_vehicles as agents using PPO with parameter sharing
- **Next steps**: Add the attention mecanism as collision detection and input to the PPO Network
- **Config**: configs/v0_2_multi_agent.py
- **Env**: envs/alpha_env_v02.py 
- **Git commit**: 5ee3ebd
- **Tensorboard dir**: tensorboard_logs/v0_2/ 
- **Results**: 
  - Collision rate: 
  - Avg Speed: 
  - Avg Reward:  
  - Training time: 

  ## 2026-03-04: Multi Agent Attention Mecanism + parameter sharing (v0.3)
  - **Goal**: Train an attention net for conflict detection + parameter sharing
  - **Next Steps**: MAPPO with centralized critic
  - **Config**: configs/v0_3_attention_plus_param_sharing.py 
  - **Git commit**:
  - **Tensorboard dir**: tensorboard_logs/v0_3/
  - **Results**:
    - Collision rate:
    - Avg Speed:
    - Avg Reward: 
    - Training Time:

  ## 2026-03-05: MAPPO with centralized critic (v0.4)
  - **Goal**: Train multiple agents in CTDE setting
  - **Next Steps**: MAPPO with centralized critic plus attention mecanism
  - **Config**: configs/v0_4_mappo.py 
  - **Git commit**:
  - **Tensorboard dir**: tensorboard_logs/v0_4/
  - **Results**:
    - Collision rate:
    - Avg Speed:
    - Avg Reward: 
    - Training Time:

  ## 2026-03-05: MAPPO with centralized critic (v0.5)
  - **Goal**: Train multiple agents in CTDE setting
  - **Next Steps**: MAPPO with centralized critic plus attention mecanism
  - **Config**: configs/v0_5_mappo.py 
  - **Git commit**:
  - **Tensorboard dir**: tensorboard_logs/v0_5/
  - **Results**:
    - Collision rate:
    - Avg Speed:
    - Avg Reward: 
    - Training Time:


## 2026-09-14: Shielded PPO extension (v0.1 + shield)
- **Goal**: Post-decision safety shield on the attention PPO controller (pre-defence report, section 11.2)
- **Env**: src/envs/alpha_env_v01_shielded_attention_continous.py (TTC, RSS, right-of-way layers)
- **Pipeline**: src/full_pipeline.py; report in docs/RESULTS.md, tables in docs/results_tables.md
- **Evaluation**: pre-defence protocol, 6 scenarios x 4 intentions x 42 episodes
- **Results** (standard benchmark, same weights with shield off / on):
  - Collision rate: 0.99% -> 0.40% (p = 0.18, not significant)
  - Travel time: 12.9 s -> 16.3 s
  - Success rate: 99.0% -> 97.0%
  - Actions overridden: 18.5% (RSS about 91% of overrides)
- **Also**: attention vs heuristic confirmed (p < 0.001); shield harms a discrete policy (4 -> 14, p = 0.03)
- **Next steps**: results in the pre-defence training environment (run_reproduction.sh), shield-aware training
