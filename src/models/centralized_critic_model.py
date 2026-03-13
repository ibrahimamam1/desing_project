"""
CentralizedCriticModel — MAPPO model with separate actor and critic networks.

Architecture (CTDE — Centralized Training, Decentralized Execution):
    Actor:  local obs → MLP → action logits  (decentralized)
    Critic: local obs → separate MLP → value  (centralized — ready for global state upgrade)

The actor and critic do NOT share parameters. This is the key architectural
difference from standard PPO parameter sharing (v0.2), where actor and critic
share a backbone.

NOTE: Currently the critic uses local obs only. To upgrade to true centralized
critic, modify the env to provide global state and update the critic input dim.
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO model with fully separate actor and critic networks.
    
    Expected flat observation: [ego_features + neighbor_features] = 12 (same as v0.2)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})
        obs_dim = int(np.prod(obs_space.shape))
        mlp_hidden = custom_cfg.get("mlp_hidden", 256)

        # --- Actor Network (decentralized — uses local obs only) ---
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_outputs),
        )

        # --- Critic Network (centralized — separate from actor) ---
        # Currently uses local obs. To upgrade to centralized global state,
        # change critic_input_dim to the global state dimension.
        critic_input_dim = obs_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

        self._obs = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        self._obs = obs
        return self.actor(obs), state

    @override(TorchModelV2)
    def value_function(self):
        assert self._obs is not None, "forward() must be called before value_function()"
        return self.critic(self._obs).squeeze(-1)
