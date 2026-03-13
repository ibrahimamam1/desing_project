"""
AttentionCentralizedCriticModel — MAPPO model with attention actor + separate critic.

Architecture (CTDE — Centralized Training, Decentralized Execution):
    Actor:  attention-based (same as AttentionPolicyModel from v0.3)
    Critic: local obs → separate MLP → value (centralized — ready for global state)

Combines the attention mechanism from v0.3 with the separate critic from MAPPO.
The actor uses cross-attention over neighbor features, while the critic uses a 
separate MLP that does NOT share parameters with the actor.
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class AttentionCentralizedCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO model with attention-based actor and separate centralized critic.
    
    Expected flat observation layout:
        [ego_features (2), neighbor_features (max_k * 2), neighbor_mask (max_k)]
    
    Default: 2 + 5*2 + 5 = 17
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})
        
        self.ego_features = custom_cfg.get("ego_features", 2)
        self.neighbor_features = custom_cfg.get("neighbor_features", 2)
        self.max_neighbors = custom_cfg.get("max_neighbors", 5)
        self.embed_dim = custom_cfg.get("embed_dim", 64)
        self.num_heads = custom_cfg.get("num_heads", 4)
        self.mlp_hidden = custom_cfg.get("mlp_hidden", 256)

        obs_dim = int(np.prod(obs_space.shape))

        # Verify obs dimension
        expected_obs_dim = self.ego_features + (self.max_neighbors * self.neighbor_features) + self.max_neighbors
        assert obs_dim == expected_obs_dim, (
            f"Observation dim mismatch: expected {expected_obs_dim}, got {obs_dim}."
        )

        # ─────────────────────────────────────────────
        # ACTOR: Attention-based (decentralized)
        # ─────────────────────────────────────────────
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_features, self.embed_dim),
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_features, self.embed_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

        context_dim = self.embed_dim * 2  # ego_embed + attn_output

        self.actor_mlp = nn.Sequential(
            nn.Linear(context_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, num_outputs),
        )

        # ─────────────────────────────────────────────
        # CRITIC: Separate MLP (centralized)
        # ─────────────────────────────────────────────
        # Uses raw obs directly — NO parameter sharing with actor
        critic_input_dim = obs_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 1),
        )

        self._raw_obs = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        self._raw_obs = obs  # Cache for critic

        # --- 1. Split observation ---
        ego_end = self.ego_features
        neighbor_end = ego_end + (self.max_neighbors * self.neighbor_features)
        
        ego_raw = obs[:, :ego_end]
        neighbor_raw = obs[:, ego_end:neighbor_end]
        mask_raw = obs[:, neighbor_end:]

        neighbor_raw = neighbor_raw.view(-1, self.max_neighbors, self.neighbor_features)

        # --- 2. Encode ---
        ego_embed = self.ego_encoder(ego_raw)
        neighbor_embeds = self.neighbor_encoder(neighbor_raw)

        # --- 3. Build attention mask ---
        key_padding_mask = (mask_raw < 0.5)
        all_masked = key_padding_mask.all(dim=1)

        # --- 4. Cross-Attention ---
        query = ego_embed.unsqueeze(1)
        key = neighbor_embeds
        value = neighbor_embeds

        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked] = False

        attn_output, _ = self.attention(
            query=query, key=key, value=value,
            key_padding_mask=safe_mask,
        )
        attn_output = attn_output.squeeze(1)
        attn_output[all_masked] = 0.0

        # --- 5. Actor output ---
        context = torch.cat([ego_embed, attn_output], dim=-1)
        actor_out = self.actor_mlp(context)

        return actor_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._raw_obs is not None, "forward() must be called before value_function()"
        return self.critic(self._raw_obs).squeeze(-1)
