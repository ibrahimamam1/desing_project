"""
AttentionPolicyModel — Custom RLlib TorchModelV2 with cross-attention.

Architecture:
    1. Split flat obs → ego features, neighbor features, neighbor mask
    2. Embed ego (query) and neighbors (key/value) through separate linear layers
    3. Masked multi-head cross-attention: Q=ego, K=V=neighbors
    4. Concatenate [ego_embed, attn_output] → MLP → policy logits + value
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class AttentionPolicyModel(TorchModelV2, nn.Module):
    """
    Cross-attention policy model for multi-agent intersection control.
    
    Expected flat observation layout:
        [ego_features (2), neighbor_features (max_k * 2), neighbor_mask (max_k)]
    
    Default: 2 + 5*2 + 5 = 17
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # --- Hyperparameters (configurable via custom_model_config) ---
        custom_cfg = model_config.get("custom_model_config", {})
        
        self.ego_features = custom_cfg.get("ego_features", 2)
        self.neighbor_features = custom_cfg.get("neighbor_features", 2)
        self.max_neighbors = custom_cfg.get("max_neighbors", 5)
        self.embed_dim = custom_cfg.get("embed_dim", 64)
        self.num_heads = custom_cfg.get("num_heads", 4)
        self.mlp_hidden = custom_cfg.get("mlp_hidden", 256)

        # Verify obs dimension matches expected layout
        expected_obs_dim = self.ego_features + (self.max_neighbors * self.neighbor_features) + self.max_neighbors
        actual_obs_dim = int(np.prod(obs_space.shape))
        assert actual_obs_dim == expected_obs_dim, (
            f"Observation dim mismatch: expected {expected_obs_dim}, got {actual_obs_dim}. "
            f"ego={self.ego_features}, neighbors={self.max_neighbors}x{self.neighbor_features}, mask={self.max_neighbors}"
        )

        # --- Encoders ---
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_features, self.embed_dim),
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_features, self.embed_dim),
            nn.ReLU(),
        )

        # --- Multi-Head Cross-Attention ---
        # PyTorch MHA expects (seq_len, batch, embed_dim) by default
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,  # Use (batch, seq, embed) format
        )

        # --- Output MLPs ---
        context_dim = self.embed_dim * 2  # ego_embed + attn_output

        self.policy_mlp = nn.Sequential(
            nn.Linear(context_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, num_outputs),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(context_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 1),
        )

        # Cache for value function
        self._context = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()  # (B, 17)
        
        # --- 1. Split observation ---
        ego_end = self.ego_features
        neighbor_end = ego_end + (self.max_neighbors * self.neighbor_features)
        
        ego_raw = obs[:, :ego_end]                          # (B, 2)
        neighbor_raw = obs[:, ego_end:neighbor_end]         # (B, 10)
        mask_raw = obs[:, neighbor_end:]                    # (B, 5)

        # Reshape neighbors: (B, 10) → (B, 5, 2)
        neighbor_raw = neighbor_raw.view(-1, self.max_neighbors, self.neighbor_features)

        # --- 2. Encode ---
        ego_embed = self.ego_encoder(ego_raw)               # (B, embed_dim)
        neighbor_embeds = self.neighbor_encoder(neighbor_raw)  # (B, 5, embed_dim)

        # --- 3. Build attention mask ---
        # key_padding_mask: True = IGNORE this position
        # mask_raw: 1.0 = real neighbor, 0.0 = padded → invert for padding mask
        key_padding_mask = (mask_raw < 0.5)  # (B, 5), True where padded

        # Handle edge case: if ALL neighbors are masked, skip attention
        all_masked = key_padding_mask.all(dim=1)  # (B,)
        
        # --- 4. Cross-Attention ---
        # Q: ego as single query → (B, 1, embed_dim)
        query = ego_embed.unsqueeze(1)   # (B, 1, embed_dim)
        key = neighbor_embeds            # (B, 5, embed_dim)
        value = neighbor_embeds          # (B, 5, embed_dim)

        # For rows where all neighbors are masked, temporarily unmask to avoid NaN
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked] = False  # unmask all for these rows (attention output will be overwritten)

        attn_output, _ = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=safe_mask,
        )  # attn_output: (B, 1, embed_dim)

        attn_output = attn_output.squeeze(1)  # (B, embed_dim)

        # Zero out attention output for rows where all neighbors were masked
        attn_output[all_masked] = 0.0

        # --- 5. Concatenate and decode ---
        context = torch.cat([ego_embed, attn_output], dim=-1)  # (B, 2*embed_dim)
        self._context = context  # Cache for value_function()

        policy_out = self.policy_mlp(context)  # (B, num_outputs)

        return policy_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._context is not None, "forward() must be called before value_function()"
        return self.value_mlp(self._context).squeeze(-1)  # (B,)
