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
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # --- Hyperparameters ---
        custom_cfg = model_config.get("custom_model_config", {})
        
        self.ego_features = custom_cfg.get("ego_features", 4)
        self.neighbor_features = custom_cfg.get("neighbor_features", 5)
        self.max_neighbors = custom_cfg.get("max_neighbors", 5)
        self.embed_dim = custom_cfg.get("embed_dim", 64)
        self.num_heads = custom_cfg.get("num_heads", 4)
        self.mlp_hidden = custom_cfg.get("mlp_hidden", 256)

        # --- Encoders (Added LayerNorm for stability) ---
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_features, self.embed_dim),
            nn.LayerNorm(self.embed_dim), 
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_features, self.embed_dim),
            nn.LayerNorm(self.embed_dim), 
            nn.ReLU(),
        )

        # --- Multi-Head Cross-Attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,  
        )

        # --- Output MLPs ---
        context_dim = self.embed_dim * 2  
        
        # Crucial Fix: Normalize the context vector before the MLP
        self.context_norm = nn.LayerNorm(context_dim)

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
        
        # Crucial Fix: Initialize final policy layer to output near-zero actions initially
        # This prevents the Gaussian mean from starting at massive values
        nn.init.orthogonal_(self.policy_mlp[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_mlp[-1].bias, 0.0)

        # Cache for value function
        self._context = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()  
        
        # --- 1. Split observation ---
        ego_end = self.ego_features
        neighbor_end = ego_end + (self.max_neighbors * self.neighbor_features)
        
        ego_raw = obs[:, :ego_end]                         
        neighbor_raw = obs[:, ego_end:neighbor_end]         
        mask_raw = obs[:, neighbor_end:]                    

        # Reshape neighbors: (B, max_neighbors, features)
        neighbor_raw = neighbor_raw.reshape(-1, self.max_neighbors, self.neighbor_features)

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
            query=query,
            key=key,
            value=value,
            key_padding_mask=safe_mask,
        )  

        attn_output = attn_output.squeeze(1)  

        # Zero out attention output for rows where all neighbors were masked
        attn_output = torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(attn_output),
            attn_output
        )

        # --- 5. Concatenate and decode ---
        context = torch.cat([ego_embed, attn_output], dim=-1)  
        
        # Apply normalization to the concatenated output
        context = self.context_norm(context)
        
        self._context = context  

        policy_out = self.policy_mlp(context)  

        return policy_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._context is not None, "forward() must be called before value_function()"
        return self.value_mlp(self._context).squeeze(-1)
