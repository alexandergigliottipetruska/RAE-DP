"""V3 Observation Encoder — matches Chi's conditioning structure exactly.

Chi's approach: all camera features + proprio are concatenated into ONE flat
vector per timestep, then projected to d_model. This produces T_o conditioning
tokens (one per observation timestep), NOT separate tokens per view.

Memory = [timestep_token, obs_t0, obs_t1] = 3 tokens for T_o=2.

Input:
  adapted_tokens: (B, T_o, K, 196, adapter_dim)  from Stage1Bridge
  proprio:        (B, T_o, proprio_dim)
  view_present:   (B, K) bool

Output dict:
  'tokens': (B, T_o, d_model)    — one token per observation timestep
  'global': (B, d_model)         — for U-Net FiLM (optional)
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes adapted visual tokens + proprio into conditioning tokens.

    Matches Chi's pipeline: pool each view → concat all views + proprio per
    timestep → project to d_model. Output is T_o tokens (NOT T_o*K + T_o).

    For robomimic lift (T_o=2, K=2): S_obs = 2 tokens.
    For RLBench (T_o=2, K=4):       S_obs = 2 tokens.
    """

    def __init__(
        self,
        adapter_dim: int = 512,
        d_model: int = 256,
        proprio_dim: int = 9,
        T_obs: int = 2,
        num_views: int = 4,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.d_model = d_model
        self.T_obs = T_obs
        self.num_views = num_views

        # Project concatenated [all_views + proprio] to d_model
        # Chi: obs_encoder outputs flat vector → cond_obs_emb projects to n_emb
        concat_dim = num_views * adapter_dim + proprio_dim
        self.obs_proj = nn.Linear(concat_dim, d_model)

        # Global conditioning vector for U-Net FiLM
        self.global_proj = nn.Sequential(
            nn.Linear(T_obs * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        proprio: torch.Tensor,
        view_present: torch.Tensor,
    ) -> dict:
        """
        Args:
            adapted_tokens: (B, T_o, K, N_patches, adapter_dim) — from Stage1Bridge
            proprio:        (B, T_o, proprio_dim)
            view_present:   (B, K) bool — which cameras are active

        Returns:
            dict with:
                'tokens': (B, T_o, d_model) — one conditioning token per timestep
                'global': (B, d_model) — for U-Net FiLM conditioning
        """
        B, T_o, K = adapted_tokens.shape[:3]

        # 1. Spatial average pool: (B, T_o, K, N, D) → (B, T_o, K, D)
        pooled = adapted_tokens.mean(dim=3)

        # 2. Mask absent views (zero out missing cameras)
        mask = view_present[:, None, :, None].float()  # (B, 1, K, 1)
        pooled = pooled * mask

        # 3. Flatten views per timestep: (B, T_o, K*adapter_dim)
        pooled_flat = pooled.reshape(B, T_o, K * self.adapter_dim)

        # 4. Concatenate proprio: (B, T_o, K*adapter_dim + proprio_dim)
        obs_concat = torch.cat([pooled_flat, proprio], dim=-1)

        # 5. Project to d_model: (B, T_o, d_model)
        obs_tokens = self.obs_proj(obs_concat)

        # 6. Global conditioning vector (for U-Net option)
        global_vec = self.global_proj(obs_tokens.reshape(B, -1))

        return {
            "tokens": obs_tokens,
            "global": global_vec,
        }
