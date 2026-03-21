"""V3 Observation Encoder — matches Chi's conditioning structure exactly.

Chi's approach: active camera features + proprio are concatenated into ONE flat
vector per timestep. This raw concat goes directly to the denoiser's cond_obs_emb
(Linear(1033, 256)) — NO extra projection in the encoder.

For robomimic (2 cameras): concat_dim = 2*512 + 9 = 1033 (matches Chi)
For RLBench (4 cameras): concat_dim = 4*512 + 8 = 2056

Memory = [timestep_token, obs_t0, obs_t1] = 3 tokens for T_o=2.

Input:
  adapted_tokens: (B, T_o, K, 196, adapter_dim)  from Stage1Bridge
  proprio:        (B, T_o, proprio_dim)
  view_present:   (B, K) bool

Output dict:
  'tokens': (B, T_o, concat_dim)  — raw concat, projected by denoiser's cond_obs_emb
  'global': (B, d_model)          — for U-Net FiLM (optional)
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes adapted visual tokens + proprio into conditioning vectors.

    Matches Chi exactly: pool each view → concat active views + proprio per timestep.
    NO LayerNorm, NO projection — raw concat goes to the denoiser's cond_obs_emb.

    Args:
        adapter_dim:    Adapter output dimension (512).
        d_model:        Transformer hidden dimension (256), only used for global_proj.
        proprio_dim:    Proprioceptive state dimension.
        T_obs:          Observation horizon.
        n_active_cams:  Number of ACTIVE cameras (2 for robomimic, 4 for RLBench).
    """

    def __init__(
        self,
        adapter_dim: int = 512,
        d_model: int = 256,
        proprio_dim: int = 9,
        T_obs: int = 2,
        n_active_cams: int = 2,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.d_model = d_model
        self.T_obs = T_obs
        self.n_active_cams = n_active_cams

        # Output dimension: concat of active views + proprio
        # Chi: 2 cameras → 1024 + 9 = 1033
        self.output_dim = n_active_cams * adapter_dim + proprio_dim

        # Global conditioning vector for U-Net FiLM
        self.global_proj = nn.Sequential(
            nn.Linear(T_obs * self.output_dim, d_model),
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
                'tokens': (B, T_o, concat_dim) — raw features for denoiser's cond_obs_emb
                'global': (B, d_model) — for U-Net FiLM conditioning
        """
        B, T_o, K = adapted_tokens.shape[:3]

        # 1. Spatial average pool: (B, T_o, K, N, D) → (B, T_o, K, D)
        pooled = adapted_tokens.mean(dim=3)

        # 2. Select only active camera features
        active_features = []
        for k in range(K):
            if view_present[:, k].any():
                active_features.append(pooled[:, :, k, :])  # (B, T_o, D)

        # Stack active cameras: (B, T_o, n_active, D)
        if len(active_features) > 0:
            active = torch.stack(active_features, dim=2)
        else:
            active = pooled[:, :, :1, :]

        # 3. Flatten active views per timestep: (B, T_o, n_active * adapter_dim)
        n_active = active.shape[2]
        active_flat = active.reshape(B, T_o, n_active * self.adapter_dim)

        # 4. Concatenate proprio: (B, T_o, n_active * adapter_dim + proprio_dim)
        obs_concat = torch.cat([active_flat, proprio], dim=-1)

        # 5. Global conditioning vector (for U-Net option)
        global_vec = self.global_proj(obs_concat.reshape(B, -1))

        return {
            "tokens": obs_concat,  # raw 1033-dim, denoiser's cond_obs_emb projects to d_model
            "global": global_vec,
        }
