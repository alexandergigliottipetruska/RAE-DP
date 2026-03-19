"""V3 Observation Encoder — projects adapted tokens + proprio into conditioning sequence.

Replaces TokenAssembly (C.4) for V3. Key differences:
  - Spatial average pool: 196 patches → 1 vector per view (not 196→49 spatial pool)
  - Returns dict with 'tokens' (for transformer cross-attention) and 'global' (for U-Net FiLM)
  - Simpler: no spatial positional embeddings (pooled away), just view + time embeddings

Input:
  adapted_tokens: (B, T_o, K, 196, adapter_dim)  from Stage1Bridge
  proprio:        (B, T_o, proprio_dim)
  view_present:   (B, K) bool

Output dict:
  'tokens': (B, S_obs, d_model)   where S_obs = T_o * K + T_o
  'global': (B, global_dim)       flattened + projected for U-Net FiLM
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes adapted visual tokens + proprio into a conditioning sequence.

    For robomimic lift (T_o=2, K=2): S_obs = 2*2 + 2 = 6 tokens.
    For RLBench (T_o=2, K=4):       S_obs = 2*4 + 2 = 10 tokens.
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

        # Project pooled visual features to d_model
        self.view_proj = nn.Linear(adapter_dim, d_model)

        # Learned embeddings
        self.view_emb = nn.Embedding(num_views, d_model)
        self.time_emb = nn.Embedding(T_obs, d_model)

        # Proprio projection (2-layer MLP)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Global conditioning vector for U-Net FiLM (flatten + project)
        # S_obs = T_obs * num_views + T_obs
        max_s_obs = T_obs * num_views + T_obs
        self.global_proj = nn.Sequential(
            nn.Linear(max_s_obs * d_model, d_model),
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
                'tokens': (B, S_obs, d_model) — for transformer cross-attention memory
                'global': (B, d_model) — for U-Net FiLM conditioning
        """
        B, T_o, K = adapted_tokens.shape[:3]

        # 1. Spatial average pool: (B, T_o, K, N, D) → (B, T_o, K, D)
        pooled = adapted_tokens.mean(dim=3)

        # 2. Mask absent views (zero out missing cameras)
        mask = view_present[:, None, :, None].float()  # (B, 1, K, 1)
        pooled = pooled * mask

        # 3. Project to d_model and add view + time embeddings
        tokens = self.view_proj(pooled)  # (B, T_o, K, d_model)
        tokens = tokens + self.view_emb.weight[None, None, :K, :]
        tokens = tokens + self.time_emb.weight[None, :T_o, None, :]

        # 4. Flatten to sequence: (B, T_o * K, d_model)
        tokens = tokens.reshape(B, T_o * K, self.d_model)

        # 5. Proprio tokens
        proprio_tok = self.proprio_proj(proprio)  # (B, T_o, d_model)

        # 6. Concatenate: view tokens + proprio tokens
        obs_tokens = torch.cat([tokens, proprio_tok], dim=1)  # (B, T_o*K + T_o, d_model)

        # 7. Global conditioning vector (for U-Net option)
        global_vec = self.global_proj(obs_tokens.reshape(B, -1))  # (B, d_model)

        return {
            "tokens": obs_tokens,
            "global": global_vec,
        }
