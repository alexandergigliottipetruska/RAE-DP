"""DiT Denoiser — adaLN-Zero diffusion transformer for action prediction.

Conditions on a single dense vector c built from:
    c = time_proj(sinusoidal(t)) + obs_proj(mean_pool(obs_tokens))

Each DiT block applies adaLN-Zero modulation to both self-attention and MLP
sublayers. Causal masking on action tokens is preserved from the cross-attention
denoiser so temporal ordering is maintained.

Key architectural differences from TransformerDenoiser (Chi cross-attn):
  - No cross-attention to obs sequence — obs is mean-pooled into c
  - Timestep goes through a proper 2-layer MLP (not raw sinusoidal)
  - adaLN-Zero: gate/shift/scale from c, adaLN_modulation zero-init'd
  - No memory sequence — conditioning is entirely via c in each block

Interface: same as TransformerDenoiser.
    forward(noisy_actions, timestep, obs_cond) -> (B, T_pred, ac_dim)
    get_optim_groups(weight_decay) -> list of param groups
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) integer timesteps → (B, dim) embeddings."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiTBlock(nn.Module):
    """Single adaLN-Zero DiT block.

    Applies shift/scale/gate modulation to both self-attention and MLP
    sublayers. The adaLN_modulation projection is zero-initialized so gates
    start at 0 (identity residual), which stabilizes early training.

    Args:
        d_model:   Hidden dimension.
        n_head:    Number of attention heads.
        mlp_ratio: MLP expansion factor (4 → 4*d_model hidden dim).
        dropout:   Dropout on attention weights.
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()

        # Pre-norm without affine (modulation replaces the learnable scale/shift)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

        # adaLN-Zero: projects c → 6 modulation params per token
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the output projection: gates=0 means identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model) action token sequence.
            c:         (B, d_model) conditioning vector (time + obs).
            attn_mask: (T, T) additive causal mask or None.

        Returns:
            (B, T, d_model) updated sequence.
        """
        # Compute modulation params from conditioning vector
        mods = self.adaLN_modulation(c)  # (B, 6*d_model)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        # Unsqueeze for broadcast over sequence: (B, 1, d_model)
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa  = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp  = gate_mlp.unsqueeze(1)

        # Self-attention sublayer with adaLN modulation
        x_normed = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x_attn, _ = self.attn(
            x_normed, x_normed, x_normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + gate_msa * x_attn

        # MLP sublayer with adaLN modulation
        x_normed2 = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_normed2)

        return x


class DiTDenoiser(nn.Module):
    """DiT-based action denoiser with adaLN-Zero conditioning.

    Builds a single conditioning vector c from the diffusion timestep and
    mean-pooled observation tokens, then runs n_layers DiT blocks over the
    noisy action sequence.

    Args:
        ac_dim:      Action dimension (10 for robomimic rot6d, 8 for RLBench).
        d_model:     Transformer hidden dimension.
        n_head:      Number of attention heads.
        n_layers:    Number of DiT blocks.
        T_pred:      Prediction horizon (action sequence length).
        cond_dim:    Dimension of obs conditioning tokens (from ObservationEncoder).
        p_drop_emb:  Dropout on action input embeddings.
        p_drop_attn: Dropout on attention weights inside each DiT block.
        causal_attn: Whether to apply causal masking on action self-attention.
    """

    def __init__(
        self,
        ac_dim: int = 10,
        d_model: int = 256,
        n_head: int = 4,
        n_layers: int = 8,
        T_pred: int = 16,
        cond_dim: int = 1033,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
    ):
        super().__init__()
        self.ac_dim = ac_dim
        self.d_model = d_model
        self.T_pred = T_pred
        self.causal_attn = causal_attn

        # --- Timestep embedding: sinusoidal → 2-layer MLP ---
        # MLP projects to d_model (standard DiT approach, more expressive than raw sinusoidal)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

        # --- Obs conditioning: mean-pool T_obs tokens → project to d_model ---
        self.obs_proj = nn.Linear(cond_dim, d_model)

        # --- Action input embedding + learned positional encoding ---
        self.input_emb = nn.Linear(ac_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T_pred, d_model))
        self.drop = nn.Dropout(p_drop_emb)

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_head, mlp_ratio=4, dropout=p_drop_attn)
            for _ in range(n_layers)
        ])

        # --- Final LayerNorm + output head ---
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, ac_dim)

        # Weight init: normal(0, 0.02) for all linear/embedding weights,
        # then re-apply zero-init for adaLN_modulation (apply() runs top-down
        # so we must redo it after)
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        # Re-zero adaLN modulation outputs (apply() overwrote them)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def _init_weights(self, module):
        """Normal(0, 0.02) init for Linear/Embedding; ones/zeros for affine LayerNorm."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]:
                w = getattr(module, name, None)
                if w is not None:
                    nn.init.normal_(w, mean=0.0, std=0.02)
            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                b = getattr(module, name, None)
                if b is not None:
                    nn.init.zeros_(b)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Split parameters into decay/no_decay groups.

        Decay:    Linear and MultiheadAttention weight matrices.
        No decay: biases, affine LayerNorm weights, positional embeddings.
        """
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        # pos_emb is a Parameter on the root module, not inside any submodule
        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        assert len(inter) == 0, f"params in both decay/no_decay: {inter}"
        union = decay | no_decay
        assert len(param_dict.keys() - union) == 0, \
            f"unclaimed params: {param_dict.keys() - union}"

        return [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: dict,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T_pred, ac_dim) — noisy action trajectory.
            timestep:      (B,) int — diffusion step k (0–99).
            obs_cond:      dict with 'tokens': (B, T_obs, cond_dim) from ObservationEncoder.

        Returns:
            (B, T_pred, ac_dim) — predicted noise (epsilon).
        """
        B = noisy_actions.shape[0]
        obs_tokens = obs_cond["tokens"]  # (B, T_obs, cond_dim)

        # 1. Normalize timestep tensor shape
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_actions.device)
        elif timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)

        # 2. Build conditioning vector c = time_proj(sin(t)) + obs_proj(mean(obs))
        t_emb = self.time_proj(self.time_emb(timestep))    # (B, d_model)
        obs_emb = self.obs_proj(obs_tokens.mean(dim=1))    # (B, d_model)
        c = t_emb + obs_emb                                # (B, d_model)

        # 3. Action tokens: embed + positional + dropout
        t = noisy_actions.shape[1]
        x = self.drop(self.input_emb(noisy_actions) + self.pos_emb[:, :t, :])  # (B, T_pred, d_model)

        # 4. Causal self-attention mask
        attn_mask = None
        if self.causal_attn:
            attn_mask = torch.triu(
                torch.full((t, t), float("-inf"), device=noisy_actions.device),
                diagonal=1,
            )

        # 5. DiT blocks
        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)

        # 6. Output head
        x = self.norm_f(x)
        return self.head(x)  # (B, T_pred, ac_dim)
