"""Stage 3 diffusion policy training loop (C.7).

Trains a DiT-Block Policy on adapted visual tokens from Stage 1 using
DDPM noise prediction. Supports optional co-training with reconstruction
loss (lambda_recon > 0).

Key differences from Stage 1:
  - DDPM noise prediction loss (MSE on predicted epsilon)
  - Per-task training (one policy per task)
  - Cosine LR with linear warmup
  - EMA model for evaluation
  - Separate LR for adapter (prevents drift from Stage 1 weights)
  - Gradient clipping (max_norm=1.0)
  - DDIM inference for evaluation

Expected component interfaces:
  noise_net.forward_enc(obs_tokens)                -> enc_cache (list of [B, S, d])
  noise_net.forward_dec(noisy_actions, time, cache) -> eps_pred [B, T_p, D_act]
  noise_net(noisy_actions, time, obs_tokens)       -> (enc_cache, eps_pred)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from data_pipeline.datasets.stage3_dataset import Stage3Dataset
from models.ema import EMAModel
from models.view_dropout import ViewDropout

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers (shared with Stage 1)
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_distributed() else 0


def _is_main() -> bool:
    return _rank() == 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_distributed() else 1


def _unwrap(model: nn.Module) -> nn.Module:
    """Get the underlying module from DDP, DataParallel, or torch.compile wrapper."""
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage3Config:
    """Hyperparameters for Stage 3 diffusion policy training."""

    # Stage 1 checkpoint
    stage1_checkpoint: str = ""

    # Data
    hdf5_paths: list = field(default_factory=list)
    batch_size: int = 64
    num_workers: int = 4
    norm_mode: str = "minmax"

    # Architecture
    T_obs: int = 2
    T_pred: int = 16
    T_act: int = 8              # execution horizon (eval only)
    hidden_dim: int = 512
    num_blocks: int = 6
    nhead: int = 8
    use_lightning: bool = True

    # Diffusion
    train_diffusion_steps: int = 100
    eval_diffusion_steps: int = 10

    # Training
    lr: float = 1e-4
    lr_adapter: float = 1e-5    # 10x lower to prevent drift
    weight_decay: float = 1e-4
    num_epochs: int = 300
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # Co-training (reconstruction alongside policy)
    lambda_recon: float = 0.0   # ablate {0, 0.1, 0.25, 0.5}

    # View dropout
    p_drop: float = 0.15

    # EMA
    ema_decay: float = 0.9999

    # Logging & checkpointing
    log_every: int = 100
    save_every_epoch: int = 10
    eval_every_epoch: int = 50
    save_dir: str = "checkpoints/stage3"


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def create_noise_scheduler(num_train_steps: int = 100) -> DDIMScheduler:
    """Create DDIM scheduler for DDPM training / DDIM inference."""
    return DDIMScheduler(
        num_train_timesteps=num_train_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    noise_net: nn.Module,
    encoder: nn.Module,
    adapter: nn.Module,
    view_dropout: ViewDropout,
    scheduler: DDIMScheduler,
    optimizer: torch.optim.Optimizer,
    config: Stage3Config,
    global_step: int,
    decoder: nn.Module | None = None,
    lpips_net=None,
    use_amp: bool = False,
) -> dict:
    """One DDPM training step.

    Args:
        batch:        Dict from Stage3Dataset (images_enc, actions, proprio, view_present).
        noise_net:    DiT noise prediction network (_DiTNoiseNet or wrapper).
        encoder:      Frozen Stage 1 encoder.
        adapter:      Trainable adapter (shared from Stage 1, lower LR).
        view_dropout: ViewDropout module.
        scheduler:    DDIM noise scheduler.
        optimizer:    AdamW optimizer.
        config:       Training config.
        global_step:  Current global step (for warmup).
        decoder:      Optional decoder for co-training L_recon.
        lpips_net:    Optional LPIPS net for co-training.
        use_amp:      Whether to use BF16 mixed precision.

    Returns:
        Dict of scalar losses for logging.
    """
    device = next(noise_net.parameters()).device
    device_type = device.type

    images_enc = batch["images_enc"].to(device)       # [B, T_o, K, 3, H, W]
    actions = batch["actions"].to(device)              # [B, T_p, D_act]
    view_present = batch["view_present"].to(device)    # [B, K]

    B, T_o, K, C, H, W = images_enc.shape

    with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
        # --- Step 1: Encode observations (frozen) ---
        with torch.no_grad():
            # Flatten [B, T_o, K, 3, H, W] -> [B*T_o*K, 3, H, W]
            flat_imgs = images_enc.reshape(B * T_o * K, C, H, W)
            flat_tokens = encoder(flat_imgs)  # [B*T_o*K, N, d]

        # Reshape back: [B, T_o, K, N, d]
        N, d = flat_tokens.shape[1], flat_tokens.shape[2]
        tokens = flat_tokens.reshape(B, T_o, K, N, d)

        # --- Step 2: Adapt (trainable, lower LR) ---
        adapted = adapter(tokens.reshape(B * T_o * K, N, d))
        d_prime = adapted.shape[-1]
        adapted = adapted.reshape(B, T_o, K, N, d_prime)

        # --- Step 3: View dropout (training only) ---
        # Apply per-timestep, reshape to [B, K, N, d'] for ViewDropout
        adapted_dropped = []
        vp_updated = view_present
        for t_idx in range(T_o):
            dropped_t, vp_updated = view_dropout(adapted[:, t_idx], vp_updated)
            adapted_dropped.append(dropped_t)
        adapted = torch.stack(adapted_dropped, dim=1)  # [B, T_o, K, N, d']

        # --- Step 4: Flatten obs tokens for noise net ---
        # Mask absent views, flatten [T_o, K, N] -> [S_obs] per sample
        # Simple: flatten all tokens (noise_net encoder handles it)
        obs_tokens = adapted.reshape(B, T_o * K * N, d_prime)

        # --- Step 5: DDPM forward process ---
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, config.train_diffusion_steps, (B,),
            device=device, dtype=torch.long,
        )
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)

        # --- Step 6: Predict noise ---
        _, eps_pred = noise_net(noisy_actions, timesteps, obs_tokens)

        # --- Step 7: MSE loss on predicted noise ---
        L_policy = nn.functional.mse_loss(eps_pred, noise)

        # --- Step 8: Optional co-training reconstruction loss ---
        L_recon = torch.tensor(0.0, device=device)
        if config.lambda_recon > 0 and decoder is not None:
            images_target = batch["images_target"].to(device)  # [B, T_o, K, 3, H, W]
            # Use adapted tokens WITHOUT dropout for reconstruction
            # Pick last timestep for efficiency
            recon_tokens = adapter(
                tokens[:, -1].reshape(B * K, N, d)
            ).reshape(B * K, N, d_prime)

            mask = view_present.reshape(B * K)
            recon_input = recon_tokens[mask]
            recon_target = images_target[:, -1].reshape(B * K, C, H, W)[mask]

            pred_imgs = decoder(recon_input)
            from models.losses import l1_loss, lpips_loss_fn
            L_l1 = l1_loss(pred_imgs, recon_target)
            L_lpips = lpips_loss_fn(pred_imgs, recon_target, lpips_net)
            L_recon = L_l1 + L_lpips

        L_total = L_policy + config.lambda_recon * L_recon

    # --- Backward + optimizer step ---
    optimizer.zero_grad()
    L_total.backward()

    # Gradient clipping
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in noise_net.parameters() if p.requires_grad] +
            [p for p in adapter.parameters() if p.requires_grad],
            config.grad_clip,
        )

    optimizer.step()

    losses = {
        "policy": L_policy.item(),
        "total": L_total.item(),
    }
    if config.lambda_recon > 0:
        losses["recon"] = L_recon.item()

    return losses


# ---------------------------------------------------------------------------
# DDIM inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_inference(
    noise_net: nn.Module,
    obs_tokens: torch.Tensor,
    ac_dim: int,
    T_pred: int,
    scheduler: DDIMScheduler,
    num_steps: int = 10,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Run DDIM denoising to produce action predictions.

    Args:
        noise_net: DiT noise prediction network.
        obs_tokens: [B, S_obs, d] encoded observation tokens.
        ac_dim:     Action dimension.
        T_pred:     Prediction horizon.
        scheduler:  DDIM scheduler.
        num_steps:  Number of DDIM denoising steps.
        device:     Device.

    Returns:
        Predicted actions [B, T_pred, ac_dim].
    """
    B = obs_tokens.shape[0]
    device = obs_tokens.device

    # Cache encoder outputs
    enc_cache = noise_net.forward_enc(obs_tokens)

    # Start from pure noise
    actions = torch.randn(B, T_pred, ac_dim, device=device)

    # Set inference timesteps
    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        _, eps_pred = noise_net.forward_dec(actions, t_batch, enc_cache)
        actions = scheduler.step(eps_pred, t, actions).prev_sample

    return actions


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup
# ---------------------------------------------------------------------------

def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    noise_net: nn.Module,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMAModel | None,
    val_metrics: dict,
    decoder: nn.Module | None = None,
):
    """Save Stage 3 training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "noise_net": _unwrap(noise_net).state_dict(),
        "adapter": _unwrap(adapter).state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if decoder is not None:
        ckpt["decoder"] = _unwrap(decoder).state_dict()
    torch.save(ckpt, path)
    log.info("Saved checkpoint: %s (epoch %d, step %d)", path, epoch, global_step)


def load_checkpoint(
    path: str,
    noise_net: nn.Module,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMAModel | None = None,
    decoder: nn.Module | None = None,
) -> tuple[int, int]:
    """Load Stage 3 checkpoint. Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    def _strip_compile_prefix(state_dict):
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            return {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return state_dict

    noise_net.load_state_dict(_strip_compile_prefix(ckpt["noise_net"]))
    adapter.load_state_dict(_strip_compile_prefix(ckpt["adapter"]))
    optimizer.load_state_dict(ckpt["optimizer"])

    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if decoder is not None and "decoder" in ckpt:
        decoder.load_state_dict(_strip_compile_prefix(ckpt["decoder"]))

    log.info("Loaded checkpoint from epoch %d, step %d: %s",
             ckpt["epoch"], ckpt["global_step"], path)
    return ckpt["epoch"] + 1, ckpt["global_step"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_stage3(
    config: Stage3Config,
    *,
    encoder: nn.Module,
    adapter: nn.Module,
    noise_net: nn.Module,
    decoder: nn.Module | None = None,
    device: torch.device | str = "cuda",
    resume_from: str | None = None,
):
    """Main Stage 3 training entry point.

    Supports single-GPU and DDP (via torchrun). DDP is auto-detected.

    Args:
        config:     Training configuration.
        encoder:    Frozen Stage 1 encoder (A.1).
        adapter:    Trainable adapter (A.2), loaded from Stage 1 checkpoint.
        noise_net:  DiT noise prediction network (from diffusion.py).
        decoder:    Optional decoder for co-training L_recon.
        device:     Device (ignored under DDP; uses LOCAL_RANK).
        resume_from: Path to checkpoint to resume from.
    """
    distributed = _is_distributed()
    rank = _rank()
    is_main = _is_main()

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)

    use_amp = (device.type == "cuda")

    # Move models to device
    encoder = encoder.to(device).eval()
    adapter = adapter.to(device).train()
    noise_net = noise_net.to(device).train()

    # View dropout
    view_dropout = ViewDropout(
        d_model=config.hidden_dim, p_drop=config.p_drop
    ).to(device)

    # DDPM scheduler
    scheduler = create_noise_scheduler(config.train_diffusion_steps)

    # Optional co-training components
    lpips_net = None
    if config.lambda_recon > 0 and decoder is not None:
        decoder = decoder.to(device).train()
        from models.losses import create_lpips_net
        lpips_net = create_lpips_net().to(device)

    # EMA
    ema = EMAModel(noise_net, decay=config.ema_decay)

    # Load checkpoint before DDP wrapping
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isfile(resume_from):
        _opt_tmp = torch.optim.AdamW(
            list(noise_net.parameters()) + list(adapter.parameters()),
            lr=config.lr,
        )
        start_epoch, global_step = load_checkpoint(
            resume_from, noise_net, adapter, _opt_tmp, ema, decoder
        )

    # DDP wrapping
    if distributed:
        noise_net = nn.parallel.DistributedDataParallel(
            noise_net, device_ids=[local_rank]
        )
        adapter = nn.parallel.DistributedDataParallel(
            adapter, device_ids=[local_rank]
        )
        if decoder is not None:
            decoder = nn.parallel.DistributedDataParallel(
                decoder, device_ids=[local_rank]
            )

    # Dataset + DataLoader
    train_ds = Stage3Dataset(
        config.hdf5_paths, split="train",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode,
    )
    valid_ds = Stage3Dataset(
        config.hdf5_paths, split="valid",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer with separate param groups
    param_groups = [
        {"params": list(noise_net.parameters()), "lr": config.lr},
        {"params": list(adapter.parameters()), "lr": config.lr_adapter},
    ]
    if decoder is not None and config.lambda_recon > 0:
        param_groups.append(
            {"params": list(decoder.parameters()), "lr": config.lr_adapter}
        )
    param_groups.append(
        {"params": list(view_dropout.parameters()), "lr": config.lr}
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )

    # Reload optimizer state if resuming
    if resume_from and os.path.isfile(resume_from) and start_epoch > 0:
        optimizer.load_state_dict(_opt_tmp.state_dict())

    total_steps = config.num_epochs * len(train_loader)
    lr_scheduler = _create_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    # --- Logging (rank 0 only) ---
    metrics_path = None
    if is_main:
        os.makedirs(config.save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.save_dir, f"train_{ts}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        log.addHandler(fh)

        metrics_path = os.path.join(config.save_dir, f"metrics_{ts}.jsonl")

        run_info = {
            "type": "run_info",
            "timestamp": ts,
            "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
            "config": {k: v for k, v in config.__dict__.items()},
            "resume_from": resume_from,
            "start_epoch": start_epoch,
            "train_samples": len(train_ds),
            "valid_samples": len(valid_ds),
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(run_info, default=str) + "\n")

        log.info("=" * 60)
        log.info("Stage 3 Diffusion Policy Training")
        log.info("=" * 60)
        log.info("Train: %d samples, Valid: %d samples", len(train_ds), len(valid_ds))
        log.info("Config: batch=%d, lr=%.1e, lr_adapter=%.1e, T_pred=%d",
                 config.batch_size, config.lr, config.lr_adapter, config.T_pred)
        log.info("Diffusion: %d train steps, %d eval steps",
                 config.train_diffusion_steps, config.eval_diffusion_steps)
        log.info("Co-training: lambda_recon=%.3f", config.lambda_recon)
        log.info("=" * 60)

    # --- Training loop ---
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        noise_net.train()
        adapter.train()
        view_dropout.train()

        epoch_losses: dict[str, float] = {}
        n_steps = 0

        loader_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
            if is_main else train_loader
        )

        for batch in loader_iter:
            step_losses = train_step(
                batch, noise_net, encoder, adapter, view_dropout,
                scheduler, optimizer, config, global_step,
                decoder=decoder, lpips_net=lpips_net, use_amp=use_amp,
            )

            # EMA update
            ema.update(_unwrap(noise_net))

            # LR schedule
            lr_scheduler.step()

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_steps += 1
            global_step += 1

        # Average epoch losses
        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # Validation + logging (rank 0 only)
        if is_main:
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()))
            log.info("Epoch %d  %s  (lr=%.2e)", epoch, train_str,
                     optimizer.param_groups[0]["lr"])

            if metrics_path:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "global_step": global_step,
                        "train": avg,
                    }) + "\n")

            # Periodic checkpointing
            if (epoch + 1) % config.save_every_epoch == 0:
                save_checkpoint(
                    os.path.join(config.save_dir, f"epoch_{epoch:03d}.pt"),
                    epoch, global_step, noise_net, adapter, optimizer, ema, avg,
                    decoder=decoder,
                )

            # Best checkpoint
            if avg.get("policy", float("inf")) < best_val_loss:
                best_val_loss = avg["policy"]
                save_checkpoint(
                    os.path.join(config.save_dir, "best.pt"),
                    epoch, global_step, noise_net, adapter, optimizer, ema, avg,
                    decoder=decoder,
                )

        if distributed:
            torch.distributed.barrier()

    if is_main:
        log.info("Training complete. Best policy loss=%.4f", best_val_loss)
        log.removeHandler(fh)
        fh.close()
