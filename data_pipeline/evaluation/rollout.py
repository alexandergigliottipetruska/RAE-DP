"""Receding-horizon evaluation rollout loop.

Benchmark-agnostic: only the environment wrapper changes per benchmark.

Protocol:
  observe (T_o frames) -> predict action chunk (T_p) -> execute first T_a actions
  -> re-observe -> repeat until done or max_steps
"""

import numpy as np


def run_rollout(env, policy, T_obs: int = 2, T_action: int = 8, max_steps: int = 400) -> dict:
    """Run one episode. Returns dict with 'success' and 'length'."""
    raise NotImplementedError


def evaluate_checkpoint(
    checkpoint_path: str,
    env_fn,
    n_episodes: int = 25,
    camera_dropout: list = None,
) -> dict:
    """Evaluate a checkpoint. Returns success_rate and episode stats."""
    raise NotImplementedError
