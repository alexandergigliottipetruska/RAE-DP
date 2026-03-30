"""V3 evaluation for RLBench tasks using JointPosition control.

Actions: 8D absolute joint positions [joint_targets(7), gripper(1)].
Normalization: minmax on all dims, inverse at eval time.
No rot6d, no OMPL, no chi normalization.

Validated via GT replay: 100% success on open_drawer.

Usage:
    from training.eval_v3_rlbench import evaluate_v3_rlbench
    sr, results, env = evaluate_v3_rlbench(wrapper, norm_stats, task="open_drawer")
"""

import logging
import os
import signal
from collections import deque

import numpy as np
import torch

log = logging.getLogger(__name__)

TASK_MAX_STEPS = {
    "close_jar": 250,
    "open_drawer": 150,
    "slide_block_to_color_target": 200,
    "meat_off_grill": 200,
    "place_wine_at_rack_location": 250,
    "push_buttons": 250,
    "sweep_to_dustpan_of_size": 250,
    "turn_tap": 200,
}


def _denorm_actions(pred: np.ndarray, action_stats: dict) -> np.ndarray:
    """Inverse minmax normalization: [-1,1] → original range."""
    a_min = action_stats["min"]
    a_max = action_stats["max"]
    return (pred + 1.0) / 2.0 * (a_max - a_min) + a_min


def _temporal_ensemble(action_history, step, T_pred, gain=0.01):
    """Weighted average of overlapping action predictions."""
    predictions = []
    for pred_step, chunk in action_history:
        offset = step - pred_step
        if 0 <= offset < chunk.shape[0]:
            predictions.append(chunk[offset])

    if len(predictions) == 0:
        raise ValueError(f"No predictions cover step {step}")
    if len(predictions) == 1:
        return predictions[0]

    preds = np.stack(predictions, axis=0)
    weights = np.exp(-gain * np.arange(len(preds))[::-1])
    weights /= weights.sum()
    return (preds * weights[:, None]).sum(axis=0)


def _run_episode(policy, env, action_stats, proprio_stats,
                 max_steps, obs_horizon, save_frames=False,
                 exec_horizon=1, demo=None) -> dict:
    """Run a single RLBench episode with JointPosition control."""
    if demo is not None:
        try:
            descriptions, obs = env._task.reset_to_demo(demo)
        except (KeyError, AttributeError):
            env._task.set_variation(demo.variation_number)
            descriptions, obs = env._task.reset()
        env._last_obs = obs
    else:
        env.reset()

    img_buffer = deque([env.get_multiview_images()] * obs_horizon, maxlen=obs_horizon)
    proprio_buffer = deque([env.get_proprio()] * obs_horizon, maxlen=obs_horizon)
    view_present = env.get_view_present()

    p_min, p_max = proprio_stats["min"], proprio_stats["max"]
    action_history = deque(maxlen=50)

    frames = []
    step_count = 0
    total_reward = 0.0
    info = {}
    pending_actions = []
    T_pred = None

    while step_count < max_steps:
        if pending_actions:
            action = pending_actions.pop(0)
        else:
            images_seq = np.concatenate(list(img_buffer), axis=0)
            proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

            # Minmax normalize proprio (all dims)
            p_range = np.clip(p_max - p_min, 1e-6, None)
            proprio_norm = 2.0 * (proprio_seq - p_min) / p_range - 1.0

            with torch.no_grad():
                pred = policy.predict(
                    torch.from_numpy(images_seq).unsqueeze(0),
                    torch.from_numpy(proprio_norm).unsqueeze(0),
                    torch.from_numpy(view_present).unsqueeze(0),
                )

            raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
            actions_8d = _denorm_actions(raw, action_stats)

            if T_pred is None:
                T_pred = actions_8d.shape[0]

            action_history.append((step_count, actions_8d))
            action = _temporal_ensemble(
                list(action_history), step_count, T_pred,
            )
            if exec_horizon > 1:
                for j in range(1, min(exec_horizon, T_pred)):
                    a = _temporal_ensemble(
                        list(action_history), step_count + j, T_pred,
                    )
                    pending_actions.append(a)

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            log.warning("  Step failed at step %d: %s", step_count, e)
            return {"success": False, "steps": step_count, "reward": total_reward}

        step_count += 1
        total_reward += reward

        img_buffer.append(env.get_multiview_images())
        proprio_buffer.append(env.get_proprio())

        if save_frames:
            img = img_buffer[-1][0]
            views = [img[i].transpose(1, 2, 0) for i in range(img.shape[0])]
            frame = (np.concatenate(views, axis=1) * 255).astype(np.uint8)
            frames.append(frame)

        if done:
            break

    success = info.get("success", False) if isinstance(info, dict) else False
    result = {"success": success, "steps": step_count, "reward": total_reward}
    if save_frames:
        result["frames"] = frames
    return result


def evaluate_v3_rlbench(
    wrapper, norm_stats, task="close_jar", num_episodes=25,
    max_steps=0, obs_horizon=2, image_size=224, headless=True,
    episode_timeout=180, save_video=False, video_dir="",
    demos=None, exec_horizon=1, _cached_env=None,
) -> tuple:
    """Run V3 evaluation on an RLBench task."""
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    if max_steps == 0:
        max_steps = TASK_MAX_STEPS.get(task, 200)

    env = _cached_env
    if env is None:
        log.info("Creating RLBenchWrapper for %s (headless=%s)...", task, headless)
        env = RLBenchWrapper(task_name=task, image_size=image_size, headless=headless)

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    if save_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)

    class _EpisodeTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _EpisodeTimeout()

    results = []
    for ep in range(num_episodes):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(episode_timeout)

        try:
            demo = demos[ep] if demos is not None and ep < len(demos) else None
            ep_result = _run_episode(
                wrapper, env, action_stats, proprio_stats,
                max_steps, obs_horizon,
                save_frames=save_video,
                exec_horizon=exec_horizon,
                demo=demo,
            )
            signal.alarm(0)

            if save_video and ep_result.get("frames"):
                import imageio
                tag = "success" if ep_result["success"] else "fail"
                path = os.path.join(video_dir, f"ep{ep:03d}_{tag}.mp4")
                imageio.mimwrite(path, ep_result.pop("frames"), fps=10)

            results.append(ep_result)
        except _EpisodeTimeout:
            log.warning("Episode %d timed out after %ds — recreating env",
                        ep, episode_timeout)
            results.append({"success": False, "steps": 0, "reward": 0.0})
            try:
                env.close()
            except Exception:
                pass
            env = RLBenchWrapper(task_name=task, image_size=image_size,
                                headless=headless)
        except Exception as e:
            signal.alarm(0)
            log.warning("Episode %d failed: %s", ep, e)
            results.append({"success": False, "steps": 0, "reward": 0.0})
        finally:
            signal.signal(signal.SIGALRM, old_handler)

        n_success = sum(1 for r in results if r["success"])
        log.info("Episode %d/%d: %s (%d steps) | Running: %d/%d (%.0f%%)",
                 ep + 1, num_episodes,
                 "SUCCESS" if results[-1]["success"] else "FAIL",
                 results[-1]["steps"],
                 n_success, len(results),
                 100 * n_success / len(results))

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / num_episodes
    log.info("RLBench eval (%s): %.1f%% success (%d/%d episodes)",
             task, success_rate * 100, n_success, num_episodes)

    return success_rate, results, env


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="V3 RLBench evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--eval_hdf5", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--ac_dim", type=int, default=8)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--n_active_cams", type=int, default=4)
    parser.add_argument("--T_pred", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--T_obs", type=int, default=1)
    parser.add_argument("--spatial_pool_size", type=int, default=1)
    parser.add_argument("--train_diffusion_steps", type=int, default=50)
    parser.add_argument("--eval_diffusion_steps", type=int, default=50)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default="checkpoints/eval_videos")
    parser.add_argument("--exec_horizon", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import pickle
    from pathlib import Path
    from data_pipeline.conversion.compute_norm_stats import load_norm_stats
    from models.policy_v3 import PolicyDiTv3
    from models.stage1_bridge import Stage1Bridge
    from training.eval_v3 import V3PolicyWrapper

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bridge = Stage1Bridge(checkpoint_path=args.stage1_checkpoint)
    policy = PolicyDiTv3(
        bridge=bridge, ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim, n_active_cams=args.n_active_cams,
        d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers,
        T_obs=args.T_obs, T_pred=args.T_pred,
        train_diffusion_steps=args.train_diffusion_steps,
        eval_diffusion_steps=args.eval_diffusion_steps,
        spatial_pool_size=args.spatial_pool_size,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt and "averaged_model" in ckpt["ema"]:
        ema_sd = ckpt["ema"]["averaged_model"]
        for key in ["denoiser.pos_emb", "denoiser.cond_pos_emb"]:
            if key in ema_sd:
                model_param = dict(policy.named_parameters())[key]
                if ema_sd[key].shape != model_param.shape:
                    old = ema_sd[key]
                    new = torch.zeros_like(model_param)
                    min_len = min(old.shape[1], new.shape[1])
                    new[:, :min_len, :] = old[:, :min_len, :]
                    ema_sd[key] = new
                    log.info("Resized EMA %s: %s -> %s",
                             key, list(old.shape), list(new.shape))
        policy.load_state_dict(ema_sd, strict=False)
        log.info("Loaded EMA weights for eval")
    policy.eval()

    wrapper = V3PolicyWrapper(policy, device=str(device))
    norm_stats = load_norm_stats(args.eval_hdf5)

    sr, results, env = evaluate_v3_rlbench(
        wrapper, norm_stats,
        task=args.task,
        num_episodes=args.num_episodes,
        obs_horizon=args.T_obs,
        save_video=args.save_video,
        video_dir=args.video_dir,
        exec_horizon=args.exec_horizon,
    )
    env.close()

    print(f"\nFinal: {sr*100:.1f}% ({int(sr*args.num_episodes)}/{args.num_episodes})")
    for i, r in enumerate(results):
        print(f"  ep{i:03d} {'SUCCESS' if r['success'] else 'FAIL'} steps={r['steps']}")
