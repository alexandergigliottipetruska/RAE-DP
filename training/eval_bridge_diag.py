"""Diagnostic bridge test: compare our wrapper's behavior with lite_physics on/off.

Also prints obs/action stats to help debug remaining gaps.
"""
from __future__ import annotations
import os, sys, warnings, logging
import numpy as np
import torch

warnings.filterwarnings("ignore", module="robosuite")
logging.getLogger("robosuite").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/CSC415/diffusion_policy"))

from collections import deque


def create_env_ours(seed, abs_action=True, lite_physics=True):
    """Create env using OUR RobomimicWrapper."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(robot="panda")
    if abs_action:
        right_cfg = controller_config["body_parts"]["right"]
        right_cfg["input_type"] = "absolute"
        right_cfg["input_ref_frame"] = "world"
        right_cfg["input_min"] = -10
        right_cfg["input_max"] = 10
        right_cfg["lite_physics"] = lite_physics

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84,
        camera_widths=84,
        use_object_obs=True,
        controller_configs=controller_config,
        ignore_done=True,
    )
    return env


def run_episode(policy, env, device, n_obs_steps=2, n_action_steps=8, max_steps=400):
    """Run one episode with diagnostics."""
    from data_pipeline.utils.rotation import convert_actions_from_rot6d

    obs_raw = env.reset()
    obs_history = deque(maxlen=n_obs_steps)
    obs_history.append(obs_raw)
    while len(obs_history) < n_obs_steps:
        obs_history.append(obs_raw)

    action_queue = []
    step = 0
    success = False
    all_actions = []

    while step < max_steps:
        if len(action_queue) == 0:
            obs_list = list(obs_history)
            obs_dict = {}
            for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                imgs = []
                for o in obs_list:
                    img = o[key][::-1].copy()  # flip bottom-up → top-up
                    img = img.astype(np.float32) / 255.0  # float [0,1]
                    img = np.moveaxis(img, -1, 0)  # HWC→CHW
                    imgs.append(img)
                obs_dict[key] = torch.from_numpy(
                    np.stack(imgs)
                ).unsqueeze(0).to(device)

            for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                vals = [o[key].astype(np.float32) for o in obs_list]
                obs_dict[key] = torch.from_numpy(
                    np.stack(vals)
                ).unsqueeze(0).to(device)

            # Print obs stats on first prediction
            if step == 0:
                for k, v in obs_dict.items():
                    print(f"    obs[{k}]: shape={list(v.shape)}, dtype={v.dtype}, "
                          f"min={v.float().min():.2f}, max={v.float().max():.2f}")

            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            actions_10d = result['action'][0].cpu().numpy()

            if step == 0:
                print(f"    pred_action[0]: {np.array2string(actions_10d[0], precision=4)}")

            actions_7d = convert_actions_from_rot6d(actions_10d)
            action_queue = list(actions_7d)

        action = action_queue.pop(0)
        all_actions.append(action.copy())
        obs_raw, reward, done, info = env.step(action)
        obs_history.append(obs_raw)
        step += 1

        if bool(env._check_success()):
            success = True
            break

    return success, step, np.array(all_actions)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load Chi's checkpoint
    import dill
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
        TrainDiffusionTransformerHybridWorkspace,
    )
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir="/tmp/eval_bridge_diag")
    workspace.load_payload(payload, exclude_keys=None, include_keys=['global_step', 'epoch'])

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    print("Policy loaded (EMA)" if cfg.training.use_ema else "Policy loaded")

    # Test with lite_physics=True (robosuite 1.5 default)
    print("\n" + "=" * 60)
    print("TEST A: lite_physics=True (robosuite 1.5 default)")
    print("=" * 60)
    successes_a = 0
    for ep in range(args.n_episodes):
        env = create_env_ours(seed=100000 + ep, lite_physics=True)
        success, steps, actions = run_episode(policy, env, device)
        successes_a += int(success)
        print(f"  Episode {ep+1}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")
        env.close()
    print(f"  Result A: {successes_a}/{args.n_episodes} ({100*successes_a/args.n_episodes:.0f}%)")

    # Test with lite_physics=False (backward compat with robosuite 1.2)
    print("\n" + "=" * 60)
    print("TEST B: lite_physics=False (backward compat)")
    print("=" * 60)
    successes_b = 0
    for ep in range(args.n_episodes):
        env = create_env_ours(seed=100000 + ep, lite_physics=False)
        success, steps, actions = run_episode(policy, env, device)
        successes_b += int(success)
        print(f"  Episode {ep+1}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")
        env.close()
    print(f"  Result B: {successes_b}/{args.n_episodes} ({100*successes_b/args.n_episodes:.0f}%)")

    print("\n" + "=" * 60)
    print(f"lite_physics=True:  {successes_a}/{args.n_episodes}")
    print(f"lite_physics=False: {successes_b}/{args.n_episodes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
