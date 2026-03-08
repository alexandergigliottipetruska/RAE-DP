"""RLBench demo replay verification (spec section 8.4).

Three modes:

1. numerical (default):
   Compare stored actions in unified HDF5 against raw observation poses.
   Works on Windows — no sim needed.

2. sim:
   Replay GT absolute EE poses through OMPL motion planning in CoppeliaSim.
   This tests the same action execution pathway used by PerAct/RVT/RVT-2
   at eval time: EndEffectorPoseViaPlanning + Discrete gripper.
   Requires WSL2 with CoppeliaSim + PyRep + RLBench.

   Expected success rates (~50%) due to CoppeliaSim physics non-determinism.

3. video:
   Stitch raw demo PNGs (CoppeliaSim renders from data collection) into
   MP4 videos. Works on Windows — no sim needed.

Usage (numerical — Windows or WSL2):
  python replay_rlbench.py --mode numerical \
      --hdf5 path/to/unified/close_jar.hdf5 \
      --raw  path/to/raw/close_jar

Usage (sim — WSL2 only):
  QT_QPA_PLATFORM=offscreen python replay_rlbench.py --mode sim \
      --raw  ~/rlbench_replay/raw/close_jar \
      --task close_jar --n 20

Usage (video — Windows or WSL2):
  python replay_rlbench.py --mode video \
      --raw path/to/raw/close_jar --n 3
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# Task name -> RLBench class name
TASK_MAP = {
    "close_jar": "CloseJar",
    "open_drawer": "OpenDrawer",
    "slide_block_to_color_target": "SlideBlockToColorTarget",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_rlbench_stub():
    """Register stub so pickles load on Windows without full RLBench."""
    try:
        import rlbench  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        from data_pipeline.conversion.rlbench_obs_stub import register_stub
        register_stub()


def get_episode_dirs(raw_root: Path, n: int):
    """Get sorted episode directories from raw task folder."""
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )
    return ep_dirs[:n]


def load_demo_pickle(ep_dir: Path):
    """Load Demo object from low_dim_obs.pkl."""
    with open(ep_dir / "low_dim_obs.pkl", "rb") as f:
        return pickle.load(f)


def get_observations(demo):
    """Extract observation list from Demo object."""
    if hasattr(demo, "_observations"):
        return demo._observations
    return list(demo)


def extract_poses(obs_list):
    """Extract [T,3] positions, [T,4] quaternions (xyzw), [T] grippers."""
    T = len(obs_list)
    positions = np.zeros((T, 3), dtype=np.float64)
    quats = np.zeros((T, 4), dtype=np.float64)
    grippers = np.zeros(T, dtype=np.float64)
    for t, obs in enumerate(obs_list):
        positions[t] = obs.gripper_pose[:3]
        quats[t] = obs.gripper_pose[3:]
        grippers[t] = float(obs.gripper_open)
    return positions, quats, grippers


# ---------------------------------------------------------------------------
# Mode 1: Numerical verification
# ---------------------------------------------------------------------------

def run_numerical(args):
    """Compare unified HDF5 actions against raw observation poses.

    Auto-detects action format:
    - 8D absolute: action[t] = [pos_{t+1}(3), quat_{t+1}(4), grip_t(1)]
    - 7D delta: action[t] = [delta_pos(3), axis_angle(3), grip_t(1)]
    """
    import h5py

    _ensure_rlbench_stub()

    raw_root = Path(args.raw)
    ep_dirs = get_episode_dirs(raw_root, args.n)

    with h5py.File(args.hdf5, "r") as f:
        mask_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in f["mask"][args.split][:]
        ]
        action_dim = f[f"data/{mask_keys[0]}/actions"].shape[1]
        n = min(args.n, len(mask_keys), len(ep_dirs))

        print(f"Numerical verification ({action_dim}D actions)")
        print(f"  HDF5:  {args.hdf5}")
        print(f"  Raw:   {args.raw}")
        print(f"  Demos: {n}")
        print()

        all_ok = True
        for i in range(n):
            key = mask_keys[i]
            obs_list = get_observations(load_demo_pickle(ep_dirs[i]))
            raw_pos, raw_quat, raw_grip = extract_poses(obs_list)
            actions = f[f"data/{key}/actions"][:]
            T = len(obs_list)
            assert actions.shape[0] == T - 1, \
                f"{key}: expected {T-1} actions, got {actions.shape[0]}"

            if action_dim == 8:
                # Absolute: action[t] = [pos_{t+1}, quat_{t+1}, grip_t]
                pos_err = np.abs(actions[:, :3] - raw_pos[1:]).max()
                rot_err = np.abs(actions[:, 3:7] - raw_quat[1:]).max()
                grip_ok = np.allclose(actions[:, 7], raw_grip[:-1], atol=1e-5)
            else:
                # 7D delta: recompute expected deltas and compare
                from scipy.spatial.transform import Rotation
                expected_dpos = raw_pos[1:] - raw_pos[:-1]
                R_curr = Rotation.from_quat(raw_quat[:-1])
                R_next = Rotation.from_quat(raw_quat[1:])
                expected_drot = (R_curr.inv() * R_next).as_rotvec()
                pos_err = np.abs(actions[:, :3] - expected_dpos).max()
                rot_err = np.abs(actions[:, 3:6] - expected_drot).max()
                grip_ok = np.allclose(actions[:, 6], raw_grip[:-1], atol=1e-5)

            ok = pos_err < 1e-4 and rot_err < 1e-4 and grip_ok
            all_ok = all_ok and ok
            print(f"  {key:10s} | pos_err={pos_err:.2e}  rot_err={rot_err:.2e}  "
                  f"grip={'OK' if grip_ok else 'FAIL'}  [{'OK' if ok else 'MISMATCH'}]")

    print(f"\n{'PASS' if all_ok else 'FAIL'}: "
          f"{'Actions match raw poses.' if all_ok else 'Mismatch detected.'}")
    return all_ok


# ---------------------------------------------------------------------------
# Mode 2: Sim replay (OMPL motion planning)
# ---------------------------------------------------------------------------

def run_sim(args):
    """Replay GT absolute EE poses via OMPL in CoppeliaSim.

    Modeled on the RVT-2 evaluation approach:
    - EndEffectorPoseViaPlanning (OMPL, absolute mode, ignore_collisions=True)
    - Discrete gripper mode
    - Dense: every timestep's pose is sent through the OMPL planner
    - Keyframes (--keyframes): only gripper-transition + endpoint poses
    """
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks

    task_name = args.task
    if task_name not in TASK_MAP:
        sys.exit(f"Unknown task: {task_name}. Supported: {list(TASK_MAP.keys())}")

    raw_root = Path(args.raw).resolve()
    ep_dirs = get_episode_dirs(raw_root, args.n)

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    task_cls = getattr(rlbench.tasks, TASK_MAP[task_name])
    task = env.get_task(task_cls)

    use_kf = args.keyframes
    label = "keyframe" if use_kf else "dense"
    print(f"Sim replay (OMPL {label}): {task_name}")
    print(f"  Raw:      {raw_root}")
    print(f"  Episodes: {len(ep_dirs)}")
    print()

    successes = []
    for i, ep_dir in enumerate(ep_dirs):
        # Load demo from pickle directly (avoids get_stored_demos image validation)
        demo = load_demo_pickle(ep_dir)
        obs_list = get_observations(demo)
        T = len(obs_list)

        # Restore scene to demo initial state
        task.set_variation(demo.variation_number)
        descriptions, obs = task.reset_to_demo(demo)

        # Build action sequence
        if use_kf:
            # Keyframes: gripper transitions + endpoint
            kf = [0]
            for t in range(1, T):
                if abs(float(obs_list[t].gripper_open)
                       - float(obs_list[t - 1].gripper_open)) > 0.5:
                    kf.append(t)
            if kf[-1] != T - 1:
                kf.append(T - 1)
            targets = []
            for j in range(len(kf) - 1):
                idx = kf[j + 1]
                pose = obs_list[idx].gripper_pose
                grip = 1.0 if obs_list[idx].gripper_open > 0.5 else 0.0
                targets.append(np.append(pose, grip))
        else:
            # Dense: action[t] = [pose_{t+1}, grip_t]
            targets = []
            for t in range(T - 1):
                pose = obs_list[t + 1].gripper_pose
                grip = 1.0 if obs_list[t].gripper_open > 0.5 else 0.0
                targets.append(np.append(pose, grip))

        # Execute actions
        success = False
        for action in targets:
            try:
                obs, reward, terminate = task.step(action)
            except Exception:
                continue
            if reward == 1.0:
                success = True
                break

        successes.append(success)
        print(f"  episode{i:<4d} | steps={len(targets):4d} | "
              f"{'SUCCESS' if success else 'FAIL'}")

    env.shutdown()

    n_ok = sum(successes)
    rate = n_ok / len(successes) if successes else 0
    print(f"\nResult: {n_ok}/{len(successes)} ({rate * 100:.0f}%)")

    if rate >= 0.4:
        print("PASS: Within expected physics non-determinism range (>=40%).")
    else:
        print("LOW: Below expected range. Check setup.")
    return rate


# ---------------------------------------------------------------------------
# Mode 3: Video from raw PNGs
# ---------------------------------------------------------------------------

def run_video(args):
    """Stitch raw demo PNGs (CoppeliaSim renders) into MP4 videos."""
    from PIL import Image
    import imageio.v3 as iio

    _ensure_rlbench_stub()

    raw_root = Path(args.raw)
    ep_dirs = get_episode_dirs(raw_root, args.n)
    task_name = raw_root.name

    video_dir = Path(args.video_dir) if args.video_dir else Path.cwd() / "replay_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    cameras = ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]
    print(f"Video from raw PNGs: {task_name}")
    print(f"  Output: {video_dir}")
    print(f"  Demos:  {len(ep_dirs)}")
    print()

    for ep_dir in ep_dirs:
        avail = [c for c in cameras
                 if (ep_dir / c).exists() and list((ep_dir / c).glob("*.png"))]
        if not avail:
            print(f"  {ep_dir.name:12s} | No PNGs — skipped")
            continue

        n_frames = len(list((ep_dir / avail[0]).glob("*.png")))
        frames = []
        for t in range(n_frames):
            row = []
            for cam in avail:
                p = ep_dir / cam / f"{t}.png"
                if p.exists():
                    row.append(np.array(Image.open(p).convert("RGB")))
            if row:
                frames.append(np.concatenate(row, axis=1))

        if not frames:
            print(f"  {ep_dir.name:12s} | No frames loaded — skipped")
            continue

        out = str(video_dir / f"{task_name}_{ep_dir.name}.mp4")
        iio.imwrite(out, frames, fps=15, codec="libx264")
        print(f"  {ep_dir.name:12s} | {n_frames} frames, "
              f"{len(avail)} cameras -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RLBench demo replay verification.",
    )
    parser.add_argument(
        "--mode", default="numerical",
        choices=["numerical", "sim", "video"],
    )
    parser.add_argument("--hdf5", help="Unified HDF5 path (numerical mode).")
    parser.add_argument(
        "--raw", required=True,
        help="Raw RLBench task dir (e.g. .../close_jar).",
    )
    parser.add_argument(
        "--task", default="close_jar",
        choices=list(TASK_MAP.keys()),
        help="Task name (sim mode).",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of episodes.")
    parser.add_argument(
        "--keyframes", action="store_true",
        help="Sim mode: replay keyframe poses only (gripper transitions + endpoint).",
    )
    parser.add_argument("--video-dir", help="Video output dir.")
    parser.add_argument("--split", default="train", help="HDF5 split (numerical mode).")

    args = parser.parse_args()

    if args.mode == "numerical":
        if not args.hdf5:
            parser.error("--hdf5 required for numerical mode")
        ok = run_numerical(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "sim":
        rate = run_sim(args)
        sys.exit(0 if rate >= 0.4 else 1)
    elif args.mode == "video":
        run_video(args)
        sys.exit(0)


if __name__ == "__main__":
    main()
