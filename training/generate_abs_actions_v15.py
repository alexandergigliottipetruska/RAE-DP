"""Generate absolute actions using robosuite 1.5's controller.

Replays original delta demos through robosuite 1.5's OSC_POSE controller,
extracts goal_pos/goal_ori as absolute actions. Produces an image_abs.hdf5
that is physics-matched with robosuite 1.5 for evaluation.

Usage:
  python training/generate_abs_actions_v15.py \
    --input data/raw/robomimic/lift/ph/image.hdf5 \
    --output data/raw/robomimic/lift/ph/image_abs_v15.hdf5 \
    --num_demos 5  # test with few first

  # Full conversion (all 200 demos):
  python training/generate_abs_actions_v15.py \
    --input data/raw/robomimic/lift/ph/image.hdf5 \
    --output data/raw/robomimic/lift/ph/image_abs_v15.hdf5
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import warnings

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def create_delta_env():
    """Create a robosuite 1.5 Lift env with default delta OSC_POSE controller."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(robot="panda")
    # Default is already delta mode — no modifications needed

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,  # no rendering needed
        use_camera_obs=False,
        use_object_obs=True,
        controller_configs=controller_config,
        ignore_done=True,
    )
    return env


def convert_demo(env, states, delta_actions):
    """Convert one demo's delta actions to absolute actions via robosuite 1.5.

    For each timestep:
    1. Reset env to state[t]
    2. Call robot.control(delta_action, policy_step=True)
    3. Read controller's goal_pos (base frame) and goal_ori (base frame)
    4. Convert to world frame
    5. Convert rotation matrix → axis-angle

    Returns:
        abs_actions: (T, 7) [world_pos(3), axis_angle(3), gripper(1)]
    """
    T = len(delta_actions)
    abs_actions = np.zeros((T, 7), dtype=np.float64)
    robot = env.robots[0]

    for t in range(T):
        # Reset to this timestep's state
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()

        # Process delta action through controller (sets goal_pos/goal_ori)
        robot.control(delta_actions[t], policy_step=True)

        # Read goal from the arm controller
        arm_ctrl = robot.composite_controller.part_controllers['right']

        # Convert from base/origin frame to world frame
        # run_controller does: desired_world_pos = origin_pos + origin_ori @ goal_pos
        world_pos = arm_ctrl.origin_pos + arm_ctrl.origin_ori @ arm_ctrl.goal_pos
        world_ori = arm_ctrl.origin_ori @ arm_ctrl.goal_ori

        # Convert rotation matrix → axis-angle
        rotvec = Rotation.from_matrix(world_ori).as_rotvec()

        # Gripper action passes through unchanged
        gripper = delta_actions[t, -1]

        abs_actions[t] = np.concatenate([world_pos, rotvec, [gripper]])

    return abs_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input delta HDF5 (with states)")
    parser.add_argument("--output", required=True, help="Output absolute HDF5")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Number of demos to convert (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Compare first demo's abs actions to Chi's 1.2 version")
    parser.add_argument("--chi_abs_hdf5", default=None,
                        help="Chi's image_abs.hdf5 for verification comparison")
    args = parser.parse_args()

    # Copy input to output (preserves all metadata, images, obs, etc.)
    log.info("Copying %s → %s", args.input, args.output)
    shutil.copy2(args.input, args.output)

    # Create env
    log.info("Creating robosuite 1.5 delta env")
    env = create_delta_env()
    env.reset()

    # Print controller info
    arm_ctrl = env.robots[0].composite_controller.part_controllers['right']
    log.info("Controller: type=%s, input_type=%s, input_ref_frame=%s",
             type(arm_ctrl).__name__, arm_ctrl.input_type, arm_ctrl.input_ref_frame)
    log.info("Origin pos: %s", arm_ctrl.origin_pos)

    with h5py.File(args.output, "a") as f:
        demo_keys = sorted(f["data"].keys())
        if args.num_demos:
            demo_keys = demo_keys[:args.num_demos]

        log.info("Converting %d demos", len(demo_keys))

        for i, key in enumerate(demo_keys):
            grp = f[f"data/{key}"]
            states = grp["states"][:]
            delta_actions = grp["actions"][:]

            # Convert
            abs_actions = convert_demo(env, states, delta_actions)

            # Overwrite actions in-place
            grp["actions"][:] = abs_actions

            if i == 0:
                log.info("Demo %s: delta[0]=%s", key,
                         np.array2string(delta_actions[0], precision=4))
                log.info("Demo %s: abs[0]  =%s", key,
                         np.array2string(abs_actions[0], precision=4))

            if (i + 1) % 20 == 0 or (i + 1) == len(demo_keys):
                log.info("  [%d/%d] done", i + 1, len(demo_keys))

    env.close()

    # Optional: compare to Chi's 1.2 absolute actions
    if args.verify and args.chi_abs_hdf5:
        log.info("Comparing to Chi's 1.2 absolute actions...")
        with h5py.File(args.output, "r") as f15, h5py.File(args.chi_abs_hdf5, "r") as f12:
            key = sorted(f15["data"].keys())[0]
            a15 = f15[f"data/{key}/actions"][:]
            a12 = f12[f"data/{key}/actions"][:]
            diff = np.abs(a15 - a12)
            log.info("  Per-dim max diff: %s", np.array2string(diff.max(0), precision=6))
            log.info("  Per-dim mean diff: %s", np.array2string(diff.mean(0), precision=6))
            log.info("  Overall max diff: %.6f", diff.max())
            log.info("  v1.5 abs[0]: %s", np.array2string(a15[0], precision=4))
            log.info("  v1.2 abs[0]: %s", np.array2string(a12[0], precision=4))

    log.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
